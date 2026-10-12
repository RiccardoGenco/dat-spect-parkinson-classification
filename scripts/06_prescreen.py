#!/usr/bin/env python3
"""Cancelli gratuiti su una feature candidata, prima di spendere seed.

Perche' esiste
--------------
Un arco appaiato su otto seed costa qualche minuto; questi tre numeri costano
qualche secondo e si leggono sul CSV che c'e' gia'. Servono a non portare
all'esperimento candidati che si possono escludere prima, e soprattutto a non
leggere un risultato al limite del rumore come se fosse una scoperta.

Le tre domande, in ordine di quanto pesano:

1. **Quanto di questa feature e' il centro invece del paziente.** Eta quadro
   della feature rispetto all'impronta dello scanner, calcolato DENTRO ciascuna
   classe cosi' che la diversa prevalenza fra centri non lo gonfi. Misurato sul
   vettore attuale: le feature SBR stanno a 0,26-0,35, `localization_confidence`
   a 0,13, i rapporti fra code del profilo a 0,01-0,03. E la correlazione fra
   quanto una feature e' contaminata e quanto sembra discriminante e' +0,71:
   **le feature che paiono migliori a dati aggregati sono le piu' confuse dal
   centro.** Per questo l'AUC da sola non e' un criterio.

2. **Se e' un duplicato di qualcosa che c'e' gia'.** Massima correlazione contro
   ogni altra colonna. Il confine e' calibrato su due verdetti noti e non su una
   convenzione: `min_shape_corr` correla +0,841 con `localization_confidence` ed
   e' stata respinta (+0,0023 su 0 seed su 6), `max_shape_corr` correla +0,777
   ed e' stata accettata. Sopra 0,85 un candidato va provato come SOSTITUZIONE,
   mai come aggiunta: due colonne quasi identiche sotto penalita' L2 si dividono
   il coefficiente e aggiungono varianza senza aggiungere informazione.

3. **Se puo' spostare la log loss dove la log loss si perde.** Il 40% della
   perdita residua sta nel 17% di esami con predizione fra 0,23 e 0,65, a
   prevalenza 0,494: casi genuinamente dubbi, non errori di calibrazione. Una
   feature che non separa dentro quella zona non puo' spostare il totale, per
   quanto separi bene altrove. Sotto AUC 0,62 nella zona grigia il guadagno
   atteso e' sotto il pavimento di rumore.

Un candidato bocciato qui non e' necessariamente inutile: questi cancelli
ordinano le priorita' e distinguono le aggiunte dalle sostituzioni. La decisione
resta del confronto appaiato.

Uso:
    python scripts/06_prescreen.py --features features_v4.csv
    python scripts/06_prescreen.py --features features_v4.csv --pattern log_slope tail_ratio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import TARGET, feature_columns  # noqa: E402

# Soglie, ciascuna con la misura che la calibra. Vedi il docstring del modulo.
ETA2_GATE = 0.10
COLLINEARITY_GATE = 0.85
GREY_AUC_GATE = 0.62
GREY_LOW, GREY_HIGH = 0.23, 0.65
MIN_GROUP = 10


def eta_squared_within_class(values: np.ndarray, groups: np.ndarray, y: np.ndarray) -> float:
    """Quota di varianza spiegata dall'impronta scanner, dentro ciascuna classe.

    Il "dentro ciascuna classe" e' il punto. La prevalenza di casi patologici
    varia da 0,154 a 0,778 fra i gruppi di spacing: calcolando eta quadro sui
    dati aggregati, una feature perfettamente pulita ma discriminante
    risulterebbe contaminata solo perche' i gruppi hanno composizioni diverse.
    Sommando numeratore e denominatore separatamente per classe si misura solo
    la parte che dipende dal centro a parita' di diagnosi.
    """
    numerator = denominator = 0.0
    for label in (0, 1):
        mask = (y == label) & np.isfinite(values)
        if mask.sum() < 20:
            continue
        v, g = values[mask], groups[mask]
        overall = v.mean()
        for group in np.unique(g):
            sample = v[g == group]
            if sample.size < MIN_GROUP:
                continue
            numerator += sample.size * (sample.mean() - overall) ** 2
        denominator += ((v - overall) ** 2).sum()
    return float(numerator / denominator) if denominator > 0 else np.nan


def rank_auc(values: np.ndarray, y: np.ndarray) -> float:
    """AUC per ranghi, orientata cosi' che 0,5 sia l'assenza di segnale.

    Si prende max(a, 1-a) perche' il verso con cui una feature separa non e'
    un'informazione da valutare qui: una feature che ordina al contrario separa
    esattamente quanto una che ordina nel verso giusto, e il segno lo sistema il
    coefficiente del modello.
    """
    from scipy.stats import rankdata

    mask = np.isfinite(values)
    if mask.sum() < 20:
        return np.nan
    labels = y[mask]
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(values[mask])
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1 - auc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--oof", type=Path, default=Path("models/oof.npy"),
                        help="Predizioni out-of-fold, per la zona grigia")
    parser.add_argument("--pattern", nargs="*", default=None, metavar="SOTTOSTRINGA",
                        help="Esamina solo le colonne che contengono una di queste "
                             "sottostringhe (default: tutte le colonne del modello)")
    args = parser.parse_args()

    frame = pd.read_csv(args.features).dropna(subset=[TARGET]).reset_index(drop=True)
    y = frame[TARGET].to_numpy(dtype=int)
    groups = frame["_group"].to_numpy()
    columns = feature_columns(frame)

    targets = columns
    if args.pattern:
        targets = [c for c in columns if any(p in c for p in args.pattern)]
        if not targets:
            print(f"Nessuna colonna contiene {args.pattern}", file=sys.stderr)
            return 1

    grey = None
    if args.oof.exists():
        oof = np.load(args.oof)
        if len(oof) == len(frame):
            grey = (oof > GREY_LOW) & (oof < GREY_HIGH)
            print(f"Zona grigia da {args.oof}: {grey.sum()} esami ({grey.mean():.1%}), "
                  f"prevalenza {y[grey].mean():.3f}")
        else:
            print(f"ATTENZIONE: {args.oof} ha {len(oof)} righe e il CSV {len(frame)}: "
                  f"zona grigia non calcolabile. Rilancia 02_train_cv sul CSV nuovo.")
    print(f"{len(frame)} esami, {len(columns)} colonne nel modello, "
          f"{len(targets)} da esaminare\n")

    values = {c: frame[c].to_numpy(dtype=float) for c in columns}
    rows = []
    for name in targets:
        v = values[name]
        # Massima correlazione contro ogni ALTRA colonna del modello.
        best_r, best_with = 0.0, ""
        for other in columns:
            if other == name:
                continue
            mask = np.isfinite(v) & np.isfinite(values[other])
            if mask.sum() < 20:
                continue
            a, b = v[mask], values[other][mask]
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if abs(r) > abs(best_r):
                best_r, best_with = r, other
        rows.append({
            "feature": name,
            "eta2": eta_squared_within_class(v, groups, y),
            "auc": rank_auc(v, y),
            "auc_grigia": rank_auc(v[grey], y[grey]) if grey is not None else np.nan,
            "max_r": best_r,
            "con": best_with,
        })

    table = pd.DataFrame(rows).sort_values("eta2")
    table["esito"] = [
        "SOSTITUZIONE" if abs(r) > COLLINEARITY_GATE
        else ("centro" if e > ETA2_GATE and not f.startswith("asym_") else "ok")
        for f, e, r in zip(table["feature"], table["eta2"], table["max_r"])
    ]

    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"eta2 > {ETA2_GATE}  ->  'centro': la feature segue lo scanner piu' del paziente")
    print(f"|r| > {COLLINEARITY_GATE}   ->  'SOSTITUZIONE': va provata al posto di quella "
          f"correlata, non in aggiunta")
    print(f"auc_grigia < {GREY_AUC_GATE} ->  da sola non puo' superare il pavimento di rumore")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
