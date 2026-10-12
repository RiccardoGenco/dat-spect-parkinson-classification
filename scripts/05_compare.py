#!/usr/bin/env python3
"""Confronto appaiato di piu' configurazioni di feature su piu' seed.

Perche' esiste
--------------
Questo progetto dichiara da tempo il proprio standard di misura - "sotto ~0,007
su un seed solo non e' un guadagno", perche' cambiando il seed dei fold la log
loss del blend si sposta di sd 0,0036 - e cita sei risultati nella forma "6 seed
su 6". Ma quel confronto non era codificato da nessuna parte: `02_train_cv.py` e'
monoseed per costruzione e sovrascrive `models/metrics.json` a ogni giro, quindi
nulla si accumula. I numeri erano stati ottenuti a mano e non erano riproducibili
da nessuno, autore incluso.

Il ciclo di iterazione documentato nel README - "modifica il codice, lancia
02_train_cv, leggi la log loss, tieni se migliora" - contraddiceva quindi
l'avvertimento sul rumore scritto qualche riga sopra. Questo script e' la
risoluzione di quella contraddizione: e' il ciclo di iterazione vero.

Cosa vuol dire "appaiato"
-------------------------
Per ogni seed i fold si calcolano UNA volta e vanno identici a tutti gli archi.
La quantita' che si legge non e' la log loss di un arco - che porta dentro tutta
la varianza del sorteggio dei fold, sd 0,0036 - ma la DIFFERENZA per seed fra
due archi che hanno visto esattamente la stessa partizione. La varianza del
sorteggio si cancella nella sottrazione, ed e' il motivo per cui otto seed
bastano a risolvere differenze di due millesimi.

La statistica che si riporta e' il numero di seed vinti, non solo la media. Con
n=8 la media di una differenza piccola resta fragile a un singolo seed anomalo,
mentre "6 su 6 nella stessa direzione" e' un'affermazione che un valore anomalo
non produce.

Cosa NON fa
-----------
Non scrive in `models/` e non stampa nulla dentro il ciclo di addestramento: gli
archi devono poter girare decine di volte senza effetti collaterali. Non
ri-estrae le feature: ogni arco legge lo STESSO CSV e differisce solo per quali
colonne riceve. Una configurazione che richieda una ri-estrazione non e' un arco,
e' un altro esperimento.

Uso:
    python scripts/05_compare.py --features features.csv --arms experiments/morphology.json
    python scripts/05_compare.py --features features.csv --arms ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv import make_folds  # noqa: E402
from src.model import TARGET, feature_columns, fit_and_score  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    """Accetta "0-7" oppure "0,3,5" oppure "0-3,7"."""
    seeds: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        elif chunk:
            seeds.append(int(chunk))
    return sorted(set(seeds))


def load_arms(path: Path) -> dict:
    """Carica la definizione degli archi.

    Sta in un file versionato e non su una riga di shell di proposito: un
    esperimento che non si puo' rileggere fra un mese non e' un esperimento, e'
    un ricordo. Il file dice anche PERCHE' ogni arco esiste, che e' l'unica parte
    che i numeri da soli non conservano.
    """
    spec = json.loads(path.read_text(encoding="utf-8"))
    if "arms" not in spec:
        raise ValueError(f"{path}: manca la chiave 'arms'")
    return spec


def resolve_arm_columns(
    arm: dict, available: list[str], excluded: list[str]
) -> tuple[list[str], list[str]]:
    """Colonne di un arco, piu' l'elenco dei nomi ignoti.

    Un arco toglie con `drop` e rimette con `include`. `include` serve a
    riportare nel modello una colonna che sta in `EXCLUDED_FROM_MODEL`, ed e'
    necessario perche' quella lista contiene due tipi diversi di voce: alcune
    escluse perche' la grandezza non e' cio' che dice di essere (intensita'
    grezze, artefatti di soglia), altre perche' e' stato MISURATO che non
    aggiungono nulla. Le seconde sono verdetti empirici presi su una versione
    passata del vettore di feature, e vanno poter essere riaperte quando il
    vettore cambia - e' scritto in features.py e finora non c'era modo di farlo.

    Le colonne rimesse vanno in coda, in ordine deterministico: l'ordine delle
    colonne entra nel sottocampionamento di LightGBM, quindi un arco senza
    `include` deve avere esattamente l'ordine di `02_train_cv.py`, o il
    confronto con la baseline sarebbe contro un modello leggermente diverso.

    Un nome sconosciuto e' quasi sempre un refuso, e un refuso qui non darebbe
    errore: l'arco girerebbe con una colonna diversa dal previsto e la
    differenza verrebbe letta come effetto della modifica. Va quindi fermato,
    non segnalato.
    """
    drop = set(arm.get("drop", []))
    include = [c for c in arm.get("include", []) if c not in drop]

    unknown = sorted(
        (drop - set(available) - set(excluded))
        | (set(include) - set(excluded) - set(available))
    )
    columns = [c for c in available if c not in drop]
    columns += [c for c in include if c not in columns]
    return columns, unknown


def summarize(scores: dict[str, dict[int, dict]], reference: str, seeds: list[int]) -> pd.DataFrame:
    """Tabella per arco: log loss media, Delta appaiato, seed vinti.

    Il Delta e' per seed e poi mediato, non la differenza fra le due medie. Sono lo
    stesso numero, ma solo il primo permette di contare i seed vinti e di
    stimare la dispersione della differenza, che e' cio' che dice se il risultato
    regge.
    """
    rows = []
    for arm, per_seed in scores.items():
        losses = np.array([per_seed[s]["log_loss"] for s in seeds])
        nested = np.array([per_seed[s]["log_loss_nested"] for s in seeds])
        reference_losses = np.array([scores[reference][s]["log_loss"] for s in seeds])
        deltas = losses - reference_losses
        won = int((deltas < 0).sum())
        rows.append({
            "arco": arm,
            "n_feature": per_seed[seeds[0]]["n_features"],
            "logloss": losses.mean(),
            "annidata": nested.mean(),
            "auroc": np.mean([per_seed[s]["auroc"] for s in seeds]),
            "delta": deltas.mean(),
            # Dispersione della DIFFERENZA, non della log loss: e' la quantita'
            # che il disegno appaiato riduce, e confrontarla con la sd fra seed
            # dell'arco di riferimento mostra di quanto.
            "sd_delta": deltas.std(ddof=1) if len(seeds) > 1 else np.nan,
            "vinti": f"{won}/{len(seeds)}" if arm != reference else "-",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True,
                        help="File JSON con la definizione degli archi")
    parser.add_argument("--seeds", default="0-7",
                        help="Seed dei fold, es. '0-7' o '0,2,4' (default 0-7)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ignore-groups", action="store_true",
                        help="CV stratificata semplice invece che per scanner")
    parser.add_argument("--out", type=Path, default=None,
                        help="Dove salvare i risultati "
                             "(default experiments/risultati/<nome>.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Valida gli archi e stampa il piano, senza addestrare")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    spec = load_arms(args.arms)
    arms: dict[str, dict] = spec["arms"]
    reference = spec.get("reference", next(iter(arms)))
    if reference not in arms:
        print(f"ERRORE: l'arco di riferimento '{reference}' non e' fra gli archi",
              file=sys.stderr)
        return 1

    frame = pd.read_csv(args.features).dropna(subset=[TARGET]).reset_index(drop=True)
    y = frame[TARGET].to_numpy(dtype=int)
    groups = None if args.ignore_groups else frame.get("_group", pd.Series(dtype=str)).to_numpy()
    if groups is not None and groups.size == 0:
        groups = None

    available = feature_columns(frame)
    # Colonne presenti nel CSV ma tenute fuori dal modello: sono il bacino da cui
    # `include` puo' attingere per riaprire un verdetto passato.
    excluded = [
        c for c in frame.columns
        if c not in available and c not in {TARGET, "uid"} and not c.startswith("_")
    ]

    # Validazione di TUTTI gli archi prima di addestrarne uno solo: scoprire un
    # refuso dopo dodici minuti di calcolo costerebbe l'intera esecuzione.
    plan, problems = {}, []
    for name, arm in arms.items():
        columns, unknown = resolve_arm_columns(arm, available, excluded)
        if unknown:
            problems.append(f"  {name}: nomi ignoti {unknown}")
        if not columns:
            problems.append(f"  {name}: non resta nessuna colonna")
        plan[name] = columns
    if problems:
        print("ERRORE: archi non validi.", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print(f"\nColonne nel modello ({len(available)}): {', '.join(sorted(available))}",
              file=sys.stderr)
        print(f"\nColonne escluse, riattivabili con 'include' ({len(excluded)}): "
              f"{', '.join(sorted(excluded))}", file=sys.stderr)
        return 1

    print(f"{spec.get('name', args.arms.stem)}: {len(arms)} archi x {len(seeds)} seed "
          f"= {len(arms) * len(seeds)} addestramenti")
    print(f"{len(frame)} esami, prevalenza {y.mean():.3f}, "
          f"CV {'semplice' if groups is None else 'raggruppata per scanner'}")
    print(f"riferimento: {reference}\n")
    width = max(len(n) for n in arms)
    for name, arm in arms.items():
        marker = "*" if name == reference else " "
        print(f" {marker} {name:{width}s}  {len(plan[name]):3d} feature  {arm.get('note', '')}")
    if args.dry_run:
        print("\n--dry-run: nessun addestramento eseguito.")
        return 0

    scores: dict[str, dict[int, dict]] = {name: {} for name in arms}
    started = time.time()
    total = len(arms) * len(seeds)
    done = 0

    for seed in seeds:
        # I fold si calcolano UNA volta per seed e vanno identici a ogni arco.
        # E' la definizione operativa di "appaiato": senza questa riga ogni arco
        # vedrebbe una partizione diversa e la differenza fra archi porterebbe
        # dentro la varianza del sorteggio, che e' dieci volte l'effetto cercato.
        folds = make_folds(y, groups, n_splits=args.n_splits, seed=seed)

        for name in arms:
            result = fit_and_score(
                frame, plan[name], seed=seed, n_splits=args.n_splits,
                groups=groups, folds=folds,
            )
            scores[name][seed] = {
                "log_loss": result["log_loss"],
                "log_loss_nested": result["log_loss_nested"],
                "auroc": result["auroc"],
                "weight": result["weight"],
                "n_features": len(plan[name]),
            }
            done += 1
            elapsed = time.time() - started
            eta = elapsed / done * (total - done)
            print(f"\r  {done}/{total}  seed {seed} arco {name}  "
                  f"logloss={result['log_loss']:.4f}  ETA {eta/60:.1f} min      ",
                  end="", flush=True)

    print(f"\n\nCompletato in {(time.time() - started)/60:.1f} min\n")

    table = summarize(scores, reference, seeds)
    print("=" * 78)
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("=" * 78)

    # Il pavimento di rumore, misurato in questa stessa esecuzione invece che
    # citato dal README. E' il metro con cui va letto ogni Delta della tabella.
    reference_losses = np.array([scores[reference][s]["log_loss"] for s in seeds])
    print(f"\nRumore fra seed sull'arco di riferimento: "
          f"sd {reference_losses.std(ddof=1):.4f} "
          f"(da {reference_losses.min():.4f} a {reference_losses.max():.4f})")
    print("Un Delta va letto insieme ai seed vinti: la media da sola, su otto seed,")
    print("puo' essere prodotta da un singolo seed anomalo.")

    # I risultati vanno in una cartella DIVERSA da quella della definizione degli
    # archi. Con il default precedente - stesso nome, stessa cartella - la prima
    # esecuzione sovrascriveva il file che la descriveva, e con esso le chiavi
    # che il risultato non riporta: l'esperimento cancellava la propria
    # definizione. Succede una volta sola, ma la volta che succede si perde
    # l'unica copia di cosa si stava misurando e perche'.
    out = args.out or (
        Path(__file__).resolve().parents[1] / "experiments" / "risultati"
        / f"{args.arms.stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() == args.arms.resolve():
        print(f"ERRORE: --out coincide con --arms ({out}): "
              f"salvare i risultati distruggerebbe la definizione degli archi.",
              file=sys.stderr)
        return 1
    out.write_text(json.dumps({
        "name": spec.get("name", args.arms.stem),
        "description": spec.get("description", ""),
        "features_file": str(args.features),
        "arms_file": str(args.arms),
        "reference": reference,
        "seeds": seeds,
        "grouped_cv": groups is not None,
        "n_samples": int(len(y)),
        # La definizione dell'arco si copia INTERA, non solo le chiavi che questo
        # script conosce: un arco descritto a meta' nel file dei risultati non
        # permette di rieseguirlo, che e' l'unica cosa per cui lo si conserva.
        "arms": {name: {**arms[name], "n_features": len(plan[name])} for name in arms},
        "scores": {name: {str(s): v for s, v in per_seed.items()}
                   for name, per_seed in scores.items()},
        "summary": table.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    print(f"\nRisultati in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
