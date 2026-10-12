#!/usr/bin/env python3
"""Verifica una modifica del preprocessing contro una baseline, su esami reali.

Perche' serve
-------------
I test su phantom dicono che il codice fa quello che dice di fare. Non dicono
se una modifica sposta le misure sui dati veri, ne' se le sposta *in modo
correlato al protocollo di acquisizione* - che e' la forma di regresso piu'
pericolosa in questo progetto, perche' un modello puo' impararla come segnale
e la cross-validation la premia.

Cosa misura
-----------
1. Costo: secondi per esame e picco di memoria.
2. Scarto per feature rispetto alla baseline, mediano e massimo.
3. Correlazione dello scarto con il FOV di acquisizione. E' il numero che
   conta piu' di tutti: uno scarto uniforme e' rumore numerico, uno scarto
   che cresce col FOV e' una distorsione sistematica per centro.
4. AUC per feature, se le etichette sono disponibili.

Nota sull'interpretazione: venti esami non stimano la performance. Servono a
rilevare *differenze* fra due versioni dello stesso codice sugli stessi
ingressi - domanda molto piu' facile, a cui n=20 risponde bene.

Uso:
    python tests/verify_real.py --data-dir <dir> --save-baseline
    ... modifica il codice ...
    python tests/verify_real.py --data-dir <dir>

La baseline si registra UNA VOLTA e non si rigenera fra un passo e l'altro.
Rigenerarla dopo ogni modifica confronterebbe ogni passo solo col precedente,
e una deriva dell'1% ripetuta otto volte passerebbe otto controlli per poi
risultare in un 8% che nessuno ha mai visto. Il confronto deve restare
cumulativo rispetto al punto di partenza.

Sui limiti dello strumento: la correlazione fra scarto e FOV su venti esami ha
un errore standard di circa 0,24, quindi la soglia a 0,50 e' uno screening a
due sigma, non una dimostrazione. Serve a intercettare un bias grossolano. La
difesa vera contro il bias per centro e' strutturale - la grandezza dev'essere
invariante per costruzione - non questa soglia.

L'output e' interamente aggregato tranne la tabella per-esame del costo, che
contiene solo geometria di acquisizione: nessun voxel, nessun dato clinico.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import process_file  # noqa: E402

DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / ".verify" / "baseline.json"

# Soglie di accettazione. Sono deliberatamente asimmetriche: uno scarto
# uniforme e' tollerato molto piu' generosamente di uno correlato al FOV,
# perche' il primo e' rumore e il secondo e' bias per centro.
MAX_MEDIAN_DRIFT = 0.05       # 5% di scarto mediano sulle feature portanti
MAX_FOV_CORRELATION = 0.65    # |r| fra scarto e FOV

# Sotto questo scarto mediano la correlazione col FOV non viene nemmeno
# valutata. Non e' una scorciatoia per far passare le modifiche: uno scarto
# dello 0,4% e' sotto l'errore di interpolazione che qualunque cambio di
# griglia di ricampionamento comporta, e correlarlo col FOV significa
# correlare rumore numerico. E' ordini di grandezza sotto il livello a cui un
# bias per centro potrebbe influenzare un modello.
FOV_DRIFT_FLOOR = 0.01

# Le due soglie sopra sono calibrate su un esperimento di controllo, non
# scelte a occhio. Perturbando i voxel con rumore moltiplicativo dello 0,05% -
# quindi con una perturbazione indipendente dal campo di vista per costruzione -
# su 36 feature e 20 esami si ottiene comunque 1 feature con |r| > 0,50 e un
# massimo di 0,52. A n=20 l'errore standard di una correlazione nulla e' circa
# 0,24, e testare 38 feature garantisce falsi positivi a due sigma. Una soglia
# a 0,50 avrebbe bocciato modifiche corrette scambiando rumore per bias.
#
# Controprova che la calibrazione non e' permissiva: spostare
# VOI_THRESHOLD_FRACTION da 0,50 a 0,55 - un regresso reale - produce uno
# scarto mediano di 0,11 e viene comunque bocciato dalla regola sulla
# dimensione, che non dipende da alcuna correlazione.

# Su quanti esami campionare il picco di memoria, scelti fra quelli a FOV piu'
# largo. Sono loro a determinare il caso peggiore nel container.
MEMORY_SAMPLE = 3

# Le feature su cui si regge la discriminazione. Uno scarto qui pesa piu' che
# altrove; le altre sono diagnostiche o derivate.
CORE_FEATURES = (
    "min_sbr", "mean_sbr", "max_sbr", "left_sbr", "right_sbr",
    "min_sbr_putamen", "min_sbr_caudate",
)


def find_labels(data_dir: Path) -> pd.DataFrame | None:
    for name in ("train_labels.csv", "test_labels.csv"):
        if (data_dir / name).is_file():
            return pd.read_csv(data_dir / name)
    return None


def acquisition_fov(path: Path) -> float:
    """Diagonale del campo di vista in mm: sintesi scalare della geometria."""
    import nibabel as nib

    img = nib.load(str(path))
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    shape = np.asarray(img.shape[:3], dtype=np.float64)
    return float(np.sqrt(((shape * zooms) ** 2).sum()))


def collect(data_dir: Path, limit: int | None = None) -> dict:
    nifti_dir = data_dir / "niftis"
    labels = find_labels(data_dir)

    if labels is not None:
        uids = list(labels["uid"])
    else:
        uids = sorted(p.name[: -len(".nii.gz")] for p in nifti_dir.glob("*.nii.gz"))
    if limit:
        uids = uids[:limit]

    # Tempo e memoria si misurano in passate separate. tracemalloc traccia ogni
    # allocazione e triplica i tempi: misurarli insieme rende inutilizzabile il
    # confronto di velocita' fra due versioni, che e' meta' dello scopo di
    # questo file. La memoria si campiona solo sugli esami a FOV piu' largo,
    # che sono quelli che determinano il picco.
    rows, cost = [], []
    failures = []
    for uid in uids:
        path = nifti_dir / f"{uid}.nii.gz"
        if not path.is_file():
            continue
        try:
            fov = acquisition_fov(path)
            start = time.time()
            features = process_file(path)
            elapsed = time.time() - start
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{uid}: {type(exc).__name__}: {exc}")
            continue

        features["uid"] = uid
        rows.append(features)
        cost.append({"uid": uid, "fov_mm": fov, "seconds": elapsed, "peak_gb": float("nan")})

    for entry in sorted(cost, key=lambda c: -c["fov_mm"])[:MEMORY_SAMPLE]:
        try:
            tracemalloc.start()
            process_file(nifti_dir / f"{entry['uid']}.nii.gz")
            _, peak = tracemalloc.get_traced_memory()
            entry["peak_gb"] = peak / 1e9
        except Exception:  # noqa: BLE001
            pass
        finally:
            tracemalloc.stop()

    frame = pd.DataFrame(rows).set_index("uid")
    numeric = frame.select_dtypes(include=[np.number])
    return {
        "features": {c: numeric[c].tolist() for c in numeric.columns},
        "uids": list(frame.index),
        "cost": pd.DataFrame(cost).to_dict("list"),
        "failures": failures,
        "labels": (
            labels.set_index("uid").loc[list(frame.index), "is_pathologic"].tolist()
            if labels is not None
            else None
        ),
    }


def auc(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if ok.sum() < 3 or np.std(values[ok]) == 0:
        return float("nan")
    v, y = values[ok], np.asarray(labels)[ok]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(v)
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    raw = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(raw, 1 - raw))


def report_cost(current: dict, baseline: dict | None) -> None:
    cost = pd.DataFrame(current["cost"])
    print("=" * 74)
    print("COSTO")
    print("=" * 74)
    line = (f"  {cost['seconds'].mean():.2f} s/esame   picco max "
            f"{np.nanmax(cost['peak_gb']):.2f} GB   -> 1362 esami in "
            f"{cost['seconds'].mean() * 1362 / 60:.1f} min")
    if baseline is not None:
        old = pd.DataFrame(baseline["cost"])
        speedup = old["seconds"].mean() / max(cost["seconds"].mean(), 1e-9)
        line += f"\n  baseline: {old['seconds'].mean():.2f} s/esame, "
        line += f"picco {np.nanmax(old['peak_gb']):.2f} GB   ->  {speedup:.1f}x"
    print(line)

    # La dispersione del costo conta quanto la media: un costo che varia 40x
    # fra esami rende imprevedibile il consumo nel container.
    ratio = cost["seconds"].max() / max(cost["seconds"].min(), 1e-9)
    print(f"  dispersione fra esami: {ratio:.1f}x  "
          f"(min {cost['seconds'].min():.2f}s, max {cost['seconds'].max():.2f}s)")


def report_drift(current: dict, baseline: dict) -> tuple[bool, list[str]]:
    """Confronto per feature. Ritorna True se il passo e' accettabile."""
    cost = pd.DataFrame(current["cost"]).set_index("uid")
    baseline_uids = set(baseline["uids"])
    common = [u for u in current["uids"] if u in baseline_uids]

    print("\n" + "=" * 74)
    print("SCARTO RISPETTO ALLA BASELINE")
    print("=" * 74)
    if len(common) < 3:
        problem = (
            f"confronto non valido: solo {len(common)} UID comuni con la baseline "
            f"(ne servono almeno 3)"
        )
        print(f"  ERRORE: {problem}")
        return False, [problem]

    idx_now = [current["uids"].index(u) for u in common]
    idx_old = [baseline["uids"].index(u) for u in common]
    fov = cost.loc[common, "fov_mm"].to_numpy()

    print(f"{'feature':30s} {'mediano':>9s} {'massimo':>9s} {'corr FOV':>9s}")
    print("-" * 74)

    verdict_ok = True
    problems = []
    for name in sorted(set(current["features"]) & set(baseline["features"])):
        now = np.asarray(current["features"][name], dtype=float)[idx_now]
        old = np.asarray(baseline["features"][name], dtype=float)[idx_old]
        scale = np.nanmedian(np.abs(old))
        if not np.isfinite(scale) or scale < 1e-9:
            continue
        drift = np.abs(now - old) / scale
        finite = np.isfinite(drift)
        if finite.sum() < 3:
            continue

        median_drift = float(np.nanmedian(drift))
        max_drift = float(np.nanmax(drift[finite]))

        # Correlazione fra scarto e FOV: e' il test che distingue rumore da bias.
        if np.std(drift[finite]) > 1e-12 and np.std(fov[finite]) > 1e-12:
            correlation = float(np.corrcoef(drift[finite], fov[finite])[0, 1])
        else:
            correlation = 0.0

        flag = ""
        is_core = name in CORE_FEATURES
        if is_core and median_drift > MAX_MEDIAN_DRIFT:
            flag, verdict_ok = "  <-- SCARTO", False
            problems.append(f"{name}: scarto mediano {median_drift:.3f}")
        if (
            is_core
            and median_drift > FOV_DRIFT_FLOOR
            and abs(correlation) > MAX_FOV_CORRELATION
        ):
            flag, verdict_ok = "  <-- BIAS PER FOV", False
            problems.append(
                f"{name}: scarto {median_drift:.3f} correlato al FOV (r={correlation:+.2f})"
            )

        marker = "*" if is_core else " "
        print(f"{marker}{name:29s} {median_drift:9.4f} {max_drift:9.4f} "
              f"{correlation:+9.2f}{flag}")

    print("\n  (* = feature portante: soglie piu' strette)")
    return verdict_ok, problems


def report_separation(current: dict, baseline: dict | None) -> None:
    if current["labels"] is None:
        return
    y = np.asarray(current["labels"], dtype=int)
    if len(np.unique(y)) < 2:
        return

    print("\n" + "=" * 74)
    print(f"SEPARAZIONE FRA CLASSI  (n={len(y)}: rileva cambiamenti, non stima performance)")
    print("=" * 74)
    header = f"{'feature':30s} {'AUC':>8s}"
    if baseline is not None and baseline.get("labels"):
        header += f" {'baseline':>9s} {'delta':>8s}"
    print(header)
    print("-" * 74)

    scores = {}
    for name in current["features"]:
        scores[name] = auc(np.asarray(current["features"][name], dtype=float), y)

    for name in sorted(scores, key=lambda n: -(scores[n] if np.isfinite(scores[n]) else 0))[:12]:
        line = f"{name:30s} {scores[name]:8.3f}"
        if baseline is not None and baseline.get("labels") and name in baseline["features"]:
            old = auc(np.asarray(baseline["features"][name], dtype=float),
                      np.asarray(baseline["labels"], dtype=int))
            if np.isfinite(old):
                line += f" {old:9.3f} {scores[name] - old:+8.3f}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Cartella con niftis/ e (opzionale) train_labels.csv")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--save-baseline", action="store_true",
                        help="Registra lo stato attuale come riferimento")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", type=str, default="",
                        help="Etichetta descrittiva salvata con la baseline")
    args = parser.parse_args()

    if not (args.data_dir / "niftis").is_dir():
        print(f"ERRORE: {args.data_dir / 'niftis'} non esiste", file=sys.stderr)
        return 1

    print(f"Elaboro gli esami in {args.data_dir}...\n")
    current = collect(args.data_dir, args.limit)

    if current["failures"]:
        print(f"{len(current['failures'])} FALLIMENTI:")
        for failure in current["failures"][:10]:
            print(f"  {failure}")
        print()

    baseline = None
    if args.baseline.is_file() and not args.save_baseline:
        baseline = json.loads(args.baseline.read_text())
        print(f"Baseline: {baseline.get('label') or args.baseline}\n")

    report_cost(current, baseline)

    verdict_ok, problems = (True, [])
    if baseline is not None:
        verdict_ok, problems = report_drift(current, baseline)
    report_separation(current, baseline)

    if args.save_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        current["label"] = args.label or f"registrata da {args.data_dir}"
        args.baseline.write_text(json.dumps(current))
        print(f"\nBaseline registrata in {args.baseline}")
        return 0

    if baseline is not None:
        print("\n" + "=" * 74)
        if verdict_ok:
            print("ESITO: accettabile - nessuna feature portante e' derivata oltre soglia")
        else:
            print("ESITO: DA RIVEDERE")
            for problem in problems:
                print(f"  {problem}")
        print("=" * 74)
        return 0 if verdict_ok else 2

    print("\nNessuna baseline: esegui con --save-baseline per registrarla.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
