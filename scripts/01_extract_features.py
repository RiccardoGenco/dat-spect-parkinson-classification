#!/usr/bin/env python3
"""Estrae le feature striatali da tutti gli esami di training e le salva su disco.

Va eseguito una sola volta (o dopo ogni modifica al preprocessing). Costa
qualche minuto; da li' in poi ogni esperimento sul modello parte dalla tabella
di feature ed e' questione di secondi. E' questa separazione a rendere il ciclo
di iterazione veloce abbastanza da valerne la pena.

Uso:
    python scripts/01_extract_features.py --data-dir /percorso/data --out features.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import process_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Cartella contenente niftis/ e train_labels.csv")
    parser.add_argument("--out", type=Path, default=Path("features.csv"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Processa solo i primi N esami (per un giro di prova)")
    args = parser.parse_args()

    nifti_dir = args.data_dir / "niftis"
    labels_path = args.data_dir / "train_labels.csv"

    if not nifti_dir.is_dir():
        print(f"ERRORE: {nifti_dir} non esiste", file=sys.stderr)
        return 1
    if not labels_path.is_file():
        print(f"ERRORE: {labels_path} non esiste", file=sys.stderr)
        return 1

    labels = pd.read_csv(labels_path)
    if args.limit:
        labels = labels.head(args.limit)
    print(f"{len(labels)} esami da processare")

    rows, failures = [], []
    start = time.time()

    for i, uid in enumerate(labels["uid"], 1):
        path = nifti_dir / f"{uid}.nii.gz"
        try:
            features = process_file(path)
            features["uid"] = uid
            rows.append(features)
        except Exception as exc:  # noqa: BLE001
            # Non interrompere: un file corrotto non deve costare l'intera
            # estrazione. I fallimenti vengono elencati alla fine.
            failures.append((uid, f"{type(exc).__name__}: {exc}"))

        if i % 50 == 0 or i == len(labels):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(labels) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(labels)}  {rate:.1f}/s  ~{remaining/60:.1f} min rimanenti")

    frame = pd.DataFrame(rows).merge(labels, on="uid", how="left")
    frame.to_csv(args.out, index=False)

    print(f"\nScritte {len(frame)} righe in {args.out}")
    print(f"Tempo totale: {(time.time()-start)/60:.1f} min")

    if failures:
        print(f"\n{len(failures)} FALLIMENTI:")
        for uid, err in failures[:20]:
            print(f"  {uid}: {err}")

    # Diagnostica immediata: quanti gruppi di scanner e quanto sono sbilanciati.
    if "_group" in frame.columns:
        counts = frame["_group"].value_counts()
        print(f"\nImpronte scanner distinte: {len(counts)}")
        print(counts.head(12).to_string())

    n_missing = frame.isna().sum()
    problematic = n_missing[n_missing > len(frame) * 0.05]
    if len(problematic):
        print("\nFeature con oltre il 5% di valori mancanti:")
        print(problematic.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
