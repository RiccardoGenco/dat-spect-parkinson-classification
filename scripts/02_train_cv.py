#!/usr/bin/env python3
"""Allena e valuta in cross-validation raggruppata per scanner.

Produce: log loss out-of-fold, modelli per fold, calibratore. La log loss OOF e'
il numero su cui si itera; e' l'unico segnale onesto disponibile prima di
consumare una submission.

Il modello vive in `src/model.py`, non qui: questo file e' la CLI che lo
esegue una volta e ne salva i prodotti. La separazione serve a `05_compare.py`,
che deve poter addestrare decine di volte senza stampare ne' scrivere - e
soprattutto deve misurare lo STESSO modello che finisce nel container.

Uso:
    python scripts/02_train_cv.py --features features.csv --out-dir models/
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv import describe_groups, evaluate_oof  # noqa: E402
from src.model import (FUSION_LOGISTIC_C_GRID, TARGET, feature_columns,  # noqa: E402
                       fit_and_score)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-groups", action="store_true",
                        help="Usa CV stratificata semplice (per misurare quanto e' ottimistica)")
    parser.add_argument("--drop", nargs="*", default=[], metavar="FEATURE",
                        help="Feature da escludere, per ablazione. Una feature che "
                             "porta segnale clinico vero peggiora la CV raggruppata "
                             "quando viene tolta; una che codifica l'identita' del "
                             "centro peggiora molto di piu' quella non raggruppata.")
    parser.add_argument("--include-meta", nargs="*", default=[], metavar="FEATURE",
                        help="Metadati con prefisso '_' da aggiungere esplicitamente. "
                             "Non cambia il default prudente di feature_columns().")
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)

    columns = feature_columns(frame)
    if args.include_meta:
        mancanti = set(args.include_meta) - set(frame.columns)
        non_meta = [c for c in args.include_meta if not c.startswith("_")]
        if mancanti or non_meta:
            print(f"ERRORE: --include-meta non valido: mancanti={sorted(mancanti)}, "
                  f"senza prefisso _={sorted(non_meta)}", file=sys.stderr)
            return 1
        columns += [c for c in args.include_meta if c not in columns]
        print(f"Metadati inclusi: {', '.join(args.include_meta)}")
    if args.drop:
        unknown = set(args.drop) - set(columns)
        if unknown:
            print(f"ERRORE: --drop nomina feature inesistenti: {sorted(unknown)}",
                  file=sys.stderr)
            return 1
        columns = [c for c in columns if c not in set(args.drop)]
        print(f"Ablazione: {len(args.drop)} feature escluse ({', '.join(args.drop)})")
    y = frame[TARGET].to_numpy(dtype=int)
    groups = None if args.ignore_groups else frame.get("_group", pd.Series(dtype=str)).to_numpy()
    if groups is not None and groups.size == 0:
        groups = None

    print(f"{len(frame)} esami, {len(columns)} feature, prevalenza {y.mean():.3f}")

    if groups is not None:
        summary = describe_groups(groups, y)
        print(f"\n{len(summary)} impronte scanner distinte")
        print(summary.head(10).to_string())
        if summary["share"].iloc[0] > 0.6:
            print("\nATTENZIONE: un solo gruppo copre oltre il 60% dei dati.")
            print("La CV per gruppi sara' instabile: leggi i risultati con cautela.")

    print("\nAlleno LightGBM e regressione logistica...")
    result = fit_and_score(
        frame, columns, seed=args.seed, n_splits=args.n_splits, groups=groups
    )
    print(f"  arresto per fold: {result['stopped_at']}")

    weight = result["weight"]
    print("\n" + "=" * 58)
    for name, key_ll, key_auc in [
        ("LightGBM", "log_loss_gbm", "auroc_gbm"),
        ("Logistica", "log_loss_lr", "auroc_lr"),
        (f"Blend (w={weight:.2f})", "log_loss", "auroc"),
    ]:
        print(f"{name:22s} logloss={result[key_ll]:.4f}  auroc={result[key_auc]:.4f}")

    # Il peso del blend e' scelto guardando tutte le etichette, quindi la riga
    # sopra e' leggermente ottimistica. Questa non lo e': il peso viene
    # riottimizzato in cross-validation annidata e valutato su dati che non ha
    # visto. La differenza fra le due e' il costo della scelta del peso.
    print(f"{'Blend (annidato)':22s} logloss={result['log_loss_nested']:.4f}   <- stima onesta")
    print(f"{'Baseline (prevalenza)':22s} logloss={result['baseline_log_loss']:.4f}")
    print("=" * 58)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "models.pkl", "wb") as handle:
        pickle.dump(
            {
                "gbm": result["models_gbm"],
                "lr": result["models_lr"],
                "fusion_lr": result["models_fusion_lr"],
                "fusion_lr_C": list(FUSION_LOGISTIC_C_GRID),
                "weight": weight,
                "columns": columns,
            },
            handle,
        )
    np.save(args.out_dir / "oof.npy", result["oof"])
    np.save(args.out_dir / "oof_fusion_lr.npy", result["oof_fusion_lr"])
    with open(args.out_dir / "metrics.json", "w") as handle:
        json.dump(
            {
                "blend_weight": weight,
                "n_samples": int(len(y)),
                "grouped_cv": groups is not None,
                "n_features": len(columns),
                "fusion_lr_C": list(FUSION_LOGISTIC_C_GRID),
                "lgb_stopped_at": [int(s) for s in result["stopped_at"]],
                "blend_log_loss_nested": result["log_loss_nested"],
                **evaluate_oof(y, result["oof"]),
            },
            handle,
            indent=2,
        )

    importance = pd.Series(
        np.mean([m.feature_importances_ for m in result["models_gbm"]], axis=0),
        index=columns,
    ).sort_values(ascending=False)
    print("\nFeature piu' importanti:")
    print(importance.head(15).to_string())

    print(f"\nModelli salvati in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
