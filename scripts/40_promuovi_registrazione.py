#!/usr/bin/env python3
"""Promuove il ramo gold con simmetria registrata senza toccare l'incumbent."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import SYMMETRY_FEATURE_NAMES  # noqa: E402
from src.cv import make_folds  # noqa: E402
from src.model import FUSION_LOGISTIC_C_GRID, TARGET, train_fusion_logistic  # noqa: E402
from src.voxnet import fondi, stima_fusione  # noqa: E402


BRANCH = "gold_registered_equal_ensemble_v1"


def metrics(y, p):
    from sklearn.metrics import log_loss, roc_auc_score

    return {
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "auroc": float(roc_auc_score(y, p)),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=root / "features_v4.csv")
    ap.add_argument(
        "--derived", type=Path,
        default=root / "experiments/risultati/features_registration_symmetry.csv",
    )
    ap.add_argument("--base-bundle", type=Path, default=root / "models/gold/models.pkl")
    ap.add_argument("--out-dir", type=Path, default=root / "models/registered_ensemble")
    ap.add_argument("--checkpoint", type=Path, default=root / "models/voxnet_mip.pt")
    ap.add_argument(
        "--checkpoint-out", type=Path,
        default=root / "models/voxnet_mip_registered_ensemble.pt",
    )
    ap.add_argument(
        "--cnn-oof", type=Path,
        default=root / "experiments/risultati/voxel_premip_final_s012.oof.npy",
    )
    ap.add_argument(
        "--arrays", type=Path,
        default=root / "experiments/risultati/registration_symmetry.npz",
    )
    ap.add_argument(
        "--external-source", type=Path,
        default=root / "experiments/risultati/metro_esterno_gruppi_cnn_batch_rid1_premip_imagenet",
    )
    ap.add_argument(
        "--meter-out", type=Path,
        default=root / "experiments/risultati/metro_registered_ensemble_premip_s012",
    )
    args = ap.parse_args()

    base = pd.read_csv(args.features).dropna(subset=[TARGET]).reset_index(drop=True)
    derived = pd.read_csv(args.derived).reset_index(drop=True)
    if not np.array_equal(base["uid"].astype(str), derived["uid"].astype(str)):
        raise SystemExit("features derivate non allineate per uid")
    frame = pd.concat([base, derived[list(SYMMETRY_FEATURE_NAMES)]], axis=1)
    y = frame[TARGET].to_numpy(dtype=int)
    groups = frame["_group"].to_numpy()

    with open(args.base_bundle, "rb") as handle:
        bundle = pickle.load(handle)
    columns = list(bundle["columns"]) + list(SYMMETRY_FEATURE_NAMES)
    folds = make_folds(y, groups, n_splits=5, seed=0)
    oof_registered, models = train_fusion_logistic(
        frame[columns].to_numpy(dtype=float), y, folds, columns
    )
    experiment = np.load(args.arrays)
    recorded_registered = experiment["registered_symmetry_tabular_oof"]
    if not np.allclose(oof_registered, recorded_registered, rtol=0, atol=1e-12):
        raise SystemExit("OOF registrata rifittata diversa dall'esperimento promosso")
    oof = np.mean([experiment["gold_tabular_oof"], oof_registered], axis=0)
    recorded = experiment["gold_registered_equal_ensemble_tabular_oof"]
    if not np.allclose(oof, recorded, rtol=0, atol=1e-12):
        raise SystemExit("OOF ensemble diversa dall'esperimento promosso")

    bundle["fusion_lr_base"] = bundle["fusion_lr"]
    bundle["fusion_lr"] = models
    bundle["fusion_lr_C"] = list(FUSION_LOGISTIC_C_GRID)
    bundle["fusion_lr_columns"] = columns
    bundle["fusion_lr_feature_version"] = BRANCH
    bundle["fusion_lr_blend_weight"] = 0.5
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "models.pkl", "wb") as handle:
        pickle.dump(bundle, handle)
    np.save(args.out_dir / "oof_fusion_lr.npy", oof)

    p_cnn = np.load(args.cnn_oof)
    a, b = stima_fusione(y, oof, p_cnn)
    (args.out_dir / "metrics.json").write_text(json.dumps({
        "branch": BRANCH, "n_features": len(columns),
        "tabular_oof": metrics(y, oof), "a": a, "b": b,
    }, indent=2), encoding="utf-8")

    import torch

    blob = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    blob["fusione"] = {"a": a, "b": b}
    blob.setdefault("provenienza", {})["tabular_branch"] = BRANCH
    blob["provenienza"]["tabular_oof"] = str(args.out_dir / "oof_fusion_lr.npy")
    torch.save(blob, str(args.checkpoint_out))
    sidecar = {
        "a": a, "b": b,
        "arco": blob.get("provenienza", {}).get("arco", "premip_final_s012"),
        "n_reti": len(blob["state_dicts"]),
        "norm": blob["config"].get("norm", "batch"),
        "arch": blob["config"]["arch"],
        "autonorm": bool(blob["config"].get("autonorm", False)),
        "tabular_branch": BRANCH,
    }
    args.checkpoint_out.with_suffix(".fusione.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )

    arrays = np.load(args.arrays)
    p_ext_tab = arrays["gold_registered_equal_ensemble_tabular_external"]
    ext_npz = np.load(args.external_source.with_suffix(".npz"))
    p_ext_fused = fondi(p_ext_tab, ext_npz["p_cnn"], a, b)
    source = json.loads(args.external_source.with_suffix(".json").read_text(
        encoding="utf-8"
    ))
    result = json.loads((args.arrays.with_suffix(".json")).read_text(encoding="utf-8"))
    rows = []
    arm_name = "gold_registered_equal_ensemble"
    named = [r["fold"] for r in result["arms"][arm_name]["external"]["per_fold"]
             if "x" in r["fold"]]
    for row in result["arms"][arm_name]["external"]["per_fold"]:
        mask = groups == row["fold"] if "x" in row["fold"] else ~np.isin(groups, named)
        rows.append({
            "fold": row["fold"], "n": int(mask.sum()),
            "solo_sbr": metrics(y[mask], p_ext_tab[mask]),
            "fusione": metrics(y[mask], p_ext_fused[mask]),
        })
    meter = {
        "partizione": "gruppi", "n": int(len(y)), "tabular_branch": BRANCH,
        "config_cnn": source["config_cnn"],
        "train_seeds": source.get("train_seeds", []),
        "fold_seed": source.get("fold_seed", 0),
        "complessivo": {
            "solo_sbr": metrics(y, p_ext_tab),
            "fusione": metrics(y, p_ext_fused),
        },
        "per_fold": rows,
    }
    args.meter_out.with_suffix(".json").write_text(
        json.dumps(meter, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.meter_out.with_suffix(".npz"), solo_sbr=p_ext_tab, p_cnn=ext_npz["p_cnn"]
    )
    print(f"Promosso {BRANCH}: a={a:.6f}, b={b:.6f}")
    print(f"Bundle: {args.out_dir / 'models.pkl'}")
    print(f"Checkpoint: {args.checkpoint_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
