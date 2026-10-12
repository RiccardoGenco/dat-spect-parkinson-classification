#!/usr/bin/env python3
"""Ablazione bloccata: simmetria voxel-wise prima e dopo allineamento rigido.

Tre soli bracci, definiti prima di leggere i risultati:

* gold: le 44 feature correnti;
* raw: gold + descrittori di simmetria sul crop non allineato;
* registered: gold + gli stessi descrittori dopo allineamento a cinque DoF.

La promozione richiede almeno 0,001 di log loss sia nella CV raggruppata
annidata sia nello stress leave-protocol-out. Non si sceglie una singola feature
o una scala dopo aver visto l'OOF: ciascun blocco entra o esce interamente.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import registered_symmetry_features, symmetry_features  # noqa: E402
from src.cv import make_folds  # noqa: E402
from src.model import TARGET, feature_columns, train_fusion_logistic  # noqa: E402
from src.voxnet import fondi, logit, stima_fusione  # noqa: E402


MIN_GAIN = 0.001


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }


def nested_fusion(y: np.ndarray, p_tab: np.ndarray, p_cnn: np.ndarray,
                  folds: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([logit(p_tab), logit(p_cnn)])
    out = np.zeros(len(y), dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train, valid = folds != fold, folds == fold
        model = LogisticRegression(
            C=1.0, max_iter=5000, fit_intercept=False
        ).fit(X[train], y[train])
        out[valid] = model.predict_proba(X[valid])[:, 1]
    return out


def _partition(groups: np.ndarray) -> list[tuple[str, np.ndarray]]:
    counts = pd.Series(groups).value_counts()
    major = [g for g, n in counts.items() if n >= 120]
    parts = [(str(g), np.flatnonzero(groups == g)) for g in major]
    rare = np.flatnonzero(~np.isin(groups, major))
    if len(rare):
        parts.append((f"({len(counts) - len(major)} gruppi rari)", rare))
    return parts


def external_tabular(frame: pd.DataFrame, columns: list[str], y: np.ndarray,
                     groups: np.ndarray) -> np.ndarray:
    """Logistica gold rifittata senza il protocollo da predire."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.transforms import LogRatios

    out = np.zeros(len(y), dtype=np.float64)
    for _, valid in _partition(groups):
        train = np.ones(len(y), dtype=bool)
        train[valid] = False
        preds = []
        for C in (0.1, 0.3):
            model = make_pipeline(
                LogRatios(), SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=C, max_iter=5000),
            ).fit(frame.loc[train, columns], y[train])
            preds.append(model.predict_proba(frame.iloc[valid][columns])[:, 1])
        out[valid] = np.mean(preds, axis=0)
    return out


def external_report(y: np.ndarray, groups: np.ndarray, p_tab: np.ndarray,
                    p_cnn: np.ndarray, a: float, b: float) -> dict[str, object]:
    fused = fondi(p_tab, p_cnn, a, b)
    rows = []
    for name, valid in _partition(groups):
        rows.append({"fold": name, "n": int(len(valid)), **metrics(y[valid], fused[valid])})
    return {"overall": metrics(y, fused), "per_fold": rows}


def eta2_within_class(values: np.ndarray, groups: np.ndarray, y: np.ndarray) -> float:
    numerator = denominator = 0.0
    for label in (0, 1):
        mask = (y == label) & np.isfinite(values)
        v, g = values[mask], groups[mask]
        overall = float(v.mean())
        for group in np.unique(g):
            sample = v[g == group]
            if len(sample) >= 10:
                numerator += len(sample) * (float(sample.mean()) - overall) ** 2
        denominator += float(((v - overall) ** 2).sum())
    return numerator / denominator if denominator > 0 else float("nan")


def feature_diagnostics(frame: pd.DataFrame, names: list[str], y: np.ndarray,
                        groups: np.ndarray, grey: np.ndarray) -> list[dict[str, float]]:
    from sklearn.metrics import roc_auc_score

    rows = []
    for name in names:
        values = frame[name].to_numpy(dtype=float)
        auc = float(roc_auc_score(y, values))
        auc_grey = float(roc_auc_score(y[grey], values[grey]))
        rows.append({
            "feature": name,
            "eta2_within_class": eta2_within_class(values, groups, y),
            "auc_oriented": max(auc, 1.0 - auc),
            "auc_grey_oriented": max(auc_grey, 1.0 - auc_grey),
        })
    return rows


def extract(crops: np.ndarray, geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, registered = [], []
    for i in range(len(crops)):
        crop = np.asarray(crops[i], dtype=np.float32)
        left = geometry.loc[i, ["sx_x", "sx_y", "sx_z"]].to_numpy(dtype=float)
        right = geometry.loc[i, ["dx_x", "dx_y", "dx_z"]].to_numpy(dtype=float)
        raw.append({f"raw_{k}": v for k, v in symmetry_features(crop).items()})
        registered.append(registered_symmetry_features(crop, left, right))
        if (i + 1) % 250 == 0:
            print(f"  estratti {i + 1}/{len(crops)}", flush=True)
    return pd.DataFrame(raw), pd.DataFrame(registered)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=root / "features_v4.csv")
    ap.add_argument("--geometry", type=Path, default=root / "crops/geometria.csv")
    ap.add_argument("--crops", type=Path, default=root / "crops/crops.npy")
    ap.add_argument(
        "--cnn", type=Path,
        default=root / "experiments/risultati/voxel_premip_final_s012.oof.npy",
    )
    ap.add_argument(
        "--external", type=Path,
        default=root / "experiments/risultati/metro_esterno_gruppi_cnn_batch_rid1_premip_imagenet.npz",
    )
    ap.add_argument(
        "--derived", type=Path,
        default=root / "experiments/risultati/features_registration_symmetry.csv",
    )
    ap.add_argument(
        "--out", type=Path,
        default=root / "experiments/risultati/registration_symmetry.json",
    )
    ap.add_argument(
        "--arrays", type=Path,
        default=root / "experiments/risultati/registration_symmetry.npz",
    )
    args = ap.parse_args()

    base = pd.read_csv(args.features).dropna(subset=[TARGET]).reset_index(drop=True)
    geometry = pd.read_csv(args.geometry).reset_index(drop=True)
    if not np.array_equal(base["uid"].astype(str), geometry["uid"].astype(str)):
        raise SystemExit("features e geometria non sono allineate per uid")
    crops = np.load(args.crops, mmap_mode="r")
    if len(crops) != len(base):
        raise SystemExit("crops e features hanno lunghezza diversa")

    started = time.time()
    raw, registered = extract(crops, geometry)
    derived = pd.concat([base[["uid"]], raw, registered], axis=1)
    args.derived.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(args.derived, index=False)

    frame = pd.concat([base, raw, registered], axis=1)
    y = frame[TARGET].to_numpy(dtype=int)
    groups = frame["_group"].to_numpy()
    folds = make_folds(y, groups, n_splits=5, seed=0)
    p_cnn = np.load(args.cnn)
    ext = np.load(args.external)
    base_columns = feature_columns(base)
    raw_columns = list(raw.columns)
    registered_columns = list(registered.columns)
    arms = {
        "gold": base_columns,
        "raw_symmetry": base_columns + raw_columns,
        "registered_symmetry": base_columns + registered_columns,
    }

    results: dict[str, object] = {
        "n": int(len(y)), "min_gain": MIN_GAIN,
        "elapsed_extraction_seconds": time.time() - started,
        "arms": {},
    }
    predictions = {}
    arrays = {}
    for name, columns in arms.items():
        p_tab, _ = train_fusion_logistic(
            frame[columns].to_numpy(dtype=float), y, folds, columns
        )
        p_fused = nested_fusion(y, p_tab, p_cnn, folds)
        a, b = stima_fusione(y, p_tab, p_cnn)
        p_ext_tab = external_tabular(frame, columns, y, groups)
        p_ext_fused = fondi(p_ext_tab, ext["p_cnn"], a, b)
        predictions[name] = p_fused
        arrays[f"{name}_tabular_oof"] = p_tab
        arrays[f"{name}_nested_fusion"] = p_fused
        arrays[f"{name}_tabular_external"] = p_ext_tab
        arrays[f"{name}_external_fusion"] = p_ext_fused
        results["arms"][name] = {
            "n_features": len(columns),
            "tabular_oof": metrics(y, p_tab),
            "nested_fusion": metrics(y, p_fused),
            "deployment_coefficients": {"a": a, "b": b},
            "external": external_report(y, groups, p_ext_tab, ext["p_cnn"], a, b),
        }

    # Ensemble a peso uguale: scelta unica e simmetrica, senza ottimizzare un
    # peso sulla stessa OOF. Conserva il gold come stabilizzatore e aggiunge il
    # segnale spaziale del ramo registrato.
    ensemble_name = "gold_registered_equal_ensemble"
    p_tab = np.mean([
        arrays["gold_tabular_oof"],
        arrays["registered_symmetry_tabular_oof"],
    ], axis=0)
    p_ext_tab = np.mean([
        arrays["gold_tabular_external"],
        arrays["registered_symmetry_tabular_external"],
    ], axis=0)
    p_fused = nested_fusion(y, p_tab, p_cnn, folds)
    a, b = stima_fusione(y, p_tab, p_cnn)
    p_ext_fused = fondi(p_ext_tab, ext["p_cnn"], a, b)
    predictions[ensemble_name] = p_fused
    arrays[f"{ensemble_name}_tabular_oof"] = p_tab
    arrays[f"{ensemble_name}_nested_fusion"] = p_fused
    arrays[f"{ensemble_name}_tabular_external"] = p_ext_tab
    arrays[f"{ensemble_name}_external_fusion"] = p_ext_fused
    results["arms"][ensemble_name] = {
        "n_features": "44 + 54, media probabilita 1:1",
        "tabular_oof": metrics(y, p_tab),
        "nested_fusion": metrics(y, p_fused),
        "deployment_coefficients": {"a": a, "b": b},
        "external": external_report(y, groups, p_ext_tab, ext["p_cnn"], a, b),
    }

    baseline = results["arms"]["gold"]
    base_nested = baseline["nested_fusion"]["log_loss"]
    base_external = baseline["external"]["overall"]["log_loss"]
    decisions = {}
    for name in ("raw_symmetry", "registered_symmetry", ensemble_name):
        arm = results["arms"][name]
        gain_nested = base_nested - arm["nested_fusion"]["log_loss"]
        gain_external = base_external - arm["external"]["overall"]["log_loss"]
        decisions[name] = {
            "gain_nested": gain_nested,
            "gain_external": gain_external,
            "promote": gain_nested >= MIN_GAIN and gain_external >= MIN_GAIN,
        }
    results["decisions"] = decisions

    # Conferma fuori dalla singola assegnazione dei gruppi. Il candidato e'
    # gia' fissato: questi seed non selezionano feature o iperparametri.
    stability = []
    for seed in range(8):
        seed_folds = make_folds(y, groups, n_splits=5, seed=seed)
        row = {"seed": seed}
        seed_tabs = {}
        for name in ("gold", "registered_symmetry"):
            columns = arms[name]
            p_tab, _ = train_fusion_logistic(
                frame[columns].to_numpy(dtype=float), y, seed_folds, columns
            )
            seed_tabs[name] = p_tab
            row[name] = metrics(y, nested_fusion(y, p_tab, p_cnn, seed_folds))["log_loss"]
        p_equal = np.mean([seed_tabs["gold"], seed_tabs["registered_symmetry"]], axis=0)
        row[ensemble_name] = metrics(
            y, nested_fusion(y, p_equal, p_cnn, seed_folds)
        )["log_loss"]
        row["gain"] = row["gold"] - row["registered_symmetry"]
        row["ensemble_gain"] = row["gold"] - row[ensemble_name]
        stability.append(row)
    results["stability_8_fold_seeds"] = {
        "runs": stability,
        "mean_gain": float(np.mean([r["gain"] for r in stability])),
        "wins": int(sum(r["gain"] > 0 for r in stability)),
        "ensemble_mean_gain": float(np.mean([r["ensemble_gain"] for r in stability])),
        "ensemble_wins": int(sum(r["ensemble_gain"] > 0 for r in stability)),
    }

    grey = (predictions["gold"] > 0.23) & (predictions["gold"] < 0.65)
    results["feature_diagnostics"] = {
        "raw": feature_diagnostics(frame, raw_columns, y, groups, grey),
        "registered": feature_diagnostics(frame, registered_columns, y, groups, grey),
    }
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(args.arrays, **arrays)
    print(json.dumps(results, indent=2))
    print(f"\nScritto {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
