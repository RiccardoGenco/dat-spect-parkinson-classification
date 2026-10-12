#!/usr/bin/env python3
"""Confronta il tetto del ML classico con l'ensemble MIP attivo.

Il confronto e' deliberatamente piccolo e pre-specificato. Provare decine di
classificatori e riportare il migliore sulla stessa CV trasformerebbe il
benchmark in una ricerca sul rumore. Le famiglie qui rappresentano le quattro
alternative classiche sensate per 1362 esami e 44 misure: lineare fortemente
regolarizzata, kernel RBF, foresta randomizzata e boosting poco profondo.

La quantita' che decide non e' soltanto la metrica del ramo tabellare: per ogni
candidato si misura anche la fusione logit annidata con le OOF MIP. Un ramo
peggiore da solo puo' essere migliore nel sistema se conserva informazione
complementare alla rete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv import make_folds  # noqa: E402
from src.model import TARGET, feature_columns  # noqa: E402
from src.transforms import LogRatios  # noqa: E402
from src.voxnet import fondi, logit, stima_fusione  # noqa: E402


GOLD_C = (0.1, 0.3)


def _pipeline_logistica(C: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        LogRatios(), SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=C, max_iter=5000),
    )


def _modelli_classici() -> dict[str, object]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "svm_rbf": make_pipeline(
            LogRatios(), SimpleImputer(strategy="median"), StandardScaler(),
            SVC(C=1.0, kernel="rbf", probability=True, random_state=0),
        ),
        "extra_trees": make_pipeline(
            LogRatios(), SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=5, max_features=0.7,
                class_weight="balanced", n_jobs=-1, random_state=0,
            ),
        ),
        "hist_boost": make_pipeline(
            LogRatios(), SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.04, max_leaf_nodes=7,
                min_samples_leaf=30, l2_regularization=5, random_state=0,
            ),
        ),
    }


def _oof_modello(model, X: pd.DataFrame, y: np.ndarray,
                 folds: np.ndarray) -> np.ndarray:
    from sklearn.base import clone

    out = np.zeros(len(y), dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train, valid = folds != fold, folds == fold
        fitted = clone(model).fit(X.loc[train], y[train])
        out[valid] = fitted.predict_proba(X.loc[valid])[:, 1]
    return out


def oof_gold(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Media su due regolarizzazioni fissate, non scelta sul risultato."""
    return np.mean([
        _oof_modello(_pipeline_logistica(C), X, y, folds) for C in GOLD_C
    ], axis=0)


def _metriche(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }


def fusione_annidata(y: np.ndarray, p_tab: np.ndarray, p_cnn: np.ndarray,
                     folds: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    """Stacking logit senza intercetta, stimato fuori dal fold valutato."""
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([logit(p_tab), logit(p_cnn)])
    out = np.zeros(len(y), dtype=np.float64)
    coefficienti = []
    for fold in sorted(np.unique(folds)):
        train, valid = folds != fold, folds == fold
        model = LogisticRegression(
            C=1.0, max_iter=5000, fit_intercept=False
        ).fit(X[train], y[train])
        out[valid] = model.predict_proba(X[valid])[:, 1]
        coefficienti.append([float(v) for v in model.coef_[0]])
    return out, coefficienti


def _intervallo_delta(y: np.ndarray, incumbent: np.ndarray, candidato: np.ndarray,
                      groups: np.ndarray, seed: int = 0,
                      n_boot: int = 4000) -> dict[str, object]:
    """CI appaiati della riduzione di log loss, per esami e per gruppi."""
    rng = np.random.default_rng(seed)
    loss_i = -(y * np.log(incumbent) + (1 - y) * np.log(1 - incumbent))
    loss_c = -(y * np.log(candidato) + (1 - y) * np.log(1 - candidato))
    delta = loss_i - loss_c

    exam = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        exam[b] = delta[idx].mean()

    unici = np.unique(groups)
    per_group = np.array([delta[groups == g].mean() for g in unici])
    block = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(unici), len(unici))
        block[b] = per_group[idx].mean()

    def summary(values):
        return {
            "media": float(values.mean()),
            "p2_5": float(np.quantile(values, 0.025)),
            "p97_5": float(np.quantile(values, 0.975)),
            "probabilita_migliora": float(np.mean(values > 0)),
        }

    return {
        "delta_osservato": float(delta.mean()),
        "bootstrap_esami": summary(exam),
        "bootstrap_macro_gruppi": summary(block),
    }


def _stress_esterno(frame: pd.DataFrame, columns: list[str], y: np.ndarray,
                    groups: np.ndarray, modo: str, p_cnn: np.ndarray,
                    p_sbr: np.ndarray, a_old: float, b_old: float,
                    a_gold: float, b_gold: float) -> dict[str, object]:
    """Predice gruppi o strati esclusi con la logistica gold rifittata."""
    from sklearn.model_selection import StratifiedKFold

    if modo == "gruppi":
        counts = pd.Series(groups).value_counts()
        grandi = [g for g, n in counts.items() if n >= 120]
        parti = [(str(g), np.flatnonzero(groups == g)) for g in grandi]
        resto = np.flatnonzero(~np.isin(groups, grandi))
        if len(resto):
            parti.append((f"({len(counts) - len(grandi)} gruppi rari)", resto))
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        parti = [(f"strato {i}", valid)
                 for i, (_, valid) in enumerate(skf.split(np.zeros(len(y)), y))]

    X = frame[columns]
    p_gold = np.zeros(len(y), dtype=np.float64)
    righe = []
    for nome, valid in parti:
        train = np.ones(len(y), dtype=bool)
        train[valid] = False
        pred = []
        for C in GOLD_C:
            model = _pipeline_logistica(C).fit(X.loc[train], y[train])
            pred.append(model.predict_proba(X.iloc[valid])[:, 1])
        p_gold[valid] = np.mean(pred, axis=0)
        righe.append({"fold": nome, "n": int(len(valid))})

    old = fondi(p_sbr, p_cnn, a_old, b_old)
    gold = fondi(p_gold, p_cnn, a_gold, b_gold)
    for row, (_, valid) in zip(righe, parti):
        row["incumbent"] = _metriche(y[valid], old[valid])
        row["gold"] = _metriche(y[valid], gold[valid])
        row["delta_log_loss"] = (
            row["incumbent"]["log_loss"] - row["gold"]["log_loss"]
        )
    return {
        "incumbent": _metriche(y, old),
        "gold": _metriche(y, gold),
        "tabellare_gold": _metriche(y, p_gold),
        "per_fold": righe,
    }, p_gold


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=root / "features_v4.csv")
    ap.add_argument("--sbr", type=Path, default=root / "models/oof.npy")
    ap.add_argument(
        "--cnn", type=Path,
        default=root / "experiments/risultati/voxel_premip_final_s012.oof.npy",
    )
    ap.add_argument(
        "--fusione", type=Path, default=root / "models/voxnet_mip.fusione.json",
    )
    ap.add_argument(
        "--out", type=Path,
        default=root / "experiments/risultati/gold_standard.json",
    )
    ap.add_argument(
        "--meter-out", type=Path,
        default=root / "experiments/risultati/metro_gold_premip_s012",
        help="base .json/.npz dello stress a gruppi usato dal pacchettatore",
    )
    args = ap.parse_args()

    frame = pd.read_csv(args.features).dropna(subset=[TARGET]).reset_index(drop=True)
    columns = feature_columns(frame)
    X = frame[columns]
    y = frame[TARGET].to_numpy(dtype=int)
    groups = frame["_group"].to_numpy()
    folds = make_folds(y, groups, n_splits=5, seed=0)
    p_sbr, p_cnn = np.load(args.sbr), np.load(args.cnn)

    candidati = {"logistica_gold": oof_gold(X, y, folds)}
    for nome, model in _modelli_classici().items():
        candidati[nome] = _oof_modello(model, X, y, folds)

    p_base, coef_base = fusione_annidata(y, p_sbr, p_cnn, folds)
    risultati = {
        "n": int(len(y)), "n_feature": int(len(columns)),
        "fold_seed": 0, "gold_C_mediati": list(GOLD_C),
        "incumbent": {
            "tabellare": _metriche(y, p_sbr),
            "fusione_annidata": _metriche(y, p_base),
            "coefficienti_per_fold": coef_base,
        },
        "classici": {},
    }
    fusioni = {}
    for nome, pred in candidati.items():
        p_fuso, coefficienti = fusione_annidata(y, pred, p_cnn, folds)
        fusioni[nome] = p_fuso
        risultati["classici"][nome] = {
            "da_solo": _metriche(y, pred),
            "fusione_annidata": _metriche(y, p_fuso),
            "coefficienti_per_fold": coefficienti,
        }

    p_gold = candidati["logistica_gold"]
    f_gold = fusioni["logistica_gold"]
    risultati["logistica_gold_vs_incumbent"] = _intervallo_delta(
        y, p_base, f_gold, groups,
    )

    a_gold, b_gold = stima_fusione(y, p_gold, p_cnn)
    old = json.loads(args.fusione.read_text(encoding="utf-8"))
    risultati["coefficienti_deployment_gold"] = {"a": a_gold, "b": b_gold}
    for modo in ("gruppi", "strati"):
        path = root / "experiments/risultati" / (
            f"metro_esterno_{modo}_cnn_batch_rid1_premip_imagenet.npz"
        )
        if path.is_file():
            z = np.load(path)
            stress, p_gold_esterno = _stress_esterno(
                frame, columns, y, groups, modo, z["p_cnn"], z["solo_sbr"],
                float(old["a"]), float(old["b"]), a_gold, b_gold,
            )
            risultati[f"stress_{modo}"] = stress
            if modo == "gruppi":
                sorgente = json.loads(path.with_suffix(".json").read_text(
                    encoding="utf-8"))
                meter = {
                    "partizione": "gruppi",
                    "n": int(len(y)),
                    "tabular_branch": "gold_logistic_C0.1_0.3",
                    "config_cnn": sorgente["config_cnn"],
                    "train_seeds": sorgente.get("train_seeds", []),
                    "fold_seed": sorgente.get("fold_seed", 0),
                    "complessivo": {
                        "solo_sbr": stress["tabellare_gold"],
                        "fusione_gold": stress["gold"],
                    },
                    "per_fold": stress["per_fold"],
                }
                args.meter_out.parent.mkdir(parents=True, exist_ok=True)
                args.meter_out.with_suffix(".json").write_text(
                    json.dumps(meter, indent=2), encoding="utf-8"
                )
                np.savez_compressed(
                    args.meter_out.with_suffix(".npz"),
                    p_cnn=z["p_cnn"], solo_sbr=p_gold_esterno,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(risultati, indent=2), encoding="utf-8")
    print(json.dumps(risultati, indent=2))
    print(f"\nScritto {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
