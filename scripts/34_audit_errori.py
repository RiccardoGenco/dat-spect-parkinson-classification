#!/usr/bin/env python3
"""Autopsia aggregata degli errori OOF della soluzione candidata.

La media della log loss non dice *come* un modello fallisce. Questo audit
separa le letture in cui sbaglia un solo ramo da quelle in cui sbagliano
entrambi, misura quanto pochi casi estremi dominano la metrica, e verifica se
la dispersione fra seed e' un indicatore affidabile di errore.

Non stampa UID, righe, predizioni individuali o contenuto delle immagini. Il
report contiene solo statistiche aggregate, cosi' puo' essere conservato nel
repository senza trasformarsi in una copia indiretta dei dati di gara.

Uso::

    python scripts/34_audit_errori.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import TARGET  # noqa: E402


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)


def _metriche(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = _clip(p)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _perdita(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _riassumi_maschera(
    mask: np.ndarray,
    loss: np.ndarray,
    wrong: np.ndarray,
) -> dict[str, float | int]:
    n = int(mask.sum())
    return {
        "n": n,
        "quota": float(mask.mean()),
        "tasso_errore_finale": float(wrong[mask].mean()) if n else float("nan"),
        "perdita_media": float(loss[mask].mean()) if n else float("nan"),
        "quota_perdita_totale": float(loss[mask].sum() / loss.sum()) if n else 0.0,
    }


def _effetti_feature(
    frame: pd.DataFrame,
    riferimento: np.ndarray,
    casi: np.ndarray,
    massimo: int = 12,
) -> list[dict]:
    """Differenza robusta casi-riferimento, in unita' di IQR.

    Si confrontano mediane invece di correlare una feature con la loss: una
    correlazione grezza confonde quasi sempre il failure mode con la classe.
    L'IQR rende confrontabili feature con scale diverse senza dipendere dalle
    code estreme tipiche delle misure SBR.
    """
    righe = []
    for nome in frame.select_dtypes(include=[np.number]).columns:
        if nome == TARGET or nome.startswith("_"):
            continue
        x = frame[nome].to_numpy(np.float64)
        buoni = np.isfinite(x) & (riferimento | casi)
        ref = x[buoni & riferimento]
        err = x[buoni & casi]
        if len(ref) < 20 or len(err) < 10:
            continue
        q25, q75 = np.percentile(ref, [25, 75])
        scala = float(q75 - q25)
        if scala <= 1e-12:
            continue
        effetto = float((np.median(err) - np.median(ref)) / scala)
        righe.append({
            "feature": nome,
            "effetto_iqr": effetto,
            "n_riferimento": int(len(ref)),
            "n_casi": int(len(err)),
        })
    return sorted(righe, key=lambda r: abs(r["effetto_iqr"]), reverse=True)[:massimo]


def _vicini(
    impronta: Path | None,
    y: np.ndarray,
    maschere: dict[str, np.ndarray],
) -> dict | None:
    """Concordanza di etichetta dei cinque vicini nell'impronta MIP congelata."""
    if impronta is None or not impronta.exists():
        return None

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    x = np.load(impronta, mmap_mode="r")
    if len(x) != len(y):
        raise ValueError(f"impronta con {len(x)} righe, attese {len(y)}")
    x = normalize(np.asarray(x, dtype=np.float32))
    # Il primo vicino e' il campione stesso e viene scartato.
    indici = NearestNeighbors(n_neighbors=6, metric="cosine").fit(x).kneighbors(
        x, return_distance=False
    )[:, 1:]
    accordo = (y[indici] == y[:, None]).mean(axis=1)
    fuori = {"k": 5, "accordo_medio": float(accordo.mean())}
    for nome, mask in maschere.items():
        fuori[nome] = {
            "n": int(mask.sum()),
            "accordo_medio": float(accordo[mask].mean()) if mask.any() else float("nan"),
            "quota_senza_maggioranza": float((accordo[mask] < 0.6).mean())
            if mask.any() else float("nan"),
        }
    return fuori


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=ROOT / "features_v4.csv")
    ap.add_argument("--sbr", type=Path, default=ROOT / "models/oof.npy")
    ap.add_argument(
        "--cnn", type=Path,
        default=ROOT / "experiments/risultati/voxel_premip_final_s012.oof.npy",
    )
    ap.add_argument(
        "--per-seed", type=Path,
        default=ROOT / "experiments/risultati/voxel_premip_final_s012.oof_per_seed.npy",
    )
    ap.add_argument(
        "--fusione", type=Path, default=ROOT / "models/voxnet_mip.fusione.json",
        help="sidecar dell'artefatto realmente spedito, non models/voxnet.fusione.json",
    )
    ap.add_argument(
        "--impronta", type=Path, default=ROOT / "crops/impronta_mip_r18.npy",
    )
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "experiments/risultati/audit_errori_premip_s012.json",
    )
    args = ap.parse_args()

    frame = pd.read_csv(args.features)
    y = frame[TARGET].to_numpy(np.float64)
    gruppi = frame["_group"].astype(str).to_numpy()
    sbr = np.load(args.sbr)
    cnn = np.load(args.cnn)
    per_seed = np.load(args.per_seed)
    fusione = json.loads(args.fusione.read_text(encoding="utf-8"))
    a, b = float(fusione["a"]), float(fusione["b"])

    if len(sbr) != len(y) or len(cnn) != len(y) or per_seed.shape[1] != len(y):
        raise ValueError("OOF, predizioni per seed e feature non sono allineate")

    finale = expit(a * logit(_clip(sbr)) + b * logit(_clip(cnn)))
    loss = _perdita(y, finale)
    y_bool = y.astype(bool)
    sbr_wrong = (sbr >= 0.5) != y_bool
    cnn_wrong = (cnn >= 0.5) != y_bool
    wrong = (finale >= 0.5) != y_bool
    ordine = np.argsort(loss)[::-1]

    categorie = {
        "entrambi_corretti": ~sbr_wrong & ~cnn_wrong,
        "solo_sbr_sbaglia": sbr_wrong & ~cnn_wrong,
        "solo_cnn_sbaglia": ~sbr_wrong & cnn_wrong,
        "entrambi_sbagliano": sbr_wrong & cnn_wrong,
    }
    top50 = np.zeros(len(y), dtype=bool)
    top50[ordine[:50]] = True
    falsi_negativi = y_bool & wrong
    falsi_positivi = ~y_bool & wrong

    report: dict = {
        "n": int(len(y)),
        "prevalenza": float(y.mean()),
        "artefatto": {
            "fusione": str(args.fusione), "a": a, "b": b,
            "arco": fusione.get("arco"), "n_reti": fusione.get("n_reti"),
        },
        "metriche": {
            "sbr": _metriche(y, sbr),
            "cnn": _metriche(y, cnn),
            "fusione_spedita": _metriche(y, finale),
        },
        "coda_perdita": {},
        "categorie_rami": {
            nome: _riassumi_maschera(mask, loss, wrong)
            for nome, mask in categorie.items()
        },
        "errori_per_classe": {
            "falsi_negativi": _riassumi_maschera(falsi_negativi, loss, wrong),
            "falsi_positivi": _riassumi_maschera(falsi_positivi, loss, wrong),
        },
    }

    for k in (10, 20, 50, 100, 200):
        idx = ordine[:k]
        report["coda_perdita"][str(k)] = {
            "quota_perdita_totale": float(loss[idx].sum() / loss.sum()),
            "perdita_media": float(loss[idx].mean()),
            "quota_patologici": float(y[idx].mean()),
        }

    sd_seed = np.std(per_seed, axis=0)
    report["instabilita_seed"] = {
        "n_seed": int(per_seed.shape[0]),
        "sd_media_tutti": float(sd_seed.mean()),
        "sd_media_errori": float(sd_seed[wrong].mean()),
        "sd_media_top50": float(sd_seed[top50].mean()),
        "rho_sd_perdita": float(spearmanr(sd_seed, loss).statistic),
    }

    report["calibrazione_decili"] = [
        {
            "n": int(len(idx)),
            "predizione_media": float(finale[idx].mean()),
            "frequenza_osservata": float(y[idx].mean()),
            "perdita_media": float(loss[idx].mean()),
        }
        for idx in np.array_split(np.argsort(finale), 10)
    ]

    gruppi_out = []
    for gruppo in pd.unique(gruppi):
        mask = gruppi == gruppo
        if mask.sum() < 20:
            continue
        r = {
            "gruppo": str(gruppo),
            "n": int(mask.sum()),
            "prevalenza": float(y[mask].mean()),
            "tasso_errore": float(wrong[mask].mean()),
            "quota_perdita_totale": float(loss[mask].sum() / loss.sum()),
            **_metriche(y[mask], finale[mask]),
        }
        gruppi_out.append(r)
    report["per_gruppo_n_almeno_20"] = sorted(
        gruppi_out, key=lambda r: r["n"], reverse=True
    )

    # Confronti dentro la stessa classe: descrivono il failure mode senza
    # confondere semplicemente normale e patologico.
    report["fenotipo_feature"] = {
        "falsi_negativi_vs_veri_positivi": _effetti_feature(
            frame, y_bool & ~wrong, falsi_negativi
        ),
        "falsi_positivi_vs_veri_negativi": _effetti_feature(
            frame, ~y_bool & ~wrong, falsi_positivi
        ),
    }
    report["vicini_impronta"] = _vicini(
        args.impronta,
        y,
        {
            "top50_perdita": top50,
            "entrambi_sbagliano": categorie["entrambi_sbagliano"],
            "falsi_negativi": falsi_negativi,
            "falsi_positivi": falsi_positivi,
        },
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"fusione {a:.3f}/{b:.3f}: log loss {report['metriche']['fusione_spedita']['log_loss']:.4f}, "
          f"AUROC {report['metriche']['fusione_spedita']['auroc']:.4f}")
    print(f"entrambi sbagliano: {categorie['entrambi_sbagliano'].sum()} esami, "
          f"{report['categorie_rami']['entrambi_sbagliano']['quota_perdita_totale']:.1%} della perdita")
    print(f"peggiori 100: {report['coda_perdita']['100']['quota_perdita_totale']:.1%} della perdita")
    print(f"SD seed: tutti {sd_seed.mean():.4f}, errori {sd_seed[wrong].mean():.4f}, "
          f"rho con perdita {report['instabilita_seed']['rho_sd_perdita']:.3f}")
    print(f"scritto {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
