#!/usr/bin/env python3
"""Crea un checkpoint MIP che usa il ramo tabellare gold in fusione.

I tensori della CNN non vengono toccati. Si scrive un nuovo artefatto, mai
sopra l'incumbent, aggiornando insieme checkpoint e sidecar con coefficienti
ricalcolati dalle OOF del ramo gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voxnet import stima_fusione  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=root / "models/voxnet_mip.pt")
    ap.add_argument("--output", type=Path,
                    default=root / "models/voxnet_mip_gold.pt")
    ap.add_argument("--features", type=Path, default=root / "features_v4.csv")
    ap.add_argument("--gold-oof", type=Path,
                    default=root / "models/gold/oof_fusion_lr.npy")
    ap.add_argument("--cnn-oof", type=Path, default=root /
                    "experiments/risultati/voxel_premip_final_s012.oof.npy")
    args = ap.parse_args()

    import torch

    y = pd.read_csv(args.features)["is_pathologic"].to_numpy(dtype=int)
    p_gold, p_cnn = np.load(args.gold_oof), np.load(args.cnn_oof)
    a, b = stima_fusione(y, p_gold, p_cnn)

    blob = torch.load(str(args.input), map_location="cpu", weights_only=True)
    blob["fusione"] = {"a": a, "b": b}
    blob.setdefault("provenienza", {})["tabular_branch"] = (
        "gold_logistic_C0.1_0.3"
    )
    blob["provenienza"]["tabular_oof"] = str(args.gold_oof)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, str(args.output))

    sidecar = {
        "a": a, "b": b,
        "arco": blob.get("provenienza", {}).get("arco", "premip_final_s012"),
        "n_reti": len(blob["state_dicts"]),
        "norm": blob["config"].get("norm", "batch"),
        "arch": blob["config"]["arch"],
        "autonorm": bool(blob["config"].get("autonorm", False)),
        "tabular_branch": "gold_logistic_C0.1_0.3",
    }
    args.output.with_suffix(".fusione.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    print(f"Scritto {args.output} con a={a:.6f}, b={b:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
