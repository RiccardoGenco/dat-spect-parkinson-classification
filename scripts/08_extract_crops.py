#!/usr/bin/env python3
"""Crop cerebrali a estensione fissa per il ramo a immagini.

Perche' esiste
--------------
Le 44 feature sono una compressione con perdita enorme: da ~10^6 voxel a 44
scalari. Il podio richiede un segnale a AUC ~0,73 dentro la zona grigia, piu' di
quanto ne abbia l'intero modello attuale (0,6938), e nessuna delle 44 supera
0,5884 li' dentro. Un modello che vede i voxel e' l'unica strada rimasta che
aggiunga informazione di tipo nuovo (`docs/DECISIONS.md`).

Due scelte che decidono la qualita' del dataset
-----------------------------------------------
**Il crop e' centrato sul baricentro cerebrale, non sull'ancora striatale.**
L'ancora dello striato deriva dall'intensita', quindi si sposta quando l'uptake
degrada; il baricentro della maschera cerebrale no - la testa e' approssimativamente
simmetrica e il suo centroide cade sulla linea mediana anche con uptake fortemente
asimmetrico, ed e' la stessa ragione per cui `midline_x` lo usa. Centrare sul
segnale che si sta misurando significherebbe togliere alla rete proprio
l'informazione su *dove* lo striato si trova rispetto all'anatomia, che e' parte
di cio' che distingue un caso patologico.

**Nessun riallineamento del verso antero-posteriore, e non per pigrizia.** La
pipeline non si fida di `as_closest_canonical` e deduce il verso dai dati
(`infer_anterior_sign`), perche' i metadati della SPECT clinica sono spesso
approssimativi; un crop con l'asse AP invertito rispetto agli altri sarebbe
veleno per una rete convoluzionale, in silenzio. Misurato su tutti i 1362:
`header_orientation_agrees` vale 1,000 ovunque, quindi non c'e' niente da
ribaltare. Se un giorno smettesse di valere, questo e' il punto in cui accorgersene.

**Il volume e' normalizzato sul riferimento occipitale del paziente stesso.**
I conteggi grezzi sono dose iniettata per tempo di acquisizione per scala di
ricostruzione, cioe' l'identita' del centro travestita da misura: darli a una rete
significherebbe insegnarle il centro. Diviso per il riferimento, ogni voxel diventa
un rapporto interno al singolo paziente - la stessa grandezza dell'SBR, non
mediata su una VOI. E' anche cio' che mantiene la conformita' alla regola «ogni
esame processato indipendentemente»: nessuna statistica sull'insieme del test set.

Uso:
    python scripts/08_extract_crops.py --data-dir ../parkinson-data --out crops
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

from src.pipeline import CROP_VOXELS, TARGET_SPACING_MM, process_file  # noqa: E402

# `CROP_VOXELS` (64 voxel a 2 mm = 128 mm di lato) vive in `src/pipeline.py`
# insieme al codice che lo usa: il crop lo produce l'inferenza, non questo
# script, e una costante geometrica duplicata e' una divergenza in attesa.
#
# Margine SOLO sull'asse infero-superiore, e per un motivo misurato.
#
# Il crop e' centrato sul baricentro della maschera cerebrale. Su x e y quel
# baricentro e' stabile - la testa entra sempre tutta nel campo di vista in
# quelle direzioni - ma su z **non lo e'**: l'estensione assiale della maschera
# va da 106 a 234 mm (sd 20,5) perche' i protocolli includono quantita' diverse
# di collo e cervelletto, e il baricentro si sposta con essa.
#
# Misurato sui crop gia' estratti, usando come sonda il centroide del 0,5% di
# voxel piu' intensi e limitandosi ai soli **normali**, dove quella sonda e'
# affidabile perche' lo striato e' di gran lunga la struttura piu' brillante:
#
#     posizione dello striato dentro il crop, sd    x 4,2 mm   y 5,6 mm   z 10,6 mm
#     corr(estensione z della maschera, z striato)  +0,578
#
# Il segno di quella correlazione dice anche **dove** sta la variabilita': piu'
# la maschera e' estesa, piu' in alto cade lo striato nel crop, cioe' l'estensione
# in piu' viene aggiunta in basso. Il vertice del cranio e' il riferimento stabile,
# la base no.
#
# Estrarre piu' fette in z che in x,y permette di scegliere il riferimento **dopo**
# averlo misurato invece di indovinarlo ora: qualunque ricentratura entro +/-16
# voxel (32 mm) diventa un'operazione di affettamento, e il dataset a 64^3 attuale
# si riottiene esattamente prendendo le 64 fette centrali. E' anche il cancello A0
# del ramo: se la fetta centrale non riproduce `crops/crops.npy` bit per bit,
# qualcosa e' cambiato in un posto in cui non doveva cambiare.
Z_VOXELS_DEFAULT = 64

# float16: 64^3 x 2 byte = 512 KB per esame, 0,7 GB per i 1362. In float32
# sarebbero 1,4 GB, e la precisione in piu' non serve - il rapporto sul
# riferimento sta in [0, ~10] e le SPECT hanno risoluzione effettiva 8-10 mm FWHM.
STORE_DTYPE = np.float16


def process(path: Path, z_voxels: int = Z_VOXELS_DEFAULT) -> dict:
    """Da NIfTI a crop normalizzato, piu' i controlli di qualita'.

    **Il crop viene da `pipeline.process_file`, non da qui.** Fino al 2 agosto
    questa funzione ripercorreva a mano i primi passi della pipeline - stessa
    maschera in spazio nativo, stesso ritaglio, stesso ricampionamento - con la
    giustificazione che erano "le tre righe che precedono la divergenza". Era
    accettabile finche' il crop serviva a un esperimento; nel momento in cui
    entra in inferenza smette di esserlo, perche' due copie dello stesso
    preprocessing sono la divergenza silenziosa contro cui `pipeline.py` esiste.

    Che la sostituzione sia neutra non e' un'opinione: il cancello A0 di
    `11_allinea_crop.py` confronta il dataset riestratto con quello precedente
    **bit per bit**.
    """
    features, crop = process_file(path, with_crop=True, crop_z_voxels=z_voxels)
    return {
        "crop": crop.astype(STORE_DTYPE),
        "ancore_dentro": int(features["_crop_anchors_inside"]),
        "frazione_non_nulla": float((crop > 0).mean()),
        "massimo": float(crop.max()),
        "centro_z": features["_crop_center_z"],
        "maschera_z_lo": features["_crop_mask_z_lo"],
        "maschera_z_hi": features["_crop_mask_z_hi"],
        "maschera_z_p95": features["_crop_mask_z_p95"],
        "localizzazione": features["localization_confidence"],
        **{f"ancora_{n}_{a}": features[f"_crop_anchor_{n}_{a}"]
           for n in ("sx", "dx") for a in "xyz"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=Path("features_v4.csv"),
                        help="da cui prendere uid ed etichette, nello stesso ordine")
    parser.add_argument("--out", type=Path, default=Path("crops"))
    parser.add_argument("--z-voxels", type=int, default=Z_VOXELS_DEFAULT,
                        help="fette in z; >64 lascia margine per ricentrare dopo")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.z_voxels % 2 or args.z_voxels < CROP_VOXELS:
        print(f"ERRORE: --z-voxels dev'essere pari e >= {CROP_VOXELS}", file=sys.stderr)
        return 1

    frame = pd.read_csv(args.features)
    if args.limit:
        frame = frame.head(args.limit)
    n = len(frame)
    args.out.mkdir(parents=True, exist_ok=True)

    # Scritto su memmap invece che in RAM: 0,7 GB ci starebbero, ma il consumatore
    # e' un dataloader che legge a batch, e un file gia' su disco si apre in
    # sola lettura da piu' processi senza copiarlo.
    store = np.lib.format.open_memmap(
        args.out / "crops.npy", mode="w+",
        dtype=STORE_DTYPE, shape=(n, CROP_VOXELS, CROP_VOXELS, args.z_voxels),
    )

    righe, started = [], time.time()
    for i, uid in enumerate(frame["uid"]):
        try:
            r = process(args.data_dir / "niftis" / f"{uid}.nii.gz", args.z_voxels)
            store[i] = r.pop("crop")
            righe.append({"uid": uid, "ok": True, **r})
        except Exception as exc:  # noqa: BLE001
            print(f"  {uid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            righe.append({"uid": uid, "ok": False, "ancore_dentro": 0,
                          "frazione_non_nulla": 0.0, "massimo": 0.0})
        if (i + 1) % 100 == 0:
            rate = (time.time() - started) / (i + 1)
            print(f"  {i+1}/{n}  {rate:.2f} s/esame  resta {(n-i-1)*rate/60:.0f} min",
                  flush=True)
    store.flush()

    qc = pd.DataFrame(righe)
    entrambe = int((qc["ancore_dentro"] == 2).sum())
    print(f"\n{n} crop in {(time.time()-started)/60:.0f} min -> {args.out}/crops.npy "
          f"({(args.out / 'crops.npy').stat().st_size/1e9:.2f} GB)")
    print(f"  estratti senza errori        {int(qc['ok'].sum())}/{n}")
    print(f"  con ENTRAMBE le ancore dentro {entrambe}/{n} ({entrambe/n:.1%})")
    print(f"  con una sola                  {int((qc['ancore_dentro']==1).sum())}")
    print(f"  con nessuna                   {int((qc['ancore_dentro']==0).sum())}")
    print(f"  frazione di voxel non nulli, mediana {qc['frazione_non_nulla'].median():.3f}")
    print(f"  massimo del rapporto, mediana        {qc['massimo'].median():.2f}")

    qc.to_csv(args.out / "qc.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps({
        "n": n, "voxel": CROP_VOXELS, "z_voxel": args.z_voxels,
        "spacing_mm": TARGET_SPACING_MM,
        "lato_mm": CROP_VOXELS * TARGET_SPACING_MM,
        "dtype": str(np.dtype(STORE_DTYPE)),
        "centratura": "baricentro della maschera cerebrale",
        "normalizzazione": "diviso il riferimento occipitale del paziente",
        "ancore_entrambe_dentro": entrambe,
        "features": str(args.features),
    }, indent=2))

    if entrambe < n:
        print("\nATTENZIONE: qualche crop non contiene entrambe le ancore striatali.")
        print("Vedi qc.csv. Sopra una quota trascurabile, va allargato CROP_VOXELS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
