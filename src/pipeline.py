"""Estrazione feature da un singolo file NIfTI: punto di ingresso condiviso.

Training e inferenza DEVONO passare da qui. Se il preprocessing divergesse anche
di poco tra le due fasi, il modello vedrebbe in produzione una distribuzione
diversa da quella su cui e' stato allenato, e il divario non comparirebbe in
cross-validation: si scoprirebbe solo sul punteggio della leaderboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cv import scanner_fingerprint
from .alignment import SYMMETRY_FEATURE_NAMES, registered_symmetry_features
from .features import build_features
from .striatum import brain_mask, measure
from .volume import bounding_box, resample_isotropic, resample_mask

TARGET_SPACING_MM = 2.0

# Lato del crop striatale dato alla rete a voxel, in voxel da 2 mm: 128 mm.
# Lo striato ne occupa ~50 di larghezza e ~40 in antero-posteriore, quindi il
# margine e' abbondante di proposito - la rete deve vedere il contesto, perche'
# e' rispetto a quello che la perdita striatale si legge.
CROP_VOXELS = 64

def crop_centered(volume: np.ndarray, center: np.ndarray, sizes) -> np.ndarray:
    """Ritaglio attorno a `center`, con riempimento a zero ai bordi.

    `sizes` e' uno scalare (cubo) o una terna per asse. Il riempimento e' zero e
    non un riflesso: fuori dalla testa c'e' aria, che nel volume vale zero,
    quindi lo zero e' il valore fisicamente corretto. Un riflesso inventerebbe
    tessuto dove non ce n'e', e la rete imparerebbe la forma del bordo del campo
    di vista - cioe' il centro di provenienza.
    """
    dims = (int(sizes),) * 3 if np.isscalar(sizes) else tuple(int(s) for s in sizes)
    out = np.zeros(dims, dtype=np.float32)
    start = np.round(center).astype(int) - np.array(dims) // 2
    src, dst = [], []
    for axis in range(3):
        lo = max(start[axis], 0)
        hi = min(start[axis] + dims[axis], volume.shape[axis])
        if hi <= lo:
            return out
        src.append(slice(lo, hi))
        dst.append(slice(lo - start[axis], hi - start[axis]))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def process_file(
    path: str | Path,
    target_mm: float = TARGET_SPACING_MM,
    with_crop: bool = False,
    crop_z_voxels: int = CROP_VOXELS,
):
    """Da percorso NIfTI a dizionario di feature, metadati inclusi.

    L'ordine delle operazioni non e' arbitrario: la maschera cerebrale si
    calcola in spazio nativo sul volume INTERO, e solo dopo si ritaglia. Il
    contrario - ritagliare e poi mascherare - sarebbe piu' veloce da scrivere e
    introdurrebbe un errore sottile: la soglia di Otsu dipende dall'istogramma,
    l'istogramma dalla quantita' di aria nel campo di vista, e il campo di vista
    varia da 190 a 630 mm fra i centri. La distorsione risultante sarebbe
    correlata al centro di provenienza, cioe' proprio l'asse su cui il modello
    deve generalizzare. Misurata: fino al 2,8% sul riferimento, e solo sugli
    scanner a FOV largo.

    I metadati di acquisizione (shape e spacing originali) vengono conservati:
    servono a costruire l'impronta dello scanner usata per raggruppare i fold.
    Non vanno mai passati al modello come feature - identificherebbero il centro,
    e il modello imparerebbe la prevalenza per centro invece della patologia.

    `with_crop` — il crop del ramo a voxel esce da QUI, e non da uno script
    -------------------------------------------------------------------------
    Il crop condivide con le feature i primi tre passi: maschera in spazio
    nativo, ritaglio al bounding box, ricampionamento isotropo. Fino al 2 agosto
    quei passi erano **duplicati** in `scripts/08_extract_crops.py`, ed era
    accettabile finche' il crop serviva solo a un esperimento. Nel momento in cui
    entra in inferenza smette di esserlo: due copie dello stesso preprocessing
    che divergono in silenzio sono esattamente il guasto che questo modulo esiste
    per rendere impossibile, e la cross-validation non lo vedrebbe perche' la
    esegue una delle due.

    Restituisce quindi `(features, crop)` invece del solo dizionario. Il crop e'
    normalizzato sul **riferimento occipitale del paziente stesso**: ogni voxel
    diventa un rapporto interno al singolo esame, con la stessa immunita' al
    centro dell'SBR e la stessa conformita' alla regola sull'indipendenza degli
    esami di test. Ed e' centrato sul **baricentro della maschera cerebrale**,
    non sull'ancora striatale, perche' l'ancora deriva dall'intensita' e si
    sposta quando l'uptake degrada.

    Costa quasi nulla in piu': i passi condivisi sono gia' stati eseguiti, e
    quello che resta e' una divisione e un affettamento.

    """
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
    original_shape = tuple(int(s) for s in data.shape[:3])

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)

    mask = brain_mask(data, tuple(float(z) for z in zooms))
    box = bounding_box(mask, zooms)
    if box is not None:
        data, mask = data[box], mask[box]

    resampled = resample_isotropic(data, zooms, target_mm=target_mm)
    resampled_mask = resample_mask(mask, zooms, target_mm=target_mm)

    measurement = measure(resampled, target_mm, mask=resampled_mask)
    features = build_features(measurement)
    features["_group"] = scanner_fingerprint(original_shape, tuple(float(z) for z in zooms))
    features["_orig_shape"] = "x".join(str(s) for s in original_shape)
    features["_orig_spacing"] = float(zooms[0])

    # Diagnostica sul campo di vista, non feature: serve a misurare su tutti gli
    # esami quanto il contenuto della maschera - e quindi della regione di
    # riferimento - dipenda da quanto in alto e in basso ha acquisito il centro.
    # Il prefisso "_" le tiene fuori dal modello per costruzione.
    features["_fov_mm"] = float(np.max(np.asarray(original_shape) * zooms))
    features["_mask_cm3"] = float(resampled_mask.sum()) * target_mm**3 / 1000.0
    if resampled_mask.any():
        z_indices = np.nonzero(resampled_mask)[2]
        features["_mask_extent_z_mm"] = float(np.ptp(z_indices) + 1) * target_mm
    else:
        features["_mask_extent_z_mm"] = float("nan")
    features["_reference_slab_ratio"] = (
        float(measurement.reference_slab / measurement.reference_mean)
        if measurement.reference_mean > 0
        else float("nan")
    )
    if not with_crop:
        return features
    # Il crop non solleva: restituisce None. Un riferimento occipitale inservibile
    # rende inservibile anche l'SBR, quindi in pratica quell'esame e' perso in
    # ogni caso — ma far fallire l'intera `process_file` per il ramo a voxel
    # significherebbe che un difetto del componente *nuovo* butta via anche la
    # predizione del componente che funziona. Il chiamante decide, e in inferenza
    # decide di ripiegare sul solo modello SBR.
    if not np.isfinite(measurement.reference_mean) or measurement.reference_mean <= 0:
        return features, None

    where = np.array(np.nonzero(resampled_mask), dtype=np.float64)
    center = where.mean(axis=1)
    dims = (CROP_VOXELS, CROP_VOXELS, int(crop_z_voxels))
    crop = crop_centered(resampled / measurement.reference_mean, center, dims)

    # Geometria del ritaglio, con il prefisso "_" che la tiene fuori dal modello
    # per costruzione, come ogni altro metadato. Serve al controllo di qualita'
    # dell'estrazione — le ancore striatali devono cadere DENTRO il crop, e un
    # crop mutilato ha esattamente la stessa forma di uno buono — e agli studi
    # sull'allineamento. L'inferenza la ignora: seleziona le colonne per nome.
    origin = np.round(center).astype(int) - np.array(dims) // 2
    features["_crop_center_z"] = float(center[2] - origin[2])
    features["_crop_mask_z_lo"] = float(where[2].min() - origin[2])
    features["_crop_mask_z_hi"] = float(where[2].max() - origin[2])
    features["_crop_mask_z_p95"] = float(np.quantile(where[2], 0.95) - origin[2])
    inside = 0
    relative_anchors = {}
    for name, hemi in (("sx", measurement.left), ("dx", measurement.right)):
        if hemi is None or hemi.centroid is None:
            for axis in "xyz":
                features[f"_crop_anchor_{name}_{axis}"] = float("nan")
            continue
        rel = np.asarray(hemi.centroid, dtype=np.float64) - origin
        relative_anchors[name] = rel
        inside += int(np.all(rel >= 0) and np.all(rel < np.array(dims)))
        for i, axis in enumerate("xyz"):
            features[f"_crop_anchor_{name}_{axis}"] = float(rel[i])
    features["_crop_anchors_inside"] = float(inside)
    # Il blocco registrato serve soltanto al ramo tabellare che affianca la
    # CNN. Un difetto geometrico non deve mai buttare via il crop o il modello
    # SBR: in quel caso l'imputer del bundle riceve NaN.
    try:
        features.update(registered_symmetry_features(
            # Il dataset di training dei crop e' float16; quantizzare soltanto
            # la copia destinata ai descrittori rende training e inferenza
            # bit-riproducibili senza degradare il crop float32 dato alla CNN.
            crop.astype(np.float16), relative_anchors["sx"], relative_anchors["dx"]
        ))
    except (KeyError, ValueError, FloatingPointError):
        features.update(dict.fromkeys(SYMMETRY_FEATURE_NAMES, float("nan")))
    return features, crop
