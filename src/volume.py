"""Caricamento e normalizzazione geometrica dei volumi DaT SPECT.

Il dataset arriva da 10 centri diversi: dimensioni (128x128x128, 142x142x112, ...)
e spacing (2.46 mm, 3.895 mm, ...) variano da scan a scan. Tutto il resto della
pipeline assume un volume gia' portato in orientamento canonico e spacing isotropo
comune, quindi questo modulo e' l'unico punto in cui la geometria viene gestita.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Volume:
    """Volume 3D con spacing isotropo noto.

    Attributes:
        data: array float32 di shape (X, Y, Z) in orientamento RAS canonico.
        spacing_mm: spacing isotropo in millimetri.
    """

    data: np.ndarray
    spacing_mm: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Legge un .nii.gz e lo riporta in orientamento canonico RAS.

    Ritorna (data float32, zooms) dove zooms sono i 3 spacing in mm.

    L'orientamento canonico e' essenziale: senza, sinistra e destra possono
    risultare scambiate tra centri diversi, e tutte le feature di asimmetria
    (che sono le piu' informative per il parkinsonismo) diventerebbero rumore.
    """
    import nibabel as nib

    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)  # porta in RAS+
    data = np.asarray(img.dataobj, dtype=np.float32)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
    return data, zooms


def resample_isotropic(
    data: np.ndarray, zooms: np.ndarray, target_mm: float = 2.0
) -> np.ndarray:
    """Ricampiona a spacing isotropo `target_mm` con interpolazione trilineare.

    Usa order=1: le SPECT DaT sono intrinsecamente lisce (risoluzione effettiva
    ~8-10 mm FWHM), quindi interpolazioni di ordine superiore introdurrebbero
    ringing senza aggiungere informazione reale.
    """
    from scipy.ndimage import zoom as ndzoom

    factors = tuple(float(z) / float(target_mm) for z in zooms)
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return np.ascontiguousarray(data, dtype=np.float32)
    out = ndzoom(data, factors, order=1, mode="nearest", prefilter=False)
    return np.ascontiguousarray(out, dtype=np.float32)


def resample_mask(
    mask: np.ndarray, zooms: np.ndarray, target_mm: float = 2.0
) -> np.ndarray:
    """Ricampiona una maschera booleana con interpolazione al vicino piu' prossimo.

    order=0 e non order=1: interpolare linearmente una maschera produrrebbe
    valori intermedi da risogliare, e la soglia sarebbe una costante arbitraria
    in piu'. Al vicino piu' prossimo la maschera resta booleana per costruzione.

    Usa gli stessi fattori di `resample_isotropic`, quindi a parita' di forma in
    ingresso le due funzioni producono la stessa forma in uscita.
    """
    factors = tuple(float(z) / float(target_mm) for z in zooms)
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return np.ascontiguousarray(mask, dtype=bool)
    from scipy.ndimage import zoom as ndzoom

    out = ndzoom(mask.astype(np.uint8), factors, order=0, mode="nearest", prefilter=False)
    return np.ascontiguousarray(out, dtype=bool)


# Margine attorno alla testa nel ritaglio, in millimetri. Deve superare il
# raggio del filtro gaussiano applicato dopo (3 sigma = 12 mm a sigma 4 mm),
# altrimenti i voxel di bordo della testa verrebbero filtrati contro il bordo
# replicato dell'array invece che contro il fondo reale, e le misure ai margini
# cambierebbero rispetto al volume non ritagliato.
CROP_PAD_MM = 20.0


def bounding_box(
    mask: np.ndarray, zooms: np.ndarray, pad_mm: float = CROP_PAD_MM
) -> tuple[slice, ...] | None:
    """Riquadro che contiene la maschera piu' un margine, in indici di voxel.

    Ritaglia via l'aria: il campo di vista varia da 190 a 630 mm fra i centri, e
    su un FOV da 630 mm la testa occupa l'1,3% del volume. Ricampionare e
    filtrare il restante 99% costa il 90% del tempo e tutta la memoria di picco.

    Il ritaglio e' sicuro solo perche' la maschera cerebrale viene calcolata
    PRIMA, sul volume intero: se venisse ricalcolata dopo, l'istogramma amputato
    sposterebbe la soglia di Otsu in modo diverso a seconda del FOV originale,
    cioe' introdurrebbe una distorsione correlata al centro di provenienza.
    """
    if not mask.any():
        return None
    indices = np.nonzero(mask)
    pad = np.ceil(float(pad_mm) / np.asarray(zooms, dtype=np.float64)).astype(int)
    return tuple(
        slice(
            max(int(axis_idx.min()) - int(p), 0),
            min(int(axis_idx.max()) + 1 + int(p), int(dim)),
        )
        for axis_idx, p, dim in zip(indices, pad, mask.shape)
    )


def load_volume(path: str | Path, target_mm: float = 2.0) -> Volume:
    """Pipeline completa di caricamento: NIfTI -> RAS -> spacing isotropo."""
    data, zooms = load_nifti(path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)  # i conteggi SPECT non possono essere negativi
    resampled = resample_isotropic(data, zooms, target_mm=target_mm)
    return Volume(data=resampled, spacing_mm=float(target_mm))
