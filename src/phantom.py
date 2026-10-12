"""Generatore di volumi DaT sintetici, per collaudare la pipeline senza dati reali.

Serve a due scopi:

1. Verificare che la localizzazione striatale e le feature si comportino come
   previsto - un phantom "patologico" deve produrre SBR piu' basso e asimmetria
   piu' alta di uno "normale". Se un test cosi' fallisce, c'e' un bug nella
   geometria o negli assi.
2. Permettere di eseguire l'intera pipeline end-to-end prima che i dati reali
   siano scaricati.

Il phantom non pretende realismo anatomico: riproduce solo le tre proprieta'
da cui dipende la pipeline, cioe' testa approssimativamente ellissoidale,
due strutture striatali molto piu' brillanti del fondo, e la possibilita' di
degradare selettivamente il putamen posteriore su un lato.
"""

from __future__ import annotations

import numpy as np


def _ellipsoid(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    gx, gy, gz = grids
    return (
        ((gx - center[0]) / radii[0]) ** 2
        + ((gy - center[1]) / radii[1]) ** 2
        + ((gz - center[2]) / radii[2]) ** 2
    )


def make_phantom(
    shape: tuple[int, int, int] = (64, 64, 48),
    spacing_mm: float = 2.0,
    left_caudate: float = 1.0,
    left_putamen: float = 1.0,
    right_caudate: float = 1.0,
    right_putamen: float = 1.0,
    background: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
    rotation_deg: float = 0.0,
    offset_voxels: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Costruisce un volume DaT sintetico in orientamento RAS.

    I quattro parametri di uptake sono moltiplicatori relativi: 1.0 rappresenta
    un uptake normale, 0.2 una perdita marcata. Il fondo rappresenta il tessuto
    di riferimento privo di trasportatori.

    Args:
        shape: dimensioni del volume in voxel, assi (x=L->R, y=P->A, z=I->S).
        spacing_mm: spacing isotropo.
        left_caudate: moltiplicatore di uptake del caudato sinistro.
        left_putamen: moltiplicatore di uptake del putamen sinistro.
        right_caudate: moltiplicatore di uptake del caudato destro.
        right_putamen: moltiplicatore di uptake del putamen destro.
        background: livello del tessuto di riferimento.
        noise_sigma: deviazione standard del rumore gaussiano additivo.
        seed: seme del generatore di rumore.

    Returns:
        Volume float32 non negativo.
    """
    grids = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )

    cx = (shape[0] - 1) / 2.0 + offset_voxels[0]
    cy = (shape[1] - 1) / 2.0 + offset_voxels[1]
    cz = (shape[2] - 1) / 2.0 + offset_voxels[2]
    volume = np.zeros(shape, dtype=np.float32)

    # Testa: ellissoide riempito con il livello di fondo.
    head = _ellipsoid(
        grids,
        (cx, cy, cz),
        (shape[0] * 0.42, shape[1] * 0.46, shape[2] * 0.42),
    ) <= 1.0
    volume[head] = background

    # Offset in voxel delle quattro sottostrutture striatali rispetto al centro.
    offset_lateral = 9.0 / spacing_mm * 2.0  # separazione dalla linea mediana
    offset_anterior = 7.0 / spacing_mm  # caudato anteriore rispetto al putamen
    radii = (5.0 / spacing_mm, 5.0 / spacing_mm, 5.0 / spacing_mm)

    # Il caudato e' anteromediale, il putamen posterolaterale: il putamen sta
    # piu' lontano dalla linea mediana. Non e' un dettaglio decorativo, e' la
    # firma geometrica che permette di dedurre la direzione anteriore dai dati
    # invece che dall'header (vedi striatum.infer_anterior_sign).
    medial_offset = offset_lateral * 0.72
    lateral_offset = offset_lateral * 1.18

    structures = [
        # (offset x, offset y, moltiplicatore)
        (-medial_offset, +offset_anterior, left_caudate),
        (-lateral_offset, -offset_anterior, left_putamen),
        (+medial_offset, +offset_anterior, right_caudate),
        (+lateral_offset, -offset_anterior, right_putamen),
    ]

    # Inclinazione della testa nel piano assiale. Nella pratica clinica il
    # paziente non e' mai allineato perfettamente agli assi dello scanner.
    if rotation_deg:
        angle = np.deg2rad(rotation_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        structures = [
            (dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a, uptake)
            for dx, dy, uptake in structures
        ]

    for dx, dy, uptake in structures:
        # Profilo gaussiano invece che a gradino: la SPECT ha una risoluzione
        # effettiva di ~8-10 mm FWHM, quindi non produce mai bordi netti.
        squared = _ellipsoid(grids, (cx + dx, cy + dy, cz), radii)
        blob = np.exp(-1.5 * squared).astype(np.float32)
        volume += (background * 7.0 * uptake) * blob

    volume *= head  # azzera tutto fuori dalla testa

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        volume = volume + rng.normal(0.0, noise_sigma, size=shape).astype(np.float32)

    return np.clip(volume, 0.0, None).astype(np.float32)


def normal_scan(**kwargs) -> np.ndarray:
    """Phantom con uptake normale e simmetrico."""
    return make_phantom(**kwargs)


def pathologic_scan(severity: float = 0.75, **kwargs) -> np.ndarray:
    """Phantom parkinsoniano: perdita asimmetrica a esordio putaminale.

    Riproduce il gradiente clinico: il putamen perde piu' del caudato, e un lato
    piu' dell'altro.
    """
    loss = float(np.clip(severity, 0.0, 1.0))
    return make_phantom(
        left_caudate=1.0 - 0.35 * loss,
        left_putamen=1.0 - 0.90 * loss,
        right_caudate=1.0 - 0.10 * loss,
        right_putamen=1.0 - 0.40 * loss,
        **kwargs,
    )
