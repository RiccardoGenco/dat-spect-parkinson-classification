"""Allineamento rigido striatale e descrittori di simmetria voxel-wise.

Il crop e' gia' canonico RAS e isotropo. Restano pero' traslazioni e piccole
rotazioni fra pazienti che rendono rumoroso il confronto diretto fra emisferi.
Le due ancore striatali forniscono cinque gradi di liberta': tre traslazioni e
le due rotazioni necessarie ad allineare la congiungente sinistra-destra.

L'allineamento usa soltanto il singolo esame. Il riferimento geometrico e' una
costante fisica congelata, quindi in inferenza non dipende dagli altri casi.
"""

from __future__ import annotations

import numpy as np


TARGET_MIDPOINT = np.array([32.0, 42.0, 34.0], dtype=np.float64)
# 41 voxel su x: il piano x=32 resta al centro e non viene assegnato a nessun
# emisfero; x=31 si confronta con x=33, x=22 con x=42, eccetera.
ROI = (slice(12, 53), slice(24, 60), slice(20, 50))
SMOOTHING_SIGMAS: tuple[float, ...] = (1.0, 2.5)
SYMMETRY_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"reg_{measure}_s{str(sigma).replace('.', 'p')}"
    for sigma in SMOOTHING_SIGMAS
    for measure in ("asym_l1", "asym_energy", "sym_corr", "sym_dice", "asym_ap")
)


def rotation_to_x(vector: np.ndarray) -> np.ndarray:
    """Rotazione minima che porta ``vector`` sul semiasse x positivo."""
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    a = v / norm
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(a, b))
    if sine < 1e-10:
        return np.eye(3, dtype=np.float64) if cosine > 0 else np.diag([-1.0, -1.0, 1.0])
    axis = cross / sine
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sine * k + (1.0 - cosine) * (k @ k)


def align_striata(
    crop: np.ndarray,
    left_anchor: np.ndarray,
    right_anchor: np.ndarray,
    target_midpoint: np.ndarray = TARGET_MIDPOINT,
) -> np.ndarray:
    """Trasla e ruota un crop affinche' le ancore siano orizzontali e centrate."""
    from scipy.ndimage import affine_transform

    left = np.asarray(left_anchor, dtype=np.float64)
    right = np.asarray(right_anchor, dtype=np.float64)
    midpoint = (left + right) / 2.0
    rotation = rotation_to_x(right - left)

    # affine_transform vuole la mappa output -> input. Se
    # output = R @ (input - midpoint) + target, allora:
    # input = R.T @ output + midpoint - R.T @ target.
    inverse = rotation.T
    offset = midpoint - inverse @ np.asarray(target_midpoint, dtype=np.float64)
    return affine_transform(
        np.asarray(crop, dtype=np.float32), inverse, offset=offset,
        output_shape=tuple(int(v) for v in crop.shape), order=1,
        mode="constant", cval=0.0, prefilter=False,
    ).astype(np.float32, copy=False)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 4 or a.std() < 1e-10 or b.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def symmetry_features(aligned: np.ndarray) -> dict[str, float]:
    """Descrittori continui e binari fra i due emisferi, a due scale fisiche."""
    from scipy.ndimage import gaussian_filter

    volume = np.asarray(aligned, dtype=np.float32)[ROI]
    center = volume.shape[0] // 2
    width = min(center, volume.shape[0] - center - 1)
    out: dict[str, float] = {}

    for sigma in SMOOTHING_SIGMAS:
        # Il crop e' uptake/riferimento: sottrarre 1 isola il legame specifico.
        specific = np.clip(gaussian_filter(volume, sigma=sigma) - 1.0, 0.0, None)
        left = specific[center - width:center]
        right = specific[center + 1:center + 1 + width][::-1]
        total = float((left + right).sum()) + 1e-8
        tag = str(sigma).replace(".", "p")

        out[f"reg_asym_l1_s{tag}"] = float(np.abs(left - right).sum() / total)
        out[f"reg_asym_energy_s{tag}"] = float(abs(left.sum() - right.sum()) / total)

        active = (left + right) > np.quantile(left + right, 0.65)
        out[f"reg_sym_corr_s{tag}"] = _safe_corr(left[active], right[active])

        positive = specific[specific > 0]
        threshold = float(np.quantile(positive, 0.70)) if positive.size else np.inf
        ml, mr = left >= threshold, right >= threshold
        out[f"reg_sym_dice_s{tag}"] = float(
            2.0 * np.logical_and(ml, mr).sum() / (ml.sum() + mr.sum() + 1e-8)
        )

        # Confronto dei profili antero-posteriori dopo aver marginalizzato x,z.
        pl, pr = left.sum(axis=(0, 2)), right.sum(axis=(0, 2))
        out[f"reg_asym_ap_s{tag}"] = float(np.abs(pl - pr).sum() / (pl.sum() + pr.sum() + 1e-8))

    return out


def registered_symmetry_features(
    crop: np.ndarray, left_anchor: np.ndarray, right_anchor: np.ndarray
) -> dict[str, float]:
    """Punto d'ingresso unico usato sia dall'esperimento sia dall'inferenza."""
    return symmetry_features(align_striata(crop, left_anchor, right_anchor))
