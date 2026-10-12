import numpy as np

from src.alignment import align_striata, registered_symmetry_features, rotation_to_x


def test_rotation_to_x_allinea_la_congiungente():
    vector = np.array([12.0, 3.0, -2.0])
    rotated = rotation_to_x(vector) @ vector
    assert rotated[0] > 0
    assert np.allclose(rotated[1:], 0.0, atol=1e-10)


def test_allineamento_porta_due_picchi_alla_stessa_quota():
    from scipy.ndimage import center_of_mass

    crop = np.zeros((64, 64, 64), dtype=np.float32)
    left = np.array([20.0, 38.0, 31.0])
    right = np.array([44.0, 44.0, 35.0])
    crop[tuple(left.astype(int))] = 10.0
    crop[tuple(right.astype(int))] = 10.0
    aligned = align_striata(crop, left, right)
    center = np.asarray(center_of_mass(aligned))
    assert np.allclose(center, [32.0, 42.0, 34.0], atol=0.6)


def test_descrittori_riconoscono_un_volume_simmetrico():
    grid = np.indices((64, 64, 64), dtype=np.float32)
    left = np.array([22.0, 42.0, 34.0])
    right = np.array([42.0, 42.0, 34.0])
    crop = np.ones((64, 64, 64), dtype=np.float32)
    for center in (left, right):
        d2 = sum((grid[i] - center[i]) ** 2 for i in range(3))
        crop += 3.0 * np.exp(-d2 / (2 * 3.0**2))
    features = registered_symmetry_features(crop, left, right)
    assert features["reg_asym_l1_s1p0"] < 1e-4
    assert features["reg_asym_energy_s1p0"] < 1e-4
    assert features["reg_sym_corr_s1p0"] > 0.999
