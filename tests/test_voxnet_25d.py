"""Collaudo del solo arco MIP preaddestrato usato dalla submission."""

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import torch
    import torchvision
    C_E_TORCH = True
except ImportError:
    C_E_TORCH = False

if __name__ != "__main__" and not C_E_TORCH:
    import pytest
    pytest.skip("test MIP da eseguire nell'immagine Docker", allow_module_level=True)

from src.voxnet import ARCHITETTURE_25D, VoxelEnsemble  # noqa: E402
from src.voxnet import build_model, carica_imagenet  # noqa: E402


def test_griglia_di_fusione_e_forte_e_prespecificata():
    from src.model import FUSION_LOGISTIC_C_GRID, LOGISTIC_C_GRID

    assert FUSION_LOGISTIC_C_GRID == (0.1, 0.3)
    assert max(FUSION_LOGISTIC_C_GRID) < min(LOGISTIC_C_GRID)


def _salta():
    if not C_E_TORCH:
        print("      (saltato: torch assente, esegui dentro l'immagine)")
        return True
    return False


def test_il_registro_contiene_solo_l_arco_attivo():
    assert set(ARCHITETTURE_25D) == {"premip"}


def test_build_model_non_scarica_pesi():
    if _salta():
        return
    vero = torch.hub.load_state_dict_from_url

    def vietato(*_args, **_kwargs):
        raise AssertionError("build_model ha provato a scaricare dei pesi")

    torch.hub.load_state_dict_from_url = vietato
    try:
        model = build_model("premip")
    finally:
        torch.hub.load_state_dict_from_url = vero
    assert model.testa[1].in_features == 3 * 256


def test_premip_restituisce_un_logit_per_esame():
    if _salta():
        return
    model = build_model("premip").eval()
    x = torch.randn(2, 1, 32, 40, 48)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1)
    assert torch.isfinite(y).all()


def test_state_dict_ricaricabile_senza_ambiguita():
    if _salta():
        return
    origine = build_model("premip").eval()
    copia = build_model("premip").eval()
    copia.load_state_dict(origine.state_dict(), strict=True)
    x = torch.randn(2, 1, 24, 28, 32)
    with torch.no_grad():
        assert torch.allclose(origine(x), copia(x), atol=1e-6)


def test_carica_imagenet_copia_i_primi_tre_stadi():
    if _salta():
        return
    with tempfile.TemporaryDirectory() as d:
        radice = pathlib.Path(d)
        try:
            carica_imagenet(build_model("premip"), radice)
            raise AssertionError("pesi assenti: doveva alzare")
        except SystemExit:
            pass

        torch.manual_seed(1)
        resnet = torchvision.models.resnet18(weights=None)
        (radice / "hub").mkdir()
        torch.save(resnet.state_dict(), radice / "hub" / "resnet18-test.pth")
        model = build_model("premip")
        n = carica_imagenet(model, radice)
        assert torch.allclose(model.stem[0].weight, resnet.conv1.weight)
        assert torch.allclose(model.strati[2][0].conv1.weight,
                              resnet.layer3[0].conv1.weight)
        assert 2e6 < n < 4e6


def test_voxel_ensemble_riproduce_la_rete_mip():
    if _salta():
        return
    import numpy as np

    model = build_model("premip").eval()
    ensemble = VoxelEnsemble({
        "arch": "premip", "norm": "batch", "in_ch": 1,
        "dropout": 0.3, "riduzione": 1, "tta": False,
        "autonorm": False,
    }, [model.state_dict()], "cpu")
    x = np.random.default_rng(0).standard_normal((2, 32, 32, 32)).astype("float32")
    p = ensemble.predici(x)
    with torch.no_grad():
        atteso = torch.sigmoid(model(torch.from_numpy(x)[:, None]).ravel()).numpy()
    assert np.allclose(p, atteso, atol=1e-5)


if __name__ == "__main__":
    test = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    falliti = 0
    for nome, funzione in test:
        try:
            funzione()
            print(f"  PASS  {nome}")
        except Exception as exc:
            falliti += 1
            print(f"  FAIL  {nome}: {type(exc).__name__}: {exc}")
    print(f"\n{len(test) - falliti}/{len(test)} test superati")
    raise SystemExit(1 if falliti else 0)
