"""Contratto fra pacchettatore, manifest e runtime di gara."""

import importlib.util
import json
import pathlib
import pickle
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack():
    return _load(ROOT / "scripts" / "03_pack_submission.py", "pack_contract")


def _runtime():
    return _load(ROOT / "submission_src" / "main.py", "runtime_contract")


def test_cli_obbliga_a_scegliere_il_ramo_e_il_metro(tmp_path):
    pack = _pack()
    with pytest.raises(SystemExit):
        pack.parse_args(["--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit):
        pack.parse_args([
            "--out", str(tmp_path / "out"),
            "--voxnet", str(tmp_path / "rete.pt"),
        ])

    args = pack.parse_args(["--out", str(tmp_path / "out"), "--senza-voxnet"])
    assert args.senza_voxnet and args.voxnet is None and args.metro is None


def test_manifest_controlla_i_byte_dentro_lo_zip(tmp_path):
    pack = _pack()
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("pass\n", encoding="utf-8")
    (source / "LICENSE").write_text("MIT\n", encoding="utf-8")
    model = source / "models.pkl"
    model.write_bytes(b"modello esatto")
    manifest = {
        "format": pack.MANIFEST_FORMAT,
        "source": {"git_commit": "test", "git_dirty": False},
        "models": pack.file_identity(model, "models.pkl"),
        "voxel": None,
    }
    (source / pack.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    archive = tmp_path / "submission.zip"
    pack.write_zip(source, archive)
    pack.check_manifest(archive)

    # Lo ZIP resta formalmente valido, ma i byte non sono piu' quelli autorizzati.
    model.write_bytes(b"modello sostituito")
    pack.write_zip(source, archive)
    with pytest.raises(SystemExit, match="non coincide"):
        pack.check_manifest(archive)


def test_runtime_rifiuta_un_models_pkl_sostituito(tmp_path):
    runtime = _runtime()
    model = tmp_path / "models.pkl"
    model.write_bytes(pickle.dumps({"columns": ["synthetic_feature"]}))
    runtime.MODEL_PATH = model
    identity = {
        "file": "models.pkl",
        "bytes": model.stat().st_size,
        "sha256": runtime.sha256_file(model),
    }
    manifest = {"format": runtime.MANIFEST_FORMAT, "models": identity, "voxel": None}
    assert "columns" in runtime.load_models(manifest)

    with open(model, "ab") as handle:
        handle.write(b"vecchio-o-manomesso")
    with pytest.raises(ValueError, match="diversa dal manifest"):
        runtime.load_models(manifest)


def test_drift_senza_uid_comuni_non_puo_passare():
    verify = _load(ROOT / "tests" / "verify_real.py", "verify_contract")
    current = {
        "uids": ["nuovo-1", "nuovo-2"],
        "cost": [
            {"uid": "nuovo-1", "fov_mm": 100.0},
            {"uid": "nuovo-2", "fov_mm": 110.0},
        ],
        "features": {},
    }
    baseline = {"uids": ["vecchio-1", "vecchio-2"], "features": {}}
    ok, problems = verify.report_drift(current, baseline)
    assert not ok
    assert problems and "UID comuni" in problems[0]
