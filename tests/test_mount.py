"""Collaudo delle difese di main.py contro il mount lento dei dati.

Perche' esiste
--------------
Gli organizzatori dichiarano che `/code_execution/data` impiega ~15-20 secondi a
diventare disponibile. Il modo in cui quella corsa si perde e' silenzioso: un
volume non ancora montato solleva un'eccezione, il gestore la degrada a
prevalenza, e la riga di diagnostica che lo direbbe viene filtrata dalla
piattaforma. Misurato sulla OOF, venti esami in fallback su ~500 valgono +0,016
di log loss - meta' del divario dal terzo posto, e indistinguibile da un modello
peggiore.

Le difese sono quindi state scritte contro un guasto che **non si vede**, ed e'
esattamente la categoria che marcisce senza che nessuno se ne accorga: un
refactoring che togliesse l'attesa non farebbe fallire nessun altro test, e il
prezzo si pagherebbe mesi dopo su una leaderboard. Da qui questo file.

I test end-to-end usano phantom generati da src/phantom.py e un modello
models/models.pkl allenato in locale. Quando il modello non e' presente,
pytest salta solo questi test; nessun dato della competizione entra nel repo.
"""
import importlib.util
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_SRC = ROOT / "submission_src" / "main.py"
MODEL = ROOT / "models" / "models.pkl"

# I phantom sono piccoli e l'inferenza gira 35 modelli per esame: tre bastano a
# distinguere "tutti", "nessuno" e "uno solo", che sono i casi che contano.
N_EXAMS = 3


def _main_module():
    """Carica submission_src/main.py come modulo, per i test di unita'."""
    spec = importlib.util.spec_from_file_location("submission_main", MAIN_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Unita': la guardia sul mount. Veloci, senza modello e senza sottoprocessi.
# --------------------------------------------------------------------------

def test_wait_returns_at_once_when_file_is_there():
    m = _main_module()
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "c'e.txt"
        p.write_bytes(b"contenuto")
        waited = m.wait_until_readable(p, timeout=5.0)
    assert 0 <= waited < 0.5, f"attesa inattesa su un file gia' presente: {waited}"


def test_wait_gives_up_and_says_so():
    m = _main_module()
    with tempfile.TemporaryDirectory() as d:
        waited = m.wait_until_readable(pathlib.Path(d) / "mai.txt", timeout=1.5)
    assert waited < 0, "un file che non arriva deve dare -1, non un'attesa valida"


def test_wait_survives_a_file_that_appears_late():
    m = _main_module()
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "tardi.txt"
        # Il timer va cancellato comunque vada: se l'assert fallisce, scriverebbe
        # in una directory temporanea gia' rimossa e la traccia di quel thread
        # seppellirebbe il messaggio del test. Il rumore in un fallimento e' un
        # costo reale - il fallimento e' l'unico momento in cui si legge.
        timer = threading.Timer(1.5, lambda: p.write_bytes(b"eccomi"))
        timer.start()
        try:
            waited = m.wait_until_readable(p, timeout=10.0)
        finally:
            timer.cancel()
    assert waited > 0, "l'attesa non ha atteso"
    assert waited < 6, f"ha atteso troppo oltre l'arrivo del file: {waited}"


def test_empty_file_does_not_pass_the_guard():
    """Il caso per cui la guardia legge un byte invece di usare exists().

    Su un mount in corso l'entrata di directory puo' comparire prima del
    contenuto. Una guardia basata su `exists()` - o su una `open()` riuscita -
    passerebbe su un file da zero byte, cioe' proprio nella situazione da cui
    deve difendere, e il guasto si sposterebbe piu' avanti dove non e' piu'
    riconoscibile come un problema di mount.
    """
    m = _main_module()
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "vuoto.txt"
        p.write_bytes(b"")
        assert m.wait_until_readable(p, timeout=1.0) < 0, \
            "un file da zero byte ha superato la guardia"
        # E appena il contenuto arriva, la guardia deve sbloccarsi.
        p.write_bytes(b"ora c'e'")
        assert m.wait_until_readable(p, timeout=1.0) >= 0, \
            "il file e' pieno ma la guardia non passa"


# --------------------------------------------------------------------------
# End-to-end: main.py vero, in un sottoprocesso, contro dati che arrivano tardi.
# --------------------------------------------------------------------------

def _stage(work: pathlib.Path) -> pathlib.Path:
    """Replica la struttura dell'archivio: main.py, models.pkl, src/."""
    pack = work / "pack"
    pack.mkdir(parents=True)
    shutil.copy2(MAIN_SRC, pack / "main.py")
    shutil.copy2(MODEL, pack / "models.pkl")
    shutil.copytree(ROOT / "src", pack / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    payload = (pack / "models.pkl").read_bytes()
    (pack / "artifact_manifest.json").write_text(json.dumps({
        "format": "dat-parkinson-artifact/v1",
        "source": {"git_commit": "test", "git_dirty": False},
        "models": {
            "file": "models.pkl",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "voxel": None,
    }), encoding="utf-8")
    return pack


def _write_phantom(path: pathlib.Path) -> None:
    import nibabel as nib
    from src.phantom import normal_scan

    path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.asarray(normal_scan(), dtype=np.float32)
    nib.save(nib.Nifti1Image(volume, np.diag([2.0, 2.0, 2.0, 1.0])), path)


def _data_dir(work: pathlib.Path, uids, with_format=True, with_volumes=()) -> pathlib.Path:
    data = work / "data"
    (data / "niftis").mkdir(parents=True)
    if with_format:
        (data / "submission_format.csv").write_text(
            "uid,is_pathologic\n" + "".join(f"{u},0.5\n" for u in uids)
        )
    for uid in with_volumes:
        _write_phantom(data / "niftis" / f"{uid}.nii.gz")
    return data


def _run(pack: pathlib.Path, data: pathlib.Path, work: pathlib.Path):
    out = work / "out"
    out.mkdir()
    proc = subprocess.run(
        [sys.executable, str(pack / "main.py")], cwd=out,
        capture_output=True, text=True,
        env=dict(os.environ, DAT_DATA_DIR=str(data)),
    )
    log = proc.stdout + proc.stderr
    csv = out / "submission.csv"
    rows = csv.read_text().strip().splitlines()[1:] if csv.exists() else None
    return proc.returncode, log, rows


@pytest.mark.skipif(not MODEL.is_file(), reason="requires a locally trained models.pkl")
def test_slow_volumes_do_not_become_silent_fallbacks():
    """Il guasto principale: i volumi arrivano dopo, e senza attesa vale il 100%.

    Prima della difesa questo identico scenario produceva exit code 0, un
    submission.csv completo, e tutte le predizioni alla prevalenza fissa.
    """
    uids = [f"phantom{i}" for i in range(N_EXAMS)]
    with tempfile.TemporaryDirectory() as d:
        work = pathlib.Path(d)
        pack = _stage(work)
        data = _data_dir(work, uids)

        def late():
            time.sleep(4)
            try:
                for uid in uids:
                    _write_phantom(data / "niftis" / f"{uid}.nii.gz")
            except OSError:
                pass  # il test e' gia' finito e la temporanea rimossa: vedi sopra

        threading.Thread(target=late, daemon=True).start()
        code, log, rows = _run(pack, data, work)

    assert code == 0, f"exit non zero: {log[-400:]}"
    assert rows is not None and len(rows) == N_EXAMS, "submission.csv assente o incompleta"
    assert "elaborato il 100.0%" in log, f"non ha atteso i volumi:\n{log}"
    assert "fallback 0.0%" in log, f"qualche esame e' finito in fallback:\n{log}"


@pytest.mark.skipif(not MODEL.is_file(), reason="requires a locally trained models.pkl")
def test_late_submission_format_does_not_lose_the_run():
    """Il guasto peggiore: senza il formato non esiste alcun submission.csv.

    Era l'unica lettura non protetta del file, e contraddiceva la difesa n. 2
    dichiarata nel docstring del modulo - che non puo' scattare prima di sapere
    quali uid predire.
    """
    uids = [f"phantom{i}" for i in range(N_EXAMS)]
    with tempfile.TemporaryDirectory() as d:
        work = pathlib.Path(d)
        pack = _stage(work)
        data = _data_dir(work, uids, with_format=False, with_volumes=uids)

        def late():
            time.sleep(3)
            try:
                (data / "submission_format.csv").write_text(
                    "uid,is_pathologic\n" + "".join(f"{u},0.5\n" for u in uids)
                )
            except OSError:
                pass  # il test e' gia' finito e la temporanea rimossa: vedi sopra

        threading.Thread(target=late, daemon=True).start()
        code, log, rows = _run(pack, data, work)

    assert code == 0, f"exit non zero: {log[-400:]}"
    assert rows is not None, "submission.csv assente: l'invio sarebbe stato perso"
    assert len(rows) == N_EXAMS, f"submission.csv incompleta: {len(rows)} righe"


@pytest.mark.skipif(not MODEL.is_file(), reason="requires a locally trained models.pkl")
def test_a_genuinely_missing_volume_is_declared_not_hidden():
    """Un volume che non arriva mai deve ripiegare E dirlo, con tassi non conteggi."""
    uids = [f"phantom{i}" for i in range(N_EXAMS)]
    with tempfile.TemporaryDirectory() as d:
        work = pathlib.Path(d)
        pack = _stage(work)
        # Il primo c'e', cosi' la sentinella passa subito; manca quello in mezzo.
        _data_dir(work, uids, with_volumes=[uids[0], uids[2]])
        code, log, rows = _run(pack, work / "data", work)

    assert code == 0, f"exit non zero: {log[-400:]}"
    assert rows is not None and len(rows) == N_EXAMS, "submission.csv assente o incompleta"
    assert "esito degradato" in log, f"il fallback non e' stato dichiarato:\n{log}"
    assert "FileNotFoundError" in log, f"la causa non compare:\n{log}"
    assert "ritentati" in log, f"il ritentativo non e' riportato:\n{log}"


@pytest.mark.skipif(not MODEL.is_file(), reason="requires a locally trained models.pkl")
def test_diagnostics_never_leak_an_absolute_count():
    """Nessuna riga deve contenere la dimensione del test set.

    Se il filtro della piattaforma esiste per non rivelarla, quello e' un
    controllo legittimo e la diagnostica lo rispetta: a noi serve un tasso, che
    non dice nulla sul denominatore. Il test guarda che nelle righe non compaia
    il numero di esami come intero isolato.
    """
    import re

    uids = [f"phantom{i}" for i in range(N_EXAMS)]
    with tempfile.TemporaryDirectory() as d:
        work = pathlib.Path(d)
        pack = _stage(work)
        _data_dir(work, uids, with_volumes=uids)
        _, log, _ = _run(pack, work / "data", work)

    righe = [r for r in log.splitlines() if r.startswith("[")]
    assert righe, f"nessuna riga di diagnostica emessa:\n{log}"
    for riga in righe:
        # Interi non seguiti da % o da un separatore decimale: un conteggio.
        nudi = re.findall(r"(?<![\d.,])(\d+)(?![\d.,]*\s*(?:%|s/|s,|\.\d))", riga)
        assert str(N_EXAMS) not in nudi, \
            f"conteggio assoluto ({N_EXAMS}) nella diagnostica: {riga}"


@pytest.mark.skipif(not MODEL.is_file(), reason="requires a locally trained models.pkl")
def test_all_three_diagnostic_shapes_reach_both_streams():
    """La sonda del filtro: tre forme, due flussi. Vedi emit() in main.py.

    Finche' non si sa quale forma sopravvive al filtro della piattaforma, devono
    esserci tutte. Quando si sapra', questo test va stretto a quella rimasta.
    """
    uids = [f"phantom{i}" for i in range(N_EXAMS)]
    with tempfile.TemporaryDirectory() as d:
        work = pathlib.Path(d)
        pack = _stage(work)
        _data_dir(work, uids, with_volumes=uids)
        out = work / "out"; out.mkdir()
        proc = subprocess.run(
            [sys.executable, str(pack / "main.py")], cwd=out,
            capture_output=True, text=True,
            env=dict(os.environ, DAT_DATA_DIR=str(work / "data")),
        )

    for flusso, testo in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        for tag in ("[A]", "[B]", "[C]"):
            assert tag in testo, f"forma {tag} assente da {flusso}:\n{testo}"
    assert "%" in proc.stdout, "nessuna percentuale: la forma B non e' utilizzabile"
    assert any(not any(c.isdigit() for c in r) for r in proc.stdout.splitlines()
               if r.startswith("[C]")), "la forma C contiene cifre"


if __name__ == "__main__":
    if not MODEL.is_file():
        print(f"  SALTATI i test end-to-end: {MODEL} assente (esegui 02_train_cv.py)")
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    if not MODEL.is_file():
        fns = [(n, f) for n, f in fns if "wait" in n or "empty" in n]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1; print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} test superati")
    sys.exit(1 if failed else 0)
