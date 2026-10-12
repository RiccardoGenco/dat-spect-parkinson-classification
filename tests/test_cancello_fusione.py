"""Collaudo del cancello sulla fusione, su centri finti di cui si sa la risposta.

Perche' esiste
--------------
`check_fusione_per_centro` e' l'unica cosa che sta fra un coefficiente sbagliato e
la submission. Un cancello che non fallisce sull'ipotesi che deve bocciare non e'
un cancello (R3), e questo cancello ha sostituito una soglia che invece bocciava
— quindi il rischio nuovo e' opposto a quello vecchio: **essere troppo
permissivo**. Qui si verifica che rifiuti davvero.

Il caso costruito e' quello vero in miniatura: due centri nominati piu' un
residuo di impronte rare, un modello SBR decente, e una CNN che su un centro
informa e su un altro no. La fusione a leva deve passare quando la CNN consegna
ovunque e deve essere **respinta** quando perde anche su un solo centro — che e'
esattamente cio' che `a=1, b=0,498` fa sui dati veri (3 centri su 5).

Si verifica anche che le due guardie R13 mordano: un metro girato su una
normalizzazione diversa da quella dei pesi, e un ordine di righe che non
corrisponde all'artefatto. Entrambe producono numeri plausibili e sbagliati se
non vengono fermate.

Nessun dato della competizione, nessuna GPU:

    python tests/test_cancello_fusione.py
"""
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

CENTRI = (("2.00x2.00x2.00", 60), ("3.00x3.00x3.00", 40), ("(impronte rare)", 30))


def _pack():
    spec = importlib.util.spec_from_file_location(
        "pack", ROOT / "scripts" / "03_pack_submission.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _perdita(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _dati(cnn_inutile_su=None, seed=0):
    """SBR informativo ovunque; CNN informativa tranne dove indicato.

    Le probabilita' si costruiscono da un logit lineare nell'etichetta piu'
    rumore: e' il minimo che riproduce la struttura che il cancello legge, cioe'
    un guadagno della fusione che cambia segno da centro a centro.
    """
    rng = np.random.default_rng(seed)
    gruppi = np.concatenate([[nome] * n for nome, n in CENTRI])
    n = len(gruppi)
    y = np.tile([0.0, 1.0], n // 2 + 1)[:n]

    z_sbr = 2.2 * (2 * y - 1) + rng.normal(0, 1.5, n)
    z_cnn = 2.0 * (2 * y - 1) + rng.normal(0, 1.6, n)
    if cnn_inutile_su is not None:
        # Sul centro indicato la CNN non informa e sbaglia **di poco**: e' il caso
        # che serve, non quello comodo. Un disastro su un centro renderebbe
        # negativa anche la media pesata, e allora un cancello sul livello medio
        # basterebbe. Qui la media resta positiva e solo il minimo e' negativo —
        # la stessa forma di `a=1, b=0,498` sui dati veri, che rende +0,0105 in
        # media e perde su due centri su cinque.
        m = gruppi == cnn_inutile_su
        z_cnn[m] = -0.30 * (2 * y[m] - 1) + rng.normal(0, 1.0, int(m.sum()))

    return y, gruppi, 1 / (1 + np.exp(-z_sbr)), 1 / (1 + np.exp(-z_cnn))


def _scrivi(radice, y, gruppi, p_sbr, p_cnn, a, b, norm="istanza",
            norm_metro=None, mescola_righe=False,
            autonorm=False, autonorm_metro=None, tabular_branch="incumbent",
            tabular_branch_metro=None):
    """Costruisce una radice di progetto finta: artefatti, features, sidecar."""
    art = radice / "experiments" / "risultati"
    art.mkdir(parents=True)
    (radice / "models").mkdir()
    shutil.copy2(ROOT / "scripts" / "20_risoluzione.py",
                 (radice / "scripts").mkdir() or radice / "scripts" / "20_risoluzione.py")

    per_fold = [{"fold": nome, "n": n} for nome, n in CENTRI]
    metro = "metro_finto"
    (art / f"{metro}.json").write_text(json.dumps({
        "tabular_branch": (tabular_branch if tabular_branch_metro is None
                            else tabular_branch_metro),
        "config_cnn": {"norm": norm_metro or norm,
                       "autonorm": autonorm if autonorm_metro is None
                       else autonorm_metro},
        "complessivo": {"solo_sbr": {"log_loss": _perdita(y, p_sbr)}},
        "per_fold": per_fold,
    }, indent=2), encoding="utf-8")
    np.savez(art / f"{metro}.npz", solo_sbr=p_sbr, p_cnn=p_cnn)

    ordine = np.arange(len(y))
    if mescola_righe:
        ordine = np.roll(ordine, 7)
    pd.DataFrame({"is_pathologic": y[ordine].astype(int),
                  "_group": gruppi[ordine]}).to_csv(radice / "features_v4.csv",
                                                    index=False)

    pesi = radice / "models" / "voxnet_finto.pt"
    fus = {"a": a, "b": b, "arco": "finto", "norm": norm, "n_reti": 20,
           "autonorm": autonorm, "tabular_branch": tabular_branch}
    pesi.with_suffix(".fusione.json").write_text(json.dumps(fus), encoding="utf-8")
    try:
        import torch  # noqa: PLC0415

        torch.save({"fusione": {"a": a, "b": b}, "config": {"norm": norm},
                    "state_dicts": []}, str(pesi))
    except ImportError:
        pesi.write_bytes(b"")  # senza torch il checkpoint non viene mai aperto
    return pesi, metro


def _esegui(a, b, cnn_inutile_su=None, **kw):
    """Esegue il cancello in una radice temporanea. Restituisce l'errore, o None."""
    pack = _pack()
    y, gruppi, p_sbr, p_cnn = _dati(cnn_inutile_su)
    with tempfile.TemporaryDirectory() as tmp:
        radice = pathlib.Path(tmp)
        pesi, metro = _scrivi(radice, y, gruppi, p_sbr, p_cnn, a, b, **kw)
        pack.ROOT = radice
        try:
            pack.check_fusione_per_centro(pesi, metro=metro)
            return None
        except SystemExit as e:
            return str(e)


def _guadagni(a, b, cnn_inutile_su=None):
    """I guadagni per centro che il caso costruito DEVE avere.

    Il test verifica il fixture prima di verificare il cancello: se la struttura
    non fosse quella voluta, l'asserzione sul cancello passerebbe per il motivo
    sbagliato.
    """
    sys.path.insert(0, str(ROOT))
    from src.voxnet import fondi  # noqa: PLC0415

    y, gruppi, p_sbr, p_cnn = _dati(cnn_inutile_su)
    fusa = fondi(p_sbr, p_cnn, a, b)
    return {nome: _perdita(y[gruppi == nome], p_sbr[gruppi == nome])
            - _perdita(y[gruppi == nome], fusa[gruppi == nome])
            for nome, _ in CENTRI}


def test_passa_se_guadagna_su_ogni_centro():
    g = _guadagni(0.7, 0.6)
    assert min(g.values()) > 0, f"fixture sbagliato, non guadagna ovunque: {g}"
    assert _esegui(0.7, 0.6) is None, "ha rifiutato una fusione che vince 3/3"


def test_rifiuta_se_perde_su_un_solo_centro():
    """R3: e' l'ipotesi che il cancello esiste per bocciare.

    Un centro su cui si perde in misura e' un centro su cui si perdera' in gara,
    e la media pesata resta positiva — quindi un cancello sul livello medio lo
    lascerebbe passare. E' la ragione per cui il criterio e' sul minimo.
    """
    g = _guadagni(0.7, 0.6, cnn_inutile_su="3.00x3.00x3.00")
    assert g["3.00x3.00x3.00"] < 0, f"fixture sbagliato, non perde: {g}"
    pesi = {nome: n for nome, n in CENTRI}
    medio = sum(pesi[k] * v for k, v in g.items()) / sum(pesi.values())
    assert medio > 0, (
        f"fixture sbagliato: con media pesata {medio:+.4f} anche un cancello sul "
        f"livello medio boccerebbe, e il test non proverebbe niente sul minimo")
    errore = _esegui(0.7, 0.6, cnn_inutile_su="3.00x3.00x3.00")
    assert errore is not None, "ha accettato una fusione che perde su un centro"
    assert "PERDE su un centro" in errore, errore
    assert "3.00x3.00x3.00" in errore, errore


def test_b_zero_non_muove_nulla_e_a_uno_passa():
    """A (a=1, b=0) la fusione E' l'SBR: guadagno esattamente zero, quindi rifiuto.

    Serve a fissare il bordo. Il cancello chiede un guadagno STRETTAMENTE
    positivo su ogni centro: una fusione che non fa niente non e' una fusione da
    spedire, e' un ramo a voxel spento che si crede acceso.
    """
    g = _guadagni(1.0, 0.0)
    assert max(abs(v) for v in g.values()) < 1e-9, g
    assert _esegui(1.0, 0.0) is not None


def test_r13_normalizzazione_del_metro_diversa_dai_pesi():
    errore = _esegui(0.7, 0.6, norm="istanza", norm_metro="batch")
    assert errore is not None and "norm" in errore, errore
    assert "modello diverso da quello spedito" in errore, errore


def test_r13_autonorm_del_metro_diverso_dai_pesi():
    """`autonorm` cambia cio' che la rete vede quanto `norm`, e va guardato uguale.

    Dal 6 agosto sul disco convivono due metri che differiscono SOLO per questo
    — `_cnn_istanza` e `_cnn_istanza_autonorm` — e i pesi corrispondenti si
    chiamano `voxnet_istanza.pt` e `voxnet_autonorm.pt`. Con il solo controllo su
    `norm`, scambiarli darebbe un cancello superato su un modello diverso da
    quello che parte: nessun errore, e una submission autorizzata da una misura
    che riguarda altro.
    """
    errore = _esegui(0.7, 0.6, autonorm=True, autonorm_metro=False)
    assert errore is not None, "metro senza autonorm su pesi con autonorm: accettato"
    assert "autonorm" in errore, errore
    assert "modello diverso da quello spedito" in errore, errore

    # e nel verso opposto, che e' quello che capiterebbe spedendo i pesi vecchi
    # con il metro nuovo
    errore = _esegui(0.7, 0.6, autonorm=False, autonorm_metro=True)
    assert errore is not None and "autonorm" in errore, errore


def test_autonorm_concorde_passa():
    """Il controllo nuovo non deve rifiutare la configurazione legittima."""
    assert _esegui(0.7, 0.6, autonorm=True, autonorm_metro=True) is None
    assert _esegui(0.7, 0.6, autonorm=False, autonorm_metro=False) is None


def test_r13_righe_non_allineate_con_artefatto():
    """L'ordine di features_v4.csv deve essere quello con cui il metro ha scritto.

    Se non lo fosse, ogni guadagno per centro sarebbe calcolato su esami diversi
    da quelli che l'artefatto descrive: numeri plausibili e falsi, che e' il modo
    di fallire che questo progetto teme di piu'.
    """
    errore = _esegui(0.7, 0.6, mescola_righe=True)
    assert errore is not None, "righe mescolate accettate in silenzio"
    assert ("non allineate" in errore or "non coincide" in errore), errore


def test_r13_ramo_tabellare_del_metro_diverso_dai_pesi():
    """Il metro gold non puo' autorizzare un checkpoint con il ramo incumbent."""
    errore = _esegui(
        0.7, 0.6, tabular_branch="incumbent", tabular_branch_metro="gold_logistic"
    )
    assert errore is not None, "rami tabellari diversi accettati in silenzio"
    assert "ramo tabellare" in errore, errore


def test_senza_pesi_non_fa_nulla():
    pack = _pack()
    assert pack.check_fusione_per_centro(None) is None


if __name__ == "__main__":
    fns = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1; print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} test superati")
    sys.exit(1 if failed else 0)
