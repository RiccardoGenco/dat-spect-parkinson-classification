"""Collaudo di `20_risoluzione.py`, su differenze di cui si conosce la risposta.

Perche' esiste
--------------
Questo script produce **errori standard**, cioe' numeri che nessuno controlla
guardandoli: un errore gonfiato di dieci volte non si vede, si legge come «il
Delta non e' significativo» e chiude una strada che era aperta. E' la stessa
categoria del guasto silenzioso che `test_voxel.py` e `test_metro.py`
sorvegliano, applicata pero' allo strumento che decide invece che alla catena
che predice.

L'ipotesi sbagliata contro cui mordono questi test
---------------------------------------------------
Calcolare l'errore dalla dispersione delle **log loss** invece che da quella
delle **differenze appaiate**. E' l'errore naturale — «ho due medie, prendo la
loro incertezza» — ed e' sbagliato perche' i due modelli sbagliano sugli stessi
esami: quella covarianza si cancella nella differenza e non nei livelli.

`test_appaiato_batte_non_appaiato` fallisce su quella versione. Sui dati veri
del metro il fattore fra i due errori e' **4,3x** sulla fusione stimata nel fold
e **7,0x** su `a=1, b=0,30`: abbastanza da trasformare un |t| di 3,5 in uno di
0,5, cioe' un guadagno solido in «non significativo».

Gli altri test reggono anche su alcune versioni sbagliate — ciascuno nomina nel
proprio docstring l'ipotesi che uccide, e solo quella.

Nessun dato della competizione: si esegue ovunque, in un secondo.

    python tests/test_risoluzione.py
"""
import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "risoluzione", ROOT / "scripts" / "20_risoluzione.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _coppia(n=1362, seed=0, spinta=0.35):
    """Due predizioni correlate su etichette vere: la struttura del metro.

    `pb` e' `pa` migliorata di una spinta nella direzione giusta, quindi le due
    sbagliano sugli stessi esami — che e' esattamente cio' che rende il calcolo
    appaiato diverso da quello non appaiato.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.55).astype(int)
    z = rng.normal(0, 1, n) + 2.2 * (y - 0.5)
    pa = 1 / (1 + np.exp(-z))
    pb = 1 / (1 + np.exp(-(z + spinta * (y - 0.5))))
    return y, pa, pb


def test_se_scala_come_radice_di_n():
    """Raddoppiando gli esami l'errore deve scendere di sqrt(2), non di 2."""
    m = _mod()
    y, pa, pb = _coppia(n=4000)
    dif = m.perdita(y, pa) - m.perdita(y, pb)
    r1 = m.risoluzione(dif[:1000], 525)
    r2 = m.risoluzione(dif[:4000], 525)
    rapporto = r1["se"] / r2["se"]
    assert 1.8 < rapporto < 2.2, rapporto


def test_delta_nullo_non_e_significativo():
    """Due predizioni identiche: Delta esattamente 0, |t| non deve esplodere."""
    m = _mod()
    y, pa, _ = _coppia()
    r = m.risoluzione(m.perdita(y, pa) - m.perdita(y, pa), 525)
    assert r["delta"] == 0.0, r["delta"]
    assert r["t"] == 0.0 or not np.isfinite(r["t"]) or r["t"] < 1e-9, r["t"]


def test_appaiato_batte_non_appaiato():
    """IL test: l'errore appaiato dev'essere molto piu' stretto di quello no.

    Se qualcuno riscrivesse `risoluzione` sulla dispersione delle log loss
    invece che su quella delle differenze, questo confronto crollerebbe a ~1 e
    il test fallirebbe. E' il controllo che l'intero script esiste per superare.
    """
    m = _mod()
    y, pa, pb = _coppia()
    la, lb = m.perdita(y, pa), m.perdita(y, pb)
    se_appaiato = m.risoluzione(la - lb, 525)["se"]
    # La versione sbagliata: due medie indipendenti, errori sommati in quadratura.
    se_non_appaiato = float(np.sqrt(la.var(ddof=1) / len(y) + lb.var(ddof=1) / len(y)))
    assert se_non_appaiato / se_appaiato > 5, (se_non_appaiato, se_appaiato)


def test_se_gara_e_piu_largo_di_quello_del_metro():
    """525 esami non possono misurare meglio di 1362. Se accadesse, e' un bug."""
    m = _mod()
    y, pa, pb = _coppia()
    r = m.risoluzione(m.perdita(y, pa) - m.perdita(y, pb), 525)
    assert r["se_gara"] > r["se"], (r["se_gara"], r["se"])
    assert abs(r["se_gara"] / r["se"] - np.sqrt(len(y) / 525)) < 1e-9


def test_leva_pura_e_sempre_negativa_su_un_modello_calibrato():
    """`spazza_a` deve dire «leva» quando il modello e' gia' calibrato.

    Su predizioni ben calibrate per costruzione, restringere il logit non puo'
    che peggiorare. Se `spazza_a` restituisse un guadagno positivo qui, la
    diagnosi «leva o calibrazione» direbbe il contrario del vero proprio nel
    caso in cui la risposta e' certa.
    """
    m = _mod()
    y, pa, _ = _coppia(n=6000)
    curva = m.spazza_a(y, pa)
    assert curva[0]["a"] == 1.0 and abs(curva[0]["guadagno"]) < 1e-12
    assert all(c["guadagno"] < 0 for c in curva[1:]), curva
    assert curva[-1]["guadagno"] < curva[1]["guadagno"], "non monotono"


def test_r10_degrada_il_rango_e_non_la_scala():
    """Il degrado di R10 deve rovinare l'ORDINE, non rimpicciolire il peso.

    La scorciatoia naturale — «la CNN rende meta'» implementato come `b -> b/2` —
    misura una cosa diversa e piu' benevola: un peso piu' piccolo su un segnale
    ancora buono. La domanda vera e' «e se la CNN ordinasse peggio», e le due si
    separano al limite. A q=1:

    - **permutazione**: la CNN e' rumore puro, sommarla a un modello che
      funziona *danneggia*, quindi il guadagno degradato e' **negativo**;
    - **riscalatura**: b diventa 0, la fusione torna a essere l'SBR, e il
      guadagno e' **esattamente zero**.

    Questa asserzione fallisce sulla versione a riscalatura. Il resto del test
    ne fissa gli estremi: a q=0 il degrado non degrada niente.
    """
    m = _mod()
    y, pa, pb = _coppia()
    r_pieno = m.robustezza(y, pa, pb, 1.0, 0.3, q=0.0, ripetizioni=3)
    r_meta = m.robustezza(y, pa, pb, 1.0, 0.3, q=0.5, ripetizioni=20)
    r_nulla = m.robustezza(y, pa, pb, 1.0, 0.3, q=1.0, ripetizioni=20)
    assert abs(r_pieno["quota_conservata"] - 1.0) < 1e-9, r_pieno
    assert 0.0 < r_meta["quota_conservata"] < 1.0, r_meta
    assert r_nulla["guadagno_degradato"] < -1e-6, (
        "a q=1 la CNN e' rumore: sommarla deve costare, non costare zero. "
        f"guadagno degradato {r_nulla['guadagno_degradato']:+.5f}")


def test_autonorm_cancella_la_regione_di_normalizzazione():
    """`autonormalizza` deve rendere il crop indipendente dal riferimento.

    Il crop arriva diviso per `measurement.reference_mean`. Se quella divisione
    conta ancora dopo la trasformazione, la trasformazione non fa quello che
    dice — e la scelta della regione occipitale resterebbe un parametro libero
    nel percorso della CNN.

    L'ipotesi sbagliata che questo test uccide e' la sottrazione della media al
    posto della divisione per la norma: centrare non toglie un fattore
    moltiplicativo, e un crop diviso per un riferimento due volte piu' grande
    resterebbe diverso. Uccide anche la normalizzazione calcolata **sul batch**
    invece che per esame, che reintrodurrebbe una statistica di dataset.
    """
    sys.path.insert(0, str(ROOT))
    from src.voxnet import autonormalizza

    rng = np.random.default_rng(3)
    grezzo = rng.gamma(2.0, 1.0, size=(4, 8, 8, 8))
    riferimenti = np.array([0.7, 1.0, 2.4, 11.0]).reshape(4, 1, 1, 1)
    a = autonormalizza(grezzo / riferimenti)          # come arriva dalla pipeline
    b = autonormalizza(grezzo)                        # senza alcun riferimento
    assert np.abs(a - b).max() < 1e-9, np.abs(a - b).max()

    # e ogni esame e' normalizzato per conto suo, non insieme agli altri
    solo = autonormalizza(grezzo[:1] / riferimenti[:1])
    assert np.abs(solo[0] - a[0]).max() < 1e-9, "la normalizzazione usa il batch"


def test_carica_rifiuta_etichette_disallineate():
    """R13: la confrontabilita' si verifica, non si assume.

    Un `.npz` letto con le etichette di un altro ordine produce numeri
    plausibili e falsi. Il controllo deve fermarsi, non stampare.
    """
    import json
    import tempfile

    m = _mod()
    y, pa, _ = _coppia(n=400)
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)
        np.savez(p / "finto.npz", solo_sbr=pa, p_cnn=pa)
        vero = float(m.perdita(y, pa).mean())
        (p / "finto.json").write_text(json.dumps({
            "config_cnn": {"norm": "istanza"}, "train_seeds": [0], "fold_seed": 0,
            "per_fold": [], "complessivo": {"solo_sbr": {"log_loss": vero}},
        }), encoding="utf-8")
        m.carica(p, "finto", y)                       # allineato: passa
        try:
            m.carica(p, "finto", 1 - y)               # etichette invertite
        except SystemExit as e:
            assert "non allineate" in str(e), str(e)
        else:
            raise AssertionError("etichette invertite accettate in silenzio")


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
