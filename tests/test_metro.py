"""Collaudo del metro esterno, su un dataset sintetico con centri finti.

Perche' esiste
--------------
`16_metro_esterno.py` costa ~55 minuti di GPU per eseguirsi una volta. Un difetto
banale — un indice fuori posto, una fusione applicata al vettore sbagliato, un
fold che si sovrappone a un altro — si scoprirebbe dopo un'ora, e nel caso
peggiore **non** si scoprirebbe affatto: produrrebbe un numero plausibile. E'
la stessa classe di guasto silenzioso di `test_voxel.py`, spostata dal container
al metro.

Qui il dataset e' costruito a mano con la struttura che conta: **un guadagno
moltiplicativo per centro**, cioe' la contaminazione vera misurata sui dati
(il livello dell'SBR varia del 50% fra centri, il rapporto fra le classi resta
fra 1,65 e 2,13). Un metro che non regge su questo non reggera' sul resto.

Nessun dato della competizione: si esegue ovunque, anche senza GPU.

    python tests/test_metro.py
"""
import importlib.util
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _metro():
    spec = importlib.util.spec_from_file_location(
        "metro_esterno", ROOT / "scripts" / "16_metro_esterno.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _dataset_finto(seed: int = 0):
    """Sei centri grandi, otto rari, e un guadagno moltiplicativo per centro.

    I livelli riproducono l'ordine di grandezza misurato: normali attorno a 2,4
    di SBR, patologici attorno a 1,4, e un fattore di centro fra 0,7 e 1,4 che
    moltiplica entrambi — cioe' che sposta il livello senza toccare il rapporto.
    E' la struttura per cui `LogRatios` esiste, quindi un metro collaudato senza
    di essa non collauderebbe il modello vero.
    """
    rng = np.random.default_rng(seed)
    taglie = {"A": 240, "B": 150, "C": 110, "D": 90, "E": 40, "F": 30}
    for i in range(8):
        taglie[f"rara{i}"] = 7

    righe = []
    for centro, k in taglie.items():
        guadagno = float(rng.uniform(0.7, 1.4))
        prevalenza = float(rng.uniform(0.45, 0.62))
        for _ in range(k):
            y = int(rng.random() < prevalenza)
            base = (1.4 if y else 2.4) * float(np.exp(rng.normal(0, 0.22)))
            sbr = guadagno * base
            righe.append({
                "uid": f"{centro}_{len(righe)}",
                "min_sbr": sbr * float(rng.uniform(0.88, 0.98)),
                "max_sbr": sbr * float(rng.uniform(1.02, 1.12)),
                "mean_sbr": sbr,
                "asym_sbr": abs(float(rng.normal(0.05 + 0.05 * y, 0.04))),
                "min_putamen_caudate_ratio": float(
                    np.clip(rng.normal(0.88 - 0.10 * y, 0.07), 0.3, 1.6)),
                "localization_confidence": float(
                    np.clip(rng.normal(0.55 - 0.12 * y, 0.12), 0.0, 1.0)),
                "_group": centro,
                "is_pathologic": y,
            })
    return pd.DataFrame(righe).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------
# Le partizioni esterne
# --------------------------------------------------------------------------

def test_fold_a_gruppi_sono_una_partizione_e_isolano_i_centri_grandi():
    m = _metro()
    f = _dataset_finto()
    g = f["_group"].to_numpy()
    y = f["is_pathologic"].to_numpy()

    fold = m.fold_esterni(g, y, "gruppi", min_n=80)
    tutti = np.concatenate([idx for _, idx in fold])
    assert len(tutti) == len(np.unique(tutti)) == len(f), (
        f"i fold si sovrappongono o non coprono tutto: {len(tutti)} vs {len(f)}")

    grandi = [nome for nome, _ in fold if not nome.startswith("(")]
    assert set(grandi) == {"A", "B", "C", "D"}, grandi
    for nome, idx in fold:
        if nome.startswith("("):
            continue
        assert set(g[idx]) == {nome}, (nome, set(g[idx]))


def test_fold_a_strati_hanno_lo_stesso_numero_di_ripetizioni():
    """Il confronto fra i due lati del metro regge solo a pari ripetizioni.

    Se la partizione a strati avesse piu' fold, ogni fold avrebbe piu' dati di
    addestramento e il suo vantaggio non verrebbe piu' dalla composizione — che
    e' la variabile in esame — ma dalla taglia. Il metro misurerebbe due cose
    insieme.
    """
    m = _metro()
    f = _dataset_finto()
    g, y = f["_group"].to_numpy(), f["is_pathologic"].to_numpy()

    a = m.fold_esterni(g, y, "gruppi", min_n=80)
    b = m.fold_esterni(g, y, "strati", min_n=80)
    assert len(a) == len(b), (len(a), len(b))
    tutti = np.concatenate([idx for _, idx in b])
    assert len(tutti) == len(np.unique(tutti)) == len(f)


# --------------------------------------------------------------------------
# Il ramo SBR fuori campione
# --------------------------------------------------------------------------

def test_sbr_fuori_campione_batte_la_prevalenza_su_un_centro_mai_visto():
    """Il controllo minimo: il modello deve trasferire almeno un po'.

    Non e' una misura di qualita' — il dataset e' finto — e' la verifica che il
    percorso di predizione sia collegato. Un errore di indicizzazione darebbe
    predizioni scorrelate dalle etichette, cioe' una log loss attorno a quella
    della prevalenza, e passerebbe qualunque controllo di forma.
    """
    from sklearn.metrics import log_loss

    from src.model import feature_columns

    m = _metro()
    f = _dataset_finto()
    g, y = f["_group"].to_numpy(), f["is_pathologic"].to_numpy()
    colonne = feature_columns(f)
    assert "_group" not in colonne and "is_pathologic" not in colonne, colonne

    fuori = np.flatnonzero(g == "A")
    dentro = np.setdiff1d(np.arange(len(f)), fuori)
    res = m.sbr_fuori_campione(f, colonne, g, dentro, fuori, seed=0)

    p = np.clip(res["p_out"], 0.005, 0.995)
    assert p.shape == fuori.shape, (p.shape, fuori.shape)
    assert np.all((p > 0) & (p < 1))
    quota = y[fuori].mean()
    baseline = log_loss(y[fuori], np.full(len(fuori), quota))
    assert log_loss(y[fuori], p) < baseline - 0.05, (log_loss(y[fuori], p), baseline)


# --------------------------------------------------------------------------
# L'aritmetica della fusione
# --------------------------------------------------------------------------

def test_la_fusione_guadagna_con_un_modello_vero_e_non_con_uno_finto():
    """Il controllo che discrimina, nei due versi.

    Un modello nuovo informativo deve far scendere la log loss; uno che e' solo
    rumore con la stessa distribuzione marginale non deve. Un metro che approva
    entrambi non e' un metro (R3), ed e' esattamente il difetto che la
    submission 2 ha pagato.
    """
    from sklearn.metrics import log_loss

    from src.voxnet import fondi

    rng = np.random.default_rng(3)
    n = 1200
    y = (rng.random(n) < 0.55).astype(int)
    logit_sbr = rng.normal(0, 1.4, n) + 1.7 * (2 * y - 1)
    p_sbr = 1 / (1 + np.exp(-logit_sbr))

    logit_vero = rng.normal(0, 1.4, n) + 1.5 * (2 * y - 1)
    p_vero = 1 / (1 + np.exp(-logit_vero))
    p_finto = p_vero[rng.permutation(n)]

    def ll(p):
        return log_loss(y, np.clip(p, 0.005, 0.995))

    base = ll(p_sbr)
    con_vero = ll(fondi(p_sbr, p_vero, 1.0, 0.30))
    con_finto = ll(fondi(p_sbr, p_finto, 1.0, 0.30))
    assert base - con_vero > 0.01, (base, con_vero)
    assert con_finto > base - 0.002, (base, con_finto)


def test_fusione_a_b_nullo_e_esattamente_l_incumbent():
    from src.voxnet import fondi

    rng = np.random.default_rng(4)
    p_sbr = rng.uniform(0.01, 0.99, 500)
    p_cnn = rng.uniform(0.01, 0.99, 500)
    assert np.allclose(fondi(p_sbr, p_cnn, 1.0, 0.0), p_sbr, atol=1e-12)


def test_verdetto_distingue_i_tre_casi():
    m = _metro()
    assert "racchiude" in m.verdetto(0.300, 0.270, 0.2884, 0.005)
    assert "PEGGIO" in m.verdetto(0.280, 0.270, 0.2884, 0.005)
    assert "MEGLIO" in m.verdetto(0.320, 0.300, 0.2884, 0.005)


# --------------------------------------------------------------------------
# La corsa intera, senza CNN
# --------------------------------------------------------------------------

def test_corsa_intera_senza_cnn_copre_ogni_esame_una_volta_sola():
    """End-to-end del percorso `--senza-cnn`: due minuti, e prova il montaggio.

    Verifica la proprieta' che rende il metro un metro: **ogni esame e' predetto
    esattamente una volta, da una catena che non ha visto il suo centro**. Se
    un indice slittasse, alcuni esami sarebbero predetti due volte e altri
    nessuna — e la log loss risultante sarebbe comunque un numero leggibile.
    """
    m = _metro()
    f = _dataset_finto()

    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        csv = d / "finte.csv"
        f.to_csv(csv, index=False)
        argv = sys.argv
        sys.argv = ["16_metro_esterno.py", "--senza-cnn", "--features", str(csv),
                    "--min-n", "80", "--outdir", str(d)]
        try:
            assert m.main() == 0
        finally:
            sys.argv = argv

        dati = json.loads((d / "metro_esterno_gruppi.json").read_text(encoding="utf-8"))
        assert dati["n"] == len(f)
        assert sum(r["n"] for r in dati["per_fold"]) == len(f), dati["per_fold"]
        # `dict(...)` e non l'oggetto pigro: un NpzFile tiene aperto il file
        # finche' non viene chiuso, e su Windows la cartella temporanea non si
        # cancella con un handle vivo — il test falliva con PermissionError
        # dopo essere passato. Su Linux non si vedeva.
        arrays = dict(np.load(d / "metro_esterno_gruppi.npz"))
        p = arrays["solo_sbr"]
        assert p.shape == (len(f),) and not np.isnan(p).any()
        assert dati["complessivo"]["solo_sbr"]["log_loss"] < 0.65


# --------------------------------------------------------------------------
# Il ramo caro: la CNN fuori campione, su CPU e in miniatura
# --------------------------------------------------------------------------

def test_cnn_fuori_campione_predice_ogni_esame_escluso_una_volta_sola():
    """Il percorso da 55 minuti di GPU, eseguito in pochi secondi su CPU.

    Non misura nulla — due epoche su volumi 16³ non imparano niente — e non deve.
    Verifica il **montaggio**: che le OOF interne coprano tutti e soli i centri
    rimasti, che la predizione esterna copra tutti e soli quelli esclusi, che
    l'ensemble abbia il numero di reti che dice di avere, e che nessun indice
    slitti fra il sistema di riferimento globale e quello del sottoinsieme.

    Quello slittamento e' il difetto piu' probabile di tutto il file, ed e'
    silenzioso per costruzione: produrrebbe probabilita' valide, in numero
    giusto, semplicemente attribuite all'esame sbagliato.

    Serve torch, quindi si salta fuori dall'immagine ufficiale — come fa
    `test_voxel.py`. Un ERROR da import mancante e un test che fallisce davvero
    hanno lo stesso colore nell'output, e il primo insegna a ignorare il secondo.
    """
    try:
        import torch
    except ImportError:
        print("      (saltato: torch assente, esegui dentro l'immagine)")
        return

    m = _metro()
    rng = np.random.default_rng(7)
    # Sei impronte fra i centri rimasti, cosi' che la partizione interna usi
    # davvero `StratifiedGroupKFold` e non il suo ripiego stratificato.
    groups = np.array(["A"] * 24 + sum([[c] * 8 for c in "BCDEFG"], []))
    n = len(groups)
    y = (rng.random(n) < 0.5).astype(np.float32)
    # 32³ con `riduci=1` riproduce la geometria vera dell'ultimo blocco (2³ di
    # lato prima della media globale) a un ottavo del costo di 64³ con
    # `riduci=2`. A 16³ l'ultimo blocco degenererebbe a 1³ e BatchNorm
    # fallirebbe su un batch da un esame: un guasto del collaudo, non del codice.
    vol = torch.from_numpy((rng.random((n, 1, 32, 32, 32)) * 3).astype(np.float16))

    cfg = {"arch": "toy", "epochs": 2, "patience": 2, "batch": 8, "lr": 3e-4,
           "wd": 1e-4, "dropout": 0.3, "cosine": False, "mixup": 0.0,
           "smooth": 0.0, "gradi": 6.0, "scala": 0.05, "shift": 2.0,
           "sfoca": 4.0, "rumore": 0.05, "gamma": 0.05, "guadagno": 0.05,
           "flip": True, "mirror_channel": False, "tta": True, "amp": False,
           "riduci": 1, "early_stop": True}

    fuori = np.flatnonzero(groups == "A")
    dentro = np.setdiff1d(np.arange(n), fuori)

    import torch.nn as nn
    import torch.nn.functional as F

    res = m.cnn_fuori_campione(
        m.carica_arco(), torch, nn, F, vol, y, groups, dentro, fuori, cfg,
        seeds=[0, 1], fold_seed=0, tag=0, log=lambda _s: None, dev="cpu")

    assert res["p_out"].shape == fuori.shape, (res["p_out"].shape, fuori.shape)
    assert res["oof_dentro"].shape == dentro.shape
    assert np.all((res["p_out"] > 0) & (res["p_out"] < 1))
    assert np.all((res["oof_dentro"] > 0) & (res["oof_dentro"] < 1))
    assert res["n_reti"] == 2 * 5, res["n_reti"]
    # Nessuna OOF interna e' rimasta al valore iniziale: ogni esame di `dentro`
    # e' stato predetto da un fold che non lo conteneva.
    assert not np.any(res["oof_dentro"] == 0.0)



def test_etichetta_distingue_le_configurazioni():
    """R3 applicata al nome di un file: due corse diverse non devono collidere.

    Il guasto e' silenzioso per costruzione — un file sovrascritto ha lo stesso
    nome e un contenuto plausibile — ed e' gia' successo una volta il 3 agosto.
    La riparazione di allora copriva `norm`; questa copre tutto cio' che cambia
    la misura, e questo test e' cio' che impedisce alla prossima aggiunta di
    ripetere il giro.
    """
    import types

    m = _metro()

    def args(**kw):
        base = dict(norm="istanza", autonorm=False, gradi=0.0, scala=0.0,
                    sfoca=0.0, rumore=0.0, gamma=0.0, guadagno=0.0, shift=4.0,
                    flip=True, tta=True, riduci=2, arch="media")
        base.update(kw)
        return types.SimpleNamespace(**base)

    # I valori predefiniti conservano l'etichetta storica.
    assert m.etichetta_configurazione(args()) == "_cnn_istanza"

    # Ogni parametro che cambia la misura cambia il nome.
    varianti = [args(), args(gradi=12.0), args(scala=0.1), args(sfoca=5.0),
                args(rumore=0.05), args(norm="gruppo"), args(autonorm=True),
                args(tta=False), args(riduci=1), args(arch="toy"),
                args(shift=8.0), args(flip=False), args(gradi=12.0, scala=0.1)]
    nomi = [m.etichetta_configurazione(v) for v in varianti]
    assert len(set(nomi)) == len(nomi), sorted(nomi)

    # E nessuna etichetta contiene un punto: finirebbe in un nome di file dove
    # la parte dopo l'ultimo punto e' l'estensione.
    assert not any("." in n for n in nomi), nomi

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
