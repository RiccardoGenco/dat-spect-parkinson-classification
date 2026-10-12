"""Collaudo del ramo a voxel, dal crop alla fusione al ripiego.

Perche' esiste, e perche' e' separato da test_core
---------------------------------------------------
Il ramo a voxel introduce tre modi di fallire che il resto del progetto non
aveva, e tutti e tre sono **silenziosi**:

1. **Il crop dell'inferenza diverso da quello dell'addestramento.** La rete
   riceverebbe una distribuzione che non ha mai visto e predirebbe rumore
   plausibile. Nessun errore. Il crop viene ora da `pipeline.process_file`
   proprio per rendere la divergenza impossibile, ma "impossibile per
   costruzione" e' un'affermazione che va verificata almeno una volta contro i
   dati veri (`test_crop_identico_a_quello_di_addestramento`).

2. **`state_dict` caricato in un'architettura diversa.** Un dizionario di
   tensori non porta con se' la rete. Se le forme combaciano ancora dopo una
   modifica — ed e' facile che combacino — non c'e' errore, ci sono predizioni
   sbagliate.

3. **Il ramo che sparisce.** `load_voxnet` non solleva mai, per progetto: se
   torch manca, se i pesi mancano, se la VRAM non basta, l'inferenza prosegue
   col solo modello SBR. E' la difesa giusta ed e' anche il guasto piu'
   pericoloso, perche' si presenta come un punteggio peggiore di ~0,02 che si
   legge come "la CNN non ha reso". Da qui i test sul ripiego, che verificano
   che degradi **verso l'SBR** e non verso la prevalenza.

I test che non toccano i dati della competizione sono autosufficienti: volumi da
`src/phantom.py`, pesi casuali generati sul momento. Quelli che richiedono
`crops/` e i NIfTI si saltano da soli quando non ci sono.

Serve torch, quindi si esegue **dentro l'immagine ufficiale**:
    python tests/test_voxel.py
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

# Questo modulo contiene anche test sul preprocessing, ma il suo contratto e'
# end-to-end e richiede PyTorch. Nel venv leggero locale si salta come unita';
# l'esecuzione diretta nel container resta gestita dal runner in fondo al file.
if __name__ != "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("test voxel da eseguire nell'immagine Docker", allow_module_level=True)

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOXNET = ROOT / "models" / "voxnet_mip.pt"
CROPS = ROOT / "crops" / "crops.npy"
FEATURES = ROOT / "features_v4.csv"
DATA = ROOT.parent / "parkinson-data" / "niftis"

# Quanti esami veri confrontare nel test di identita' del crop. Il costo e' ~1 s
# ciascuno e la domanda e' binaria: o il percorso e' lo stesso o non lo e'.
N_REALI = 8


def _phantom_nifti(directory: pathlib.Path, severity: float | None = None) -> pathlib.Path:
    """Un NIfTI sintetico da `src/phantom.py`, come fa gia' `test_mount.py`."""
    import nibabel as nib

    from src.phantom import normal_scan, pathologic_scan

    volume = (np.asarray(normal_scan(), dtype=np.float32) if severity is None
              else np.asarray(pathologic_scan(severity=severity), dtype=np.float32))
    path = directory / f"phantom_{severity}.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.diag([2.0, 2.0, 2.0, 1.0])), str(path))
    return path


# --------------------------------------------------------------------------
# Il crop: forma, normalizzazione, e identita' con l'addestramento
# --------------------------------------------------------------------------

def test_process_file_restituisce_crop_di_forma_attesa():
    from src.pipeline import CROP_VOXELS, process_file

    with tempfile.TemporaryDirectory() as d:
        path = _phantom_nifti(pathlib.Path(d))
        features, crop = process_file(path, with_crop=True)
    assert crop is not None, "nessun crop prodotto su un phantom sano"
    assert crop.shape == (CROP_VOXELS,) * 3, crop.shape
    assert np.all(np.isfinite(crop)), "crop con valori non finiti"
    assert crop.min() >= 0.0, "il crop contiene valori negativi"
    assert crop.max() > 1.0, ("il crop e' normalizzato sul riferimento: lo striato "
                              "deve stare sopra 1")


def test_crop_alto_contiene_quello_normale_al_centro():
    """La finestra centrale di un crop alto e' il crop normale, bit per bit.

    E' la proprieta' su cui poggia tutto lo studio sull'allineamento in z: le
    varianti sono fette dello stesso volume, quindi appaiate per costruzione.
    Se smettesse di valere, quei confronti smetterebbero di essere appaiati
    senza che nulla lo segnali.
    """
    from src.pipeline import CROP_VOXELS, process_file

    with tempfile.TemporaryDirectory() as d:
        path = _phantom_nifti(pathlib.Path(d))
        _, corto = process_file(path, with_crop=True)
        _, alto = process_file(path, with_crop=True, crop_z_voxels=96)
    lo = 96 // 2 - CROP_VOXELS // 2
    assert np.array_equal(alto[:, :, lo:lo + CROP_VOXELS], corto)


def test_metadati_del_crop_hanno_il_prefisso_di_esclusione():
    """La geometria del ritaglio non deve poter entrare nel modello.

    Sono lunghezze e posizioni: dipendono dal protocollo di acquisizione, cioe'
    identificano il centro. Il prefisso "_" le tiene fuori per costruzione, come
    ogni altro metadato, e questo test impedisce che una futura chiave nasca
    senza.
    """
    from src.model import feature_columns
    from src.pipeline import process_file
    import pandas as pd

    with tempfile.TemporaryDirectory() as d:
        features, _ = process_file(_phantom_nifti(pathlib.Path(d)), with_crop=True)
    nuove = [k for k in features if "crop" in k]
    assert nuove, "nessun metadato del crop: il test non sta verificando nulla"
    assert all(k.startswith("_") for k in nuove), [k for k in nuove if k[0] != "_"]
    colonne = feature_columns(pd.DataFrame([features]))
    assert not [c for c in colonne if "crop" in c], "un metadato del crop e' nel modello"


def test_crop_identico_a_quello_di_addestramento():
    """Il crop calcolato in inferenza e' bit per bit quello del dataset.

    E' il test che rende vera l'affermazione "una sola sorgente". Confronta il
    risultato di `process_file(with_crop=True)` con le righe di `crops/crops.npy`,
    che sono state prodotte dallo stesso percorso ma in un'altra esecuzione e su
    un'altra macchina logica.
    """
    import pandas as pd

    from src.pipeline import process_file

    uids = pd.read_csv(FEATURES)["uid"].tolist()[:N_REALI]
    store = np.load(CROPS, mmap_mode="r")
    diversi = []
    for i, uid in enumerate(uids):
        _, crop = process_file(DATA / f"{uid}.nii.gz", with_crop=True)
        if not np.array_equal(crop.astype(np.float16), np.asarray(store[i])):
            diversi.append(uid)
    assert not diversi, f"crop diversi da quelli di addestramento: {diversi}"


# --------------------------------------------------------------------------
# La rete: caricamento, determinismo, invarianze
# --------------------------------------------------------------------------

def _ensemble_finto(n: int = 2, arch: str = "media"):
    import torch

    from src.voxnet import VoxelEnsemble, build_model

    stati = []
    for s in range(n):
        torch.manual_seed(s)
        m = build_model(arch)
        stati.append({k: v.half() for k, v in m.state_dict().items()})
    return VoxelEnsemble({"arch": arch, "in_ch": 1, "dropout": 0.3,
                          "riduzione": 2, "tta": True}, stati, "cpu")


def test_ensemble_predice_probabilita_valide_e_deterministiche():
    ens = _ensemble_finto()
    rng = np.random.default_rng(0)
    x = rng.random((3, 64, 64, 64)).astype(np.float32) * 5
    a, b = ens.predici(x), ens.predici(x)
    assert a.shape == (3,), a.shape
    assert np.all((a > 0) & (a < 1)), a
    assert np.array_equal(a, b), "predizioni non deterministiche: dropout attiva?"


def test_tta_rende_la_predizione_invariante_al_ribaltamento():
    """Con la TTA attiva, ribaltare l'ingresso non deve cambiare la predizione.

    Non e' una proprieta' estetica: il progetto non da' mai al modello i lati
    separati, perche' *quale* emisfero sia il piu' colpito non ha significato
    diagnostico. Se la TTA fosse applicata sull'asse sbagliato — ed e' un errore
    di un carattere — l'invarianza sparirebbe.

    **Usa i pesi spediti e non pesi casuali**, e la prima versione di questo test
    sbagliava proprio li'. Una rete appena inizializzata ha le statistiche di
    BatchNorm ai valori di default e in `eval()` produce un'uscita quasi
    indipendente dall'ingresso: l'invarianza risultava vera per qualunque asse,
    e il test passava anche sulla versione rotta. Verificato: con pesi casuali
    non morde, con i pesi veri si'.
    """
    from src.voxnet import VoxelEnsemble

    ens = VoxelEnsemble.carica(VOXNET, device="cpu")
    rng = np.random.default_rng(1)
    x = (rng.random((2, 64, 64, 64)).astype(np.float32) * 6) ** 2
    diretta = ens.predici(x)
    ribaltata = ens.predici(np.ascontiguousarray(x[:, ::-1]))
    # Il test discrimina solo se l'ingresso muove davvero la predizione: senza
    # questa riga un modello degenere lo supererebbe senza dire nulla.
    assert diretta.std() > 1e-4 or abs(diretta[0] - diretta[1]) > 1e-4, (
        "le predizioni non dipendono dall'ingresso: il test non discrimina")
    assert np.allclose(diretta, ribaltata, atol=1e-6), (diretta, ribaltata)


def test_architettura_sbagliata_fallisce_invece_di_predire_a_caso():
    """Caricare pesi di `media` dentro `toy` deve sollevare, non funzionare."""
    import torch

    from src.voxnet import VoxelEnsemble, build_model

    torch.manual_seed(0)
    stati = [{k: v.half() for k, v in build_model("media").state_dict().items()}]
    try:
        VoxelEnsemble({"arch": "toy", "in_ch": 1, "dropout": 0.3,
                       "riduzione": 2, "tta": True}, stati, "cpu")
    except Exception:
        return
    raise AssertionError("pesi di un'architettura diversa caricati senza errore")


# --------------------------------------------------------------------------
# La fusione
# --------------------------------------------------------------------------

def test_fusione_a_peso_nullo_riproduce_esattamente_il_modello_sbr():
    """Con b=0 e a=1 la fusione deve essere l'identita' sull'incumbent.

    E' cio' che rende attribuibile il guadagno: il modello nuovo non puo'
    guadagnare per il solo fatto di essere aggiunto.
    """
    from src.voxnet import fondi

    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    q = np.array([0.9, 0.1, 0.7, 0.3, 0.5])
    assert np.allclose(fondi(p, q, 1.0, 0.0), p, atol=1e-9)


def test_un_modello_nuovo_neutro_non_intacca_una_predizione_confidente():
    """Con la CNN a 0,50 — cioe' che non dice nulla — una predizione SBR
    confidente deve restare confidente.

    E' **il** meccanismo per cui sommare in logit rende il triplo che sommare in
    probabilita', ed e' anche la ragione per cui il blend a peso scalare gia'
    spedito butta via due terzi del contributo: una media convessa in
    probabilita' trascina verso 0,5 anche quando il modello nuovo non ha nulla
    da aggiungere, e su un esame confidente e corretto quella e' pura perdita.
    Il 70,6% del dataset sta in quel regime.

    In numeri, con i coefficienti spediti e p_sbr = 0,995: la fusione in logit
    da' ~0,97, una convessa a peso equivalente ~0,78. La prima versione di
    questo test confrontava invece quanto la fusione muove nella zona grigia
    rispetto ai confidenti, e **non mordeva**: anche la convessa muove di piu'
    dove l'incumbent e' incerto. Misurava una proprieta' vera di entrambe.
    """
    from src.voxnet import fondi

    a, b = 0.658, 0.498
    confidente = float(fondi(0.995, 0.50, a, b))
    assert confidente > 0.95, (
        f"una CNN neutra ha portato 0,995 a {confidente:.3f}: la fusione sta "
        f"mediando in probabilita' invece che sommare in logit")
    # E nella zona grigia, dove logit_sbr vale circa zero, la CNN deve invece
    # muovere la predizione in modo netto.
    incerto = float(fondi(0.50, 0.90, a, b))
    assert incerto > 0.60, incerto


def test_fusione_e_monotona_nel_modello_nuovo():
    """A parita' di SBR, una CNN piu' convinta non puo' abbassare la predizione."""
    from src.voxnet import fondi

    q = np.linspace(0.01, 0.99, 25)
    p = fondi(np.full_like(q, 0.4), q, 0.658, 0.498)
    assert np.all(np.diff(p) > 0), "fusione non monotona: segno del peso sbagliato?"


def test_pesi_spediti_coerenti_con_la_configurazione_documentata():
    """Il file dei pesi dice di essere cio' che `src/voxnet.py` documenta."""
    import torch

    blob = torch.load(str(VOXNET), map_location="cpu", weights_only=True)
    cfg = blob["config"]
    assert cfg["arch"] == "premip", cfg["arch"]
    assert cfg["riduzione"] == 1, "il MIP attivo usa i crop a 2 mm"
    assert cfg["tta"] is True, "il TTA e' parte della configurazione misurata"
    assert len(blob["state_dicts"]) == 15, len(blob["state_dicts"])
    a, b = blob["fusione"]["a"], blob["fusione"]["b"]
    assert 0.0 < a < 2.0 and 0.0 < b < 2.0, (a, b)


def test_pesi_spediti_si_caricano_e_predicono():
    from src.voxnet import VoxelEnsemble

    ens = VoxelEnsemble.carica(VOXNET, device="cpu")
    p = ens.predici(np.zeros((64, 64, 64), dtype=np.float32))
    assert p.shape == (1,) and 0.0 < p[0] < 1.0, p


# --------------------------------------------------------------------------
# Il ripiego: verso l'SBR, mai verso la prevalenza
# --------------------------------------------------------------------------

def test_senza_pesi_il_ramo_si_disattiva_invece_di_fallire():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "sub_main", ROOT / "submission_src" / "main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.VOXNET_PATH = ROOT / "models" / "non_esiste.pt"
    vox, motivo = m.load_voxnet()
    assert vox is None and motivo == "pesi assenti", (vox, motivo)
    assert os.environ.get("DAT_DISABLE_VOXNET") is None


def test_crop_mancante_ripiega_sull_sbr_non_sulla_prevalenza():
    """Con `crop=None` la predizione deve restare quella del modello SBR.

    E' il livello di ripiego nuovo, e vale ~0,39 di log loss per esame contro
    l'alternativa: una predizione SBR piena perde in media 0,296, la prevalenza
    0,688. Se questo si rompesse, un difetto del componente nuovo butterebbe via
    anche il componente che funziona.
    """
    import importlib.util
    import pickle

    spec = importlib.util.spec_from_file_location(
        "sub_main2", ROOT / "submission_src" / "main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with open(ROOT / "models" / "models.pkl", "rb") as h:
        bundle = pickle.load(h)

    with tempfile.TemporaryDirectory() as d:
        path = _phantom_nifti(pathlib.Path(d))
        solo_sbr, fuso_a = m.predict_one(path, bundle, None)
        # Un finto ramo a voxel che restituisce sempre crop=None non esiste:
        # si simula chiedendo la fusione con un ensemble che solleva sempre.
        class Rotto:
            def predici(self, _):
                raise RuntimeError("rete rotta")

        degradato, fuso_b = m.predict_one(path, bundle,
                                          {"ens": Rotto(), "a": 1.0, "b": 1.0})
    assert fuso_a is False and fuso_b is False
    assert abs(solo_sbr - degradato) < 1e-12, (solo_sbr, degradato)
    assert abs(solo_sbr - m.FALLBACK_PROBABILITY) > 1e-6, (
        "la predizione coincide con la prevalenza: il test non discrimina")


# --------------------------------------------------------------------------
# L'augmentation fisica: sfocatura, rumore, intensita'
# --------------------------------------------------------------------------
#
# Queste tre voci esistono perche' la differenza fra due scanner del dataset e'
# fisica prima che geometrica: risoluzione fra ~8 e ~10 mm FWHM, conteggi
# diversi, ricostruzioni che comprimono la scala in modo diverso. L'arco spedito
# non ne simulava nessuna. I test qui sotto verificano che facciano quello che
# dicono — e, il piu' importante di tutti, che **a zero non facciano nulla**:
# altrimenti l'aggiunta avrebbe cambiato in silenzio ogni arco gia' misurato.

def _arco():
    """`scripts/10_voxel_cnn.py` come modulo: il nome non e' importabile."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "arco_voxel_test", ROOT / "scripts" / "10_voxel_cnn.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _cfg_spento(**extra):
    base = {"flip": False, "mirror_channel": False, "gradi": 0.0, "scala": 0.0,
            "shift": 0.0, "sfoca": 0.0, "rumore": 0.0, "gamma": 0.0,
            "guadagno": 0.0}
    base.update(extra)
    return base


def test_augment_con_tutto_a_zero_e_l_identita_bit_per_bit():
    """Il test che protegge i dieci archi gia' misurati.

    Se una delle voci nuove mordesse con il suo valore di default, ogni numero
    prodotto prima diventerebbe incomparabile con ogni numero prodotto dopo, e
    non ci sarebbe nessun errore a dirlo. Verificato che fallisce: mettendo
    `sfoca` a 1.0 nel cfg, l'uguaglianza salta.
    """
    import torch
    import torch.nn.functional as F

    arco = _arco()
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(3, 1, 16, 16, 16)
    out = arco.augment(torch, F, x, gen, _cfg_spento())
    assert torch.equal(out, x), (out - x).abs().max().item()


def test_sfoca_allarga_e_conserva_la_massa():
    """Sfocare deve ridurre il picco e lasciare l'integrale dov'era."""
    import torch
    import torch.nn.functional as F

    arco = _arco()
    gen = torch.Generator().manual_seed(1)
    x = torch.zeros(2, 1, 24, 24, 24)
    x[:, :, 12, 12, 12] = 100.0
    out = arco._sfoca(torch, F, x, gen, fwhm_max_mm=8.0)
    assert out.max() < x.max(), (out.max().item(), x.max().item())
    assert abs(float(out.sum()) - float(x.sum())) / float(x.sum()) < 1e-3, (
        float(out.sum()), float(x.sum()))


def test_sfoca_ha_un_nucleo_diverso_per_esame():
    """Due esami identici nello stesso batch devono uscire diversi.

    Con un solo valore di FWHM per batch la diversita' arriverebbe solo fra
    batch. Il test discrimina proprio quella scorciatoia.
    """
    import torch
    import torch.nn.functional as F

    arco = _arco()
    gen = torch.Generator().manual_seed(2)
    x = torch.zeros(8, 1, 24, 24, 24)
    x[:, :, 12, 12, 12] = 100.0
    out = arco._sfoca(torch, F, x, gen, fwhm_max_mm=8.0)
    picchi = out[:, 0, 12, 12, 12]
    assert float(picchi.std()) > 1e-3, picchi.tolist()


def test_rumore_ha_varianza_proporzionale_al_segnale():
    """sd ∝ sqrt(segnale): quadruplicare il valore raddoppia la dispersione.

    E' la forma poissoniana, e distingue questa augmentation da un rumore
    gaussiano omogeneo — che insegnerebbe robustezza nel fondo e nessuna dentro
    lo striato, cioe' l'invarianza sbagliata.
    """
    import torch

    arco = _arco()
    gen = torch.Generator().manual_seed(3)
    basso = torch.full((64, 1, 8, 8, 8), 1.0)
    alto = torch.full((64, 1, 8, 8, 8), 4.0)
    sd_basso = float((arco._rumore(torch, basso, gen, 1.0) - basso).std())
    sd_alto = float((arco._rumore(torch, alto, gen, 1.0) - alto).std())
    assert 1.7 < sd_alto / sd_basso < 2.3, (sd_basso, sd_alto)


def test_intensita_e_monotona_e_non_produce_valori_negativi():
    """Guadagno e gamma non devono mai invertire l'ordine dentro un esame."""
    import torch

    arco = _arco()
    gen = torch.Generator().manual_seed(4)
    x = torch.linspace(0.0, 6.0, 512).reshape(2, 1, 8, 8, 4)
    out = arco._intensita(torch, x, gen, gamma_max=0.2, guadagno_max=0.2)
    assert float(out.min()) >= 0.0, float(out.min())
    for i in range(x.shape[0]):
        piatto_in = x[i].flatten()
        piatto_out = out[i].flatten()
        ordine = torch.argsort(piatto_in)
        assert torch.all(torch.diff(piatto_out[ordine]) >= -1e-5), i


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SALTATI: torch assente. Esegui dentro l'immagine ufficiale.")
        sys.exit(0)

    salta = set()
    if not VOXNET.is_file():
        salta |= {"test_pesi_spediti_coerenti_con_la_configurazione_documentata",
                  "test_pesi_spediti_si_caricano_e_predicono"}
        print(f"  SALTATI i test sui pesi: {VOXNET} assente")
    if not (CROPS.is_file() and FEATURES.is_file() and DATA.is_dir()):
        salta.add("test_crop_identico_a_quello_di_addestramento")
        print("  SALTATO il confronto coi crop di addestramento: dati assenti")
    if not (ROOT / "models" / "models.pkl").is_file():
        salta.add("test_crop_mancante_ripiega_sull_sbr_non_sulla_prevalenza")
        print("  SALTATO il test di ripiego: models.pkl assente")

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and n not in salta]
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
