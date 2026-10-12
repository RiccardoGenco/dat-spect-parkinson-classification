"""Entrypoint di inferenza per il runtime della competizione.

Contratto imposto dalla piattaforma:
  - eseguito come `python main.py` con working directory /code_execution/
  - il test set e' in /code_execution/data/niftis/
  - deve scrivere submission.csv nella working directory
  - nessun accesso a internet: ogni dipendenza deve stare nell'immagine ufficiale
    e ogni peso del modello dentro submission.zip
  - i log sono limitati a 300 righe da 300 caratteri

Il principio che governa questo file: **il fallimento inaccettabile non e' un
punteggio mediocre, e' non produrre alcun file.** Predire la prevalenza su un
esame costa qualche centesimo di log loss su quell'esame; un container che va in
crash o in timeout costa l'intera submission, e con essa il giro.

Da qui quattro difese, in ordine di quanto sono costate a scrivere:

1. Un esame che va in errore ripiega sulla prevalenza del training.
2. submission.csv viene scritto **subito**, riempito di prevalenza, e
   riscritto man mano. Se il processo muore a meta' - per memoria, per timeout,
   per un bug - sul disco resta comunque un file valido e completo.
3. Un budget di tempo. Superato, gli esami rimanenti prendono la prevalenza
   invece di essere elaborati. Meglio una coda di predizioni deboli che una
   submission che non esiste.
4. Un'attesa esplicita del mount dei dati, prima di leggere qualunque cosa.

La quarta esiste **a causa** della prima, ed e' il motivo per cui vale la pena
scriverla. La difesa n. 1 trasforma ogni errore in una predizione debole, il che
e' giusto per un volume malformato e disastroso per una directory non ancora
montata: il guasto smette di essere un errore e diventa un punteggio un po'
peggiore, indistinguibile da un modello un po' peggiore. Una difesa che rende
silenzioso un guasto sistemico va accompagnata da una che quel guasto lo
previene.

Due scelte progettuali ereditate dal resto della pipeline:

- Ogni esame e' processato in modo indipendente, come impone il regolamento
  ("each test data sample should be processed independently during inference"),
  quindi niente normalizzazione calcolata sull'intero test set. E' anche il
  motivo per cui l'SBR e' la feature giusta: e' gia' normalizzato al suo
  interno, sul riferimento dello stesso paziente.
- La diagnostica e' aggregata, non per esame: con il limite di 300 righe di
  log, stampare una riga per fallimento significa perdere proprio le
  informazioni che servono quando i fallimenti sono tanti.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

# I log di gara sono limitati a 300 righe da 300 caratteri. Un warning ripetuto
# da una qualunque libreria - dentro un ciclo che gira una volta per esame -
# li esaurisce e seppellisce la diagnostica proprio quando serve. "once" lascia
# passare ogni avviso distinto una volta sola: si vede comunque cosa e'
# successo, senza perdere il resto. Non sostituisce la correzione delle cause
# note (i modelli sono allenati con nomi di colonna espliciti apposta), e' la
# difesa contro quelle che non si conoscono.
warnings.simplefilter("once")

SRC_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_ROOT))

from src.pipeline import process_file  # noqa: E402

# Il percorso e' quello imposto dal contratto; la variabile d'ambiente serve
# solo a poter eseguire questo file fuori dal container, contro lo smoke test,
# senza doverlo modificare. Nel runtime non e' definita e vale il default.
DATA_DIR = Path(os.environ.get("DAT_DATA_DIR", "/code_execution/data"))
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")
MODEL_PATH = SRC_ROOT / "models.pkl"
VOXNET_PATH = SRC_ROOT / "voxnet.pt"
MANIFEST_PATH = SRC_ROOT / "artifact_manifest.json"
MANIFEST_FORMAT = "dat-parkinson-artifact/v1"

# --- Il ramo a voxel, e la regola che lo governa ---------------------------
#
# La CNN sui crop striatali aggiunge informazione che le 44 feature non vedono:
# dentro la zona grigia l'AUROC passa da 0,6938 a 0,7802, e la log loss OOF da
# 0,2959 a 0,2759 con la fusione a due parametri valutata tenendo fuori interi
# centri (`scripts/fusione.py`).
#
# **E' un componente aggiuntivo, mai un sostituto, e non deve poter peggiorare
# nulla.** Il modello SBR e' stato validato dall'esterno — OOF 0,2956 contro
# leaderboard 0,2902 — mentre la rete non lo e' ancora. Da qui una gerarchia di
# ripieghi a tre livelli invece dei due precedenti:
#
#   1. la rete non si carica          -> tutti gli esami col solo modello SBR
#   2. un esame non produce un crop   -> quell'esame col solo modello SBR
#   3. il modello SBR stesso fallisce -> quell'esame con la prevalenza
#
# Il livello 2 e' quello nuovo e il piu' importante: prima dell'aggiunta, un
# guasto del preprocessing mandava un esame alla prevalenza (perdita ~0,688);
# ora lo manda a una predizione SBR piena (perdita ~0,296). Il componente nuovo
# puo' solo aggiungere, per costruzione del codice e non per buona volonta'.
USE_VOXNET = os.environ.get("DAT_DISABLE_VOXNET", "") == ""

# Prevalenza di casi patologici nel training set (747 su 1362).
# Usata per ogni esame che non e' stato possibile elaborare: e' la predizione
# che minimizza la log loss attesa in assenza di qualunque informazione.
FALLBACK_PROBABILITY = 0.5485

# La log loss non e' limitata: una singola predizione confidente e sbagliata puo'
# dominare il punteggio di tutto il test set. Restringere l'intervallo costa
# pochissimo sui casi facili e limita il danno su quelli anomali.
CLIP_LOW, CLIP_HIGH = 0.005, 0.995

# Budget di tempo.
#
# ATTENZIONE, e la versione precedente di questo commento affermava il
# contrario: `IS_SMOKE_TEST` **non e' una variabile della piattaforma**. E' una
# variabile del justfile locale, e nel log di gara compare come vuota
# (`+ '[' -n '' ']'`), quindi anche durante lo smoke test - il cui limite reale
# e' 6 minuti - viene scelto il budget da 150. Il ramo da 5 minuti e' quindi
# codice morto in gara e la difesa che dovrebbe proteggere lo smoke test non
# c'e'. Nella pratica non ha ancora morso, perche' 20 esami girano in 16
# secondi; resta un commento che descriveva un contratto inesistente, cioe' la
# forma peggiore di documentazione.
#
# Il budget deve stare SOTTO il limite, non sopra. Un budget piu' lungo del
# limite e' una difesa inerte: la piattaforma uccide il container prima che il
# controllo scatti, e il file resta a meta'. Il margine serve a garantire che
# esista sempre un istante in cui il processo smette di elaborare e va a
# scrivere, con tempo davanti per farlo.
#
# Il margine e' abbondante perche' costa poco: la pipeline gira a 0,42 s/esame
# nel container, quindi 2,5 ore coprirebbero oltre 20.000 esami, contro i 1362
# dell'intero training set. Il budget non dovrebbe mai attivarsi; se si attiva,
# significa che e' successo qualcosa di imprevisto, ed e' esattamente la
# situazione per cui esiste.
SMOKE_TEST_BUDGET_SECONDS = 5 * 60          # limite 6 min
FULL_RUN_BUDGET_SECONDS = int(2.5 * 3600)   # limite 3 ore

TIME_BUDGET_SECONDS = (
    SMOKE_TEST_BUDGET_SECONDS
    if os.environ.get("IS_SMOKE_TEST")
    else FULL_RUN_BUDGET_SECONDS
)

# Ogni quanti esami riscrivere il file. Il costo e' trascurabile rispetto
# all'elaborazione di un volume, e in cambio la finestra in cui un crash
# lascerebbe un file non aggiornato resta piccola.
CHECKPOINT_EVERY = 25

# --- Attesa del mount ------------------------------------------------------
#
# Gli organizzatori dichiarano che /code_execution/data impiega ~15-20 secondi a
# diventare elencabile, e che l'accesso per percorso diretto "dovrebbe"
# funzionare subito. "Dovrebbe" non e' un contratto, e il modo in cui questa
# corsa si perde e' silenzioso: un file non ancora montato solleva
# FileNotFoundError, la difesa n. 1 lo degrada a prevalenza, e la riga di
# diagnostica che lo direbbe viene filtrata dalla piattaforma.
#
# Quanto costa, misurato sulla OOF: la perdita media di un esame elaborato e'
# 0,296, quella di un esame in fallback 0,688. A 0,72 s/esame, venti esami
# cadono dentro una finestra di quindici secondi; su un test set di ~500 esami
# valgono +0,016 di log loss - meta' del divario che ci separa dal terzo posto,
# e indistinguibile da un modello peggiore.
#
# L'attesa e' generosa perche' non costa nulla: si paga una volta sola, contro
# un budget di 150 minuti, e nel caso normale il primo tentativo riesce subito.
MOUNT_WAIT_SECONDS = 120
MOUNT_POLL_SECONDS = 1.0

# Finestra dall'avvio entro cui un fallimento viene ritentato invece che
# accettato. Oltre, un esame che fallisce sta fallendo per ragioni sue - un
# volume malformato, un caso che la pipeline non gestisce - e ritentarlo
# spenderebbe budget senza cambiare esito. Dentro, l'ipotesi piu' probabile e'
# il mount, e una pausa la risolve. Il costo peggiore e' limitato per
# costruzione: passata la finestra non si ritenta piu'.
MOUNT_GRACE_SECONDS = 180
RETRY_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 2.0


def wait_until_readable(path: Path, timeout: float) -> float:
    """Attende che `path` sia apribile in lettura. Restituisce i secondi attesi.

    Il controllo e' la lettura di **un byte che ci sia davvero**, non
    `path.exists()` e nemmeno una `open()` riuscita. Su un mount in corso
    l'entrata di directory puo' comparire prima del suo contenuto: `exists()`
    risponderebbe di si' su un file ancora vuoto, e `open()` riesce su un file
    da zero byte. Entrambe sarebbero guardie che passano proprio nel caso da cui
    difendono. Ne' `submission_format.csv` ne' un `.nii.gz` sono mai vuoti per
    legittimita', quindi pretendere un byte non ha falsi negativi.

    Restituisce -1 se il timeout scade, invece di sollevare: la risposta giusta
    e' diversa per il formato di submission - senza il quale non c'e' niente da
    consegnare - e per un singolo volume, che un fallback ce l'ha. La decisione
    resta del chiamante.
    """
    started = time.time()
    while True:
        try:
            with open(path, "rb") as handle:
                if not handle.read(1):
                    raise OSError("file ancora vuoto")
            return time.time() - started
        except OSError:
            if time.time() - started >= timeout:
                return -1.0
            time.sleep(MOUNT_POLL_SECONDS)


def emit(tag: str, message: str) -> None:
    """Una riga di diagnostica, marcata, su stdout e su stderr.

    SONDA TEMPORANEA, non la forma definitiva della diagnostica.

    Sulla submission del 1 agosto l'unica riga stampata da questo file e' stata
    sostituita nei log da `< ... WARNING: line filtered >`. L'ipotesi era che il
    filtro scattasse sui conteggi assoluti, perche' rivelano la dimensione del
    test set; ma poggiava anche sul listato vuoto di `data/`, che gli
    organizzatori hanno poi spiegato come mount non ancora pronto. Rimasta senza
    quel secondo appoggio, l'ipotesi non e' verificata, e riscrivere la
    diagnostica secondo una regola indovinata significherebbe restare ciechi
    credendo di aver risolto.

    Quindi si misura invece di indovinare. Ogni informazione viene emessa in tre
    forme - con numeri ma senza conteggi, con una sola percentuale, senza cifre -
    e su entrambi i flussi. Il prossimo smoke test dira' quali sopravvivono, e
    allora restera' solo quella: le altre vanno cancellate, non lasciate qui.

    Nota deliberata: **nessuna forma contiene un conteggio assoluto.** Se il
    filtro esiste per non rivelare la dimensione del test set, quello e' un
    controllo legittimo della piattaforma e questa diagnostica lo rispetta. Cio'
    che serve a noi e' il tasso di fallback dei nostri esami, che e' un rapporto
    e non dice nulla sul denominatore.
    """
    line = f"[{tag}] {message}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("formato del manifest sconosciuto")
    if not isinstance(manifest.get("models"), dict):
        raise ValueError("identita' di models.pkl assente dal manifest")
    return manifest


def verify_identity(path: Path, identity: dict, expected_name: str) -> None:
    """Rifiuta un file vecchio o sostituito dopo il packaging."""
    if identity.get("file") != expected_name:
        raise ValueError(f"il manifest non dichiara {expected_name}")
    if path.stat().st_size != identity.get("bytes"):
        raise ValueError(f"dimensione di {expected_name} diversa dal manifest")
    if sha256_file(path) != identity.get("sha256"):
        raise ValueError(f"SHA-256 di {expected_name} diverso dal manifest")


def load_models(manifest: dict):
    verify_identity(MODEL_PATH, manifest["models"], "models.pkl")
    with open(MODEL_PATH, "rb") as handle:
        return pickle.load(handle)


def load_voxnet(manifest: dict | None = None):
    """Carica il ramo a voxel, oppure restituisce None e il motivo.

    Non solleva mai. Ogni cosa che puo' andare storta qui — torch assente, file
    mancante, architettura incompatibile, VRAM insufficiente — deve produrre una
    submission col solo modello SBR, che e' il modello gia' validato dalla
    leaderboard. Il ramo nuovo non e' autorizzato a costare nulla.
    """
    if not USE_VOXNET:
        return None, "disattivato da ambiente"
    if not VOXNET_PATH.is_file():
        return None, "pesi assenti"
    try:
        if manifest is None:
            return None, "manifest assente"
        identity = manifest.get("voxel")
        if not isinstance(identity, dict):
            return None, "checkpoint non dichiarato"
        verify_identity(VOXNET_PATH, identity, "voxnet.pt")

        from src.voxnet import VoxelEnsemble

        ensemble = VoxelEnsemble.carica(VOXNET_PATH)
        import torch

        blob = torch.load(str(VOXNET_PATH), map_location="cpu", weights_only=True)
        fusione = blob["fusione"]
        config = blob["config"]
        expected_config = identity.get("config", {})
        actual_config = {
            "arch": config.get("arch"),
            "norm": config.get("norm", "batch"),
            "autonorm": bool(config.get("autonorm", False)),
            "n_reti": len(blob["state_dicts"]),
        }
        if actual_config != expected_config:
            raise ValueError(
                f"config checkpoint {actual_config} diversa dal manifest "
                f"{expected_config}"
            )
        expected_fusion = identity.get("fusion", {})
        for key in ("a", "b"):
            if abs(float(fusione[key]) - float(expected_fusion[key])) > 1e-9:
                raise ValueError(f"fusione {key} diversa dal manifest")
        # Una prova a vuoto PRIMA del primo esame: se la rete non sa predire un
        # cubo di zeri non sapra' predire nemmeno un paziente, e scoprirlo qui
        # costa un millisecondo invece di un errore ripetuto centinaia di volte.
        import numpy as _np

        prova = ensemble.predici(_np.zeros((64, 64, 64), dtype=_np.float32))
        if not _np.all(_np.isfinite(prova)):
            return None, "prova a vuoto non finita"
        return {
            "ens": ensemble,
            "a": float(fusione["a"]),
            "b": float(fusione["b"]),
            "device": ensemble.device,
            "n": len(ensemble),
            "artifact_id": identity["sha256"][:12],
            "arch": actual_config["arch"],
            "norm": actual_config["norm"],
            "autonorm": actual_config["autonorm"],
            "arco": expected_fusion.get("arco", "?"),
            "meter": identity.get("meter", {}).get("name", "?"),
        }, "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}"


def predict_one(path: Path, bundle, vox=None) -> tuple[float, bool]:
    """Probabilita' per un esame, piu' se il ramo a voxel ha contribuito."""
    crop = None
    if vox is None:
        features = process_file(path)
    else:
        features, crop = process_file(path, with_crop=True)
    # DataFrame con i nomi delle colonne, non array nudo: i modelli sono stati
    # allenati con i nomi, e passare una matrice anonima fa emettere a sklearn
    # un warning per ogni modello e per ogni esame. Con cinque modelli e i log
    # di gara limitati a 300 righe, su qualche centinaio di esami la
    # diagnostica verrebbe sepolta. Il contratto per nome ha comunque un
    # valore suo: un disallineamento nell'ordine delle colonne diventa un
    # errore invece di una predizione sbagliata in silenzio.
    row = pd.DataFrame(
        [[features.get(name, np.nan) for name in bundle["columns"]]],
        columns=list(bundle["columns"]),
        dtype=np.float64,
    )

    gbm_predictions = [m.predict_proba(row)[0, 1] for m in bundle["gbm"]]
    lr_predictions = [m.predict_proba(row)[0, 1] for m in bundle["lr"]]

    weight = bundle["weight"]
    probability = weight * float(np.mean(gbm_predictions)) + (1 - weight) * float(
        np.mean(lr_predictions)
    )
    if not np.isfinite(probability):
        raise ValueError("predizione non finita")

    if vox is None or crop is None:
        return float(np.clip(probability, CLIP_LOW, CLIP_HIGH)), False

    # Livello 2 del ripiego. Un guasto qui dentro riguarda SOLO il contributo
    # della rete: la predizione SBR e' gia' calcolata e resta valida, quindi
    # l'esame non perde nulla rispetto a prima che il ramo a voxel esistesse.
    try:
        from src.voxnet import fondi

        # Il ramo fortemente regolarizzato e' stato misurato specificamente
        # accanto alla MIP. Resta separato dal modello SBR di ripiego, che da
        # solo e' migliore e deve sopravvivere intatto a qualunque guasto CNN.
        fusion_models = bundle.get("fusion_lr", [])
        fusion_columns = bundle.get("fusion_lr_columns", bundle["columns"])
        fusion_row = pd.DataFrame(
            [[features.get(name, np.nan) for name in fusion_columns]],
            columns=list(fusion_columns), dtype=np.float64,
        )
        registered_probability = (
            float(np.mean([m.predict_proba(fusion_row)[0, 1] for m in fusion_models]))
            if fusion_models else probability
        )
        base_fusion_models = bundle.get("fusion_lr_base", [])
        if base_fusion_models:
            base_probability = float(np.mean([
                m.predict_proba(row)[0, 1] for m in base_fusion_models
            ]))
            weight = float(bundle.get("fusion_lr_blend_weight", 0.5))
            fusion_probability = (
                weight * registered_probability + (1.0 - weight) * base_probability
            )
        else:
            fusion_probability = registered_probability
        p_cnn = float(vox["ens"].predici(crop)[0])
        fused = float(fondi(fusion_probability, p_cnn, vox["a"], vox["b"]))
        if not np.isfinite(fused):
            raise ValueError("fusione non finita")
        return float(np.clip(fused, CLIP_LOW, CLIP_HIGH)), True
    except Exception:  # noqa: BLE001
        return float(np.clip(probability, CLIP_LOW, CLIP_HIGH)), False


def main() -> None:
    boot = time.time()

    # Prima lettura in assoluto, e quindi la piu' esposta al mount. Senza il
    # formato non si sa nemmeno quali uid predire: non c'e' fallback possibile,
    # e l'unica difesa e' aspettare.
    if wait_until_readable(SUBMISSION_FORMAT_PATH, MOUNT_WAIT_SECONDS) < 0:
        emit("A", "FORMATO DI SUBMISSION NON LEGGIBILE dopo l'attesa del mount: "
                  "impossibile sapere quali esami predire")
        return

    submission = pd.read_csv(SUBMISSION_FORMAT_PATH)
    n = len(submission)

    # Submission valida su disco PRIMA di elaborare qualunque cosa. Da questo
    # momento nessun fallimento successivo puo' produrre l'assenza di un file.
    submission["is_pathologic"] = FALLBACK_PROBABILITY
    submission.to_csv(WRITE_SUBMISSION_PATH, index=False)

    try:
        manifest = load_manifest()
        bundle = load_models(manifest)
    except Exception as exc:  # noqa: BLE001
        # Senza modello non c'e' niente da fare se non consegnare la prevalenza.
        # Vale comunque piu' di un crash: la log loss sara' quella della
        # baseline invece di essere indefinita.
        emit("A", f"MODELLO NON CARICABILE ({type(exc).__name__}: {exc}) - "
                  f"consegno la prevalenza sul 100% degli esami")
        return

    source = manifest.get("source", {})
    commit = source.get("git_commit")
    dirty = source.get("git_dirty")
    source_state = "sconosciuto" if dirty is None else ("sporco" if dirty else "pulito")
    emit("M", f"artefatto verificato: models {manifest['models']['sha256'][:12]}, "
              f"sorgente {commit[:12] if commit else 'non identificato'}, "
              f"worktree {source_state}")

    # Il ramo a voxel si carica UNA volta, prima del ciclo, e se non si carica
    # si prosegue col solo modello SBR. Caricarlo qui e non al primo esame e'
    # deliberato: un guasto scoperto dentro il ciclo si ripeterebbe per ogni
    # esame, consumando budget e riempiendo i log — che sono limitati a 300
    # righe — proprio mentre si degrada.
    vox, vox_reason = load_voxnet(manifest)

    # Sentinella del mount dei volumi: il primo esame che andra' letto. Se
    # arriva quello sono arrivati anche gli altri. Se non arriva si procede lo
    # stesso - un singolo file puo' mancare per ragioni sue, e il retry per
    # esame resta come seconda rete - ma l'attesa scaduta finisce nella
    # diagnostica, perche' e' esattamente il sintomo da non perdere.
    mount_wait = wait_until_readable(
        NIFTI_DIR / f"{submission['uid'].iloc[0]}.nii.gz", MOUNT_WAIT_SECONDS
    )

    predictions = [FALLBACK_PROBABILITY] * n
    reasons: Counter[str] = Counter()
    started = time.time()
    n_processed = 0
    n_retried = 0        # esami che hanno avuto bisogno di almeno un ritentativo
    n_recovered = 0      # ...e che grazie a quello sono riusciti
    n_fused = 0          # esami in cui il ramo a voxel ha davvero contribuito
    failed_at: list[int] = []   # posizione nella sequenza di ogni esame fallito

    for i, uid in enumerate(submission["uid"]):
        # Il budget si misura da `boot` e non da qui: comprende quindi anche
        # l'attesa del mount, che nel caso peggiore vale due minuti. Misurarlo
        # dall'inizio del ciclo lo farebbe sommare al limite invece che starci
        # dentro, ed e' proprio l'errore contro cui il commento sul budget mette
        # in guardia: una difesa piu' lunga del limite non e' una difesa.
        if time.time() - boot > TIME_BUDGET_SECONDS:
            reasons["budget di tempo esaurito"] += n - i
            failed_at.extend(range(i, n))
            break

        path = NIFTI_DIR / f"{uid}.nii.gz"
        retried = False
        for attempt in range(RETRY_ATTEMPTS):
            try:
                predictions[i], fused = predict_one(path, bundle, vox)
                n_processed += 1
                n_fused += fused
                n_recovered += retried
                break
            except Exception as exc:  # noqa: BLE001
                # Il tipo di eccezione non distingue i due casi, ed e' il motivo
                # per cui la finestra temporale fa da criterio al posto suo: un
                # gzip a meta' mount fa sollevare a nibabel ImageFileError o un
                # errore di decompressione, non FileNotFoundError. Filtrare per
                # tipo lascerebbe fuori proprio la forma piu' probabile del
                # guasto da cui questa difesa esiste per proteggere.
                within_grace = time.time() - boot < MOUNT_GRACE_SECONDS
                if attempt + 1 < RETRY_ATTEMPTS and within_grace:
                    retried = True
                    time.sleep(RETRY_PAUSE_SECONDS)
                    continue
                reasons[type(exc).__name__] += 1
                failed_at.append(i)
                break
        n_retried += retried

        if (i + 1) % CHECKPOINT_EVERY == 0:
            submission["is_pathologic"] = predictions
            submission.to_csv(WRITE_SUBMISSION_PATH, index=False)

    submission["is_pathologic"] = predictions
    submission.to_csv(WRITE_SUBMISSION_PATH, index=False)

    # Tre forme della stessa informazione, nessuna con conteggi assoluti. Quale
    # sopravviva al filtro della piattaforma e' una domanda aperta: vedi emit().
    elapsed = time.time() - started
    done = n_processed / n
    emit("A", f"elaborato il {done:.1%} degli esami, "
              f"{elapsed / max(n_processed, 1):.2f} s/esame, "
              f"attesa mount {'SCADUTA' if mount_wait < 0 else f'{mount_wait:.0f} s'}, "
              f"budget usato {(time.time() - boot) / TIME_BUDGET_SECONDS:.1%}")
    # Tre tassi distinti e non uno solo: "esami ritentati" dice quanto e' servita
    # la difesa, "recuperati" quanto ha reso. La versione precedente di questa
    # riga divideva il numero di TENTATIVI per il numero di esami, che con tre
    # tentativi a testa poteva superare il 100% e non era il tasso di niente.
    emit("B", f"fallback {1 - done:.1%}, "
              f"ritentati {n_retried / n:.1%}, recuperati {n_recovered / n:.1%}")
    emit("C", "esito nominale, nessun esame in fallback" if n_processed == n
              else "esito degradato, esami in fallback")
    # Il ramo a voxel come TASSO, non come conteggio, per la stessa ragione di
    # tutto il resto. Se il ramo non si e' caricato la riga lo dice comunque:
    # un componente che sparisce in silenzio si presenta come un modello un po'
    # peggiore, ed e' il modo di sbagliare direzione che questo file combatte.
    if vox is None:
        emit("V", f"ramo a voxel INATTIVO ({vox_reason}) - solo modello SBR")
    else:
        emit("V", f"voxel {vox['artifact_id']} attivo su {vox['device']}, "
                  f"{vox['n']} reti {vox['arch']}/{vox['norm']}, "
                  f"autonorm={'si' if vox['autonorm'] else 'no'}, arco {vox['arco']}, "
                  f"metro {vox['meter']}, fusione a={vox['a']:.3f} b={vox['b']:.3f}, "
                  f"usato sul {n_fused / max(n_processed, 1):.1%}")
    if failed_at:
        # Aggregata e non per esame: con 300 righe di log a disposizione, una
        # riga per fallimento cancellerebbe proprio il riepilogo che serve.
        emit("D", "cause: " + ", ".join(f"{reason} {count / n:.1%}"
                                        for reason, count in reasons.most_common(6)))
        # DOVE cadono i fallimenti separa le due diagnosi che altrimenti si
        # confondono, ed e' la distinzione che questo intero lavoro esiste per
        # rendere possibile: un mount lento colpisce l'inizio della sequenza e
        # basta, un difetto della pipeline colpisce ovunque. Senza questo numero
        # un "fallback 4%" resta ambiguo, e un ambiguo si interpreta come il
        # modello - cioe' si ottimizza nella direzione sbagliata per settimane.
        median = sorted(failed_at)[len(failed_at) // 2] / n
        emit("E", f"fallimenti concentrati intorno al {median:.0%} della sequenza"
                  f" ({'inizio: compatibile col mount' if median < 0.10 else 'sparsi: non e la finestra di mount'})")


if __name__ == "__main__":
    main()
