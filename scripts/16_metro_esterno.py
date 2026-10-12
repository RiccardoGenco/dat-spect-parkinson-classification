#!/usr/bin/env python3
"""Il metro esterno: un centro intero predetto come lo predirebbe il container.

Perche' esiste, in una riga
---------------------------
Dieci archi di CNN sono stati ordinati su un metro che ha promesso +0,0213 e
consegnato +0,0018. Qualunque strategia che passi per un modello di immagine ha
bisogno di un metro che *ordini* i candidati, e quello attuale ha dimostrato di
non farlo nel regime ad alta capacita'. Questo file e' il tentativo di
costruirne uno.

Cosa fa di diverso dalla CV raggruppata
---------------------------------------
`02_train_cv.py` tiene fuori un fold ma riusa il resto della catena addestrato
altrove: le OOF della CNN vengono da un'altra esecuzione, i coefficienti (a, b)
sono stimati su tutte le OOF, e il punteggio si legge su una composizione di
reti che in inferenza non esiste.

Qui, per ogni centro tenuto fuori, si rifa' **l'intera catena** sui soli centri
rimasti — LightGBM, logistica, peso del blend, N reti, stima di (a, b) — e si
predice il centro escluso con gli artefatti congelati, esattamente come
`main.py` predice un esame mai visto.

Cio' che NON viene corretto, di proposito
------------------------------------------
Due divergenze note fra il percorso misurato e quello del container restano
dentro: la TTA si media in **logit** durante l'addestramento e in
**probabilita'** in inferenza, e le OOF interne vengono da una rete per seed
mentre la predizione esterna ne usa N. Toglierle renderebbe il metro piu' bello
e meno utile — il loro posto e' dentro la stima, non fuori. Per questo la
predizione sul centro escluso passa da `VoxelEnsemble`, cioe' dalla classe che
gira nel container, e non da una media riscritta qui.

Perche' due partizioni, e perche' non si valida contro un numero solo
----------------------------------------------------------------------
La tentazione e' «il metro e' buono se riproduce 0,2884». Non regge: su ~525
esami quel numero ha un errore standard di ~0,005, quindi *qualunque* metro fra
0,278 e 0,298 lo «riprodurrebbe». Validare contro un punto rumoroso e' come
tarare una bilancia su un oggetto di peso ignoto.

Questo metro produce invece **due numeri che devono racchiudere la gara**:

- **a gruppi** (`--esterni gruppi`): ogni impronta scanner grande e' un fold.
  Piu' severo del test vero, che viene dagli stessi dieci ospedali del
  training: tenere fuori un centro intero e' uno scenario che in gara non
  capita. Limite **pessimistico**.
- **a strati** (`--esterni strati`): fold casuali stratificati, stessa taglia,
  gruppi ignorati. Ogni centro e' rappresentato in addestramento — la
  situazione della gara — ma senza il pezzo di generalizzazione che il test
  comunque richiede. Limite **ottimistico**.

| dove cade il punteggio di gara | lettura |
|---|---|
| fra i due | il metro racchiude la gara: si puo' usare per ordinare |
| **peggio** del pessimistico | non e' la stima a sbagliare, e' il container a degradare qualcosa |
| meglio dell'ottimistico | il metro e' severo oltre la gara: ordina, non predice |

E resta vero che un metro simulato ordina e non predice (R12). Il modo di
ancorarlo e' spendere **due submission appaiate** con un contrasto noto — la
stessa configurazione con `b=0` e con `b` scelto — e verificare che il segno
del Delta esterno e quello del Delta di gara coincidano. Un contrasto appaiato
sopravvive al rumore di 0,005 che un livello assoluto non sopravvive.

Che cosa si porta a casa gratis
--------------------------------
L'AUROC della CNN **su ciascun centro tenuto fuori**: e' la misura diretta di
quanto il ramo a voxel generalizza fuori dai centri che ha visto, e non
esisteva. Sta in `per_fold[].cnn_auroc_out`.

Costo
-----
Il collo di bottiglia sono le reti: (fold esterni) x (5 fold interni) x (seed).
Con 5 fold esterni e `--train-seeds 0-3` sono 100 reti, ~55 minuti sulla GPU su
cui l'arco spedito ne ha fatte 20 in 11. Con `--train-seeds 0-1` sono 50 e la
dispersione fra seed resta misurabile. Gira solo dentro l'immagine ufficiale.

Uso:
    python scripts/16_metro_esterno.py --esterni gruppi --arch media --tta --riduci 2 --train-seeds 0-3
    python scripts/16_metro_esterno.py --esterni strati --arch media --tta --riduci 2 --train-seeds 0-3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cv import make_folds  # noqa: E402
from src.model import TARGET, as_frame, feature_columns, fit_and_score  # noqa: E402
from src.voxnet import (ARCHITETTURE, ARCHITETTURE_25D, VoxelEnsemble,  # noqa: E402
                        autonormalizza, fondi, stima_fusione)

# Gli stessi di `submission_src/main.py`: il metro deve clippare dove clippa il
# container, altrimenti misura una funzione che non viene spedita.
CLIP_LO, CLIP_HI = 0.005, 0.995
LATO = 64

# Le fusioni messe a confronto. `None` = solo SBR (il controllo che deve uscire
# identico all'incumbent), `"stima"` = i coefficienti stimati dentro il fold,
# cioe' la regola davvero spedita.
FUSIONI: dict[str, object] = {
    "solo_sbr": None,
    "stimata_nel_fold": "stima",
    "spedita_a0.658_b0.498": (0.658, 0.498),
    "robusta_a1_b0.498": (1.0, 0.498),
    "robusta_a1_b0.30": (1.0, 0.30),
    "robusta_a1_b0.25": (1.0, 0.25),
    "robusta_a1_b0.20": (1.0, 0.20),
}


def carica_arco():
    """Carica `10_voxel_cnn.py` come modulo, nonostante il nome.

    Il file non e' importabile per nome — comincia con una cifra — e riscrivere
    qui il ciclo di addestramento produrrebbe due definizioni divergenti di
    "come si addestra una rete": esattamente la classe di difetto contro cui
    `src/voxnet.py` e `src/pipeline.py` esistono. Meglio un importlib brutto di
    una copia elegante.
    """
    percorso = Path(__file__).resolve().parent / "10_voxel_cnn.py"
    spec = importlib.util.spec_from_file_location("arco_voxel", percorso)
    modulo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modulo)
    return modulo


# --------------------------------------------------------------------------- #
# Le partizioni esterne
# --------------------------------------------------------------------------- #

def fold_esterni(groups: np.ndarray, y: np.ndarray, modo: str, min_n: int,
                 seed: int = 0, n_strati: int = 0) -> list[tuple[str, np.ndarray]]:
    """Lista di `(nome, indici tenuti fuori)`.

    A gruppi: ogni impronta con almeno `min_n` esami e' un fold suo, e tutte le
    rare finiscono in un fold unico. Non e' una scelta estetica: un fold da 8
    esami non ha un errore standard leggibile, e mescolare le rare in un solo
    blocco riproduce la riga «35 impronte rare» della diagnosi, dove la fusione
    danneggiava.

    A strati: lo stesso numero di fold, stessa taglia complessiva, gruppi
    ignorati. Le taglie dei singoli fold non coincidono con quelle a gruppi, e
    non devono: cio' che si tiene fisso e' il numero di ripetizioni e il totale
    degli esami valutati, non la composizione — che e' proprio la variabile in
    esame.
    """
    if modo == "gruppi":
        conteggi = pd.Series(groups).value_counts()
        grandi = [g for g, n in conteggi.items() if n >= min_n]
        fold = [(str(g), np.flatnonzero(groups == g)) for g in grandi]
        resto = np.flatnonzero(~np.isin(groups, grandi))
        if resto.size:
            fold.append((f"({len(conteggi) - len(grandi)} impronte rare)", resto))
        return fold

    from sklearn.model_selection import StratifiedKFold

    conteggi = pd.Series(groups).value_counts()
    n_fold = len([g for g, n in conteggi.items() if n >= min_n]) + \
        (1 if (conteggi < min_n).any() else 0)
    if n_strati:
        n_fold = n_strati
    splitter = StratifiedKFold(n_splits=max(n_fold, 2), shuffle=True, random_state=seed)
    return [(f"strato {i}", np.asarray(out))
            for i, (_, out) in enumerate(splitter.split(np.zeros(len(y)), y))]


# --------------------------------------------------------------------------- #
# I due rami, addestrati sui soli centri rimasti
# --------------------------------------------------------------------------- #

def sbr_fuori_campione(frame: pd.DataFrame, columns: list[str], groups: np.ndarray,
                       dentro: np.ndarray, fuori: np.ndarray, seed: int) -> dict:
    """Addestra il modello SBR su `dentro` e predice `fuori`.

    Il percorso di predizione e' quello di `main.py` riga per riga: media delle
    probabilita' dei LightGBM, media di quelle dei logistici, blend col peso
    scelto **dentro** il fold. Il DataFrame con i nomi delle colonne non e'
    pignoleria: e' cio' che trasforma un disallineamento d'ordine in un errore
    invece che in una predizione sbagliata in silenzio.
    """
    sotto = frame.iloc[dentro].reset_index(drop=True)
    res = fit_and_score(sotto, columns, seed=seed, n_splits=5, groups=groups[dentro])

    X_out = as_frame(frame.iloc[fuori][columns].to_numpy(dtype=np.float64), columns)
    p_gbm = float_mean([m.predict_proba(X_out)[:, 1] for m in res["models_gbm"]])
    p_lr = float_mean([m.predict_proba(X_out)[:, 1] for m in res["models_lr"]])
    peso = res["weight"]

    return {
        "p_out": peso * p_gbm + (1 - peso) * p_lr,
        "oof_dentro": res["oof"],
        "peso": peso,
        "log_loss_dentro": res["log_loss"],
    }


def float_mean(colonne: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.asarray(colonne, dtype=np.float64), axis=0)


def cnn_fuori_campione(arco, torch, nn, F, vol, y, groups, dentro, fuori, cfg,
                       seeds: list[int], fold_seed: int, tag: int, log,
                       dev: str = "cuda") -> dict:
    """Addestra N reti sui soli centri di `dentro`, predice `fuori` con l'ensemble.

    Le OOF interne servono a una cosa sola: stimare (a, b) senza guardare il
    centro escluso. Vengono dal percorso di addestramento, con la sua media in
    logit; la predizione esterna viene da `VoxelEnsemble`, con la sua media in
    probabilita'. La differenza fra i due percorsi e' proprio cio' che il metro
    deve contenere, non correggere.

    `dev` esiste per una ragione sola e non e' la portabilita': permette di
    eseguire questo percorso su CPU con volumi minuscoli dentro
    `tests/test_metro.py`. E' il pezzo piu' caro del metro — 55 minuti di GPU —
    e senza un modo di collaudarlo a costo zero un errore di indicizzazione si
    scoprirebbe dopo un'ora, o mai, perche' produrrebbe un numero plausibile.
    """
    from sklearn.model_selection import GroupShuffleSplit

    folds = make_folds(y[dentro].astype(int), groups[dentro], n_splits=5, seed=fold_seed)
    per_seed = []
    stati = []

    for s in seeds:
        oof = np.zeros(len(dentro))
        for f in sorted(np.unique(folds)):
            tr = dentro[folds != f]
            va = dentro[folds == f]
            rs = fold_seed + int(f) + 1000 * int(s) + 100000 * tag
            if cfg["early_stop"]:
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                fit_pos, stop_pos = next(splitter.split(tr, y[tr], groups[tr]))
                fit_idx, stop_idx = tr[fit_pos], tr[stop_pos]
            else:
                fit_idx, stop_idx = tr, tr[:1]

            pred, interna, epoche, _, stato = arco.train_one_fold(
                torch, nn, F, vol, y, fit_idx, stop_idx, va, cfg,
                seed=int(s) * 100 + int(f) + 10000 * tag,
            )
            oof[folds == f] = pred
            stati.append({k: v.detach().half().cpu() for k, v in stato.items()})
            if dev == "cuda":
                torch.cuda.empty_cache()
            log(f"      seed {s} fold {f}: {epoche} epoche, interna {interna:.4f}")
        per_seed.append(oof)

    ens = VoxelEnsemble(
        {"arch": cfg["arch"], "in_ch": 1, "dropout": cfg["dropout"],
         "norm": cfg.get("norm", "batch"),
         "riduzione": cfg["riduci"], "tta": cfg["tta"]},
        stati, device=dev,
    )
    p_out = np.concatenate([
        ens.predici(vol[torch.from_numpy(fuori[k:k + 32]).to(dev), 0]
                    .float().cpu().numpy())
        for k in range(0, len(fuori), 32)
    ])
    del ens, stati
    if dev == "cuda":
        torch.cuda.empty_cache()

    return {"p_out": p_out, "oof_dentro": np.mean(per_seed, axis=0), "n_reti": len(seeds) * 5}


# --------------------------------------------------------------------------- #
# Lettura
# --------------------------------------------------------------------------- #

def metriche(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    p = np.clip(p, CLIP_LO, CLIP_HI)
    return {"log_loss": float(log_loss(y, p)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")}


def verdetto(pessimistico: float | None, ottimistico: float | None,
             gara: float, errore: float) -> str:
    if pessimistico is None or ottimistico is None:
        return ("un solo lato del metro: esegui anche l'altra partizione "
                "(--esterni strati/gruppi) prima di leggere un verdetto")
    if gara > pessimistico + errore:
        return ("la gara e' PEGGIO del limite pessimistico: non e' la stima a "
                "sbagliare, e' il percorso del container a degradare qualcosa")
    if gara < ottimistico - errore:
        return ("la gara e' MEGLIO del limite ottimistico: il metro e' severo "
                "oltre la gara — ordina, non predice")
    return "il metro racchiude la gara: utilizzabile per ordinare i candidati"


def etichetta_configurazione(args) -> str:
    """Nome della copia che nessun'altra configurazione puo' sovrascrivere.

    Deve nominare OGNI parametro che cambia cio' che si misura, non solo quelli
    che erano in discussione il giorno in cui la copia e' stata introdotta. Con
    la sola `norm` dentro — com'era fino al 4 agosto — una corsa con augmentation
    geometrica avrebbe scritto sopra l'artefatto della corsa senza: stesso nome,
    contenuto diverso, nessun errore. Sarebbero andati persi insieme il confronto
    appaiato e l'ingresso di `check_fusione_per_centro`, che quell'artefatto lo
    legge per autorizzare una submission.

    E' esattamente il guasto del 3 agosto — la corsa `istanza` che sovrascrisse
    quella `batch` — con la riparazione che allora copriva un parametro solo. Un
    difetto che si ripresenta identico in un posto vicino non e' sfortuna: e' una
    riparazione fatta sul sintomo.

    I valori predefiniti non entrano nel nome, cosi' le corse gia' fatte
    conservano l'etichetta che hanno.
    """
    pezzi = [args.norm]
    if args.autonorm:
        pezzi.append("autonorm")
    for sigla, valore in (("g", args.gradi), ("s", args.scala), ("sf", args.sfoca),
                          ("ru", args.rumore), ("ga", args.gamma),
                          ("gu", args.guadagno)):
        if valore:
            pezzi.append(f"{sigla}{valore:g}".replace(".", ""))
    if args.shift != 4.0:
        pezzi.append(f"sh{args.shift:g}".replace(".", ""))
    if not args.flip:
        pezzi.append("noflip")
    if not args.tta:
        pezzi.append("notta")
    if args.riduci != 2:
        pezzi.append(f"rid{args.riduci}")
    if args.arch != "media":
        pezzi.append(args.arch)
    if getattr(args, "preaddestrato", None):
        pezzi.append("imagenet")
    return "_cnn_" + "_".join(pezzi)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--esterni", choices=("gruppi", "strati"), default="gruppi")
    p.add_argument("--min-n", type=int, default=120,
                   help="taglia minima perche' un'impronta sia un fold suo")
    p.add_argument("--features", type=Path, default=Path("features_v4.csv"))
    p.add_argument("--crops", type=Path, default=Path("crops"))
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="seed del modello SBR")
    p.add_argument("--train-seeds", default="0-3")
    p.add_argument("--arch", choices=sorted(ARCHITETTURE) + sorted(ARCHITETTURE_25D),
                   default="premip")
    p.add_argument("--preaddestrato", type=Path, default=None,
                   help="radice dei pesi ImageNet per archi 2,5D/MIP; usata "
                        "solo durante l'addestramento del metro")
    p.add_argument("--autonorm", action="store_true",
                   help="toglie la regione di normalizzazione occipitale; "
                        "vedi src/voxnet.py:autonormalizza")
    p.add_argument("--norm", choices=("batch", "istanza", "gruppo"), default="batch",
                   help="come in 10_voxel_cnn.py: `batch` porta dentro le "
                        "statistiche del training set, dominato da un solo "
                        "centro; `istanza` e `gruppo` normalizzano dentro il "
                        "singolo esame. Qui serve a misurare sul metro esterno "
                        "cio' che il protocollo interno ha gia' mostrato.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--riduci", type=int, default=2)
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no-tta", dest="tta", action="store_false")
    p.add_argument("--gradi", type=float, default=0.0)
    p.add_argument("--scala", type=float, default=0.0)
    p.add_argument("--shift", type=float, default=4.0)
    p.add_argument("--sfoca", type=float, default=0.0)
    p.add_argument("--rumore", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--guadagno", type=float, default=0.0)
    p.add_argument("--no-flip", dest="flip", action="store_false")
    p.add_argument("--dispositivo", choices=("cuda", "cpu"), default="cuda",
                   help="cpu solo per collaudo: il ramo a voxel diventa impraticabile")
    p.add_argument("--senza-cnn", action="store_true",
                   help="solo il ramo SBR: due minuti, e dice se il metro regge "
                        "sul modello di cui gia' conosciamo la risposta")
    p.add_argument("--n-strati", type=int, default=0,
                   help="numero di fold della partizione a strati. 0 = quello della "
                        "partizione a gruppi, l'unico confrontabile. Serve a "
                        "misurare quanto della forbice viene dalla QUANTITA' di "
                        "dati invece che dalla loro composizione: a 10 fold ogni "
                        "modello vede il 90% invece dell'80%, e la differenza "
                        "fra le due corse e' la pendenza della curva di "
                        "apprendimento, non un effetto di centro")
    p.add_argument("--gara", type=float, default=0.2884,
                   help="il punteggio di gara della configurazione misurata")
    p.add_argument("--gara-auroc", type=float, default=0.9447,
                   help="l'AUROC di gara della stessa configurazione. E' la "
                        "lettura che conta di piu': invariante a qualunque "
                        "ricalibrazione monotona, quindi isola la separazione — "
                        "che e' l'unica leva rimasta — dalla calibrazione, che e' "
                        "chiusa")
    p.add_argument("--errore-gara", type=float, default=0.0239,
                   help="errore standard ASSOLUTO del punteggio di gara su ~525 "
                        "esami. Misurato, non assunto: la deviazione standard "
                        "della perdita per esame e' 0,5470, e 0,5470/sqrt(525) "
                        "= 0,0239. Il default precedente, 0,005, non era mai "
                        "stato misurato — e siccome questo numero decide se la "
                        "stima del metro e' compatibile con la gara, un valore "
                        "cinque volte troppo piccolo faceva dichiarare "
                        "incompatibili scarti che sono rumore. Da non "
                        "confondere con l'errore APPAIATO fra due invii "
                        "successivi, che e' 0,0075 perche' gli stessi esami si "
                        "elidono: quello e' il numero giusto per confrontare "
                        "due archi fra loro, questo per confrontare il metro "
                        "con il tabellone.")
    p.add_argument("--outdir", type=Path, default=Path("experiments/risultati"))
    args = p.parse_args()

    frame = pd.read_csv(args.features).dropna(subset=[TARGET]).reset_index(drop=True)
    columns = feature_columns(frame)
    y = frame[TARGET].to_numpy(np.float32)
    groups = frame["_group"].to_numpy()
    n = len(frame)

    arco = torch = nn = F = vol = None
    if not args.senza_cnn:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F

        torch, nn, F = _torch, _nn, _F
        if args.dispositivo == "cuda" and not torch.cuda.is_available():
            print("ERRORE: serve la GPU. Esegui dentro l'immagine ufficiale, "
                  "oppure --dispositivo cpu (solo per collaudo: e' ~100x piu' lento).",
                  file=sys.stderr)
            return 1
        arco = carica_arco()
        crops = np.load(args.crops / "crops.npy", mmap_mode="r")
        assert len(crops) == n, (len(crops), n)
        if crops.ndim != 4:
            print(f"ERRORE: forma crop {crops.shape}, attesa (N,X,Y,Z)",
                  file=sys.stderr)
            return 1
        # Un crop **piu' alto** del lato voluto va finestrato prima, e questo
        # script non sa farlo: e' un'operazione di `10_voxel_cnn.py`, e farla qui
        # in un secondo posto sarebbe la duplicazione che il progetto evita.
        if crops.shape[3] > LATO:
            print(f"ERRORE: crop con {crops.shape[3]} fette in z, piu' di {LATO}: "
                  "va finestrato prima. Usa --crops crops.", file=sys.stderr)
            return 1
        if min(crops.shape[1:]) < 16:
            print(f"ERRORE: crop {crops.shape[1:]}: con quattro blocchi a passo 2 "
                  "un asse sotto i 16 voxel collassa a 1 prima della media "
                  "globale.", file=sys.stderr)
            return 1
        if tuple(crops.shape[1:]) != (LATO, LATO, LATO):
            print(f"ERRORE: crop {tuple(crops.shape[1:])}, "
                  f"atteso {(LATO, LATO, LATO)}", file=sys.stderr)
            return 1
        vol = torch.from_numpy(
            np.array(crops, dtype=np.float16)).to(args.dispositivo)[:, None]
        if args.autonorm:
            vol = autonormalizza(vol)

    cfg = {
        # `autonorm` sta qui perche' la prima corsa che lo usava NON lo
        # registrava: i due artefatti da confrontare erano indistinguibili
        # dal proprio JSON, e l'unica prova di quale fosse quale era la riga
        # di comando in un log. R13 — se due artefatti devono essere
        # confrontabili, cio' che li distingue deve stare dentro di loro.
        "arch": args.arch, "norm": args.norm, "autonorm": bool(args.autonorm),
        "epochs": args.epochs, "patience": args.patience,
        "batch": args.batch, "lr": args.lr, "wd": args.wd, "dropout": args.dropout,
        "cosine": False, "mixup": 0.0, "smooth": 0.0,
        "gradi": args.gradi, "scala": args.scala, "shift": args.shift,
        "sfoca": args.sfoca, "rumore": args.rumore, "gamma": args.gamma,
        "guadagno": args.guadagno,
        "flip": args.flip, "mirror_channel": False,
        "tta": args.tta, "amp": True, "riduci": args.riduci, "early_stop": True,
        "preaddestrato": str(args.preaddestrato) if args.preaddestrato else None,
    }
    seeds = arco.parse_seeds(args.train_seeds) if arco is not None else []

    esterni = fold_esterni(groups, y, args.esterni, args.min_n, seed=args.fold_seed,
                           n_strati=args.n_strati)
    print(f"{n} esami, {len(columns)} feature, partizione esterna '{args.esterni}': "
          f"{len(esterni)} fold")
    for nome, idx in esterni:
        print(f"  {nome:28s} n={len(idx):4d}  prevalenza {y[idx].mean():.3f}")
    print()

    pred = {k: np.full(n, np.nan) for k in FUSIONI}
    p_cnn_tutti = np.full(n, np.nan)
    per_fold = []
    started = time.time()

    for tag, (nome, fuori) in enumerate(esterni):
        dentro = np.setdiff1d(np.arange(n), fuori)
        print(f"[{tag + 1}/{len(esterni)}] fuori: {nome}  (n={len(fuori)}, "
              f"dentro {len(dentro)})", flush=True)

        sbr = sbr_fuori_campione(frame, columns, groups, dentro, fuori, args.seed)
        riga = {"fold": nome, "n": int(len(fuori)),
                "prevalenza": float(y[fuori].mean()),
                "peso_blend": sbr["peso"],
                **{f"sbr_{k}": v for k, v in metriche(y[fuori], sbr["p_out"]).items()}}
        pred["solo_sbr"][fuori] = sbr["p_out"]

        if arco is not None:
            cnn = cnn_fuori_campione(
                arco, torch, nn, F, vol, y, groups, dentro, fuori, cfg, seeds,
                args.fold_seed, tag, log=lambda s: print(s, flush=True),
                dev=args.dispositivo)
            p_cnn_tutti[fuori] = cnn["p_out"]
            a_hat, b_hat = stima_fusione(y[dentro], sbr["oof_dentro"], cnn["oof_dentro"])
            riga.update({
                "a_stimato": a_hat, "b_stimato": b_hat, "n_reti": cnn["n_reti"],
                **{f"cnn_{k}_out": v for k, v in metriche(y[fuori], cnn["p_out"]).items()},
            })
            for chiave, regola in FUSIONI.items():
                if regola is None:
                    continue
                a, b = (a_hat, b_hat) if regola == "stima" else regola
                pred[chiave][fuori] = fondi(sbr["p_out"], cnn["p_out"], a, b)
                riga[f"ll::{chiave}"] = metriche(y[fuori], pred[chiave][fuori])["log_loss"]
            print(f"      -> CNN fuori: AUROC {riga['cnn_auroc_out']:.4f} | "
                  f"SBR {riga['sbr_auroc']:.4f} | (a,b) stimati "
                  f"({a_hat:.3f}, {b_hat:.3f})", flush=True)

        per_fold.append(riga)
        print(f"      -> SBR fuori: log loss {riga['sbr_log_loss']:.4f}, "
              f"{(time.time() - started) / 60:.1f} min\n", flush=True)

    complessivo = {}
    for chiave, colonna in pred.items():
        if np.isnan(colonna).all():
            continue
        assert not np.isnan(colonna).any(), chiave
        complessivo[chiave] = metriche(y, colonna)

    riferimento = complessivo["solo_sbr"]["log_loss"]
    for chiave, m in complessivo.items():
        m["guadagno_su_sbr"] = riferimento - m["log_loss"]

    minuti = (time.time() - started) / 60
    larghezza = 78
    print("=" * larghezza)
    print(f"METRO ESTERNO '{args.esterni}' — {n} esami, ogni esame predetto da una "
          f"catena\nche non ha mai visto il suo {'centro' if args.esterni == 'gruppi' else 'strato'}."
          f"  {minuti:.1f} min")
    print("=" * larghezza)
    for chiave, m in complessivo.items():
        print(f"  {chiave:26s} log loss {m['log_loss']:.4f}   AUROC {m['auroc']:.4f}"
              f"   guadagno {m['guadagno_su_sbr']:+.4f}")
    print("=" * larghezza)

    if arco is not None:
        print("\nAUROC della CNN sul centro tenuto fuori — quanto generalizza il ramo:")
        for r in per_fold:
            print(f"  {r['fold']:28s} n={r['n']:4d}  CNN {r['cnn_auroc_out']:.4f}   "
                  f"SBR {r['sbr_auroc']:.4f}   b stimato {r['b_stimato']:.3f}")

    chiave_spedita = "stimata_nel_fold" if arco is not None else "solo_sbr"
    stima = complessivo[chiave_spedita]["log_loss"]
    print(f"\nQuesto lato del metro: {stima:.4f}   (gara {args.gara:.4f} "
          f"± {args.errore_gara:.4f})")
    print("Per il verdetto servono entrambe le partizioni: esegui anche "
          f"--esterni {'strati' if args.esterni == 'gruppi' else 'gruppi'}.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    # Il numero di strati entra nel nome: una corsa a 10 fold misura un'altra
    # cosa (la curva di apprendimento) e sovrascrivere con essa il limite
    # ottimistico a 5 fold lo distruggerebbe senza dirlo.
    suffisso = f"_{args.n_strati}fold" if args.n_strati else ""
    base = args.outdir / f"metro_esterno_{args.esterni}{suffisso}"
    # Lo stesso ragionamento vale per la configurazione della CNN, e il 3 agosto
    # e' costato quasi il reperto piu' importante del progetto: la corsa con
    # `--norm istanza` ha scritto qui sopra il risultato di quella con `batch`,
    # e solo una copia fatta a mano ha salvato il confronto. Ottanta minuti di
    # GPU non possono dipendere da chi si ricorda di copiare un file, quindi
    # ogni corsa con la CNN lascia **anche** una copia che nomina la propria
    # configurazione, e che nessuna corsa con una configurazione diversa puo'
    # sovrascrivere. Il nome generico resta per non rompere chi lo legge.
    etichetta = "" if arco is None else etichetta_configurazione(args)
    np.savez(base.with_suffix(".npz"), p_cnn=p_cnn_tutti,
             **{k: v for k, v in pred.items() if not np.isnan(v).all()})
    base.with_suffix(".json").write_text(json.dumps({
        "partizione": args.esterni, "min_n": args.min_n, "n": n,
        "config_cnn": None if arco is None else cfg,
        "train_seeds": seeds, "fold_seed": args.fold_seed, "minuti": minuti,
        "gara": {"log_loss": args.gara, "auroc": args.gara_auroc,
                 "errore": args.errore_gara},
        "complessivo": complessivo, "per_fold": per_fold,
    }, indent=2), encoding="utf-8")
    print(f"\nScritto {base.with_suffix('.json')} e {base.with_suffix('.npz')}")

    if etichetta:
        import shutil  # noqa: PLC0415

        copia = args.outdir / f"{base.name}{etichetta}"
        shutil.copy2(base.with_suffix(".json"), copia.with_suffix(".json"))
        shutil.copy2(base.with_suffix(".npz"), copia.with_suffix(".npz"))
        print(f"Copia che nessun'altra configurazione sovrascrive: "
              f"{copia.with_suffix('.json')}")

    altro = args.outdir / (
        f"metro_esterno_{'strati' if args.esterni == 'gruppi' else 'gruppi'}.json")
    if altro.exists():
        dati = json.loads(altro.read_text(encoding="utf-8"))
        opp = dati["complessivo"].get(chiave_spedita)
        # Il default silenzioso e' costato 31 minuti di GPU gia' spesi: con
        # `.get(chiave, {})` un'altra partizione NON confrontabile — perche'
        # girata `--senza-cnn`, quindi senza la chiave della fusione — passava
        # il controllo e faceva esplodere la riga dopo, con un KeyError che non
        # nomina il problema. Il confronto o e' possibile o si dice perche' no.
        if opp is None:
            print("\n" + "=" * larghezza)
            print(f"Nessun verdetto: {altro.name} non ha '{chiave_spedita}'.")
            print(f"  ha: {', '.join(sorted(dati['complessivo']))}")
            print("  Le due partizioni vanno girate con la STESSA configurazione: "
                  "una corsa\n  --senza-cnn non e' confrontabile con una che la CNN "
                  "ce l'ha.")
            print("=" * larghezza)
            return 0
        qui = complessivo[chiave_spedita]
        pess, ott = ((qui, opp) if args.esterni == "gruppi" else (opp, qui))

        def quota(chiave: str, gara: float) -> float:
            """Dove cade la gara fra i due limiti: 0 = come un centro mai visto.

            E' la quantita' che il metro serve a produrre, e non e' un dettaglio
            di presentazione: dice **in quale regime vive il test set**. Vicino a
            zero, la generalizzazione fra centri e' l'intero problema e ogni
            scorciatoia condizionata al centro e' inutile. Vicino a uno, il test
            e' in dominio e la CV raggruppata sta sprecando informazione.
            """
            a, b = pess[chiave], ott[chiave]
            return float("nan") if a == b else (gara - a) / (b - a)

        print("\n" + "=" * larghezza)
        print(f"{'':29s}{'pessimistico':>14s}{'gara':>10s}{'ottimistico':>14s}"
              f"{'quota':>9s}")
        print(f"  {'log loss':27s}{pess['log_loss']:>14.4f}{args.gara:>10.4f}"
              f"{ott['log_loss']:>14.4f}{quota('log_loss', args.gara):>9.0%}")
        print(f"  {'AUROC (isola la separazione)':27s}{pess['auroc']:>14.4f}"
              f"{args.gara_auroc:>10.4f}{ott['auroc']:>14.4f}"
              f"{quota('auroc', args.gara_auroc):>9.0%}")
        print(f"\n  -> {verdetto(pess['log_loss'], ott['log_loss'], args.gara, args.errore_gara)}")
        print("=" * larghezza)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
