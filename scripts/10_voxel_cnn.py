#!/usr/bin/env python3
"""Allena e misura il ramo a immagini che finisce nella submission.

La configurazione attiva e' `premip`; ogni corsa produce predizioni OOF e puo'
scrivere il checkpoint soltanto quando `--salva` e' esplicito. La valutazione
resta appaiata al ramo tabellare e separata per gruppi scanner.

**1. Il rumore dell'addestramento si misura, non si assume.** Ogni arco gira su
piu' seed di addestramento a fold FISSI. La dispersione fra quei seed e' il
pavimento di rumore del ramo C, ed e' una quantita' diversa - e finora ignota -
dal pavimento di 0,0021 fra seed dei fold del modello SBR. Senza quel numero
nessun confronto qui dentro e' leggibile (R7: un cancello si specifica contro
l'incumbent, e per farlo serve sapere quanto vale il rumore).

**2. La media sui seed non e' solo una misura, e' il modello.** K reti mediate
sono un unico predittore, meglio calibrato di ciascuna: e' la forma in cui il
ramo C verrebbe spedito, quindi e' la forma in cui va valutato.

**3. Ogni arco e' appaiato con tutti gli altri per costruzione.** Stessi fold
(quelli di `make_folds` con lo stesso seed del modello SBR), stessa zona grigia
presa da `models/oof.npy`, stesso SBR con cui misurare il blend. La differenza
fra due archi non porta dentro la varianza del sorteggio dei fold.

Il numero che decide
--------------------
Non l'AUC dentro la zona grigia - un livello, e la R7 dice che la domanda e' un
contrasto - ma **il guadagno di log loss del blend con peso scelto in CV
annidata, contro il modello SBR da solo**. Il verdetto dell'incumbent e le
direzioni chiuse sono in `docs/DECISIONS.md`.

Cosa NON fa
-----------
Non tocca `src/`, non scrive in `models/`, non ri-estrae i crop. Un arco che
richiedesse crop diversi non e' un arco: e' un altro dataset, e va estratto con
`08_extract_crops.py`.

Uso (dentro l'immagine ufficiale, che ha torch e la GPU):
    python scripts/10_voxel_cnn.py --name premip --arch premip \
        --preaddestrato models/torchhub --train-seeds 0-2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusione import tabella, verdetto  # noqa: E402
from src.cv import make_folds  # noqa: E402
from src.model import TARGET, find_best_blend  # noqa: E402
from src.voxnet import ARCHITETTURE, ARCHITETTURE_25D  # noqa: E402
from src.voxnet import build_model as _build_condiviso  # noqa: E402
from src.voxnet import autonormalizza, carica_imagenet, stima_fusione  # noqa: E402

GREY_LOW, GREY_HIGH = 0.2, 0.8
CLIP_LO, CLIP_HI = 0.005, 0.995
LATO = 64          # lato del cubo dato alla rete, in voxel da 2 mm


# --------------------------------------------------------------------------- #
# Modello
# --------------------------------------------------------------------------- #

def build_model(torch, nn, arch: str, in_ch: int, dropout: float,
                norm: str = "batch"):
    """Delega a `src/voxnet.py`, che e' l'unica definizione della rete.

    La firma prende ancora `torch` e `nn` per non toccare i richiami interni,
    ma non li usa: la rete la costruisce il modulo che **viaggia dentro
    l'archivio di submission**. Un `state_dict` e' solo un dizionario di
    tensori e non porta con se' l'architettura, quindi due definizioni che
    divergono produrrebbero, nel caso fortunato, un errore di caricamento, e nel
    caso sfortunato predizioni sbagliate senza alcun errore.
    """
    return _build_condiviso(arch, in_ch, dropout, norm)


# --------------------------------------------------------------------------- #
# Augmentation, tutta su GPU
# --------------------------------------------------------------------------- #

def _flip_lr(torch, x, mirror_channel: bool):
    """Ribaltamento destra-sinistra. Con il canale speculare, scambia i canali.

    Il ribaltamento e' legittimo e raddoppia i dati: il progetto non da' mai al
    modello i lati separati, perche' *quale* emisfero sia il piu' colpito non ha
    significato diagnostico - conta l'entita' dell'asimmetria, non il verso. Una
    rete che imparasse una lateralita' preferenziale starebbe imparando rumore.

    Con `--mirror-channel` l'ingresso e' [volume, volume speculare]: ribaltare
    lo spazio senza scambiare i due canali produrrebbe una coppia incoerente,
    cioe' un esempio che non esiste in natura. Lo scambio e' obbligatorio, non
    un'opzione.
    """
    out = torch.flip(x, dims=[2])            # dim 2 = asse destra-sinistra
    if mirror_channel:
        out = torch.flip(out, dims=[1])      # dim 1 = canale: 0 <-> 1
    return out


def _affine(torch, F, x, gen, gradi: float, scala: float, shift_vox: float):
    """Rototraslazione rigida piccola, in batch, con riempimento a zero.

    Il volume e' cubico e isotropo a 2 mm, quindi le coordinate normalizzate di
    `affine_grid` sono isometriche: una matrice di rotazione costruita in quello
    spazio **e'** una rotazione fisica, a meno di una permutazione degli assi che
    per una rotazione casuale isotropa non cambia nulla. E' il motivo per cui qui
    non compare nessuna conversione fra sistemi di riferimento - sarebbe la
    classica riga giusta per caso.

    Le ampiezze sono quelle dell'esperimento sul jitter (+/-3 mm, +/-5 gradi),
    che ha misurato quanto la pipeline SBR oscilla sotto perturbazioni di quella
    taglia. Qui pero' non servono a *misurare* il rumore di localizzazione: la
    rete non ha un'ancora da spostare. Servono a insegnarle che la stessa testa
    posata diversamente nel gantry e' lo stesso paziente.
    """
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, 3, generator=gen, device=dev) * 2 - 1) * math.radians(gradi)
    ca, sa = torch.cos(ang), torch.sin(ang)
    zero = torch.zeros(b, device=dev)
    one = torch.ones(b, device=dev)

    def stack3(rows):
        return torch.stack([torch.stack(r, dim=1) for r in rows], dim=1)

    rx = stack3([[one, zero, zero],
                 [zero, ca[:, 0], -sa[:, 0]],
                 [zero, sa[:, 0], ca[:, 0]]])
    ry = stack3([[ca[:, 1], zero, sa[:, 1]],
                 [zero, one, zero],
                 [-sa[:, 1], zero, ca[:, 1]]])
    rz = stack3([[ca[:, 2], -sa[:, 2], zero],
                 [sa[:, 2], ca[:, 2], zero],
                 [zero, zero, one]])
    rot = rx @ ry @ rz

    if scala > 0:
        s = 1.0 + (torch.rand(b, 1, 1, generator=gen, device=dev) * 2 - 1) * scala
        rot = rot / s

    # --- correzione per volumi NON cubici ------------------------------------
    #
    # `affine_grid` lavora in coordinate normalizzate, dove ogni asse va da -1 a
    # +1 **indipendentemente da quanti voxel ha**. Su un cubo e' innocuo; su una
    # scatola 48x34x33 non lo e': la stessa matrice di rotazione produrrebbe
    # angoli fisici diversi asse per asse, e una traslazione espressa in frazione
    # di `shape[-1]` varrebbe voxel diversi sui tre assi.
    #
    # La conversione e' una coniugazione: un punto a p voxel dal centro sta a
    # u = 2p/D in coordinate normalizzate, quindi una rotazione fisica R diventa
    # diag(1/D) R diag(D). L'ordine degli assi e' quello di `grid_sample`, che
    # per un tensore (B, C, d0, d1, d2) e' (d2, d1, d0) — invertito rispetto alle
    # dimensioni del tensore, ed e' la trappola classica di questa funzione.
    #
    # Su un volume cubico D e' costante e la coniugazione e' l'identita': questo
    # blocco **non cambia nessun numero gia' misurato**, ed e' il motivo per cui
    # si puo' aggiungere senza rimettere in discussione gli archi passati.
    dim = torch.tensor([x.shape[-1], x.shape[-2], x.shape[-3]],
                       dtype=rot.dtype, device=dev)
    if not bool(torch.all(dim == dim[0])):
        rot = torch.diag(1.0 / dim) @ rot @ torch.diag(dim)

    # La traslazione e' in coordinate normalizzate: 2 unita' = tutto il lato,
    # quindi `shift_vox` voxel valgono 2*shift_vox/D su ciascun asse.
    tra = (torch.rand(b, 3, generator=gen, device=dev) * 2 - 1) * (2.0 * shift_vox / dim)
    theta = torch.cat([rot, tra[:, :, None]], dim=2)

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


def _intensita(torch, x, gen, gamma_max: float, guadagno_max: float):
    """Guadagno moltiplicativo e gamma, per esame.

    Il crop e' gia' normalizzato sul riferimento occipitale del paziente, quindi
    un guadagno globale non dovrebbe esistere. **E' esattamente per questo che
    va simulato**: cio' che varia fra centri non e' il guadagno vero ma
    l'*errore residuo* di quella normalizzazione — il riferimento occipitale e'
    esso stesso una misura, e la sua distorsione dipende da FOV, ricostruzione e
    diffusione. Il gamma aggiunge la parte non lineare: due ricostruzioni
    diverse non comprimono la scala allo stesso modo agli estremi.
    """
    b = x.shape[0]
    forma = (b, 1, 1, 1, 1)
    if guadagno_max > 0:
        g = 1.0 + (torch.rand(forma, generator=gen, device=x.device) * 2 - 1) * guadagno_max
        x = x * g
    if gamma_max > 0:
        e = 1.0 + (torch.rand(forma, generator=gen, device=x.device) * 2 - 1) * gamma_max
        x = x.clamp_min(0.0) ** e
    return x


def _rumore(torch, x, gen, frazione: float):
    """Rumore con varianza proporzionale al segnale, cioe' di forma poissoniana.

    Una SPECT conta fotoni: la deviazione standard di un voxel va come la radice
    del suo valore, non come una costante. Un rumore gaussiano omogeneo
    insegnerebbe alla rete l'invarianza sbagliata — robustezza dove non serve
    (nel fondo) e nessuna dove serve (nello striato, dove i conteggi sono alti e
    la differenza fra due pazienti e' piccola in proporzione).

    L'ampiezza assoluta e' arbitraria perche' il crop e' un rapporto e i
    conteggi originali non sono piu' recuperabili: cio' che si riproduce e' la
    **forma** della dipendenza, non il livello. E' un'augmentation, non una
    simulazione.
    """
    if frazione <= 0:
        return x
    b = x.shape[0]
    amp = torch.rand((b, 1, 1, 1, 1), generator=gen, device=x.device) * frazione
    e = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return x + amp * torch.sqrt(x.clamp_min(0.0)) * e


def _sfoca(torch, F, x, gen, fwhm_max_mm: float, mm_per_voxel: float = 2.0):
    """Sfocatura gaussiana separabile, con FWHM sorteggiata per esame.

    **E' l'augmentation che manca e che il fallimento della submission 2 indica
    per prima.** La differenza fisica principale fra i 43 scanner del dataset non
    e' la posa del paziente: e' la risoluzione effettiva, fra ~8 e ~10 mm FWHM,
    piu' il filtro di ricostruzione. Una rete addestrata su un solo livello di
    sfocatura impara la nitidezza del centro che l'ha vista, e quella e'
    un'informazione sul centro travestita da informazione sul paziente.

    Si puo' solo sfocare, mai affilare: l'arco allarga l'insieme verso gli
    scanner piu' morbidi, non verso quelli piu' nitidi. E' un'asimmetria reale,
    non un limite dell'implementazione.

    Il nucleo e' diverso per esame — `groups=B*C` in una convoluzione sola per
    asse — e non per batch: un batch da 16 con un solo valore di FWHM darebbe
    alla rete 16 copie della stessa condizione, e la diversita' arriverebbe
    solo fra batch, cioe' molto piu' lentamente.
    """
    if fwhm_max_mm <= 0:
        return x
    b, c = x.shape[0], x.shape[1]
    fwhm = torch.rand(b, generator=gen, device=x.device) * fwhm_max_mm
    sigma = (fwhm / 2.3548 / mm_per_voxel).clamp_min(1e-6)
    raggio = max(1, int(math.ceil(3.0 * float(sigma.max().item()))))

    d = torch.arange(-raggio, raggio + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-(d[None, :] ** 2) / (2 * sigma[:, None] ** 2))
    k = (k / k.sum(dim=1, keepdim=True)).to(x.dtype)

    out = x
    for asse in range(3):
        forma = [1, 1, 1]
        forma[asse] = 2 * raggio + 1
        w = k.view(b, 1, *forma).repeat_interleave(c, dim=0)
        pad = [0, 0, 0]
        pad[asse] = raggio
        out = F.conv3d(out.reshape(1, b * c, *out.shape[2:]), w,
                       groups=b * c, padding=tuple(pad))
        out = out.reshape(b, c, *out.shape[2:])
    return out


def riduci(F, x, fattore: int):
    """Sottocampiona il volume con una media, se l'arco lo chiede.

    2 mm di campionamento su una SPECT sono un sovracampionamento di fatto: la
    risoluzione effettiva di una gamma camera e' 8-10 mm FWHM, quindi a 4 mm si
    e' ancora ben dentro Nyquist e non si butta via informazione fisica — si
    butta via rumore di ricostruzione e sette ottavi del calcolo. Resta un arco
    e non un default perche' "non si butta via nulla" e' un argomento, e gli
    argomenti in questo progetto si misurano.
    """
    return x if fattore <= 1 else F.avg_pool3d(x, fattore)


def augment(torch, F, x, gen, cfg):
    """Applica l'augmentation dell'arco. `x` e' (B, C, 64, 64, 64) su GPU.

    L'ordine non e' libero, segue la catena fisica: il paziente sta nel gantry
    in un certo modo (flip, affine), il segnale esce con un guadagno e una
    scala di ricostruzione (intensita'), i fotoni arrivano contati (rumore), e
    la camera piu' il filtro di ricostruzione sfocano tutto (sfoca). Mettere la
    sfocatura prima del rumore darebbe rumore bianco su un'immagine morbida, che
    non e' cosa che una SPECT ricostruita produca: il rumore reale e' correlato
    proprio sulla scala della risoluzione.

    Le quattro voci di intensita' sono a zero per difetto. Non sono un default
    mancato: valgono quanto vale la loro misura, e finche' un arco appaiato non
    l'ha fatta restano un'opzione dichiarata.
    """
    if cfg["flip"]:
        scelti = torch.rand(x.shape[0], generator=gen, device=x.device) < 0.5
        if scelti.any():
            x = x.clone()
            x[scelti] = _flip_lr(torch, x[scelti], cfg["mirror_channel"])
    if cfg["gradi"] > 0 or cfg["scala"] > 0 or cfg["shift"] > 0:
        x = _affine(torch, F, x, gen, cfg["gradi"], cfg["scala"], cfg["shift"])
    if cfg.get("gamma", 0.0) > 0 or cfg.get("guadagno", 0.0) > 0:
        x = _intensita(torch, x, gen, cfg.get("gamma", 0.0), cfg.get("guadagno", 0.0))
    x = _rumore(torch, x, gen, cfg.get("rumore", 0.0))
    x = _sfoca(torch, F, x, gen, cfg.get("sfoca", 0.0))
    return x


# --------------------------------------------------------------------------- #
# Addestramento di un fold
# --------------------------------------------------------------------------- #

def train_one_fold(torch, nn, F, volumes, y, fit_idx, stop_idx, pred_idx, cfg, seed):
    """Addestra su `fit_idx`, arresta su `stop_idx`, predice su `pred_idx`.

    `stop_idx` non appartiene mai al fold di validation e ha i gruppi tenuti
    separati da `fit_idx`: se lo stesso scanner comparisse in entrambe le parti,
    il criterio di arresto misurerebbe la memorizzazione del centro invece della
    generalizzazione, che e' esattamente il leak eliminato allo Step 6 del piano
    di lavoro per LightGBM.
    """
    dev = volumes.device
    radice = int(seed) * 9973 + 17
    gen = torch.Generator(device=dev).manual_seed(radice)
    torch.manual_seed(radice)

    model = build_model(torch, nn, cfg["arch"], volumes.shape[1],
                        cfg["dropout"], cfg.get("norm", "batch")).to(dev)
    if cfg["arch"] in ARCHITETTURE_25D:
        # `channels_last_3d` e' un layout per tensori a 5 assi: i pesi Conv2d ne
        # hanno 4 e non lo accettano. E l'inizializzazione ImageNet si mette qui,
        # nell'unico posto che non finisce nell'archivio di submission.
        if cfg.get("preaddestrato"):
            n = carica_imagenet(model, cfg["preaddestrato"])
            if seed == 0:
                print(f"      pesi ImageNet caricati ({n/1e6:.1f}M parametri)")
    else:
        model = model.to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    lossf = nn.BCEWithLogitsLoss()

    sched = None
    if cfg["cosine"]:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    yt = torch.from_numpy(y).to(dev)
    fit_t = torch.from_numpy(fit_idx).to(dev)
    n_fit = len(fit_idx)
    def predici(idx, tta: bool):
        model.eval()
        out = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=cfg["amp"]):
            for k in range(0, len(idx), 64):
                sel = torch.from_numpy(idx[k:k + 64]).to(dev)
                xb = riduci(F, volumes[sel].float(), cfg["riduci"])
                xb = xb.to(memory_format=torch.channels_last_3d)
                logit = model(xb).float().ravel()
                if tta:
                    xf = _flip_lr(torch, xb, cfg["mirror_channel"])
                    logit = (logit + model(xf).float().ravel()) / 2
                out.append(logit)
        return torch.cat(out).cpu().numpy()

    # Senza arresto anticipato il criterio di fine e' il calendario, non i dati:
    # si addestra per un numero di epoche fissato PRIMA, e in cambio il fold di
    # training viene usato intero invece che all'80%. Il numero di epoche non
    # puo' venire dal fold di validation — verrebbe scelto guardando le etichette
    # da predire, che e' il leak eliminato allo Step 6 — e infatti viene dalla
    # mediana delle epoche a cui l'arco con arresto si e' fermato sulla porzione
    # interna. E' una scelta tarata su dati che il fold di validation non vede.
    senza_arresto = not cfg["early_stop"]
    best, best_state, since, epoch, best_epoch = float("inf"), None, 0, 0, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n_fit, generator=gen, device=dev)
        for k in range(0, n_fit, cfg["batch"]):
            sel = fit_t[perm[k:k + cfg["batch"]]]
            xb = volumes[sel].float()
            yb = yt[sel][:, None]
            xb = riduci(F, augment(torch, F, xb, gen, cfg), cfg["riduci"])

            if cfg["mixup"] > 0:
                lam = float(np.random.default_rng(int(gen.initial_seed()) + epoch * 1000 + k)
                            .beta(cfg["mixup"], cfg["mixup"]))
                lam = max(lam, 1 - lam)
                rp = torch.randperm(xb.shape[0], generator=gen, device=dev)
                xb = lam * xb + (1 - lam) * xb[rp]
                yb = lam * yb + (1 - lam) * yb[rp]

            if cfg["smooth"] > 0:
                yb = yb * (1 - cfg["smooth"]) + 0.5 * cfg["smooth"]

            xb = xb.to(memory_format=torch.channels_last_3d)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if sched is not None:
            sched.step()

        if senza_arresto:
            continue
        p = 1 / (1 + np.exp(-predici(stop_idx, cfg["tta"])))
        stop_loss = float(-np.mean(y[stop_idx] * np.log(np.clip(p, 1e-6, 1))
                                   + (1 - y[stop_idx]) * np.log(np.clip(1 - p, 1e-6, 1))))
        if stop_loss < best - 1e-4:
            best, since, best_epoch = stop_loss, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if senza_arresto:
        best_epoch = epoch + 1
    return (1 / (1 + np.exp(-predici(pred_idx, cfg["tta"]))), best, epoch + 1,
            best_epoch, model.state_dict())


# --------------------------------------------------------------------------- #
# Lettura del risultato
# --------------------------------------------------------------------------- #

def analizza(y, oof_cnn, oof_sbr, grey, groups=None) -> dict:
    """Tutte le quantita' che decidono, in un posto solo.

    Il blend usa `find_best_blend` di `src/model.py`, cioe' **la stessa funzione
    che sceglie il peso fra LightGBM e logistica nel modello spedito**: il peso
    riportato e' quello per l'inferenza, il punteggio riportato e' quello
    annidato, che non ha mai visto le etichette con cui e' valutato. Riscriverne
    una copia qui avrebbe prodotto due definizioni di "onesto" divergenti in
    silenzio, che e' la classe di difetto contro cui `src/model.py` esiste.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import log_loss, roc_auc_score

    def ll(p):
        return float(log_loss(y, np.clip(p, CLIP_LO, CLIP_HI)))

    peso, annidata = find_best_blend(y, oof_cnn, oof_sbr)
    blend = np.clip(peso * oof_cnn + (1 - peso) * oof_sbr, CLIP_LO, CLIP_HI)
    sbr_ll = ll(oof_sbr)

    return {
        "cnn_log_loss": ll(oof_cnn),
        "cnn_auroc": float(roc_auc_score(y, oof_cnn)),
        "cnn_auc_grigia": float(roc_auc_score(y[grey], oof_cnn[grey])),
        "sbr_log_loss": sbr_ll,
        "sbr_auroc": float(roc_auc_score(y, oof_sbr)),
        "sbr_auc_grigia": float(roc_auc_score(y[grey], oof_sbr[grey])),
        "blend_peso": peso,
        "blend_log_loss_in_sample": ll(blend),
        "blend_log_loss_annidato": annidata,
        # Il numero del cancello: positivo = guadagno.
        "guadagno_annidato": sbr_ll - annidata,
        "guadagno_in_sample": sbr_ll - ll(blend),
        "blend_auroc": float(roc_auc_score(y, blend)),
        "blend_auc_grigia": float(roc_auc_score(y[grey], blend[grey])),
        "rho_grigia": float(spearmanr(oof_cnn[grey], oof_sbr[grey]).statistic),
        "rho_totale": float(spearmanr(oof_cnn, oof_sbr).statistic),
        # Il verdetto vero, dal modulo condiviso con `12_fusione.py`. Il peso
        # scalare qui sopra e' il numero su cui il piano ha fissato il cancello
        # di 0,008 e resta riportato; ma quel cancello e' stato tarato prima di
        # sapere che una fusione a un parametro in probabilita' butta via due
        # terzi del contributo, e che una con intercetta non sopravvive a un
        # centro mai visto. Vedi `scripts/fusione.py`.
        **({} if groups is None else
           {f"fus_{k}": v for k, v in
            verdetto(tabella(y, oof_sbr, oof_cnn, groups, grey)).items()}),
    }


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="nome dell'arco; decide i file scritti")
    p.add_argument("--crops", type=Path, default=Path("crops"))
    p.add_argument("--z-offsets", type=Path, default=None,
                   help="npy con il centro della finestra z per esame, nel sistema del "
                        "crop alto. Senza, si prende la finestra centrale — che su "
                        "crops_z96 riproduce `crops/` fetta per fetta.")
    p.add_argument("--features", type=Path, default=Path("features_v4.csv"))
    p.add_argument("--oof", type=Path, default=Path("models/oof.npy"))
    p.add_argument("--fold-seed", type=int, default=0,
                   help="seed dei fold; DEVE essere quello con cui e' stata prodotta --oof")
    p.add_argument("--train-seeds", default="0", help="es. '0-4'")
    p.add_argument("--arch", choices=sorted(ARCHITETTURE) + sorted(ARCHITETTURE_25D),
                   required=True)
    p.add_argument("--preaddestrato", type=Path, default=None,
                   help="radice in cui cercare i pesi ImageNet (es. models/torchhub) "
                        "per gli archi 2,5D. Serve SOLO in addestramento: "
                        "nell'archivio viaggia il state_dict messo a punto, e "
                        "`build_model` non costruisce mai con `weights=`")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--autonorm", action="store_true",
                   help="divide ogni crop per la propria norma, togliendo la "
                        "regione di normalizzazione occipitale (Zhou & Tagare "
                        "2021). Vedi src/voxnet.py:autonormalizza")
    p.add_argument("--norm", choices=("batch", "istanza", "gruppo"), default="batch",
                   help="`batch` accumula statistiche sul dataset, dominato al 39%% "
                        "da un solo centro: e' un canale di fuga per l'identita' "
                        "dello scanner. `istanza` e `gruppo` normalizzano dentro "
                        "il singolo esame")
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--riduci", type=int, default=1,
                   help="fattore di sottocampionamento in ingresso (2 = voxel da 4 mm)")
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                   help="niente arresto anticipato: epoche fisse e fold di training intero")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--gradi", type=float, default=0.0, help="rotazione max in gradi")
    p.add_argument("--scala", type=float, default=0.0, help="jitter di scala, es. 0.08")
    p.add_argument("--shift", type=float, default=4.0, help="traslazione max in voxel")
    # Le quattro voci fisiche: simulano cio' che distingue davvero due scanner —
    # risoluzione, conteggi, e l'errore residuo della normalizzazione — invece
    # di come il paziente e' posato. Zero per difetto: vanno misurate.
    p.add_argument("--sfoca", type=float, default=0.0,
                   help="FWHM max della sfocatura aggiunta, in mm (es. 5.0)")
    p.add_argument("--rumore", type=float, default=0.0,
                   help="ampiezza max del rumore di forma poissoniana (es. 0.05)")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="jitter dell'esponente di intensita' (es. 0.10)")
    p.add_argument("--guadagno", type=float, default=0.0,
                   help="jitter del guadagno moltiplicativo (es. 0.10)")
    p.add_argument("--no-flip", dest="flip", action="store_false")
    p.add_argument("--mirror-channel", action="store_true",
                   help="ingresso a 2 canali: volume e suo speculare")
    p.add_argument("--tta", action="store_true", help="media col ribaltamento in predizione")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--outdir", type=Path, default=Path("experiments/risultati"))
    p.add_argument("--salva", type=Path, default=None,
                   help="scrive i pesi spedibili e i coefficienti di fusione "
                        "(es. models/voxnet.pt). Senza, l'arco e' solo una misura")
    args = p.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import GroupShuffleSplit

    if not torch.cuda.is_available():
        print("ERRORE: serve la GPU. Esegui dentro l'immagine ufficiale.", file=sys.stderr)
        return 1
    dev = "cuda"

    cfg = {
        # `autonorm` non e' un parametro dell'ottimizzatore: si applica ai
        # volumi prima dell'addestramento, ed era registrato solo nel
        # checkpoint. Il JSON dell'arco quindi non lo vedeva, e due archi che
        # differiscono SOLO per questo — `istanza_spedibile` e
        # `istanza_autonorm`, AUROC 0,9314 contro 0,9348 — avevano `config`
        # byte per byte identiche. Un record che non distingue due esperimenti
        # diversi non e' un record: da li' non si puo' riprodurre nessuno dei
        # due, e la scelta fra i due diventa irripetibile.
        "autonorm": bool(args.autonorm),
        "arch": args.arch, "norm": args.norm,
        "epochs": args.epochs, "patience": args.patience,
        "batch": args.batch, "lr": args.lr, "wd": args.wd, "dropout": args.dropout,
        "cosine": args.cosine, "mixup": args.mixup, "smooth": args.smooth,
        "gradi": args.gradi, "scala": args.scala, "shift": args.shift,
        "sfoca": args.sfoca, "rumore": args.rumore, "gamma": args.gamma,
        "guadagno": args.guadagno,
        "flip": args.flip, "mirror_channel": args.mirror_channel,
        "tta": args.tta, "amp": args.amp, "riduci": args.riduci,
        "early_stop": args.early_stop,
        "preaddestrato": str(args.preaddestrato) if args.preaddestrato else None,
    }
    if args.arch in ARCHITETTURE_25D and not args.preaddestrato:
        # Un arco 2,5D senza pesi ImageNet e' un ResNet inizializzato a caso:
        # gira, non alza, e misura una cosa diversa da quella che si crede di
        # misurare. Se lo si vuole davvero — ed e' il controllo giusto da fare
        # una volta — va chiesto esplicitamente.
        print("ATTENZIONE: arco 2,5D senza --preaddestrato: backbone CASUALE.",
              file=sys.stderr)
    seeds = parse_seeds(args.train_seeds)

    frame = pd.read_csv(args.features)
    y = frame[TARGET].to_numpy(np.float32)
    groups = frame["_group"].to_numpy()
    crops = np.load(args.crops / "crops.npy", mmap_mode="r")
    assert len(crops) == len(frame), (len(crops), len(frame))
    if crops.ndim != 4:
        raise SystemExit(f"forma crop non valida {crops.shape}: attesa (N,X,Y,Z)")

    # Finestra di 64 fette dentro un crop piu' alto in z. Il dataset alto esiste
    # perche' la ricentratura in z sia un affettamento invece di una
    # ri-estrazione: cosi' due varianti di allineamento sono confrontabili senza
    # che nulla ALTRO sia cambiato fra loro, che e' la definizione di appaiato.
    # La finestra si applica solo quando il crop e' PIU' ALTO del lato voluto.
    z_alto = crops.shape[3]
    if z_alto > LATO or args.z_offsets is not None:
        if args.z_offsets is None:
            centri = np.full(len(frame), z_alto // 2, dtype=int)
        else:
            centri = np.load(args.z_offsets).astype(int)
            assert len(centri) == len(frame)
        inizio = np.clip(centri - LATO // 2, 0, z_alto - LATO)
        fuori = int((np.abs(centri - LATO // 2 - inizio) > 0).sum())
        stretti = np.empty((len(frame), LATO, LATO, LATO), dtype=np.float16)
        for i in range(len(frame)):
            stretti[i] = crops[i][:, :, inizio[i]:inizio[i] + LATO]
        crops = stretti
        print(f"finestra z da {z_alto} a {LATO} fette, "
              f"{fuori} esami tosati al bordo del crop alto")

    if tuple(crops.shape[1:]) != (LATO, LATO, LATO):
        raise SystemExit(
            f"{args.crops/'crops.npy'} produce {tuple(crops.shape[1:])}, "
            f"atteso {(LATO, LATO, LATO)}")

    folds = make_folds(y.astype(int), groups, n_splits=5, seed=args.fold_seed)
    oof_sbr = np.load(args.oof)
    grey = (oof_sbr > GREY_LOW) & (oof_sbr < GREY_HIGH)

    # Tutto il dataset in VRAM: 1362 x 64^3 x 2 byte = 0,71 GB su 8. E' cio' che
    # rende praticabile un ensemble di seed - l'indicizzazione dei batch e
    # l'augmentation non toccano mai il bus PCIe, e un arco passa da ore a
    # minuti. Il ciclo di iterazione e' un requisito del piano (G4), non comodita'.
    vol = torch.from_numpy(np.array(crops, dtype=np.float16)).to(dev)
    if args.autonorm:
        # Sulla stessa funzione che gira in inferenza: se fosse riscritta qui,
        # il giorno in cui una delle due cambia i pesi resterebbero caricabili.
        vol = autonormalizza(vol)
    if args.mirror_channel:
        vol = torch.stack([vol, torch.flip(vol, dims=[1])], dim=1)
    else:
        vol = vol[:, None]

    n_par = sum(q.numel() for q in build_model(torch, nn, args.arch,
                                               vol.shape[1], args.dropout,
                                               args.norm).parameters())
    print(f"arco '{args.name}': {args.arch} ({n_par/1000:.0f}k parametri), "
          f"{vol.shape[1]} canali, seed di addestramento {seeds}")
    print(f"{len(frame)} esami, zona grigia {int(grey.sum())}, "
          f"fold seed {args.fold_seed}, VRAM dati {vol.numel()*2/1e9:.2f} GB\n")

    per_seed_oof = []
    per_seed_metriche = []
    pesi: list[dict] = []
    started = time.time()

    for s in seeds:
        oof = np.zeros(len(frame))
        for fold in sorted(np.unique(folds)):
            train_idx = np.flatnonzero(folds != fold)
            valid_idx = np.flatnonzero(folds == fold)
            if args.early_stop:
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                             random_state=args.fold_seed + fold + 1000 * s)
                fit_pos, stop_pos = next(splitter.split(train_idx, y[train_idx],
                                                        groups[train_idx]))
                fit_idx, stop_idx = train_idx[fit_pos], train_idx[stop_pos]
            else:
                fit_idx, stop_idx = train_idx, train_idx[:1]

            pred, interna, epoche, epoca_migliore, stato = train_one_fold(
                torch, nn, F, vol, y, fit_idx, stop_idx, valid_idx, cfg,
                seed=s * 100 + fold,
            )
            oof[valid_idx] = pred
            if args.salva is not None:
                # fp16: dimezza l'archivio e non tocca le predizioni in modo
                # misurabile — la rete gira comunque in fp32 dopo il
                # caricamento, e l'errore di arrotondamento dei pesi e' ordini
                # di grandezza sotto la dispersione fra seed (sd 0,0034).
                pesi.append({k: v.detach().half().cpu() for k, v in stato.items()})
            torch.cuda.empty_cache()
            print(f"  seed {s} fold {fold}: stop {epoche} epoche, minimo "
                  f"all'epoca {epoca_migliore}, interna {interna:.4f}, "
                  f"{(time.time()-started)/60:.1f} min", flush=True)

        # Per seed basta il quadro sintetico: la tabella completa delle fusioni
        # si calcola una volta sola, sull'ensemble, che e' il modello vero.
        m = analizza(y, oof, oof_sbr, grey)
        per_seed_oof.append(oof)
        per_seed_metriche.append(m)
        print(f"  -> seed {s}: CNN {m['cnn_log_loss']:.4f} / AUROC {m['cnn_auroc']:.4f} / "
              f"grigia {m['cnn_auc_grigia']:.4f} | guadagno annidato "
              f"{m['guadagno_annidato']:+.4f}\n", flush=True)

    # La media si fa in probabilita' e non in logit: e' la media bayesiana dei
    # modelli, ed e' la stessa forma in cui `main.py` gia' media i cinque
    # LightGBM e i trenta logistici in inferenza. Mediare i logit darebbe una
    # media geometrica delle odds, piu' confidente della media dei modelli —
    # cioe' proprio il tipo di sicurezza in piu' che la log loss punisce.
    ens = np.mean(per_seed_oof, axis=0)
    m_ens = analizza(y, ens, oof_sbr, grey, groups)

    # Il pavimento di rumore del ramo C, che nessuno aveva ancora misurato: la
    # dispersione fra seed di addestramento a fold FISSI. Ogni Delta letto in
    # questo ramo va confrontato con questo numero, non con lo 0,0021 del
    # modello SBR, che e' la dispersione fra seed dei FOLD - un'altra quantita'.
    disp = {}
    if len(seeds) > 1:
        for k in ("cnn_log_loss", "cnn_auroc", "cnn_auc_grigia", "guadagno_annidato"):
            v = np.array([q[k] for q in per_seed_metriche])
            disp[k] = {"media": float(v.mean()), "sd": float(v.std(ddof=1)),
                       "min": float(v.min()), "max": float(v.max())}

    minuti = (time.time() - started) / 60
    print("=" * 74)
    print(f"arco '{args.name}'  —  {len(seeds)} seed x 5 fold in {minuti:.1f} min")
    print("=" * 74)
    if disp:
        print("\nDispersione fra seed di addestramento (fold fissi) — il pavimento del ramo C:")
        for k, v in disp.items():
            print(f"  {k:22s} media {v['media']:+.4f}  sd {v['sd']:.4f}  "
                  f"[{v['min']:+.4f}, {v['max']:+.4f}]")
    print(f"\nEnsemble delle {len(seeds)} reti:")
    print(f"  CNN da sola        log loss {m_ens['cnn_log_loss']:.4f}   "
          f"AUROC {m_ens['cnn_auroc']:.4f}")
    print(f"  AUC zona grigia    CNN {m_ens['cnn_auc_grigia']:.4f}   "
          f"SBR {m_ens['sbr_auc_grigia']:.4f}   blend {m_ens['blend_auc_grigia']:.4f}")
    print(f"  rho di rango       grigia {m_ens['rho_grigia']:+.3f}   "
          f"totale {m_ens['rho_totale']:+.3f}")
    print(f"\n  SBR da solo        {m_ens['sbr_log_loss']:.4f}")
    print(f"  blend (peso {m_ens['blend_peso']:.2f})  annidato {m_ens['blend_log_loss_annidato']:.4f}"
          f"   in-sample {m_ens['blend_log_loss_in_sample']:.4f}")
    print("\n" + "=" * 74)
    print(f"  guadagno che DECIDE  (1 par., logit, raggruppato)  "
          f"{m_ens['fus_guadagno_deciso']:+.4f}")
    print(f"  controllo nullo, deve essere zero                  "
          f"{m_ens['fus_controllo_nullo']:+.4f}")
    print(f"  lo stesso, annidamento stratificato                "
          f"{m_ens['fus_guadagno_logit_stratificato']:+.4f}")
    print(f"  il numero del cancello del piano (peso, prob.)     "
          f"{m_ens['fus_guadagno_piano']:+.4f}   (cancello 0,0080)")
    print(f"  peso del modello nuovo, b                          "
          f"{m_ens['fus_b_raggruppato']:.2f}")
    print(f"  AUC nella zona grigia dopo la fusione              "
          f"{m_ens['fus_auc_grigia_fusa']:.4f}   (SBR da solo "
          f"{m_ens['sbr_auc_grigia']:.4f})")
    print("=" * 74)

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / f"voxel_{args.name}"
    np.save(base.with_suffix(".oof.npy"), ens)
    np.save(base.with_suffix(".oof_per_seed.npy"), np.array(per_seed_oof))
    base.with_suffix(".json").write_text(json.dumps({
        "arco": args.name, "config": cfg, "fold_seed": args.fold_seed,
        "train_seeds": seeds, "parametri": int(n_par), "minuti": minuti,
        "per_seed": per_seed_metriche, "dispersione": disp, "ensemble": m_ens,
    }, indent=2), encoding="utf-8")
    print(f"\nScritto {base.with_suffix('.json')}")

    if args.salva is not None:
        # I coefficienti di fusione si stimano su TUTTE le OOF: sono quelli che
        # servono a predire. Il punteggio riportato resta quello annidato di
        # `scripts/fusione.py`, che non ha mai visto le etichette con cui e'
        # valutato — la stessa distinzione che `find_best_blend` fa gia' per il
        # peso fra LightGBM e logistica.
        a, b = stima_fusione(y, oof_sbr, ens)
        args.salva.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": {
                "arch": args.arch, "norm": args.norm,
                "autonorm": bool(args.autonorm),
                "in_ch": int(vol.shape[1]),
                "dropout": args.dropout, "riduzione": args.riduci,
                "tta": bool(args.tta), "lato": LATO,
            },
            "state_dicts": pesi,
            "fusione": {"a": a, "b": b},
            "provenienza": {
                "arco": args.name, "fold_seed": args.fold_seed,
                "train_seeds": seeds, "n_reti": len(pesi),
                "guadagno_annidato_raggruppato_2p": m_ens.get("fus_guadagno_2p_raggruppato"),
                "guadagno_metro_del_piano": m_ens.get("fus_guadagno_piano"),
            },
        }, str(args.salva))
        # Sidecar leggibile senza torch. Il pacchettatore gira nel venv locale,
        # che torch non ce l'ha, e il cancello di robustezza (R10) ha bisogno
        # dei coefficienti: senza questo file l'unico modo di leggerli sarebbe
        # aprire il checkpoint, cioe' il controllo si salterebbe da solo
        # proprio nell'ambiente in cui si impacchetta. Non viaggia nell'archivio.
        sidecar = args.salva.with_suffix(".fusione.json")
        sidecar.write_text(json.dumps({
            "a": a, "b": b, "arco": args.name, "n_reti": len(pesi),
            "norm": args.norm, "arch": args.arch,
            "autonorm": bool(args.autonorm),
        }, indent=2), encoding="utf-8")
        mb = args.salva.stat().st_size / 1e6
        print(f"Scritti {len(pesi)} pesi in {args.salva} ({mb:.1f} MB), "
              f"fusione a={a:.3f} b={b:.3f}")
        print(f"Scritto {sidecar}")
        if args.riduci != 2 or not args.tta:
            print("ATTENZIONE: `src/voxnet.py` documenta riduzione 2 e TTA come la "
                  "configurazione misurata. Questo file ne salva un'altra.",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
