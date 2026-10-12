"""La rete a voxel: definizione condivisa fra addestramento e inferenza.

Perche' sta in `src/` e non nello script che la addestra
---------------------------------------------------------
Vale la stessa ragione di `pipeline.py` e `model.py`, con una posta piu' alta.
Un `state_dict` di PyTorch e' **solo un dizionario di tensori**: non contiene
l'architettura. Se la definizione della rete vivesse nello script di
addestramento e l'inferenza ne tenesse una copia, il giorno in cui una delle due
cambia larghezza, ordine dei blocchi o posizione della dropout, `load_state_dict`
fallirebbe — nel caso fortunato. Nel caso sfortunato **non** fallirebbe, perche'
le forme combaciano ancora, e il container produrrebbe predizioni sbagliate senza
un errore da nessuna parte. E' il modo di fallire peggiore che questo progetto
conosca, ed e' lo stesso del pickle che nomina `src.transforms`.

Qui c'e' quindi l'unica definizione, piu' il percorso di inferenza completo. Lo
script di addestramento la importa; `main.py` la importa; nessuno la riscrive.

Cosa la rete vede, e perche' quelle scelte
-------------------------------------------
Un crop di 64³ voxel a 2 mm centrato sul **baricentro della maschera cerebrale**
e normalizzato sul **riferimento occipitale del paziente stesso** — quindi ogni
voxel e' un rapporto interno al singolo esame, con la stessa immunita' al centro
di provenienza dell'SBR e la stessa conformita' alla regola sull'indipendenza
degli esami di test. Lo produce `pipeline.process_file(with_crop=True)`.

**Il volume viene poi sottocampionato a 4 mm** (`RIDUZIONE = 2`). Non e' un
compromesso: la risoluzione effettiva di una gamma camera e' 8-10 mm FWHM, quindi
a 4 mm si e' ancora ben dentro Nyquist. Misurato con un arco appaiato a cinque
seed: a 2 mm la rete da sola e' marginalmente migliore (AUROC 0,9291 contro
0,9249), la separazione dentro la zona grigia e' **peggiore**, il guadagno del
blend non si muove, e il costo e' **venti volte** tanto.

**Il ribaltamento destra-sinistra come test-time augmentation** e' legittimo e
gratuito: il progetto non da' mai al modello i lati separati, perche' *quale*
emisfero sia il piu' colpito non ha significato diagnostico — conta l'entita'
dell'asimmetria, non il verso.

La fusione con il modello SBR
------------------------------
`sigmoid(a·logit(p_sbr) + b·logit(p_cnn))`, senza intercetta. Due parametri e non
uno, e in logit e non in probabilita': entrambe le scelte sono misurate, e il
ragionamento sta in `scripts/fusione.py`. In sintesi — una media convessa in
probabilita' abbassa le predizioni confidenti e corrette, che sono il 70,6% degli
esami, quindi l'ottimizzatore sceglie un peso piccolo e la rete smette di muovere
anche la zona grigia; l'intercetta invece non sopravvive a un centro mai visto,
perche' la prevalenza differisce fra centri.
"""

from __future__ import annotations

import numpy as np

# Le larghezze dei quattro blocchi. `media` e' l'architettura scelta: `toy` e
# `wide` restano perche' un verdetto su cui non si puo' tornare non e' un
# verdetto, e perche' `wide` — piu' grande — e' misurabilmente peggiore.
ARCHITETTURE: dict[str, tuple[int, ...]] = {
    "toy": (16, 32, 64, 96),
    "media": (24, 48, 96, 144),
    "wide": (32, 64, 128, 192),
}

RIDUZIONE = 2          # sottocampionamento in ingresso: 64³ a 2 mm -> 32³ a 4 mm
CLIP_LO, CLIP_HI = 1e-6, 1 - 1e-6

# Il registro contiene soltanto l'arco MIP attivo. I parametri restano nel nome
# dell'architettura per ricostruire senza ambiguita' lo state_dict spedito.
ARCHITETTURE_25D: dict[str, dict] = {
    "premip": dict(profondita=3, lato=128),
}


def _costruisci_mip(profondita: int, lato: int, in_ch: int, dropout: float):
    """ResNet2D sulle tre MIP ortogonali.

    Perche' 2,5D e non 3D: i pesi 3D medici stanno dietro download manuali e
    licenze da verificare; quelli di ImageNet sono pubblici, gia' sul disco, e
    sono il caso di transfer piu' DEBOLE che esista — fotografie a colori contro
    fette di SPECT sfocata. Congelato, senza una sola epoca di adattamento,
    quel backbone arriva da solo a AUROC 0,8307 e aggiunge +0,0027 sopra la CNN
    3D con correlazione 0,63 (contro 0,91-0,998 fra le nostre varianti 3D).
    E' l'unica direzione in cui il progetto abbia trovato diversita' vera.

    Media **e** massimo, non solo media: il segnale striatale sta in poche fette
    su decine e una media le diluisce — lo stesso difetto che la rete 3D ha con
    `AdaptiveAvgPool3d`. Il massimo dice se ALMENO una fetta contiene la
    struttura, che e' proprio cio' che una media cancella.

    La standardizzazione e' **per immagine 2D**: nessuna statistica attraversa gli
    esami, quindi la regola di gara sull'indipendenza dei casi di test resta
    soddisfatta esattamente come per l'SBR.

    Nessun `weights=` qui dentro, mai: in gara il container non ha rete e un
    tentativo di download fallirebbe al primo esame e poi a tutti gli altri.
    L'inizializzazione ImageNet la mette `carica_imagenet`, che vive solo nel
    percorso di addestramento; nell'archivio viaggia il `state_dict` gia' messo
    a punto.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    if in_ch != 1:
        raise ValueError(f"l'arco 2,5D vuole un canale solo, non {in_ch} "
                         "(--mirror-channel non e' combinabile)")

    class Rete25D(nn.Module):
        def __init__(self):
            super().__init__()
            r = torchvision.models.resnet18(weights=None)      # MAI weights=...
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
            self.strati = nn.Sequential(
                *[r.layer1, r.layer2, r.layer3, r.layer4][:profondita])
            canali = [64, 128, 256, 512][profondita - 1]
            self.testa = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(3 * canali, 1))

        def forward(self, x):
            b = x.shape[0]
            v = x[:, 0]
            viste = [v.amax(dim=3), v.amax(dim=2), v.amax(dim=1)]
            pronte = []
            for vista in viste:
                vista = vista[:, None]
                if vista.shape[-2:] != (lato, lato):
                    vista = F.interpolate(vista, size=(lato, lato),
                                          mode="bilinear", align_corners=False)
                pronte.append(vista)
            x = torch.stack(pronte, dim=1).flatten(0, 1)
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            s = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            x = (x - m) / s
            x = x.repeat(1, 3, 1, 1)
            f = self.strati(self.stem(x))
            f = F.adaptive_avg_pool2d(f, 1).flatten(1).reshape(b, 3, -1)
            return self.testa(f.flatten(1))

    return Rete25D()


def carica_imagenet(modello, radice) -> int:
    """Mette i pesi ImageNet dentro un modello gia' costruito. Solo in addestramento.

    Non c'e' nessuna mappatura di chiavi scritta a mano: si costruisce un
    ResNet18 vuoto, ci si caricano i pesi con `strict=True` — se il file non
    fosse un ResNet18 esploderebbe qui — e poi si copiano i sottomoduli uno per
    uno, ancora con `strict=True`. Una mappatura scritta a mano potrebbe
    combaciare per forma e sbagliare per contenuto; questa o funziona o alza.
    """
    import torch
    import torchvision
    from pathlib import Path

    trovati = sorted(Path(radice).rglob("resnet18-*.pth"))
    if len(trovati) != 1:
        raise SystemExit(
            f"servono i pesi ImageNet in {radice}: trovati {len(trovati)} file "
            "'resnet18-*.pth'. Vanno scaricati sull'host — il container di gara "
            "non ha rete.")
    r = torchvision.models.resnet18(weights=None)
    r.load_state_dict(torch.load(trovati[0], map_location="cpu", weights_only=True))

    modello.stem[0].load_state_dict(r.conv1.state_dict())
    modello.stem[1].load_state_dict(r.bn1.state_dict())
    strati = [r.layer1, r.layer2, r.layer3, r.layer4]
    for i, s in enumerate(modello.strati):
        s.load_state_dict(strati[i].state_dict())
    n = sum(p.numel() for p in modello.stem.parameters())
    n += sum(p.numel() for p in modello.strati.parameters())
    return n


def build_model(arch: str = "media", in_ch: int = 1, dropout: float = 0.3,
                norm: str = "batch"):
    """CNN 3D a quattro blocchi: 32³ -> 2³ -> media globale -> logit.

    La capacita' e' il rischio principale del ramo, perche' quattro gruppi
    coprono l'88,8% dei dati e la CV raggruppata ha circa quattro ripetizioni
    indipendenti. Misurate tre larghezze con archi appaiati a cinque seed:
    `media` e' l'unica leva del modello che abbia superato nettamente il proprio
    errore, e `wide` torna indietro. L'ottimo non e' al bordo.
    """
    import torch.nn as nn

    if arch in ARCHITETTURE_25D:
        return _costruisci_mip(in_ch=in_ch, dropout=dropout,
                              **ARCHITETTURE_25D[arch])

    w = ARCHITETTURE[arch]

    def normalizza(c):
        """La normalizzazione e' un canale di fuga per l'identita' del centro.

        `BatchNorm3d` accumula media e varianza **su tutto il training set** e le
        usa in `eval()`. Il training set e' dominato al 39% da una sola impronta
        scanner, quindi quelle statistiche descrivono soprattutto quel centro: su
        un esame che ne viene da un altro la normalizzazione e' disallineata, e
        il disallineamento si propaga fino al logit come uno **spostamento
        sistematico**. E' il meccanismo che spiega il reperto del 3 agosto
        (verdetto consolidato in `docs/DECISIONS.md`): `p_cnn` sta a 0,9139
        aggregata contro 0,9335 dentro i centri, mentre il modello SBR — che non
        ha nessuna statistica di dataset dentro di se' — non ha il divario.

        `istanza` e `gruppo` normalizzano dentro il singolo esame: nessuna
        statistica di dataset, quindi niente da disallineare. E' anche la forma
        piu' coerente col resto del progetto, dove ogni misura e' un rapporto
        interno al paziente.

        `batch` resta il default perche' e' cio' con cui sono stati addestrati i
        pesi spediti: cambiarlo di nascosto renderebbe `voxnet.pt` incaricabile
        oppure — peggio — caricabile e sbagliato.
        """
        if norm == "batch":
            return nn.BatchNorm3d(c)
        if norm == "istanza":
            return nn.InstanceNorm3d(c, affine=True)
        if norm == "gruppo":
            return nn.GroupNorm(min(8, c), c)
        raise ValueError(f"normalizzazione sconosciuta: {norm!r}")

    def block(ci, co):
        return nn.Sequential(
            nn.Conv3d(ci, co, 3, padding=1, bias=False),
            normalizza(co), nn.ReLU(inplace=True),
            nn.Conv3d(co, co, 3, padding=1, stride=2, bias=False),
            normalizza(co), nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        block(in_ch, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3]),
        nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        nn.Dropout(dropout), nn.Linear(w[3], 1),
    )


def autonormalizza(x, eps: float = 1e-8):
    """Divide ogni crop per la propria norma quadratica: toglie il riferimento.

    Il crop arriva gia' diviso per `measurement.reference_mean`
    (`pipeline.py:153`), cioe' per l'uptake occipitale del paziente. Quella
    scelta e' una **regione di normalizzazione**, ed e' un parametro libero che
    il progetto ha tarato a mano: restringerla al piano striatale vale −0,0082.
    Un parametro che si tara e' un parametro che puo' essere tarato male, e
    soprattutto e' una statistica calcolata su una regione la cui resa dipende
    dal campo di vista, che varia da 190 a 630 mm fra i centri.

    Zhou & Tagare (IEEE BIBM 2021, *Self-normalized Classification of
    Parkinson's Disease DaTscan Images*) mostrano che la classificazione
    normalizzata equivale a separare i **semiraggi** delle immagini a meno di
    equivalenza moltiplicativa, e che quindi la regione di normalizzazione si
    puo' togliere del tutto: basta proiettare ogni immagine sulla sfera unitaria.

    Qui costa una riga, perche' i due fattori si compongono. Dividere per la
    norma del crop gia' normalizzato equivale a dividere per la norma del crop
    **grezzo**: `reference_mean` si semplifica e sparisce dal percorso. Niente
    regione, niente parametro da tarare, e nessuna statistica che dipenda dal
    FOV.

    E' la stessa forma del reperto del 3 agosto — `BatchNorm3d` toglieva una
    statistica di *dataset*, questa toglie una statistica di *regione* — con la
    differenza che quella era un difetto e questa e' una scelta di progetto che
    finora non era mai stata messa in discussione.

    Funziona su ndarray e su tensori torch, perche' vive sul percorso condiviso.
    """
    if hasattr(x, "detach"):          # torch
        piatto = x.reshape(x.shape[0], -1) if x.ndim > 3 else x.reshape(1, -1)
        n = piatto.float().pow(2).mean(dim=1).sqrt().clamp_min(eps)
        return (x.float() / n.reshape((-1,) + (1,) * (x.ndim - 1))).to(x.dtype)
    a = np.asarray(x, dtype=np.float64)
    assi = tuple(range(1, a.ndim)) if a.ndim > 3 else None
    n = np.sqrt(np.mean(a * a, axis=assi, keepdims=True))
    return a / np.maximum(n, eps)


def logit(p, eps: float = 1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fondi(p_sbr, p_cnn, a: float, b: float):
    """`sigmoid(a·logit(p_sbr) + b·logit(p_cnn))`, la fusione a due parametri."""
    z = a * logit(np.asarray(p_sbr, dtype=np.float64)) + b * logit(
        np.asarray(p_cnn, dtype=np.float64))
    return 1.0 / (1.0 + np.exp(-z))


def stima_fusione(y, p_sbr, p_cnn) -> tuple[float, float]:
    """(a, b) stimati su tutte le OOF: i coefficienti che vanno in inferenza.

    Il punteggio che si **riporta** non e' quello di questi coefficienti — e'
    quello annidato, calcolato da `scripts/fusione.py` su partizioni che non
    hanno visto le etichette con cui vengono valutate. Questi servono solo a
    predire, ed e' la stessa distinzione che `find_best_blend` fa gia' per il
    peso fra LightGBM e logistica.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([logit(p_sbr), logit(p_cnn)])
    m = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(X, y)
    return float(m.coef_[0][0]), float(m.coef_[0][1])


class VoxelEnsemble:
    """Le reti addestrate, pronte a predire un esame alla volta.

    Il caricamento e' separato dalla predizione di proposito: `main.py` deve
    poter scoprire *subito* se il ramo a voxel e' utilizzabile, e in caso
    contrario proseguire con il solo modello SBR invece di scoprirlo al primo
    esame e ripeterlo per tutti gli altri.
    """

    def __init__(self, config: dict, state_dicts: list, device: str = "cpu"):
        import torch

        self.config = config
        self.device = device
        self.riduzione = int(config.get("riduzione", RIDUZIONE))
        self.tta = bool(config.get("tta", True))
        self.models = []
        for sd in state_dicts:
            m = build_model(config["arch"], config.get("in_ch", 1),
                            config.get("dropout", 0.3),
                            config.get("norm", "batch"))
            m.load_state_dict({k: v.float() for k, v in sd.items()})
            m.eval().to(device)
            self.models.append(m)
        self._torch = torch

    @classmethod
    def carica(cls, path, device: str | None = None) -> "VoxelEnsemble":
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # `weights_only=True`: il file contiene solo tensori e tipi elementari,
        # e caricarlo senza quella garanzia significherebbe eseguire pickle
        # arbitrario. Non e' paranoia teorica — e' il nostro stesso archivio, ma
        # la regola giusta e' che un formato di pesi non sia un vettore di codice.
        blob = torch.load(str(path), map_location=device, weights_only=True)
        return cls(blob["config"], blob["state_dicts"], device)

    def __len__(self) -> int:
        return len(self.models)

    def predici(self, crops: np.ndarray) -> np.ndarray:
        """Probabilita' per ciascun crop. `crops` e' (N, 64, 64, 64) oppure (64,64,64).

        La media si fa **in probabilita'** e non in logit: e' la media bayesiana
        dei modelli, ed e' la stessa forma in cui `main.py` gia' media i cinque
        LightGBM e i trenta logistici. Mediare i logit darebbe una media
        geometrica delle odds, piu' confidente della media dei modelli — cioe'
        proprio il tipo di sicurezza in piu' che la log loss punisce.
        """
        torch = self._torch
        import torch.nn.functional as F

        x = np.asarray(crops, dtype=np.float32)
        if x.ndim == 3:
            x = x[None]
        # La stessa trasformazione dell'addestramento, letta dalla config del
        # checkpoint. Se qui e in addestramento divergessero, i pesi si
        # caricherebbero senza errori e predirebbero su una scala diversa: il
        # guasto che questo modulo esiste per rendere impossibile.
        if self.config.get("autonorm", False):
            x = autonormalizza(x).astype(np.float32)
        xb = torch.from_numpy(x)[:, None].to(self.device)
        if self.riduzione > 1:
            xb = F.avg_pool3d(xb, self.riduzione)

        somma = None
        with torch.no_grad():
            viste = [xb, torch.flip(xb, dims=[2])] if self.tta else [xb]
            for m in self.models:
                for v in viste:
                    p = torch.sigmoid(m(v).float().ravel())
                    somma = p if somma is None else somma + p
        media = (somma / (len(self.models) * (2 if self.tta else 1))).cpu().numpy()
        return np.clip(media.astype(np.float64), CLIP_LO, CLIP_HI)
