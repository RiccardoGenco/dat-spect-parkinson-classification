"""Come si misura quanto vale un modello nuovo accanto a quello che c'e' gia'.

Sta in un modulo e non dentro uno script perche' due consumatori la usano —
`10_voxel_cnn.py` mentre addestra e `12_fusione.py` come analisi a posteriori — e
due copie di "onesto" che divergono in silenzio sono la classe di difetto contro
cui `src/model.py` e `src/pipeline.py` esistono. Qui la posta e' piu' alta che
altrove: e' il numero su cui si decide se spendere le settimane che restano.

Le tre cose che questo modulo sa e che sono costate una misura
---------------------------------------------------------------
**1. Lo spazio in cui si somma conta piu' del numero di parametri.**
`w*p_a + (1-w)*p_b` e' una media convessa in probabilita': su un esame in cui
l'incumbent dice 0,99 e il modello nuovo 0,70, qualunque peso non nullo trascina
verso il basso una predizione confidente e corretta, e quelle sono il 70,6% del
dataset. L'ottimizzatore si difende scegliendo w piccolo, e con w piccolo il
modello nuovo non muove piu' nemmeno la zona grigia. In logit — `logit_a +
b*logit_b` — il vincolo sparisce: dove `logit_a` e' grande domina lui, e dove
vale circa zero (cioe' nella zona grigia) l'unico termine che muove la predizione
e' quello del modello nuovo. La porta che serviva e' dentro la parametrizzazione.

**2. Un'intercetta nella fusione compra un guadagno che non esiste.**
Misurato: con annidamento stratificato la logistica piena (pendenza, peso,
intercetta) sulla sonda del ramo C dava +0,0083, contro +0,0032 della versione a
un parametro. Con annidamento **raggruppato** la stessa logistica piena crolla a
-0,0321. Il colpevole e' identificato per fold: tenendo fuori il gruppo a 2,46 mm
— 528 esami, il 39% del dataset, prevalenza 0,597 contro 0,548 — l'intercetta
stimata sul resto cambia segno, e riapplicata proprio a quel centro costa da sola
piu' di quanto il modello nuovo renda sugli altri quattro fold. E' lo spostamento
di prevalenza fra centri gia' registrato come non correggibile: non un difetto
della fusione, ma una cosa che una fusione con intercetta si mette nella
condizione di sbagliare.

**3. L'annidamento raggruppato e' il metro giusto, e lo si e' verificato su un
caso di cui si conosce la risposta.** Sul blend LightGBM/logistica gia' spedito,
`peso_prob` rende +0,0014 stratificato e **+0,0010 raggruppato**: passa. Le
fusioni con intercetta, sullo stesso identico caso, crollano a -0,0123 e -0,0161.
Quindi il metro non e' severo in generale — e' severo esattamente con cio' che
deve bocciare. Un metro che non distingue non e' un metro (R3).

Da cui la regola operativa: **si riportano tutte le fusioni, ma quella che decide
e' a un parametro e con annidamento raggruppato.**
"""

from __future__ import annotations

import numpy as np

CLIP_LO, CLIP_HI = 0.005, 0.995


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def log_loss_semplice(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fai_split(y, groups, raggruppati: bool, seed: int = 0):
    """Partizione interna per la scelta dei parametri di fusione."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if raggruppati:
        s = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(s.split(np.zeros(len(y)), y, groups))
    s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(s.split(np.zeros(len(y)), y))


def combina(base, nuovo, b, spazio, idx=slice(None)):
    if spazio == "prob":
        return b * nuovo[idx] + (1 - b) * base[idx]
    z = logit(base)[idx] + b * logit(nuovo)[idx]
    return 1 / (1 + np.exp(-z))


def annidata_scalare(y, base, nuovo, splits, spazio: str):
    """Fusione a UN parametro, cercato su griglia, valutata fuori campione.

    La ricerca su griglia invece della discesa del gradiente e' deliberata: e' la
    stessa forma di `find_best_blend` in `src/model.py`, cosi' i due numeri sono
    confrontabili senza doversi chiedere se a differire sia l'ottimizzatore.

    A b=0 entrambi gli spazi riproducono l'incumbent **esattamente**. E' cio' che
    rende il guadagno attribuibile al modello nuovo e a nient'altro: non c'e'
    nessun grado di liberta' che possa ricalibrare l'incumbent per conto suo.
    """
    griglia = np.linspace(0, 1, 101)

    def migliore(idx):
        perdite = [log_loss_semplice(y[idx], combina(base, nuovo, b, spazio, idx))
                   for b in griglia]
        return float(griglia[int(np.argmin(perdite))])

    pred = np.zeros(len(y))
    for tr, va in splits:
        pred[va] = combina(base, nuovo, migliore(tr), spazio, va)
    return pred, migliore(np.arange(len(y)))


def annidata_logistica(y, X, splits, intercetta: bool) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    pred = np.zeros(len(y))
    for tr, va in splits:
        m = LogisticRegression(C=1.0, max_iter=2000,
                               fit_intercept=intercetta).fit(X[tr], y[tr])
        pred[va] = m.predict_proba(X[va])[:, 1]
    return pred


def valuta(y, p, grey) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    q = np.clip(p, CLIP_LO, CLIP_HI)
    return {
        "log_loss": float(log_loss(y, q)),
        "auroc": float(roc_auc_score(y, q)),
        "auc_grigia": float(roc_auc_score(y[grey], q[grey])),
    }


def tabella(y, base, nuovo, groups, grey, seed: int = 0) -> list[dict]:
    """Tutte le fusioni, entrambi gli annidamenti, piu' il controllo nullo.

    Il controllo nullo — la stessa fusione con le predizioni del modello nuovo
    mescolate — deve dare esattamente zero, perche' la ricerca su griglia sceglie
    b=0 quando il secondo ingresso e' rumore. Se non lo desse, l'estimatore
    starebbe comprando guadagno dal nulla e nessun'altra riga della tabella
    varrebbe niente.
    """
    lb, ln = logit(base), logit(nuovo)
    rng = np.random.default_rng(seed)
    permutato = nuovo[rng.permutation(len(nuovo))]

    righe = [{"fusione": "incumbent grezzo", "dof": 0, "annidamento": "-",
              **valuta(y, base, grey)},
             {"fusione": "nuovo da solo", "dof": 0, "annidamento": "-",
              **valuta(y, nuovo, grey)}]

    for raggr in (False, True):
        eti = "raggruppato" if raggr else "stratificato"
        splits = fai_split(y, groups, raggr, seed)
        for spazio, nome in (("prob", "peso_prob"), ("logit", "offset_logit")):
            pred, pieno = annidata_scalare(y, base, nuovo, splits, spazio)
            righe.append({"fusione": nome, "dof": 1, "annidamento": eti,
                          "parametro": pieno, **valuta(y, pred, grey)})
        pred, _ = annidata_scalare(y, base, permutato, splits, "logit")
        righe.append({"fusione": "offset_logit permutato", "dof": 1,
                      "annidamento": eti, **valuta(y, pred, grey)})
        # `solo_base_temp` sta a `logit_2p` come `solo_base_platt` sta a
        # `logit_3p`: e' la stessa fusione **senza il modello nuovo**. Una
        # logistica su `logit_base` con intercetta bloccata a zero e' una
        # ricalibrazione di temperatura dell'incumbent, e puo' guadagnare per
        # conto suo. Senza questa riga, tutto cio' che `logit_2p` fa in piu' del
        # peso a un parametro verrebbe attribuito al modello nuovo per
        # esclusione, che e' esattamente il difetto B1 del piano.
        for nome, X, inter in (("logit_2p", np.column_stack([lb, ln]), False),
                               ("logit_3p", np.column_stack([lb, ln]), True),
                               ("solo_base_temp", lb[:, None], False),
                               ("solo_base_platt", lb[:, None], True)):
            righe.append({"fusione": nome, "dof": X.shape[1] + int(inter),
                          "annidamento": eti,
                          **valuta(y, annidata_logistica(y, X, splits, inter), grey)})
    return righe


def verdetto(righe: list[dict]) -> dict:
    """I due numeri che decidono, estratti dalla tabella.

    `guadagno_deciso` e' contro l'incumbent grezzo, a un parametro, in logit, con
    annidamento raggruppato. `guadagno_piano` e' il numero su cui il piano per il
    podio ha fissato il cancello di 0,008 — peso scalare in probabilita' con
    annidamento stratificato — e si riporta perche' una soglia risponde alla
    domanda su cui e' tarata e a nessun'altra (R5).
    """
    def prendi(f, a):
        for r in righe:
            if r["fusione"] == f and r["annidamento"] == a:
                return r
        raise KeyError((f, a))

    rif = righe[0]["log_loss"]
    return {
        "incumbent": rif,
        "guadagno_piano": rif - prendi("peso_prob", "stratificato")["log_loss"],
        "guadagno_deciso": rif - prendi("offset_logit", "raggruppato")["log_loss"],
        "guadagno_2p_raggruppato": rif - prendi("logit_2p", "raggruppato")["log_loss"],
        # Il contributo NETTO del modello nuovo nella fusione a due parametri:
        # quanto batte la stessa fusione senza di lui.
        "contributo_netto_2p": (prendi("solo_base_temp", "raggruppato")["log_loss"]
                                - prendi("logit_2p", "raggruppato")["log_loss"]),
        "guadagno_logit_stratificato": rif - prendi("offset_logit",
                                                    "stratificato")["log_loss"],
        "controllo_nullo": rif - prendi("offset_logit permutato",
                                        "raggruppato")["log_loss"],
        "b_raggruppato": prendi("offset_logit", "raggruppato")["parametro"],
        "auc_grigia_fusa": prendi("offset_logit", "raggruppato")["auc_grigia"],
    }
