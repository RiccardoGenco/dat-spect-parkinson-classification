"""Schema di cross-validation.

Questa e' la decisione piu' importante di tutto il progetto, piu' della scelta
del modello. Una CV che sovrastima porta a ottimizzare per settimane nella
direzione sbagliata, e il divario si scopre solo sulla leaderboard privata,
quando non c'e' piu' tempo.

Il problema specifico
---------------------
I dati vengono da 10 centri ospedalieri francesi con scanner e protocolli
diversi. Gli organizzatori dichiarano esplicitamente che generalizzare tra
centri e' parte della sfida. Una StratifiedKFold normale mescola i centri tra
train e validation: il modello impara le idiosincrasie di ciascuno scanner e la
CV lo premia, ma sul test set - che puo' contenere proporzioni diverse di centri
- quel vantaggio svanisce.

L'impronta dello scanner
------------------------
Il set di identificatori dei centri non e' fornito. Ma i metadati NIfTI lo
tradiscono: dimensioni della matrice e voxel spacing sono determinati dal
protocollo di ricostruzione, quindi la coppia (shape, spacing) e' di fatto
un'impronta digitale dello scanner. Raggruppare su questa impronta e tenere
i gruppi separati tra fold rende la CV una misura di generalizzazione
cross-centro, cioe' della stessa cosa che misura il test set.

Va detto onestamente: e' un proxy, non la verita'. Due centri con lo stesso
modello di gamma camera collassano nello stesso gruppo, e un centro che ha
cambiato protocollo si divide in due. Resta comunque molto piu' vicino alla
realta' di una CV che ignora del tutto la struttura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scanner_fingerprint(shape: tuple[int, ...], spacing: tuple[float, ...]) -> str:
    """Chiave di raggruppamento derivata dalla geometria di acquisizione.

    Usa il voxel spacing e SOLO quello. La forma della matrice non entra, ed e'
    una correzione fatta due volte contro i dati.

    Il primo tentativo usava la forma completa. Ma sui 1362 esami di training
    quattro acquisizioni a 3,90 mm avevano forme 128x128x36, 128x128x41,
    128x128x42 e 128x128x128: stesso protocollo, campo di vista assiale
    ritagliato diversamente da paziente a paziente. Il numero di fette e' una
    variabile del paziente.

    Il secondo tentativo teneva la matrice nel piano, assumendola fissata dal
    protocollo. Non lo e': a 2,46 mm i 528 esami di quel gruppo si distribuiscono
    su una quindicina di matrici diverse (128, 134, 138, 140, 142, 144, 148,
    150, 152, 156, 164, 170, 256...), cioe' anche l'estensione nel piano viene
    ritagliata per paziente. Un solo scanner finiva spezzato in quindici gruppi.

    Misurato sui 1362 esami, contando quanti finiscono in gruppi abbastanza
    grandi da reggere un fold (almeno 20 esami):

        forma completa + spacing   131 gruppi   79,2%
        matrice nel piano + spacing 67 gruppi   87,5%
        solo spacing                43 gruppi   92,5%

    Lo spacing resta a 2 decimali: a 1 decimale i gruppi scenderebbero a 19 ma
    si fonderebbero protocolli che differiscono per spessore di fetta
    (3,30x3,30x3,30 con 3,30x3,30x3,00), che sono acquisizioni diverse.

    Resta un proxy, e sbaglia nella direzione giusta. Due centri con la stessa
    gamma camera collassano in un gruppo unico: questo rende la CV piu' severa
    del vero, non piu' indulgente. L'errore pericoloso e' l'opposto - spezzare
    un centro fra fold diversi - perche' quello rende la CV ottimistica, ed e'
    esattamente cio' che le due versioni precedenti facevano.

    Perche' tutto questo conta: la prevalenza di casi patologici varia da 0,154
    a 0,778 fra i gruppi di spacing. Un modello capace di riconoscere lo scanner
    ha a disposizione una scorciatoia enorme, e la cross-validation la premia se
    i gruppi non sono tenuti separati fra i fold.
    """
    return "x".join(f"{float(s):.2f}" for s in spacing[:3])


def make_folds(
    labels: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Assegna a ogni campione un indice di fold.

    Usa StratifiedGroupKFold quando i gruppi sono disponibili, cosi' che ogni
    impronta di scanner compaia in un solo fold, mantenendo comunque bilanciata
    la proporzione di casi patologici.

    Returns:
        Array int di lunghezza len(labels) con l'indice di fold di validation.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    fold_ids = np.full(len(labels), -1, dtype=int)

    if groups is not None and len(np.unique(groups)) >= n_splits:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        iterator = splitter.split(np.zeros(len(labels)), labels, groups)
    else:
        # Fallback: se i gruppi distinti sono meno dei fold, raggrupparli
        # renderebbe i fold troppo sbilanciati. Meglio una stratificazione
        # semplice, ma va segnalato perche' la CV diventa ottimistica.
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(labels)), labels)

    for fold, (_, validation_idx) in enumerate(iterator):
        fold_ids[validation_idx] = fold

    return fold_ids


def describe_groups(groups: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Riepilogo dei gruppi, per capire quanto e' eterogeneo il dataset.

    Da guardare prima di allenare qualunque cosa: se un solo gruppo copre l'80%
    dei dati, la CV per gruppi diventa instabile e va ripensata.
    """
    frame = pd.DataFrame({"group": groups, "label": labels})
    summary = (
        frame.groupby("group")
        .agg(n=("label", "size"), positive_rate=("label", "mean"))
        .sort_values("n", ascending=False)
    )
    summary["share"] = summary["n"] / summary["n"].sum()
    return summary


def evaluate_oof(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Metriche out-of-fold.

    La metrica ufficiale e' la log loss. L'AUROC e' riportata solo come
    diagnostica: separa il potere discriminante dalla calibrazione. Se l'AUROC
    e' alta ma la log loss e' mediocre, il problema e' la calibrazione e si
    risolve con un post-processing, non cambiando modello.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    clipped = np.clip(predictions, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(labels, clipped)),
        "auroc": float(roc_auc_score(labels, clipped)),
        "brier": float(np.mean((clipped - labels) ** 2)),
        "base_rate": float(np.mean(labels)),
        "baseline_log_loss": float(
            log_loss(labels, np.full_like(clipped, np.mean(labels)))
        ),
    }
