"""Definizione del modello: punto di ingresso condiviso fra misura e submission.

Perche' sta qui e non dentro lo script di training
--------------------------------------------------
Vale per il modello la stessa ragione per cui `pipeline.py` e' l'unico punto da
cui passa il preprocessing. Finche' la definizione viveva dentro una CLI,
qualunque altro consumatore - un harness di confronto, un'ablazione, uno studio
sulla calibrazione - poteva solo duplicarla o importare uno script per percorso.
Entrambe le strade producono la stessa classe di difetto: **due definizioni che
divergono in silenzio**, qui fra il modello che si misura e il modello che si
spedisce. Un divario del genere non comparirebbe in cross-validation, perche' la
cross-validation la esegue una delle due.

Cosa contiene
-------------
I due modelli, il loro blend, e la funzione che li compone in un unico risultato
valutato. Nessuna funzione qui stampa o scrive su disco: sono decisioni della
CLI, non del modello, ed e' cio' che rende questo modulo riusabile 72 volte di
fila da un harness senza effetti collaterali.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import evaluate_oof, make_folds
from .features import EXCLUDED_FROM_MODEL

TARGET = "is_pathologic"
# Le colonne con prefisso "_" sono metadati di acquisizione. Tenerle fuori dal
# modello e' deliberato: identificano il centro, e se la prevalenza di casi
# patologici differisce tra centri il modello imparerebbe quella scorciatoia
# invece della patologia, con un crollo sul test set.
META_PREFIX = "_"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Colonne date al modello.

    Due esclusioni distinte. Il prefisso "_" toglie i metadati di acquisizione,
    che identificano il centro in modo palese. EXCLUDED_FROM_MODEL toglie
    feature che sembrano misure ma non lo sono - intensita' grezze e artefatti
    della soglia relativa al picco - e che il prefisso non intercetterebbe
    perche' hanno nomi da grandezza legittima. Le ragioni, una per feature,
    stanno in src/features.py.
    """
    return [
        c
        for c in frame.columns
        if c not in {TARGET, "uid"}
        and not c.startswith(META_PREFIX)
        and c not in EXCLUDED_FROM_MODEL
    ]


def as_frame(X, columns):
    """Avvolge la matrice in un DataFrame con i nomi delle feature.

    Il contratto fra training e inferenza diventa cosi' per NOME invece che
    posizionale. Non e' pignoleria: allenando su un array nudo, LightGBM
    genera nomi propri (Column_0, Column_1, ...) e nel container - dove la
    versione di libreria e' piu' recente ed espone feature_names_in_ - sklearn
    emette un warning a ogni singola predizione. Con 5 modelli per esame sono
    5 righe di log a esame, e i log di gara sono limitati a 300 righe: su un
    test set di qualche centinaio di esami la diagnostica verrebbe troncata
    via, lasciando solo warning.

    Il beneficio vero e' comunque un altro: con i nomi espliciti, un
    disallineamento nell'ordine delle colonne fra training e inferenza diventa
    un errore invece che una predizione sbagliata in silenzio.
    """
    return pd.DataFrame(np.asarray(X, dtype=np.float64), columns=list(columns))


def train_lightgbm(X, y, folds, columns, groups=None, seed=0):
    """LightGBM con early stopping su una porzione interna del training.

    Il punto delicato e' da dove viene il criterio di arresto. La versione
    precedente passava il fold di validation come eval_set e poi prediceva lo
    stesso fold: il numero di alberi veniva scelto guardando le etichette dei
    campioni da predire. La log loss out-of-fold che ne usciva era ottimistica
    per costruzione, ed e' l'unico numero su cui si itera per sei settimane.

    Quanto vale, misurato invece che supposto: su 1362 campioni di rumore puro
    con queste dimensioni (27 feature, 8 gruppi, prevalenza 0,548) e cinque
    ripetizioni, la versione con leak si assesta sulla baseline di prevalenza
    (-0,00003) mentre quella corretta le sta sopra di +0,0017, che e' il
    comportamento giusto quando non c'e' segnale. L'ottimismo e' quindi
    dell'ordine di 0,002 di log loss, non di piu': gli iperparametri
    conservativi - learning rate 0,02, otto foglie - limitano molto quanto
    l'arresto anticipato riesca a sfruttare le etichette che vede. Vale la pena
    correggerlo perche' non costa nulla e perche' su una classifica fitta due
    millesimi non sono irrilevanti, ma non era il difetto che ribalta il
    progetto. La misura sta in tests/ come nota, non come test automatico:
    richiede minuti di calcolo e non ha senso eseguirla a ogni modifica.

    Qui il fold di validation non viene mai osservato durante l'addestramento.
    L'arresto si decide su una porzione ritagliata DAL TRAINING, tenendo i
    gruppi separati anche li': se lo stesso scanner comparisse in entrambe le
    parti, il criterio di arresto misurerebbe la memorizzazione del centro
    invece della generalizzazione, e si reintrodurrebbe una versione attenuata
    dello stesso problema.

    Iperparametri conservativi di proposito. Con 1362 esami e poche decine di
    feature, un modello ad alta capacita' memorizza il rumore per centro;
    foglie poche e sottocampionamento marcato producono OOF piu' stabili e,
    cosa che qui conta di piu', probabilita' meglio calibrate.
    """
    import lightgbm as lgb
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    oof = np.zeros(len(y))
    models = []
    stopped_at = []

    for fold in sorted(np.unique(folds)):
        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)

        # Ritaglio interno per l'arresto, disgiunto dal fold di validation.
        inner_groups = None if groups is None else np.asarray(groups)[train_idx]
        if inner_groups is not None and len(np.unique(inner_groups)) >= 4:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
            fit_pos, stop_pos = next(splitter.split(train_idx, y[train_idx], inner_groups))
        else:
            fit_pos, stop_pos = train_test_split(
                np.arange(len(train_idx)), test_size=0.2,
                stratify=y[train_idx], random_state=seed + fold,
            )
        fit_idx, stop_idx = train_idx[fit_pos], train_idx[stop_pos]

        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=8,
            max_depth=4,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            random_state=seed + fold,
            verbose=-1,
        )
        frame = as_frame(X, columns)
        model.fit(
            frame.iloc[fit_idx], y[fit_idx],
            eval_set=[(frame.iloc[stop_idx], y[stop_idx])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[valid_idx] = model.predict_proba(frame.iloc[valid_idx])[:, 1]
        models.append(model)
        stopped_at.append(model.best_iteration_ or model.n_estimators)

    return oof, models, stopped_at


# Griglia di regolarizzazione su cui si MEDIA, invece di sceglierne un punto.
#
# In scala logaritmica il risultato e' insensibile a C - da 1 a 32 la log loss
# resta fra 0,3147 e 0,3188 - e la media sulla griglia vale quanto il miglior
# punto singolo (0,3146 contro 0,3147). Mediare invece di scegliere ha due
# vantaggi: nessun iperparametro viene tarato sulla stessa CV che si riporta,
# quindi il numero resta onesto; e la predizione non dipende da un punto scelto
# su un plateau, dove la scelta e' comunque arbitraria.
#
# Verificato che serva: con selezione ANNIDATA di C su una CV interna il
# risultato peggiora a 0,3194, perche' con pochi gruppi la scelta interna e'
# instabile e capita di prendere un C sbagliato. Mediare e' meglio che scegliere
# male.
LOGISTIC_C_GRID: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

# Quando il tabellare e' il modello finale, la griglia sopra e' il suo plateau
# migliore. Quando invece entra accanto alla MIP, una regolarizzazione molto
# piu' forte conserva meno dettagli ridondanti con la rete e migliora la
# fusione pur essendo peggiore da sola. I due punti si mediano: non si sceglie
# il migliore sulla stessa OOF che si riporta.
FUSION_LOGISTIC_C_GRID: tuple[float, ...] = (0.1, 0.3)


def train_logistic(X, y, folds, columns):
    """Regressione logistica su rapporti in scala logaritmica, mediata su C.

    Serve da contrappeso al gradient boosting: e' quasi impossibile che
    sovradatti su questo numero di feature, ed e' naturalmente ben calibrata.
    Mediarla con LightGBM migliora la log loss anche quando la sua AUROC da sola
    e' inferiore. Sui dati veri non e' nemmeno inferiore: la logistica batte il
    gradient boosting di quattro centesimi di log loss.

    Il primo stadio - `LogRatios` - e' quello che porta il guadagno, e la
    ragione per esteso sta in src/transforms.py. In sintesi: l'effetto del
    centro sull'SBR e' moltiplicativo, quindi in scala grezza cambia sia la
    pendenza sia l'intercetta da centro a centro, mentre in logaritmi diventa
    una costante additiva e un solo coefficiente vale per tutti.

    Sta dentro la pipeline sklearn di proposito, non applicato ai dati prima:
    cosi' viaggia dentro models.pkl e l'inferenza nel container lo applica senza
    saperlo. Una trasformazione fatta a mano nello script di training sarebbe
    una divergenza fra training e inferenza in attesa di succedere, ed e'
    esattamente il tipo di divario che la cross-validation non mostrerebbe.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .transforms import LogRatios

    oof = np.zeros(len(y))
    models = []
    frame = as_frame(X, columns)

    for fold in sorted(np.unique(folds)):
        train_idx, valid_idx = folds != fold, folds == fold
        predictions = []
        for C in LOGISTIC_C_GRID:
            model = make_pipeline(
                LogRatios(),
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=C, max_iter=5000),
            )
            model.fit(frame.loc[train_idx], y[train_idx])
            predictions.append(model.predict_proba(frame.loc[valid_idx])[:, 1])
            models.append(model)
        # main.py media indistintamente tutti i modelli della lista, quindi la
        # media su C e quella sui fold coincidono con questa.
        oof[valid_idx] = np.mean(predictions, axis=0)

    return oof, models


def train_fusion_logistic(X, y, folds, columns):
    """Logistica fortemente regolarizzata destinata soltanto alla fusione MIP.

    Non sostituisce il ramo SBR di ripiego: senza CNN il blend LightGBM +
    logistica esistente resta nettamente migliore. Questo ramo viene salvato
    separatamente nel bundle e usato soltanto quando la rete ha prodotto una
    predizione valida.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .transforms import LogRatios

    oof = np.zeros(len(y))
    models = []
    frame = as_frame(X, columns)
    for fold in sorted(np.unique(folds)):
        train_idx, valid_idx = folds != fold, folds == fold
        predictions = []
        for C in FUSION_LOGISTIC_C_GRID:
            model = make_pipeline(
                LogRatios(), SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=C, max_iter=5000),
            )
            model.fit(frame.loc[train_idx], y[train_idx])
            predictions.append(model.predict_proba(frame.loc[valid_idx])[:, 1])
            models.append(model)
        oof[valid_idx] = np.mean(predictions, axis=0)
    return oof, models


def find_best_blend(y, oof_a, oof_b):
    """Peso di miscelazione ottimo, e stima onesta di cosa rende.

    Scegliere il peso sulle predizioni OOF e poi riportare la log loss di
    quelle stesse predizioni e' la versione attenuata del leak di early
    stopping: il peso ha visto tutte le etichette. Con 51 valori su una griglia
    monodimensionale l'ottimismo e' piccolo, ma non e' zero, e questo progetto
    non ha altri numeri di cui fidarsi.

    Il peso viene quindi scelto in cross-validation annidata: per ogni fold si
    ottimizza sugli altri e si valuta su quello escluso. Il peso restituito e'
    quello che serve in inferenza (stimato su tutto), ma il punteggio riportato
    e' quello annidato, che non ha mai visto le etichette con cui e' valutato.
    """
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold

    grid = np.linspace(0, 1, 51)

    def best_on(mask):
        scores = [
            log_loss(y[mask], np.clip(w * oof_a[mask] + (1 - w) * oof_b[mask], 1e-6, 1 - 1e-6))
            for w in grid
        ]
        return float(grid[int(np.argmin(scores))])

    everything = np.ones(len(y), dtype=bool)
    weight = best_on(everything)

    nested = np.zeros(len(y))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for inner_train, inner_valid in splitter.split(np.zeros(len(y)), y):
        mask = np.zeros(len(y), dtype=bool)
        mask[inner_train] = True
        w = best_on(mask)
        nested[inner_valid] = w * oof_a[inner_valid] + (1 - w) * oof_b[inner_valid]

    honest = float(log_loss(y, np.clip(nested, 1e-6, 1 - 1e-6)))
    return weight, honest


def fit_and_score(
    frame: pd.DataFrame,
    columns: list[str],
    seed: int = 0,
    n_splits: int = 5,
    groups: np.ndarray | None = None,
    folds: np.ndarray | None = None,
) -> dict:
    """Addestra i due modelli, li fonde, e restituisce tutto senza stampare nulla.

    `folds` esiste per il confronto appaiato. I fold dipendono solo da
    etichette, gruppi, seed e numero di split - mai dalle colonne - quindi due
    archi con lo stesso seed li otterrebbero identici anche ricalcolandoli. Ma
    "identici per coincidenza" e "identici per costruzione" si distinguono il
    giorno in cui qualcuno aggiunge una dipendenza dalle colonne: passarli
    esplicitamente rende il disegno appaiato una proprieta' del codice invece
    che una conseguenza da ricontrollare. E' la stessa distinzione fra una
    grandezza invariante per costruzione e una la cui invarianza e' stata
    misurata su venti esami.
    """
    y = frame[TARGET].to_numpy(dtype=int)
    X = frame[columns].to_numpy(dtype=np.float64)

    if folds is None:
        folds = make_folds(y, groups, n_splits=n_splits, seed=seed)

    oof_gbm, models_gbm, stopped_at = train_lightgbm(
        X, y, folds, columns, groups=groups, seed=seed
    )
    oof_lr, models_lr = train_logistic(X, y, folds, columns)
    oof_fusion_lr, models_fusion_lr = train_fusion_logistic(
        X, y, folds, columns
    )

    weight, nested = find_best_blend(y, oof_gbm, oof_lr)
    oof_blend = weight * oof_gbm + (1 - weight) * oof_lr

    metrics = evaluate_oof(y, oof_blend)
    return {
        "log_loss": metrics["log_loss"],
        "log_loss_nested": nested,
        "auroc": metrics["auroc"],
        "baseline_log_loss": metrics["baseline_log_loss"],
        "log_loss_gbm": evaluate_oof(y, oof_gbm)["log_loss"],
        "log_loss_lr": evaluate_oof(y, oof_lr)["log_loss"],
        "auroc_gbm": evaluate_oof(y, oof_gbm)["auroc"],
        "auroc_lr": evaluate_oof(y, oof_lr)["auroc"],
        "weight": weight,
        "oof": oof_blend,
        "oof_gbm": oof_gbm,
        "oof_lr": oof_lr,
        "oof_fusion_lr": oof_fusion_lr,
        "models_gbm": models_gbm,
        "models_lr": models_lr,
        "models_fusion_lr": models_fusion_lr,
        "stopped_at": stopped_at,
        "folds": folds,
    }
