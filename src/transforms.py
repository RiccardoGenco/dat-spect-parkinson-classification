"""Trasformazione in scala logaritmica delle grandezze che sono rapporti.

Il reperto che la motiva
------------------------
Sui 1362 esami, a classe fissa, il 31-36% della varianza di OGNI feature SBR e'
spiegato dall'impronta dello scanner (eta quadro calcolato dentro ciascuna
classe, quindi al netto della diversa prevalenza fra centri). Per confronto:
`localization_confidence` sta a 0,13 e gli indici di asimmetria a 0,01. E la
correlazione fra quanto una feature e' contaminata dal centro e quanto sembra
discriminante e' +0,71 - cioe' le feature che paiono migliori a dati aggregati
sono esattamente le piu' confuse dal centro di provenienza.

La forma della contaminazione, misurata sui gruppi grandi:

    spacing   SBR normali   SBR patologici   rapporto
    1,47         2,286          1,236          1,85
    2,30         2,441          1,475          1,65
    2,40         2,300          1,081          2,13
    2,46         3,263          1,974          1,65
    3,90         2,199          1,316          1,67

Il livello assoluto varia del 50%, il rapporto fra le classi resta fra 1,65 e
2,13. **L'effetto centro e' moltiplicativo, la separazione fra classi anche.**

Da notare che non e' un effetto di risoluzione, che sarebbe stato la spiegazione
ovvia: a 2,40 mm l'SBR medio dei normali e' 2,30 e a 2,46 mm e' 3,26, cioe' quasi
lo stesso campionamento e il 40% di differenza. Dipende da altro nel protocollo
di ricostruzione - filtro, correzioni, iterazioni - e questo esclude di poterlo
correggere uniformando la risoluzione con un filtraggio.

Cosa cambia passando ai logaritmi
---------------------------------
Con SBR = (segnale - riferimento) / riferimento, si ha 1 + SBR = segnale /
riferimento, che e' positivo per costruzione. In log:

    log(1 + SBR) = log(fattore del centro) + log(rapporto vero)

Il fattore del centro smette di essere un guadagno che moltiplica anche la
differenza fra le classi e diventa una costante additiva. Un solo coefficiente
logistico vale allora per tutti i centri, e a sbagliarsi resta la sola
intercetta. In scala grezza invece cambiavano insieme pendenza e intercetta, ed
e' il motivo per cui `min_sbr` da sola arriva a 0,79 di AUROC ma a una log loss
di appena 0,61 contro 0,688 della baseline: ordina bene i pazienti dentro un
centro, e sbaglia la soglia appena il centro cambia.

Misurato in CV raggruppata, media su 8 seed appaiati, sola regressione logistica:

    grezze, C=0,5 (attuale)      0,3236
    grezze, al suo C migliore    0,3217
    log,    al suo C migliore    0,3147
    log,    media su C=1..32     0,3146

Un effetto collaterale che vale piu' del guadagno: in scala logaritmica il
risultato smette di dipendere dalla forza di regolarizzazione. Da C=1 a C=32 la
log loss resta fra 0,3147 e 0,3188, mentre in scala grezza degrada da 0,3217 a
0,3354. Si puo' quindi mediare su tutta la griglia invece di sceglierne un
punto, e nessun iperparametro viene tarato sulla metrica che si riporta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def log_transform_kind(name: str) -> str | None:
    """Quale trasformazione spetta a una feature, dedotta dal suo nome.

    Il criterio non e' il nome in se' ma cosa la grandezza e':

    - "shift": vale (segnale - riferimento) / riferimento, quindi 1 + valore e'
      un rapporto positivo e il logaritmo va preso di quello.
    - "ratio": e' gia' un rapporto positivo fra due misure dello stesso esame.
    - None: non e' un rapporto. Correlazioni, indici di asimmetria, lunghezze e
      pendenze restano dove sono; prenderne il logaritmo non corrisponderebbe a
      nessuna proprieta' dei dati e per meta' di loro non sarebbe definito.

    Gli indici `asym_*` sono esclusi di proposito benche' siano rapporti: sono
    gia' invarianti al fattore del centro per costruzione (eta quadro 0,01), e
    infatti sono le uniche feature che il problema non ce l'hanno.
    """
    if name.startswith("asym_"):
        return None
    if "sbr" in name:
        return "shift"
    if name.endswith(("_ratio", "concentration")):
        return "ratio"
    return None


class LogRatios(BaseEstimator, TransformerMixin):
    """Porta in scala logaritmica le colonne che sono rapporti, lasciando le altre.

    Va messo per PRIMO nella pipeline, prima dell'imputazione, per due ragioni.
    La prima e' pratica: qui le colonne hanno ancora un nome, e dopo
    `SimpleImputer` sarebbero una matrice anonima. La seconda conta di piu': la
    mediana di un campione con cui si imputa e' meglio calcolarla in scala
    logaritmica, dove la distribuzione e' simmetrica, che in scala grezza dove
    ha una coda lunga.

    I NaN attraversano intatti: il logaritmo di NaN e' NaN, e a riempirli ci
    pensa lo stadio successivo, che e' l'unico posto dove si deve decidere.
    """

    def fit(self, X, y=None):  # noqa: ARG002, D102
        names = list(getattr(X, "columns", []))
        if not names:
            raise ValueError(
                "LogRatios ha bisogno dei nomi delle colonne: passagli un DataFrame. "
                "Con una matrice anonima non c'e' modo di sapere quali colonne "
                "siano rapporti, e trasformare quelle sbagliate sarebbe un errore "
                "silenzioso."
            )
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.kinds_ = {name: log_transform_kind(name) for name in names}
        self.n_features_in_ = len(names)
        return self

    def transform(self, X):  # noqa: D102
        frame = pd.DataFrame(X, columns=list(self.feature_names_in_)).astype(np.float64)
        out = frame.copy()
        for name, kind in self.kinds_.items():
            if kind is None:
                continue
            values = frame[name].to_numpy(dtype=np.float64)
            # Il clip protegge dal caso degenere in cui il segnale scende sotto
            # il riferimento e il rapporto arriva a zero. Sui dati reali non
            # succede - il piu' basso e' ben sopra - ma un esame di test corrotto
            # non deve produrre -inf e propagarlo a tutta la predizione.
            shifted = (1.0 + values) if kind == "shift" else values
            out[name] = np.log(np.clip(shifted, 1e-3, None))
        return out

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002, D102
        return np.asarray(self.feature_names_in_, dtype=object)
