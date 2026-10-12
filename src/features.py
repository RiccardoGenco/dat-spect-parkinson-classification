"""Costruzione del vettore di feature da una misurazione striatale.

Le feature sono scelte per corrispondere a cio' che un medico nucleare guarda
davvero, perche' con poche migliaia di esami un modello con un prior corretto
batte quasi sempre un modello che deve scoprire il prior dai dati.

Le tre famiglie che portano il segnale:

1. SBR assoluti      - quanto uptake resta, normalizzato sul riferimento
2. Rapporto putamen/caudato - la perdita nel parkinsonismo inizia dal putamen
                              posteriore, quindi questo rapporto cala prima
                              dell'SBR complessivo
3. Asimmetria        - il parkinsonismo esordisce quasi sempre da un lato

Sull'asimmetria: l'indice usa il valore assoluto perche' quale sia il lato piu'
colpito non ha significato diagnostico, mentre l'entita' della differenza si'.
Usare la differenza con segno insegnerebbe al modello una lateralita' spuria.
"""

from __future__ import annotations

import numpy as np

from .morphology import MORPHOLOGY_KEYS
from .striatum import Hemisphere, StriatalMeasurement, binding_ratio

NAN = float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return NAN
    if abs(denominator) < 1e-8:
        return NAN
    return numerator / denominator


def _asymmetry_index(left: float, right: float) -> float:
    """Indice di asimmetria normalizzato, in [0, 2].

    AI = 2 * |L - R| / (L + R)

    Normalizzato sulla somma cosi' che non dipenda dal livello assoluto di
    uptake: un paziente con uptake globalmente basso ma simmetrico deve
    risultare simmetrico.
    """
    if not np.isfinite(left) or not np.isfinite(right):
        return NAN
    total = left + right
    if abs(total) < 1e-8:
        return NAN
    return 2.0 * abs(left - right) / abs(total)


def _shape_features(hemi: Hemisphere | None, prefix: str) -> dict[str, float]:
    """Descrittori di forma della VOI.

    Nel parkinsonismo la classica forma a virgola degenera in punto: la VOI
    diventa piu' piccola e piu' compatta. Gli autovalori della matrice di
    covarianza dei voxel catturano questo allungamento senza richiedere un
    atlante.
    """
    if hemi is None:
        return {
            f"{prefix}_volume_mm3": NAN,
            f"{prefix}_elongation": NAN,
            f"{prefix}_flatness": NAN,
        }

    coords = np.array(np.nonzero(hemi.voi_mask), dtype=np.float64)
    if coords.shape[1] < 4:
        return {
            f"{prefix}_volume_mm3": hemi.voi_volume_mm3,
            f"{prefix}_elongation": NAN,
            f"{prefix}_flatness": NAN,
        }

    centered = coords - coords.mean(axis=1, keepdims=True)
    cov = np.cov(centered)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    major = float(np.sqrt(eigenvalues[0]))
    middle = float(np.sqrt(eigenvalues[1]))
    minor = float(np.sqrt(eigenvalues[2]))

    return {
        f"{prefix}_volume_mm3": hemi.voi_volume_mm3,
        f"{prefix}_elongation": _safe_ratio(major, middle),
        f"{prefix}_flatness": _safe_ratio(middle, minor),
    }


def _hemisphere_features(
    hemi: Hemisphere | None, reference: float, prefix: str
) -> dict[str, float]:
    if hemi is None:
        base = {
            f"{prefix}_sbr": NAN,
            f"{prefix}_sbr_caudate": NAN,
            f"{prefix}_sbr_putamen": NAN,
            f"{prefix}_sbr_peak": NAN,
            f"{prefix}_putamen_caudate_ratio": NAN,
        }
        base.update(_shape_features(None, prefix))
        return base

    sbr = binding_ratio(hemi.voi_mean, reference)
    sbr_caudate = binding_ratio(hemi.caudate_mean, reference)
    sbr_putamen = binding_ratio(hemi.putamen_mean, reference)
    sbr_peak = binding_ratio(hemi.peak_value, reference)

    features = {
        f"{prefix}_sbr": sbr,
        f"{prefix}_sbr_caudate": sbr_caudate,
        f"{prefix}_sbr_putamen": sbr_putamen,
        f"{prefix}_sbr_peak": sbr_peak,
        f"{prefix}_putamen_caudate_ratio": _safe_ratio(sbr_putamen, sbr_caudate),
    }
    features.update(_shape_features(hemi, prefix))
    return features


# Chiavi morfologiche di cui si tiene il valore peggiore e migliore fra i due
# emisferi. Solo min e max, non anche la media: con due soli lati la media e'
# la loro semisomma, cioe' una colonna esattamente collineare che non aggiunge
# nulla a un modello lineare e diluisce l'importanza in uno ad alberi.
#
# E non i valori sinistro e destro separati, che pure sarebbero disponibili:
# quale lato sia il piu' colpito non ha significato diagnostico, e darli al
# modello gli insegnerebbe una lateralita' spuria. E' la stessa ragione per cui
# gli indici di asimmetria usano il valore assoluto.
_ALL_MORPHOLOGY_PAIRED: tuple[str, ...] = (
    "sbr_core",
    "sbr_full",
    "sbr_core_slab",
    "sbr_full_slab",
    "concentration",
    "ap_skew",
    "shape_corr",
    # `ap_slope` resta QUI, prima di `post_ant_ratio`, benche' sia appena uscita
    # dal modello. La posizione nella tupla fissa l'ordine delle colonne nel CSV,
    # e l'ordine delle colonne entra nel sottocampionamento di LightGBM: spostarla
    # renderebbe una ri-estrazione futura non confrontabile con i numeri misurati
    # qui, per un motivo che non comparirebbe da nessuna parte.
    "ap_slope",
    "post_ant_ratio",
    # Blocco direzionale, dopo l'ablazione appaiata a otto archi
    # (verdetto consolidato in `docs/DECISIONS.md`). Dei ventuno
    # candidati ne sopravvive UNO, ed e' uno scambio a parita' di colonne:
    # `ap_log_slope` prende il posto di `ap_slope`, -0,0010 su 8 seed su 8.
    #
    # Questa tupla dice cosa si CALCOLA, non cosa entra nel modello: il filtro e'
    # EXCLUDED_FROM_MODEL, e i venti bocciati sono elencati la' con il loro costo
    # accanto. Restano calcolati di proposito, perche' `include` di 05_compare
    # puo' riaprire un verdetto solo finche' la colonna esiste nel CSV - ed e'
    # cio' che rende quei verdetti riapribili invece che definitivi. Il prezzo
    # sono ~12 minuti di estrazione in piu' sui 1362 esami, quasi tutti dovuti
    # ai due profili in piu' per emisfero (obliquo e medio-laterale): e' il costo
    # da rivedere per primo se il ciclo di iterazione tornasse a essere stretto.
    "ap_log_slope",
    "ap_tail_ratio",
    "ap_bin_m15_ratio",
    "ap_bin_m09_ratio",
    "ap_bin_p09_ratio",
    "ap_bin_p15_ratio",
    "obl_log_slope",
    "obl_tail_ratio",
    "ml_log_slope",
    "ml_tail_ratio",
)

# Quelle davvero accoppiate sono le chiavi che `morphology.py` produce davvero.
# Il filtro passa da MORPHOLOGY_KEYS e non da una seconda lista scritta qui: due
# elenchi delle stesse chiavi divergerebbero in silenzio alla prima direzione
# riattivata, e il sintomo sarebbe una colonna di soli NaN - che non fa fallire
# niente e sposta le predizioni. E' la stessa ragione per cui il modello ha una
# definizione sola in model.py.
_MORPHOLOGY_PAIRED: tuple[str, ...] = tuple(
    key for key in _ALL_MORPHOLOGY_PAIRED if key in frozenset(MORPHOLOGY_KEYS)
)

# Chiavi di cui basta un solo riassunto bilaterale, con la ragione della scelta.
_MORPHOLOGY_SINGLE: dict[str, str] = {
    "center_shift_mm": "max",  # instabilita' dell'ancora: conta il caso peggiore
    "ap_extent_mm": "mean",  # estensione: e' una misura di taglia, si media
    "ml_extent_mm": "mean",
    "si_extent_mm": "mean",
    # Diagnostica sull'asse striatale, non feature: vedi EXCLUDED_FROM_MODEL.
    # Si media perche' l'anatomia e' la stessa sui due lati e la media e' la
    # stima piu' stabile dell'angolo vero.
    "ap_axis_angle_deg": "mean",
}


def _morphology_features(measurement: StriatalMeasurement) -> dict[str, float]:
    """Riassunti bilaterali del blocco a geometria fissa.

    Il blocco arriva gia' calcolato da `morphology.py`, che lo produce per
    emisfero. Qui si sceglie soltanto come combinare i due lati.
    """
    from .morphology import MORPHOLOGY_KEYS

    left = measurement.morphology_left or dict.fromkeys(MORPHOLOGY_KEYS, NAN)
    right = measurement.morphology_right or dict.fromkeys(MORPHOLOGY_KEYS, NAN)

    def pair(key: str) -> np.ndarray:
        return np.array([left.get(key, NAN), right.get(key, NAN)], dtype=np.float64)

    features: dict[str, float] = {}

    for key in _MORPHOLOGY_PAIRED:
        values = pair(key)
        finite = np.isfinite(values).any()
        features[f"min_{key}"] = float(np.nanmin(values)) if finite else NAN
        features[f"max_{key}"] = float(np.nanmax(values)) if finite else NAN

    for key, how in _MORPHOLOGY_SINGLE.items():
        values = pair(key)
        finite = np.isfinite(values).any()
        reducer = {"max": np.nanmax, "mean": np.nanmean, "min": np.nanmin}[how]
        features[f"{how}_{key}"] = float(reducer(values)) if finite else NAN

    # Asimmetria calcolata su uptake grezzi anziche' su SBR. E' la stessa
    # grandezza clinica di `asym_sbr`, ma ben condizionata: |L-R|/(L+R) fra due
    # quantita' positive resta in [0, 2] e varia con continuita', mentre la
    # versione su SBR ha al denominatore il legame specifico totale, che tende a
    # zero proprio nei casi bilaterali gravi. E' il motivo per cui una
    # perturbazione dell'1% sugli ingressi sposta `asym_sbr` del 30-170%.
    features["asym_uptake_fixed"] = _asymmetry_index(
        left.get("uptake_full", NAN), right.get("uptake_full", NAN)
    )
    features["asym_uptake_voi"] = _asymmetry_index(
        measurement.left.voi_mean if measurement.left is not None else NAN,
        measurement.right.voi_mean if measurement.right is not None else NAN,
    )

    # Asimmetria del GRADIENTE, non del livello. `asym_uptake_fixed` chiede se un
    # lato capta meno dell'altro; questa chiede se un lato ha perso la coda
    # posteriore piu' dell'altro, che e' letteralmente il modo in cui il
    # parkinsonismo esordisce - putamen posteriore per primo, e da un lato solo.
    #
    # Vale la pena provarla per due ragioni misurate. La famiglia `asym_*` e' la
    # meno contaminata dal centro di tutta la pipeline (eta quadro 0,010 contro
    # 0,28-0,35 delle feature SBR), e `asym_uptake_fixed` e' gia' fra le prime
    # tre per importanza NONOSTANTE sia calcolata su una sfera da 22 mm, che
    # media via proprio la coda posteriore. La zona grigia dove si concentra il
    # 40% della log loss residua ha prevalenza 0,494, cioe' e' fatta di casi
    # dubbi: e' li' che una perdita precoce e unilaterale vive.
    #
    # E' ben condizionata per costruzione, a differenza di `asym_sbr`: e' un
    # rapporto fra due quantita' positive vicine a 1, non fra due legami
    # specifici che tendono a zero insieme nei casi gravi.
    features["asym_ap_tail_ratio"] = _asymmetry_index(
        left.get("ap_tail_ratio", NAN), right.get("ap_tail_ratio", NAN)
    )

    return features


def _anchor_drift_features(measurement: StriatalMeasurement) -> dict[str, float]:
    """Dove sta l'ancora rispetto a un riferimento che l'intensita' non muove.

    Tutte le VOI del progetto sono a geometria fissa ma **ancorate a un punto
    derivato dall'intensita'**: il modo dell'uptake trovato con mean shift.
    Misurato il 3 agosto sui 1362 esami
    (verdetto consolidato in `docs/DECISIONS.md`), rispetto al baricentro della
    maschera cerebrale l'ancora dei patologici sta **1,45 mm piu' avanti e
    1,37 mm piu' medialmente**: un vettore di 2,0 mm diretto verso il caudato,
    che e' esattamente dove il residuo di uptake si concentra quando il putamen
    posteriore degenera. Sopravvive al condizionamento sul volume della
    maschera, quindi non e' atrofia.

    Il baricentro della maschera e' l'unico riferimento geometrico disponibile
    che non dipenda dall'intensita' dello striato — e' per questo che la deriva
    si puo' misurare rispetto a lui e non rispetto all'ancora stessa.

    **Le due colonne restano fuori dal modello** finche' l'harness appaiato non
    dice quale delle due letture sia vera: la deriva puo' essere un errore di
    centratura che attenua il contrasto, oppure informazione diagnostica che la
    pipeline incassa gia' gratis. I millimetri non lo distinguono; il Delta di
    log loss si'. Vedi `experiments/deriva_ancora.json`.
    """
    centro = measurement.mask_centroid
    sinistra = measurement.left.centroid if measurement.left is not None else None
    destra = measurement.right.centroid if measurement.right is not None else None
    mm = measurement.spacing_mm

    if centro is None or sinistra is None or destra is None or not np.isfinite(mm):
        return {"anchor_ap_mm": NAN, "anchor_semiwidth_mm": NAN}

    sinistra = np.asarray(sinistra, dtype=np.float64)
    destra = np.asarray(destra, dtype=np.float64)
    return {
        # Asse 1 = antero-posteriore: dopo `as_closest_canonical` gli assi sono
        # RAS, e y cresce verso l'anteriore.
        "anchor_ap_mm": float(
            ((sinistra[1] + destra[1]) / 2 - float(centro[1])) * mm),
        # Semidistanza fra le due ancore: la larghezza apparente dello striato,
        # che si stringe quando il residuo si concentra sul caudato.
        "anchor_semiwidth_mm": float(abs(sinistra[0] - destra[0]) / 2 * mm),
    }


def build_features(measurement: StriatalMeasurement) -> dict[str, float]:
    """Vettore di feature completo per un esame."""
    reference = measurement.reference_mean
    features: dict[str, float] = {}

    features.update(_hemisphere_features(measurement.left, reference, "left"))
    features.update(_hemisphere_features(measurement.right, reference, "right"))

    # Aggregati bilaterali. Il minimo tra i due lati e' spesso piu' informativo
    # della media: la diagnosi si fonda sul lato piu' compromesso, e mediare
    # diluisce il segnale proprio nei casi asimmetrici precoci.
    for key in ("sbr", "sbr_caudate", "sbr_putamen", "putamen_caudate_ratio"):
        left_value = features.get(f"left_{key}", NAN)
        right_value = features.get(f"right_{key}", NAN)
        pair = np.array([left_value, right_value], dtype=np.float64)

        features[f"min_{key}"] = float(np.nanmin(pair)) if np.isfinite(pair).any() else NAN
        features[f"max_{key}"] = float(np.nanmax(pair)) if np.isfinite(pair).any() else NAN
        features[f"mean_{key}"] = float(np.nanmean(pair)) if np.isfinite(pair).any() else NAN
        features[f"asym_{key}"] = _asymmetry_index(left_value, right_value)

    features["reference_mean"] = reference

    # Confidenza della deduzione geometrica sull'orientamento. Esposta al
    # modello di proposito: quando e' bassa, la localizzazione striatale non e'
    # affidabile - tipicamente per un focus extracerebrale intenso o per uno
    # striato quasi assente - e le altre feature vanno prese con cautela.
    # Darla in pasto al modello gli permette di imparare quella cautela invece
    # di trattare misure spurie come misure valide.
    features["localization_confidence"] = measurement.anterior_confidence
    features["header_orientation_agrees"] = float(measurement.header_agrees)

    left_volume = features.get("left_volume_mm3", NAN)
    right_volume = features.get("right_volume_mm3", NAN)
    features["total_volume_mm3"] = (
        left_volume + right_volume
        if np.isfinite(left_volume) and np.isfinite(right_volume)
        else NAN
    )
    features["asym_volume"] = _asymmetry_index(left_volume, right_volume)

    features.update(_morphology_features(measurement))
    features.update(_anchor_drift_features(measurement))

    return features


FEATURE_NAMES: tuple[str, ...] = tuple(
    sorted(
        build_features(
            StriatalMeasurement(left=None, right=None, reference_mean=1.0, midline_x=0.0)
        ).keys()
    )
)

# Feature calcolate ma tenute fuori dal modello, ciascuna con la sua ragione.
#
# Restano calcolate di proposito invece di essere cancellate: servono come
# diagnostica, e l'esclusione va riverificata sui 1362 esami di training in CV
# raggruppata per scanner, dove la domanda "questa feature generalizza fra
# centri" ha una risposta vera. Le decisioni qui sotto sono prese su venti
# esami dello smoke test, che bastano a mostrare un artefatto ma non a
# quantificarne il costo.
EXCLUDED_FROM_MODEL: dict[str, str] = {
    # Intensita' grezza in conteggi. Sui 20 esami reali va da 1,0 a 10.513:
    # e' dose iniettata, tempo di acquisizione e scala di ricostruzione, cioe'
    # l'identita' del centro travestita da misura. E' esattamente cio' che il
    # prefisso "_" esclude altrove, solo che questa non lo aveva.
    "reference_mean": "intensita' grezza: identifica lo scanner, non la patologia",
    # La VOI e' definita come i voxel sopra il 50% del picco emisferico. Quando
    # la patologia abbassa il picco, la soglia si abbassa con lui e la VOI si
    # allarga nel tessuto circostante. Misurato sui dati reali: VOI dei
    # patologici 1900-5200 voxel contro 1300-2600 dei normali, cioe' il volume
    # discrimina nella direzione OPPOSTA all'anatomia, dove uno striato
    # degenerato dovrebbe risultare piu' piccolo. Il segnale c'e' (AUC 0,79)
    # ma e' l'artefatto di soglia, e non c'e' ragione di credere che si ripeta
    # con la stessa intensita' su uno scanner diverso.
    "left_volume_mm3": "artefatto della soglia relativa al picco",
    "right_volume_mm3": "artefatto della soglia relativa al picco",
    "total_volume_mm3": "artefatto della soglia relativa al picco",
    "asym_volume": "artefatto della soglia relativa al picco",
    # Stessa causa: descrivono la forma di una VOI la cui estensione dipende dal
    # picco anziche' dall'anatomia.
    "left_elongation": "forma di una VOI a soglia: stesso artefatto del volume",
    "right_elongation": "forma di una VOI a soglia: stesso artefatto del volume",
    "left_flatness": "forma di una VOI a soglia: stesso artefatto del volume",
    "right_flatness": "forma di una VOI a soglia: stesso artefatto del volume",
    # Costante a 1,0 su 20 esami reali su 20: tutti gli header sono in RAS.
    # Varianza nulla, nessuna informazione. Resta calcolata perche' se mai
    # diventasse variabile su un centro sarebbe un segnale diagnostico
    # importante, ma darla al modello ora e' solo peso morto.
    "header_orientation_agrees": "costante sui dati reali: varianza nulla",
    # --- blocco a geometria fissa: le parti che sono state MISURATE inutili ---
    #
    # Escluse dopo ablazione appaiata su 6 seed della CV raggruppata, non per
    # sospetto. Le altre voci di questa lista rispondono alla domanda "questa
    # grandezza e' cio' che dice di essere"; queste rispondono a "questa
    # grandezza aggiunge qualcosa a quelle che ho gia'", che e' una domanda
    # diversa e ha bisogno di una misura, non di un ragionamento.
    #
    # Il caso piu' istruttivo e' `min_shape_corr`. Da sola separa quanto la
    # feature piu' forte del modello (AUC 0,904 su 140 esami veri, identica a
    # `localization_confidence`) ed e' la sua versione senza soglia, quindi era
    # la candidata piu' promettente di tutto il blocco. Aggiunta, peggiora:
    # +0,0023 di log loss, 0 seed su 6. Misura la stessa cosa dell'originale, e
    # due colonne quasi identiche in un modello con penalita' L2 si dividono il
    # coefficiente e aggiungono varianza senza aggiungere informazione.
    # `max_shape_corr` invece resta, e la ragione e' che non e' ridondante:
    # `localization_confidence` e' una media pesata dei due lati, quindi somiglia
    # al minimo; il massimo dice se ALMENO UN emisfero ha ancora la forma a
    # virgola, che e' un'informazione che nel modello non c'era.
    "min_shape_corr": "duplicato di localization_confidence: +0,0023, 0 seed su 6",
    # I momenti di terzo ordine e le estensioni sono le uniche grandezze del
    # blocco che hanno la dimensione di una lunghezza invece di essere rapporti,
    # e infatti si comportano come tali: la larghezza apparente di una struttura
    # dipende dalla risoluzione effettiva della ricostruzione, cioe' dal centro.
    # Insieme costano +0,0087 di log loss su 0 seed su 6.
    "min_ap_skew": "asimmetria del profilo: nulla su dati veri (AUC 0,52)",
    "max_ap_skew": "asimmetria del profilo: nulla su dati veri (AUC 0,54)",
    "mean_ap_extent_mm": "e' una lunghezza: segue la risoluzione, cioe' il centro",
    "mean_ml_extent_mm": "e' una lunghezza: segue la risoluzione, cioe' il centro",
    "mean_si_extent_mm": "e' una lunghezza: segue la risoluzione, cioe' il centro",
    # Distanza fra argmax e baricentro dell'uptake. Nata come diagnostica sulla
    # stabilita' dell'ancora e utile in quel ruolo, ma come feature non porta
    # nulla (AUC 0,56) e resta una lunghezza come le precedenti.
    "max_center_shift_mm": "diagnostica sull'ancora, non una misura del paziente",
    # Le due componenti della DERIVA dell'ancora, misurate il 3 agosto su tutti i
    # 1362 esami: rispetto al
    # baricentro della maschera cerebrale, l'ancora dei patologici sta 1,45 mm
    # piu' avanti e 1,37 mm piu' medialmente — un vettore di 2,0 mm diretto
    # antero-medialmente, cioe' verso il caudato, che e' esattamente dove il
    # residuo di uptake si concentra quando il putamen posteriore degenera. Il
    # d di Cohen e' 0,27 sull'asse antero-posteriore e 0,53 sulla semilarghezza,
    # con risposta monotona alla dose (rho -0,074 e +0,255 con `min_sbr`), e
    # sopravvive al condizionamento sul volume della maschera, quindi non e'
    # atrofia.
    #
    # Restano ESCLUSE perche' la misura osservazionale non distingue le due
    # letture opposte: la deriva puo' essere un errore di centratura che
    # ATTENUA il contrasto, e allora va rimossa ancorando le VOI all'anatomia;
    # oppure puo' essere informazione diagnostica che la pipeline oggi incassa
    # gratis, e allora rimuoverla costerebbe. A deciderlo e' l'harness appaiato
    # (`experiments/deriva_ancora.json`), non altri millimetri.
    "anchor_ap_mm": "deriva dell'ancora: misurata, verdetto aperto",
    "anchor_semiwidth_mm": "deriva dell'ancora: misurata, verdetto aperto",
    # Angolo dell'asse principale dello striato nel piano assiale. Esclusa PRIMA
    # di misurarla, che e' un'eccezione in questa lista e ha bisogno di una
    # ragione forte: sarebbe la TERZA misura della stessa grandezza di
    # `shape_corr` e `localization_confidence` - tutte e tre chiedono quanto
    # l'asse antero-posteriore e quello medio-laterale sono accoppiati dentro lo
    # striato - e la seconda, `min_shape_corr`, ha gia' prezzato quella
    # ridondanza a +0,0023 su 0 seed su 6 nonostante da sola separasse quanto la
    # feature piu' forte del modello. Un terzo tentativo non ha bisogno di essere
    # misurato per essere previsto.
    #
    # Resta calcolata perche' nel suo ruolo vero - dire se OBLIQUE_ANGLE_DEG e'
    # l'angolo giusto - risponde con i dati a una domanda a cui altrimenti si
    # risponderebbe per asserzione, e a rischio zero.
    "mean_ap_axis_angle_deg": "diagnostica sull'asse: terza misura di localization_confidence",
    #
    # --- I venti bocciati del blocco direzionale --------------------------------
    #
    # Ablazione appaiata a otto archi su otto seed, riferimento A0 = 0,2965
    # (verdetto consolidato in `docs/DECISIONS.md`). Il rumore fra seed
    # sull'arco di riferimento e' sd 0,0021: un Delta va letto insieme ai seed
    # vinti, e nessuno di questi ne vince abbastanza.
    #
    # `ap_slope` esce dal modello ed e' l'unico movimento dell'esperimento. Non
    # perche' non porti nulla - era fra le prime dieci per importanza - ma perche'
    # `ap_log_slope` porta la STESSA cosa con meno centro dentro: eta quadro da
    # 0,087 a 0,055/0,025, a parita' di numero di colonne. Il fattore
    # moltiplicativo del centro, in logaritmo, e' una costante additiva uguale in
    # ogni bin e sparisce dalla pendenza.
    "min_ap_slope": "sostituita da min_ap_log_slope: -0,0010 su 8 seed su 8",
    "max_ap_slope": "sostituita da max_ap_log_slope: -0,0010 su 8 seed su 8",
    # `ap_tail_ratio` misura la stessa cosa di `ap_log_slope` e lo dice il dato:
    # correlano -0,984. Provata come scambio al posto di `post_ant_ratio`, e'
    # l'arco A4: +0,0010 su 0 seed su 8. Fra due misure ridondanti ha vinto quella
    # in logaritmo, ed e' coerente con il perche' entrambe esistono.
    "min_ap_tail_ratio": "duplicato di ap_log_slope (r = -0,984): +0,0010, 0 seed su 8",
    "max_ap_tail_ratio": "duplicato di ap_log_slope (r = -0,984): +0,0010, 0 seed su 8",
    # I quattro bin esterni grezzi al posto della sintesi, arco A2: +0,0011 su
    # 0 seed su 8. Tenendo ANCHE `post_ant_ratio`, arco A3, peggiora ancora a
    # +0,0043 - la stessa informazione due volte, che e' esattamente il
    # meccanismo con cui era gia' stata respinta `min_shape_corr`. Otto colonne
    # in piu' che aggiungono varianza e nessuna informazione.
    "min_ap_bin_m15_ratio": "profilo grezzo al posto della sintesi: +0,0011, 0 seed su 8",
    "max_ap_bin_m15_ratio": "profilo grezzo al posto della sintesi: +0,0011, 0 seed su 8",
    "min_ap_bin_m09_ratio": "profilo grezzo al posto della sintesi: +0,0011, 0 seed su 8",
    "max_ap_bin_m09_ratio": "profilo grezzo al posto della sintesi: +0,0011, 0 seed su 8",
    "min_ap_bin_p09_ratio": "profilo grezzo al posto della sintesi: +0,0011, 0 seed su 8",
    "max_ap_bin_p09_ratio": "profilo grezzo al posto della sintesi: +0,0011, 0 seed su 8",
    "min_ap_bin_p15_ratio": "profilo grezzo al posto della sintesi: +0,0011, 0 seed su 8",
    "max_ap_bin_p15_ratio": "profilo grezzo al posto della sintesi: +0,0011, 0 seed su 8",
    # Profilo campionato a 30 gradi dall'asse AP invece che lungo y, arco A6:
    # +0,0048 su 0 seed su 8. La diagnostica `mean_ap_axis_angle_deg` misura 28,8
    # gradi di mediana, quindi l'angolo NON era sbagliato: e' il campionamento
    # obliquo a costare, perche' il cilindro attraversa piu' tessuto eterogeneo
    # per la stessa lunghezza. La risposta e' netta e vale la pena tenerla, visto
    # che l'ipotesi "l'asse anatomico batte l'asse dell'immagine" era ragionevole.
    "min_obl_log_slope": "campionamento obliquo a 30 gradi: +0,0048, 0 seed su 8",
    "max_obl_log_slope": "campionamento obliquo a 30 gradi: +0,0048, 0 seed su 8",
    "min_obl_tail_ratio": "campionamento obliquo a 30 gradi: +0,0048, 0 seed su 8",
    "max_obl_tail_ratio": "campionamento obliquo a 30 gradi: +0,0048, 0 seed su 8",
    # Blocco medio-laterale, arco A7: +0,0064 su 0 seed su 8, il peggiore di
    # tutti - previsto, e per la ragione prevista. Fra il centroide del putamen e
    # quello del caudato ci sono 12-14 mm, poco piu' di un elemento di risoluzione
    # SPECT: un contrasto su una base cosi' corta misura la sfocatura della
    # ricostruzione, cioe' il centro, non il paziente.
    "min_ml_log_slope": "base piu' corta della risoluzione SPECT: +0,0064, 0 seed su 8",
    "max_ml_log_slope": "base piu' corta della risoluzione SPECT: +0,0064, 0 seed su 8",
    "min_ml_tail_ratio": "base piu' corta della risoluzione SPECT: +0,0064, 0 seed su 8",
    "max_ml_tail_ratio": "base piu' corta della risoluzione SPECT: +0,0064, 0 seed su 8",
    # Asimmetria del gradiente invece che del livello, arco A5: +0,0001 su 4 seed
    # su 8, cioe' testa o croce. E' l'unico risultato che sorprende - la famiglia
    # `asym_` e' la meno contaminata dal centro di tutta la pipeline (eta quadro
    # 0,013) e la perdita precoce e unilaterale del putamen posteriore e'
    # letteralmente il modo in cui la malattia esordisce. Il prescreen aveva pero'
    # gia' detto perche' non poteva funzionare: AUC 0,545 in generale e 0,503
    # nella zona grigia, cioe' rumore esattamente dove si perde il 40% della log
    # loss. Un'ipotesi clinicamente giusta su una misura che non separa.
    "asym_ap_tail_ratio": "asimmetria del gradiente: +0,0001, 4 seed su 8 (rumore)",
}

# Feature effettivamente passate al modello.
MODEL_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name in FEATURE_NAMES if name not in EXCLUDED_FROM_MODEL
)

# `localization_confidence`: il caso che NON e' escluso, e la ragione.
#
# Il sospetto era che condividesse l'artefatto del volume - misura quanto la
# geometria della VOI e' conclusiva, e una VOI diffusa risulta meno conclusiva.
# Il criterio per decidere era: se codifica l'identita' del centro, toglierla
# deve danneggiare piu' la CV semplice, dove quella scorciatoia e' sfruttabile,
# che quella raggruppata per scanner.
#
# Misurato sui 1362 esami, log loss del blend:
#
#                                    CV raggruppata   CV semplice
#     tutte le feature                   0,3178         0,2948
#     senza localization_confidence      0,4075         0,3682
#     senza min_sbr                      0,3180         0,2951
#     senza asym_sbr                     0,3180         0,2956
#     senza min_putamen_caudate_ratio    0,3185         0,2944
#
# Il verdetto e' opposto al sospetto. Il danno e' MAGGIORE in CV raggruppata
# (+0,0897 contro +0,0734): e' la feature che porta generalizzazione fra
# centri, non quella che la aggira. E togliere una qualunque feature SBR non
# costa nulla - sono collineari e si sostituiscono a vicenda - mentre per
# questa non c'e' sostituto: quattrocento volte l'impatto delle altre.
#
# Da sola vale 0,4497 di log loss (AUROC 0,867), quasi quanto le altre
# ventisei insieme (0,4075). Unite danno 0,3178, un salto che nessuna delle due
# spiega: portano informazione complementare.
#
# Perche' funziona, a posteriori. La confidenza e' la correlazione fra
# coordinata antero-posteriore e distanza dalla linea mediana dentro la VOI.
# In uno striato sano il caudato e' anteromediale e il putamen posterolaterale,
# quindi la correlazione e' forte: e' la forma a virgola. Quando la virgola
# degrada a punto - il segno clinico della DaTscan parkinsoniana - la
# correlazione si perde. Nata come diagnostica per dedurre l'orientamento
# dall'anatomia, e' finita per essere la misura piu' fedele del criterio
# clinico vero presente in tutta la pipeline, e per caso.
#
# Implicazione operativa: l'asse morfologico rende quanto l'asse di intensita'
# ed e' rappresentato da una sola feature ottenuta di rimbalzo. Svilupparlo di
# proposito vale piu' di qualsiasi affinamento delle feature SBR, che sono
# gia' sature.


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """Converte un dizionario di feature in un array nell'ordine canonico."""
    return np.array([features.get(name, NAN) for name in FEATURE_NAMES], dtype=np.float64)
