"""Descrittori morfologici dello striato, calcolati senza soglia.

Perche' esiste questo modulo
----------------------------
La VOI di `striatum.py` e' definita come i voxel sopra il 50% del picco
emisferico. E' la convenzione clinica, ma ha una conseguenza che i dati reali
hanno reso evidente: **la VOI e' definita dal segnale che dovrebbe misurare**.
Quando la patologia abbassa il picco, la soglia scende con lui e la regione si
allarga nel tessuto circostante. Da qui l'esclusione di volume, elongation e
flatness dal modello: non descrivono l'anatomia, descrivono la soglia.

Il danno peggiore pero' non e' sulle feature escluse, e' sul rapporto
putamen/caudato, che sui 1362 esami resta inchiodato fra 1,06 e 1,10 in
entrambe le classi mentre clinicamente dovrebbe spaziare da 1 a 3.
La causa e' strutturale e vale la pena scriverla per esteso, perche' e' il
motivo per cui la feature piu' importante della diagnosi non porta segnale:

    la VOI contiene per costruzione solo voxel fra il 50% e il 100% del picco;
    il taglio caudato/putamen sta al centroide della VOI stessa;
    quindi si confronta la meta' anteriore con la meta' posteriore di un blocco
    il cui intervallo di intensita' e' gia' fissato a [0.5, 1.0] del picco.

Il rapporto fra le due medie e' allora quasi una costante geometrica, e lo
resta qualunque cosa faccia la malattia. Peggio: quando il putamen posteriore
degenera, i suoi voxel scendono sotto soglia ed **escono dalla VOI**, il
centroide si sposta in avanti e il taglio si sposta con lui. La misura insegue
la patologia invece di misurarla.

Cosa fa questo modulo
---------------------
Sostituisce la soglia con una geometria fissa in millimetri. Nessuna regione
dipende dall'intensita': i raggi sono costanti fisiche, i pesi sono il legame
specifico (uptake meno riferimento), e tutte le grandezze prodotte sono
rapporti o momenti normalizzati, quindi invarianti rispetto al guadagno dello
scanner esattamente come l'SBR.

L'ancora resta il picco emisferico, che e' un punto di repere robusto anche in
patologia - lo striato e' comunque la struttura piu' captante del centro
encefalo - ma viene raffinato con qualche iterazione di mean shift sul legame
specifico, perche' l'argmax di un volume rumoroso non e' stabile mentre il
baricentro dell'uptake lo e'.

Cosa questo modulo NON e'
-------------------------
Non e' registrazione a template. Un'ancora derivata dai dati si sposta con la
patologia: in un caso parkinsoniano il baricentro dell'uptake migra verso il
caudato, quindi la posizione dell'origine e' essa stessa un'informazione
diagnostica e non un sistema di riferimento neutro. Per questo la posizione
dell'origine viene esposta come feature (`center_shift_mm`, `ap_peak_mm`)
invece di essere trattata come una costante: e' meglio darla al modello che
fingere che non esista.
"""

from __future__ import annotations

import numpy as np

# Raggi delle regioni sferiche, in millimetri. Costanti fisiche, mai derivate
# dall'intensita': e' l'intero punto del modulo.
#
# 10 mm: il nucleo dello striato, dominato dal massimo captante.
# 22 mm: l'intera struttura piu' un margine (lo striato umano si estende per
#        circa 30 mm lungo l'asse maggiore), quindi include anche il tessuto
#        dove l'uptake e' andato perso.
# Il rapporto fra le medie nelle due sfere e' una misura di quanto l'uptake e'
# concentrato, che e' il segno "virgola contro punto" espresso senza soglia.
RADIUS_CORE_MM = 10.0
RADIUS_FULL_MM = 22.0

# Mean shift: raggio della finestra e numero di iterazioni. Poche iterazioni
# bastano perche' il picco parte gia' dentro lo striato; servono solo a
# spostare l'ancora dall'argmax (rumoroso) al baricentro (stabile).
MEANSHIFT_RADIUS_MM = 16.0
MEANSHIFT_ITERATIONS = 3

# Profilo antero-posteriore: semilarghezza dei bin e mezza estensione totale.
# 3 mm di semilarghezza a 2 mm di voxel significa bin da 6 mm, cioe' tre fette:
# sotto la risoluzione effettiva della SPECT non ci sarebbe informazione in
# piu', solo rumore.
PROFILE_HALF_WIDTH_MM = 3.0
PROFILE_EXTENT_MM = 18.0
# Il profilo si campiona dentro un cilindro attorno all'asse antero-posteriore:
# senza questo vincolo i bin anteriori e posteriori raccoglierebbero tessuto
# via via piu' lontano dallo striato e il profilo misurerebbe la forma della
# sfera invece di quella della struttura.
PROFILE_LATERAL_MM = 14.0
PROFILE_AXIAL_MM = 10.0

# Posizioni dei bin, in millimetri dall'ancora lungo la direzione di
# campionamento. Negativo = posteriore. Sono esattamente i centri che
# l'implementazione precedente calcolava con arange, ma scritti per esteso:
# adesso un bin ha un nome proprio (`_bin_m15_ratio`) e un nome ha significato
# solo se la posizione e' fissa e uguale in ogni esame.
PROFILE_OFFSETS_MM: tuple[float, ...] = (-15.0, -9.0, -3.0, 3.0, 9.0, 15.0)

# Numero minimo di voxel perche' la media di un bin sia una media e non rumore.
PROFILE_MIN_VOXELS = 10

# Asse lungo dello striato nel piano assiale, in gradi dall'asse antero-
# posteriore. Il caudato e' anteromediale e il putamen posterolaterale, quindi
# la struttura e' allungata lungo una direzione obliqua: campionando lungo y i
# bin a +-15 mm tagliano gli angoli della struttura, campionando lungo l'asse
# vero restano dentro.
#
# E' una COSTANTE FISICA, come RADIUS_CORE_MM, e non un asse ricavato dai dati.
# La distinzione e' la stessa che questo modulo fa gia' per l'ancora, ma qui e'
# piu' severa. Il modo di fallire dell'ancora e' una traslazione: limitata dalla
# finestra del mean shift e uguale su entrambe le estremita' del profilo, quindi
# attenua il gradiente senza invertirlo. Il modo di fallire di un asse ricavato
# dalla PCA sarebbe una rotazione, che non e' limitata da nulla - e in uno
# striato degenerato a punto i due autovalori nel piano assiale sono quasi
# uguali, quindi l'autovettore e' deciso dal rumore. Il rumore di ricostruzione
# e' una proprieta' del centro: si otterrebbe una misura il cui asse di
# campionamento e' scelto dallo scanner, proprio nella classe che serve
# riconoscere. L'angolo misurato si espone come diagnostica (`ap_axis_angle_deg`)
# e non entra nel modello.
OBLIQUE_ANGLE_DEG = 30.0

# Sotto questo valore il logaritmo non si prende: e' il pavimento che impedisce
# a un bin vuoto di produrre -inf e propagarlo a tutta la predizione.
LOG_FLOOR = 1e-6

NAN = float("nan")

# Direzioni lungo cui si campiona il profilo: angolo, prefisso, se calcolare i
# rapporti di bin, se usare la formulazione storica.
#
# **Solo `ap` e' attiva.** Le altre due sono state misurate su otto archi appaiati
# e respinte entrambe - obliquo +0,0048, medio-laterale +0,0064, zero seed su otto
# (verdetto consolidato in `docs/DECISIONS.md`) - e calcolarle costava
# ~12 minuti sui 1362 esami, cioe' un terzo del tempo di estrazione speso per due
# verdetti gia' dati. Con 45 giorni davanti, il tempo di estrazione determina
# quante ipotesi si riescono a provare, ed e' diventato un vincolo prima che un
# fastidio.
#
# Restano elencate qui invece di essere cancellate perche' il verdetto sia
# riapribile con **una riga** invece che con un'archeologia in git: si sposta una
# voce da REJECTED_PROFILE_DIRECTIONS a PROFILE_DIRECTIONS, si ri-estrae, e
# `include` di 05_compare puo' riprovarle. E' lo stesso principio per cui i venti
# candidati respinti stanno in EXCLUDED_FROM_MODEL con il costo accanto, applicato
# un livello piu' in basso - dove pero' si paga in minuti e non in colonne.
PROFILE_DIRECTIONS: tuple[tuple[float, str, bool, bool], ...] = (
    (0.0, "ap", True, True),
)
REJECTED_PROFILE_DIRECTIONS: tuple[tuple[float, str, bool, bool], ...] = (
    (OBLIQUE_ANGLE_DEG, "obl", False, False),
    (90.0, "ml", False, False),
)

# Quale direzione produce quali chiavi. Mappa esplicita, e non un filtro sul
# prefisso, per una ragione precisa: `ml_extent_mm` comincia per "ml_" ed e' una
# misura di taglia, non un profilo medio-laterale. Un filtro per prefisso la
# cancellerebbe insieme al profilo, e il difetto si presenterebbe come una colonna
# mancante a estrazione finita - tre quarti d'ora dopo.
_KEYS_OF_DIRECTION: dict[str, tuple[str, ...]] = {
    "ap": ("ap_log_slope", "ap_tail_ratio", "ap_bin_m15_ratio",
           "ap_bin_m09_ratio", "ap_bin_p09_ratio", "ap_bin_p15_ratio"),
    "obl": ("obl_log_slope", "obl_tail_ratio"),
    "ml": ("ml_log_slope", "ml_tail_ratio"),
}

# Chiavi prodotte per emisfero. Esposte perche' `features.py` deve poter
# costruire il vettore completo anche quando un emisfero non e' stato
# localizzato, e quindi deve conoscere i nomi senza poterli calcolare.
_ALL_MORPHOLOGY_KEYS: tuple[str, ...] = (
    "sbr_core",
    "sbr_full",
    "concentration",
    "center_shift_mm",
    "ap_extent_mm",
    "ml_extent_mm",
    "si_extent_mm",
    "ap_skew",
    "shape_corr",
    "ap_slope",
    "post_ant_ratio",
    "uptake_full",
    "sbr_core_slab",
    "sbr_full_slab",
    # Profilo lungo y (invariato), piu' le due sintesi nuove e i rapporti di bin.
    "ap_log_slope",
    "ap_tail_ratio",
    "ap_bin_m15_ratio",
    "ap_bin_m09_ratio",
    "ap_bin_p09_ratio",
    "ap_bin_p15_ratio",
    # Stesso profilo campionato lungo l'asse obliquo dello striato.
    "obl_log_slope",
    "obl_tail_ratio",
    # Lungo l'asse medio-laterale. Calcolato perche' non costa nulla nella
    # stessa passata, ma atteso contaminato dal centro: dal centroide del
    # putamen a quello del caudato ci sono ~12-14 mm, poco piu' di un elemento
    # di risoluzione SPECT, cioe' la stessa geometria di `concentration`, che
    # misura eta quadro 0,32 - quanto l'SBR grezzo - benche' sia anch'essa un
    # rapporto di uptake. Un contrasto su una base cosi' corta misura la
    # sfocatura della ricostruzione, non l'anatomia.
    "ml_log_slope",
    "ml_tail_ratio",
    # Diagnostica sull'asse, esclusa dal modello: vedi OBLIQUE_ANGLE_DEG.
    "ap_axis_angle_deg",
)

# Le chiavi effettivamente prodotte, dopo aver tolto quelle delle direzioni
# disattivate. L'ordine relativo di cio' che resta e' invariato, e non e' un
# dettaglio: fissa l'ordine delle colonne nel CSV, che entra nel
# sottocampionamento di LightGBM. Togliere voci dalla coda e' neutro; spostarle
# no. Le quattro rimosse sono comunque tutte in EXCLUDED_FROM_MODEL, quindi le 44
# colonne del modello non si accorgono di niente - ed e' una proprieta' che
# `tests/test_directions.py` verifica invece di darla per buona.
INACTIVE_DIRECTION_KEYS: frozenset[str] = frozenset(
    key
    for prefix, keys in _KEYS_OF_DIRECTION.items()
    if prefix not in {p for _, p, _, _ in PROFILE_DIRECTIONS}
    for key in keys
)

MORPHOLOGY_KEYS: tuple[str, ...] = tuple(
    key for key in _ALL_MORPHOLOGY_KEYS if key not in INACTIVE_DIRECTION_KEYS
)


def _normalized(signal: float, reference: float) -> float:
    """SBR = (segnale - riferimento) / riferimento, con la stessa guardia di striatum.py."""
    if not np.isfinite(signal) or not np.isfinite(reference) or reference <= 0:
        return NAN
    return float((signal - reference) / reference)


def _safe_ratio(numerator: float, denominator: float, floor: float = 1e-8) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return NAN
    if abs(denominator) < floor:
        return NAN
    return float(numerator / denominator)


def _local_box(
    data: np.ndarray, center: tuple[int, int, int], half_mm: float, spacing_mm: float
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Sottovolume attorno al centro, con le coordinate globali dei suoi voxel.

    Lavorare su una scatola locale invece che sul volume intero non e'
    un'ottimizzazione cosmetica: le griglie di coordinate a piena risoluzione
    sono la voce di costo dominante di questa famiglia di misure, e qui ne
    servirebbero diverse per emisfero.
    """
    half = max(1, int(np.ceil(half_mm / spacing_mm)))
    slices = tuple(
        slice(max(0, int(c) - half), min(int(dim), int(c) + half + 1))
        for c, dim in zip(center, data.shape)
    )
    grids = np.meshgrid(
        *(np.arange(s.start, s.stop, dtype=np.float64) for s in slices), indexing="ij"
    )
    return data[slices], tuple(grids)


def _specific_binding(values: np.ndarray, reference: float) -> np.ndarray:
    """Legame specifico: uptake meno fondo, mai negativo.

    Pesare i momenti con il legame specifico invece che con l'uptake grezzo
    conta: il fondo e' una costante additiva che trascina ogni baricentro verso
    il centro della finestra e appiattisce ogni momento verso quello di una
    sfera uniforme. Sottrarlo lascia solo la struttura.
    """
    return np.clip(values - reference, 0.0, None)


def _refine_center(
    data: np.ndarray,
    peak_index: tuple[int, int, int],
    reference: float,
    spacing_mm: float,
) -> np.ndarray:
    """Sposta l'ancora dall'argmax al baricentro del legame specifico.

    L'argmax di un volume SPECT e' un singolo voxel e si muove di parecchi
    millimetri fra due ricostruzioni dello stesso paziente. Il baricentro
    dell'uptake in una finestra fissa e' invece stabile, ed e' l'unico punto
    dell'ancora che serve: tutte le misure sotto sono relative a esso.
    """
    center = np.array(peak_index, dtype=np.float64)
    radius_vox = MEANSHIFT_RADIUS_MM / spacing_mm

    for _ in range(MEANSHIFT_ITERATIONS):
        box, grids = _local_box(
            data, tuple(int(round(c)) for c in center), MEANSHIFT_RADIUS_MM, spacing_mm
        )
        distance_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
        inside = distance_sq <= radius_vox**2
        if not inside.any():
            break
        weights = _specific_binding(box[inside], reference)
        total = float(weights.sum())
        if total <= 0:
            break
        center = np.array(
            [float((g[inside] * weights).sum() / total) for g in grids],
            dtype=np.float64,
        )

    return center


def _sphere_mean(
    box: np.ndarray, grids: tuple[np.ndarray, ...], center: np.ndarray, radius_mm: float,
    spacing_mm: float,
) -> float:
    radius_vox = radius_mm / spacing_mm
    distance_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
    inside = distance_sq <= radius_vox**2
    if not inside.any():
        return NAN
    return float(box[inside].mean())


def _weighted_moments(
    box: np.ndarray,
    grids: tuple[np.ndarray, ...],
    center: np.ndarray,
    reference: float,
    spacing_mm: float,
    midline_x: float,
    anterior_sign: int,
) -> dict[str, float]:
    """Momenti del legame specifico dentro la sfera grande.

    Tutte le grandezze sono normalizzate o hanno dimensione di lunghezza, quindi
    nessuna dipende dal guadagno dello scanner: moltiplicare i conteggi per un
    fattore arbitrario moltiplica i pesi per lo stesso fattore e li lascia
    identiche. E' la stessa invarianza che rende utilizzabile l'SBR, applicata
    alla forma invece che all'intensita'.
    """
    radius_vox = RADIUS_FULL_MM / spacing_mm
    distance_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
    inside = distance_sq <= radius_vox**2

    empty = {
        "ap_extent_mm": NAN, "ml_extent_mm": NAN, "si_extent_mm": NAN,
        "ap_skew": NAN, "shape_corr": NAN, "ap_axis_angle_deg": NAN,
    }
    if inside.sum() < 20:
        return empty

    weights = _specific_binding(box[inside], reference)
    total = float(weights.sum())
    if total <= 0:
        return empty
    weights = weights / total

    # Coordinate in millimetri rispetto al centro raffinato. Il segno anteriore
    # e' applicato qui una volta sola, cosi' l'asse y punta sempre verso il
    # caudato indipendentemente da come e' orientato il volume in ingresso.
    x = (grids[0][inside] - center[0]) * spacing_mm
    y = anterior_sign * (grids[1][inside] - center[1]) * spacing_mm
    z = (grids[2][inside] - center[2]) * spacing_mm
    lateral = np.abs(grids[0][inside] - midline_x) * spacing_mm

    def moment(values: np.ndarray) -> tuple[float, float]:
        mean = float((weights * values).sum())
        var = float((weights * (values - mean) ** 2).sum())
        return mean, var

    _, var_x = moment(x)
    mean_y, var_y = moment(y)
    _, var_z = moment(z)

    sd_y = float(np.sqrt(var_y))
    if sd_y > 1e-6:
        skew = float((weights * ((y - mean_y) / sd_y) ** 3).sum())
    else:
        skew = NAN

    # Stessa grandezza di `localization_confidence`, ma su una regione a raggio
    # fisso invece che sulla VOI a soglia. Se le due concordano, la feature piu'
    # importante del modello non era un artefatto della soglia; se divergono,
    # questa e' la versione da tenere perche' e' l'unica non contaminata.
    mean_lat = float((weights * lateral).sum())
    cov = float((weights * (y - mean_y) * (lateral - mean_lat)).sum())
    sd_lat = float(np.sqrt((weights * (lateral - mean_lat) ** 2).sum()))
    shape_corr = -cov / (sd_y * sd_lat) if sd_y > 1e-6 and sd_lat > 1e-6 else NAN

    return {
        "ap_extent_mm": sd_y,
        "ml_extent_mm": float(np.sqrt(var_x)),
        "si_extent_mm": float(np.sqrt(var_z)),
        "ap_skew": skew,
        "shape_corr": float(shape_corr) if np.isfinite(shape_corr) else NAN,
        "ap_axis_angle_deg": _principal_axis_angle(
            x, y, weights, midline_x, center[0]
        ),
    }


def _principal_axis_angle(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    midline_x: float,
    center_x: float,
) -> float:
    """Angolo dell'asse principale nel piano assiale, in gradi dall'asse AP.

    DIAGNOSTICA, non feature. Esiste per rispondere con i dati alla domanda se
    OBLIQUE_ANGLE_DEG sia l'angolo giusto, invece che per asserzione: nella
    classe normale l'atteso e' una mediana attorno ai 25-35 gradi, e nella
    classe patologica una dispersione verso l'uniforme man mano che i due
    autovalori nel piano si avvicinano e l'autovettore smette di significare
    qualcosa.

    Quella dispersione e' anche il motivo per cui l'angolo NON entra nel
    modello, oltre alla ragione gia' registrata (sarebbe la terza misura della
    stessa grandezza di `shape_corr` e `localization_confidence`, e
    `min_shape_corr` ha prezzato quella trappola a +0,0023 su 0 seed su 6).

    Positivo significa che l'asse si sposta verso la linea mediana andando in
    avanti, cioe' l'anatomia attesa: caudato anteromediale, putamen
    posterolaterale.
    """
    medial_sign = 1.0 if center_x < midline_x else -1.0
    x_medial = medial_sign * x

    mean_x = float((weights * x_medial).sum())
    mean_y = float((weights * y).sum())
    var_x = float((weights * (x_medial - mean_x) ** 2).sum())
    var_y = float((weights * (y - mean_y) ** 2).sum())
    cov_xy = float((weights * (x_medial - mean_x) * (y - mean_y)).sum())

    if var_x <= 1e-9 or var_y <= 1e-9:
        return NAN

    eigenvalues, eigenvectors = np.linalg.eigh(
        np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)
    )
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    # L'autovettore e' definito a meno del segno: si orienta verso l'anteriore
    # cosi' che l'angolo abbia sempre lo stesso verso di lettura.
    if principal[1] < 0:
        principal = -principal
    return float(np.degrees(np.arctan2(principal[0], principal[1])))


def _axial_direction(
    angle_deg: float, anterior_sign: int, medial_sign: int
) -> np.ndarray:
    """Versore nel piano assiale, a `angle_deg` dall'asse antero-posteriore.

    Punta sempre dalla struttura vulnerabile verso quella risparmiata: dal
    putamen posterolaterale al caudato anteromediale. Serve perche' il segno
    della pendenza abbia lo stesso significato clinico in ogni esame e su
    entrambi i lati, qualunque sia l'orientamento del volume in ingresso.

    `medial_sign` viene dalla posizione dell'ancora rispetto alla linea mediana,
    e la linea mediana e' il centroide della maschera cerebrale - `find_midline`
    e' scritta apposta per non dipendere dal segnale patologico. Quindi nulla di
    derivato dall'uptake entra nella geometria del campionamento.
    """
    theta = np.deg2rad(angle_deg)
    return np.array(
        [medial_sign * np.sin(theta), anterior_sign * np.cos(theta), 0.0],
        dtype=np.float64,
    )


def _profile_bins(
    box: np.ndarray,
    grids: tuple[np.ndarray, ...],
    center: np.ndarray,
    spacing_mm: float,
    direction: np.ndarray,
) -> np.ndarray:
    """Uptake medio GREZZO nei sei bin da 6 mm lungo `direction`.

    E' il rapporto putamen/caudato rifatto in modo che possa variare. Il taglio
    non sta piu' al centroide della VOI - che si sposta con la patologia e
    annulla la misura - ma a offset costanti in millimetri dall'ancora, dentro
    un cilindro di sezione fissa. Un bin posteriore resta posteriore anche
    quando li' non e' rimasto uptake, e la sua media crolla invece di uscire
    dalla regione.

    Restituisce sempre sei valori, con NaN nei bin che non raccolgono almeno
    `PROFILE_MIN_VOXELS` voxel. La lunghezza fissa non e' cosmetica: e' cio' che
    permette di dare un nome proprio al singolo bin, perche' `_bin_m15_ratio`
    significa qualcosa solo se in ogni esame indica la stessa posizione fisica.

    Il cilindro e' quello di sempre - 14 mm di raggio perpendicolare, 10 mm
    assiali - ma il raggio si misura ora rispetto all'asse di campionamento
    invece che rispetto a x. Per `direction = (0, +-1, 0)` le due definizioni
    coincidono voxel per voxel, ed e' la condizione che tiene `ap_slope` e
    `post_ant_ratio` identiche al valore che avevano prima di questo modulo
    diventare direzionale. C'e' un test su phantom che lo verifica.
    """
    offsets = np.asarray(PROFILE_OFFSETS_MM, dtype=np.float64)
    means = np.full(len(offsets), NAN, dtype=np.float64)

    delta = [(grids[i] - center[i]) * spacing_mm for i in range(3)]
    # Coordinata lungo l'asse di campionamento.
    along = sum(d * u for d, u in zip(delta, direction))
    # Distanza dall'asse dentro il piano assiale: il perpendicolare a
    # `direction` nel piano e' (-uy, ux, 0).
    perpendicular = np.abs(-delta[0] * direction[1] + delta[1] * direction[0])
    axial = np.abs(delta[2])

    cylinder = (perpendicular <= PROFILE_LATERAL_MM) & (axial <= PROFILE_AXIAL_MM)
    if not cylinder.any():
        return means

    for i, offset in enumerate(offsets):
        selected = cylinder & (np.abs(along - offset) <= PROFILE_HALF_WIDTH_MM)
        if selected.sum() < PROFILE_MIN_VOXELS:
            continue
        means[i] = float(box[selected].mean())

    return means


def _profile_summaries(
    means: np.ndarray,
    reference: float,
    prefix: str,
    with_bins: bool = False,
    legacy: bool = False,
) -> dict[str, float]:
    """Sintesi scalari di un profilo, tutte invarianti al guadagno per costruzione.

    `{prefix}_log_slope` - pendenza di log(uptake grezzo) per millimetro. E' la
    grandezza migliore di tutto il blocco, e la ragione e' aritmetica. Il modello
    di contaminazione di questo progetto e' un fattore MOLTIPLICATIVO del centro:
    misurato sui gruppi grandi, il livello assoluto dell'SBR varia del 50% fra
    centri mentre il rapporto fra le classi resta fra 1,65 e 2,13. In logaritmo
    quel fattore diventa `log(gamma)`, una costante additiva **uguale in tutti i
    bin**, quindi sparisce esattamente dalla pendenza. E poiche'
    log(m / riferimento) = log(m) - log(riferimento) e il riferimento non dipende
    dal bin, la pendenza non dipende nemmeno dalla stima del riferimento - cioe'
    ritira dal calcolo l'intera fonte di errore dipendente dal campo di vista che
    il resto della pipeline combatte con la fetta striatale.

    La pendenza precedente, `ap_slope`, era in unita' di SBR per millimetro e
    quindi il fattore del centro ci restava dentro: misurato, eta quadro 0,087
    contro 0,014 di `post_ant_ratio`, sei volte tanto, ed e' una feature fra le
    prime dieci per importanza. Non e' una svista isolata: questo stesso modulo
    argomenta gia', due funzioni piu' sotto, che il rapporto va preso fra uptake
    grezzi e non fra SBR. La pendenza accanto faceva l'opposto.

    `{prefix}_tail_ratio` - rapporto fra la media dei due bin posteriori e quella
    dei due anteriori. Rispetto a `post_ant_ratio` esclude i due bin centrali,
    che stanno a cavallo dell'ancora e sono lo stesso tessuto da entrambe le
    parti: diluiscono il contrasto senza aggiungere baseline.

    `with_bins` espone i quattro bin esterni come rapporto sulla coppia centrale.
    Quattro e non sei: i due bin centrali SONO il denominatore, esporli
    significherebbe esporre la costante 1. E il denominatore e' la coppia
    centrale e non il bin caudale perche' in malattia avanzata l'ancora migra in
    avanti - con un denominatore anteriore il rapporto schizzerebbe verso
    "normale" proprio nei casi gravi, che e' il modo di sbagliare peggiore
    possibile. La coppia centrale invece straddia l'ancora, e' la zona a miglior
    rapporto segnale-rumore del profilo, ed e' l'unica parte della geometria
    stabile per costruzione.

    `legacy` riemette `ap_slope` e `post_ant_ratio` calcolate esattamente come
    prima. Serve al confronto appaiato: se la baseline si muovesse, ogni Delta
    misurato contro di essa sarebbe contro un riferimento diverso da quello
    pubblicato, e l'intero esperimento non direbbe piu' nulla.
    """
    offsets = np.asarray(PROFILE_OFFSETS_MM, dtype=np.float64)
    finite = np.isfinite(means)

    out: dict[str, float] = {f"{prefix}_log_slope": NAN, f"{prefix}_tail_ratio": NAN}
    if with_bins:
        out.update({
            f"{prefix}_bin_m15_ratio": NAN,
            f"{prefix}_bin_m09_ratio": NAN,
            f"{prefix}_bin_p09_ratio": NAN,
            f"{prefix}_bin_p15_ratio": NAN,
        })
    if legacy:
        out.update({"ap_slope": NAN, "post_ant_ratio": NAN})

    if finite.sum() < 4:
        return out

    positive = finite & (means > 0)
    if positive.sum() >= 4:
        out[f"{prefix}_log_slope"] = float(
            np.polyfit(
                offsets[positive], np.log(np.clip(means[positive], LOG_FLOOR, None)), 1
            )[0]
        )

    posterior = means[:2]
    anterior = means[-2:]
    if np.isfinite(posterior).any() and np.isfinite(anterior).any():
        out[f"{prefix}_tail_ratio"] = _safe_ratio(
            float(np.nanmean(posterior)), float(np.nanmean(anterior))
        )

    if with_bins:
        core = float(np.nanmean(means[2:4])) if np.isfinite(means[2:4]).any() else NAN
        for name, index in (("m15", 0), ("m09", 1), ("p09", 4), ("p15", 5)):
            out[f"{prefix}_bin_{name}_ratio"] = _safe_ratio(float(means[index]), core)

    if legacy:
        # Ricostruzione fedele del calcolo precedente: si lavora sui soli bin
        # popolati, come faceva il `continue`, e si esce se sono meno di quattro
        # o se il riferimento non e' utilizzabile.
        positions_legacy = offsets[finite]
        means_legacy = means[finite]
        if reference > 0:
            sbr = (means_legacy - reference) / reference
            if np.isfinite(sbr).all():
                out["ap_slope"] = float(np.polyfit(positions_legacy, sbr, 1)[0])
                back = means_legacy[positions_legacy < 0]
                front = means_legacy[positions_legacy > 0]
                if back.size and front.size:
                    # Rapporto fra uptake grezzi, non fra SBR: entrambi sono
                    # positivi per costruzione, mentre una differenza di SBR ha
                    # al denominatore il legame specifico totale, che tende a
                    # zero proprio nei casi gravi. E' lo stesso
                    # malcondizionamento che tiene `asym_sbr` a 0,77 di AUC.
                    out["post_ant_ratio"] = _safe_ratio(
                        float(back.mean()), float(front.mean())
                    )

    return out


def _directional_profile(
    box: np.ndarray,
    grids: tuple[np.ndarray, ...],
    center: np.ndarray,
    reference: float,
    spacing_mm: float,
    angle_deg: float,
    anterior_sign: int,
    medial_sign: int,
    prefix: str,
    with_bins: bool = False,
    legacy: bool = False,
) -> dict[str, float]:
    """Profilo campionato lungo una direzione, e le sue sintesi.

    Una terza sintesi, la posizione del bin massimo, e' stata provata e tolta:
    su 140 esami veri e' risultata costante. Il motivo e' strutturale e vale la
    pena registrarlo, perche' riguarda ogni misura ancorata a un baricentro:
    l'origine del profilo E' il baricentro dell'uptake, quindi il massimo ci
    cade sopra quasi per definizione e la quantizzazione in bin da 6 mm
    cancella il residuo. Sui phantom sembrava viva (si spostava di 12 mm fra
    normale e patologico) perche' li' la virgola e' molto piu' marcata del vero.
    """
    direction = _axial_direction(angle_deg, anterior_sign, medial_sign)
    means = _profile_bins(box, grids, center, spacing_mm, direction)
    return _profile_summaries(
        means, reference, prefix, with_bins=with_bins, legacy=legacy
    )


def hemisphere_morphology(
    data: np.ndarray,
    peak_index: tuple[int, int, int] | None,
    reference: float,
    spacing_mm: float,
    midline_x: float,
    anterior_sign: int,
    reference_slab: float = NAN,
) -> dict[str, float]:
    """Blocco morfologico completo per un emisfero, senza alcuna soglia."""
    if peak_index is None or not np.isfinite(reference) or reference <= 0:
        return dict.fromkeys(MORPHOLOGY_KEYS, NAN)

    center = _refine_center(data, peak_index, reference, spacing_mm)
    box, grids = _local_box(
        data, tuple(int(round(c)) for c in center), RADIUS_FULL_MM + 4.0, spacing_mm
    )

    core = _sphere_mean(box, grids, center, RADIUS_CORE_MM, spacing_mm)
    full = _sphere_mean(box, grids, center, RADIUS_FULL_MM, spacing_mm)

    result = {
        "sbr_core": _normalized(core, reference),
        "sbr_full": _normalized(full, reference),
        # Rapporto fra uptake grezzi: quanto l'uptake e' concentrato nel nucleo
        # rispetto all'intera struttura. Vale 1 per una distribuzione uniforme e
        # cresce quanto piu' la virgola si e' ridotta a un punto.
        "concentration": _safe_ratio(core, full),
        "center_shift_mm": float(
            np.linalg.norm(center - np.array(peak_index, dtype=np.float64)) * spacing_mm
        ),
        # Serve agli indici di asimmetria bilaterali: e' una quantita' positiva,
        # quindi il rapporto fra i due lati resta ben condizionato anche quando
        # il legame specifico e' quasi nullo su entrambi.
        "uptake_full": full,
        # Gli stessi due uptake normalizzati sul riferimento della sola fetta
        # striatale invece che su tutto il tessuto lontano. Coppia di controllo:
        # se generalizzano meglio fra centri, il riferimento globale stava
        # portandosi dentro il campo di vista del protocollo.
        "sbr_core_slab": _normalized(core, reference_slab),
        "sbr_full_slab": _normalized(full, reference_slab),
    }
    result.update(
        _weighted_moments(
            box, grids, center, reference, spacing_mm, midline_x, anterior_sign
        )
    )

    # Verso della linea mediana, dall'ancora. Serve a orientare i profili obliqui
    # cosi' che "mediale" significhi la stessa cosa sui due lati; per il profilo
    # puramente antero-posteriore non entra nel calcolo (sin 0 = 0).
    medial_sign = 1 if center[0] < midline_x else -1

    # Le direzioni attive, calcolate nella stessa passata. Quale sopravvivesse era
    # una domanda empirica e ha avuto risposta: vedi PROFILE_DIRECTIONS, dove sta
    # anche come riaprirla.
    for angle, prefix, with_bins, legacy in PROFILE_DIRECTIONS:
        result.update(
            _directional_profile(
                box, grids, center, reference, spacing_mm,
                angle_deg=angle,
                anterior_sign=anterior_sign,
                medial_sign=medial_sign,
                prefix=prefix,
                with_bins=with_bins,
                legacy=legacy,
            )
        )
    return result
