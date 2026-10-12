"""Localizzazione dello striato e calcolo dello Striatal Binding Ratio (SBR).

Perche' questo approccio invece di dare il volume grezzo a una CNN
--------------------------------------------------------------------
La lettura clinica di una DaTscan non si basa sull'intensita' assoluta, che
dipende da scanner, dose iniettata e tempo di acquisizione, ma su un rapporto:
l'uptake nello striato relativo a una regione di riferimento priva di trasportatori
della dopamina. Questo rapporto e' lo Striatal Binding Ratio.

    SBR = (media_striato - media_riferimento) / media_riferimento

Poiche' e' un rapporto, e' intrinsecamente invariante rispetto alla calibrazione
dello scanner. Con dati provenienti da 10 centri diversi, questa invarianza vale
piu' di qualunque architettura: e' esattamente l'asse su cui il modello deve
generalizzare.

Il pattern diagnostico
----------------------
Normale: uptake simmetrico a forma di virgola in entrambi gli emisferi.
Parkinsoniano: perdita che inizia dal putamen posteriore, tipicamente asimmetrica,
e la virgola degrada a punto. Quindi le feature che contano sono il rapporto
putamen/caudato e l'asimmetria destra/sinistra, non l'intensita' assoluta.

Convenzione degli assi
----------------------
Tutti i volumi arrivano in orientamento canonico RAS:
    asse 0 (x): sinistra -> destra
    asse 1 (y): posteriore -> anteriore   (caudato anteriore, putamen posteriore)
    asse 2 (z): inferiore -> superiore

Questo modulo usa solo numpy, cosi' e' collaudabile su phantom sintetici senza
dipendenze pesanti.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Frazione del massimo emisferico usata come soglia per la VOI striatale.
# Il 50% del massimo e' la convenzione nella letteratura sulla quantificazione
# DaT: e' un compromesso stabile tra includere tessuto non striatale (soglia
# troppo bassa) e collassare su pochi voxel rumorosi (soglia troppo alta).
VOI_THRESHOLD_FRACTION = 0.5

# Raggio massimo, in mm, entro cui la VOI puo' estendersi dal picco emisferico.
# Lo striato umano si estende per circa 30 mm; 25 mm evita di agganciare
# strutture adiacenti mantenendo l'intera struttura.
VOI_RADIUS_MM = 25.0

# Distanza minima dai picchi striatali oltre la quale il tessuto e' considerato
# regione di riferimento (privo di trasportatori della dopamina).
REFERENCE_MIN_DISTANCE_MM = 40.0

# Sigma del filtro gaussiano, in millimetri. Sotto la risoluzione effettiva
# della SPECT (~8-10 mm FWHM), quindi stabilizza la ricerca del picco e la
# soglia senza cancellare struttura reale.
BRAIN_SIGMA_MM = 4.0


@dataclass(frozen=True)
class Hemisphere:
    """Misure estratte da un singolo emisfero striatale."""

    peak_value: float
    voi_mask: np.ndarray
    voi_mean: float
    voi_volume_mm3: float
    centroid: np.ndarray  # (x, y, z) in voxel
    caudate_mean: float  # meta' anteriore della VOI
    putamen_mean: float  # meta' posteriore della VOI
    # Indice del voxel di picco, in coordinate del volume. Serve come ancora
    # alle misure di morphology.py, che sono definite su geometrie a raggio
    # fisso e quindi non possono partire dalla VOI, che una soglia la ha.
    peak_index: tuple[int, int, int] | None = None


def gaussian_blur(data: np.ndarray, sigma_vox: float | tuple[float, ...]) -> np.ndarray:
    """Filtro gaussiano separabile.

    `sigma_vox` puo' essere scalare o una sigma per asse: serve nel caso di
    voxel anisotropi, dove una sigma fisica di 4 mm corrisponde a un numero di
    voxel diverso lungo ciascun asse.

    Usa scipy quando e' disponibile - e' circa 5 volte piu' veloce, e su questa
    pipeline il filtraggio e' la voce di costo dominante. La versione in numpy
    puro resta come fallback perche' tutto il modulo e' deliberatamente
    collaudabile su phantom senza dipendenze pesanti; le due implementazioni
    concordano entro il 2 per mille, verificato sui volumi reali.
    """
    sigmas = (
        (float(sigma_vox),) * data.ndim
        if np.isscalar(sigma_vox)
        else tuple(float(s) for s in sigma_vox)  # type: ignore[union-attr]
    )
    if all(s <= 0 for s in sigmas):
        return data

    try:
        from scipy.ndimage import gaussian_filter

        # mode="nearest" replica il bordo, come il np.pad(mode="edge") sotto:
        # con lo zero-padding i voxel di bordo verrebbero artificialmente
        # attenuati, e la soglia calcolata sul volume filtrato ne risentirebbe.
        return gaussian_filter(
            data.astype(np.float32, copy=False), sigmas, mode="nearest"
        ).astype(np.float32)
    except ImportError:
        pass

    out = data.astype(np.float32, copy=True)
    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        radius = max(1, int(round(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(
            out,
            [(radius, radius) if i == axis else (0, 0) for i in range(out.ndim)],
            mode="edge",
        )
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis, padded
        )
    return out.astype(np.float32)


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Soglia di Otsu: massimizza la varianza inter-classe.

    Preferita a una soglia relativa a un percentile fisso perche' non introduce
    costanti arbitrarie e si adatta da sola alla proporzione tra fondo e tessuto,
    che varia con il campo di vista e quindi con il centro di provenienza.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        return vmin

    hist, edges = np.histogram(finite, bins=n_bins, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = hist.astype(np.float64)

    weight_low = np.cumsum(hist)
    weight_high = weight_low[-1] - weight_low
    cumulative_mean = np.cumsum(hist * centers)
    total_mean = cumulative_mean[-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_low = cumulative_mean / weight_low
        mean_high = (total_mean - cumulative_mean) / weight_high
        between = weight_low * weight_high * (mean_low - mean_high) ** 2

    between = np.nan_to_num(between, nan=-np.inf)
    return float(centers[int(np.argmax(between))])


def _shift_or(mask: np.ndarray, axis: int, step: int) -> np.ndarray:
    """OR della maschera con se stessa traslata di un voxel, senza wraparound.

    np.roll qui sarebbe sbagliato: farebbe rientrare i voxel dal lato opposto
    del volume, collegando regioni che non si toccano.
    """
    out = np.zeros_like(mask)
    src = [slice(None)] * mask.ndim
    dst = [slice(None)] * mask.ndim
    if step > 0:
        src[axis] = slice(0, -1)
        dst[axis] = slice(1, None)
    else:
        src[axis] = slice(1, None)
        dst[axis] = slice(0, -1)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Componente connessa piu' estesa della maschera.

    Si prende la piu' grande e non quella contenente il voxel piu' brillante
    perche' il voxel piu' brillante non e' sempre striatale: una captazione
    tiroidea o salivare nel campo di vista puo' superare lo striato, e usarla
    come seme farebbe collassare la maschera cerebrale sul focus sbagliato.
    La testa e' invece sempre la struttura piu' voluminosa.
    """
    if not mask.any():
        return mask

    try:
        from scipy.ndimage import label as _label

        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, :] = structure[1, :, 1] = structure[:, 1, 1] = True
        labels, n = _label(mask, structure=structure)
        if n <= 1:
            return mask
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        return labels == int(np.argmax(counts))
    except ImportError:
        pass

    # Fallback senza scipy: si semina dal voxel della maschera piu' vicino al
    # suo centroide. La testa domina in volume, quindi il centroide vi cade
    # dentro anche in presenza di focolai spuri.
    coords = np.array(np.nonzero(mask), dtype=np.float64)
    centroid = coords.mean(axis=1)
    distances = ((coords - centroid[:, None]) ** 2).sum(axis=0)
    nearest = coords[:, int(np.argmin(distances))].astype(int)
    return connected_component(mask, tuple(nearest))


def connected_component(mask: np.ndarray, seed: tuple[int, ...]) -> np.ndarray:
    """Componente connessa della maschera che contiene `seed`.

    Preferisce scipy quando disponibile (molto piu' veloce sui volumi reali),
    con fallback a un flood fill iterativo in numpy puro per gli ambienti di
    test senza scipy. I due percorsi danno lo stesso risultato: connettivita' a
    6 vicini.
    """
    if not mask[seed]:
        return np.zeros_like(mask)

    try:
        from scipy.ndimage import label as _label

        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, :] = structure[1, :, 1] = structure[:, 1, 1] = True
        labels, _ = _label(mask, structure=structure)
        return labels == labels[seed]
    except ImportError:
        pass

    current = np.zeros_like(mask)
    current[seed] = True
    while True:
        grown = current.copy()
        for axis in range(mask.ndim):
            grown |= _shift_or(current, axis, 1)
            grown |= _shift_or(current, axis, -1)
        grown &= mask
        if int(grown.sum()) == int(current.sum()):
            return current
        current = grown


def brain_mask(
    data: np.ndarray, spacing_mm: float | tuple[float, ...] = 2.0
) -> np.ndarray:
    """Maschera grossolana della testa.

    `spacing_mm` puo' essere scalare o per asse: va calcolata anche in spazio
    nativo, dove i voxel sono anisotropi su alcuni centri.

    Fuori dalla testa i conteggi SPECT sono prossimi a zero, quindi una soglia
    relativa a un percentile alto e' sufficiente e non richiede segmentazione.
    Usa il 99.5esimo percentile invece del massimo per non farsi trascinare da
    singoli voxel caldi.

    La soglia si applica al volume FILTRATO, non a quello grezzo. E' una
    distinzione che conta: il rumore di ricostruzione fuori dalla testa produce
    voxel isolati che superano qualunque soglia ragionevole, entrano nella
    maschera e trascinano verso il basso la stima della regione di riferimento.
    Poiche' l'SBR ha il riferimento al denominatore, un riferimento sottostimato
    gonfia l'SBR di un fattore arbitrario. Filtrare prima di sogliare elimina i
    voxel isolati perche' il rumore si media a zero mentre il tessuto no.
    """
    spacing = np.atleast_1d(np.asarray(spacing_mm, dtype=np.float64))
    if spacing.size == 1:
        spacing = np.repeat(spacing, data.ndim)
    return _mask_from_smoothed(gaussian_blur(data, tuple(BRAIN_SIGMA_MM / spacing)))


def _mask_from_smoothed(smoothed: np.ndarray) -> np.ndarray:
    """Maschera della testa a partire dal volume gia' filtrato.

    Due passaggi:
    1. soglia di Otsu, che separa il fondo dal tessuto;
    2. si tiene solo la componente connessa piu' estesa.

    Il secondo passaggio e' quello che conta davvero. La ricostruzione SPECT
    lascia fuori dalla testa un fondo positivo, non zero; una soglia da sola lo
    include, e quei voxel quasi-vuoti abbassano la mediana del riferimento.
    Poiche' il riferimento sta al denominatore dell'SBR, sottostimarlo gonfia
    l'SBR di un fattore arbitrario e diverso per ogni scanner - esattamente
    l'errore che distruggerebbe la generalizzazione tra centri.
    Si sceglie la componente piu' grande, non quella del voxel piu' brillante:
    un focus extracerebrale intenso puo' superare lo striato, e usarlo come
    seme farebbe collassare la maschera sulla struttura sbagliata.
    """
    if float(smoothed.max()) <= 0:
        return np.zeros(smoothed.shape, dtype=bool)

    thresholded = smoothed > otsu_threshold(smoothed)
    if not thresholded.any():
        return thresholded

    return largest_component(thresholded)


def _coordinate_grids(shape: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    return np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )


def find_midline(mask: np.ndarray) -> float:
    """Stima la coordinata x del piano sagittale mediano.

    Usa il centroide della maschera cerebrale. La testa e' approssimativamente
    simmetrica, quindi il suo centroide cade sulla linea mediana anche quando
    l'uptake striatale e' fortemente asimmetrico: e' importante che la stima
    della linea mediana NON dipenda dal segnale patologico, altrimenti la
    misura di asimmetria si auto-annullerebbe.
    """
    if not mask.any():
        return (mask.shape[0] - 1) / 2.0
    xs = np.nonzero(mask)[0]
    return float(xs.mean())


# Frazione del raggio cerebrale entro cui cercare il picco striatale.
#
# Deliberatamente relativa e non assoluta: un raggio in millimetri fissi non
# ha significato finche' non si sa quanto e' grande la testa nel campo di vista,
# e sbaglia in entrambe le direzioni - troppo largo su acquisizioni strette
# (non filtra nulla), troppo stretto su campi ampi (taglia via lo striato).
#
# Lo striato dista dal centro dell'encefalo circa il 40% del raggio cerebrale.
# 0.55 lascia margine per teste non centrate restando ben dentro rispetto a
# tiroide e ghiandole salivari, che stanno oltre il bordo del cranio.
CENTRAL_SEARCH_FRACTION = 0.55


def _central_region(mask: np.ndarray, spacing_mm: float = 2.0) -> np.ndarray:
    """Intorno del centroide cerebrale in cui puo' trovarsi lo striato."""
    if not mask.any():
        return np.ones(mask.shape, dtype=bool)

    coords = np.array(np.nonzero(mask), dtype=np.float32)
    centroid = coords.mean(axis=1)

    # Raggio efficace della sfera di pari volume: stima robusta della scala
    # cerebrale, insensibile alla forma esatta della maschera.
    n_voxels = float(mask.sum())
    effective_radius = (3.0 * n_voxels / (4.0 * np.pi)) ** (1.0 / 3.0)

    gx, gy, gz = _coordinate_grids(mask.shape)
    distance = np.sqrt(
        (gx - centroid[0]) ** 2 + (gy - centroid[1]) ** 2 + (gz - centroid[2]) ** 2
    )
    return distance <= (CENTRAL_SEARCH_FRACTION * effective_radius)


def _extract_hemisphere(
    data: np.ndarray,
    smoothed: np.ndarray,
    hemi_mask: np.ndarray,
    spacing_mm: float,
) -> Hemisphere | None:
    """Estrae la VOI striatale da un emisfero gia' isolato."""
    if not hemi_mask.any():
        return None

    candidates = np.where(hemi_mask, smoothed, -np.inf)
    peak_flat = int(np.argmax(candidates))
    peak_idx = np.unravel_index(peak_flat, data.shape)
    peak_value = float(smoothed[peak_idx])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return None

    gx, gy, gz = _coordinate_grids(data.shape)
    dist_vox = np.sqrt(
        (gx - peak_idx[0]) ** 2 + (gy - peak_idx[1]) ** 2 + (gz - peak_idx[2]) ** 2
    )
    within_radius = dist_vox <= (VOI_RADIUS_MM / spacing_mm)

    voi = hemi_mask & within_radius & (smoothed >= VOI_THRESHOLD_FRACTION * peak_value)
    if not voi.any():
        return None

    voxel_volume = spacing_mm**3
    values = data[voi]

    coords = np.array(np.nonzero(voi), dtype=np.float32)  # (3, N)
    weights = np.clip(values, 0, None)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        centroid = coords.mean(axis=1)
    else:
        centroid = (coords * weights).sum(axis=1) / total_weight

    # Suddivisione caudato / putamen lungo l'asse antero-posteriore.
    # Il caudato e' anteriore (y maggiore), il putamen posteriore.
    # Il taglio e' al centroide y della VOI stessa, quindi si adatta alla
    # posizione individuale senza richiedere un atlante registrato.
    y_coords = coords[1]
    y_split = float(centroid[1])
    anterior = y_coords >= y_split
    posterior = ~anterior

    caudate_mean = float(values[anterior].mean()) if anterior.any() else float("nan")
    putamen_mean = float(values[posterior].mean()) if posterior.any() else float("nan")

    return Hemisphere(
        peak_value=float(data[voi].max()),
        voi_mask=voi,
        voi_mean=float(values.mean()),
        voi_volume_mm3=float(voi.sum()) * voxel_volume,
        centroid=centroid,
        caudate_mean=caudate_mean,
        putamen_mean=putamen_mean,
        peak_index=tuple(int(i) for i in peak_idx),
    )


def infer_anterior_sign(
    voi_mask: np.ndarray, midline_x: float, weights: np.ndarray | None = None
) -> tuple[int, float]:
    """Deduce quale verso dell'asse y sia anteriore, usando solo i dati.

    Perche' serve
    -------------
    Tutto il resto della pipeline si fida di `as_closest_canonical` per sapere
    che +y e' anteriore. Su volumi ben formati e' corretto, ma la SPECT clinica
    ha spesso header approssimativi o affine identita', e in quel caso la
    riorientazione non corregge nulla. La conseguenza sarebbe la peggiore
    possibile: la separazione caudato/putamen si inverte in silenzio su una
    parte del dataset, senza errori, e il segnale si diluisce in un modo che
    non comparirebbe mai guardando la log loss.

    Il criterio anatomico
    ---------------------
    Il caudato e' anteromediale, il putamen posterolaterale. Percorrendo lo
    striato dall'estremita' anteriore a quella posteriore, la distanza dalla
    linea mediana aumenta. Quindi il segno della correlazione tra la coordinata
    y e la distanza dalla linea mediana identifica il verso anteriore, senza
    consultare l'header.

    Returns:
        (segno, forza) dove segno e' +1 se +y e' anteriore e -1 altrimenti.
        La forza e' il valore assoluto della correlazione, in [0, 1]: valori
        bassi significano che la geometria non e' conclusiva su quel volume e
        che la deduzione non va usata.
    """
    coords = np.array(np.nonzero(voi_mask), dtype=np.float64)
    if coords.shape[1] < 20:
        return 1, 0.0

    y = coords[1]
    distance_from_midline = np.abs(coords[0] - midline_x)

    if weights is not None:
        w = np.clip(weights[voi_mask].astype(np.float64), 0, None)
        if w.sum() <= 0:
            w = None
    else:
        w = None

    if w is None:
        w = np.ones(coords.shape[1], dtype=np.float64)

    w = w / w.sum()
    y_centered = y - (w * y).sum()
    d_centered = distance_from_midline - (w * distance_from_midline).sum()

    covariance = float((w * y_centered * d_centered).sum())
    y_std = float(np.sqrt((w * y_centered**2).sum()))
    d_std = float(np.sqrt((w * d_centered**2).sum()))
    if y_std < 1e-9 or d_std < 1e-9:
        return 1, 0.0

    correlation = covariance / (y_std * d_std)
    # Correlazione negativa: y crescente avvicina alla linea mediana, quindi
    # +y punta verso il caudato, cioe' e' anteriore.
    return (1 if correlation < 0 else -1), float(abs(correlation))


@dataclass(frozen=True)
class StriatalMeasurement:
    """Risultato completo della localizzazione striatale su un volume."""

    left: Hemisphere | None
    right: Hemisphere | None
    reference_mean: float
    midline_x: float
    # Riferimento alternativo, calcolato sulla sola fetta assiale che contiene
    # lo striato. Coesiste con quello globale invece di sostituirlo: quale dei
    # due generalizzi meglio fra centri e' una domanda empirica, e tenerli
    # entrambi permette di rispondere con una sola estrazione.
    reference_slab: float = float("nan")
    anterior_sign: int = 1
    anterior_confidence: float = 0.0
    header_agrees: bool = True
    # Descrittori a geometria fissa, per emisfero. Tenuti separati dal resto
    # perche' non condividono la definizione della VOI: vedi morphology.py.
    morphology_left: dict[str, float] | None = None
    morphology_right: dict[str, float] | None = None
    # Baricentro della maschera cerebrale, in voxel. E' l'unico riferimento
    # geometrico del volume che NON dipende dall'intensita' dello striato,
    # quindi l'unico rispetto a cui si possa dire se l'ancora si sia spostata
    # invece che il paziente. Serve alle due feature di deriva in features.py.
    mask_centroid: tuple[float, float, float] | None = None
    spacing_mm: float = float("nan")


def _split_caudate_putamen(
    data: np.ndarray, hemi: Hemisphere, anterior_sign: int
) -> Hemisphere:
    """Ricalcola le medie di caudato e putamen dato il verso anteriore."""
    import dataclasses

    coords = np.array(np.nonzero(hemi.voi_mask), dtype=np.float32)
    values = data[hemi.voi_mask]
    projection = anterior_sign * coords[1]
    threshold = anterior_sign * float(hemi.centroid[1])

    anterior = projection >= threshold
    posterior = ~anterior

    return dataclasses.replace(
        hemi,
        caudate_mean=float(values[anterior].mean()) if anterior.any() else float("nan"),
        putamen_mean=float(values[posterior].mean()) if posterior.any() else float("nan"),
    )


def measure(
    data: np.ndarray, spacing_mm: float, mask: np.ndarray | None = None
) -> StriatalMeasurement:
    """Localizza entrambi gli striati e stima la regione di riferimento.

    `mask` permette di passare una maschera cerebrale gia' calcolata invece di
    ricavarla qui. Serve alla pipeline reale, che la calcola una sola volta in
    spazio nativo prima di ritagliare il volume: cosi' la soglia di Otsu vede
    sempre l'istogramma completo e non dipende da quanta aria e' stata tolta,
    che varia da 190 a 630 mm di campo di vista a seconda del centro.

    Senza argomento la maschera viene calcolata sul volume ricevuto, che e' il
    comportamento su cui si reggono i test su phantom.
    """
    smoothed = gaussian_blur(data, sigma_vox=BRAIN_SIGMA_MM / spacing_mm)

    if mask is None:
        mask = _mask_from_smoothed(smoothed)

    midline_x = find_midline(mask)
    gx, _, _ = _coordinate_grids(data.shape)

    # In RAS, x crescente va verso destra.
    # Lo striato e' una struttura profonda e centrale. Confinare la ricerca del
    # picco a un intorno del centroide cerebrale rende la localizzazione immune
    # ai focolai extracerebrali (tiroide, ghiandole salivari) che possono
    # comparire ai margini del campo di vista con intensita' anche superiore.
    central = _central_region(mask, spacing_mm)

    left_mask = mask & central & (gx < midline_x)
    right_mask = mask & central & (gx >= midline_x)

    left = _extract_hemisphere(data, smoothed, left_mask, spacing_mm)
    right = _extract_hemisphere(data, smoothed, right_mask, spacing_mm)

    reference_mean = _reference_mean(data, mask, [left, right], spacing_mm)
    reference_slab = _reference_mean(
        data, mask, [left, right], spacing_mm, slab_half_mm=REFERENCE_SLAB_HALF_MM
    )

    # Verso anteriore dedotto dalla geometria, non dall'header. Si combinano i
    # due emisferi pesandoli per la confidenza: se uno e' compromesso dalla
    # patologia, l'altro porta comunque il segnale geometrico.
    votes = []
    for hemi in (left, right):
        if hemi is not None:
            sign, confidence = infer_anterior_sign(hemi.voi_mask, midline_x, weights=data)
            votes.append((sign, confidence))

    if votes:
        score = sum(sign * confidence for sign, confidence in votes)
        anterior_sign = 1 if score >= 0 else -1
        anterior_confidence = abs(score) / max(len(votes), 1)
    else:
        anterior_sign, anterior_confidence = 1, 0.0

    # La deduzione si applica solo quando la geometria e' conclusiva. Sotto
    # questa soglia il dato non e' informativo e ci si affida all'header, che
    # nella maggior parte dei casi e' corretto.
    trust_geometry = anterior_confidence >= 0.15
    effective_sign = anterior_sign if trust_geometry else 1

    if effective_sign == -1:
        left = _split_caudate_putamen(data, left, -1) if left is not None else None
        right = _split_caudate_putamen(data, right, -1) if right is not None else None

    # Il blocco morfologico si calcola qui, dopo che il verso anteriore e' noto:
    # pendenza e asimmetria del profilo cambiano segno con l'orientamento, e
    # calcolarle prima significherebbe mescolare volumi orientati diversamente.
    from .morphology import hemisphere_morphology

    morphology = {
        side: hemisphere_morphology(
            data,
            hemi.peak_index if hemi is not None else None,
            reference_mean,
            spacing_mm,
            midline_x,
            effective_sign,
            reference_slab=reference_slab,
        )
        for side, hemi in (("left", left), ("right", right))
    }

    return StriatalMeasurement(
        left=left,
        right=right,
        reference_mean=reference_mean,
        reference_slab=reference_slab,
        midline_x=midline_x,
        anterior_sign=effective_sign,
        anterior_confidence=anterior_confidence,
        header_agrees=(effective_sign == 1),
        morphology_left=morphology["left"],
        morphology_right=morphology["right"],
        mask_centroid=(
            tuple(np.array(np.nonzero(mask), dtype=np.float64).mean(axis=1))
            if mask is not None and mask.any() else None
        ),
        spacing_mm=float(spacing_mm),
    )


# Semispessore, in millimetri, della fetta assiale entro cui si puo' calcolare
# un riferimento alternativo. 25 mm attorno al piano striatale coprono la
# corteccia occipitale e parietale allo stesso livello dello striato, che e' la
# regione di riferimento della quantificazione DaT in clinica.
REFERENCE_SLAB_HALF_MM = 25.0


def _reference_mean(
    data: np.ndarray,
    mask: np.ndarray,
    hemispheres: list[Hemisphere | None],
    spacing_mm: float,
    slab_half_mm: float | None = None,
) -> float:
    """Uptake medio nel tessuto privo di trasportatori della dopamina.

    Definito come il tessuto cerebrale sufficientemente lontano da entrambi i
    picchi striatali. Usa la mediana invece della media perche' e' robusta a
    residui di spillover striatale ai margini della regione.

    `slab_half_mm` restringe la regione a una fetta assiale attorno al piano
    striatale, e serve a rispondere a un problema misurato sui dati reali.
    Senza restrizione la regione e' "tutto il tessuto lontano dallo striato",
    cioe' tutto quello che l'acquisizione ha incluso: su 24 esami veri
    l'estensione in z della maschera correla con il campo di vista (r = 0,37),
    quindi un centro che acquisisce alto include collo e cervelletto e un altro
    no. Il contenuto della regione di riferimento diventa allora una proprieta'
    del protocollo, e siccome il riferimento sta al DENOMINATORE dell'SBR,
    l'invarianza allo scanner su cui si regge tutta la pipeline si incrina
    esattamente dove dovrebbe reggere.

    Restringere a una fetta rende la regione approssimativamente la stessa
    anatomia in ogni esame, qualunque sia il campo di vista.
    """
    peaks = [h.centroid for h in hemispheres if h is not None]
    if not peaks:
        brain_values = data[mask]
        return float(np.median(brain_values)) if brain_values.size else 1.0

    gx, gy, gz = _coordinate_grids(data.shape)
    min_dist_vox = REFERENCE_MIN_DISTANCE_MM / spacing_mm

    far_from_striatum = np.ones(data.shape, dtype=bool)
    for centroid in peaks:
        dist = np.sqrt(
            (gx - centroid[0]) ** 2 + (gy - centroid[1]) ** 2 + (gz - centroid[2]) ** 2
        )
        far_from_striatum &= dist > min_dist_vox

    reference_region = mask & far_from_striatum
    if slab_half_mm is not None:
        striatal_z = float(np.mean([c[2] for c in peaks]))
        reference_region = reference_region & (
            np.abs(gz - striatal_z) <= slab_half_mm / spacing_mm
        )
    if reference_region.sum() < 50:
        # Fallback: se la testa e' piccola o il ritaglio e' stretto, usa tutto
        # il tessuto sotto la mediana, che e' dominato dal fondo.
        brain_values = data[mask]
        if brain_values.size == 0:
            return 1.0
        return float(np.median(brain_values))

    return float(np.median(data[reference_region]))


def binding_ratio(signal: float, reference: float) -> float:
    """SBR = (segnale - riferimento) / riferimento."""
    if not np.isfinite(signal) or not np.isfinite(reference) or reference <= 0:
        return float("nan")
    return (signal - reference) / reference
