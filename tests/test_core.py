"""Collaudo della localizzazione striatale e delle feature su phantom sintetici.

Questi test non richiedono i dati della competizione: verificano che la geometria
e le formule si comportino come previsto. Se falliscono, il bug e' nel codice,
non nei dati.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.striatum import measure
from src.features import build_features, FEATURE_NAMES, features_to_array
from src.morphology import PROFILE_DIRECTIONS
from src.phantom import normal_scan, pathologic_scan

SPACING = 2.0

def _feat(vol):
    return build_features(measure(vol, SPACING))

def test_finds_both_hemispheres():
    m = measure(normal_scan(), SPACING)
    assert m.left is not None, "emisfero sinistro non trovato"
    assert m.right is not None, "emisfero destro non trovato"
    assert m.reference_mean > 0, "riferimento non positivo"

def test_hemispheres_on_correct_sides():
    m = measure(normal_scan(), SPACING)
    assert m.left.centroid[0] < m.midline_x < m.right.centroid[0], \
        "gli emisferi non sono dai lati giusti della linea mediana"

def test_caudate_anterior_of_putamen():
    """Il caudato deve stare in y maggiore (anteriore) rispetto al putamen."""
    vol = normal_scan(left_caudate=1.0, left_putamen=0.2,
                      right_caudate=1.0, right_putamen=0.2)
    f = _feat(vol)
    assert f["left_sbr_caudate"] > f["left_sbr_putamen"], \
        "asse antero-posteriore invertito: il caudato dovrebbe essere piu' brillante"

def test_normal_has_higher_sbr_than_pathologic():
    n, p = _feat(normal_scan()), _feat(pathologic_scan())
    assert n["mean_sbr"] > p["mean_sbr"], "SBR non discrimina normale da patologico"
    assert n["min_sbr"] > p["min_sbr"]

def test_pathologic_is_more_asymmetric():
    n, p = _feat(normal_scan()), _feat(pathologic_scan())
    assert p["asym_sbr"] > n["asym_sbr"], "asimmetria non rilevata"

def test_putamen_caudate_ratio_drops():
    n, p = _feat(normal_scan()), _feat(pathologic_scan())
    assert p["min_putamen_caudate_ratio"] < n["min_putamen_caudate_ratio"], \
        "il rapporto putamen/caudato non cala nel patologico"

def test_sbr_invariant_to_scanner_gain():
    """L'SBR e' un rapporto: raddoppiare i conteggi non deve cambiarlo.

    E' la proprieta' su cui si regge la generalizzazione tra i 10 centri.
    """
    # Si usa un phantom asimmetrico di proposito: su uno simmetrico l'indice di
    # asimmetria vale esattamente zero, e confrontare due zeri con tolleranza
    # relativa non verifica nulla. Qui i valori sono tutti lontani da zero,
    # quindi il test ha mordente.
    vol = pathologic_scan(severity=0.7)
    a, b = _feat(vol), _feat(vol * 3.7)
    for k in ("mean_sbr", "asym_sbr", "min_sbr", "min_putamen_caudate_ratio"):
        assert abs(a[k]) > 1e-3, f"{k} degenere nel phantom: test senza valore"
        assert np.isclose(a[k], b[k], rtol=1e-4), f"{k} dipende dal guadagno dello scanner"

def test_severity_is_monotonic():
    sev = [0.0, 0.3, 0.6, 0.9]
    vals = [_feat(pathologic_scan(severity=s))["min_sbr"] for s in sev]
    assert all(vals[i] > vals[i+1] for i in range(len(vals)-1)), \
        f"min_sbr non decresce monotonicamente con la severita': {vals}"

def test_robust_to_noise():
    clean = _feat(normal_scan())
    noisy = _feat(normal_scan(noise_sigma=0.35, seed=1))
    assert abs(clean["mean_sbr"] - noisy["mean_sbr"]) / clean["mean_sbr"] < 0.25, \
        "SBR troppo sensibile al rumore"

def test_infers_anterior_direction():
    """Il verso anteriore deve essere dedotto dalla geometria, non dall'header."""
    m = measure(normal_scan(), SPACING)
    assert m.anterior_sign == 1, "verso anteriore sbagliato su volume ben orientato"
    assert m.anterior_confidence > 0.15, \
        f"geometria non conclusiva (conf={m.anterior_confidence:.3f})"

def test_detects_flipped_volume():
    """Un volume capovolto lungo y deve essere riconosciuto come tale."""
    flipped = normal_scan()[:, ::-1, :].copy()
    m = measure(flipped, SPACING)
    assert m.anterior_sign == -1, "inversione dell'asse antero-posteriore non rilevata"
    assert not m.header_agrees

def test_caudate_putamen_survives_flip():
    """Il punto cruciale: con l'header sbagliato le feature devono restare corrette.

    Se questo test passa, un dataset con orientamenti misti non degrada in
    silenzio - che era il rischio principale della pipeline.
    """
    vol = normal_scan(left_caudate=1.0, left_putamen=0.2,
                      right_caudate=1.0, right_putamen=0.2)
    upright = build_features(measure(vol, SPACING))
    flipped = build_features(measure(vol[:, ::-1, :].copy(), SPACING))

    assert upright["left_sbr_caudate"] > upright["left_sbr_putamen"]
    assert flipped["left_sbr_caudate"] > flipped["left_sbr_putamen"], \
        "con il volume capovolto caudato e putamen risultano invertiti"
    assert np.isclose(upright["min_putamen_caudate_ratio"],
                      flipped["min_putamen_caudate_ratio"], rtol=0.05), \
        "il rapporto putamen/caudato cambia se il volume e' capovolto"

def test_flip_detection_survives_pathology():
    """La deduzione deve reggere anche quando la patologia deforma lo striato."""
    for severity in (0.4, 0.7, 1.0):
        m = measure(pathologic_scan(severity=severity)[:, ::-1, :].copy(), SPACING)
        assert m.anterior_sign == -1, f"flip non rilevato con severita' {severity}"


def _add_hot_focus(vol, intensity, pos=(32, 6, 10), radius=3.5):
    """Aggiunge un focus extracerebrale (tiroide o ghiandole salivari)."""
    grids = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in vol.shape], indexing="ij")
    squared = sum(((grids[i] - pos[i]) / radius) ** 2 for i in range(3))
    return vol + intensity * np.exp(-1.5 * squared).astype(np.float32)

def test_survives_extracerebral_hot_focus():
    """Un focus extracerebrale intenso non deve dirottare la localizzazione.

    E' il fallimento piu' insidioso trovato negli stress test: senza difesa, la
    pipeline misurava il focus invece dello striato e il margine tra normale e
    patologico crollava esattamente a zero, senza alcun errore visibile.
    """
    base_n, base_p = normal_scan(), pathologic_scan()
    peak = float(base_n.max())
    clean_margin = _feat(base_n)["mean_sbr"] - _feat(base_p)["mean_sbr"]

    for multiple in (1.0, 3.0, 6.0):
        n = _feat(_add_hot_focus(base_n, peak * multiple))
        p = _feat(_add_hot_focus(base_p, peak * multiple))
        margin = n["mean_sbr"] - p["mean_sbr"]
        assert margin > 0.5 * clean_margin, \
            f"focus a {multiple}x: margine crollato a {margin:.3f} (pulito {clean_margin:.3f})"

def test_confidence_is_exposed_as_feature():
    """La confidenza di localizzazione deve arrivare al modello."""
    f = _feat(normal_scan())
    assert "localization_confidence" in f
    assert f["localization_confidence"] > 0.15, "confidenza bassa su un volume pulito"


def test_feature_vector_is_wellformed():
    f = _feat(normal_scan())
    arr = features_to_array(f)
    assert arr.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(arr).sum() >= len(FEATURE_NAMES) - 2, \
        f"troppi NaN: {[n for n,v in zip(FEATURE_NAMES, arr) if not np.isfinite(v)]}"

def test_handles_empty_volume():
    """Un volume vuoto non deve far crashare la pipeline."""
    f = _feat(np.zeros((32, 32, 24), dtype=np.float32))
    assert isinstance(f, dict) and len(f) > 0


def test_exclusion_list_has_no_typos():
    """Ogni nome in EXCLUDED_FROM_MODEL deve corrispondere a una feature vera.

    Un refuso qui non darebbe errore: escluderebbe semplicemente nulla, e la
    feature che si voleva togliere continuerebbe ad arrivare al modello senza
    che nessuno se ne accorga.
    """
    from src.features import EXCLUDED_FROM_MODEL, FEATURE_NAMES, MODEL_FEATURE_NAMES
    from src.morphology import INACTIVE_DIRECTION_KEYS

    # Le direzioni disattivate non producono piu' colonne, ma il loro verdetto -
    # con il costo misurato accanto - resta in EXCLUDED_FROM_MODEL: e' il
    # registro di cosa e' stato provato, non l'elenco delle colonne di oggi.
    # Cancellarlo per far passare questo test perderebbe proprio l'informazione
    # che impedisce di riprovare la stessa cosa fra tre settimane.
    dormienti = {f"{how}_{key}"
                 for key in INACTIVE_DIRECTION_KEYS for how in ("min", "max")}

    unknown = set(EXCLUDED_FROM_MODEL) - set(FEATURE_NAMES) - dormienti
    assert not unknown, f"nomi inesistenti nella lista di esclusione: {sorted(unknown)}"
    assert set(MODEL_FEATURE_NAMES).isdisjoint(EXCLUDED_FROM_MODEL)
    escluse_esistenti = set(EXCLUDED_FROM_MODEL) & set(FEATURE_NAMES)
    assert len(MODEL_FEATURE_NAMES) == len(FEATURE_NAMES) - len(escluse_esistenti)


def test_crop_preserves_the_head():
    """Il ritaglio non deve perdere voxel della maschera cerebrale.

    Se ne perdesse, cambierebbe la regione di riferimento - che sta al
    denominatore dell'SBR - e sarebbe un cambio di semantica mascherato da
    ottimizzazione.
    """
    from src.striatum import brain_mask
    from src.volume import bounding_box

    vol = normal_scan()
    zooms = np.array([SPACING] * 3, dtype=np.float32)
    mask = brain_mask(vol, tuple(float(z) for z in zooms))
    box = bounding_box(mask, zooms)

    assert box is not None
    assert int(mask[box].sum()) == int(mask.sum()), \
        "il ritaglio ha tagliato via parte della testa"


def test_resampled_mask_matches_resampled_data():
    """Dati e maschera devono avere la stessa forma dopo il ricampionamento.

    Sono ricampionati da due funzioni diverse con ordini di interpolazione
    diversi: se le forme divergessero, measure() userebbe una maschera
    disallineata rispetto ai voxel, e lo farebbe in silenzio sui soli scanner
    con quello spacing.
    """
    from src.striatum import brain_mask
    from src.volume import resample_isotropic, resample_mask

    vol = normal_scan()
    for zoom in (1.48, 2.0, 2.46, 3.9, 4.42):
        zooms = np.array([zoom, zoom, zoom * 1.3], dtype=np.float32)  # anche anisotropo
        mask = brain_mask(vol, tuple(float(z) for z in zooms))
        data_shape = resample_isotropic(vol, zooms, SPACING).shape
        mask_shape = resample_mask(mask, zooms, SPACING).shape
        assert data_shape == mask_shape, \
            f"forme divergenti a spacing {zooms}: {data_shape} vs {mask_shape}"


def test_precomputed_mask_matches_internal_one():
    """measure() con maschera passata deve dare lo stesso risultato che senza.

    E' l'assunzione su cui si regge il ritaglio: la maschera calcolata prima e
    passata dentro deve essere equivalente a quella che measure() calcolerebbe
    da sola sullo stesso volume.
    """
    from src.striatum import brain_mask

    vol = pathologic_scan(severity=0.6)
    internal = build_features(measure(vol, SPACING))
    external = build_features(measure(vol, SPACING, mask=brain_mask(vol, SPACING)))

    for key in ("mean_sbr", "min_sbr", "asym_sbr", "reference_mean"):
        assert np.isclose(internal[key], external[key], rtol=1e-6), \
            f"{key} cambia se la maschera viene passata dall'esterno"

def test_morphology_invariant_to_scanner_gain():
    """Il blocco a geometria fissa deve essere invariante al guadagno come l'SBR.

    E' meno ovvio che per l'SBR e per questo va verificato: i momenti sono pesati
    sul legame specifico, che con il guadagno scala, e la sottrazione del
    riferimento avviene PRIMA della normalizzazione. Se i pesi non fossero
    rinormalizzati a somma uno, ogni momento dipenderebbe dai conteggi assoluti,
    cioe' dal centro, ed e' esattamente l'errore che ha gia' costretto a
    escludere `reference_mean` dal modello.

    La versione precedente di questo test saltava ogni chiave con valore sotto
    1e-3 e si accontentava di quattro verifiche su sette: fino a tre grandezze
    potevano non essere mai controllate senza che il test lo segnalasse, e quali
    fossero dipendeva dal phantom. Qui il valore vicino a zero si gestisce con
    una tolleranza assoluta invece che saltando il confronto, cosi' OGNI chiave
    finita viene verificata e il conteggio serve solo a impedire che il test
    passi a vuoto se la localizzazione fallisse del tutto.
    """
    vol = pathologic_scan(severity=0.7)
    a, b = _feat(vol), _feat(vol * 4.3)

    keys = [
        "min_sbr_core", "min_sbr_full", "min_concentration", "min_shape_corr",
        "min_post_ant_ratio", "mean_ap_extent_mm", "min_sbr_core_slab",
        # Blocco direzionale. Le pendenze in logaritmo sono il caso meno ovvio:
        # log(gamma * m) = log(gamma) + log(m), quindi il guadagno e' una
        # costante additiva uguale in ogni bin e sparisce dalla pendenza. Se
        # sparisse solo quasi, questo test lo direbbe.
        "min_ap_log_slope", "max_ap_log_slope",
        "min_ap_tail_ratio", "max_ap_tail_ratio",
        "min_ap_bin_m15_ratio", "max_ap_bin_m15_ratio",
        "min_ap_bin_p15_ratio", "min_ap_bin_m09_ratio", "min_ap_bin_p09_ratio",
        "asym_ap_tail_ratio", "mean_ap_axis_angle_deg",
    ]
    # Le sintesi comuni a ogni direzione attiva, derivate invece che elencate: se
    # una direzione viene riattivata la copertura torna da sola, e se una attiva
    # smettesse di produrre la sua chiave il test lo direbbe. Elencarle a mano
    # significherebbe che riaprire un verdetto lascia il blocco nuovo scoperto.
    keys += [f"{how}_{prefix}_{suffix}"
             for _, prefix, _, _ in PROFILE_DIRECTIONS
             for suffix in ("log_slope", "tail_ratio")
             for how in ("min", "max")]
    checked = 0
    for key in keys:
        assert key in a, f"{key} non e' arrivata nel vettore di feature"
        if not np.isfinite(a[key]) and not np.isfinite(b[key]):
            continue
        assert np.isclose(a[key], b[key], rtol=1e-4, atol=1e-9), \
            f"{key} dipende dal guadagno dello scanner: {a[key]} contro {b[key]}"
        checked += 1
    assert checked >= len(keys) - 2, \
        f"solo {checked} chiavi su {len(keys)} sono finite: il test non ha mordente"


def test_fixed_geometry_ignores_the_threshold():
    """Le misure a raggio fisso non devono muoversi cambiando la soglia della VOI.

    E' la proprieta' per cui il blocco esiste. Volume, elongation e flatness sono
    fuori dal modello perche' descrivono una regione la cui estensione dipende
    dal picco; se le nuove feature condividessero quella dipendenza, avrebbero
    ereditato lo stesso difetto sotto un altro nome.
    """
    import src.striatum as striatum

    vol = pathologic_scan(severity=0.5)
    original = striatum.VOI_THRESHOLD_FRACTION
    try:
        striatum.VOI_THRESHOLD_FRACTION = 0.5
        a = _feat(vol)
        striatum.VOI_THRESHOLD_FRACTION = 0.35
        b = _feat(vol)
    finally:
        striatum.VOI_THRESHOLD_FRACTION = original

    assert not np.isclose(a["total_volume_mm3"], b["total_volume_mm3"], rtol=1e-3), \
        "la soglia non ha cambiato la VOI: il test non sta verificando nulla"

    # Il profilo direzionale mancava del tutto da questo elenco, ed era il buco
    # di copertura piu' grosso del blocco: `ap_slope` e `post_ant_ratio` esistono
    # proprio perche' la versione a soglia della stessa misura - il rapporto
    # putamen/caudato della VOI - era autoreferenziale. Non verificarne
    # l'indipendenza dalla soglia lasciava senza rete l'unica proprieta' per cui
    # sono state riscritte.
    for key in ["min_sbr_core", "min_sbr_full", "min_concentration", "min_shape_corr",
                "min_ap_slope", "max_ap_slope", "min_post_ant_ratio",
                "min_ap_bin_m15_ratio", "min_ap_bin_p15_ratio",
                "asym_ap_tail_ratio", "mean_ap_axis_angle_deg"] + [
                # Derivate dalle direzioni attive, per la ragione detta sopra.
                f"min_{prefix}_{suffix}"
                for _, prefix, _, _ in PROFILE_DIRECTIONS
                for suffix in ("log_slope", "tail_ratio")] + [
                f"max_{prefix}_{suffix}"
                for _, prefix, _, _ in PROFILE_DIRECTIONS
                for suffix in ("log_slope", "tail_ratio")]:
        assert np.isclose(a[key], b[key], rtol=1e-6, equal_nan=True), \
            f"{key} dipende dalla soglia della VOI, che e' cio' che doveva evitare"


def test_profile_along_y_matches_the_old_cylinder():
    """Il campionamento direzionale a theta=0 deve dare la maschera di prima.

    E' la condizione che rende valido ogni confronto contro la baseline
    pubblicata. Il profilo e' diventato parametrico nella direzione; se per
    `direction = (0, +-1, 0)` selezionasse anche un solo voxel diverso, `ap_slope`
    e `post_ant_ratio` si sposterebbero, e la log loss di riferimento a cui si
    confrontano tutte le varianti nuove non sarebbe piu' quella misurata.

    Si confrontano le maschere e non i valori: due maschere identiche danno
    valori identici per costruzione, mentre due valori vicini possono nascondere
    maschere diverse che si compensano.
    """
    from src.morphology import (
        PROFILE_AXIAL_MM, PROFILE_HALF_WIDTH_MM, PROFILE_LATERAL_MM,
        PROFILE_OFFSETS_MM, _axial_direction, _local_box, RADIUS_FULL_MM,
    )

    vol = pathologic_scan(severity=0.6)
    m = measure(vol, SPACING)
    hemi = m.left or m.right
    assert hemi is not None and hemi.peak_index is not None

    center = np.array(hemi.peak_index, dtype=np.float64)
    box, grids = _local_box(
        vol, tuple(int(round(c)) for c in center), RADIUS_FULL_MM + 4.0, SPACING
    )

    # Selezione di riferimento: la formula testuale dell'implementazione
    # precedente, scritta qui a mano perche' quel codice non esiste piu'.
    anterior_sign = m.anterior_sign
    y = anterior_sign * (grids[1] - center[1]) * SPACING
    lateral = np.abs(grids[0] - center[0]) * SPACING
    axial = np.abs(grids[2] - center[2]) * SPACING
    cylinder_old = (lateral <= PROFILE_LATERAL_MM) & (axial <= PROFILE_AXIAL_MM)

    # Selezione nuova, per la stessa direzione.
    direction = _axial_direction(0.0, anterior_sign, medial_sign=1)
    delta = [(grids[i] - center[i]) * SPACING for i in range(3)]
    along = sum(d * u for d, u in zip(delta, direction))
    perpendicular = np.abs(-delta[0] * direction[1] + delta[1] * direction[0])
    cylinder_new = (perpendicular <= PROFILE_LATERAL_MM) & (np.abs(delta[2]) <= PROFILE_AXIAL_MM)

    assert np.array_equal(cylinder_old, cylinder_new), \
        "il cilindro a theta=0 non coincide con quello dell'implementazione precedente"

    for offset in PROFILE_OFFSETS_MM:
        old_bin = cylinder_old & (np.abs(y - offset) <= PROFILE_HALF_WIDTH_MM)
        new_bin = cylinder_new & (np.abs(along - offset) <= PROFILE_HALF_WIDTH_MM)
        assert np.array_equal(old_bin, new_bin), \
            f"il bin a {offset:+.0f} mm seleziona voxel diversi"
        assert old_bin.sum() > 0, f"il bin a {offset:+.0f} mm e' vuoto: test senza mordente"


def test_posterior_bin_collapses_instead_of_moving():
    """Svuotare la coda posteriore deve far crollare quel bin, non spostarlo.

    E' l'affermazione centrale del modulo, e finora non aveva alcun test. La
    misura precedente - il rapporto putamen/caudato sulla VOI - falliva proprio
    qui: quando il putamen posteriore degenera i suoi voxel scendono sotto la
    soglia ed escono dalla VOI, il centroide si sposta in avanti e il taglio si
    sposta con lui, quindi la misura insegue la patologia invece di misurarla.

    Il criterio: azzerando l'uptake nella meta' posteriore dello striato, la
    media del bin piu' posteriore deve crollare, mentre il bin anteriore resta
    dov'e'. Se la regione seguisse il segnale, il bin posteriore troverebbe
    altrove tessuto da misurare e la sua media resterebbe alta.
    """
    from src.morphology import (
        PROFILE_OFFSETS_MM, _axial_direction, _local_box, _profile_bins,
        RADIUS_FULL_MM,
    )

    vol = normal_scan()
    m = measure(vol, SPACING)
    hemi = m.left or m.right
    assert hemi is not None and hemi.peak_index is not None
    center = np.array(hemi.peak_index, dtype=np.float64)

    def bins_of(volume):
        box, grids = _local_box(
            volume, tuple(int(round(c)) for c in center), RADIUS_FULL_MM + 4.0, SPACING
        )
        direction = _axial_direction(0.0, m.anterior_sign, medial_sign=1)
        return _profile_bins(box, grids, center, SPACING, direction)

    intact = bins_of(vol)

    # Azzera tutto cio' che sta dietro l'ancora lungo l'asse antero-posteriore,
    # portandolo al livello del fondo.
    damaged_vol = vol.copy()
    y_index = np.arange(vol.shape[1])
    behind = (m.anterior_sign * (y_index - center[1])) < -4.0
    background = float(np.median(vol[vol > 0])) if (vol > 0).any() else 0.0
    damaged_vol[:, behind, :] = background
    damaged = bins_of(damaged_vol)

    posterior = 0                      # bin a -15 mm
    anterior = len(PROFILE_OFFSETS_MM) - 1  # bin a +15 mm

    # Si confronta il LEGAME SPECIFICO, cioe' l'uptake sopra il fondo, non
    # l'uptake grezzo. Il fondo e' una costante additiva presente ovunque: un
    # bin completamente svuotato non va a zero, va al livello del fondo, e
    # leggere il grezzo confonderebbe "crollato del tutto" con "sceso del 30%".
    intact_specific = intact[posterior] - background
    damaged_specific = damaged[posterior] - background
    assert intact_specific > 0.1, \
        f"il bin posteriore non aveva segnale da perdere ({intact_specific:.2f}): test senza mordente"

    assert np.isfinite(damaged[posterior]), \
        "il bin posteriore e' sparito invece di crollare: la regione ha seguito il segnale"
    assert damaged_specific < 0.1 * intact_specific, \
        (f"il legame specifico del bin posteriore non e' crollato: "
         f"{intact_specific:.3f} -> {damaged_specific:.3f}")
    assert np.isclose(damaged[anterior], intact[anterior], rtol=1e-6), \
        "il bin anteriore si e' mosso: il danno posteriore ha spostato la geometria"


def test_log_transform_touches_only_ratios():
    """LogRatios deve trasformare i rapporti e lasciare stare tutto il resto.

    Sbagliare bersaglio qui non darebbe errore: produrrebbe NaN silenziosi sulle
    colonne che possono essere negative (pendenze, correlazioni) e il modello si
    limiterebbe a peggiorare senza che nulla lo segnali.
    """
    import pandas as pd

    from src.transforms import LogRatios, log_transform_kind

    assert log_transform_kind("min_sbr") == "shift"
    assert log_transform_kind("min_concentration") == "ratio"
    assert log_transform_kind("asym_sbr") is None, \
        "gli indici di asimmetria sono gia' invarianti al centro: non vanno toccati"
    assert log_transform_kind("min_shape_corr") is None
    assert log_transform_kind("min_ap_slope") is None, \
        "la pendenza puo' essere negativa: il logaritmo non e' definito"

    # Il blocco direzionale. Ogni riga qui sotto e' l'unica difesa contro un
    # fallimento silenzioso: sbagliare bersaglio non darebbe errore, la feature
    # renderebbe semplicemente meno e sembrerebbe che l'idea non funzionasse.
    assert log_transform_kind("min_ap_log_slope") is None, \
        "e' gia' in scala logaritmica, e puo' essere negativa"
    assert log_transform_kind("min_obl_log_slope") is None
    assert log_transform_kind("min_ap_tail_ratio") == "ratio"
    assert log_transform_kind("min_ap_bin_m15_ratio") == "ratio", \
        "il suffisso _ratio deve restare in fondo al nome: endswith non perdona"
    assert log_transform_kind("max_ap_bin_p15_ratio") == "ratio"
    # Dipende dall'ORDINE delle istruzioni in log_transform_kind: il nome finisce
    # per _ratio e verrebbe trasformato, se il controllo su asym_ non venisse
    # prima. E' corretto che non lo sia - gli indici di asimmetria sono gia'
    # invarianti al centro, eta quadro 0,010 - ma e' un equilibrio fragile.
    assert log_transform_kind("asym_ap_tail_ratio") is None, \
        "il controllo su asym_ deve precedere quello sul suffisso _ratio"
    # Una pendenza in scala logaritmica non deve MAI contenere "sbr" nel nome:
    # farebbe match sul ramo "shift", cioe' log(1 + pendenza), e con pendenze
    # piccole e di segno variabile il clip inchioderebbe meta' colonna a -6,9.
    assert log_transform_kind("min_ap_log_sbr_slope") == "shift", \
        "documenta la trappola: un nome cosi' NON va usato"

    frame = pd.DataFrame({
        "min_sbr": [1.0, 3.0, np.nan],
        "min_concentration": [1.0, 2.0, 4.0],
        "min_shape_corr": [0.5, -0.2, 0.9],
        "min_ap_slope": [-0.01, 0.02, 0.0],
    })
    out = LogRatios().fit(frame).transform(frame)

    assert np.isclose(out["min_sbr"][0], np.log(2.0))
    assert np.isnan(out["min_sbr"][2]), "il NaN deve passare intatto all'imputatore"
    assert np.isclose(out["min_concentration"][2], np.log(4.0))
    assert np.allclose(out["min_shape_corr"], frame["min_shape_corr"])
    assert np.allclose(out["min_ap_slope"], frame["min_ap_slope"])


def test_slab_reference_excludes_distant_slices():
    """Il riferimento sulla fetta non deve vedere il tessuto lontano lungo z.

    E' la ragione per cui esiste: su 24 esami veri l'estensione in z della
    maschera correla con il campo di vista (r = 0,37), quindi quanto collo e
    cervelletto finiscono nella regione di riferimento e' una proprieta' del
    protocollo del centro. Siccome il riferimento sta al denominatore dell'SBR,
    quella dipendenza si trasferisce a ogni feature di intensita'.

    Sui phantom il confronto diretto fra i due riferimenti non direbbe nulla -
    il fondo e' uniforme, quindi le due mediane coincidono per costruzione e il
    test passerebbe anche con la restrizione rotta. Serve un volume in cui il
    tessuto lontano lungo z sia diverso da quello vicino.
    """
    from types import SimpleNamespace

    from src.striatum import REFERENCE_SLAB_HALF_MM, _reference_mean

    shape = (60, 60, 80)
    data = np.full(shape, 100.0, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)

    # Due striati al centro, e un tessuto molto piu' brillante nelle fette
    # estreme: e' il ruolo che nei dati veri hanno collo e cervelletto quando il
    # centro acquisisce alto.
    z_center = shape[2] // 2
    far = REFERENCE_SLAB_HALF_MM / SPACING
    data[:, :, : int(z_center - far - 5)] = 400.0
    data[:, :, int(z_center + far + 5) :] = 400.0

    hemispheres = [
        SimpleNamespace(centroid=np.array([20.0, 30.0, float(z_center)])),
        SimpleNamespace(centroid=np.array([40.0, 30.0, float(z_center)])),
    ]

    globale = _reference_mean(data, mask, hemispheres, SPACING)
    fetta = _reference_mean(data, mask, hemispheres, SPACING,
                            slab_half_mm=REFERENCE_SLAB_HALF_MM)

    assert fetta == 100.0, \
        f"la fetta ha raccolto tessuto lontano lungo z (riferimento {fetta})"
    assert globale > fetta, \
        "il riferimento globale non ha visto il tessuto lontano: test senza mordente"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1; print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} test superati")
    sys.exit(1 if failed else 0)
