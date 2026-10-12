#!/usr/bin/env python3
"""Assembla main.py + src/ + modelli, e scrive submission.zip.

Il runtime ufficiale offre `just pack-submission`, che usa `uvx rpzip`. Qui
l'archivio viene scritto direttamente per non dipendere da `just`, da `uv` o
dall'eseguibile `zip`, che su Windows tipicamente non c'e'.

Due dettagli che non sono cosmetici, entrambi scoperti impacchettando a mano:

1. **I separatori dentro l'archivio devono essere `/`.** `Compress-Archive` di
   PowerShell scrive `src\\pipeline.py`, e `unzip` dentro il container Linux
   interpreta quel backslash come parte del NOME del file invece che come
   separatore di cartella: si ritrova un file chiamato "src\\pipeline.py" nella
   radice, l'import di `src.pipeline` fallisce, e la submission va interamente
   in fallback senza che nulla segnali il perche'.

2. **`__pycache__` va escluso.** L'ambiente locale e' Python 3.11 e il
   container 3.12: il bytecode compilato e' inutile la' dentro, e averlo
   nell'archivio significa spedire artefatti di una versione diversa da quella
   che li eseguira'. La cartella di destinazione va ripulita a ogni giro,
   perche' collaudare la submission in locale la ricrea.

3. **Cosa si puo' escludere non lo decide l'analisi degli import.** `models.pkl`
   contiene Pipeline di sklearn, e una Pipeline serializza il transformer
   custom *per riferimento*: dentro il pickle c'e' la stringa "src.transforms",
   non il codice. Nessun `import` la nomina, quindi qualunque grafo di import
   dichiara quel modulo irraggiungibile - ma toglierlo fa fallire
   `pickle.load`, e main.py degrada l'INTERA submission a prevalenza fissa
   senza che il container segnali un errore. Da qui `pickle_module_refs()`:
   l'archivio viene verificato contro cio' che il pickle chiedera' davvero.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import pickletools
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Moduli che non servono in inferenza. La lista e' volutamente corta: escludere
# e' un'ottimizzazione da pochi kilobyte, sbagliare l'esclusione costa la
# submission. Ogni voce e' verificata irraggiungibile sia dagli import sia dai
# riferimenti dentro models.pkl (vedi check_model_deps).
#   - phantom.py  volumi sintetici per i test
#   - model.py    codice di addestramento: non gira mai nel container, e la
#                 riproducibilita' richiesta ai vincitori passa dal repo, non
#                 dallo zip di inferenza
EXCLUDE = shutil.ignore_patterns("__pycache__", "*.pyc", "phantom.py", "model.py")
MANIFEST_NAME = "artifact_manifest.json"
MANIFEST_FORMAT = "dat-parkinson-artifact/v1"


def sha256_file(path: Path) -> str:
    """Identita' del contenuto, indipendente da nome e data del file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, archived_as: str) -> dict:
    return {
        "file": archived_as,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_identity() -> dict:
    """Commit e stato del sorgente; valori nulli se il repo non e' Git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def build_artifact_manifest(models: Path, voxnet: Path | None,
                            metro: str | None) -> dict:
    """Descrive esattamente gli artefatti che il runtime eseguira'."""
    manifest = {
        "format": MANIFEST_FORMAT,
        "source": git_identity(),
        "models": file_identity(models, "models.pkl"),
        "voxel": None,
    }
    if voxnet is None:
        return manifest

    sidecar = voxnet.with_suffix(".fusione.json")
    if not sidecar.is_file():
        raise SystemExit(f"ERRORE: manca {sidecar}; checkpoint non identificabile")
    fusion = json.loads(sidecar.read_text(encoding="utf-8"))
    required = ("a", "b", "arco", "n_reti", "norm", "arch")
    missing = [key for key in required if key not in fusion]
    if missing:
        raise SystemExit(
            f"ERRORE: {sidecar.name} non identifica {', '.join(missing)}"
        )
    if not metro:
        raise SystemExit("ERRORE: un checkpoint voxel richiede un metro esplicito")
    result_dir = ROOT / "experiments" / "risultati"
    meter_json = result_dir / f"{metro}.json"
    meter_npz = result_dir / f"{metro}.npz"
    missing_files = [str(p) for p in (meter_json, meter_npz) if not p.is_file()]
    if missing_files:
        raise SystemExit(
            f"ERRORE: artefatti del metro mancanti: {', '.join(missing_files)}"
        )

    manifest["voxel"] = {
        **file_identity(voxnet, "voxnet.pt"),
        "source_file": voxnet.name,
        "config": {
            "arch": fusion["arch"],
            "norm": fusion["norm"],
            "autonorm": bool(fusion.get("autonorm", False)),
            "n_reti": int(fusion["n_reti"]),
        },
        "fusion": {
            "a": float(fusion["a"]),
            "b": float(fusion["b"]),
            "arco": fusion["arco"],
        },
        "meter": {
            "name": metro,
            "json": file_identity(meter_json, meter_json.name),
            "npz": file_identity(meter_npz, meter_npz.name),
        },
    }
    return manifest


def check_manifest(archive: Path) -> None:
    """Il manifest deve descrivere i byte realmente contenuti nello ZIP."""
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        if MANIFEST_NAME not in names:
            raise SystemExit(f"ERRORE: {MANIFEST_NAME} assente dall'archivio")
        manifest = json.loads(bundle.read(MANIFEST_NAME))
        if manifest.get("format") != MANIFEST_FORMAT:
            raise SystemExit("ERRORE: formato del manifest sconosciuto")
        identities = [manifest.get("models")]
        if manifest.get("voxel") is not None:
            identities.append(manifest["voxel"])
        for identity in identities:
            filename = identity.get("file") if isinstance(identity, dict) else None
            if not filename or filename not in names:
                raise SystemExit(f"ERRORE: file dichiarato dal manifest assente: {filename}")
            payload = bundle.read(filename)
            actual = hashlib.sha256(payload).hexdigest()
            if len(payload) != identity.get("bytes") or actual != identity.get("sha256"):
                raise SystemExit(
                    f"ERRORE: {filename} non coincide con l'identita' nel manifest"
                )


def pickle_module_refs(pickle_path: Path) -> set[str]:
    """I moduli `src.*` che models.pkl nominera' al momento di deserializzare.

    Si leggono le opcode senza eseguire il pickle: i riferimenti a classi
    esterne sono stringhe nel flusso, non oggetti. E' l'unico modo di sapere
    cosa serve davvero senza importare nulla.
    """
    refs: set[str] = set()
    with open(pickle_path, "rb") as handle:
        for _, arg, _ in pickletools.genops(io.BytesIO(handle.read())):
            if isinstance(arg, str) and (arg == "src" or arg.startswith("src.")):
                refs.add(arg)
    return refs


def check_voxnet(archive: Path, voxnet_path: Path | None) -> None:
    """Il ramo a voxel viaggia intero, o non viaggia.

    Un `state_dict` di PyTorch contiene **solo tensori**: l'architettura la
    fornisce `src/voxnet.py`. Se i pesi fossero nell'archivio e il modulo no,
    l'import in `main.py` fallirebbe e `load_voxnet` degraderebbe l'intero test
    set al solo modello SBR — con exit code 0, un `submission.csv` valido, e un
    punteggio peggiore di ~0,02 che si leggerebbe come "la CNN non ha reso".
    E' la stessa classe di guasto del pickle che nomina `src.transforms`, e
    merita lo stesso controllo esplicito invece di fidarsi del grafo degli
    import.
    """
    names = set(zipfile.ZipFile(archive).namelist())
    if voxnet_path is None:
        if "voxnet.pt" in names:
            raise SystemExit("ERRORE: voxnet.pt nell'archivio ma non richiesto")
        print("  ramo a voxel: ASSENTE (solo modello SBR)")
        return
    mancanti = [f for f in ("voxnet.pt", "src/voxnet.py") if f not in names]
    if mancanti:
        raise SystemExit(
            f"ERRORE: il ramo a voxel e' incompleto nell'archivio: manca "
            f"{', '.join(mancanti)}. Con i pesi ma senza il modulo — o viceversa "
            f"— ogni esame ripiega sul solo modello SBR senza alcun errore."
        )
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        # Il venv locale non ha torch: l'impacchettamento gira qui, i pesi li ha
        # prodotti il container. La verifica della struttura interna si fa
        # comunque, ma nel container, dove `main.py` verra' eseguito davvero —
        # ed e' `load_voxnet` a farla, con la sua prova a vuoto.
        print(f"  ramo a voxel: presente ({voxnet_path.stat().st_size / 1e6:.0f} MB), "
              f"struttura non verificabile senza torch in locale")
        return
    blob = torch.load(str(voxnet_path), map_location="cpu", weights_only=True)
    for chiave in ("config", "state_dicts", "fusione"):
        if chiave not in blob:
            raise SystemExit(f"ERRORE: voxnet.pt senza la chiave '{chiave}'")
    a, b = blob["fusione"]["a"], blob["fusione"]["b"]
    print(f"  ramo a voxel: {len(blob['state_dicts'])} reti, arch "
          f"'{blob['config']['arch']}', fusione a={a:.3f} b={b:.3f}")


def check_fusione_per_centro(voxnet_path: Path | None, models_path: Path | None = None,
                             metro: str = "metro_esterno_gruppi_cnn_istanza",
                             q: float = 0.4) -> None:
    """La fusione non deve perdere su NESSUN centro mai visto. Misurato, non supposto.

    Cosa sostituisce, e perche'
    ---------------------------
    Fino al 3 agosto qui c'era la R10 resa eseguibile: «con la CNN degradata a
    q=0,4 la fusione deve conservare almeno meta' del guadagno». Quel criterio
    rifiutava i coefficienti stimati dalla procedura (conservano il 21%) e
    imponeva la colonna pre-specificata `a=1, b=0,30`, che sul metro esterno
    rende +0,0107 invece di +0,0193: l'assicurazione costava ~0,0086.

    Il premio era pero' tarato su un evento che il progetto ha poi rimisurato.
    La motivazione scritta era «la submission ha consegnato +0,0018 dove la
    stima diceva +0,0155»; ma l'errore standard di quel contrasto sui ~525 esami
    di gara e' 0,0071-0,0081, cioe' **inghiotte l'intera discrepanza**, e la
    causa vera si e' rivelata un'altra — le statistiche di `BatchNorm3d`
    disallineate fuori centro, identificate e rimosse. Un modello di degrado del
    RANGO, che e' cio' che `20_risoluzione.robustezza` simula permutando, non
    descrive uno spostamento sistematico del logit: lo dice quella funzione
    stessa nella propria docstring.

    E il rischio residuo e' piu' stretto di quanto sembrasse. `a<1` toglie
    confidenza all'SBR per far posto alla CNN, quindi scoprirebbe l'SBR il giorno
    in cui la CNN non consegna. Ma se la CNN **fallisce** — pesi non caricabili,
    crop assente, eccezione in predizione — `main.py` non applica `a` affatto:
    `predict_one` restituisce la probabilita' SBR grezza, e il livello 2 del
    ripiego la lascia intatta. La leva morde solo nel caso in cui la rete gira e
    non informa, ed e' esattamente il caso che il metro esterno misura cinque
    volte tenendo fuori un centro intero.

    Il criterio nuovo
    -----------------
    **Su ogni centro tenuto fuori dal metro, la fusione deve battere il modello
    SBR da sola.** E' la stessa domanda della R10 — «e se la CNN non consegnasse
    su un centro nuovo?» — posta a cinque centri veri invece che a una
    permutazione con una probabilita' inventata addosso.

    Il cancello discrimina, che e' la R3: `a=1, b=0,498` — una colonna
    pre-specificata, quindi legittima sotto il criterio vecchio — **perde su due
    centri su cinque** e viene rifiutata; i coefficienti stimati vincono su
    cinque su cinque con peggior centro +0,0047, che e' migliore del peggior
    centro della colonna conservativa (+0,0010). Non e' severo in generale: e'
    severo con cio' che deve bocciare.

    Il numero della R10 resta STAMPATO e non piu' vincolante. La frontiera fra
    guadagno e assicurazione deve restare visibile a chi spedisce; non deve piu'
    decidere al posto di una misura.
    """
    if voxnet_path is None:
        return
    try:
        import importlib.util  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            f"ERRORE: cancello di robustezza non eseguibile ({e.name} assente). "
            f"Un controllo che si salta da solo e' peggio di nessun controllo."
        ) from e

    # I coefficienti si leggono dal sidecar, non dal checkpoint: il pacchettatore
    # gira nel venv locale, che torch non ce l'ha. Quando torch c'e' i due si
    # confrontano — R13: fra due artefatti che devono coincidere, si verifica.
    sidecar = voxnet_path.with_suffix(".fusione.json")
    if not sidecar.is_file():
        raise SystemExit(
            f"ERRORE: manca {sidecar.name}, e senza non si leggono i coefficienti "
            f"di fusione senza torch.\n"
            f"  Lo scrive `10_voxel_cnn.py --salva`. Se questi pesi sono anteriori, "
            f"rigenerali o scrivi il sidecar a mano con a, b e arco."
        )
    fus = json.loads(sidecar.read_text(encoding="utf-8"))
    a, b, arco = fus["a"], fus["b"], fus.get("arco")
    try:
        import torch  # noqa: PLC0415

        blob = torch.load(str(voxnet_path), map_location="cpu", weights_only=True)
        if (abs(blob["fusione"]["a"] - a) > 1e-9
                or abs(blob["fusione"]["b"] - b) > 1e-9):
            raise SystemExit(
                f"ERRORE: il sidecar dice (a={a:.4f}, b={b:.4f}) e il checkpoint "
                f"(a={blob['fusione']['a']:.4f}, b={blob['fusione']['b']:.4f}).\n"
                f"  In inferenza contano quelli del checkpoint: il cancello "
                f"starebbe verificando dei numeri che non verranno usati."
            )
    except ImportError:
        pass

    art = ROOT / "experiments" / "risultati"
    mj, mz = art / f"{metro}.json", art / f"{metro}.npz"
    feats = ROOT / "features_v4.csv"
    mancanti = [str(p) for p in (mj, mz, feats) if not p.is_file()]
    if mancanti:
        raise SystemExit(
            f"ERRORE: cancello non verificabile, manca {', '.join(mancanti)}.\n"
            f"  Serve l'artefatto del metro esterno che ha misurato QUESTI pesi:\n"
            f"    python scripts/16_metro_esterno.py --esterni gruppi --arch "
            f"{fus.get('arch', 'media')} --norm {fus.get('norm', 'batch')} --tta "
            f"--riduci 2 --train-seeds 0-3\n"
            f"  Oppure spedisci senza ramo a voxel (--senza-voxnet)."
        )

    j = json.loads(mj.read_text(encoding="utf-8"))
    z = dict(np.load(mz, allow_pickle=True))

    ramo_tabellare = fus.get("tabular_branch", "incumbent")
    ramo_metro = j.get("tabular_branch", "incumbent")
    if ramo_tabellare != ramo_metro:
        raise SystemExit(
            f"ERRORE: il checkpoint usa il ramo tabellare '{ramo_tabellare}', "
            f"il metro ha misurato '{ramo_metro}'."
        )
    if ramo_tabellare != "incumbent":
        if models_path is None or not models_path.is_file():
            raise SystemExit("ERRORE: manca il bundle del ramo tabellare gold")
        sys.path.insert(0, str(ROOT))
        with open(models_path, "rb") as handle:
            modelli = pickle.load(handle)
        if not modelli.get("fusion_lr"):
            raise SystemExit(
                "ERRORE: il checkpoint gold richiede 'fusion_lr' nel bundle"
            )
        if modelli.get("fusion_lr_C") != [0.1, 0.3]:
            raise SystemExit(
                "ERRORE: il bundle gold non dichiara fusion_lr_C=[0.1, 0.3]"
            )
        if ramo_tabellare == "gold_registered_equal_ensemble_v1":
            from src.alignment import SYMMETRY_FEATURE_NAMES  # noqa: PLC0415

            colonne = modelli.get("fusion_lr_columns", [])
            mancanti = sorted(set(SYMMETRY_FEATURE_NAMES) - set(colonne))
            if mancanti:
                raise SystemExit(
                    "ERRORE: il bundle registrato non contiene le feature "
                    f"dichiarate: {', '.join(mancanti)}"
                )
            if not modelli.get("fusion_lr_base"):
                raise SystemExit("ERRORE: il bundle ensemble non contiene fusion_lr_base")
            if modelli.get("fusion_lr_blend_weight") != 0.5:
                raise SystemExit("ERRORE: il bundle ensemble non dichiara il peso fisso 0.5")

    # R13, prima verifica: il metro deve aver misurato la STESSA rete che si
    # spedisce. Un metro girato su `norm=batch` che autorizza pesi `norm=istanza`
    # sarebbe un cancello che verifica un altro modello - passerebbe, e non
    # direbbe niente su cio' che parte.
    # `autonorm` sta accanto a `norm` e non dopo: sono due parametri che cambiano
    # cio' che la rete vede, e controllarne uno solo lascia aperta esattamente
    # meta' del guasto. Dal 6 agosto sul disco esistono due metri che
    # differiscono SOLO per autonorm — `_cnn_istanza` e `_cnn_istanza_autonorm` —
    # e i pesi corrispondenti hanno nomi altrettanto vicini
    # (`voxnet_istanza.pt`, `voxnet_autonorm.pt`). Scambiarli e' un errore di
    # una lettera che produrrebbe un cancello superato su un modello diverso da
    # quello spedito.
    for chiave, difetto in (("norm", "batch"), ("autonorm", False)):
        v_metro = j.get("config_cnn", {}).get(chiave, difetto)
        v_pesi = fus.get(chiave, difetto)
        if chiave == "autonorm":               # None, 0 e False dicono la stessa cosa
            v_metro, v_pesi = bool(v_metro), bool(v_pesi)
        if v_metro != v_pesi:
            raise SystemExit(
                f"ERRORE: il metro '{metro}' ha misurato {chiave}={v_metro}, "
                f"i pesi sono {chiave}={v_pesi}.\n"
                f"  Il cancello verificherebbe un modello diverso da quello spedito."
            )

    frame = pd.read_csv(feats)
    y = frame["is_pathologic"].to_numpy(dtype=float)
    gruppi = frame["_group"].to_numpy()
    p_sbr, p_cnn = z["solo_sbr"], z["p_cnn"]

    # R13, seconda verifica: le righe di features_v4.csv devono essere
    # nell'ordine con cui il metro ha scritto l'.npz. Se non lo fossero, ogni
    # numero sotto sarebbe plausibile e falso. Si ricalcola una quantita' che
    # l'artefatto riporta gia'.
    atteso = j["complessivo"]["solo_sbr"]["log_loss"]
    ottenuto = float(-np.mean(y * np.log(np.clip(p_sbr, 1e-9, 1))
                              + (1 - y) * np.log(np.clip(1 - p_sbr, 1e-9, 1))))
    if abs(atteso - ottenuto) > 1e-3:
        raise SystemExit(
            f"ERRORE: etichette non allineate con l'artefatto del metro.\n"
            f"  log loss di solo_sbr ricalcolata {ottenuto:.4f}, nel JSON "
            f"{atteso:.4f}."
        )

    sys.path.insert(0, str(ROOT))
    from src.voxnet import fondi  # noqa: PLC0415

    def perdita(mask, p):
        return float(-np.mean(y[mask] * np.log(np.clip(p[mask], 1e-9, 1))
                              + (1 - y[mask]) * np.log(np.clip(1 - p[mask], 1e-9, 1))))

    # I centri tenuti fuori, ricostruiti dai nomi che il metro ha scritto. Il
    # conteggio si VERIFICA contro quello riportato: se il raggruppamento qui e
    # quello del metro divergessero, i guadagni sarebbero calcolati su insiemi
    # diversi da quelli che il JSON descrive.
    nominati = [f["fold"] for f in j["per_fold"] if "x" in f["fold"]]
    fusa = fondi(p_sbr, p_cnn, a, b)
    righe, peggiore = [], None
    for f in j["per_fold"]:
        m = (gruppi == f["fold"]) if "x" in f["fold"] else ~np.isin(gruppi, nominati)
        if int(m.sum()) != f["n"]:
            raise SystemExit(
                f"ERRORE: il centro '{f['fold']}' ha {int(m.sum())} esami qui e "
                f"{f['n']} nell'artefatto: il raggruppamento non coincide."
            )
        g = perdita(m, p_sbr) - perdita(m, fusa)
        righe.append((f["fold"], f["n"], g))
        peggiore = g if peggiore is None else min(peggiore, g)

    print(f"  cancello per centro: a={a:.3f} b={b:.3f}, arco '{arco}', "
          f"metro '{metro}'")
    for nome, n, g in righe:
        print(f"      {'OK ' if g > 0 else 'NO '} {nome:22s} n={n:4d}  {g:+.4f}")
    medio = sum(n * g for _, n, g in righe) / sum(n for _, n, _ in righe)
    print(f"      medio pesato {medio:+.4f} · peggior centro {peggiore:+.4f} · "
          f"vinti {sum(g > 0 for _, _, g in righe)}/{len(righe)}")

    # La R10 resta come numero, non come veto: il prezzo dell'assicurazione va
    # visto da chi spedisce anche quando decide di non comprarla.
    spec = importlib.util.spec_from_file_location(
        "risoluzione", ROOT / "scripts" / "20_risoluzione.py")
    ris = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ris)
    r = ris.robustezza(y, p_sbr, p_cnn, a, b, q=q, ripetizioni=200)
    print(f"      (R10, non vincolante: a q={q} conserva "
          f"{r['quota_conservata']:.0%} del guadagno)")

    if peggiore <= 0:
        perdenti = ", ".join(f"{nome} {g:+.4f}" for nome, _, g in righe if g <= 0)
        raise SystemExit(
            f"ERRORE: la fusione (a={a:.3f}, b={b:.3f}) PERDE su un centro mai "
            f"visto: {perdenti}.\n"
            f"  Il criterio e' che ogni centro tenuto fuori dal metro guadagni "
            f"rispetto al solo SBR. Un centro su cui si perde in misura e' un "
            f"centro su cui si perdera' in gara.\n"
            f"  Rimedio: `20_risoluzione.py --arco {metro}` stampa la frontiera; "
            f"le colonne che il metro riporta gia' sono le uniche non "
            f"selezionate in-sample."
        )


def check_model_deps(archive: Path, pickle_path: Path) -> None:
    """Ogni modulo che il pickle nominera' deve essere dentro l'archivio.

    Il fallimento che questo controllo intercetta e' silenzioso e totale:
    `pickle.load` solleva ModuleNotFoundError, main.py lo cattura per non
    andare in crash, e il container scrive una submission valida in cui OGNI
    esame ha la prevalenza. Nessun errore, punteggio pari alla baseline.
    """
    names = set(zipfile.ZipFile(archive).namelist())
    missing = sorted(
        ref for ref in pickle_module_refs(pickle_path)
        if ref != "src" and f"{ref.replace('.', '/')}.py" not in names
    )
    if missing:
        raise SystemExit(
            f"ERRORE: models.pkl referenzia moduli assenti dall'archivio: "
            f"{', '.join(missing)}. Sono probabilmente esclusi da EXCLUDE. "
            f"Senza di loro il modello non si carica e ogni esame va a prevalenza."
        )


def write_zip(source_dir: Path, archive: Path) -> None:
    """Scrive l'archivio con percorsi relativi e separatori POSIX."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()

    files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.relative_to(source_dir).as_posix())

    names = zipfile.ZipFile(archive).namelist()
    if "main.py" not in names:
        raise SystemExit("ERRORE: main.py non e' nella radice dell'archivio")
    if any("\\" in name for name in names):
        raise SystemExit("ERRORE: separatori backslash nell'archivio")
    if "LICENSE" not in names:
        raise SystemExit("ERRORE: LICENSE assente dall'archivio")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models" / "models.pkl")
    branch = parser.add_mutually_exclusive_group()
    branch.add_argument(
        "--voxnet", type=Path, default=None,
        help="checkpoint voxel scelto esplicitamente (nessun default implicito)",
    )
    branch.add_argument("--senza-voxnet", action="store_true",
                        help="spedisce deliberatamente il solo modello SBR")
    parser.add_argument("--out", type=Path, required=True,
                        help="submission_src/ del repo runtime")
    parser.add_argument("--zip", type=Path, default=None,
                        help="Dove scrivere submission.zip "
                             "(default: <out>/../submission/submission.zip)")
    # Il cancello e' parametrico nel metro ma non disattivabile: un criterio che
    # si puo' spegnere con un flag viene spento la sera prima di una scadenza,
    # che e' esattamente il momento per cui esiste. Si puo' cambiare QUALE metro
    # lo alimenta - un arco nuovo ne produce uno nuovo - non se eseguirlo.
    parser.add_argument("--metro", default=None,
                        help="artefatto di 16_metro_esterno.py che misura questi pesi")
    parser.add_argument("--q-robustezza", type=float, default=0.4,
                        help="degrado della CNN nel numero R10, stampato non vincolante")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.voxnet is None and not args.senza_voxnet:
        parser.error("scegli esplicitamente --voxnet CHECKPOINT oppure --senza-voxnet")
    if args.voxnet is not None and not args.metro:
        parser.error("--voxnet richiede --metro: il checkpoint deve avere una misura associata")
    if args.senza_voxnet and args.metro:
        parser.error("--metro non ha significato insieme a --senza-voxnet")
    return args


def main() -> int:
    args = parse_args()

    if not args.models.is_file():
        print(f"ERRORE: {args.models} non trovato. Esegui prima 02_train_cv.py", file=sys.stderr)
        return 1
    if args.voxnet is not None and not args.voxnet.is_file():
        print(f"ERRORE: checkpoint esplicito non trovato: {args.voxnet}", file=sys.stderr)
        return 1

    voxnet = args.voxnet
    # Tutti i cancelli girano prima di cancellare l'output buono del giro
    # precedente. Un pacchetto rifiutato non deve lasciare una mezza cartella.
    check_fusione_per_centro(
        voxnet, models_path=args.models, metro=args.metro, q=args.q_robustezza
    )
    manifest = build_artifact_manifest(args.models, voxnet, args.metro)

    # Ripulita e non solo sovrascritta: un giro precedente puo' averci lasciato
    # __pycache__ o file di un modello vecchio, che finirebbero nell'archivio.
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    shutil.copy2(ROOT / "submission_src" / "main.py", args.out / "main.py")
    shutil.copy2(args.models, args.out / "models.pkl")
    shutil.copytree(ROOT / "src", args.out / "src", ignore=EXCLUDE)

    if voxnet is not None:
        shutil.copy2(voxnet, args.out / "voxnet.pt")
    # Il regolamento impone la MIT alla soluzione vincente. Nello zip la licenza
    # non e' obbligatoria, ma e' il posto dove un revisore la cerca, e costa
    # un kilobyte: l'archivio si dichiara da solo.
    shutil.copy2(ROOT / "LICENSE", args.out / "LICENSE")
    (args.out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    archive = args.zip or (args.out.parent / "submission" / "submission.zip")
    write_zip(args.out, archive)
    check_manifest(archive)
    check_model_deps(archive, args.models)
    check_voxnet(archive, voxnet)

    total_mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"Submission preparata in {args.out} ({total_mb:.1f} MB)")
    print(f"Archivio scritto in {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    stampa_checklist(args.out, archive, voxnet)
    return 0


def stampa_checklist(out: Path, archive: Path, voxnet: Path | None) -> None:
    """La checklist di invio, stampata dove viene letta.

    Stava nel README, dove si legge quando si aggiorna il README e non quando
    si spedisce. Qui compare nel momento in cui serve, e le voci verificabili
    sono gia' state verificate sopra: `check_model_deps` e `check_voxnet`
    sollevano invece di avvisare, quindi arrivare a questa riga significa che
    sono passate.

    Restano le voci che **nessun controllo automatico puo' vedere**, perche'
    riguardano cio' che succede sulla piattaforma dopo l'invio. Sono elencate
    per ultime di proposito: sono le uniche che chiedono qualcosa a chi legge.
    """
    from submission_src.main import FALLBACK_PROBABILITY

    names = zipfile.ZipFile(archive).namelist()
    posix = not any("\\" in n for n in names)
    radice = "main.py" in names

    print("\n" + "=" * 66)
    print("CHECKLIST DI INVIO")
    print("=" * 66)
    for esito, voce in (
        (radice, "main.py alla radice dell'archivio"),
        (posix, "separatori POSIX (Compress-Archive scrive '\\' e unzip lo "
                "tratta\n         come parte del nome: ogni import fallirebbe "
                "senza dirlo)"),
        (abs(FALLBACK_PROBABILITY - 0.5485) < 5e-4,
         f"FALLBACK_PROBABILITY = {FALLBACK_PROBABILITY} (prevalenza del training)"),
        (True, "check_model_deps: ogni modulo nominato da models.pkl e' nell'archivio"),
        (MANIFEST_NAME in names,
         "artifact_manifest: hash di models.pkl"
         f"{' e voxnet.pt' if voxnet is not None else ''} verificati nello ZIP"),
        (voxnet is not None, "check_voxnet: voxnet.pt e src/voxnet.py entrambi presenti"
                             if voxnet is not None else
                             "ramo a voxel ASSENTE: si spedisce il solo modello SBR"),
        (voxnet is not None,
         "check_fusione_per_centro: la fusione guadagna su ogni centro tenuto\n"
         "         fuori dal metro esterno" if voxnet is not None else
         "cancello sulla fusione non applicabile: nessuna fusione da spedire"),
    ):
        print(f"  [{'x' if esito else ' '}]  {voce}")

    print("\n  Non verificabili da qui — restano a chi invia:")
    print("      -  smoke test sulla piattaforma superato")
    print("      -  nel log di gara, la riga [V] dice 'attivo' e non 'INATTIVO'")
    print("      -  diagnostica in tassi e percentuali, mai conteggi assoluti")
    print("=" * 66)
    print("\nOra, dal repo runtime: just test-submission")


if __name__ == "__main__":
    raise SystemExit(main())
