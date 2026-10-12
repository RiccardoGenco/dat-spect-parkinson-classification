#!/usr/bin/env python3
"""Controlli di sanita' sul dataset, da eseguire PRIMA di qualunque training.

Cerca i problemi che non si manifestano come errori ma come rumore silenzioso.
Il piu' insidioso e' l'orientamento: la pipeline assume che dopo la
riorientazione canonica l'asse y sia postero-anteriore, e su questo si regge
tutta la separazione caudato/putamen. Se una parte degli esami ha header
malformati - cosa comune nella SPECT clinica - quelle feature si invertono su
quel sottoinsieme senza alcun errore, diluendo il segnale in un modo che non si
vedrebbe mai guardando la log loss.

L'output e' interamente aggregato: conteggi e statistiche, nessun voxel e
nessun dato di paziente. Puo' essere condiviso liberamente.

Uso:
    python scripts/00_inspect_data.py --data-dir /percorso/data
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv import scanner_fingerprint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=400,
                        help="Quanti esami ispezionare (0 = tutti)")
    args = parser.parse_args()

    import nibabel as nib

    nifti_dir = args.data_dir / "niftis"
    labels_path = args.data_dir / "train_labels.csv"

    if not nifti_dir.is_dir():
        print(f"ERRORE: {nifti_dir} non esiste", file=sys.stderr)
        return 1

    labels = pd.read_csv(labels_path) if labels_path.is_file() else None
    all_files = sorted(nifti_dir.glob("*.nii.gz"))
    print(f"File NIfTI trovati: {len(all_files)}")

    if labels is not None:
        print(f"Righe in train_labels.csv: {len(labels)}")
        print(f"Prevalenza casi patologici: {labels['is_pathologic'].mean():.4f}")
        print(f"  normali={int((labels['is_pathologic']==0).sum())}  "
              f"anomali={int((labels['is_pathologic']==1).sum())}")
        missing = set(labels["uid"]) - {f.name[:-7] for f in all_files}
        if missing:
            print(f"  ATTENZIONE: {len(missing)} uid etichettati senza file immagine")

    files = all_files if args.sample == 0 else all_files[: args.sample]
    print(f"\nIspeziono {len(files)} esami...\n")

    orientations = Counter()
    shapes = Counter()
    spacings = Counter()
    fingerprints = Counter()
    dtypes = Counter()
    stats, failures = [], []

    for path in files:
        try:
            img = nib.load(str(path))
            raw_codes = nib.aff2axcodes(img.affine)
            orientations["".join(raw_codes)] += 1
            dtypes[str(img.get_data_dtype())] += 1

            canonical = nib.as_closest_canonical(img)
            shape = tuple(int(s) for s in canonical.shape[:3])
            zooms = tuple(round(float(z), 2) for z in canonical.header.get_zooms()[:3])
            shapes[shape] += 1
            spacings[zooms] += 1
            # La stessa funzione usata per raggruppare i fold, non una copia:
            # un diagnostico che misura qualcosa di diverso da cio' che la CV
            # fa davvero e' peggio che inutile, perche' rassicura sul numero
            # sbagliato.
            fingerprints[scanner_fingerprint(shape, zooms)] += 1

            data = np.asarray(canonical.dataobj, dtype=np.float32)
            stats.append({
                "min": float(data.min()),
                "max": float(data.max()),
                "p50": float(np.percentile(data, 50)),
                "p999": float(np.percentile(data, 99.9)),
                "n_negative": int((data < 0).sum()),
                "n_nan": int(np.isnan(data).sum()),
            })
        except Exception as exc:  # noqa: BLE001
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))

    print("=" * 62)
    print("1. ORIENTAMENTO  <- il controllo critico")
    print("=" * 62)
    for codes, count in orientations.most_common():
        share = count / max(len(files), 1)
        print(f"  {codes:6s} {count:5d}  ({share:5.1%})")
    if len(orientations) > 1:
        print("\n  Piu' orientamenti presenti. Non e' di per se' un problema:")
        print("  as_closest_canonical li normalizza tutti a RAS. Diventa un")
        print("  problema solo se qualche affine e' degenere o non informativo")
        print("  (vedi il controllo 4).")
    else:
        print("\n  Orientamento unico: nessun rischio di inversione degli assi.")

    print("\n" + "=" * 62)
    print("2. IMPRONTE SCANNER  <- determina se la CV per gruppi e' praticabile")
    print("=" * 62)
    print(f"  Impronte distinte (matrice nel piano + spacing): {len(fingerprints)}")
    print(f"  Shape complete distinte: {len(shapes)}   Spacing distinti: {len(spacings)}")
    print("\n  Le piu' frequenti:")
    for fingerprint, count in fingerprints.most_common(12):
        share = count / max(len(files), 1)
        print(f"    {fingerprint:34s} {count:5d}  ({share:5.1%})")

    if len(fingerprints) < 5:
        print("\n  PROBLEMA: meno di 5 impronte distinte.")
        print("  La CV raggruppata per scanner non e' praticabile: servira'")
        print("  un altro proxy per la struttura tra centri.")
    elif fingerprints.most_common(1)[0][1] / max(len(files), 1) > 0.6:
        print("\n  ATTENZIONE: un'impronta copre oltre il 60% dei dati.")
        print("  La CV per gruppi sara' instabile.")
    else:
        print("\n  Buono: eterogeneita' sufficiente per una CV raggruppata.")

    print("\n" + "=" * 62)
    print("3. INTENSITA'")
    print("=" * 62)
    if stats:
        frame = pd.DataFrame(stats)
        print(frame[["min", "max", "p50", "p999"]].describe().to_string())
        n_neg = int((frame["n_negative"] > 0).sum())
        n_nan = int((frame["n_nan"] > 0).sum())
        print(f"\n  Esami con voxel negativi: {n_neg}")
        print(f"  Esami con NaN: {n_nan}")
        dynamic_range = frame["p999"].max() / max(frame["p999"].min(), 1e-9)
        print(f"  Escursione del p99.9 tra esami: {dynamic_range:.1f}x")
        print("\n  Un'escursione ampia e' attesa e non preoccupa: l'SBR e' un")
        print("  rapporto, quindi e' invariante rispetto al guadagno.")
    print(f"\n  Tipi di dato: {dict(dtypes)}")

    print("\n" + "=" * 62)
    print("4. AFFINE DEGENERI")
    print("=" * 62)
    suspicious = 0
    for path in files[:100]:
        try:
            affine = nib.load(str(path)).affine
            if np.allclose(affine[:3, :3], np.eye(3)) or abs(np.linalg.det(affine[:3, :3])) < 1e-6:
                suspicious += 1
        except Exception:  # noqa: BLE001, S110
            pass
    print(f"  Affine identita' o singolari su 100 esami: {suspicious}")
    if suspicious:
        print("\n  Un affine identita' significa che l'header non contiene vera")
        print("  informazione spaziale: as_closest_canonical non puo' correggere")
        print("  nulla e l'orientamento resta quello dei dati grezzi.")
        print("  Se il numero e' alto, la separazione caudato/putamen non e'")
        print("  affidabile e va verificata con un metodo indipendente.")
    else:
        print("  Nessuno: gli header contengono informazione spaziale valida.")

    print("\n" + "=" * 62)
    print("5. VERSO ANTERIORE DEDOTTO DALLA GEOMETRIA")
    print("=" * 62)
    print("  Confronta l'orientamento dichiarato dall'header con quello dedotto")
    print("  dall'anatomia (il caudato e' anteromediale, il putamen posterolaterale).")
    print("  Un disaccordo diffuso significa header inaffidabili.\n")

    from src.striatum import brain_mask, measure  # noqa: E402
    from src.volume import bounding_box, resample_isotropic, resample_mask  # noqa: E402

    agree = disagree = inconclusive = 0
    confidences = []
    for path in files[:150]:
        try:
            # Stesso percorso di src/pipeline.py, non una versione semplificata:
            # se il diagnostico misurasse una pipeline diversa da quella che
            # produce le feature, il suo verdetto non direbbe nulla su di essa.
            img = nib.as_closest_canonical(nib.load(str(path)))
            data = np.clip(np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32)), 0, None)
            zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
            mask = brain_mask(data, tuple(float(z) for z in zooms))
            box = bounding_box(mask, zooms)
            if box is not None:
                data, mask = data[box], mask[box]
            m = measure(
                resample_isotropic(data, zooms, 2.0),
                2.0,
                mask=resample_mask(mask, zooms, 2.0),
            )
            confidences.append(m.anterior_confidence)
            if m.anterior_confidence < 0.15:
                inconclusive += 1
            elif m.anterior_sign == 1:
                agree += 1
            else:
                disagree += 1
        except Exception:  # noqa: BLE001, S110
            pass

    checked = agree + disagree + inconclusive
    if checked:
        print(f"  Header confermato dalla geometria: {agree:4d}  ({agree/checked:5.1%})")
        print(f"  Header in disaccordo:              {disagree:4d}  ({disagree/checked:5.1%})")
        print(f"  Geometria non conclusiva:          {inconclusive:4d}  ({inconclusive/checked:5.1%})")
        print(f"  Confidenza mediana: {np.median(confidences):.3f}")
        if disagree / checked > 0.02:
            print("\n  RILEVANTE: una parte dei volumi ha l'asse antero-posteriore")
            print("  invertito rispetto all'header. La pipeline lo corregge da sola,")
            print("  ma senza questa correzione le feature caudato/putamen sarebbero")
            print("  state rumore su quel sottoinsieme.")
        elif inconclusive / checked > 0.3:
            print("\n  ATTENZIONE: su molti volumi la geometria non e' conclusiva.")
            print("  Verificare visivamente che lo striato sia localizzato bene.")
        else:
            print("\n  Header affidabili: rischio di inversione silenziosa escluso.")

    if failures:
        print("\n" + "=" * 62)
        print(f"6. FILE ILLEGGIBILI: {len(failures)}")
        print("=" * 62)
        for name, err in failures[:10]:
            print(f"  {name}: {err}")

    print("\nFatto. Questo output e' aggregato e condivisibile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
