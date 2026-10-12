#!/usr/bin/env python3
"""Fissa i coefficienti di fusione di un checkpoint, con il motivo accanto.

Perche' esiste
--------------
`10_voxel_cnn.py --salva` stima (a, b) con `stima_fusione` su tutte le OOF. Il
checkpoint puo' pero' dover portare coefficienti diversi — perche' una colonna
pre-specificata e' stata preferita alla stima, o perche' un cancello ha bocciato
la stima — e il rimedio non e' modificare il file a mano. Un file di pesi i cui
coefficienti sono stati cambiati da qualcuno, in un momento imprecisato, per un
motivo che non e' scritto da nessuna parte, e' esattamente l'artefatto che
questo progetto non vuole avere. Qui il cambio lascia una traccia dentro il
checkpoint stesso.

Quali coefficienti sono legittimi
----------------------------------
**Solo quelli che non sono stati scelti guardando il numero che poi si riporta.**
`20_risoluzione.py` stampa la frontiera su una griglia: prenderne il massimo
sarebbe selezione in-sample sullo stesso metro, cioe' la R11 violata nel modo in
cui e' gia' stata violata una volta. Restano legittimi i coefficienti **stimati
dalla procedura** — che nessuno sceglie, li produce una logistica — e le
**colonne pre-specificate** che il metro riporta (`a=1` con b in 0,20 / 0,25 /
0,30 / 0,498).

Sulla configurazione `norm=istanza`, misurata tenendo fuori un centro alla volta:

| coefficienti | guadagno | peggior centro | centri vinti |
|---|---|---|---|
| **stimati dalla procedura (0,667 / 0,575)** | **+0,0193** | **+0,0047** | **5/5** |
| a=1, b=0,30 | +0,0107 | +0,0010 | 5/5 |
| a=1, b=0,498 | +0,0105 | −0,0053 | 3/5 |

La terza riga e' il motivo per cui «pre-specificata» non basta da sola: e' una
colonna decisa in anticipo, e perde su due centri su cinque. Chi decide e'
`check_fusione_per_centro` in `03_pack_submission.py`, e la sua docstring spiega
perche' quel criterio ha sostituito la soglia R10.

Uso (serve torch: dentro l'immagine ufficiale):
    python scripts/21_fissa_fusione.py --pesi models/voxnet_istanza.pt \\
        -a 0.6674 -b 0.5747 --motivo "coefficienti della procedura: sul metro \\
        esterno guadagnano su 5 centri su 5, peggior centro +0,0047"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesi", type=Path, required=True)
    ap.add_argument("-a", type=float, required=True)
    ap.add_argument("-b", type=float, required=True)
    ap.add_argument("--motivo", required=True,
                    help="perche' questi e non quelli stimati. Obbligatorio: "
                         "un coefficiente senza motivo non e' tracciabile")
    args = ap.parse_args()

    import torch

    blob = torch.load(str(args.pesi), map_location="cpu", weights_only=True)
    vecchi = dict(blob["fusione"])
    sidecar = args.pesi.with_suffix(".fusione.json")
    d = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}

    # `stimati_dalla_procedura` sono i coefficienti che `stima_fusione` ha
    # prodotto, e restano quelli PER SEMPRE. Alla prima applicazione coincidono
    # con `vecchi`; alla seconda no, perche' `vecchi` sono gia' i fissati a mano,
    # e sovrascriverli col valore corrente cancellerebbe l'unico numero che dice
    # da dove si e' partiti. Il difetto e' silenzioso — il file resta valido e
    # plausibile — ed e' la stessa forma di guasto per cui questo script esiste.
    stimati = (blob.get("provenienza", {})
               .get("fusione_fissata", {})
               .get("stimati_dalla_procedura")
               or d.get("stimati_dalla_procedura")
               or vecchi)

    blob["fusione"] = {"a": float(args.a), "b": float(args.b)}
    # La storia resta nel file: chi lo aprira' fra sei settimane deve poter
    # vedere che i coefficienti non sono quelli che la procedura ha prodotto, e
    # perche'. E' la stessa ragione per cui `provenienza` esiste gia'.
    storia = blob.setdefault("provenienza", {}).setdefault("fusione_storia", [])
    storia.append({
        "quando": dt.datetime.now().isoformat(timespec="seconds"),
        "da": vecchi, "a": {"a": float(args.a), "b": float(args.b)},
        "motivo": args.motivo,
    })
    blob["provenienza"]["fusione_fissata"] = {
        "quando": dt.datetime.now().isoformat(timespec="seconds"),
        "stimati_dalla_procedura": stimati,
        "motivo": args.motivo,
    }
    torch.save(blob, str(args.pesi))

    d.update({"a": float(args.a), "b": float(args.b),
              "stimati_dalla_procedura": stimati, "motivo": args.motivo})
    d.setdefault("arco", blob.get("provenienza", {}).get("arco"))
    d.setdefault("n_reti", len(blob["state_dicts"]))
    d.setdefault("norm", blob["config"].get("norm", "batch"))
    # `autonorm` accanto a `norm`, perche' il cancello ora confronta entrambi
    # con la configurazione del metro. Il sidecar lo riceve gia' da
    # `10_voxel_cnn.py --salva`; questa riga serve al caso in cui non ci sia —
    # un sidecar scritto a mano, o rigenerato da un checkpoint anteriore alla
    # riga che la chiave la scrive. Senza, il cancello leggerebbe False su pesi
    # autonorm e rifiuterebbe una submission valida.
    d.setdefault("autonorm", bool(blob["config"].get("autonorm", False)))
    sidecar.write_text(json.dumps(d, indent=2), encoding="utf-8")

    print(f"{args.pesi}: fusione (a={vecchi['a']:.4f}, b={vecchi['b']:.4f}) "
          f"-> (a={args.a:.4f}, b={args.b:.4f})")
    print(f"  motivo: {args.motivo}")
    print(f"  aggiornato anche {sidecar.name}")
    print("\nOra `03_pack_submission.py` eseguira' il cancello di robustezza su "
          "questi coefficienti: se non passa, non impacchetta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
