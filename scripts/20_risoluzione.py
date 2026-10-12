#!/usr/bin/env python3
"""Quanto e' risolvibile un Delta: l'errore standard che nessun metro stampava.

Perche' esiste
--------------
Il progetto ha due strumenti di misura e per nessuno dei due conosceva la
risoluzione. Il metro esterno stampa «guadagno +0,0036»; la leaderboard risponde
«+0,0018»; e non c'era modo di sapere se i due numeri fossero in disaccordo o se
fossero *lo stesso numero letto due volte da uno strumento con la mano che
trema*. Sono la stessa domanda della R6 — un numero non e' una misura finche'
non se ne conosce la larghezza — applicata pero' alla dimensione che la R6 non
copre: non il rumore fra addestramenti, ma quello **fra campioni di esami**.

Le due larghezze sono diverse e vanno sommate, non confuse:

| sorgente | come si misura | dove sta |
|---|---|---|
| seed di addestramento, fold fissi | ripetere l'arco | `voxel_*.json`, `dispersione` |
| **campione di esami, arco fisso** | **questo script** | qui |

Come si calcola, e l'ipotesi sbagliata che sostituisce
-------------------------------------------------------
L'unita' di campionamento e' l'esame. Per ogni esame si prende la **differenza
appaiata** di log loss fra due predizioni, e l'errore standard e' `sd/sqrt(n)`
di quella differenza.

L'ipotesi sbagliata, che questo script esiste per rimpiazzare, e' calcolare
l'errore dalla dispersione delle *log loss* invece che da quella delle
*differenze*: l'SBR e la fusione sbagliano sugli stessi esami, quella
correlazione si cancella nella differenza e non nei livelli, e ignorarla gonfia
l'errore di un ordine di grandezza — abbastanza da dichiarare inconoscibile un
Delta che invece si vede. `tests/test_risoluzione.py` fallisce su quella
versione.

Cosa se ne fa chi decide
-------------------------
1. **|t| accanto a ogni guadagno del metro.** Un guadagno con |t| sotto 2 non e'
   un guadagno: e' un campione di 1362 esami che non esclude lo zero.
2. **La risoluzione della gara**, cioe' lo stesso errore standard proiettato sui
   ~525 esami del test. Dice, *prima* di spendere una submission, se il Delta
   che si vuole leggere e' leggibile. Sotto ~2x quell'errore non lo e', e
   nessun numero di submission lo rende tale: submission identiche danno lo
   stesso punteggio, e ripetere non media niente.
3. **Il contrasto appaiato fra due archi** (`--contro`), che e' la forma giusta
   di un cancello secondo la R5: non «guadagno sopra 0,0080» ma «meglio
   dell'arco che c'e' gia', di X, con |t| sopra 2». Due archi valutati sugli
   stessi esami condividono l'SBR e gran parte della varianza, quindi il
   contrasto e' molto piu' risolvibile del livello — ed e' la ragione per cui il
   metro ordina meglio di quanto misuri.

Cio' che questo script NON dice
--------------------------------
Il rumore fra seed di addestramento: qui l'arco e' fisso. Un cancello onesto
vuole entrambe le larghezze, e la seconda si ottiene rieseguendo l'arco.

Uso:
    python scripts/20_risoluzione.py
    python scripts/20_risoluzione.py --arco metro_esterno_gruppi_cnn_istanza \
                                     --contro metro_esterno_gruppi_cnn_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

EPS = 1e-9


def perdita(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Log loss per esame, non mediata: l'unita' su cui si campiona."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def risoluzione(dif: np.ndarray, n_gara: int,
                centri: np.ndarray | None = None) -> dict:
    """Media, larghezza e |t| di una differenza appaiata di log loss.

    Due errori standard, e il secondo e' quello che decide
    ------------------------------------------------------
    Dividere per la radice del numero di ESAMI presuppone che gli esami siano
    repliche indipendenti della quantita' che deve generalizzare. Non lo sono:
    cio' che si vuole sapere e' se il guadagno tiene su **un centro mai visto**,
    e i centri sono cinque. Un esame in piu' dentro un centro gia' rappresentato
    non aggiunge una replica di quella domanda.

    La differenza non e' accademica, ed e' costata due submission. Sulla fusione
    con `norm=istanza` il guadagno e' +0,0184 con errore 0,0047 contando 1362
    esami (|t| 4,1, apparentemente schiacciante) e **0,0066 con 4 gradi di
    liberta'** contando cinque centri — intervallo al 95% da +0,004 a +0,040. La
    gara ha poi consegnato +0,0027, sotto l'estremo inferiore anche di quello.

    Peggio: la media pesata sugli esami e' dominata dai centri piccoli, dove la
    CNN rende di piu' (+0,043 su 207 esami) che sul centro grande (+0,003 su
    528). Il centro grande e' l'unico paragonabile a un test set vero, ed e'
    l'unico che ha previsto la gara alla terza cifra. Va letto per primo.

    `centri` e' l'etichetta di gruppo per esame. Senza, si riportano solo i
    numeri per esame — e la chiave `se_centri` resta assente invece di contenere
    un valore inventato.
    """
    n = len(dif)
    sd = float(dif.std(ddof=1))
    se = sd / np.sqrt(n)
    out = {
        "delta": float(dif.mean()),
        "sd_per_esame": sd,
        "se": float(se),
        "t": float(abs(dif.mean()) / se) if se > 0 else float("inf"),
        "se_gara": float(sd / np.sqrt(n_gara)),
        "n": int(n),
    }
    if centri is None:
        return out

    unici = [u for u in np.unique(centri) if (centri == u).sum() >= 5]
    per_centro = {str(u): float(dif[centri == u].mean()) for u in unici}
    g = np.array(list(per_centro.values()))
    pesi = np.array([float((centri == u).sum()) for u in unici])
    if len(g) >= 3:
        se_c = float(g.std(ddof=1) / np.sqrt(len(g)))
        # t di Student a len(g)-1 gradi, non normale: con cinque centri il
        # valore critico al 95% e' 2,78 e non 1,96, cioe' l'intervallo e' il 42%
        # piu' largo di quanto una normale direbbe.
        from scipy.stats import t as student  # noqa: PLC0415

        critico = float(student.ppf(0.975, len(g) - 1))
        out.update({
            "per_centro": per_centro,
            "n_centri": len(g),
            "delta_medio_centri": float(g.mean()),
            "se_centri": se_c,
            "t_centri": float(abs(g.mean()) / se_c) if se_c > 0 else float("inf"),
            "ic95_centri": [float(g.mean() - critico * se_c),
                            float(g.mean() + critico * se_c)],
            "peggior_centro": float(g.min()),
            "centri_vinti": int((g > 0).sum()),
            # Il centro piu' grande, che e' la riga da leggere per prima.
            "centro_grande": str(unici[int(np.argmax(pesi))]),
            "delta_centro_grande": float(g[int(np.argmax(pesi))]),
        })
    return out


def carica(dirart: Path, nome: str, y: np.ndarray) -> tuple[dict, dict]:
    """Legge arco + artefatto e **verifica** che le etichette siano allineate.

    R13: fra due artefatti che devono essere confrontabili la confrontabilita'
    si verifica, non si assume. Se l'ordine delle righe di `features` non fosse
    quello con cui il metro ha scritto l'`.npz`, ogni numero qui sotto sarebbe
    plausibile e sbagliato — l'unico modo di accorgersene e' ricalcolare una
    quantita' che l'artefatto riporta gia'.
    """
    j = json.loads((dirart / f"{nome}.json").read_text(encoding="utf-8"))
    z = dict(np.load(dirart / f"{nome}.npz", allow_pickle=True))
    atteso = j["complessivo"]["solo_sbr"]["log_loss"]
    ottenuto = float(perdita(y, z["solo_sbr"]).mean())
    if abs(atteso - ottenuto) > 1e-3:
        raise SystemExit(
            f"{nome}: etichette non allineate con l'artefatto.\n"
            f"  log loss di solo_sbr ricalcolata {ottenuto:.4f}, "
            f"riportata nel JSON {atteso:.4f}.\n"
            f"  Le righe di --features non sono nell'ordine con cui il metro "
            f"ha scritto l'.npz: ogni confronto sarebbe silenziosamente falso."
        )
    return j, z


def spazza_b(y: np.ndarray, ps: np.ndarray, pc: np.ndarray) -> tuple[list, dict]:
    """La curva del peso della fusione, e il suo ottimo col senno di poi.

    Serve a separare due cose che il guadagno da solo non separa: quanto della
    resa viene da segnale nuovo e quanto dal semplice restringimento di un peso
    troppo grande. Se l'ottimo cade sul bordo b -> 0 la fusione sta solo
    limitando i danni; se e' interno, dentro la CNN c'e' qualcosa.

    L'ottimo su (a, b) e' stimato **sugli stessi esami su cui e' valutato**: e'
    un tetto ottimistico, non un risultato. Sta qui per dare al cancello un
    termine di paragone che non puo' essere superato onestamente.
    """
    ls, lc = logit(np.clip(ps, EPS, 1 - EPS)), logit(np.clip(pc, EPS, 1 - EPS))
    base = float(perdita(y, ps).mean())
    curva = []
    for b in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.498, 0.60):
        p = expit(ls + b * lc)
        curva.append({"b": b, "log_loss": float(perdita(y, p).mean()),
                      "guadagno": base - float(perdita(y, p).mean()),
                      "auroc": float(roc_auc_score(y, p))})
    r = minimize(lambda t: float(perdita(y, expit(t[0] * ls + t[1] * lc)).mean()),
                 x0=[1.0, 0.2], method="Nelder-Mead")
    tetto = {"a": float(r.x[0]), "b": float(r.x[1]), "log_loss": float(r.fun),
             "guadagno": base - float(r.fun)}
    return curva, tetto


def spazza_a(y: np.ndarray, ps: np.ndarray) -> list:
    """`a` da solo, con b=0: la fusione a leva e' calibrazione o scommessa?

    Il metro stima sistematicamente `a` fra 0,69 e 0,77, cioe' restringe il logit
    dell'SBR. Due letture, e non sono la stessa cosa:

    - **calibrazione**: l'SBR e' troppo confidente su un centro mai visto, e
      restringerlo aiuta *comunque*, CNN o no. Sarebbe un guadagno gratis, e
      contraddirebbe «siamo sul pavimento di Bayes» — misurato pero' sulla CV
      interna, dove ogni centro e' rappresentato in addestramento. La R5 dice di
      non fidarsi di una soglia fuori dalla domanda su cui e' tarata.
    - **leva**: `a<1` toglie confidenza all'SBR *per farne posto* alla CNN, e
      paga solo se la CNN consegna.

    Le separa una misura sola: il guadagno di `a` con `b=0`. Misurato su entrambe
    le partizioni del metro, e' **monotono negativo** — a=0,85 rende -0,0037 a
    gruppi e -0,0047 a strati. Quindi e' leva, la calibrazione non ha niente da
    dare nemmeno fuori centro, e la R10 ha ragione a punirla: il giorno in cui la
    CNN rende meta', `a<1` lascia scoperto l'unico modello che funziona.
    """
    zs = logit(np.clip(ps, EPS, 1 - EPS))
    base = float(perdita(y, ps).mean())
    return [{"a": a, "guadagno": base - float(perdita(y, expit(a * zs)).mean())}
            for a in (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70)]


def robustezza(y: np.ndarray, ps: np.ndarray, pc: np.ndarray, a: float, b: float,
               q: float = 0.4, ripetizioni: int = 40, seed: int = 0) -> dict:
    """R10: quanto del guadagno sopravvive se la CNN rende una frazione (1-q).

    Il degrado e' quello di `15_diagnosi_fusione.py`: si mescola il logit della
    CNN con una sua permutazione, cosi' la distribuzione marginale resta identica
    e a peggiorare e' **solo** l'ordinamento. Non serve a prevedere niente: serve
    a sapere se la fusione scelta e' quella che, il giorno in cui il ramo nuovo
    rende meta' di quanto promette, perde meta' del guadagno invece di tutto.

    Nota su cosa questo *non* copre piu'. Il modello a `q` e' nato per spiegare
    la submission 2 come «la CNN perde potere discriminante nel container». La
    causa vera si e' poi rivelata un'altra — le statistiche di BatchNorm
    disallineate su un centro mai visto, che spostano il logit senza toccarne
    l'ordine. Un modello di degrado del rango non descrive uno spostamento
    sistematico. La R10 resta come assicurazione a basso costo; la diagnosi che
    l'aveva motivata no.
    """
    rng = np.random.default_rng(seed)
    zs, zc = logit(np.clip(ps, EPS, 1 - EPS)), logit(np.clip(pc, EPS, 1 - EPS))
    base = float(perdita(y, ps).mean())
    pieno = base - float(perdita(y, expit(a * zs + b * zc)).mean())
    resa = []
    for _ in range(ripetizioni):
        zm = (1 - q) * zc + q * zc[rng.permutation(len(zc))]
        resa.append(base - float(perdita(y, expit(a * zs + b * zm)).mean()))
    return {"a": a, "b": b, "q": q, "guadagno_pieno": pieno,
            "guadagno_degradato": float(np.mean(resa)),
            "sd": float(np.std(resa, ddof=1)),
            "quota_conservata": float(np.mean(resa)) / pieno if pieno > 0 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("experiments/risultati"))
    ap.add_argument("--arco", default="metro_esterno_gruppi",
                    help="nome base dell'artefatto del metro, senza estensione")
    ap.add_argument("--contro", default=None,
                    help="secondo arco: stampa il contrasto appaiato fra i due")
    ap.add_argument("--features", type=Path, default=Path("features_v4.csv"))
    ap.add_argument("--etichetta", default="is_pathologic")
    ap.add_argument("--n-gara", type=int, default=525,
                    help="taglia del test set, per proiettare la risoluzione")
    ap.add_argument("--cancello", type=float, default=0.0080)
    ap.add_argument("--q", type=float, default=0.4,
                    help="degrado della CNN per la verifica R10")
    args = ap.parse_args()

    frame = pd.read_csv(args.features)
    y = frame[args.etichetta].to_numpy()
    ja, za = carica(args.dir, args.arco, y)

    # I centri, ricostruiti dalla partizione che il metro ha davvero usato e non
    # dal CSV: le impronte rare sono raccolte in un fold unico, e contarle come
    # centri separati darebbe gradi di liberta' che non esistono. Il conteggio si
    # verifica contro quello riportato (R13).
    centri = None
    if "_group" in frame.columns and ja.get("per_fold"):
        g = frame["_group"].to_numpy().astype(object)
        nominati = [f["fold"] for f in ja["per_fold"] if "x" in f["fold"]]
        raro = next((f["fold"] for f in ja["per_fold"] if "x" not in f["fold"]), None)
        if raro is not None:
            g = np.where(np.isin(g, nominati), g, raro)
        ok = all(int((g == f["fold"]).sum()) == f["n"] for f in ja["per_fold"])
        if ok:
            centri = g
        else:
            print("ATTENZIONE: i gruppi del CSV non riproducono i fold "
                  "dell'artefatto; l'errore sui centri non viene riportato.")

    print(f"ARCO  {args.arco}   n={len(y)}   norm={ja['config_cnn'].get('norm', 'batch')}"
          f"   seed={ja['train_seeds']}")
    print()
    print(f"{'fusione':<26}{'log loss':>10}{'guadagno':>10}"
          f"{'se esami':>10}{'|t|':>6}{'se CENTRI':>11}{'|t|c':>6}"
          f"{'peggior c.':>11}{'vinti':>7}{'grande':>9}")
    print("-" * 108)
    lls = perdita(y, za["solo_sbr"])
    righe = {}
    for nome in za:
        if nome in ("p_cnn", "solo_sbr"):
            continue
        r = risoluzione(lls - perdita(y, za[nome]), args.n_gara, centri)
        righe[nome] = r
        coda = ""
        if "se_centri" in r:
            vinti = f"{r['centri_vinti']}/{r['n_centri']}"
            coda = (f"{r['se_centri']:>11.4f}{r['t_centri']:>6.2f}"
                    f"{r['peggior_centro']:>+11.4f}{vinti:>7}"
                    f"{r['delta_centro_grande']:>+9.4f}")
        print(f"{nome:<26}{float(perdita(y, za[nome]).mean()):>10.4f}"
              f"{r['delta']:>+10.4f}{r['se']:>10.4f}{r['t']:>6.2f}{coda}")

    if centri is not None:
        print("\nLa colonna che decide e' 'grande': il guadagno sul centro piu'")
        print("popoloso, l'unico paragonabile a un test set vero. La media pesata")
        print("sugli esami e' dominata dai centri piccoli, dove la CNN rende di")
        print("piu', ed e' cio' che ha fatto sbagliare due submission.")
        print("L'errore sui centri usa la t di Student: con 5 centri il valore")
        print("critico al 95% e' 2,78 e non 1,96.")

    curva, tetto = spazza_b(y, za["solo_sbr"], za["p_cnn"])
    for c in curva:
        r = robustezza(y, za["solo_sbr"], za["p_cnn"], 1.0, c["b"], q=args.q)
        c["guadagno_degradato"] = r["guadagno_degradato"]
        c["quota_conservata"] = r["quota_conservata"]
    print()
    print("peso della fusione, a=1 fisso:")
    print("  b        " + "".join(f"{c['b']:>8.3f}" for c in curva))
    print("  guadagno " + "".join(f"{c['guadagno']:>+8.4f}" for c in curva))
    print(f"  a q={args.q}   " + "".join(f"{c['guadagno_degradato']:>+8.4f}" for c in curva))
    print("  R10      " + "".join(
        f"{c['quota_conservata']:>7.0%} " if c["guadagno"] > 1e-5 else f"{'-':>8}"
        for c in curva))
    interno = max(range(len(curva)), key=lambda i: curva[i]["guadagno"])
    sul_bordo = interno in (0, len(curva) - 1)
    lettura = ("sul bordo: la fusione sta limitando i danni, non aggiungendo segnale"
               if sul_bordo else "interno: dentro la CNN c'e' segnale, non solo restringimento")
    print(f"  ottimo a b={curva[interno]['b']:.3f} ({lettura})")
    print(f"  tetto col senno di poi (a,b)=({tetto['a']:.2f}, {tetto['b']:.2f}) "
          f"-> guadagno {tetto['guadagno']:+.4f}  [ottimistico: stimato dove e' valutato]")

    # I coefficienti che andrebbero davvero in inferenza sono la mediana di
    # quelli che il metro ha stimato fold per fold: e' cio' che `stima_fusione`
    # produrra' sull'insieme, non un punto della curva qui sopra.
    a_med = float(np.median([f["a_stimato"] for f in ja["per_fold"]]))
    b_med = float(np.median([f["b_stimato"] for f in ja["per_fold"]]))
    rob = robustezza(y, za["solo_sbr"], za["p_cnn"], a_med, b_med, q=args.q)
    curva_a = spazza_a(y, za["solo_sbr"])
    print()
    print("`a` da solo, b=0 (leva o calibrazione?):")
    print("  a        " + "".join(f"{c['a']:>8.2f}" for c in curva_a))
    print("  guadagno " + "".join(f"{c['guadagno']:>+8.4f}" for c in curva_a))
    print("  monotono negativo = e' LEVA sulla CNN, non calibrazione dell'SBR")
    print()
    print(f"R10 — robustezza a q={args.q} con (a,b)=({a_med:.2f}, {b_med:.2f}), "
          f"i coefficienti mediani stimati dal metro:")
    print(f"  guadagno pieno {rob['guadagno_pieno']:+.4f} -> degradato "
          f"{rob['guadagno_degradato']:+.4f} +- {rob['sd']:.4f}   "
          f"conservato {rob['quota_conservata']:.0%}   "
          f"[{'PASSA' if rob['quota_conservata'] >= 0.5 else 'NON PASSA'}: la soglia e' 50%]")

    fuori = {}
    for f in ja["per_fold"]:
        fuori[f["fold"]] = (f["n"], f["cnn_auroc_out"], f["sbr_auroc"])
    print()
    print("AUROC sul centro tenuto fuori:")
    for k, (n, c, s) in fuori.items():
        print(f"  {k:<30}n={n:>4}   CNN {c:.4f}   SBR {s:.4f}   {c - s:+.4f}")
    tot = sum(n for n, _, _ in fuori.values())
    print(f"  {'media pesata':<30}n={tot:>4}   "
          f"CNN {sum(n * c for n, c, _ in fuori.values()) / tot:.4f}   "
          f"SBR {sum(n * s for n, _, s in fuori.values()) / tot:.4f}")

    contrasto = None
    if args.contro:
        jb, zb = carica(args.dir, args.contro, y)
        print()
        print(f"CONTRASTO APPAIATO   {args.arco}  contro  {args.contro}")
        print(f"  (norm {ja['config_cnn'].get('norm', 'batch')} contro "
              f"{jb['config_cnn'].get('norm', 'batch')}, "
              f"fold_seed {ja['fold_seed']}/{jb['fold_seed']}, "
              f"seed {ja['train_seeds']}/{jb['train_seeds']})")
        print()
        print(f"{'fusione':<26}{'arco':>10}{'contro':>10}{'Delta':>10}"
              f"{'se esami':>10}{'|t|':>6}{'se CENTRI':>11}{'|t|c':>6}"
              f"{'vinti':>7}{'grande':>9}")
        print("-" * 105)
        contrasto = {}
        for nome in za:
            if nome in ("p_cnn", "solo_sbr") or nome not in zb:
                continue
            r = risoluzione(perdita(y, zb[nome]) - perdita(y, za[nome]),
                            args.n_gara, centri)
            contrasto[nome] = r
            coda = ""
            if "se_centri" in r:
                vinti = f"{r['centri_vinti']}/{r['n_centri']}"
                coda = (f"{r['se_centri']:>11.4f}{r['t_centri']:>6.2f}"
                        f"{vinti:>7}{r['delta_centro_grande']:>+9.4f}")
            print(f"{nome:<26}{righe[nome]['delta']:>+10.4f}"
                  f"{float(perdita(y, zb[nome]).mean() - perdita(y, zb['solo_sbr']).mean()):>+10.4f}"
                  f"{r['delta']:>+10.4f}{r['se']:>10.4f}{r['t']:>6.2f}{coda}")
        print()
        print("  Il Delta e' quanto l'arco guadagna SULL'ALTRO, appaiato esame per")
        print("  esame. E' la quantita' su cui va posto un cancello (R5), perche' un")
        print("  livello assoluto porta dentro tutta la varianza dell'SBR.")

    print()
    print(f"cancello del piano: {args.cancello:+.4f}")
    n_serve = (2 * righe[max(righe, key=lambda k: righe[k]['delta'])]['sd_per_esame']
               / args.cancello) ** 2
    print(f"  per leggerlo a |t|=2 servono ~{n_serve:.0f} esami; il metro ne ha "
          f"{len(y)}, la gara {args.n_gara}.")
    print(f"  larghezza dei soli seed di addestramento: non e' qui, sta in "
          f"voxel_*.json -> dispersione. Vanno sommate in quadratura.")

    out = args.dir / f"risoluzione_{args.arco}.json"
    out.write_text(json.dumps({
        "arco": args.arco, "contro": args.contro, "n": len(y), "n_gara": args.n_gara,
        "norm": ja["config_cnn"].get("norm", "batch"),
        "risoluzione": righe, "curva_b": curva, "tetto_senno_di_poi": tetto,
        "contrasto": contrasto, "robustezza_r10": rob, "curva_a": curva_a,
        "auroc_fuori": {k: {"n": n, "cnn": c, "sbr": s} for k, (n, c, s) in fuori.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nScritto {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
