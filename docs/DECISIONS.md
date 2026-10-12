# Registro decisionale

Questo file conserva i verdetti utili senza mantenere codice, log e risultati
di ogni ablation. I numeri sono OOF raggruppati salvo indicazione diversa.

## Incumbent

- Ramo immagini: `premip`, primi tre stadi di ResNet18 ImageNet sulle MIP
  assiale, coronale e sagittale del crop cerebrale 64³.
- Ensemble: 3 seed × 5 fold, 15 reti, TTA destra-sinistra.
- Fusione: `sigmoid(0.4464*logit(SBR) + 0.9700*logit(MIP))`.
- OOF fusione: log loss 0.2348, AUROC 0.9663.
- Ultimo pubblico: log loss 0.2675, AUROC 0.9544.

Gli artefatti aggregati minimi sono
`experiments/risultati/voxel_premip_final_s012.json`,
`experiments/risultati/metro_esterno_gruppi_cnn_batch_rid1_premip_imagenet.json`
e `experiments/risultati/audit_errori_premip_s012.json`.

## Come fallisce

- 90 casi sbagliati da entrambi i rami generano il 49,6% della perdita OOF.
- I falsi negativi dominano: 99 casi e 48,5% della perdita.
- La dispersione fra seed e' associata alla perdita, ma usarla come gate o
  calibrazione non migliora la CV raggruppata.
- Il divario fra OOF e pubblico indica generalizzazione imperfetta fra
  protocolli; gli header NIfTI non ricostruiscono in modo affidabile i centri.

## Direzioni chiuse

Sono state rimosse dopo confronti appaiati negativi:

- CNN 3D piu' larga, ensemble di varianti 3D e doppia normalizzazione;
- fette 2,5D, MIP con statistiche massimo/top-k/media e MIP a slab;
- crop striatale stretto, sequenza ordinata di fette e sfocatura fisica;
- BatchNorm congelata durante il fine-tuning;
- correzione col gradiente antero-posteriore, calibrazione globale e gate
  sull'incertezza;
- deduzione dei centri dagli header NIfTI.

Il crop stretto ha raggiunto 0.2803/0.9544 a tre seed ma ha peggiorato la fusione
in tutte le dieci ripetizioni raggruppate. `premipstat` ha prodotto log loss
0.2995; la BatchNorm congelata 0.3053 contro 0.2768 del controllo gemello.

## Gold standard contro capacita' moderna (20 agosto)

Il benchmark pre-specificato in `scripts/37_gold_standard.py` ha confrontato
logistica regolarizzata, SVM RBF, ExtraTrees e boosting poco profondo. SVM e
alberi peggiorano sia da soli sia in fusione. Una logistica sui rapporti molto
piu' regolarizzata (`C=0,1` e `0,3`, mediati senza scegliere il migliore) e'
peggiore da sola ma piu' complementare alla MIP:

- fusione annidata: 0,2339 / 0,9667 contro 0,2374 / 0,9657;
- coefficienti stimati su tutte le OOF: 0,2315 / 0,9674;
- leave-protocol-out pesato: 0,2387 / 0,9648 contro 0,2409 / 0,9647;
- vince sui tre protocolli maggiori, perde di 0,0004 sul quarto e di 0,0089
  sul blocco dei 39 protocolli rari;
- bootstrap appaiato per esami: delta log loss +0,0036, CI95%
  [+0,0003, +0,0069]; macro-gruppo: effetto incerto e mediamente negativo.

Il candidato e' separato dall'incumbent: `models/gold/models.pkl` contiene il
ramo `fusion_lr`, mentre il ripiego senza CNN continua a usare il blend SBR
originale. `models/voxnet_mip_gold.pt` cambia soltanto i coefficienti di
fusione. Il pacchettatore verifica che bundle, sidecar e metro nominino lo
stesso ramo tabellare.

Un EfficientNet-B0 ImageNet a seed gemello ha fallito nettamente (CNN
0,3566 / 0,9303) e aggiungerlo all'incumbent peggiora la fusione a 0,2422.
Anche rotazioni casuali di +/-5 gradi peggiorano la MIP a 0,2835 / 0,9517.
Quindi piu' capacita' 2D e semplice invarianza di posa non spiegano il tetto.

## Registrazione rigida e simmetria voxel-wise (20 agosto)

Il gold standard lasciava aperta una differenza precisa rispetto alla pipeline:
orientamento canonico e centratura non sono registrazione. Il braccio bloccato
in `scripts/39_registration_symmetry.py` ha quindi confrontato gli stessi dieci
descrittori multiscala prima e dopo un allineamento a cinque gradi di liberta',
guidato dalle due ancore striatali e indipendente dagli altri esami.

- senza allineamento: 0,2343 / 0,9668 annidato e 0,2401 / 0,9646 esterno,
  quindi respinto;
- con allineamento: **0,2324 / 0,9672** annidato e **0,2377 / 0,9651**
  leave-protocol-out;
- guadagno annidato presente su 8 assegnazioni dei fold su 8, media +0,00154;
- il blocco dei 39 protocolli rari passa da 0,2883 / 0,9491 a
  **0,2701 / 0,9565**;
- costo misurato nello smoke test: da 0,62 a 0,64 secondi per esame.

La media fissa 1:1 fra logistica gold e logistica registrata migliora ancora
senza stimare un peso sulle OOF: **0,2323 / 0,9673** annidato e
**0,2375 / 0,9652** esterno, con guadagno su 8 fold seed su 8. E' questo il
candidato finale, perche' conserva parte della stabilita' del ramo gold.

Il confronto raw/registered e' la prova causale importante: non basta aggiungere
altre misure di asimmetria, serve prima rendere omologhi i voxel confrontati.
Il candidato promosso resta separato negli artefatti
`models/registered_ensemble/models.pkl` e
`models/voxnet_mip_registered_ensemble.pt`; il bundle di ripiego SBR e la CNN
non vengono modificati.

## Verdetto e prossimo esperimento legittimo

Il gold standard classico e' gia' sostanzialmente saturo: una migliore
regolarizzazione consegna qualche millesimo e raggiunge il target sul metro,
ma non spiega da sola il divario pubblico di circa 0,03. Il salto successivo
deve aggiungere informazione sui falsi negativi ad alto uptake e preservare
l'ordine spaziale perso dalle MIP. La registrazione con simmetria e' il primo
passo che supera il target su entrambi gli assi senza cambiare la CNN. Solo se
il suo riscontro pubblico non confermera' il proxy, il candidato successivo
sara' un ramo multi-piano/multi-slice con aggregazione leggera e
preaddestramento permissivo. Deve passare un seed, tre seed e il
leave-center-out; nessuna submission si prepara soltanto sul miglior numero di
una singola partizione.
