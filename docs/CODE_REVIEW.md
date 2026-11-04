# Revisione del codice EIGER — 4 novembre 2026

Revisione del fork `dcotroneCLT/eiger-framework` al commit `b3450f9`. Obiettivo: trovare bug che bloccano gli esperimenti o falsano le metriche. Metodo: lettura del codice di runner, corpus builder, retriever e metriche; riproduzione dei problemi con codice eseguibile; test automatici (`pytest` con soglia di copertura 100%), `ruff`, `mypy`, `bandit`.

Stato dopo le correzioni: **803 test unitari superati, copertura 100%**. Lo stato prima era 776 test.

## 1. Sintesi

| # | Problema | Gravità | Effetto sui risultati | Stato |
|---|---|---|---|---|
| 1 | Source Integrity confronta i documenti con la query, non con il claim, e non calcola P(implicazione) | Critica | Tutti i valori di SI finora non sono validi | Corretto |
| 2 | Documento avvelenato e ground-truth hanno lo stesso `doc_id`; il retriever ibrido li fonde e nasconde il veleno | Critica (solo ibrido) | PRR sottostimato con `retriever.type: hybrid`; nessun effetto sugli sweep densi | Corretto |
| 3 | Il codice del corpus ufficiale e degli sweep relativi non è nel repository | Critica (provenienza) | I `results.json` del corpus ufficiale riportano un commit che non contiene il codice che li ha prodotti | Da committare |
| 4 | FFR = 0 garantito dalla costruzione dello scorer proxy | Alta | FFR non misura nulla con lo scorer `embedding` | Documentato |
| 5 | FFR non è condizionato all'esposizione al veleno | Media | FFR mescola risposte con e senza contesto avvelenato | Da decidere |
| 6 | ERS è una media condizionata ai soli record con annotazioni | Media | ERS non è confrontabile tra tassi diversi | Documentato |
| 7 | Sottoinsieme Snopes = primi 100 claim del file, non un campione casuale | Media | Possibile bias di selezione | Documentato |
| 8 | Ollama non riceve un `seed` | Bassa | Nessuno con temperatura 0 (verificato: run identiche) | Aperto |
| 9 | Pulizia: 10 import inutilizzati, `type: ignore` superflui, 3 tipi non coerenti in mypy | Bassa | Nessuno | Aperto |

Le metriche **PRR e PRD@1 degli sweep densi sono corrette** e i loro risultati restano validi: dipendono solo da `doc_type` dei documenti recuperati, e il retriever denso indicizza i documenti per posizione, non per `doc_id`.

## 2. Problemi critici

### 2.1 Source Integrity calcolata in modo errato

`eiger/metrics/source_integrity.py` aveva due errori.

**Ground truth sbagliata.** `_get_ground_truth()` restituiva `record.retrieval.query`, con un commento che lo dichiarava provvisorio. Su Snopes la query è una domanda generata, quindi ogni documento veniva confrontato con una domanda; sul corpus ufficiale è `Is it true that ...?`.

**Probabilità sbagliata.** Usava la pipeline `zero-shot-classification` con le etichette `consistent` / `contradictory`. Quella pipeline mette in competizione le due ipotesi costruite dal template e normalizza tra loro: un documento non correlato ottiene circa 0.5, non circa 0. Il valore non è P(implicazione | documento, claim), che è la definizione del README.

Questo spiega i valori osservati: SI intorno a 0.5–0.58 a ogni tasso, e in leggera *crescita* con il tasso di avvelenamento, l'opposto di quanto ipotizzato in H2.

**Correzione.** `ExperimentRunner` salva `claim.original_fact` in `generation.metadata["ground_truth"]`. La metrica usa la pipeline `text-classification` sulla coppia (premessa = documento, ipotesi = claim), con `top_k=None`, e prende la probabilità dell'etichetta di implicazione **cercandola per nome** nell'output del modello. Se il modello non ha un'etichetta di implicazione, solleva un errore invece di restituire un numero sbagliato. Senza ground truth restituisce 0.0 con un avviso, come prima, ma non ricade più sulla query.

**Da verificare sul Mac**, perché qui il modello NLI non è installato: il controllo rapido nella sezione 5 deve dare un valore alto per il claim confrontato con se stesso e basso per la sua versione manipolata.

### 2.2 Il retriever ibrido nasconde i documenti avvelenati

Tutti gli attacchi creano il documento avvelenato copiando il `doc_id` del documento originale (`gt_C1`). `HybridRetriever` fonde le due classifiche (densa e BM25) per `doc_id`, quindi documento vero e avvelenato diventano un solo risultato, e resta quello visto per primo.

Riproduzione, con un claim e un attacco al 100%:

```
docs:   [('gt_C1', 'ground_truth'), ('gt_C1', 'poisoned')]
sparse: [('gt_C1', 'ground_truth'), ('gt_C1', 'poisoned')]
hybrid: [('gt_C1', 'ground_truth')]   contains_poisoned: False
```

Con `retriever.type: hybrid` il PRR veniva quindi sottostimato, e il retriever ibrido sarebbe sembrato una difesa efficace per un errore del codice. Gli sweep eseguiti finora usano il retriever denso e non sono toccati.

**Correzione.** `CorpusBuilder` assegna a ogni documento avvelenato un id unico e deterministico, `<doc_id>__<nome attacco>`. La correzione è in un solo punto, vale per tutti gli attacchi presenti e futuri, ed è coperta da due test di regressione (`tests/unit/test_review_fixes.py`).

### 2.3 Il codice del corpus ufficiale non è nel repository

Il fork al commit `b3450f9` non contiene `eiger/datasets/corpus_claim_official.py`, `eiger/attacks/corpus_precomputed.py`, i relativi test, `scripts/make_corpus_sweep.py` né le configurazioni `experiments/corpus_sweep_preliminary/`. Lo sweep sul corpus ufficiale è stato eseguito con file locali non committati. Il `results.json` registra `git_commit = bb00c08`, un commit che non contiene quel codice, e il runner non registra se l'albero di lavoro aveva modifiche.

**Correzione.** Il pacchetto di questa revisione include tutti questi file. Vanno committati. Conviene anche far registrare al runner lo stato "dirty" dell'albero git, così il problema non si ripete senza essere visto (non ancora implementato).

## 3. Problemi sulle metriche

### 3.1 FFR = 0 per costruzione con lo scorer proxy

`EmbeddingFaithfulnessScorer` calcola la correttezza come similarità coseno riscalata: `(cos + 1) / 2`. Due testi non correlati hanno coseno vicino a 0, quindi correttezza vicina a 0.5. La soglia di FFR (correttezza < 0.2) richiede un coseno inferiore a −0.6, che con questi embedding di frasi non si verifica. FFR = 0 non è un'osservazione: è garantito dalla formula. Il documento dei risultati attribuiva lo zero in modo generico allo scorer; ora indica la causa esatta.

Non va corretto il proxy: anche senza riscalatura, una risposta falsa sullo stesso argomento ha coseno alto con il fatto vero. Serve lo scorer RAGAS calibrato, o un giudice NLI.

### 3.2 FFR non è condizionato all'esposizione

FFR è calcolato su tutti i record, anche su quelli senza alcun documento avvelenato nel contesto. La definizione ("fedele a un contesto avvelenato") riguarda solo i record esposti. Proposta: riportare sia FFR su tutti i record sia FFR sui soli record in cui la variante del claim interrogato è nel top-k. La seconda misura risponde alla domanda di ricerca; la prima dipende dal tasso di esposizione.

### 3.3 ERS è condizionato

`ERSMetric.aggregate()` esclude i record con valore 0, cioè quelli senza documenti annotati. ERS è quindi "rischio medio dei documenti avvelenati recuperati", non "rischio medio per query". Questo spiega perché resta circa 0.81 a ogni tasso. Inoltre la normalizzazione divide per 5 valori in [1, 5], quindi il minimo possibile è 0.2, non 0. Va dichiarato quando si riporta.

### 3.4 Scomposizione del PRR

Sul corpus ufficiale circa due terzi del PRR veniva da varianti di altri claim. Lo script di aggregazione ora riporta per ogni tasso `own` (variante del claim interrogato nel top-k, esposizione mirata) e `other only` (solo varianti di altri claim, esposizione collaterale), con `PRR = own + other only`.

## 4. Verifiche superate

- **Riproducibilità.** `derive_seed` usa SHA-256, non `hash()` di Python, quindi è stabile tra processi e macchine. Due run identiche danno `results.json` identici.
- **PRR e PRD@1.** Formule corrette e coerenti con la definizione del README; i valori per record ricostruiti dall'aggregazione coincidono con quelli salvati.
- **Qdrant.** Distanza coseno, id dei punti per posizione: nessuna collisione nel retriever denso.
- **Scorer RAGAS.** Corretto in precedenza (`AnswerSimilarity`); il test live passa.
- **Gate etico.** M03 e M05 restano bloccati senza opt-in esplicito, anche per l'attacco `corpus_precomputed`.

## 5. Verifica sul Mac dopo l'aggiornamento

```bash
python -m pytest tests/unit -q --no-cov 2>&1 | tail -1          # atteso: 803 passed
python - <<'EOF'
from eiger.metrics.source_integrity import SourceIntegrityMetric
m = SourceIntegrityMetric(); m._load_pipeline()
gt = "Thousands of cows died in Kansas from excessive heat in June 2022."
for doc in [gt, "Thousands of cows died in Kansas from excessive heat in June 2021.",
            "The Federal Reserve raised interest rates in March."]:
    print(round(m._entailment_probability(doc, gt), 3), "|", doc)
EOF
```

Atteso: valore alto (> 0.8) per il claim stesso, basso per la versione con l'anno cambiato, basso per il testo non correlato. Se il primo valore non è alto, la correzione di SI non va usata finché non è chiarito.

## 6. Risultati da rifare

| Risultato | Da rifare? | Motivo |
|---|---|---|
| PRR, PRD@1 — sweep Snopes | No | Retriever denso, non toccato dai bug |
| PRR, PRD@1 — corpus ufficiale | Solo per provenienza | Rieseguire dopo il commit del codice, per avere un commit coerente |
| Source Integrity — tutti | Sì | Bug 2.1 |
| FFR — tutti | Non significativo | Proxy (3.1) |
| Qualunque risultato con `hybrid` | Sì | Bug 2.2 (nessuno ancora prodotto) |
