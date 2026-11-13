# Esperimenti EIBench: documentazione e risultati

Stato al 4 novembre 2026, dopo la revisione del codice (`docs/CODE_REVIEW.md`). Sostituisce le versioni del 29 ottobre e del 3 novembre. Descrive gli esperimenti eseguiti, i risultati e i limiti con cui vanno letti. Per l'ambiente e la riproduzione vedi `ENVIRONMENT.md` e `docs/REPRODUCING.md`.

## 1. Sintesi

Lo sweep su Snopes è stato rieseguito dopo la revisione del codice del 4 novembre, che ha trovato un errore importante: metà dei documenti "avvelenati" erano copie identiche del documento vero, prodotte da attacchi che non avevano trovato nulla da cambiare. Ora sono esclusi dal corpus. I numeri del 3 novembre (PRR 0.82 al 10%) erano quindi sovrastimati e non vanno più citati.

Con i soli quattro attacchi del benchmark (M01, M02, M04, M06), cinque seed e il codice corretto, la quota di query con almeno un documento avvelenato nel top-5 (PRR) cresce in modo monotono con il tasso di avvelenamento, da 0.00 a **0.54 al 10%** (IC 95% 0.49–0.58). Il risultato resta coerente con H1 (RQ1), con un effetto più piccolo di quanto riportato prima.

Due risultati nuovi, comuni ai due corpus:

- **L'esposizione è soprattutto collaterale.** Al 10%, su Snopes, 0.16 delle query contengono la variante avvelenata del proprio claim e 0.37 contengono solo varianti di *altri* claim. Sul corpus ufficiale il rapporto è simile (0.10 contro 0.17).
- **Quando un claim è avvelenato, la sua variante entra quasi sempre nel contesto**, ma raramente al primo posto.

Limiti invariati: FFR non è misurabile con lo scorer proxy (sezione 5); la Source Integrity, ora calcolata correttamente, ha un problema di definizione (sezione 3.2); l'esperimento sul corpus ufficiale (sezione 4) è preliminare.

## 2. Setup comune

| Componente | Valore |
|---|---|
| Dataset (sez. 3) | Snopes verified-true ripulito (`data/snopes/snopes_enriched.json`, 2.928 claim, 0 contaminati), primi 100 claim, split `test` |
| SHA-256 del dataset | identico sulle due macchine (registrato in `reference/data_hashes.txt`) |
| Attacchi | M01 `numerical_shift`, M02 `date_manipulation`, M04 `causal_manipulation`, M06 `missing_context`; M03 e M05 esclusi, `allow_non_benchmark_attacks: false` |
| Tasso di avvelenamento | 0, 1, 3, 5, 10% (applicato indipendentemente a ogni attacco) |
| Seed | 42, 43, 44, 45, 46 (25 run in totale) |
| Retriever | Denso, `sentence-transformers/all-MiniLM-L6-v2`, top-k = 5 |
| Vector store | Qdrant server 1.9.4 (Docker), `qdrant-client` 1.9.2 |
| Generazione | `llama3.1:8b` via Ollama 0.40.13 (server), temperatura 0.0, max 512 token |
| Scorer di faithfulness | `EmbeddingFaithfulnessScorer` (proxy a similarità coseno) |
| Ambiente Python | 3.12.16, `requirements/lock-darwin-arm64.txt` (hash SHA-256): numpy 1.26.4, pyarrow 25.0.1, torch 2.15.0, transformers 4.57.6, sentence-transformers 3.4.1, ragas 0.2.15 |
| Macchina | Mac-116, Apple Silicon (darwin-arm64) |
| Commit | `270c9d6` (configurazioni generate da `scripts/make_benchmark_sweep.py`; correzioni della revisione incluse) |

Verifiche prima dello sweep: 776 test unitari e 5 test di integrazione superati (incluso il test live con Qdrant e Ollama). Due run identiche di `baseline_v1` e di `0pct_seed42` producono `results.json` identici, escluso il timestamp.

## 3. Risultati: sweep del tasso di avvelenamento (Snopes)

### 3.1 Metriche di retrieval

Commit `270c9d6`. Media ± deviazione standard sui cinque seed; IC 95% con bootstrap (10.000 ricampionamenti) sui 500 valori per record di ogni tasso.

| Tasso | PRR (≥1 avvelenato in top-5) | IC 95% | PRD@1 (avvelenato al rank 1) | IC 95% |
|---|---|---|---|---|
| 0% | 0.000 ± 0.000 | [0.000, 0.000] | 0.000 ± 0.000 | [0.000, 0.000] |
| 1% | 0.116 ± 0.048 | [0.088, 0.144] | 0.006 ± 0.005 | [0.000, 0.014] |
| 3% | 0.254 ± 0.030 | [0.216, 0.292] | 0.012 ± 0.008 | [0.004, 0.022] |
| 5% | 0.342 ± 0.045 | [0.300, 0.384] | 0.020 ± 0.014 | [0.010, 0.032] |
| 10% | 0.538 ± 0.016 | [0.494, 0.582] | 0.046 ± 0.029 | [0.028, 0.064] |

Scomposizione del PRR (vedi anche sezione 4.5):

| Tasso | PRR | variante del claim interrogato (mirata) | solo varianti di altri claim (collaterale) |
|---|---|---|---|
| 1% | 0.116 | 0.020 | 0.096 |
| 3% | 0.254 | 0.050 | 0.204 |
| 5% | 0.342 | 0.082 | 0.260 |
| 10% | 0.538 | 0.164 | 0.374 |

Lettura. L'esposizione cresce in modo monotono e gli intervalli di tassi adiacenti non si sovrappongono. Circa due terzi dell'esposizione sono collaterali: un attacco su pochi claim contamina anche le risposte su claim non attaccati, semanticamente vicini. Il documento avvelenato arriva al primo posto in meno del 5% delle query anche al 10%.

Confronto con le versioni precedenti, al 10%: PRR 0.77 (29 ottobre, un seed, sei attacchi), 0.82 (3 novembre, con le copie identiche), 0.54 (ora). La differenza tra le ultime due è dovuta interamente alle copie identiche escluse.

### 3.2 Metriche diagnostiche (non sono risultati)

| Tasso | Source Integrity | ERS | FFR (proxy) |
|---|---|---|---|
| 0% | 0.189 ± 0.000 | 0.000 ± 0.000 | 0.000 |
| 1% | 0.191 ± 0.002 | 0.796 ± 0.033 | 0.000 |
| 3% | 0.194 ± 0.002 | 0.784 ± 0.011 | 0.000 |
| 5% | 0.198 ± 0.002 | 0.779 ± 0.011 | 0.000 |
| 10% | 0.207 ± 0.004 | 0.777 ± 0.009 | 0.000 |

**Source Integrity.** Ora è calcolata come previsto dal README: media su top-5 di P(implicazione | documento, claim), con il claim come riferimento (prima della revisione usava la query, ed era errata). Il calcolo è verificato: 0.992 per il claim confrontato con se stesso, 0.084 con l'anno cambiato, 0.0 per un testo non correlato. Però la metrica **sale** leggermente con l'avvelenamento. Il motivo è la definizione: un documento su un altro claim e un documento che contraddice il claim ricevono entrambi circa 0, quindi il valore di base è circa 0.2 (il solo documento vero su cinque), e una variante avvelenata, simile al claim, riceve un po' di implicazione e fa salire la media. La definizione va rivista (proposta nella sezione 7).

**ERS** è la media delle annotazioni fissate dal disegno degli attacchi, calcolata solo sui record con documenti annotati: è quasi costante per costruzione. **FFR** è 0 per costruzione dello scorer proxy (sezione 5).

### 3.3 Avvertenze di lettura

- **Tasso nominale e manipolazioni effettive.** Ogni attacco si applica in modo indipendente, e circa metà delle applicazioni non produce alcuna modifica (soprattutto l'attacco numerico, perché molti claim non contengono cifre): queste applicazioni sono escluse dal corpus. Media sui cinque seed, 100 claim per run:

  | Tasso nominale | Documenti davvero modificati | Applicazioni senza effetto (escluse) | Quota del corpus avvelenata |
  |---|---|---|---|
  | 0% | 0.0 | 0.0 | 0.0% |
  | 1% | 2.0 | 2.8 | 2.0% |
  | 3% | 5.2 | 6.0 | 4.9% |
  | 5% | 8.4 | 9.6 | 7.7% |
  | 10% | 17.6 | 21.0 | 15.0% |

  La quota del corpus avvelenata è circa una volta e mezza il tasso nominale, perché i quattro attacchi si sommano. Quando si riporta il PRR, va indicata accanto al tasso nominale: al 10% nominale, il 15% dei documenti del corpus è avvelenato. Fonte: campi `n_poisoned`, `skipped_noop` e `poison_ratio` della riga `corpus_builder.complete` nei log.

- **Query che ripetono il valore vero.** 17 delle 100 query Snopes ripetono un numero o un anno presente nel claim. Quando l'attacco cambia proprio quel valore, la query favorisce il documento vero, e il PRD@1 è sottostimato. La scelta di un prompt che eviti numeri e date è in discussione con la collega.
- **Intervallo di confidenza.** Il bootstrap tratta i 500 record di un tasso come indipendenti, ma i cinque seed usano gli stessi 100 claim; la deviazione standard tra seed è la misura più prudente.
- **Parità tra macchine.** Tutte le run sono su Mac-116. La verifica su un'altra macchina non è ancora stata fatta.

## 4. Esperimento sul corpus ufficiale in git (preliminare)

### 4.1 Cosa cambia

Il corpus della collega è versionato in `corpus_claim_official/`: 3.638 claim originali (`Original_Claims/`) e, per ognuno, la sua manipolazione controllata (`Modified_Claims/`), con categoria e elemento modificato. Due componenti nuovi lo rendono utilizzabile:

- dataset `corpus_claim_official` e `corpus_claim_official_unreviewed` (`eiger/datasets/corpus_claim_official.py`);
- attacco `corpus_precomputed` (`eiger/attacks/corpus_precomputed.py`), che usa come documento avvelenato la manipolazione già prodotta dalla collega invece di un attacco meccanico.

Ogni claim ha una sola variante manipolata, quindi qui il tasso nominale coincide con la quota attesa di claim avvelenati: la prima avvertenza della sezione 3.3 non si applica.

### 4.2 Quali righe entrano

| Filtro | Righe escluse |
|---|---|
| Etichetta diversa da `verified_true` | 1.951 |
| Lingua diversa da inglese | 3 |
| Non revisionate da un umano (solo per `corpus_claim_official`) | 1.659 |

Restano 1.684 claim verified-true in inglese. Di questi, solo **25** sono revisionati (`kept_after_manual_audit` o `approved_final_version`); i 230 `corrected_after_manual_audit` non contano come revisionati, perché il loro output è ancora da ricontrollare. Le categorie presenti sono M06 (814), M01 (581), M02 (287) e M04 (2): M04 è quasi assente.

I claim `verified_false` sono esclusi perché il documento ground-truth è il testo del claim stesso: per un claim falso servirebbe un documento correttivo, che il corpus non contiene.

### 4.3 Configurazione

25 run (5 tassi × 5 seed) su 300 claim del dataset `corpus_claim_official_unreviewed`, generati da `scripts/make_corpus_sweep.py`. Il sottoinsieme di 300 claim è fisso (ordinato per SHA-256 di `item_id`, così mescola Snopes e PolitiFact); il seed cambia solo quali claim ricevono la variante avvelenata. Il resto del setup è quello della sezione 2.

Tutti i risultati di questo esperimento vanno etichettati **"preliminare, corpus non ancora validato"**, come richiede il protocollo di allineamento (punto 6). Con soli 25 claim revisionati, una versione paper-facing (`--reviewed-only`) non ha ancora una dimensione utile.

### 4.4 Risultati (preliminari, corpus non ancora validato)

25 run, commit `cd95b48` (rieseguito dopo la revisione, risultati identici alla prima esecuzione), 300 claim per run, 1.500 record per tasso. Media ± deviazione standard sui cinque seed; IC 95% con bootstrap sui record.

| Tasso | PRR | IC 95% | PRD@1 | IC 95% |
|---|---|---|---|---|
| 0% | 0.000 ± 0.000 | [0.000, 0.000] | 0.000 ± 0.000 | [0.000, 0.000] |
| 1% | 0.050 ± 0.040 | [0.040, 0.061] | 0.001 ± 0.001 | [0.000, 0.002] |
| 3% | 0.100 ± 0.068 | [0.085, 0.115] | 0.001 ± 0.002 | [0.000, 0.003] |
| 5% | 0.148 ± 0.037 | [0.131, 0.167] | 0.002 ± 0.003 | [0.000, 0.005] |
| 10% | 0.261 ± 0.030 | [0.239, 0.283] | 0.006 ± 0.004 | [0.003, 0.010] |

Diagnostiche (codice corretto): Source Integrity 0.201–0.205, con lo stesso problema di definizione della sezione 3.2; ERS 0.000 (annotazioni assenti nel corpus, non inventate); FFR 0.000 (proxy).

### 4.5 Scomposizione del PRR

Il PRR conta qualunque documento avvelenato nel top-5. Sui `results.json` è stato separato il caso in cui il documento avvelenato è la variante del claim interrogato da quello in cui appartiene a un altro claim.

| Tasso | Documento vero del claim al rank 1 | Variante del claim nel top-5 | di cui al rank 2 | Solo varianti di altri claim |
|---|---|---|---|---|
| 1% | 0.999 | 0.015 | 21 su 22 | 0.035 |
| 5% | 0.998 | 0.050 | 72 su 75 | 0.098 |
| 10% | 0.994 | 0.096 | 135 su 144 | 0.165 |

Tre osservazioni.

1. **Quando un claim è avvelenato, la sua variante entra quasi sempre nel contesto.** La quota di query con la propria variante nel top-5 coincide con la quota attesa di claim avvelenati (circa 1%, 5%, 10%). Il retrieval non filtra la manipolazione.
2. **La variante resta quasi sempre al secondo posto, dietro l'originale.** Il documento vero è al rank 1 in oltre il 99% delle query. La causa è la query: il template `Is it true that <claim>?` contiene il testo originale, quindi l'originale è quasi identico alla query. Il PRD@1 vicino a zero misura questo vantaggio, non la robustezza del retrieval. La previsione iniziale (PRD@1 più alto che su Snopes) era sbagliata.
3. **Circa due terzi del PRR al 10% vengono da varianti di altri claim** (0.165 su 0.261). Sono documenti avvelenati di claim semanticamente vicini, che contaminano query su cui non c'è stato un attacco mirato. È un'esposizione collaterale, distinta da quella mirata.

### 4.6 Perché non è confrontabile con Snopes

- **Query**: generata da un LLM su Snopes, template con il testo originale qui (osservazione 2).
- **Quota di documenti avvelenati**: una variante per claim qui; quattro attacchi indipendenti su Snopes, circa quattro volte tanti documenti avvelenati allo stesso tasso nominale.
- **Dimensione**: 300 claim qui, 100 su Snopes.

Il risultato utilizzabile di questo esperimento non è il valore di PRR, ma la scomposizione: la manipolazione viene quasi sempre recuperata insieme alla verità, e il contesto dato al modello contiene quindi una contraddizione. È lo scenario in cui misurare FFR ha più senso.

## 4.7 Confronto dei retriever con query v2 (corpus ufficiale, curva completa)

Stesse 300 claim, stesse query v2, stessi documenti avvelenati e stessi seed. Il denso è sul commit della revisione (Mac-116), l'ibrido sul commit `4690d28` (Mac-116, 25 run completate il 13 novembre). Il BM25 è disponibile solo al 10% (Mac-Susy, commit `4690d28`).

PRR e PRD@1 (media su 5 seed; IC claims × seed tra parentesi):

| Tasso | Denso PRR | Ibrido PRR | Denso PRD@1 | Ibrido PRD@1 |
|---|---|---|---|---|
| 1% | 0.071 [0.034, 0.115] | 0.056 [0.029, 0.087] | 0.006 | 0.006 |
| 3% | 0.142 [0.071, 0.218] | 0.121 [0.061, 0.183] | 0.017 | 0.015 |
| 5% | 0.201 [0.149, 0.262] | 0.174 [0.125, 0.225] | 0.023 | 0.021 |
| 10% | 0.333 [0.300, 0.367] | 0.297 [0.261, 0.333] | 0.041 [0.026, 0.059] | 0.039 [0.025, 0.055] |

Esposizione mirata (variante propria nel top-5) e collaterale (solo varianti di altri claim), al 10%:

| Retriever | Mirata | Collaterale |
|---|---|---|
| Denso | 0.095 | 0.239 |
| BM25 | 0.087 | 0.225 |
| Ibrido | 0.094 | 0.203 |

Lettura.
- L'ibrido abbassa il PRR a ogni tasso, ma la differenza è nella parte collaterale (0.203 contro 0.239 al 10%). La parte mirata è identica (0.094 contro 0.095): il claim attaccato viene recuperato allo stesso modo.
- Il PRD@1 è praticamente uguale (differenze dentro gli intervalli): con query neutre, il primo posto non dipende dal retriever.
- Il valore del PRD@1 è basso a tutti i tassi, e questo è coerente con quanto visto sopra: con query che non contengono il valore cambiato, il documento avvelenato non prende quasi mai il primo posto.
- Confronto con Snopes: lì il PRD@1 del denso era 0.046 al 10% con query a template; qui 0.041 con query neutre, quindi il valore non dipende dal tipo di query nei limiti di questa misura. Il risultato su Snopes per BM25 (0.002) era causato dal bug del pareggio, già corretto.

## 4.8 Stato delle misure di sensibilità e difesa (4 novembre -> 13 novembre 2026)

Obiettivo: misurare quanto il sistema dipende da un contesto manipolato e se una difesa semplice lo riduce. Le manipolazioni restano quelle già nel corpus, revisionate dove possibile; non se ne generano di nuove.

| Misura | Cosa risponde | Stato | Cosa serve |
|---|---|---|---|
| 1. PCS (sensibilità della risposta al contesto avvelenato) | La risposta cambia quando si rimuovono i documenti avvelenati? | Implementata nel runner (generazione controfattuale). Non ancora eseguita sul corpus con query v2. | Rilanciare lo sweep al 10% con `pcs` tra le metriche. Il costo è circa il doppio per run. PCS è un proxy a similarità di embedding: si riporta come "PCS (embedding-similarity proxy)". |
| 2. Calibrazione di FFR e Source Integrity | Le metriche automatiche correlano con il giudizio umano? | Strumenti pronti (`sample_for_faithfulness_calibration.py`, `compute_calibration_correlation.py`). Nessuna annotazione raccolta. | Un annotatore con codebook, circa 40-60 record, più un secondo annotatore su una parte del campione per misurare l'accordo. Chi annota è ancora da decidere. |
| 3. Audit di contraddizione (difesa) | Una coppia di documenti in contraddizione nel top-5 segnala la manipolazione? | Da implementare. Non richiede generazioni: usa il modello NLI sui risultati già prodotti. | Script `contradiction_audit.py`. Soglia da calibrare sui record a 0% (falsi allarmi). L'audit non usa il testo del claim come riferimento, perché in un sistema reale il vero non è noto. |

Decisioni aperte prima di implementare la misura 3:
- definizione di contraddizione: probabilità NLI di contraddizione sopra soglia (punto di partenza 0.5, da calibrare);
- trattamento delle coppie: si considerano solo documenti con lo stesso claim_id, e si escludono le coppie non varianti, per evitare un segnale banale;
- chi decide le soglie e i criteri di validazione: la collega e il gruppo.

Stato dei risultati al 13 novembre 2026: corpus ufficiale con query v2, retriever densi (curva completa, Mac-116); BM25 solo al 10% (Mac-Susy, commit 4690d28; curva completa da rilanciare); ibrido completo su tutti i tassi (25 run, Mac-116, commit 4690d28, completate il 13 novembre, tabella in 4.7). Lo sweep ibrido su Snopes resta da rifare con il commit 4690d28, perché quello precedente è pre-correzione.

## 5. Limiti e interpretazione

FFR è zero per una ragione strutturale. FFR conta una risposta come falsità fedele solo se la faithfulness supera 0.8 e la correttezza scende sotto 0.2. Lo scorer proxy confronta similarità di embedding, e una risposta sullo stesso argomento del fatto vero ottiene comunque una correttezza alta anche se contiene il falso. Serve uno scorer che verifichi l'implicazione.

Lo scorer RAGAS (`faithfulness_scorer: ragas`) ora funziona: il bug su `AnswerSimilarity` è corretto e il test live passa. Non è però calibrato: usa `llama3.1:8b` come giudice, con problemi di affidabilità documentati. Va calibrato su annotazioni umane prima di produrre numeri da riportare, sempre come "FFR (RAGAS, llama3.1:8b judge)".

Source Integrity: fino al commit `b3450f9` era calcolata in modo errato (sezione 3.2 e `docs/CODE_REVIEW.md`). Dopo la correzione usa P(implicazione | documento, claim) del modello `cross-encoder/nli-MiniLM2-L6-H768`, che resta non calibrato e non fissato a una revisione.

FFR con lo scorer proxy: la correttezza è la similarità coseno riscalata in [0, 1] come (cos + 1) / 2. Due testi non correlati hanno coseno vicino a 0, quindi correttezza vicina a 0.5: la soglia 0.2 richiederebbe un coseno sotto −0.6, che con questi embedding non si verifica. FFR = 0 è quindi garantito dalla costruzione dello scorer, non osservato.

Warning noti, non bloccanti: `ascore` deprecato in RAGAS 0.2.15, `recreate_collection` deprecato in `qdrant_store.py`, `datetime.utcnow()` deprecato in `eiger/core/models.py`, `project.license` come tabella in `pyproject.toml`.

## 6. Storico (superati, da non citare)

- **29 ottobre.** 100 claim Snopes, un seed, sei attacchi inclusi M03 e M05. PRR da 0.28 (1%) a 0.77 (10%).
- **3 novembre.** Cinque seed, quattro attacchi. PRR da 0.20 (1%) a 0.82 (10%). Includeva come documenti avvelenati le copie identiche all'originale, e la Source Integrity era calcolata rispetto alla query.

Entrambe le versioni confermano il trend crescente, ma i valori sono sovrastimati.

## 7. Cosa serve ancora

1. **Definizione di Source Integrity** (da decidere con la collega). Proposta: SI = 1 − media di P(contraddizione | documento, claim). Con questa definizione un documento coerente o non correlato vale circa 1, uno che contraddice il claim circa 0, e la metrica scenderebbe solo se il contesto contiene affermazioni contrarie alla verità.
2. **Prompt delle query** (da decidere con la collega): metà delle query generate per il corpus ufficiale, e 17 su 100 per Snopes, ripetono il valore vero.
3. **FFR**: calibrare lo scorer RAGAS su annotazioni umane, e riportare FFR anche sui soli record esposti alla variante del proprio claim.
4. **Parità** sull'altra macchina con `10pct_seed42`.
5. **Revisione umana del corpus ufficiale**, poi sezione 4 con `--reviewed-only`.
6. Fissare la revisione dei modelli NLI e di embedding (`HF_HUB_OFFLINE=1`).

## 8. Riproduzione

```bash
scripts/install_locked.sh && make up
python scripts/make_benchmark_sweep.py --seeds 42 43 44 45 46
caffeinate -i scripts/run_benchmark_sweep.sh experiments/poison_rate_sweep_benchmark
python scripts/aggregate_benchmark_sweep.py

python scripts/make_corpus_sweep.py --seeds 42 43 44 45 46 --max-claims 300
caffeinate -i scripts/run_benchmark_sweep.sh experiments/corpus_sweep_preliminary
python scripts/aggregate_benchmark_sweep.py --results-dir results/corpus_sweep_preliminary \
  --title "Official corpus sweep, PRELIMINARY (corpus not yet human-validated)"
```
