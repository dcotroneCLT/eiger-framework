# Esperimenti EIBench: documentazione e risultati

Stato al 3 novembre 2026. Sostituisce la versione del 29 ottobre. Descrive gli esperimenti eseguiti, i risultati e i limiti con cui vanno letti. Per l'ambiente e la riproduzione vedi `ENVIRONMENT.md` e `docs/REPRODUCING.md`.

## 1. Sintesi

Rispetto al 29 ottobre, il risultato sul retrieval è ora sostenibile. Lo sweep è stato rieseguito con i soli quattro attacchi ammessi dal benchmark (M01, M02, M04, M06), su cinque seed, con intervalli di confidenza, in un ambiente fissato da lockfile.

Al crescere del tasso di avvelenamento (0, 1, 3, 5, 10%), la quota di query con almeno un documento avvelenato nel top-5 (PRR) cresce in modo monotono, da 0.00 a 0.82 (IC 95% 0.79–0.86). Gli intervalli di tassi adiacenti non si sovrappongono. Il risultato è coerente con H1 (RQ1).

Restano invariati due limiti: FFR non è misurabile con lo scorer proxy (vale 0.0 per costruzione, sezione 5), e il dataset della sezione 3 è Snopes. Il primo esperimento sul corpus ufficiale in git (sezione 4) è preliminare, perché il corpus non è ancora validato. Mostra che una manipolazione, quando presente, viene quasi sempre recuperata accanto al documento vero, ma il suo PRD@1 è falsato dalla costruzione della query.

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
| Commit | `4a709b2` (configurazioni generate da `scripts/make_benchmark_sweep.py`) |

Verifiche prima dello sweep: 776 test unitari e 5 test di integrazione superati (incluso il test live con Qdrant e Ollama). Due run identiche di `baseline_v1` e di `0pct_seed42` producono `results.json` identici, escluso il timestamp.

## 3. Risultati: sweep del tasso di avvelenamento (Snopes)

### 3.1 Metriche di retrieval

Media ± deviazione standard sui cinque seed. Intervallo di confidenza al 95% con bootstrap (10.000 ricampionamenti) sui 500 valori per record di ogni tasso.

| Tasso | PRR (≥1 avvelenato in top-5) | IC 95% | PRD@1 (avvelenato al rank 1) | IC 95% |
|---|---|---|---|---|
| 0% | 0.000 ± 0.000 | [0.000, 0.000] | 0.000 ± 0.000 | [0.000, 0.000] |
| 1% | 0.204 ± 0.056 | [0.168, 0.240] | 0.022 ± 0.008 | [0.010, 0.036] |
| 3% | 0.434 ± 0.068 | [0.390, 0.476] | 0.052 ± 0.011 | [0.032, 0.072] |
| 5% | 0.584 ± 0.099 | [0.540, 0.628] | 0.064 ± 0.011 | [0.044, 0.086] |
| 10% | 0.824 ± 0.083 | [0.790, 0.856] | 0.156 ± 0.030 | [0.124, 0.190] |

Lettura. L'esposizione del retrieval cresce in modo monotono e più che proporzionale al tasso nominale: al 10% più di quattro query su cinque hanno almeno un documento avvelenato nel top-5, e circa una su sei lo ha al primo posto. PRR e PRD dipendono solo da embedding, vector store e corpus, quindi non risentono dei limiti degli scorer.

La variabilità tra seed è ampia: il solo seed 42 al 10% dà PRR 0.68, contro una media di 0.82. I numeri a un solo seed della versione del 29 ottobre vanno quindi letti come indicativi.

### 3.2 Metriche diagnostiche (non sono risultati)

| Tasso | Source Integrity | ERS | FFR (proxy) |
|---|---|---|---|
| 0% | 0.560 ± 0.000 | 0.000 ± 0.000 | 0.000 |
| 1% | 0.565 ± 0.003 | 0.811 ± 0.037 | 0.000 |
| 3% | 0.569 ± 0.004 | 0.821 ± 0.015 | 0.000 |
| 5% | 0.572 ± 0.004 | 0.817 ± 0.014 | 0.000 |
| 10% | 0.581 ± 0.002 | 0.817 ± 0.015 | 0.000 |

Source Integrity cresce leggermente con il tasso invece di diminuire. Non è interpretabile finché il modello NLI non è calibrato. ERS è la media di annotazioni fissate dal disegno dell'attacco, quindi è quasi costante per costruzione. FFR è 0 per la ragione strutturale descritta nella sezione 5.

### 3.3 Avvertenze di lettura

- **Tasso nominale e quota effettiva.** Ogni attacco si applica in modo indipendente. Con quattro attacchi al 10%, un claim riceve in media circa 0.4 documenti avvelenati in più rispetto al suo ground-truth. La quota effettiva per run è nel campo `n_poisoned` dei log (`logs/`) e va riportata accanto al tasso nominale.
- **Intervallo di confidenza.** Il bootstrap tratta i 500 record di un tasso come indipendenti, ma i cinque seed usano gli stessi 100 claim e cambiano solo i documenti avvelenati. L'intervallo è quindi probabilmente un po' stretto; la deviazione standard tra seed è la misura più prudente.
- **Parità tra macchine.** Tutte le run sono su Mac-116. Su un'altra macchina con la stessa piattaforma e lo stesso lockfile, `10pct_seed42` deve dare PRR e PRD identici. Questa verifica non è ancora stata fatta.

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

25 run, commit `bb00c08`, 300 claim per run, 1.500 record per tasso. Media ± deviazione standard sui cinque seed; IC 95% con bootstrap sui record.

| Tasso | PRR | IC 95% | PRD@1 | IC 95% |
|---|---|---|---|---|
| 0% | 0.000 ± 0.000 | [0.000, 0.000] | 0.000 ± 0.000 | [0.000, 0.000] |
| 1% | 0.050 ± 0.040 | [0.040, 0.061] | 0.001 ± 0.001 | [0.000, 0.002] |
| 3% | 0.100 ± 0.068 | [0.085, 0.115] | 0.001 ± 0.002 | [0.000, 0.003] |
| 5% | 0.148 ± 0.037 | [0.131, 0.167] | 0.002 ± 0.003 | [0.000, 0.005] |
| 10% | 0.261 ± 0.030 | [0.239, 0.283] | 0.006 ± 0.004 | [0.003, 0.010] |

Diagnostiche: Source Integrity 0.503–0.507, ERS 0.000 (annotazioni assenti nel corpus, non inventate), FFR 0.000 (proxy).

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

## 5. Limiti e interpretazione

FFR è zero per una ragione strutturale. FFR conta una risposta come falsità fedele solo se la faithfulness supera 0.8 e la correttezza scende sotto 0.2. Lo scorer proxy confronta similarità di embedding, e una risposta sullo stesso argomento del fatto vero ottiene comunque una correttezza alta anche se contiene il falso. Serve uno scorer che verifichi l'implicazione.

Lo scorer RAGAS (`faithfulness_scorer: ragas`) ora funziona: il bug su `AnswerSimilarity` è corretto e il test live passa. Non è però calibrato: usa `llama3.1:8b` come giudice, con problemi di affidabilità documentati. Va calibrato su annotazioni umane prima di produrre numeri da riportare, sempre come "FFR (RAGAS, llama3.1:8b judge)".

Source Integrity dipende da un modello NLI (`cross-encoder/nli-MiniLM2-L6-H768`) non calibrato e non ancora fissato a una revisione.

Warning noti, non bloccanti: `ascore` deprecato in RAGAS 0.2.15, `recreate_collection` deprecato in `qdrant_store.py`, `datetime.utcnow()` deprecato in `eiger/core/models.py`, `project.license` come tabella in `pyproject.toml`.

## 6. Storico: risultati del 29 ottobre (superati)

Sweep su 100 claim Snopes, un solo seed (42), sei attacchi inclusi M03 e M05, ambiente non fissato. PRR passava da 0.28 (1%) a 0.77 (10%). Questi numeri non vanno più citati: includono attacchi esclusi dal benchmark e non hanno stima della variabilità. Restano utili come conferma qualitativa del trend.

## 7. Cosa serve ancora

1. Verificare la parità sull'altra macchina con `10pct_seed42`.
2. Dare al corpus ufficiale query neutre, che non contengano il testo originale né l'elemento modificato, generate una sola volta e versionate con hash; poi rieseguire la sezione 4. Aggiungere a PRR la scomposizione mirata/collaterale della sezione 4.5.
3. Calibrare lo scorer RAGAS su annotazioni umane e misurare FFR.
4. Completare la revisione umana del corpus, poi rieseguire la sezione 4 con `--reviewed-only`.
5. Fissare la revisione del modello NLI e del modello di embedding (`HF_HUB_OFFLINE=1`).

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
