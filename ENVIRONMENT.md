# Ambiente riproducibile di EIBench

Questo documento descrive come ottenere lo stesso ambiente su più macchine e come
verificare che sia davvero identico prima di lanciare esperimenti. Vale per qualunque
macchina del gruppo: non c'è una macchina "di riferimento" fissa, c'è un ambiente
definito da file versionati e da una verifica.

## 1. Cosa viene fissato

| Componente | Come viene fissato | Dove si verifica |
|---|---|---|
| Python | 3.12.x (gli script rifiutano altre versioni) | `env_fingerprint.py` → `python.version` |
| Pacchetti Python | lockfile con hash SHA-256 per piattaforma: `requirements/lock-<os>-<arch>.txt` | `scripts/install_locked.sh` (`--require-hashes`) |
| Qdrant | immagine Docker `qdrant/qdrant:v1.9.4` in `docker-compose.yml` | `curl localhost:6333/healthz` |
| Ollama | versione fissata in `OLLAMA_VERSION` (vedi §4) | `ollama --version` |
| Modello LLM | `llama3.1:8b`, digest registrato nella fingerprint | `ollama list` |
| Modello di embedding | `sentence-transformers/all-MiniLM-L6-v2`, revisione HF registrata | `env_fingerprint.py` → `embedder.hf_snapshot` |
| Dati | SHA-256 di corpus e configurazioni YAML | `env_fingerprint.py` → `data_sha256` |
| Codice | commit git, senza modifiche non committate | `env_fingerprint.py` → `git` |

Le versioni dei pacchetti non sono più intervalli (`>=`): sono quelle del lockfile.

## 2. Lockfile per piattaforma

PyTorch e alcune librerie numeriche usano wheel diverse per macOS, Linux, arm64 e
x86_64. Un solo lockfile non può quindi valere per tutte le macchine. Il progetto
mantiene un lockfile per ogni coppia `<os>-<arch>`:

- `requirements/lock-darwin-arm64.txt`: Mac con Apple Silicon.
- `requirements/lock-darwin-x86_64.txt`: Mac Intel.
- `requirements/lock-linux-x86_64.txt`: Linux x86_64.

Due macchine con la stessa coppia usano lo stesso lockfile e devono ottenere
pacchetti identici. Due macchine con coppie diverse non possono essere identiche
al bit: il confronto tra loro va fatto con la verifica del §5, con tolleranza.

## 3. Procedura su una nuova macchina

```bash
git clone <repo> && cd eiger-framework
# 1. Python 3.12 e Docker/OrbStack installati; Ollama alla versione di §4
scripts/install_locked.sh          # crea venv-<host> dal lockfile della piattaforma
source venv-$(hostname -s)/bin/activate
make up                             # Qdrant v1.9.4
ollama pull llama3.1:8b
scripts/verify_env.sh               # confronta con reference/fingerprint_reference.json
make test-unit
```

Non lanciare esperimenti finché `verify_env.sh` non riporta zero differenze, oppure
finché le differenze sono state spiegate e accettate per iscritto (vedi §6).

## 4. Ollama

Ollama va installato alla versione fissata, non a quella più recente. Il valore
è in `OLLAMA_VERSION` nel file `reference/OLLAMA_VERSION` (da compilare: la prima
versione da registrare è quella usata nel primo run verificato). Su macOS
`brew install ollama` installa l'ultima versione, quindi dopo l'installazione va
controllato `ollama --version` e confrontato con il file. Se non coincide, scaricare
la versione esatta dalla pagina delle release ufficiali di Ollama su GitHub.

## 5. Verifica di parità tra due macchine

La verifica ha due livelli.

**Livello A, ambiente.** `verify_env.sh` deve riportare zero differenze per: Python,
pacchetti chiave, versione Ollama, digest del modello, revisione dell'embedder, hash
dei dati, commit git. Se la piattaforma è diversa, le differenze di pacchetti sono
attese e vanno elencate.

**Livello B, risultati.** Lanciare `baseline_v1` su entrambe le macchine con lo stesso
commit e confrontare le metriche aggregate:

| Metrica | Criterio di parità |
|---|---|
| PRR, PRD | identiche (dipendono solo da embedding, Qdrant e corpus) |
| Source Integrity | differenza assoluta ≤ 1e-4 |
| ERS | identica (annotazioni fissate dal disegno dell'attacco) |
| FFR | identica o spiegata (dipende dallo scorer e dalla generazione) |

Se il livello B fallisce, non si procede con gli sweep: prima si capisce la causa
(versione Ollama, digest del modello, thread di torch, piattaforma).

## 6. Quando le differenze sono accettabili

Una differenza è accettabile solo se è registrata con: quali macchine, quali valori,
perché è attesa (per esempio arm64 contro x86_64), e l'effetto misurato sul
livello B. Va scritta nel registro decisioni del progetto con data e responsabile.

## 7. Cosa non è ancora coperto

- Il digest del modello e la revisione dell'embedder vanno registrati la prima volta
  che si scaricano; da quel momento usare `HF_HUB_OFFLINE=1` per evitare aggiornamenti
  silenziosi.
- `results.json` registra ancora solo commit, Python e piattaforma. Va esteso per
  includere la fingerprint completa in ogni run.
- Il numero di test riportato nel README (776) e nella guida (485) va allineato
  dopo il primo `make test-unit` su una macchina pulita.
- Il repository contiene ancora claim manipolate pubbliche, come deciso per gli
  esperimenti iniziali. La bonifica è registrata nel registro decisioni (vedi
  `docs/ETHICS_AND_THREAT_MODEL.md`).
