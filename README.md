# Milo 1

Milo 1 è l'assistente generale di Ticly: risponde alle domande e cerca informazioni sul web.
È un modello linguistico **costruito da zero** (nessuna API di terzi): qui dentro ci sono
il tokenizer, la rete neurale, gli script per addestrarla e quelli per usarla.
Vextor resta l'AI per il codice.

Il modello non è ancora addestrato: questo repository è tutto ciò che serve per farlo.

## Com'è fatto

| File | Cosa fa |
|---|---|
| `milo/tokenizer.py` | Tokenizer BPE a livello di byte: trasforma il testo in numeri |
| `milo/model.py` | Il modello: transformer decoder stile GPT (RoPE, RMSNorm, cache in generazione) |
| `milo/chat_format.py` | Formato delle conversazioni (`<|user|>`, `<|assistant|>`, `<|context|>`) |
| `milo/search.py` | Ricerca web (DuckDuckGo) e lettura del testo delle pagine |
| `milo/assistant.py` | Mette insieme modello, tokenizer e ricerca |
| `build_pretrain.py` | Scarica il testo italiano e lo trasforma in token (miliardi di token, riprendibile) |
| `build_chat.py` | Scarica e prepara le conversazioni per il fine-tuning |
| `train_tokenizer.py`, `prepare_data.py`, `train.py` | I tre passi dell'addestramento |
| `notebooks/addestra_milo.ipynb` | Tutto l'addestramento, pronto per Colab e Kaggle |
| `chat.py`, `server.py` | Uso da terminale e API HTTP |

## Installazione

```bash
pip install -r requirements.txt
```

Per addestrare sul serio serve una GPU NVIDIA; su CPU funziona solo la taglia `tiny` di prova.

## Addestramento su Colab

Apri `notebooks/addestra_milo.ipynb` su [Google Colab](https://colab.research.google.com)
(caricando il file con *File → Carica notebook*) e segui le istruzioni in cima. In breve:

1. **Parte A, dati** (runtime senza GPU, 1-2 ore): *Esegui tutto*. Si ferma da solo quando i dati sono pronti.
2. **Parte B, addestramento** (runtime T4 GPU): *Esegui tutto* di nuovo. Il pre-addestramento dura
   `PRETRAIN_HOURS` ore di GPU (48 di default), il fine-tuning 1-3 ore.

Tutto viene salvato su Google Drive (`Milo1/`, servono circa 10 GB liberi). Le sessioni di Colab si
chiudono dopo qualche ora: riapri il notebook e premi di nuovo *Esegui tutto*, riparte dall'ultimo salvataggio.

### Cosa impara

| Fonte (Hugging Face) | Cosa contiene | Quota |
|---|---|---|
| `wikimedia/wikipedia` 20231101.it | Tutta Wikipedia in italiano (1,8 milioni di voci) | fino al 50% |
| `wikimedia/wikisource` 20231201.it | Testi classici e documenti (es. la Costituzione) | fino al 5% |
| `manu/project_gutenberg` split `it` | Circa 1.000 libri di pubblico dominio | fino al 5% |
| `epfml/FineWeb2-HQ` ita_Latn | Pagine web italiane, il 10% di qualità più alta | il resto |

Con le impostazioni di default sono 2,5 miliardi di token. Per il fine-tuning: `FreedomIntelligence/alpaca-gpt4-italian`,
`teelinsan/camoscio`, le conversazioni in italiano di `OpenAssistant/oasst2`, `crux82/squad_it`
(trasformato in esempi di "rispondi usando queste pagine web", con pagine esca) e le conversazioni
sull'identità di Milo 1 scritte in `build_chat.py`.

## Addestramento a mano

L'addestramento ha due fasi: prima il modello impara la lingua da tanto testo
(pre-addestramento), poi impara a fare l'assistente da conversazioni di esempio (fine-tuning).

```bash
# 1. Dati
python build_chat.py --out dati/chat.jsonl
python build_pretrain.py sample --out dati/campione.txt --mb 150
python train_tokenizer.py --input dati/campione.txt dati/chat.jsonl --vocab-size 32000 --max-chars 220000000
python build_pretrain.py tokens --tokenizer out/tokenizer.json --out out/pretrain --target-tokens 2.5e9
python prepare_data.py --chat dati/chat.jsonl --out out/chat

# 2. Pre-addestramento da zero, per un numero di ore (i passi si calcolano dalla velocità della GPU)
python train.py --data out/pretrain --out out/pretrain-model --preset base --hours 48 \
    --batch-size 8 --grad-accum 8 --lr 6e-4 --warmup 1000

# 3. Fine-tuning sulle conversazioni (la loss conta solo le risposte dell'assistente)
python train.py --data out/chat --out out/chat-model \
    --init-from out/pretrain-model/best.pt --epochs 2 --lr 1e-4 --batch-size 8 --grad-accum 8
```

Al posto di `--hours` si possono usare `--steps` (numero di passi) o `--epochs` (passaggi sui dati).
Un addestramento interrotto (anche con Ctrl+C) si riprende aggiungendo `--resume`.
Se la GPU ha poca memoria, abbassa `--batch-size` e alza `--grad-accum`.

### Taglie del modello (`--preset`)

| Preset | Parametri | Contesto | Per cosa |
|---|---|---|---|
| `tiny` | ~3M | 256 token | Provare che tutto funzioni, anche su CPU |
| `small` | ~42M | 1024 | Poche ore di GPU |
| `base` | ~110M | 1024 | Decine di ore su una T4 (taglia GPT-2 small): la scelta per Colab |
| `large` | ~335M | 1024 | Più GPU o settimane di calcolo (taglia GPT-2 medium) |

(Parametri con vocabolario da 32.000 token.)

### Dati

- **Testo libero** (`.txt`): più ce n'è, meglio è. Servono almeno centinaia di MB
  di testo di buona qualità perché il modello scriva frasi sensate.
- **Conversazioni** (`.jsonl`, una per riga):

  ```json
  {"messages": [{"role": "user", "content": "Ciao"}, {"role": "assistant", "content": "Ciao! Come posso aiutarti?"}]}
  ```

- **Ricerca web**: per insegnare al modello a rispondere usando le pagine trovate,
  una parte delle conversazioni deve avere, prima della domanda, un messaggio con
  `"role": "context"` che contiene i risultati nello stesso formato usato in
  produzione (vedi `data/esempio/chat.jsonl`). Senza questi esempi il modello
  riceve le pagine ma non sa cosa farne.

### Prova veloce

Con i dati minuscoli in `data/esempio` (pochi minuti su CPU, il modello li impara a memoria):

```bash
python train_tokenizer.py --input data/esempio/*.txt data/esempio/chat.jsonl --vocab-size 600
python prepare_data.py --text data/esempio/testo.txt --chat data/esempio/chat.jsonl --out out/demo --val-fraction 0.15
python train.py --data out/demo --out out/demo-model --preset tiny --steps 300 --batch-size 16 --lr 1e-3 --warmup 20 --eval-every 100
python chat.py --model out/demo-model/last.pt --no-web
```

## Uso

Dal terminale:

```bash
python chat.py --model out/chat-model/best.pt
```

Come API:

```bash
TICLY_API_KEY=una-chiave-segreta python server.py --model out/chat-model/best.pt --port 3000

curl -X POST http://localhost:3000/v1/chat \
  -H "Authorization: Bearer una-chiave-segreta" \
  -d '{"message": "Quanto è alto il Monte Bianco?"}'
```

Risposta: `{"answer": "...", "sources": [{"title": "...", "url": "..."}]}`

Campi della richiesta:

| Campo | Default | Descrizione |
|---|---|---|
| `message` / `messages` | — | Domanda singola, oppure tutta la conversazione (l'API non la salva) |
| `web` | `true` | Cerca sul web prima di rispondere |
| `stream` | `false` | Risposta a pezzi (eventi `sources`, `text`, `done`, `error`) |
| `temperature` | `0.7` | 0 = sempre la risposta più probabile, più alto = più varia |
| `max_tokens` | `512` | Lunghezza massima della risposta |

Il nome dell'assistente si cambia con la variabile d'ambiente `ASSISTANT_NAME`.

## Limiti da conoscere

- La qualità dipende quasi solo da **quanti dati e quanto calcolo** si usano. Un modello
  `small` o `base` addestrato così scrive in modo scorrevole e risponde su ciò che ha
  visto, ma resta molto lontano da Gemini o ChatGPT, che hanno migliaia di volte più
  parametri e dati.
- La ricerca usa come query l'ultima domanda dell'utente e la pagina HTML di DuckDuckGo,
  che può cambiare o limitare le richieste: per un uso serio conviene un'API di ricerca.
- Il server genera una risposta alla volta.
