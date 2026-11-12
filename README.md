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
| `download_data.py` | Scarica i dati di addestramento |
| `train_tokenizer.py`, `prepare_data.py`, `train.py` | I tre passi dell'addestramento |
| `notebooks/addestra_milo.ipynb` | Tutto l'addestramento, pronto per Colab e Kaggle |
| `chat.py`, `server.py` | Uso da terminale e API HTTP |

## Installazione

```bash
pip install -r requirements.txt
```

Per addestrare sul serio serve una GPU NVIDIA; su CPU funziona solo la taglia `tiny` di prova.

## Addestramento su Colab o Kaggle

Il modo più semplice: apri `notebooks/addestra_milo.ipynb` su
[Google Colab](https://colab.research.google.com) (*File → Apri notebook → GitHub*) oppure su
[Kaggle](https://www.kaggle.com/code) (*File → Import notebook*), attiva la GPU ed esegui le celle in ordine.
Il notebook scarica i dati, addestra Milo 1 e salva il modello (su Google Drive, se usi Colab).
Se la sessione si interrompe basta rieseguirlo: riprende da dove era arrivato.

I dati li scarica `download_data.py`: testo di Wikipedia in italiano per il pre-addestramento e,
per il fine-tuning, domande e risposte, risposte basate su una pagina (per la ricerca web)
e conversazioni sull'identità di Milo 1.

## Addestramento a mano

L'addestramento ha due fasi: prima il modello impara la lingua da tanto testo
(pre-addestramento), poi impara a fare l'assistente da conversazioni di esempio (fine-tuning).

```bash
# 1. Vocabolario
python train_tokenizer.py --input dati/*.txt dati/chat.jsonl --vocab-size 16000

# 2. Dati -> token
python prepare_data.py --text dati/*.txt --out out/pretrain
python prepare_data.py --chat dati/chat.jsonl --out out/chat

# 3a. Pre-addestramento da zero
python train.py --data out/pretrain --out out/pretrain-model --preset small --steps 100000 --compile

# 3b. Fine-tuning sulle conversazioni
python train.py --data out/chat --out out/chat-model \
    --init-from out/pretrain-model/best.pt --steps 5000 --lr 1e-4
```

Un addestramento interrotto (anche con Ctrl+C) si riprende aggiungendo `--resume`.
Se la GPU ha poca memoria, abbassa `--batch-size` e alza `--grad-accum`.

### Taglie del modello (`--preset`)

| Preset | Parametri | Contesto | Per cosa |
|---|---|---|---|
| `tiny` | ~3M | 256 token | Provare che tutto funzioni, anche su CPU |
| `small` | ~33M | 1024 | Una GPU consumer |
| `base` | ~97M | 2048 | Una GPU seria e molti dati (taglia GPT-2 small) |
| `large` | ~318M | 2048 | Più GPU o giorni di calcolo (taglia GPT-2 medium) |

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
