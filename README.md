# Local Translation Service

A self-hosted REST API for offline text translation, built with FastAPI and
Hugging Face Transformers. Designed to run on a Synology NAS (or any
x86_64 Linux host) via Docker Compose — CPU only, no GPU required.

Supports **eight translation directions** between Slovak, French, English, and Arabic —
all in a single container, with models loaded on demand.

---

## What it does

- Exposes a simple HTTP API for translating text, plus a browser UI at `/ui`.
- Routes by `source_lang` + `target_lang` in each request — one service
  handles all directions (SK↔FR, SK↔EN, EN↔FR, AR↔FR).
- Models are loaded **lazily**: only the pairs you actually use are kept in
  memory (~400 MB each). Unused directions cost nothing.
- Splits long texts into paragraph- and sentence-aware chunks so requests
  larger than the model's token limit still work correctly.
- Caches translated chunks (LRU, 256 entries per model) — repeated strings
  such as UI labels are never sent to the model twice.
- Keeps model weights on a persistent host volume so models are not
  re-downloaded when the container is recreated.

---

## Project structure

```
translation-service/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, endpoints
│   ├── translator.py    # Model loading, chunking, inference
│   ├── settings.py      # Environment-variable configuration
│   └── static/
│       └── index.html   # Browser translation UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── upload-to-synology.sh  # One-command rsync deploy script
├── .env.example
└── README.md
```

---

## Quick start

### Build and run

```bash
docker compose up -d --build
```

The container starts immediately — models are downloaded on the **first request**
for each language pair (~300 MB each) and cached in `./models/huggingface/`.
Every subsequent start reuses the cached copy and works fully offline.
Internet access is only needed during this first download per pair.

```bash
docker compose ps          # watch STATUS column
docker compose logs -f     # follow logs

# Stop and remove the container (model cache is preserved)
docker compose down
```

### Check health

```bash
curl http://NAS_IP:8088/health
# {"status":"ok"}
```

### Translate text

`source_lang` and `target_lang` determine which model is used. On the first
request for a pair the model is downloaded (~300 MB) and loaded; all
subsequent requests for that pair are served from memory.

```bash
curl -X POST http://NAS_IP:8088/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Dobrý deň, toto je testovací preklad.","source_lang":"sk","target_lang":"fr"}'
```

Expected response:

```json
{
  "translated_text": "Bonjour, ceci est une traduction de test.",
  "model": "Helsinki-NLP/opus-mt-sk-fr",
  "source_lang": "sk",
  "target_lang": "fr",
  "char_count": 37
}
```

Supported pairs: `sk→fr`, `fr→sk`, `sk→en`, `en→sk`, `en→fr`, `fr→en`, `ar→fr`, `fr→ar`.
Sending an unsupported pair returns HTTP 422.

### Browser UI

```
http://NAS_IP:8088/ui
```

DeepL-style interface — select language pair, type text, translation appears
automatically. Includes swap button, character counter, and copy button.
Link to API docs is in the top-right corner.

### Other useful endpoints

```bash
# Service index (JSON)
curl http://NAS_IP:8088/

# Loaded models and supported pairs
curl http://NAS_IP:8088/model-info

# Interactive API docs (Swagger UI)
open http://NAS_IP:8088/docs
```

---

## Configuration

All settings are environment variables. Edit `docker-compose.yml` or copy
`.env.example` to `.env` for local development.

| Variable | Default | Description |
|---|---|---|
| `MAX_CHARS_PER_REQUEST` | `10000` | Hard limit per single request |
| `CHUNK_SIZE` | `450` | Max characters per internal translation chunk fed to the model |
| `LOG_TEXT` | `false` | Set `true` to log translated text (debug only) |
| `ALLOWED_ORIGINS` | `*` | CORS origins, comma-separated or `*` |

### Supported language pairs

All six directions are served by the single container on port **8088**.
Models are loaded on the first request for each pair and stay in memory.

| Direction | Model | HF downloads |
|---|---|---|
| SK → FR | `Helsinki-NLP/opus-mt-sk-fr` | 133 |
| FR → SK | `Helsinki-NLP/opus-mt-fr-sk` | 84 |
| SK → EN | `Helsinki-NLP/opus-mt-sk-en` | 4.7 K |
| EN → SK | `Helsinki-NLP/opus-mt-en-sk` | 747 |
| EN → FR | `Helsinki-NLP/opus-mt-en-fr` | 407 K |
| FR → EN | `Helsinki-NLP/opus-mt-fr-en` | 664 K |
| AR → FR | `Helsinki-NLP/opus-mt-ar-fr` | 1.6 K |
| FR → AR | `Helsinki-NLP/opus-mt-fr-ar` | 1.9 K |

RAM usage is ~400 MB per loaded model. If you only use two or three
directions, the others are never loaded and cost nothing.

---

## Migrating from the old 6-container setup

If you previously ran six separate `translator-sk-fr`, `translator-fr-sk`, …
containers, stop them and remove the old project before deploying the new one:

```bash
# On the NAS
cd /volume1/docker/translation-service
docker compose down

# Upload the updated project files
# (from your computer)
./upload-to-synology.sh --deploy
```

**Model cache:** The old setup stored each model in a separate subfolder
(`./models/sk-fr/huggingface/`, `./models/fr-sk/huggingface/`, …). The new
single-service setup uses a flat `./models/huggingface/` cache. Models are
re-downloaded on first use (~300 MB each) — this is a one-time cost. After
that everything runs offline as before.

The old per-pair port mapping (8088–8093) is replaced by a single port
**8088** with `source_lang`/`target_lang` parameters doing the routing.

---

## Deployment on Synology NAS

### Step 1 — Copy the project to the NAS

The easiest way is the included deploy script, which uses `rsync` over SSH
(port 22222 — the default non-standard SSH port on Synology DSM):

```bash
# Upload only (inspect before starting)
./upload-to-synology.sh

# Upload and immediately build + start all services
./upload-to-synology.sh --deploy
```

Default connection parameters (edit the top of the script if yours differ):

| Parameter | Default |
|---|---|
| `NAS_USER` | `vovo` |
| `NAS_HOST` | `192.168.0.202` |
| `REMOTE_DIR` | `/volume1/docker/translation-service` |
| `SSH_PORT` | `22222` |

You can also pass them as positional arguments:

```bash
./upload-to-synology.sh OTHER_USER 192.168.1.10 /volume1/docker/translation-service --deploy
```

The script skips the `models/` directory entirely — downloaded model weights
on the NAS are never overwritten or re-transferred.

**Manual alternative** — if you prefer plain `scp` (uses the standard SSH
port 22; adjust `-P` if your NAS uses a different port):

```bash
scp -r translation-service NAS_USER@NAS_IP:/volume1/docker/translation-service
```

Or drag-and-drop the `translation-service/` folder into `/volume1/docker/`
via **DSM → File Station**.

> Make sure `/volume1/docker/` exists on the NAS. If not, create it in
> File Station first. Ensure the volume has at least **2 GB free** for the
> Docker image and model cache.

---

### Step 2 — Create a Container Manager Project

1. Open **DSM → Container Manager → Project → Create**
2. **Project name:** `translation-service`
3. **Path:** `/volume1/docker/translation-service`
   (the folder that contains `docker-compose.yml`)
4. Container Manager shows the contents of `docker-compose.yml` — confirm
   **Next → Done**
5. The image build starts automatically — **first build takes 5–15 minutes**
   (downloads Python, PyTorch CPU, and all dependencies)

---

### Step 3 — Verify the service is up

The container starts in seconds — models are loaded lazily on the first
request, not at startup. The health check turns green almost immediately.

Follow the logs over SSH to see model downloads in real time:

```bash
ssh -p 22222 NAS_USER@NAS_IP
docker compose -f /volume1/docker/translation-service/docker-compose.yml logs -f
```

The first request for each language pair triggers a download (~300 MB) and
loading. You will see this in the logs:
```
Loading model: Helsinki-NLP/opus-mt-sk-fr
Model loaded in XX.XXs | device: cpu
```

Subsequent requests for that pair are served immediately from memory.

---

### Step 4 — Verify

```bash
curl http://NAS_IP:8088/health
# {"status":"ok"}

curl -X POST http://NAS_IP:8088/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Dobrý deň, toto je testovací preklad.","source_lang":"sk","target_lang":"fr"}'
```

Or open `http://NAS_IP:8088/docs` in a browser for the interactive Swagger UI.

---

### Managing the service

```bash
# View logs
docker compose logs -f

# Restart
docker compose restart

# Stop (keeps container)
docker compose stop

# Stop and remove container — model files in ./models/ are preserved
docker compose down

# Rebuild after a code change
docker compose up -d --build
```

All commands should be run from `/volume1/docker/translation-service/` on
the NAS, or passed with `-f /volume1/docker/translation-service/docker-compose.yml`.

---

### Updating a running deployment

#### Case A — code change (`app/*.py`)

Use `upload-to-synology.sh --deploy` from your computer — it uploads only
changed files and triggers a rebuild automatically.

Or manually:

```bash
# From your computer
scp -r app/ NAS_USER@NAS_IP:/volume1/docker/translation-service/

# On the NAS (via SSH)
cd /volume1/docker/translation-service
docker compose up -d --build
```

The rebuild only affects the image layers that changed. Model files in
`./models/` are on the host volume and are never touched by a rebuild.

#### Case B — `docker-compose.yml` change (env variable, new service, port)

No rebuild needed — Compose detects what changed and restarts only the
affected services:

```bash
cd /volume1/docker/translation-service
docker compose up -d
```

#### Case C — restart

```bash
docker compose restart
```

#### Via Container Manager UI (no SSH)

1. Copy changed files to the NAS via **File Station**
2. **Container Manager → Project → translation-service → Stop**
3. Click **Build** if you changed application code, or **Start** directly
   if you only changed `docker-compose.yml`

> Model weights in `./models/` survive all of the above operations —
> they are stored on the host, not inside the image.

---

## Testing

The easiest way to test is to open the **browser UI**:

```
http://NAS_IP:8088/ui
```

Or the **Swagger UI** for raw API testing:

```
http://NAS_IP:8088/docs
```

---

For terminal testing, work through these checks in order:

### 1. Is the service alive?

```bash
curl http://NAS_IP:8088/health
```

Expected: `{"status":"ok"}`

### 2. Check loaded models

```bash
curl http://NAS_IP:8088/model-info
```

Expected (after at least one translation request):
```json
{
  "supported_pairs": ["en→fr", "en→sk", "fr→en", "fr→sk", "sk→en", "sk→fr"],
  "loaded_models": [
    {"pair": "sk-fr", "model_id": "Helsinki-NLP/opus-mt-sk-fr", "device": "cpu"}
  ]
}
```

### 3. Basic translation

```bash
curl -s -X POST http://NAS_IP:8088/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Dobrý deň, toto je testovací preklad.","source_lang":"sk","target_lang":"fr"}' \
  | python3 -m json.tool
```

Expected:
```json
{
  "translated_text": "Bonjour, ceci est une traduction de test.",
  "model": "Helsinki-NLP/opus-mt-sk-fr",
  "source_lang": "sk",
  "target_lang": "fr",
  "char_count": 37
}
```

### 4. Multi-paragraph text

```bash
curl -s -X POST http://NAS_IP:8088/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Prvý odsek textu na preklad.\n\nDruhý odsek s ďalšou vetou.",
    "source_lang": "sk",
    "target_lang": "fr"
  }' | python3 -m json.tool
```

The response `translated_text` should contain two paragraphs separated by
`\n\n`.

### 5. Validation — empty text (should be rejected)

```bash
curl -s -X POST http://NAS_IP:8088/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"","source_lang":"sk","target_lang":"fr"}'
```

Expected: HTTP 422 with a validation error message.

### 6. Validation — text too long (should be rejected)

```bash
python3 -c "
import json, urllib.request
payload = json.dumps({
    'text': 'a' * 10001,
    'source_lang': 'sk',
    'target_lang': 'fr'
}).encode()
req = urllib.request.Request(
    'http://NAS_IP:8088/translate',
    data=payload,
    headers={'Content-Type': 'application/json'}
)
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode())
"
```

Expected: HTTP 422 with `Text length 10001 exceeds MAX_CHARS_PER_REQUEST (10000)`.

---

## CPU performance and NAS hardware requirements

### Will PyTorch run on a Synology NAS?

Yes — for occasional translation workloads this is the right architecture.
The `opus-mt-sk-fr` model is small by ML standards:

- **Disk:** ~300 MB for the downloaded weights
- **RAM at runtime:** ~350–450 MB (loaded once at startup, held for all requests)

Any x86_64 Synology NAS (Celeron J4125, J4025, N5105, J6412, …) can hold
that comfortably alongside DSM and other services.

### How fast is CPU inference?

| Input size | Approximate time on a typical NAS CPU |
|---|---|
| 1 sentence (~80 chars) | 1–4 s |
| 1 paragraph (~450 chars, 1 chunk) | 3–10 s |
| 10 000 chars (max request, ~22 chunks) | 60–200 s |

At the target load of ~1 million characters/month (~33 000 chars/day, a
handful of requests per hour) the CPU is never a bottleneck.

### AVX2 — the one hardware caveat

PyTorch CPU wheels are compiled with AVX2 optimizations. Very old Synology
Atom processors — J1900, J3355, J3455 — **do not support AVX2** and will
crash with `Illegal instruction` when PyTorch tries to use it.

Check your NAS CPU in the Intel ARK database under
"Advanced Vector Extensions 2". All Celeron J4xxx, J6xxx, and N5xxx
processors (2019 onward) are fine.

### Tuning inference speed

The default `num_beams=2` in `translator.py` gives a good quality/speed
trade-off on CPU. If responses are too slow for your NAS, set it to `1`
(greedy decoding) — quality on short Slovak→French paragraphs is nearly
identical and it is roughly 2× faster:

```python
# app/translator.py
output_ids = self.model.generate(**inputs, max_length=512, num_beams=1)
```

---

## Architecture notes

### Lazy model loading

Models are not loaded at startup. The first request for a given language pair
triggers the download (once, ~300 MB) and loads the model into memory. All
subsequent requests for that pair are served instantly from the cached instance.

Only the pairs you actually use consume RAM. On a NAS with 4 GB RAM where
you primarily use `sk↔fr`, only ~400 MB is occupied — not 2.4 GB.

### Chunk-level LRU cache

Translated chunks (up to 256 per model, FIFO eviction) are cached in memory.
If the same sentence or paragraph appears in multiple requests — common with
UI strings, boilerplate text, or repeated document sections — the model is not
called again.

### Concurrency model

`model.generate()` is CPU-bound and not thread-safe to run in parallel on the
same model. A single `asyncio.Semaphore(1)` queues requests so only one
inference runs at a time, preventing CPU thrashing. The FastAPI event loop
remains unblocked while inference runs in a thread pool worker
(`asyncio.to_thread`).

---

## Networking — local IP vs. DDNS hostname

If you connect from the **same network as the NAS** (LAN, local Wi-Fi, or
VPN into the LAN) always use the **local IP**:

```bash
curl http://192.168.0.202:8088/health   # works
curl http://yourname.synology.me:8088/health  # fails on LAN
```

The DDNS hostname (e.g. `yourname.synology.me`) resolves to your router's
**public/external IP**. Most home routers do not support NAT hairpinning —
they cannot forward a packet that arrives on the WAN interface back to a
device inside the same LAN. The connection is simply dropped.

| Where you connect from | Use |
|---|---|
| Same LAN / local Wi-Fi | `http://192.168.0.202:8088` |
| Internet (remote access) | `https://yourname.synology.me` via reverse proxy (see below) |

### Remote access

Do **not** open port 8088 directly on the router. Instead use the
**Synology Reverse Proxy** built into DSM:

1. **DSM → Control Panel → Login Portal → Advanced → Reverse Proxy → Create**
2. Source: `HTTPS`, hostname `yourname.synology.me`, port `443`
3. Destination: `HTTP`, hostname `localhost`, port `8088`
4. Add an access-control rule or HTTP Basic Auth to protect the endpoint

This gives you HTTPS and keeps port 8088 off the public internet.
Nginx Proxy Manager (available as a Container Manager project) is an
alternative with a friendlier UI.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Failed to connect` on DDNS hostname | NAT hairpinning not supported by router | Use the local IP `192.168.x.x` instead |
| Container keeps restarting | Model download failed or OOM | Check logs: `docker compose logs` |
| `/health` returns 503 | Model still loading | Wait — first start can take 2–5 min |
| Translation is very slow | Large input, many chunks, or first request (model loading) | Wait for model load; reduce request size; CPU inference is slower than cloud APIs |
| First request for a pair takes minutes | Model is downloading (~300 MB) | Normal on first use — subsequent requests are fast |
| `422` with "Unsupported language pair" | Invalid `source_lang`/`target_lang` combination | Use one of: sk↔fr, sk↔en, en↔fr |
| `422 Unprocessable Entity` | Text exceeds `MAX_CHARS_PER_REQUEST` | Split your request or raise the limit |
| Model file missing after restart | Volume not mounted | Confirm `./models` exists and is writable |
| `ssh: connect to host ... port 22` fails | NAS uses non-standard SSH port | Add `-p 22222` to SSH/SCP commands, or use `upload-to-synology.sh` |

---

## Security and privacy

- **All translated text stays local.** No data leaves your network.
- Model weights are downloaded from Hugging Face on first run only; subsequent
  starts are fully offline.
- **Do not expose port 8088 directly to the public internet.**
  If you need remote access, put the service behind a reverse proxy (e.g.
  Nginx Proxy Manager on the same NAS) with HTTPS and authentication.
- Set `ALLOWED_ORIGINS` to a specific domain if you embed this API in a
  web application to restrict cross-origin access.
