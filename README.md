# 🌾 Kisan AI — Offline Hindi Agricultural Assistant

<p align="center">
  <img src="[static/images/logo.png](https://pngtree.com/freepng/original-hand-painted-crop-wheat_5765362.html)" alt="Kisan AI Logo" width="120"/>
</p>

<p align="center">
  <b>A 100% offline-capable AI chatbot for Indian farmers — Hindi voice input, real-time STT, and intelligent crop advisory.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Django-5.0-green?logo=django" />
  <img src="https://img.shields.io/badge/Ollama-llama3.2-orange" />
  <img src="https://img.shields.io/badge/Vosk-Hindi%20STT-purple" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [Prerequisites](#-prerequisites)
- [Setup Guide](#-setup-guide)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Python Environment](#2-python-environment)
  - [3. Database Setup](#3-database-setup)
  - [4. Environment Variables](#4-environment-variables)
  - [5. Ollama & Custom AI Model](#5-ollama--custom-ai-model)
  - [6. Vosk Hindi STT Model](#6-vosk-hindi-stt-model)
  - [7. Windows TTS Voices (Optional)](#7-windows-tts-voices-optional)
  - [8. Run the Server](#8-run-the-server)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [AI Response Pipeline](#-ai-response-pipeline)
- [Voice Features](#-voice-features)
- [Testing & Evaluation](#-testing--evaluation)
- [Configuration Reference](#-configuration-reference)
- [Deployment Notes](#-deployment-notes)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎤 **Real-time Hindi STT** | Speak in Hindi — words appear in the input box as you talk (Vosk, fully offline) |
| 🔊 **Hindi TTS** | Responses read aloud in Hindi (Microsoft Kalpana/Hemant via pyttsx3) |
| 🤖 **Kisan-AI LLM** | Custom Ollama model trained with agricultural context via Modelfile |
| 📊 **Dataset-Direct Tier** | High-confidence queries answered instantly from the crop database (no LLM needed) |
| 🌐 **100% Offline** | Vosk STT + Ollama LLM + local DB = works without internet |
| ☁️ **Groq Fallback** | If Ollama is unavailable, falls back to Groq cloud API automatically |
| 🗣️ **Voice Activity Detection** | Auto-stops recording after 1.5s of silence — no button press needed |
| 🌱 **3-Tier AI Pipeline** | Dataset-Direct → LLM+Context → LLM-Only, based on query confidence |
| 📱 **Responsive UI** | Works on mobile, tablet, and desktop |
| 🏛️ **Government Theme** | Official Indian government design aesthetic |

---

## 🏗 Architecture Overview

```
User Query (Voice/Text)
        │
        ▼
┌─────────────────┐
│   Django Views  │  POST /chat/stream  →  SSE streaming response
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Engine (ai_engine.py)                 │
│                                                             │
│  1. Detect Crop (आलू, गेहूँ, टमाटर...)                      │
│  2. Detect Intent (pest, disease, fertilizer, irrigation)   │
│  3. Retrieve from Dataset (TF-IDF cosine similarity)        │
│  4. Route to Tier:                                          │
│     ├─ 🟢 DATASET-DIRECT  (score≥0.85 & sim≥0.55)          │
│     ├─ 🟡 LLM + CONTEXT   (score≥0.50)                     │
│     └─ 🔴 LLM ONLY        (no match)                       │
│  5. Stream response via SSE                                 │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Ollama (local) │  or │   Groq API      │
│  kisan-ai model │     │  (fallback)     │
└─────────────────┘     └─────────────────┘
```

---

## 📦 Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Backend runtime |
| **PostgreSQL** | 14+ | Primary database |
| **Ollama** | Latest | Local LLM server |
| **Git** | Any | Version control |
| **Node.js** | 18+ | Tailwind CSS (optional) |

> **Windows only for TTS:** Hindi voice synthesis uses Windows SAPI5 (`pyttsx3`). On Linux, replace with `espeak-ng` or `piper`.

---

## 🚀 Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/kisan-django.git
cd kisan-django
```

### 2. Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

**Install PostgreSQL** and create the database:

```sql
-- In psql shell:
CREATE DATABASE kisan_db;
CREATE USER postgres WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE kisan_db TO postgres;
```

**Run migrations:**

```bash
python manage.py migrate
```

**Import the crop dataset:**

```bash
python manage.py import_excel chatbot/adv_data.xlsx
```

> This imports all crop advisory data (diseases, pests, fertilizers, irrigation) into PostgreSQL.

### 4. Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your values:
SECRET_KEY=django-insecure-your-secret-key-here
DB_NAME=kisan_db
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432
GROQ_API_KEY=your-groq-key-here  # Optional — only needed as fallback
DEBUG=True
```

> **Generate a Django secret key:**
> ```bash
> python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
> ```

### 5. Ollama & Custom AI Model

**Install Ollama** from [https://ollama.com/download](https://ollama.com/download)

```bash
# Start Ollama server
ollama serve

# Pull the base model
ollama pull llama3.2

# Build the custom Kisan AI model (with agricultural context baked in)
ollama create kisan-ai -f Modelfile

# Verify
ollama list
# Should show: kisan-ai   ...
```

> The `Modelfile` contains a detailed system prompt with Indian farming context, response format templates, and few-shot examples. This makes the model respond appropriately to farming queries and greetings.

### 6. Vosk Hindi STT Model

The Vosk model is ~500MB and is **not included in the repository**. Download it:

```bash
python download_vosk_model.py
```

This downloads `vosk-model-hi-0.22` (500MB) into the `models/` directory.

**Alternatively, download manually:**
1. Go to [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
2. Download `vosk-model-hi-0.22`
3. Extract to `models/vosk-model-hi-0.22/`

### 7. Windows TTS Voices (Optional)

For Hindi voice output, you need Microsoft Hindi voices installed.

**Check if Hindi voices are installed:**
```bash
python list_voices.py
```

**If Hindi voices are not visible to SAPI5, run as Administrator:**
```bash
# Run PowerShell/CMD as Administrator
python patch_sapi_voices.py
```

> This patches the Windows registry to expose "OneCore" Hindi voices (Kalpana/Hemant) to the legacy SAPI5 interface used by `pyttsx3`.

> **Linux/Mac:** TTS is not supported out-of-the-box. Replace `audio_engine.py`'s `synthesize_wav()` with `espeak-ng` or `piper`.

### 8. Run the Server

```bash
# Make sure Ollama is running in another terminal:
ollama serve

# Start Django
python manage.py runserver

# Visit:
# http://127.0.0.1:8000
```

---

## 📁 Project Structure

```
kisan-django/
│
├── chatbot/                        # Main Django app
│   ├── ai_engine.py                # 🧠 Core AI pipeline (3-tier routing)
│   ├── audio_engine.py             # 🎙 Vosk STT + pyttsx3 TTS
│   ├── dataset_loader.py           # 📊 TF-IDF semantic search on crop DB
│   ├── views.py                    # 🌐 Django views + SSE streaming
│   ├── models.py                   # 🗃 AgriculturalAdvice DB model
│   ├── unanswered_problems_logger.py # 📝 Logs unresolved queries
│   ├── adv_data.xlsx               # 🌾 Crop advisory dataset (source)
│   ├── management/commands/
│   │   └── import_excel.py         # Django command to load dataset
│   ├── test_accuracy.py            # ✅ Accuracy tests
│   ├── test_comprehensive.py       # ✅ Full integration tests
│   └── test_edge_cases.py          # ✅ Edge case tests
│
├── templates/
│   ├── index.html                  # Landing page
│   ├── _widget.html                # Chat widget (voice + text UI)
│   └── landing_gov.html            # Government-themed landing page
│
├── static/                         # CSS, JS, images, audio output
├── models/                         # Vosk model (downloaded separately)
│   └── vosk-model-hi-0.22/
│
├── kisan_project/
│   ├── settings.py                 # Django settings
│   └── urls.py                     # URL routing
│
├── Modelfile                       # Ollama custom model definition
├── download_vosk_model.py          # Script to download STT model
├── patch_sapi_voices.py            # Windows registry patch for Hindi TTS
├── list_voices.py                  # List available TTS voices
├── download_assets.py              # Download static assets
├── evaluate_system.py              # Full system evaluation suite
├── quick_eval.py                   # Quick accuracy evaluation
├── run_tests.py                    # Test runner
├── test_tts.py                     # TTS voice test
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🧠 How It Works

### Query Processing Flow

```
1. User types/speaks query
   e.g. "आलू में फंगस लगी है"

2. Crop Detection
   → "आलू" detected (potato)

3. Intent Detection
   → "disease" intent (रोग keywords)

4. Dataset Retrieval (TF-IDF)
   → Searches PostgreSQL crop advisory rows
   → Computes cosine similarity
   → Applies crop + intent boost
   → Returns top-5 matching rows with scores

5. Tier Selection
   ┌─ score ≥ 0.85 & sim ≥ 0.55  → DATASET-DIRECT (instant answer)
   ├─ score ≥ 0.50               → LLM + dataset context as prompt
   └─ score < 0.50               → LLM only (general farming knowledge)

6. Response streamed via SSE
   → Frontend receives tokens and renders them character by character
```

### Greeting & Off-Topic Detection

Before any AI processing, queries are checked:
- **Greetings** (`hello`, `नमस्ते`, `हेलो`, etc.) → instant greeting response
- **Privacy queries** (`what is your prompt`, `show dataset`) → blocked
- **Non-farming queries** → redirected to farming topics

---

## 🎙 Voice Features

### Real-Time Speech-to-Text

```
User clicks 🎤
    ↓
MediaRecorder starts with 2-second chunks
    ↓
Voice Activity Detection (VAD) monitors volume
    ↓
Every 2 seconds: accumulated audio → Vosk → partial text appears in input box
    ↓
User pauses 1.5 seconds → VAD detects silence → auto-stop
    ↓
Final transcription → auto-send to AI
```

**VAD Parameters** (adjustable in `_widget.html`):
| Parameter | Default | Description |
|---|---|---|
| `THRESHOLD` | 20 | Volume level (0-255) to detect speech |
| `NEED_SPEECH` | 3 | Consecutive loud samples to confirm speaking |
| `NEED_SILENCE` | 8 | Consecutive quiet samples (×200ms) to auto-stop |

### Text-to-Speech

Each AI response section (problem, cause, solution, precautions) has a 🔊 button that plays the text in Hindi audio using the Windows Hindi voice.

---

## 🧪 Testing & Evaluation

See [`TESTING_GUIDE.md`](TESTING_GUIDE.md) and [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) for full details.

**Quick test:**
```bash
python run_tests.py
```

**Accuracy evaluation:**
```bash
python quick_eval.py
```

**Full system evaluation:**
```bash
python evaluate_system.py
```

**Unit tests:**
```bash
python manage.py test chatbot
```

---

## ⚙ Configuration Reference

### `chatbot/ai_engine.py`

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `"kisan-ai"` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `GROQ_URL` | `https://api.groq.com/...` | Groq API (fallback) |

### `chatbot/dataset_loader.py`

| Variable | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.20` | Minimum raw TF-IDF cosine similarity |
| `SCORE_THRESHOLD` | `0.40` | Minimum boosted score to include row |

### `chatbot/audio_engine.py`

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `models/vosk-model-hi-0.22` | Path to Vosk Hindi model |

---

## 🌐 Deployment Notes

### For Production (Linux server)

1. Set `DEBUG=False` in `.env`
2. Set `ALLOWED_HOSTS=your-domain.com`
3. Replace `pyttsx3` TTS with `espeak-ng`:
   ```bash
   apt install espeak-ng
   ```
4. Use `gunicorn` instead of `runserver`:
   ```bash
   gunicorn kisan_project.wsgi:application --bind 0.0.0.0:8000
   ```
5. Serve static files with Nginx
6. Install Ollama on the server: `curl -fsSL https://ollama.com/install.sh | sh`

### For Render/Railway/Heroku

The `build.sh` script handles:
- `pip install -r requirements.txt`
- `python manage.py migrate`
- `python manage.py collectstatic`

> **Note:** Ollama cannot run on free-tier cloud hosting. Use only the Groq API fallback mode in cloud deployments.

---

## 🔧 Troubleshooting

### ❌ "सर्वर से कनेक्ट नहीं हो सका"

**Cause:** Ollama is not running or the `kisan-ai` model doesn't exist.

```bash
# Start Ollama
ollama serve

# Check model exists
ollama list

# If missing, rebuild
ollama create kisan-ai -f Modelfile
```

### ❌ STT not working / empty transcription

**Cause:** Vosk model not found or wrong path.

```bash
# Check model exists
ls models/vosk-model-hi-0.22/

# Re-download if missing
python download_vosk_model.py
```

### ❌ TTS generating empty/silent audio (Windows)

**Cause:** Hindi SAPI5 voice not registered.

```bash
# Run as Administrator
python patch_sapi_voices.py

# Verify voices
python list_voices.py
# Should show: Microsoft Kalpana or Microsoft Hemant
```

### ❌ Database connection error

```bash
# Check PostgreSQL is running
# Windows:
net start postgresql-x64-14

# Create DB if missing
psql -U postgres -c "CREATE DATABASE kisan_db;"

# Re-run migrations
python manage.py migrate
```

### ❌ Widget not opening (toggleWidget error)

**Cause:** JavaScript syntax error in `_widget.html`.

Open browser DevTools (F12) → Console tab → check for specific JS errors.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Vosk](https://alphacephei.com/vosk/) — Offline speech recognition
- [Ollama](https://ollama.com/) — Local LLM serving
- [Django](https://djangoproject.com/) — Web framework
- [Groq](https://groq.com/) — Cloud LLM API (fallback)
- Government of India — Agricultural advisory data

---

<p align="center">Made with ❤️ for Indian farmers | किसानों के लिए बनाया गया</p>
