<div align="center">

# ClinicDx

### Offline AI Clinical Intelligence for Edge Deployment

**Trimodal Inference · Multi-Turn RAG · Clinical Intent Reranking · Audio-to-FHIR Pipeline**

Powered by [MedGemma](https://huggingface.co/google/medgemma-4b-it) · Runs on [llama.cpp](https://github.com/ggerganov/llama.cpp) · Works with [OpenMRS O3](https://openmrs.org/) or any EMR · Fully offline

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![npm](https://img.shields.io/npm/v/@clinicdx/esm-clinicdx-app?label=npm&color=cb3837)](https://www.npmjs.com/package/@clinicdx/esm-clinicdx-app)
[![OpenMRS](https://img.shields.io/badge/OpenMRS-O3-blue?logo=data:image/svg+xml;base64,)](https://openmrs.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-ClinicDx1%2FClinicDx-yellow)](https://huggingface.co/ClinicDx1/ClinicDx)
[![Website](https://img.shields.io/badge/Website-clinicdx.org-informational)](https://clinicdx.org)

[**Website**](https://clinicdx.org) · [**npm Package**](https://www.npmjs.com/package/@clinicdx/esm-clinicdx-app) · [**Model on HuggingFace**](https://huggingface.co/ClinicDx1/ClinicDx) · [**Research**](RESEARCH.md) · [**Report a Bug**](https://github.com/ClinicDx/ClinicDx/issues)

</div>

---

## Research Contributions

ClinicDx is an open-source trimodal inference system for edge clinical AI — combining a medical ASR encoder, a learned audio projector, and a fine-tuned 4B clinical LLM in a single llama.cpp binary, deployable fully offline on consumer hardware. See [**RESEARCH.md**](RESEARCH.md) for full details, key results, open problems, and comparison to prior work.

- **AudioProjector architecture** — a 2-layer MLP + RMSNorm projector (11.8M params) that maps MedASR Conformer embeddings into Gemma3's token space with a fixed 64-token budget, enabling constant KV-cache allocation on memory-constrained devices.
- **Multi-turn ReAct CDS** — the model controls its own evidence retrieval via `<KB_QUERY>` / `<KB_RESULT>` tool-use over up to 5 turns, unlike standard single-pass RAG where retrieval is a fixed preprocessing step.
- **Clinical intent reranking** — a 4-slot reranker (condition / severity / population / task) over 27,860 WHO/MSF guideline chunks, with multiplicative penalties and additive boosts for domain-specific retrieval quality.
- **LoRA fine-tuning on clinical reasoning** — r=64, α=128 on 27,592 traces with input masking (64% context masked) and a hybrid teacher strategy, achieving 86.25% accuracy at eval loss 0.4758 on a 4B model.
- **EMR-agnostic inference architecture** — the clinical AI engine is decoupled from any specific EMR, validating that the inference pipeline generalizes across health information systems — a prerequisite for LMIC deployment.

---

## Overview

ClinicDx ships as **two independent components**:

| Component | What it is | How it ships |
|---|---|---|
| **CDS Engine** | Model inference + knowledge base + CDS/Scribe middleware | `docker compose up` — works with any EMR |
| **OpenMRS Module** | React frontend (CDS workspace, Scribe, OCR, Imaging) | npm package — `@clinicdx/esm-clinicdx-app` |

The engine is **EMR-agnostic**. Any EMR can call its REST API for clinical decision support and voice scribe capabilities. OpenMRS is one supported integration, not a requirement.

| Capability | Description |
|---|---|
| **Clinical Decision Support** | Structured 6-section assessments with WHO/MSF evidence citations, generated via multi-turn retrieval-augmented reasoning |
| **Voice Scribe** | Natural language clinical observations → structured FHIR R4 observations written directly to the EMR |
| **Document OCR** | Digitise referral letters, lab reports, and prescriptions |
| **Imaging Analysis** | AI-assisted interpretation of clinical images |

Everything runs on a single 4B-parameter multimodal model fine-tuned from Google MedGemma, operating fully offline — no data leaves the facility.

---

## Model and Training

**[`ClinicDx1/ClinicDx`](https://huggingface.co/ClinicDx1/ClinicDx)** on HuggingFace

| File | Size | Purpose |
|---|---|---|
| [`clinicdx-v1-q8.gguf`](https://huggingface.co/ClinicDx1/ClinicDx/blob/main/clinicdx-v1-q8.gguf) | 3.9 GB | Language model (Q8_0 — chosen for structured output preservation) |
| [`medasr-encoder.gguf`](https://huggingface.co/ClinicDx1/ClinicDx/blob/main/medasr-encoder.gguf) | 401 MB | MedASR Conformer audio encoder (frozen, 105M params) |
| [`audio-projector-v3-best.gguf`](https://huggingface.co/ClinicDx1/ClinicDx/blob/main/audio-projector-v3-best.gguf) | 46 MB | AudioProjector v3 — best checkpoint (step 40,000, val LM 0.1042) |
| [`who_knowledge_vec_v2.mv2`](https://huggingface.co/ClinicDx1/ClinicDx/blob/main/who_knowledge_vec_v2.mv2) | 1.1 GB | WHO/MSF knowledge base v2.1 (27,860 chunks, BM25 + semantic hybrid) |

### Training Pipeline

```
Google MedGemma 4B-IT  (base)
       │
       ▼  Stage 1 — CDS LoRA Fine-Tuning
       │  LoRA (r=64, α=128) on 27,592 quality-filtered clinical conversations
       │  Hybrid teacher: thinking model (antehoc reasoning) + instruct model (KB tool-use)
       │  Input masking: only model output turns trained (64% context masked)
       │  Best checkpoint: step 4,000  |  eval loss 0.4758  |  accuracy 86.25%
       │
       ▼  Merge LoRA  →  ClinicDx V1  (production CDS model)
       │
       ▼  Stage 2 — AudioProjector Training  (base model frozen)
          2-layer MLP + RMSNorm: MedASR (512-dim) → LLM space (2560-dim)
          Frame stacking k=4: [B, T_enc, 512] → [B, T/4, 2048] → [B, 64, 2560]
          50,000 synthetic clinical audio clips  |  10 epochs
          Best checkpoint: step 40,000  |  val LM 0.1042  |  key accuracy 84%
          11,806,720 trainable parameters
```

### AudioProjector Architecture

```
MedASR Conformer Output  [B, T_enc, 512]
    │
    ▼  Frame stacking (k=4)
  [B, T_enc/4, 2048]
    │
    ▼  Linear(2048 → 2560, no bias) → RMSNorm(2560) → GELU → Linear(2560 → 2560, no bias)
  [B, T_enc/4, 2560]
    │
    ▼  Pad (learnable audio_padding_emb) or truncate to 64 tokens
  [B, 64, 2560]  →  Gemma3 LLM embedding space
```

Token budget: 1s audio → 13 projected frames, 3s → 38, 5s → 63. The fixed 64-token output guarantees constant KV-cache consumption.

---

## System Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                Component 1: CDS Engine (Docker)                 │
 │                                                                 │
 │  ┌────────────────────────────────────────────────────────────┐ │
 │  │  FastAPI Middleware  :8321 (exposed to host)               │ │
 │  │   cds_router.py    — Multi-turn ReAct CDS + SSE streaming  │ │
 │  │   scribe_router.py — Audio → structured observations       │ │
 │  └──────────────────┬────────────────────┬────────────────────┘ │
 │                     │                    │                      │
 │           ┌─────────┘                    └──────────┐           │
 │           ▼                                         ▼           │
 │  ┌─────────────────────┐         ┌─────────────────────────┐    │
 │  │  KB Daemon v2 :4276 │         │  llama-server :8180     │    │
 │  │                     │         │                         │    │
 │  │  WHO Guidelines     │         │  MedASR Encoder (105M)  │    │
 │  │  MSF Protocols      │         │  AudioProjector v3      │    │
 │  │  BM25 + Semantic    │         │  ClinicDx LLM (4.3B, Q8)│    │
 │  │  Hybrid Retrieval   │         │                         │    │
 │  └─────────────────────┘         └─────────────────────────┘    │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                    REST API (any EMR can call)
                               │
         ┌─────────────────────┼──────────────────────┐
         │                     │                      │
         ▼                     ▼                      ▼
 ┌───────────────┐    ┌──────────────┐     ┌──────────────────┐
 │ OpenMRS O3    │    │ Other EMR    │     │ curl / any       │
 │ @clinicdx/esm │    │ (FHIR, REST) │     │ HTTP client      │
 │ -clinicdx-app │    │              │     │                  │
 │ (Component 2) │    │              │     │                  │
 └───────────────┘    └──────────────┘     └──────────────────┘
```

### CDS Flow — Multi-Turn ReAct with KB Tool-Use

```
1.  Frontend sends patient case  →  POST /cds/generate_stream
2.  Middleware builds Gemma chat prompt with encounter context
3.  Model streams thinking block  →  emits <KB_QUERY>term</KB_QUERY>
4.  Middleware queries KB daemon  →  injects <KB_RESULT>evidence</KB_RESULT>
5.  Up to 5 retrieval turns until structured response is complete
6.  SSE stream delivers 6-section markdown response with WHO citations
```

**CDS Response Schema:**
1. **Alert Level** — Routine / Urgent / Emergency
2. **Clinical Assessment** — Findings summary and reasoning
3. **Differential Considerations** — Ranked diagnoses with rationale
4. **Recommended Actions** — Investigations and management steps
5. **Safety Alerts** — Red-flag signs, interactions, contraindications
6. **Key Points** — Concise handover summary

### Scribe Flow — Direct Audio to FHIR

```
1.  Doctor records voice note in browser  →  POST /scribe/process_audio
2.  Middleware transcodes to 16kHz mono PCM-16 WAV (via ffmpeg)
3.  WAV → POST /v1/audio/extract on llama-server
4.  MedASR Conformer encodes audio  →  [T_enc, 512]
5.  AudioProjector projects  →  [64, 2560]  (fixed token budget)
6.  LLM decodes structured observations  →  "key: value" lines
7.  Middleware maps keys to CIEL concept codes  →  FHIR R4 payloads
8.  Doctor reviews and confirms  →  POST to OpenMRS FHIR API
```

---

## Knowledge Base

The KB daemon serves a 27,860-chunk index built from WHO and MSF clinical guidelines, processed through a Docling + HybridChunker pipeline with safety keyword detection for life-threatening conditions.

### Retrieval Pipeline

| Stage | Method | Details |
|---|---|---|
| Lexical search | BM25 via Tantivy | Fast, no GPU required |
| Semantic search | EmbedGemma 300M | Multi-window pooling for long chunks |
| Fusion | Reciprocal Rank Fusion (RRF) | Merges lexical + semantic rankings |
| Reranking | 4-slot clinical intent reranker | Domain-specific rescoring |

### Clinical Intent Reranker

The reranker extracts structured intent from each query and rescores hits based on slot alignment:

| Slot | Extraction | Effect on mismatched hits |
|---|---|---|
| **Condition** | 30+ clinical conditions with inclusion/exclusion regex | ×0.12 penalty for off-condition content |
| **Severity** | Light vs. severe presentation markers | ×0.35 penalty for severity mismatch |
| **Population** | Adult / child / neonate detection | ×0.30 penalty for wrong age group |
| **Task** | Dosing / diagnosis / first-line / referral / prevention | Boost for preferred content types |

Content-type boosting (×1.50 for recommendations and protocols, ×0.25 for annexes) and multi-slot match bonuses (+0.15 for 3+ slots aligned) further refine ranking. A floor of 0.10 prevents any hit from being completely suppressed.

The reranker includes condition-specific overrides (e.g., snakebite ≠ scorpion, sickle cell ACS disambiguation) and background rescue logic that promotes on-condition background chunks when no actionable content is available.

---

## Deployment

### Requirements

| Requirement | Version |
|---|---|
| Docker Engine | ≥ 24 |
| Docker Compose plugin | ≥ 2.24 |
| NVIDIA Container Toolkit *(GPU mode)* | CUDA 11.8+ |
| GPU VRAM *(GPU mode)* | ≥ 8 GB |
| Disk space | ~10 GB (models + KB index + embeddings + Docker images) |

### Engine Only (any EMR)

```bash
git clone https://github.com/ClinicDx/ClinicDx.git && cd ClinicDx
cp .env.example .env
make up           # GPU
# or: make up-cpu # CPU only
```

On first start, all artifacts (~6.6 GB) auto-download from HuggingFace. Subsequent starts are instant.

| What downloads | From | Size |
|---|---|---|
| `clinicdx-v1-q8.gguf` | `ClinicDx1/ClinicDx` | 3.9 GB |
| `medasr-encoder.gguf` | `ClinicDx1/ClinicDx` | 401 MB |
| `audio-projector-v3-best.gguf` | `ClinicDx1/ClinicDx` | 46 MB |
| `who_knowledge_vec_v2.mv2` | `ClinicDx1/ClinicDx` | 1.1 GB |
| EmbedGemma 300M | `google/embeddinggemma-300m` | ~1.2 GB |

Once running, the engine API is at `http://localhost:8321`:

```bash
curl http://localhost:8321/api/health
curl -X POST http://localhost:8321/cds/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<bos><start_of_turn>user\nPatient: 5-year-old, fever 39°C for 3 days, cough, rapid breathing.\n<end_of_turn>\n<start_of_turn>model\n"}'
```

### Full Stack (with nginx + OpenMRS)

```bash
git clone https://github.com/ClinicDx/ClinicDx.git && cd ClinicDx
cp .env.example .env

mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/server.key -out certs/server.crt -subj "/CN=localhost"

make up-full           # GPU
# or: make up-full-cpu # CPU only
```

### Air-Gap / Offline Deployment

For facilities without internet access, pre-download all artifacts on a connected machine:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='ClinicDx1/ClinicDx', allow_patterns=['*.gguf', '*.mv2'], local_dir='./artifacts/clinicdx')
snapshot_download(repo_id='google/embeddinggemma-300m', local_dir='./artifacts/embeddinggemma-300m')
"
tar -czf clinicdx_artifacts.tar.gz artifacts/
```

On the target machine:

```bash
docker compose --profile gpu up -d
docker cp artifacts/clinicdx/. $(docker volume inspect clinicdx-engine_model_data -f '{{.Mountpoint}}')
docker cp artifacts/clinicdx/. $(docker volume inspect clinicdx-engine_kb_data -f '{{.Mountpoint}}')
docker cp artifacts/embeddinggemma-300m/. $(docker volume inspect clinicdx-engine_hf_cache -f '{{.Mountpoint}}')/embeddinggemma-300m/
```

### Configuration

All configuration is via environment variables (copy `.env.example` to `.env`):

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(empty)* | HuggingFace token (only if repo is gated) |
| `ENGINE_PORT` | `8321` | Port the middleware exposes to the host |
| `N_GPU_LAYERS` | `999` | GPU offload layers (0 = CPU only) |
| `MODEL_CTX` | `8192` | Context window size (tokens) |
| `MODEL_PARALLEL` | `1` | Inference slots — **must remain 1** (audio pipeline constraint) |
| `MODEL_THREADS` | `8` | CPU inference threads |
| `KB_SEARCH_MODE` | `rrf` | `rrf` (hybrid), `semantic`, or `lexical` |
| `KB_K` | `5` | Number of KB results per retrieval turn |
| `OPENMRS_URL` | *(empty)* | OpenMRS backend URL (enables Scribe manifest/confirm endpoints) |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARN` / `ERROR` |

> **Note:** `MODEL_PARALLEL` must be set to `1`. The audio extraction endpoint performs blocking KV-cache operations that conflict with parallel slot inference.

---

## OpenMRS Integration

The OpenMRS frontend module is one deployment target for the engine. Install it into an existing OpenMRS O3 instance:

```json
{
  "frontendModules": {
    "@clinicdx/esm-clinicdx-app": "latest"
  }
}
```

Configure the engine URL in OpenMRS admin:

```json
{
  "@clinicdx/esm-clinicdx-app": {
    "middlewareUrl": "http://your-engine-host:8321"
  }
}
```

The Scribe module supports 29 CIEL clinical concepts across 7 encounter types:

**Vital Signs:** temperature, blood pressure (systolic/diastolic), pulse, SpO₂, respiratory rate, BMI, weight, height

**Lab Results:** CD4 count, CD4 percent, fasting glucose, finger-stick glucose, post-prandial glucose, serum glucose

**Clinical Observations:** duration of illness, Glasgow Coma Scale, visual analogue pain score, missed medication doses, urine output, fundal height, fetal heart rate, estimated gestational age

**Obstetric / Pediatric:** pre-gestational weight, birth weight, weight gain since last visit, weight on admission, head circumference, MUAC

---

## API Reference

### Engine endpoints (EMR-agnostic, port 8321)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Full stack health — model, KB, and middleware status |
| `POST` | `/cds/generate` | Clinical decision support (blocking) |
| `POST` | `/cds/generate_stream` | CDS with SSE token streaming |
| `POST` | `/scribe/process` | Transcription text → structured observations |
| `POST` | `/scribe/process_audio` | Raw audio → structured observations (direct pipeline) |

### OpenMRS endpoints (require `OPENMRS_URL`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/scribe/manifest?encounter_uuid=` | Build encounter concept manifest from OpenMRS |
| `POST` | `/scribe/confirm` | POST confirmed FHIR observations to OpenMRS |

OpenMRS-dependent endpoints return HTTP 501 if `OPENMRS_URL` is not configured, allowing standalone operation.

### Knowledge Base Daemon (port 4276)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns index version |
| `POST` | `/search` | Query KB: `{"query": "...", "top_k": 5}` |
| `GET` | `/stats` | Index statistics |

### Model Server (port 8180)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health |
| `POST` | `/v1/completions` | Text generation (OpenAI-compatible) |
| `POST` | `/v1/audio/extract` | Audio WAV → structured observations (multipart/form-data) |

---

## Limitations and Open Problems

This section documents known limitations honestly. See [RESEARCH.md](RESEARCH.md) for a more detailed discussion.

- **No formal clinical validation.** Accuracy metrics (86.25% CDS, 84% Scribe) are measured on held-out synthetic data, not on real clinical encounters. A prospective evaluation with practicing clinicians in LMIC facilities is the highest-priority gap.
- **English only.** All CDS outputs, Scribe extraction, and KB retrieval operate in English. Multilingual support (Swahili, Amharic, Hausa, Yoruba) is planned but not yet implemented.
- **Audio token norm mismatch.** Projected audio token norms are 360-620× larger than text token norms in the LLM embedding space. The current mitigation uses adaptive norm alignment loss with scheduled decay, but this remains an active area of investigation.
- **No ARM / low-power benchmarks.** The system has been validated on x86_64 with NVIDIA GPUs and CPU-only mode. Latency and throughput on ARM edge devices are unknown.
- **Q8 quantization chosen empirically.** Q8_0 was selected over Q4 variants because lower quantization degraded structured output behavior (THINK block coherence, KB query emission) more than perplexity. No systematic ablation study has been conducted.
- **KB coverage gaps.** The knowledge base covers WHO and MSF guidelines. Region-specific protocols, national formularies, and conditions not prominently featured in WHO guidelines are not covered.

---

## Repository Layout

```
ClinicDx/
├── RESEARCH.md                  Research contributions, results, and open problems
├── docker-compose.yml           CDS Engine stack (model + KB + middleware)
├── docker-compose.full.yml      Full stack (engine + nginx for OpenMRS)
├── Makefile                     up / up-cpu / up-full / up-full-cpu / down / logs
├── .env.example                 Configuration template
│
├── services/
│   ├── middleware/              FastAPI middleware (CDS orchestration + Scribe)
│   │   └── service/
│   │       ├── api.py           FastAPI app entry point
│   │       ├── cds_router.py    Multi-turn ReAct CDS + SSE streaming
│   │       ├── scribe_router.py Audio → FHIR pipeline
│   │       ├── manifest.py      Encounter → concept manifest (29 CIEL concepts)
│   │       ├── fhir_builder.py  FHIR R4 resource construction
│   │       └── ciel_mappings.json  CIEL concept → OpenMRS UUID map
│   │
│   ├── knowledge-base/          Knowledge Base daemon (v2.1 index)
│   │   └── kb/
│   │       ├── daemon_v2.py     HTTP server (port 4276)
│   │       ├── retrieval_core_v2.py  BM25 + semantic hybrid + intent reranker
│   │       └── embedder.py      EmbedGemma 300M wrapper
│   │
│   └── unified-model-server/    AudioProjector model definition + serving
│       └── modeling/
│           └── gemma3_audio.py  Gemma3AudioProjector architecture
│
├── openmrs-module/              OpenMRS O3 ESM microfrontend (TypeScript/React)
│   └── src/
│       ├── cds/                 CDS workspace and action button
│       │   └── case-builder/    Patient context + middleware API
│       ├── scribe/              Voice Scribe workspace
│       ├── imaging/             Imaging analysis workspace
│       └── ocr/                 OCR workspace
│
├── docker/
│   ├── kb/                      KB Dockerfile + entrypoint
│   ├── model/                   Model Dockerfile + entrypoint
│   ├── middleware/              Middleware Dockerfile
│   └── nginx/                   Nginx reverse proxy (full stack only)
│
└── docs/
    ├── architecture.md          System architecture details
    └── deployment.md            Deployment guide
```

---

## Roadmap

- [ ] Multi-language Scribe (Swahili, Amharic, Hausa, Yoruba)
- [ ] Prospective clinical validation study in LMIC facilities
- [ ] ARM / low-power edge hardware benchmarks
- [ ] Quantization ablation study (Q4/Q5/Q6/Q8 behavioral effects)
- [ ] Differential diagnosis confidence scores
- [ ] Federated learning for multi-facility model improvement
- [ ] Expanded KB: Africa CDC protocols, national formularies
- [ ] Offline model update via USB/sneakernet

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for significant changes.

```bash
git clone https://github.com/ClinicDx/ClinicDx.git
cd ClinicDx
make test-unit
```

Areas where contributions are especially needed:
- Clinical validation studies
- Additional CIEL concept mappings
- Regional language support for Scribe
- Knowledge base expansion (national formularies, Africa CDC)

---

## Built With

| Component | Project |
|---|---|
| Base LLM | [Google MedGemma 4B-IT](https://huggingface.co/google/medgemma-4b-it) |
| Inference Runtime | [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| Audio Encoder | MedASR (Conformer, LASR architecture) |
| Knowledge Base | [memvid](https://github.com/Oaynerad/memvid) |
| Embedding Model | [EmbedGemma 300M](https://huggingface.co/google/embeddinggemma-300m) |
| Clinical Terminology | [CIEL](https://www.cielterminology.org/) |
| FHIR Standard | [HL7 FHIR R4](https://hl7.org/fhir/R4/) |
| EMR Platform | [OpenMRS O3](https://openmrs.org/) |

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** — see [LICENSE](LICENSE) for details.

Model weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

<div align="center">

**[clinicdx.org](https://clinicdx.org)** · **[npm](https://www.npmjs.com/package/@clinicdx/esm-clinicdx-app)** · **[HuggingFace](https://huggingface.co/ClinicDx1/ClinicDx)** · **[GitHub](https://github.com/ClinicDx/ClinicDx)** · **[Issues](https://github.com/ClinicDx/ClinicDx/issues)**

*Built for clinicians in under-resourced settings. Every observation captured, every diagnosis supported.*

</div>
