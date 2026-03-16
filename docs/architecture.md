# ClinicDx — Architecture Deep Dive

## System Components

### 1. Model Server — llama.cpp (`llama-server`)

Production inference uses [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` (port 8180), serving three model components from a single process:

```
Model Stack (single llama-server binary):
  MedASR Conformer encoder        105M params  (frozen, GGUF)
  AudioProjector v3                11.8M params (trained, GGUF)
  ClinicDx LLM (MedGemma 4B, Q8)  4.3B params  (LoRA merged, GGUF)
```

The model server exposes:
- `POST /v1/completions` — OpenAI-compatible text generation (CDS)
- `POST /v1/audio/extract` — audio WAV → structured observations (Scribe)
- `GET /health` — server health check

`MODEL_PARALLEL` must remain `1` because the audio extraction endpoint performs blocking KV-cache operations that conflict with parallel slot inference.

### 2. AudioProjector Architecture (`services/unified-model-server/modeling/gemma3_audio.py`)

The AudioProjector maps MedASR Conformer encoder outputs into Gemma3's LLM embedding space. It mirrors how vision is integrated in Gemma3 (SigLIP → MultiModalProjector → LLM) but for audio (MedASR → AudioProjector → LLM).

```
Input:  [B, T_enc, 512]  (MedASR Conformer output)
    │
    ▼  Frame stacking (k=4): concat 4 adjacent frames
  [B, T_enc/4, 2048]
    │
    ▼  Linear(2048 → 2560, no bias)
    ▼  RMSNorm(2560)
    ▼  GELU
    ▼  Linear(2560 → 2560, no bias)
  [B, T_enc/4, 2560]
    │
    ▼  Pad (learnable audio_padding_emb) or truncate to 64 tokens
Output: [B, 64, 2560]  (Gemma3 embedding dimension)
```

**Design decisions:**
- **Frame stacking k=4** reduces temporal resolution before projection, compressing ~250 encoder frames/second down to ~63 projected tokens/second.
- **Fixed 64-token budget** via `_adjust_to_expected_length` guarantees constant KV-cache consumption regardless of audio duration. Short utterances are padded with a *learnable* `audio_padding_emb` parameter (not zeros).
- **RMSNorm** (matching Gemma3's internal normalization) instead of LayerNorm.
- **No bias** in linear layers, consistent with Gemma3's language model weights.

Token budget per audio duration (at 16kHz, hop=160, subsample=2×, stack=4×):
1s → 13 projected frames | 3s → 38 | 5s → 63 | 10s → 125 (truncated to 64)

### 3. Python Middleware (`services/middleware/`)

FastAPI service (port 8321) that orchestrates requests between frontend, model server, and KB daemon.

**CDS Router** (`service/cds_router.py`):
- Implements multi-turn ReAct loop (up to 5 turns, up to 5 KB queries)
- Model emits `<KB_QUERY>term</KB_QUERY>` → middleware intercepts, queries KB daemon, injects `<KB_RESULT source="..." score="...">...</KB_RESULT>` back into the generation context
- Streaming endpoint via SSE (`/cds/generate_stream`)
- Includes depth rules that enforce structured output format (6 sections with specific clinical detail requirements)
- KB query rules that enforce clinically-specific search terms (e.g., "severe malaria artesunate dose" not "management")

**Scribe Router** (`service/scribe_router.py`):
- `/scribe/process_audio` — transcodes audio to 16kHz mono PCM-16 WAV (ffmpeg), sends to model server `/v1/audio/extract`, maps results to FHIR R4
- `/scribe/manifest` — builds CIEL concept manifest from live OpenMRS encounter (requires `OPENMRS_URL`)
- `/scribe/confirm` — POSTs confirmed FHIR R4 Observation resources to OpenMRS FHIR API
- OpenMRS-dependent endpoints return HTTP 501 when `OPENMRS_URL` is not set

### 4. Knowledge Base Daemon (`services/knowledge-base/`)

Minimal stdlib-only HTTP server (port 4276, no FastAPI dependency) serving a single memvid v2 index:

| Index | Size | Contents |
|---|---|---|
| `who_knowledge_vec_v2.mv2` | 1.1 GB | 27,860 chunks from WHO/MSF clinical guidelines |

**Retrieval pipeline** (`retrieval_core_v2.py`):

| Stage | Implementation | Details |
|---|---|---|
| Lexical search | Tantivy BM25 | Built into memvid v2 index |
| Semantic search | EmbedGemma 300M | Multi-window pooling for long chunks |
| Fusion | Reciprocal Rank Fusion (RRF) | Merges lexical + semantic rankings |
| Content-type boost | Multiplier per chunk type | ×1.50 recommendation/protocol, ×0.25 annex |
| Retrieval priority | Per-chunk metadata | Superseded chunks (rp=0.0) hard-dropped |
| Intent reranking | 4-slot clinical reranker | Condition/severity/population/task alignment |

**4-slot intent reranker** (`_intent_rerank()`, ~500 lines):

Extracts structured clinical intent from the query and rescores each hit:
- **Condition slot**: 30+ clinical conditions with inclusion/exclusion regex. On-condition boost (+0.25), off-condition penalty (×0.12). Condition-specific overrides (snakebite≠scorpion, sickle cell ACS disambiguation, rabies≠malaria).
- **Severity slot**: Light vs. severe presentation detection. Severity mismatch penalty (×0.35).
- **Population slot**: Adult/child/neonate detection. Wrong age group penalty (×0.30 for neonatal content when adult queried).
- **Task slot**: Dosing/diagnosis/first-line/referral/prevention. Preferred content type boost.
- **Multi-slot bonus**: +0.15 for 3+ slots aligned, +0.05 for 2 slots.
- **Floor**: penalty never drops below 0.10, preventing complete suppression.
- **Background rescue**: promotes on-condition background chunks when no actionable content is available.

**Embedder** (`embedder.py`):
- EmbedGemma 300M for semantic search
- Multi-window pooling for chunks exceeding the model's context window

### 5. OpenMRS ESM Module (`openmrs-module/`)

A [single-spa](https://single-spa.js.org/) microfrontend built with the OpenMRS O3 framework. Registers four workspaces into the OpenMRS patient chart:

- **CDS workspace** — sends patient context to `/cds/generate_stream`, renders streaming markdown with live thinking panel and KB evidence cards
- **Scribe workspace** — records audio via `MediaRecorder`, posts to `/scribe/process_audio`, shows extracted observations for review and confirmation
- **OCR workspace** — image/document upload and text extraction
- **Imaging workspace** — clinical image analysis

Configuration (via OpenMRS config system):
```json
{
  "@clinicdx/esm-clinicdx-app": {
    "middlewareUrl": "http://localhost:8321"
  }
}
```

## Data Formats

### CDS Prompt Format (Gemma chat template)

The multi-turn ReAct loop builds a Gemma chat prompt with interleaved model thinking and KB results:

```
<bos><start_of_turn>user
[DEPTH_RULES]
[KB_SEARCH_RULES]
{patient_case_text}<end_of_turn>
<start_of_turn>model
<think>
  QUERY_ESTIMATE: 2
  <KB_QUERY>malaria diagnosis criteria</KB_QUERY>
</think>
<end_of_turn>
<start_of_turn>user
<KB_RESULT source="WHO Guidelines" score="18.5">
  ...WHO malaria treatment content...
</KB_RESULT><end_of_turn>
<start_of_turn>model
## Clinical Assessment
...
```

The middleware injects DEPTH_RULES (structured output format requirements) and KB_SEARCH_RULES (query shape guidance) as part of the initial prompt.

### Scribe Output Format
```
temperature: 37.8
blood_pressure: 120/80
chief_complaint: headache and fever
malaria_test_result: positive
```

### FHIR R4 Observation (built by `fhir_builder.py`)
```json
{
  "resourceType": "Observation",
  "status": "final",
  "code": {
    "coding": [
      {"code": "<local_uuid>"},
      {"system": "https://cielterminology.org", "code": "5088"}
    ]
  },
  "subject": {"reference": "Patient/<uuid>"},
  "encounter": {"reference": "Encounter/<uuid>"},
  "valueQuantity": {"value": 37.8, "unit": "Cel"}
}
```

## Deployment Topology

Production runs via Docker Compose:
- Port 8180: llama-server (GPU — model + audio encoder + projector)
- Port 4276: KB Daemon (CPU — memvid v2 index + EmbedGemma 300M)
- Port 8321: Middleware (CPU — FastAPI, exposed to host)
- Optional: nginx reverse proxy (full stack with OpenMRS)
