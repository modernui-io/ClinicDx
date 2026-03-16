# ClinicDx — Research Contributions

ClinicDx is an open-source, offline-first, trimodal clinical AI system designed for edge deployment in sub-Saharan Africa. This document describes the research contributions, key results, open problems, and relationship to prior work.

For deployment instructions, see the main [README](README.md).

---

## Contributions

### 1. Trimodal Edge Inference Architecture

A single llama.cpp binary serves text (CDS), audio (Scribe), and vision (OCR/Imaging) through a unified inference pipeline. The audio modality is integrated via a custom **AudioProjector** — a 2-layer MLP with RMSNorm that maps MedASR Conformer encoder embeddings (512-dim) into the Gemma3 LLM embedding space (2560-dim) using frame stacking (k=4) and a fixed 64-token output budget.

The fixed token budget is a deliberate architectural constraint: `_adjust_to_expected_length` pads short utterances with a learnable `audio_padding_emb` or truncates long ones, guaranteeing constant KV-cache consumption regardless of audio duration. This enables predictable memory allocation on memory-constrained edge hardware.

**Architecture:**
```
MedASR Conformer (105M, frozen)
  [B, T_enc, 512]
    → Frame stacking k=4  →  [B, T/4, 2048]
    → Linear(2048 → 2560, no bias)
    → RMSNorm(2560)
    → GELU
    → Linear(2560 → 2560, no bias)
    → Pad/truncate to 64 tokens
  [B, 64, 2560]  →  Gemma3 LLM (4.3B, frozen)
```

11,806,720 trainable parameters. The entire rest of the model (MedASR encoder + Gemma3 language model + vision tower) remains frozen.

### 2. Multi-Turn ReAct CDS with Knowledge Base Tool-Use

The CDS pipeline implements a multi-turn ReAct loop where the model controls its own evidence retrieval. During generation, the model emits `<KB_QUERY>term</KB_QUERY>` tags when it needs evidence. The middleware intercepts these, queries the KB daemon, and injects `<KB_RESULT source="..." score="...">...</KB_RESULT>` blocks back into the context. This continues for up to 5 retrieval turns before the model produces its structured clinical assessment.

This differs from standard single-pass RAG: the model decides *what* to retrieve and *when*, based on its evolving reasoning state. The model is fine-tuned to emit structured `<KB_QUERY>` tags with clinically-specific search terms (e.g., "severe malaria artesunate dose") rather than generic queries.

### 3. Clinical Intent Reranking for WHO/MSF Guideline Retrieval

The knowledge base retrieval pipeline uses a hybrid BM25 + semantic search with Reciprocal Rank Fusion (RRF), followed by a domain-specific **4-slot intent reranker** that extracts clinical intent from the query and rescores hits:

| Slot | What it matches | Effect |
|---|---|---|
| **Condition** | 30+ clinical conditions with inclusion/exclusion regex | Boost on-target chunks, penalize off-condition (×0.12) |
| **Severity** | Light vs. severe presentations | Demote mismatched severity (×0.35) |
| **Population** | Adult / child / neonate | Demote wrong population (×0.30) |
| **Task** | Dosing / diagnosis / first-line / referral / prevention | Boost preferred content types |

Multi-slot match bonus: +0.15 for 3+ slots aligned, +0.05 for 2 slots. Content-type boosting (×1.50 for recommendations, ×0.25 for annexes) and domain coherence checks further refine results.

The reranker operates over 27,860 chunks from WHO and MSF clinical guidelines, processed through a Docling + HybridChunker pipeline with safety keyword detection for life-threatening conditions.

### 4. LoRA Fine-Tuning for Clinical Reasoning Under Resource Constraints

MedGemma 4B-IT was fine-tuned using LoRA (r=64, α=128) on 27,592 quality-filtered clinical conversation traces. The training methodology uses:

- **Input masking**: only model output turns are trained; 64% of context tokens are masked. This prevents the model from learning to repeat patient data and focuses training signal on clinical reasoning and KB tool-use behavior.
- **Hybrid teacher strategy**: a thinking model generates antehoc reasoning traces (the `<think>` blocks), while an instruct model generates the KB tool-use patterns (`<KB_QUERY>` emissions). Both are combined into single training examples.
- **Single-epoch training** with early stopping (patience=5, eval every 200 steps).

### 5. EMR-Agnostic Inference Architecture

The clinical AI engine is deliberately decoupled from any specific EMR. The middleware exposes a REST API that any EMR can call; OpenMRS-dependent endpoints (`/scribe/manifest`, `/scribe/confirm`) return HTTP 501 when `OPENMRS_URL` is not configured. This architectural decision enables deployment across the heterogeneous health information systems found in LMICs, where OpenMRS, DHIS2, and custom systems coexist.

---

## Key Results

### CDS LoRA Fine-Tuning (Stage 1)

| Metric | Value |
|---|---|
| Base model | Google MedGemma 4B-IT |
| LoRA rank / alpha | r=64, α=128 |
| Training samples | 27,592 (quality-filtered) |
| Validation samples | 1,452 |
| Best checkpoint | Step 4,000 |
| Eval loss | 0.4758 |
| Accuracy | 86.25% |
| Context masking | 64% of input tokens masked |

### AudioProjector Training (Stage 2)

| Metric | Value |
|---|---|
| Base model | ClinicDx V1 (LoRA-merged MedGemma) |
| Audio encoder | MedASR Conformer (105M params, frozen) |
| Trainable parameters | 11,806,720 |
| Training clips | 50,000 synthetic clinical audio |
| Epochs | 10 |
| Best checkpoint | Step 40,000 |
| Val LM loss | 0.1042 |
| Key extraction accuracy | 84% |
| Token budget | 64 tokens (fixed) |

### Knowledge Base

| Metric | Value |
|---|---|
| Corpus | WHO + MSF clinical guidelines |
| Chunks | 27,860 |
| Search modes | BM25 (lexical), EmbedGemma 300M (semantic), RRF (hybrid) |
| Reranking | 4-slot clinical intent reranker |
| Content types | 8 (recommendation, treatment_protocol, dosage, background, etc.) |
| Condition vocabulary | 30+ clinical conditions with inclusion/exclusion patterns |
| Index format | memvid v2 (1.1 GB) |

### Deployment

| Metric | Value |
|---|---|
| Total model size | ~4.3 GB (Q8_0 GGUF) |
| Total artifact size | ~6.6 GB (model + encoder + projector + KB + embedder) |
| Quantization | Q8_0 (chosen for behavioral preservation) |
| Minimum VRAM | 8 GB (GPU mode) |
| CPU mode | Supported (no GPU required) |
| Inference runtime | llama.cpp |

---

## Research Questions

1. **Clinical reasoning under extreme model compression**: Can a 4B parameter model, constrained to Q8_0 quantization for edge deployment, maintain reliable clinical reasoning via multi-turn RAG tool-use when fine-tuned on ~28K synthetic traces? What is the minimum model size that preserves structured output integrity (THINK blocks, KB query emissions, citation discipline)?

2. **Audio token budget vs. clinical accuracy**: What is the minimum viable audio token budget for clinically-relevant scribe accuracy? The current 64-token budget represents a ~5s utterance at full resolution. What architectural choices (frame stacking factor, padding strategy, projector depth) govern the accuracy-efficiency tradeoff?

3. **Hybrid retrieval for clinical guideline corpora**: How does BM25 + semantic hybrid retrieval with clinical intent reranking perform on WHO/MSF guideline corpora compared to semantic-only or lexical-only retrieval, specifically for rare/critical condition detection where recall is safety-critical?

---

## Open Problems and Limitations

### Clinical Evaluation

The system has not undergone formal clinical validation. Accuracy metrics (86.25% CDS, 84% Scribe key accuracy) are measured on held-out synthetic data, not on real clinical encounters. A prospective evaluation with practicing clinicians in LMIC facilities is the highest-priority gap.

### Audio Token Norm Mismatch

During AudioProjector training, projected audio token norms were observed to be 360-620x larger than text token norms in the LLM embedding space. The current mitigation uses adaptive norm alignment loss with scheduled decay (`norm_weight` 0.1→0.03 over training). This remains an active area of investigation — the norm gap suggests the projector and LLM embedding spaces are not fully aligned, which may limit scribe accuracy on out-of-distribution audio.

### English Only

All CDS outputs, Scribe extraction, and KB retrieval operate in English. Multilingual support (Swahili, Amharic, Hausa, Yoruba) is on the roadmap but not yet implemented. Linguistic generalization — whether the same 64-token audio budget and projector architecture work for tonal and agglutinative languages — is an open research question.

### No ARM / Low-Power Hardware Benchmarks

The system has been validated on x86_64 with NVIDIA GPUs and CPU-only mode. No benchmarks exist for ARM-based edge hardware (e.g., Orange Pi, Jetson). Latency and throughput characteristics on low-power devices are unknown.

### Quantization Behavioral Preservation

Q8_0 was chosen empirically over Q4 variants because lower quantization degraded structured output behavior (THINK block coherence, KB query emission reliability) more than it degraded perplexity. However, no systematic ablation study has been conducted. Documenting the quantization-behavior relationship in fine-tuned clinical models is an open contribution.

### Knowledge Base Coverage

The KB covers WHO and MSF guidelines relevant to sub-Saharan Africa. Coverage gaps exist for region-specific protocols, national formularies, and conditions not prominently featured in WHO guidelines. The safety keyword detector catches life-threatening conditions but has not been evaluated for recall on edge cases.

---

## Comparison to Prior Work

| System | Parameters | Offline | Audio | EMR Integration | Open Source | Target Setting |
|---|---|---|---|---|---|---|
| **ClinicDx** | 4B | Yes | Yes (MedASR + AudioProjector) | OpenMRS, FHIR R4 | Yes | LMIC edge deployment |
| Med-PaLM 2 (Google) | 340B | No | No | No | No | Research benchmark |
| AMIE (Google) | Undisclosed | No | No | No | No | Diagnostic dialogue research |
| Almanac (UCSF) | 70B+ | No | No | No | Partial | Clinical QA retrieval |
| BioMistral | 7B | Possible | No | No | Yes | Biomedical NLP |
| MedGemma (Google) | 4B | Possible | No | No | Yes | Medical foundation model |

**What ClinicDx addresses that prior work does not:**

- **Offline-first**: Med-PaLM 2, AMIE, and Almanac require cloud inference. ClinicDx runs entirely on local hardware with zero network dependency after initial artifact download.
- **Trimodal in a single binary**: No existing open-source clinical AI system combines text, audio, and vision in a single inference process. The AudioProjector architecture enables audio understanding without a separate ASR pipeline or transcription step.
- **EMR-integrated with clinical terminology grounding**: The CIEL-mapped FHIR R4 pipeline (29 concepts, 7 encounter types) writes structured observations directly into the patient record. Prior clinical AI systems produce text outputs that require manual entry.
- **Edge-deployable model size**: At 4.3B parameters (Q8_0), ClinicDx fits in 8GB VRAM — achievable on consumer GPUs common in LMIC health facilities. Med-PaLM 2 and Almanac require datacenter infrastructure.
- **Multi-turn retrieval-augmented reasoning**: The ReAct loop with KB tool-use gives the model agency over its own evidence retrieval, unlike single-pass RAG systems where retrieval is a fixed preprocessing step.

---

## What Grant Funding Would Enable

1. **Prospective clinical validation study** — Partner with 2-3 facilities in East Africa to evaluate CDS accuracy, Scribe extraction quality, and clinician trust in real clinical encounters.
2. **Multilingual Scribe** — Extend the AudioProjector and training pipeline to Swahili, Amharic, and French, evaluating whether the 64-token budget and projection architecture generalize across language families.
3. **ARM / low-power edge benchmarks** — Profile inference latency, throughput, and memory usage on ARM-based devices (Orange Pi 5 Max, Jetson Orin Nano) to validate deployment on $100-200 hardware.
4. **Quantization ablation study** — Systematic evaluation of Q4/Q5/Q6/Q8 quantization effects on structured output behavior (THINK block integrity, KB query emission, citation discipline) in fine-tuned clinical models.
5. **Expanded KB coverage** — Add Africa CDC protocols, national formularies (Kenya, Ethiopia, Nigeria), and evaluate retrieval quality on region-specific clinical scenarios.
