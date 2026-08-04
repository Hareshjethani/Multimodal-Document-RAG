# Visual Document RAG: ColPali + Qwen2-VL

> Search 2,000 real-world document images without a single line of OCR.

A Retrieval-Augmented Generation (RAG) pipeline that retrieves and answers questions over scanned documents (invoices, forms, business letters) by treating each page as an **image**, not text — skipping OCR entirely.

---

##  Overview

Traditional document search pipelines rely on OCR to convert scanned pages into text before retrieval. This works poorly on real-world documents: tables break, handwriting fails, and layout — which often carries the actual meaning — is flattened into disconnected words.

This project takes a different approach: let the model **see** the document the way a human would.

| Component | Role |
|---|---|
| **ColPali** | Retriever — indexes document pages as images using multi-vector patch embeddings (ColBERT-style late interaction, applied to vision) |
| **Qwen2-VL-2B-Instruct** | Generator — reads the retrieved image directly and answers the question, no text extraction step |

**Corpus:** 2,000 document images from the DocVQA dataset (invoices, forms, letters, reports).

---

##  Architecture

```
User Question
     │
     ▼
┌─────────────┐      top-k page      ┌──────────────┐
│   ColPali    │ ───────────────────▶ │  Qwen2-VL     │ ───▶ Answer
│  (Retriever) │      (image)         │ (Generator)   │
└─────────────┘                      └──────────────┘
       ▲
       │
  2,000 indexed
  document images
  (Byaldi library)
```

- **Indexing:** All 2,000 pages are indexed as dense patch-level embeddings via the [Byaldi](https://github.com/AnswerDotAI/byaldi) wrapper around ColPali.
- **Retrieval:** Given a query, ColPali returns the top-k most relevant document images.
- **Generation:** Qwen2-VL takes the retrieved image + question and generates a natural-language answer directly — no OCR, no layout parser.
- **Efficiency:** Qwen2-VL is loaded in 4-bit precision (via `bitsandbytes`) to run on a single Kaggle T4 GPU.

---

##  Evaluation Methodology

To prove retrieval was actually doing meaningful work (and not just adding latency), answers were generated under **three conditions**:

| Condition | Description |
|---|---|
| **Blind** | Qwen2-VL answers with no image at all |
| **RAG** | Qwen2-VL answers using whatever ColPali retrieved |
| **Oracle** | Qwen2-VL answers using the ground-truth image (retrieval bypassed) |

If RAG accuracy tracks Oracle accuracy closely, the retriever is reliably handing the generator the correct page.

---

##  Results

### Retrieval Performance (200 test queries, 2,000-doc corpus)
| Metric | Score |
|---|---|
| Top-1 Accuracy (Exact Match) | **18.50%** |
| Top-5 Accuracy (In Top 5) | **65.50%** |

### 3-Way Retrieval Evaluation (fixed pipeline)
| Metric | Score |
|---|---|
| Recall@3 (Hits@3) | **100%** |
| MRR | **0.7667** |

> **Note:** Early runs showed Recall@3 = 0% and MRR = 0.0 — not because retrieval was broken, but because of an ID-mapping bug in the evaluation code (ground-truth string IDs weren't aligned with the retriever's internal integer IDs). Fixing the mapping — not the model — resolved the issue. See [Key Learnings](#-key-learnings).

---

##  Key Learnings

1. **A "working" retriever can still be silently broken.** Always trace exactly what a metric is comparing before trusting it — the bug was in evaluation code, not the model.
2. **Retrieval bugs hide behind generation quality.** A generator can look "bad" when the real fault lies upstream in what it was given to read. Test retrieval and generation in isolation before judging the combined system.
3. **Visual retrieval genuinely competes with OCR pipelines** on messy, real-world documents — ColPali needed zero extracted text to find the right page.
4. **Small evaluation samples lie convincingly.** Results on 5 queries looked clean but were statistically meaningless — scale up before trusting a number.
5. **A capable generator can't compensate for a broken retriever.** Qwen2-VL performed well only when shown the correct document.

---

##  Tech Stack

- **Retrieval:** ColPali (`vidore/colpali-v1.2`), Byaldi
- **Generation:** Qwen2-VL-2B-Instruct (4-bit, `bitsandbytes`)
- **Core libraries:** `transformers`, `torchao`, `peft`, `accelerate`, `datasets`
- **Interface:** Gradio (interactive demo UI)
- **Dataset:** [DocVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) (`HuggingFaceM4/DocumentVQA`)
- **Hardware:** Single Kaggle T4 GPU

---

##  Getting Started

```bash
pip install -q -U torchao peft byaldi transformers datasets accelerate bitsandbytes gradio
```

1. **Load & prepare the dataset** — downloads DocVQA and saves images for indexing.
2. **Build the index** — batch-indexes all document images with ColPali via Byaldi.
3. **Evaluate retrieval** — run test queries and compute Recall@k / MRR (watch the ID-mapping between ground truth and retriever output).
4. **Run end-to-end RAG** — retrieve top document → pass to Qwen2-VL → generate answer.
5. **(Optional) Launch the Gradio UI** for interactive querying.

---

## 📁 Repo Structure (suggested)

```
.
├── notebooks/
│   ├── 01_indexing.ipynb        # ColPali index building (2,000 docs)
│   ├── 02_retrieval_eval.ipynb  # Recall@k / MRR evaluation
│   └── 03_rag_evaluation.ipynb  # Blind vs RAG vs Oracle comparison
├── assets/
│   └── charts/                  # Accuracy & comparison charts
└── README.md
```

## 🙋 Author

**Haresh Kumar**
