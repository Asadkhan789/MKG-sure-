# Scope: TVQA+ Triplet Knowledge Base for RAG (MKG-Attention–style)

**Purpose:** Define which TVQA+ splits and files to use, how they map to the shared sample script and the MKG-Attention method (full paper: `MKG_Attention.pdf`), and how ingestion should treat **local** data in this repository.

**Constraint:** RAG / retrieval setup only—**no model training** on TVQA+ train. Use **validation and/or test** splits only for building the knowledge base and evaluation (train JSON may exist on disk for reference but is **out of scope** for corpus construction and scoring).

**References:** [TVQA+ download and description](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa_plus.html)

**Data status:** The archives listed in §3.1 are **downloaded and extracted** under `data/` as described in §3.2.

---

## 1. Objectives

1. **Build a text-first knowledge base** from TVQA+ that supports answering multiple-choice questions via retrieval (and optional LLM reasoning), consistent with the idea that **high-quality triplets** improve downstream accuracy.
2. **Mirror the collaborator’s extraction pattern** (see `test_script.py`): for each question, gather **evidence strings** (dialogue text + visual descriptions), call an LLM with constrained prompts, normalize output to **`(subject, relation, object)`** triplets, deduplicate, and persist append-only results.
3. **Scope evaluation to non-training splits:** validation (always with labels in this repo) and optionally **test** (see §3.4 for label availability).

Non-goals for initial scope: fine-tuning encoders on TVQA+ train, downloading 34GB+ ResNet features or raw frames unless a later milestone explicitly requires them.

---

## 2. Method alignment (paper + sample script)

| Piece | Role in scope |
|--------|----------------|
| **MKG_Attention.pdf** | Defines the **attention / memory over a multimodal knowledge graph** (or similar) behavior you are approximating with **RAG + triplets** instead of end-to-end training. |
| **`test_script.py`** | Reference **implementation contract**: JSONL per line → `question` + `answers` / `supporting_context` with `image` vs `text` → GPT prompts → `normalize_triplet_output` → dedupe via MD5(`question`+`evidence`) → append to output. |

**TVQA+ adaptation:** TVQA+ JSON does not match the script’s schema verbatim. The engineering task is to **project TVQA+ into the same conceptual slots**:

- **Question:** `q` (and optionally `qid`, `vid_name`, `ts` as metadata).
- **Text evidence:** subtitle lines for `vid_name` from `tvqa_plus_subtitles.json`, optionally **windowed to `ts`** to reduce noise and cost.
- **“Caption” / visual text evidence:** per-frame **visual concept sentences** from `det_visual_concepts_hq.pickle` (preferred) and/or **bbox `label` lists** from annotations (lighter but sparser than full sentences).

---

## 3. Dataset assets

### 3.1 Obtained archives (source names on the UNC page)

These four packages are **already** obtained for this project:

| Archive (download name) | Role |
|--------------------------|------|
| `tvqa_plus_annotations.tar.gz` | Raw-style splits: **train** + **val** JSON with `answer_idx`, `ts`, `bbox`, etc. |
| `tvqa_plus_annotations_preproc_with_test.tar.gz` | Preprocessed **train** + **valid** + **test** JSON (extra length fields; test file is special—§3.4). |
| `tvqa_plus_subtitles.tar.gz` | One JSON: `vid_name` → subtitle lines + start times. |
| `det_visual_concepts_hq.pickle.tar.gz` | Pickle dict: `vid_name` → list of **per-frame** visual concept sentences. |

### 3.2 Local layout (this repo)

All paths are relative to the **repository root**.

| Path | Contents |
|------|-----------|
| `data/tvqa_plus_annotations/tvqa_plus_train.json` | Train QAs (**do not use** for RAG corpus / primary eval per project stance). |
| `data/tvqa_plus_annotations/tvqa_plus_val.json` | **Validation** QAs (3,017); keys include `q`, `a0`–`a4`, `answer_idx`, `ts`, `vid_name`, `bbox`. |
| `data/tvqa_plus_annotations_with_test/tvqa_plus_train_preprocessed.json` | Preprocessed train (**out of scope** for same reasons as above). |
| `data/tvqa_plus_annotations_with_test/tvqa_plus_valid_preprocessed.json` | **Validation** QAs (3,017), preprocessed; includes `answer_idx`, `ts`, `bbox`, plus `*_len` fields. |
| `data/tvqa_plus_annotations_with_test/tvqa_plus_test_preprocessed_no_anno.json` | **Test** QAs (2,821); **no** `answer_idx` or `ts` in this checkout—see §3.4. |
| `data/tvqa_plus_subtitles.json` | Subtitles JSON for all clips keyed by `vid_name`. |
| `data/det_visual_concepts_hq.pickle` | Visual concepts pickle keyed by `vid_name`. |

### 3.3 Which annotation JSON to use (pick one convention)

- **Option A — Raw val only:** `data/tvqa_plus_annotations/tvqa_plus_val.json`  
  Minimal fields, matches public docs closely (`tvqa_plus_val.json` naming).

- **Option B — Preprocessed val (+ test when needed):**  
  - Val: `data/tvqa_plus_annotations_with_test/tvqa_plus_valid_preprocessed.json`  
  - Test: `data/tvqa_plus_annotations_with_test/tvqa_plus_test_preprocessed_no_anno.json`  

**Recommendation:** Use **one** line of annotation files per pipeline to avoid duplicate `qid` rows (raw val vs preprocessed valid are the **same count** and should align; mixing both without care causes redundant LLM calls).

### 3.4 Test split caveat (labels and timestamps)

In the extracted **`tvqa_plus_test_preprocessed_no_anno.json`**, rows **do not** include `answer_idx` or `ts` (verified on this tree). That matches the usual “held-out test” story: you can still build **subtitles + visual concepts** evidence per `vid_name` and run a **model or RAG system** to **predict** an answer, but **you cannot compute official accuracy locally** unless you obtain labels elsewhere (e.g. evaluation server / separate release). For **supervised accuracy and ablations in-house**, standardize on **`tvqa_plus_val.json`** or **`tvqa_plus_valid_preprocessed.json`**.

### 3.5 Not obtained (optional later)

| Asset | Approx. | When it would matter |
|--------|-----------|----------------------|
| `tvqa_imagenet_resnet101_pool5_hq.tar.gz` | ~34GB | Learned visual encoders / non-text retrieval—not needed for subtitle + visual-concepts triplet RAG v1. |
| `tvqa_video_frames_fps3_hq.tar.gz` | ~43GB | Raw frames; requires UNC form/agreement. |

---

## 4. Data products (what you will build)

1. **Corpus index (per `vid_name`):** aligned subtitle segments (optionally clipped to `ts` when present) + visual concept strings (frame-ordered; same downsampling story as in the TVQA+ paper / site).
2. **Triplet store:** normalized lines `(s, r, o)` keyed by `(qid or q)`, `vid_name`, evidence type (`subtitle` / `visual_concepts` / `bbox_labels`), and optional `doc_id` you invent for traceability.
3. **Evaluation protocol:** multiple-choice scoring on **val** using `answer_idx`. For **test**, document whether you only generate predictions or use an external evaluator.

---

## 5. Engineering tasks (high level)

1. **Ingestion:** load chosen annotations JSON + `data/tvqa_plus_subtitles.json` + `data/det_visual_concepts_hq.pickle` (versioned Python, pinned deps).
2. **Evidence assembly:** for each QA row, collect text blocks for that `vid_name`; optionally filter subtitles by `ts` when `ts` exists.
3. **Triplet extraction:** adapt `prompt_for_text` / `prompt_for_image` prompts; keep **temperature 0**, strict format, post-parse validation (reuse `normalize_triplet_output` pattern).
4. **Operational:** rate limits, retries, **API keys from environment variables** (do not commit secrets), cost estimates (val-only vs val+test prediction runs).
5. **Retrieval:** embed triplets or chunk text + triplets; retrieve top-k per question; optional LLM answerer with MC options `a0`–`a4`.

---

## 6. Risks and assumptions

- **Pickle compatibility:** loading `data/det_visual_concepts_hq.pickle` may require a specific Python version; pin environment early.
- **Alignment:** visual concepts are **downsampled** like ImageNet features in the TVQA+ documentation; subtitles use **raw** timing—alignment to `ts` and frame indices is **heuristic**, not guaranteed pixel-perfect.
- **Scale / cost:** val has ~3k QAs; each QA can trigger **many** LLM calls if every subtitle line is a separate prompt—**batching, windowing, and deduplication** are in scope.
- **Test JSON:** absence of `answer_idx` / `ts` limits **local** supervised metrics; scope any “test accuracy” claim accordingly.

---

## 7. Success criteria (suggested)

- Reproducible pipeline from **paths in §3.2** → **triplet file(s)** + **indexed corpus** without using train for KB construction.
- Documented **split policy** (raw val vs preprocessed valid) and **evidence sources** per question.
- Ablation note: **subtitles only** vs **subtitles + visual concepts** vs **+ bbox labels** (expected: visual concepts help “visual” questions the most).

---

## 8. Immediate next steps (post-download)

1. **Pick annotation convention** (§3.3) and load **val** only for the first end-to-end prototype.
2. **Verify keys** on a few rows: `vid_name` exists in both subtitles JSON and pickle (log misses).
3. **Prototype ingestion** on ~100 val questions: one subtitle window + one visual-concepts block per clip before scaling LLM calls.
4. **Wire outputs** to the same append-only triplet file pattern as `test_script.py`, then add retrieval + MC scoring on val.

---

*Document updated for local `data/` layout. Re-verify against the [live TVQA+ download page](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa_plus.html) if you re-download archives (checksums, filenames).*
