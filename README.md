# Two-Stage RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

**Course:** Large Language Models & Multi-Agent Systems (LMA) — Assignment 1  
**Author:** Naveen Mishra  
**Repository:** [github.com/Mveen3/lma-a1](https://github.com/Mveen3/lma-a1)  

---

## 📌 Overview

This project implements, evaluates, and contrasts two Retrieval-Augmented Generation (RAG) pipelines over the official **Catan (5th Edition) Rules & Almanac** (`data/catan_base_rules_2020.pdf`, 160 chunks):

1. **Pipeline 1 (Baseline Dense Retrieval):**
   - Dense embeddings via `sentence-transformers/all-MiniLM-L6-v2`.
   - Retrieves Top-10 candidates via cosine similarity.
   - Passes Top-3 candidates directly to Google Gemini (`gemini-flash-lite-latest`) using `PROMPT_TEMPLATE`.
2. **Pipeline 2 (Advanced Re-Ranking):**
   - Takes the Top-10 candidates from Stage 1.
   - Jointly scores each `(query, chunk)` pair using `cross-encoder/ms-marco-MiniLM-L-6-v2` via full cross-attention.
   - Passes re-ranked Top-3 candidates to Google Gemini using an **identical prompt template**.

---

## 📁 Repository Structure

```
lma-a1/
├── data/
│   └── catan_base_rules_2020.pdf        # Corpus dataset (16-page official rulebook)
├── docs/
│   └── Problem Statement.txt            # Assignment specification & rubric
├── notebooks/
│   └── RAG_Baseline_vs_Reranking.ipynb  # Primary deliverable: fully executed RAG notebook
├── reports/
│   └── REPORT.md                        # Comprehensive experimental analysis report
├── .env.example                         # Template for API keys
├── .gitignore                           # Git ignore rules
├── README.md                            # Project documentation
└── requirements.txt                     # Minimal Python dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup

Activate your Python/Conda environment (recommended: `ml`):

```bash
conda activate ml
pip install -r requirements.txt
```

### 2. Configure API Key

Copy `.env.example` to `.env` and set your Gemini API key:

```bash
cp .env.example .env
# Edit .env and replace with your active Google GenAI key
```

### 3. Run the Notebook

Open [`notebooks/RAG_Baseline_vs_Reranking.ipynb`](notebooks/RAG_Baseline_vs_Reranking.ipynb) in VS Code or JupyterLab and run all cells sequentially. 

The notebook is configured to dynamically resolve data paths whether executed from the project root or the `notebooks/` directory. All outputs from our latest run are already embedded and preserved.

---

## 📊 Summary of Results

### 1. Failure Modes & Rank Shifts

| Test Case | Category | Baseline Rank | Re-Ranked Rank | Net Promotion | Impact on LLM Output |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CASE-1** | Keyword Trap / Distractor | Rank #6 | **Rank #2** | **+4** | Promoted specific prohibition rule over generic trade text. |
| **CASE-2** | Negation & Constraints | Rank #10 | **Rank #1** | **+9** | Baseline lacked prohibition clause; Re-ranking quoted exact rule. |
| **CASE-3** | Exceptions & Corner Cases | Rank #1 | **Rank #1** | **0** | Kept primary rule at #1; promoted supporting rule chunk (#5 $\to$ #2). |
| **CASE-4** | Document-Specific Promotion | Rank #10 | **Rank #2** | **+8** | Promoted mutual production shortage rule into Top 3. |

### 2. Latency Profiling (30 Benchmark Passes on AMD Ryzen 7 7445HS)

| Pipeline Stage | Mean Latency | Std Dev | Notes |
| :--- | :---: | :---: | :--- |
| **Pipeline 1 (Bi-Encoder Search)** | **$5.47\text{ ms}$** | $\pm 1.09\text{ ms}$ | Sub-millisecond vector dot product after query embedding. |
| **Pipeline 2 (Bi-Encoder + Cross-Encoder)** | **$22.23\text{ ms}$** | $\pm 4.05\text{ ms}$ | Cross-Encoder evaluates 10 candidate pairs ($16.76\text{ ms}$ overhead). |

> 📖 **Full Analysis:** For deep-dive architectural comparisons, mathematical complexity formulas, dataset selection rationale, and enterprise scaling discussions, refer to [`reports/REPORT.md`](reports/REPORT.md).
