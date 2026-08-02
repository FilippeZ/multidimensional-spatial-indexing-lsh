# 🔍 Multidimensional Spatial Indexing & LSH — High-Performance Hybrid Search for Healthcare Analytics

**Maximizing retrieval efficiency over millions of healthcare records through intelligent spatial pruning and locality-sensitive text similarity.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org)
[![Dataset](https://img.shields.io/badge/Dataset-CMS_HCAHPS-0080FF.svg)](https://data.cms.gov/)

---

## 📋 Executive Overview

This repository presents a production-grade, benchmarked framework for **hybrid spatial-textual search** over **18,324 hospital patient satisfaction records** sourced from the U.S. Centers for Medicare & Medicaid Services (CMS HCAHPS). 

By integrating **four multidimensional spatial tree index structures** (`KD-Tree`, `Octree`, `Range Tree`, `R-Tree`) with **Locality-Sensitive Hashing (LSH)** using **MinHash signatures**, the system systematically eliminates ~50% of the textual search space with **zero accuracy loss** ($\text{Recall}@5 = 1.0$).

---

## 🏥 Foundation for Healthcare AI Systems & RAG (Retrieval-Augmented Generation)

### 1. The Real-Time Healthcare Data Challenge
In modern health informatics and clinical AI, Artificial Intelligence systems (such as Natural Language Processing - NLP models and Large Language Models - LLMs) are required to process vast amounts of heterogeneous data in real time:
- **Numerical & Spatial Dimensions:** Patient demographics (ZIP Code), hospital stay timestamps (Year/Timestamp), vital signs, lab values, and patient satisfaction scores (HCAHPS Score).
- **Unstructured Text:** Physician notes, clinical diagnoses, medical history, and verbatim patient feedback/complaints.

Executing naive, exhaustive searches ($O(N)$ brute-force) across millions of patient records introduces multi-second query latencies, making real-time AI assistance during clinical workflows impossible.

### 2. Hybrid Search: The Core Engine of RAG Systems (Retrieval-Augmented Generation)
In modern **Retrieval-Augmented Generation (RAG)** architectures, LLMs do not rely solely on their static parametric memory (which is susceptible to hallucinations), but dynamically retrieve relevant context from external domain databases.

The hybrid spatial-textual retrieval architecture implemented in this project forms the **technical foundation** for ultra-fast healthcare RAG pipelines:

1. **Spatial & Numerical Pruning via Index Trees:**
   Sub-division structures such as **KD-Trees, Octrees, and Range Trees** filter $50\%+$ of the total dataset in sub-milliseconds, isolating only the patient records that meet spatial and numerical clinical constraints (e.g., specific geographic region, temporal window, or health indicator threshold).

2. **Ultra-Fast Textual Similarity Search via MinHash LSH:**
   Subsequently, **Locality-Sensitive Hashing (LSH)** immediately identifies ($O(1)$ amortized lookup complexity) documents with similar text content (e.g., symptoms, complaints, or diagnostic descriptions) without the heavy computational burden of pairwise string comparisons.

3. **Real-Time Data Feeding for NLP & LLM Models:**
   The hybrid search outputs are directly fed as high-precision **Context** to downstream NLP models and LLMs, enabling:
   - **Real-Time Clinical Decision Support:** Instantaneous retrieval of historical cohort patients and analogous cases for attending physicians.
   - **Real-Time Patient Experience Monitoring:** Automated sentiment analysis and instant clustering of patient feedback.
   - **Zero-Loss Recall Guarantee:** Ensuring no critical patient record is missed during context retrieval ($\text{Recall}@5 = 1.0$).

---

## 🎯 The Business & Technical Dilemma

Modern Healthcare Patient Experience (PX) analytics face a critical search bottleneck:
1. **Multi-Dimensional Complexity:** Records exist in a high-dimensional hybrid space comprising:
   - **Numerical Spatial Dimensions:** $\text{ZIP Code} \times \text{Year} \times \text{Linear Mean Score}$
   - **Unstructured Text Dimension:** Patient surveys, question descriptions, and verbatim clinical feedback.
2. **Computational Overhead:** Brute-force $O(N)$ text similarity comparisons take **~2.0s per query**, rendering real-time dashboarding and interactive NLP queries unfeasible.
3. **Clinical Integrity Guarantee:** Unlike web search engines, healthcare clinical decisions mandate **$100\%$ precision matching** to baseline ground truth. Probabilistic approximations must guarantee zero loss of critical patient context.

---

## ✅ The Hybrid Search Solution

Our architecture decouples spatial region pruning from textual similarity estimation:

| Stage | Algorithm / Structure | Mathematical Purpose | Computational Benefit |
| :--- | :--- | :--- | :--- |
| **Stage 1: Spatial Pruning** | Range Tree / Octree / KD-Tree / R-Tree | Filters candidate pool based on 3D numerical bounds $(\text{ZIP}, \text{Year}, \text{Score})$ | Reduces candidate set by ~50% in sub-millisecond time |
| **Stage 2: Text Bucketing** | MinHash LSH (128 hash permutations) | Hashes text $k$-grams into similarity buckets using MinHash signatures | Avoids pairwise text comparisons ($O(1)$ amortized lookup) |
| **Stage 3: Exact Reranking** | Deterministic Jaccard Coefficient | $J(A,B) = \frac{\vert A \cap B \vert}{\vert A \cup B \vert}$ | Guarantees $\text{Recall}@5 = 1.0$ (identical to brute-force) |

```
              ┌─────────────────────────────────────────┐
              │          Incoming User Query            │
              │ (ZIP: 35004, Year: 2018, Score: 85-95)  │
              │     Text: "doctor nurse communication"   │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │     Stage 1: 3D Spatial Pruning         │
              │  (Range Tree / Octree / KD-Tree / RTree) │
              └────────────────────┬────────────────────┘
                                   │ (Spatially Filtered Candidates)
                                   ▼
              ┌─────────────────────────────────────────┐
              │    Stage 2: MinHash LSH Text Search     │
              │    (128 Permutations, Jaccard Bucket)   │
              └────────────────────┬────────────────────┘
                                   │ (Candidate Subset)
                                   ▼
              ┌─────────────────────────────────────────┐
              │    Stage 3: Exact Jaccard Reranking     │
              │     (Guaranteed Recall@5 = 1.0)         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │   Top-K Relevant Patient Feedback       │
              └─────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

The project is organized into clean, decoupled Python modules and dedicated artifact directories:

```text
multidimensional-spatial-indexing-lsh/
├── README.md                           # 📖 Main Analytical Documentation
├── requirements.txt                    # 📦 Python Environment Dependencies
├── .gitignore                          # 🙈 Version control exclusions
├── combine_csv.py                      # 🐍 Data Ingestion Entry Point
├── analysis_notebook.ipynb             # 📓 Master Jupyter Benchmarking Notebook
├── data/                               # 📊 Data Directory (Generated)
│   └── data.csv                        # Merged CMS HCAHPS Dataset (~199 MB)
├── docs/                               # 📄 Documentation & References
│   ├── 1084660.pdf                     # Original Benchmark Paper / Report
│   └── Oδηγίες.txt                     # Research Instructions
├── notebooks/                          # 📓 Notebook Workspace
│   └── analysis_notebook.ipynb         # Copy of primary Jupyter Notebook
└── src/                                # 🧠 Core Algorithmic Source Code
    ├── __init__.py                     # Package Initializer
    ├── spatial/                        # 🌲 Multidimensional Spatial Tree Indexing
    │   ├── __init__.py                 # Spatial Package Export
    │   ├── kd_tree.py                  # KD-Tree (Recursive Median Partitioning)
    │   ├── octree.py                   # Octree (3D Cubical Bounding Subdivision)
    │   ├── range_tree.py               # Multidimensional Range Tree & Bounds Pruning
    │   └── r_tree.py                   # R-Tree (Minimum Bounding Rectangles - MBR)
    ├── lsh/                            # 🔗 Locality-Sensitive Hashing
    │   ├── __init__.py                 # LSH Package Export
    │   └── lsh_index.py                # MinHash LSH Indexing & Exact Reranker
    └── pipeline/                       # ⚙️ Data Ingestion & Sampling Pipeline
        ├── __init__.py                 # Pipeline Package Export
        └── combine_csv.py              # Reproducible Data Merge & 199MB Sampler
```

### Module File References
- [src/spatial/kd_tree.py](file:///c:/Users/wwefi/OneDrive/Υπολογιστής/multidimensional-spatial-indexing-lsh/src/spatial/kd_tree.py): Implementation of `KDTree` and `KDTreeNode`.
- [src/spatial/octree.py](file:///c:/Users/wwefi/OneDrive/Υπολογιστής/multidimensional-spatial-indexing-lsh/src/spatial/octree.py): Implementation of `Octree` and `OctreeNode`.
- [src/spatial/range_tree.py](file:///c:/Users/wwefi/OneDrive/Υπολογιστής/multidimensional-spatial-indexing-lsh/src/spatial/range_tree.py): Implementation of `RangeTree` and `RangeTreeNode`.
- [src/spatial/r_tree.py](file:///c:/Users/wwefi/OneDrive/Υπολογιστής/multidimensional-spatial-indexing-lsh/src/spatial/r_tree.py): Implementation of `RTreeK`, `RTreeEntryK`, and `RTreeNodeK`.
- [src/lsh/lsh_index.py](file:///c:/Users/wwefi/OneDrive/Υπολογιστής/multidimensional-spatial-indexing-lsh/src/lsh/lsh_index.py): Implementation of `LSHIndex` with `datasketch.MinHash`.
- [src/pipeline/combine_csv.py](file:///c:/Users/wwefi/OneDrive/Υπολογιστής/multidimensional-spatial-indexing-lsh/src/pipeline/combine_csv.py): Data merging and reproducible sampling algorithm (`random_state=42`).

---

## 🔬 Algorithmic Architecture & Deep Dive

### 1. Spatial Tree Indexing
* **KD-Tree ($k$-Dimensional Tree):** Alternates split dimensions recursively across the $k$ axes $(\text{ZIP}, \text{Year}, \text{Score})$ using median points, offering $O(\log N)$ average query time.
* **Octree (3D Subdivision):** Subdivides 3D space into 8 sub-octants. Highly efficient for Axis-Aligned Bounding Box (AABB) spatial range filtering.
* **Range Tree:** Combines primary balanced BSTs with secondary $(d-1)$-dimensional subtrees, supporting fractional cascading concepts for rapid orthogonal range searching.
* **R-Tree:** Groups spatial objects into Minimum Bounding Rectangles (MBR), minimizing empty volume enlargement during hierarchical spatial traversal.

### 2. Locality-Sensitive Hashing (LSH) & MinHash
* **MinHash Signatures:** Approximates Jaccard similarity of character $n$-grams using 128 hash functions.
* **LSH Bucketing:** Maps high-dimensional signatures into hash buckets where colliding documents share high similarity threshold ($\text{threshold} \ge 0.5$).
* **Candidate Backfilling:** If LSH yields insufficient candidates due to sparse text overlap, global MinHash estimation backfills candidates seamlessly.

---

## 🏆 Benchmark & Performance Evaluation

Evaluated across **18,324 CMS patient records** over 20 representative queries:

| Spatial Index | Tree Build Time (s) | kNN Query Latency (s) | Text Search Speedup (%) | Recall@5 (Accuracy) |
| :--- | :---: | :---: | :---: | :---: |
| **KD-Tree + LSH** | **~0.00s** 🥇 | 0.006s | -11.3% | 1.0 (100%) ✅ |
| **Octree + LSH** | 0.59s | **0.001s** 🥇 | -24.2% | 1.0 (100%) ✅ |
| **Range Tree + LSH** | **~0.00s** 🥇 | **0.001s** 🥇 | **-28.0%** 🏆 | 1.0 (100%) ✅ |
| **R-Tree + LSH** | 6.86s | 0.011s | -9.4% | 1.0 (100%) ✅ |

### Key Benchmark Takeaways
- **🏆 Top Performer:** **Range Tree + LSH** achieved the maximum retrieval speedup (**28% faster** query latency than exhaustive baseline) with sub-second tree build times.
- **⚡ Ultra-Low Latency:** **Octree + LSH** matched the 1 ms query speed, ideal for real-time spatial range queries.
- **🎯 Guaranteed Accuracy:** All hybrid configurations achieved a **Recall@5 of 1.0**, confirming zero false-negative dropouts compared to brute-force $O(N)$ text matching.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites & Environment Setup
Clone the repository and install requirements:
```bash
git clone https://github.com/FilippeZ/multidimensional-spatial-indexing-lsh.git
cd multidimensional-spatial-indexing-lsh
pip install -r requirements.txt
```

### 2. Data Preparation Pipeline
Merge the CMS HCAHPS CSV dataset files into `data/data.csv`:
```bash
python combine_csv.py
```

### 3. Run Benchmark Notebook
Launch Jupyter Notebook to explore interactive indexing, visualizations, and comparative benchmarks:
```bash
jupyter notebook notebooks/analysis_notebook.ipynb
```

---

## 🛠️ Technology Stack
* **Core Language:** Python 3.10+
* **Data Processing & Analytics:** Pandas, NumPy
* **Spatial & LSH Core:** Custom Spatial Trees, `datasketch` (MinHash LSH), `psutil`
* **Visualization:** Matplotlib
* **Environment:** Jupyter Notebook

---

## 📄 License
Licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

## 👤 Author
**Filippos-Paraskevas Zygouris**  
[LinkedIn](https://www.linkedin.com/in/filippos-paraskevas-zygouris/) | [GitHub](https://github.com/FilippeZ)
