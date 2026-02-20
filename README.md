<![CDATA[# 🔍 Multidimensional Spatial Indexing & LSH — High-Performance Hybrid Search for Healthcare Data

*Proving that intelligent spatial pruning eliminates 50% of the search space with zero accuracy loss — achieving perfect Recall@5 across four spatial indexing architectures.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org)
[![datasketch](https://img.shields.io/badge/LSH-datasketch-orange.svg)](https://github.com/ekzhu/datasketch)
[![Dataset](https://img.shields.io/badge/Data-CMS_HCAHPS-0080FF.svg)](https://data.cms.gov/)

---

## 📋 Overview

**Multidimensional Spatial Indexing & LSH** is a comparative study that implements **five spatial data structures from scratch** and combines them with **Locality-Sensitive Hashing (MinHash)** to perform hybrid spatial-textual similarity search on real-world healthcare data. Built to benchmark the trade-offs between speed, memory, and accuracy, this project demonstrates that the **Range Tree + LSH** combination achieves the **best overall performance** with a **28% text query speedup** and **zero retrieval accuracy loss**.

## 🎯 The Problem

Modern healthcare data analytics faces a fundamental tension:

* **Scale:** The CMS HCAHPS dataset contains **18,324** patient satisfaction records across **4,000+ U.S. hospitals**, spanning 5 years (2016–2020).
* **Dual Search Domains:** Each record exists simultaneously in a **3D numerical space** `(ZIP Code, Year, Score)` and a **high-dimensional text space** (patient feedback descriptions).
* **Brute-Force Bottleneck:** Scanning the entire corpus for text similarity takes **~2.0 seconds** per query batch — too slow for interactive exploration.
* **Accuracy Mandate:** Healthcare analytics cannot tolerate degraded retrieval quality — relevant records must not be lost during optimization.

## ✅ The Solution

This platform implements a **two-stage retrieval architecture** that converts brute-force `O(N)` text search into an efficient hybrid pipeline:

| Stage | Method | What It Does | Performance Impact |
| :--- | :--- | :--- | :--- |
| **🌐 Stage 1: Spatial Pruning** | KD-Tree / Octree / Range Tree / R-Tree | Narrows candidates via kNN + range queries in 3D space | Eliminates ~50% of corpus |
| **🔗 Stage 2: LSH Text Ranking** | MinHash (128 perms, char 3-grams) | Approximate Jaccard similarity on pruned set | 10–28% faster than full scan |
| **🎯 Stage 3: Exact Rerank** | Set-based Jaccard ($J(A,B) = \frac{\|A \cap B\|}{\|A \cup B\|}$) | Precise reranking of top candidates | Guarantees Recall@5 = 1.0 |

---

## 🏗️ Architecture & Workflow

The system follows a modular three-layer architecture separating data ingestion, spatial indexing, and text retrieval:

```
Healthcare Patient Satisfaction Data (CMS HCAHPS 2016-2020)
    │
    ├── Data Ingestion Layer
    │   ├── combine_csv.py (Merge 5 annual CSVs → ~199MB)
    │   ├── Numeric Conversion (ZIP, Year, Score → float)
    │   └── Text Concatenation (Question + Answer → feedback_text)
    │
    ├── Spatial Indexing Layer (Stage 1)
    │   ├── KD-Tree      (~0.00s build · 1,247 MB · Lightest)
    │   ├── Octree        (0.59s build · 1,403 MB · Fastest Range)
    │   ├── Range Tree    (~0.00s build · 1,546 MB · Best Speedup)
    │   └── R-Tree        (6.86s build · 1,697 MB · Best Dynamic)
    │
    ├── LSH Text Retrieval Layer (Stage 2 + 3)
    │   ├── Character 3-gram Tokenization
    │   ├── MinHash Signature Generation (128 permutations)
    │   ├── LSH Bucket Query + Backfill
    │   └── Exact Jaccard Reranking
    │
    ├── Evaluation & Benchmarking
    │   ├── PerformanceEvaluator (Build, kNN, Range, Text queries)
    │   ├── Accuracy Verification (Recall@5, Jaccard Overlap)
    │   └── 3D Visualization (Scatter, Boxplots, Histograms)
    │
    └── Results: Recall@5 = 1.0 ✅ | Up to 28% speedup ✅
```

## 📂 Project Structure

```text
multidimensional-spatial-indexing-lsh/
├── analysis_notebook.ipynb       # 📓 Full implementation & evaluation
│   ├── Data Understanding & EDA
│   ├── LSHIndex Class            # MinHash + datasketch wrapper
│   ├── KDTree & KDTreeNode       # 🌲 k-D median-split BST
│   ├── Octree & OctreeNode       # 🧊 3D spatial subdivision
│   ├── RangeTree & RangeTreeNode # 📐 Multi-level associated BST
│   ├── RTreeK, RTreeNodeK        # 📦 MBR-based balanced indexing
│   ├── PerformanceEvaluator      # ⏱️ Benchmarking framework
│   └── Visualizations            # 📊 3D scatter, boxplots, bars
├── combine_csv.py                # 🐍 Data merging & sampling script
├── Oδηγίες.txt                   # 📄 Original instructions (Greek)
├── .gitignore                    # 🚫 Excludes data.csv (~200MB)
└── README.md                     # 📋 This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/FilippeZ/multidimensional-spatial-indexing-lsh.git
cd multidimensional-spatial-indexing-lsh
pip install pandas numpy matplotlib datasketch psutil
```

### 2. Prepare the Dataset

> **Note:** Download the raw CMS HCAHPS CSV files (2016–2020) from [data.cms.gov](https://data.cms.gov/) and place them in the project root.

```bash
python combine_csv.py
```

This merges the annual CSVs into a single `data.csv` (~199MB), using iterative random sampling with `random_state=42` for reproducibility.

### 3. Run the Analysis

```bash
jupyter notebook analysis_notebook.ipynb
```

| Section | What It Does |
| :--- | :--- |
| **Cells 0–8** | Data loading, EDA, distribution plots |
| **Cells 9–25** | From-scratch implementations of all 5 data structures |
| **Cells 26–40** | PerformanceEvaluator: build, query, accuracy benchmarks |
| **Final Cells** | Comparative visualizations and conclusions |

---

## 📊 Dataset: CMS HCAHPS

### What is HCAHPS?

**HCAHPS** (Hospital Consumer Assessment of Healthcare Providers and Systems) is the U.S. national standard for measuring patients' perspectives of hospital care. It evaluates:

* 🏥 Communication with doctors and nurses
* ⏱️ Staff responsiveness
* 🧹 Hospital cleanliness and quietness
* 💊 Pain management and medication communication
* 📋 Discharge information and care transitions
* ⭐ Overall hospital rating and recommendation likelihood

### Data Summary

| Property | Value |
| :--- | :--- |
| **Source** | CMS Hospital Compare (data.cms.gov) |
| **Period** | 2016–2020 (5 annual files) |
| **Hospitals** | 4,000+ across the United States |
| **Final Records** | **18,324** (after filtering & sampling) |
| **Spatial Dimensions** | `ZIP Code` · `Year` · `HCAHPS Linear Mean Value` |
| **Text Dimension** | `HCAHPS Question` + `HCAHPS Answer Description` |

### Key EDA Findings

* **Score Distribution:** Concentrated around **88–92**, with negative skewness confirming generally high satisfaction. Very few low-scoring outliers.
* **Temporal Stability:** Median scores remained remarkably stable (~88–90) throughout the entire 5-year period.
* **Thematic Consistency:** Most frequent HCAHPS questions address core care experiences (communication, cleanliness, responsiveness), providing an excellent foundation for LSH grouping.

---

## 🧩 Implemented Data Structures — Deep Dive

All structures are implemented **from scratch in Python** with full CRUD operations. No external spatial libraries are used.

### 🌲 KD-Tree

Recursive median-split binary search tree with axis cycling. Supports backtracking-based kNN with squared Euclidean distance pruning.

* **Build:** `O(n log n)` — bulk sort + recursive median split
* **kNN:** DFS with close-first traversal and `diff²` pruning of the away branch
* **Range Query:** Recursive descent with per-axis interval checks

### 🧊 Octree

3D spatial subdivision with configurable `max_points_per_node=8`, `min_size=1e-6`, and `max_depth=32`. Uses 3-bit bitmask octant encoding.

* **Build:** Point-by-point insertion with lazy splitting
* **kNN:** Priority-ordered child traversal using `bbox_min_sqdist` lower bounds
* **Range Query:** AABB intersection tests with early subtree inclusion

### 📐 Range Tree

Multi-level balanced BST with **associated (d-1)-dimensional subtrees** at each node. Stores per-subtree bounding boxes for efficient pruning.

* **Build:** `O(n log^(d-1) n)` — sorted construction with nested subtrees
* **Range Query:** Combines `rect_outside` / `rect_inside` checks with cascade into subtrees for `O(log^d n + k)` output
* **kNN:** BST-guided search with bounding-box lower-bound pruning

### 📦 R-Tree (Generalized k-D)

MBR-based balanced indexing with **minimal enlargement** subtree selection and quadratic split seeding (maximum L1 center distance). Includes a convenience `RTree3D` wrapper.

* **Build:** Bottom-up via sequential insertion; root splits increase tree height by 1
* **Split Strategy:** Seed selection maximizes L1 distance between MBR centers; remaining entries assigned by minimal volume enlargement, then by smallest current volume, then by fewest entries
* **kNN:** Best-first priority queue with `_bbox_min_dist_sq` lower bounds

### 🔗 LSH (MinHash + datasketch)

Locality-Sensitive Hashing for approximate text similarity using character 3-gram tokenization.

* **Signature:** 128 MinHash permutations per document
* **Query Pipeline:** Bucket query → global backfill → exact Jaccard rerank
* **Similarity:** Jaccard coefficient $J(A,B) = \frac{|A \cap B|}{|A \cup B|}$

---

## 🏆 Benchmark Results

### Build Performance

| Index | Spatial Build | LSH Build | Total Memory | Memory Rank |
| :--- | :---: | :---: | :---: | :---: |
| **KD-Tree + LSH** | ~0.00s | 1.45s | **1,247 MB** | 🥇 Lightest |
| **Octree + LSH** | 0.59s | 1.62s | 1,403 MB | 🥈 |
| **Range Tree + LSH** | ~0.00s | 1.50s | 1,546 MB | 🥉 |
| **R-Tree + LSH** | 6.86s | 1.50s | 1,697 MB | 4th |

### Query Performance (20 Test Queries)

| Index | kNN Time | Range Time | Text (Full Corpus) | Text (Hybrid) | **Speedup** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KD-Tree + LSH** | 0.006s | 0.076s | 2.025s | 1.796s | **-11.3%** |
| **Octree + LSH** | **0.001s** 🥇 | **0.063s** 🥇 | 2.153s | 1.632s | **-24.2%** |
| **Range Tree + LSH** | **0.001s** 🥇 | 0.094s | 2.079s | **1.496s** 🥇 | **-28.0%** 🏆 |
| **R-Tree + LSH** | 0.011s | 0.081s | 2.002s | 1.813s | **-9.4%** |

### Accuracy Verification

| Metric | KD+LSH | Octree+LSH | Range+LSH | RTree+LSH |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@5** | **1.000** ✅ | **1.000** ✅ | **1.000** ✅ | **1.000** ✅ |
| **Jaccard Overlap** | **1.000** ✅ | **1.000** ✅ | **1.000** ✅ | **1.000** ✅ |

> All four hybrid combinations retrieved the **exact same top-5 results** as the brute-force LSH baseline — **zero accuracy loss**.

---

## 🔬 Key Findings & Conclusions

### 🏅 Primary Finding

The hybrid architecture of **spatial pruning → LSH bucket backfill → exact Jaccard rerank** successfully eliminates ~50% of the search space while maintaining **perfect retrieval accuracy** (Recall@5 = 1.0, Jaccard = 1.0). This validates that spatial locality in healthcare data can be safely exploited for search optimization.

### ⚖️ Trade-off Analysis

| Priority | Recommended Index | Rationale |
| :--- | :--- | :--- |
| **💾 Memory Efficiency** | KD-Tree + LSH | Lightest at 1,247 MB; simplest implementation |
| **⚡ Best Text Speedup** | Range Tree + LSH | 28% reduction; fastest kNN at 0.001s |
| **🧊 Fastest 3D Range** | Octree + LSH | 0.063s range queries; cubic cell pruning |
| **🔄 Dynamic Updates** | R-Tree + LSH | MBR-based; handles insertions/deletions gracefully |

### 📌 Technical Insights

1. **LSH dominates memory:** The MinHash engine (18,324 docs × 128 permutations) accounts for the majority of memory across all configurations.
2. **Spatial pruning is safe:** Restricting candidates to ~50% of the corpus does not degrade retrieval quality when combined with LSH bucket backfill.
3. **Exact rerank is cheap:** Set-based Jaccard on the top pool adds negligible latency while guaranteeing result correctness.
4. **Character 3-grams > word tokens:** Char n-gram tokenization captures sub-word patterns, improving LSH sensitivity for healthcare terminology.
5. **R-Trees trade speed for flexibility:** Higher build cost (6.86s vs ~0.00s) and memory (1,697 MB) are justified only when dynamic MBR updates are required.

---

## 🛠️ Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core implementation |
| **Data Processing** | Pandas, NumPy | DataFrame manipulation, numeric operations |
| **LSH Engine** | datasketch | MinHash signatures & LSH bucket indexing |
| **Visualization** | Matplotlib | 3D scatter plots, histograms, boxplots |
| **System Monitoring** | psutil | Memory usage tracking during benchmarks |
| **Environment** | Jupyter Notebook | Interactive analysis & reproducibility |
| **Text Processing** | re (regex) | Tokenization, character n-gram generation |
| **Type Safety** | typing module | Type hints for code clarity |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

## 👤 Author

**Filippos-Paraskevas Zygouris**
[GitHub](https://github.com/FilippeZ) | [LinkedIn](https://www.linkedin.com/in/filippos-paraskevas-zygouris/)
]]>
