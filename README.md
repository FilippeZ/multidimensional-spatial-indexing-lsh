# 🔍 Multidimensional Spatial Indexing & LSH — High-Performance Hybrid Search for Healthcare Analytics

**Maximizing retrieval efficiency over millions of healthcare records through intelligent spatial pruning and locality-sensitive text similarity.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org)
[![HCAHPS](https://img.shields.io/badge/Dataset-CMS_HCAHPS-0080FF.svg)](https://data.cms.gov/)

---

## 📋 Executive Overview
This project implements a rigorously benchmarked framework for hybrid spatial-textual search over **18,324** hospital patient satisfaction records (CMS HCAHPS). By operationalizing **five multidimensional spatial data structures** and combining them with **Locality-Sensitive Hashing (LSH)**, we introduce a specialized search pipeline that systematically eliminates ~50% of the search space with **zero accuracy loss** (Recall@5 = 1.0).

## 🎯 The Business Problem
Modern Healthcare Patient Experience (PX) data creates a unique "Search Dilemma":
* **Scale & Latency:** Raw CMS datasets (~200MB) demand rapid, concurrent retrieval across numerical (ratings) and textual (feedback) dimensions.
* **Complex Locality:** Records exist in a multifaceted 3D numerical space `(ZIP Code, Year, Score)` coupled with a high-dimensional unstructured text space.
* **Performance Gap:** Traditional brute-force `O(N)` text comparisons take **~2.0s** per query, a critical bottleneck for interactive analytics requirements.
* **Clinical Accuracy:** Unlike probabilistic web search, healthcare retrieval mandates **identical results** to the exhaustive baseline to strictly maintain statistical integrity.

## ✅ The Solution
This platform transforms heavy computational search workloads into an efficient, "Glass Box" retrieval architecture. We employ a four-layered strategy mapping directly to domain imperatives:

| Feature | Method | Domain Alignment |
| :--- | :--- | :--- |
| **Spatial Pruning** | KD-Tree / Octree | ZIP-code based geographic patient filtering |
| **Temporal Locality** | Range Tree | Precise year-over-year performance tracking |
| **Text Similarity** | MinHash LSH | Rapid grouping of analogous patient feedback |
| **Precise Reranking** | Exact Jaccard | Guaranteed retrieval accuracy (Recall@5 = 1.0) |

---

## 🏗️ Architecture & Workflow
The system strictly adheres to a modular 4-layer architecture, decoupling ingestion logic from core algorithmic search controls:

1. **Ingestion Layer:** CMS HCAHPS data merging, reproducible iterative sampling (random-seed 42), and deterministic feature selection.
2. **Indexing Layer:** Constructing from-scratch geographic spatial structures and scalable MinHash signatures (128 permutations).
3. **Retrieval Layer (Hybrid Control):**
   - *Stage 1:* Spatial kNN/Range queries to rapidly extract geographic neighbors.
   - *Stage 2:* LSH bucket matching on neighbors coupled with deterministic backfill.
4. **Validation Layer:** End-to-end `PerformanceEvaluator` strictly measuring build time, query latency, and accuracy metrics.

## 📂 Project Structure
```text
project-root/
├── analysis_notebook.ipynb       # 📓 Core Framework & Benchmarking
├── combine_csv.py                # 🐍 Data Ingestion & Sampling Pipeline
├── requirements.txt              # 📦 Environment Dependencies
├── Oδηγίες.txt                   # 📄 Original Research Instructions
├── src/                          # 🧠 Algorithm Implementations 
│   ├── spatial/                  # KD-Tree, Octree, Range-Tree, R-Tree
│   └── lsh/                      # MinHash LSH Logic
└── data/                         # 📊 CMS HCAHPS CSVs (Not Tracked)
```

---

## 🚀 Quick Start
### 1. Installation
```bash
git clone https://github.com/FilippeZ/multidimensional-spatial-indexing-lsh.git
cd multidimensional-spatial-indexing-lsh
pip install -r requirements.txt
```

### 2. Operationalize the Pipeline
Prepare the dataset and launch the notebook:
```bash
python combine_csv.py             # Generates data.csv
jupyter notebook analysis_notebook.ipynb
```

---

## ⚖️ Strategic Value & Domain Context
We map technical outputs to healthcare analytics objectives:

#### CMS Compliance & Statistical Integrity
We measure the overlap between the hybrid search and the exhaustive baseline using the **Jaccard Similarity Coefficient**:
$$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$
In this project, $J(A,B) = 1.0$ across all 20 test queries, ensuring that speed optimizations do not introduce bias into clinical satisfaction analysis.

#### Human-Centered Decision Support
Technology assists the analyst; it does not replace them. By providing **28% faster** query responses without losing metadata, this framework allows PX analysts to perform "what-if" scenarios (using Octree range queries) and "similar sentiment" searches (via LSH) in near real-time.

---

## 🔬 Algorithmic Deep Dive

### 🌲 Multi-Dimensional Trees
* **KD-Tree:** Splits space along $k$ dimensions (ZIP, Year, Score) using recursive median-splits for $O(\log n)$ balanced search.
* **Octree:** Recursive 3D grid subdivision. Optimal for AABB (Axis-Aligned Bounding Box) range queries in $O(\log_8 n)$.
* **Range Tree:** Combines a primary BST with associated $(d-1)$-dimensional subtrees. Reached **🏆 Best Query Latency Reduction (-28%)**.

### 🔗 Locality-Sensitive Hashing (LSH)
We perform MinHash-based dimensionality reduction on character 3-grams of patient feedback. This avoids the $O(N)$ pair-wise comparison overhead, enabling amortized $O(1)$ similarity lookups within spatially-pruned buckets.

---

## 🏆 Benchmark Results

### Performance Summary
| Index | Build Time | kNN Latency | Text Speedup | Accuracy (Recall) |
| :--- | :---: | :---: | :---: | :---: |
| **KD-Tree + LSH** | **~0.00s** 🥇 | 0.006s | -11.3% | 1.0 ✅ |
| **Octree + LSH** | 0.59s | **0.001s** 🥇 | -24.2% | 1.0 ✅ |
| **Range Tree + LSH**| **~0.00s** 🥇 | **0.001s** 🥇 | **-28.0%** 🏆 | 1.0 ✅ |
| **R-Tree + LSH** | 6.86s | 0.011s | -9.4% | 1.0 ✅ |

### Memory Footprint
* **Lightweight Leader:** KD-Tree + LSH (**1,247 MB**)
* **High Performance Lead:** Range Tree + LSH (**1,546 MB**)

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Frameworks:** Pandas (Data), NumPy (Numerical), Matplotlib (Viz)
* **XAI/LSH Core:** `datasketch` (MinHash), `psutil` (Monitoring)
* **Environment:** Jupyter Notebook

---

## 📄 License
Licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author
**Filippos-Paraskevas Zygouris**
[LinkedIn](https://www.linkedin.com/in/filippos-paraskevas-zygouris/) | [GitHub](https://github.com/FilippeZ)
