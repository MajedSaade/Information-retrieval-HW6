# 🔍 Robust Information Retrieval (IR) System: OCR Error Handling with Q-Grams

## 🌟 Project Overview

This project implements a small-scale Information Retrieval (IR) system designed to demonstrate the **robustness of character $n$-gram indexing** against common text corruption caused by Optical Character Recognition (OCR) processes. The system successfully retrieves relevant documents even when key search terms are corrupted with errors like spaces or character substitutions.

## ⚙️ Technical Details

* **Core Task:** Information Retrieval (IR) for noisy data.
* **Indexing Method:** Character 3-grams ($q=3$).
* **Ranking Model:** Vector Space Model (VSM) using Cosine Similarity.
* **Language:** Python
* **Key Libraries:** `scikit-learn` (for `CountVectorizer` and `cosine_similarity`) and `numpy`.

## 📂 Data & Corpus

The system uses a corpus of 20 documents on the topic of **"Cancer Cure"**.

* **Query:** A **clean** query (e.g., "genetic research found a cure for cancer").
* **Documents:** Contain simulated OCR errors where key terms are corrupted (e.g., "r esearch" instead of "research") to challenge the system.

## 🚀 Getting Started

### Prerequisites

You need Python installed on your system along with the following libraries:

```bash
pip install numpy scikit-learn