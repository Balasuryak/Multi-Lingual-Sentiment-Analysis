

# 🌍 Multi-Lingual Sentiment Analysis

**Fine-tuning LLaMA 3.1-8B-Instruct using LoRA (NPPE-1 Competition)**

## 📌 Project Overview

This repository contains my solution for the **NPPE-1 Multilingual Sentiment Analysis Competition**, where the objective was to fine-tune **LLaMA 3.1-8B-Instruct** for **sentiment classification across 13 Indian languages** under strict compute constraints.

Although LLaMA 3.1 officially supports only Hindi, its tokenizer and pretraining corpus include all target languages. This project demonstrates how **parameter-efficient fine-tuning (LoRA)** enables effective multilingual adaptation using **limited labeled data** and **Kaggle / Colab-level compute**.

---

## 🎯 Problem Statement

* Perform sentiment classification on text written in **13 Indian languages**
* Work with **limited labeled data**
* Use **Kaggle Notebooks / Colab only** (no external GPUs)
* Fine-tune a large language model efficiently without full retraining

---

## 🧠 Key Ideas Explored

* Multilingual adaptation of instruction-tuned LLMs
* Low-resource and data-efficient learning
* Parameter-Efficient Fine-Tuning (PEFT) using **LoRA**
* Prompt-based sentiment classification
* Trade-offs between performance and compute constraints

---

## ⚙️ Model & Training Details

### 🔹 Base Model

* **LLaMA 3.1-8B-Instruct**

### 🔹 Fine-Tuning Method

* **LoRA (Low-Rank Adaptation)**
* Applied to attention layers to reduce trainable parameters
* Keeps memory usage low while enabling task adaptation

### 🔹 Training Environment

* Platform: **Kaggle Notebook / Google Colab**
* Mixed-precision training
* Optimized batch size and gradient accumulation to fit memory limits

### 🔹 Task Formulation

* Sentiment classification framed as an **instruction-following task**
* Unified label space across all languages
* Language-agnostic prompting to encourage cross-lingual generalization

---


## 📊 Evaluation

* Metric: **Classification Accuracy**
* Evaluated across all languages
* Focus on **generalization**, not just high-resource languages

---

## 📁 Repository Contents

```text
├── llama3-1-tuned-for-sentiment-classification.ipynb
│   └── Colab notebook containing preprocessing, LoRA fine-tuning, and evaluation
│
├── multi-lingual-sentiment-analysis.zip
│   └── Data used for training and testing
│
└── README.md
```

---

## 🚀 Key Outcomes

* Successfully adapted LLaMA 3.1 for multilingual sentiment analysis
* Demonstrated effectiveness of **LoRA under compute constraints**
* Achieved robust performance across multiple low-resource languages
* Gained hands-on experience with **scalable LLM fine-tuning pipelines**

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* Kaggle / Google Colab

---


