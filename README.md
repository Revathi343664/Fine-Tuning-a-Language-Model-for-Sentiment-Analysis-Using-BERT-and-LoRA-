BERT Sentiment Analysis with LoRA Fine-Tuning

Project Overview

This project implements **sentiment analysis on the IMDb movie review dataset** using a **fine-tuned BERT model with Low-Rank Adaptation (LoRA)**. The goal is to efficiently adapt a pretrained transformer model for sentiment classification while significantly reducing the number of trainable parameters.

The project demonstrates how **parameter-efficient fine-tuning (PEFT)** techniques can achieve strong performance with limited computational resources and a relatively small training dataset.

The model was trained on a **subset of 2,000 samples** from the IMDb dataset and achieved approximately **87% accuracy**, highlighting the effectiveness of LoRA-based fine-tuning.

---

Key Features

* Fine-tuning **BERT for sentiment classification**
* Implementation of **LoRA (Low-Rank Adaptation)** for parameter-efficient training
* **Hyperparameter tuning using Optuna**
* Training pipeline using **Hugging Face Trainer API**
* **Exploratory Data Analysis (EDA)** of the dataset
* Model evaluation using **accuracy, confusion matrix, and classification report**
* Custom function for **real-time sentiment prediction**

---

Dataset

The project uses the **IMDb Movie Reviews Dataset**, which contains **50,000 labeled reviews** split evenly into positive and negative sentiments.

For faster experimentation:

* **Training set:** 2,000 samples
* **Test set:** 1,000 samples

Dataset fields include:

* `text` – Movie review text
* `label` – Sentiment (positive or negative)

---

Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT (LoRA)
* Optuna
* Seaborn
* Matplotlib
* Torchinfo

---

Project Workflow

1. Exploratory Data Analysis (EDA)

Initial analysis was performed to understand the dataset.

Tasks included:

* Checking sentiment label distribution
* Visualizing class balance using **Seaborn**
* Ensuring balanced samples after dataset reduction

This step ensured unbiased model training.

---

2. Data Preprocessing & Tokenization

The dataset was tokenized using **BertTokenizerFast**.

Key preprocessing steps:

* Text tokenization
* Padding to fixed length
* Truncation at **512 tokens**
* Conversion to **PyTorch tensors**

---

3. Model Architecture

The project uses:

**BERT for Sequence Classification**

with **LoRA integration** to reduce training complexity.

LoRA adds **low-rank matrices to the attention layers**, allowing only a small subset of parameters to be trained.

---

4. Training Setup

Training was performed using the **Hugging Face Trainer API**.

Key configurations:

* Evaluation performed every epoch
* Best model saved automatically
* Accuracy used as the main evaluation metric

---

5. Hyperparameter Optimization

Hyperparameter tuning was conducted using **Optuna**.

Search space included:

| Parameter     | Range       |
| ------------- | ----------- |
| Learning Rate | 1e-5 – 5e-5 |
| Batch Size    | 8, 16       |
| Epochs        | 5, 6        |
| Weight Decay  | 0.0 – 0.3   |



6. Model Evaluation

The model was evaluated using:

* **Accuracy**
* **Classification Report**
* **Confusion Matrix**
* **Training and validation loss tracking**

Final performance achieved approximately:

**Accuracy: ~87%**

---


Model Efficiency

Using **LoRA significantly reduces the number of trainable parameters**, making training faster and more resource-efficient compared to full fine-tuning.

Model summary was generated using **torchinfo**.

---

Project Structure

```
BERT-Sentiment-LoRA
├── Bert Assignment.ipynb
└── README.md
```
 Results

| Metric       | Value                  |
| ------------ | ---------------------- |
| Accuracy     | ~87%                   |
| Dataset Size | 2,000 training samples |
| Model        | BERT + LoRA            |
| Optimization | Optuna                 |

