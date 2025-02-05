# PLR-TQA

# **Enhancing Textbook Question Answering with Large Language Models and Retrieval-Augmented Generation**

This repository contains the implementation for the paper *"Enhancing Textbook Question Answering Task with Large Language Models and Retrieval Augmented Generation"* by Hessa A. Alawwad et al. The work focuses on improving the Textbook Question Answering (TQA) task using the Pre-trained LLM, Llama-2, fine-tuned with supervised learning and enhanced with Retrieval-Augmented Generation (RAG) techniques.

## Paper Link

You can find the full paper here:
([PLR-TQA](https://www.sciencedirect.com/science/article/pii/S0031320324010835))
---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
   - [Data Preparation](#data-preparation)
   - [Training](#training)
   - [Evaluation](#evaluation)
6. [Results](#results)
7. [Acknowledgments](#acknowledgments)
8. [License](#license)

---

## **Overview**

Textbook Question Answering (TQA) is a challenging domain that requires models to reason across multimodal data and lengthy contexts. This project utilizes **Llama-2**, a state-of-the-art Large Language Model (LLM), and integrates a **Retrieval-Augmented Generation (RAG)** pipeline to address the “out-of-domain” challenges in TQA datasets. Key contributions include:
- Fine-tuning Llama-2 on domain-specific TQA datasets.
- Implementing RAG to enhance context relevance and model accuracy using **Pinecone** for vector retrieval.
- Optimizing retrieval with embeddings and re-ranking techniques.

---

## **Features**
- Fine-tuned Llama-2 with domain-specific data.
- RAG integration using **Pinecone** for semantic vector-based search.
- Scalable training pipelines with parameter-efficient fine-tuning (PEFT) methods.
- High accuracy improvements over baseline models.

---

## **Requirements**
- Python >= 3.8
- CUDA-enabled GPU (e.g., NVIDIA A100 recommended)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- PyTorch >= 1.10
- [Pinecone](https://www.pinecone.io/) for vector search
- Additional libraries:
  - `datasets`
  - `bitsandbytes`
  - `peft`
  - `transformers`
  - `scipy`

---

## **Setup and Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/username/TQA-Llama2-RAG.git
    cd TQA-Llama2-RAG
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## **Usage**

### **Data Preparation**
Download the CK12-QA dataset and structure it as follows:

data/ ├── train/ ├── validation/ ├── test/


### **Training**
To fine-tune Llama-2:
python fine_tune_llama2_with_rag.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --epochs 2 \
    --batch_size 4 \
    --lr 2e-4

### **Evaluation**
Evaluate the fine-tuned model on the test set:
python evaluate.py \
    --model_dir ./outputs \
    --test_data ./data/test

---

## **Results**

### **Model Accuracy**
- **Validation Accuracy:** 82.40%
- **Test Accuracy:** 84.24%

### **Comparison with Baselines**
This approach outperforms state-of-the-art baselines, with significant improvements in accuracy for non-diagram multiple-choice questions.

---

## **Repository Structure**
├── data/ # Dataset files ├── models/ # Pre-trained and fine-tuned models ├── retrieval/ # RAG-related components ├── outputs/ # Model outputs and logs ├── cleaned_fine_tune_llama2_with_rag.py # Main training script ├── evaluate.py # Optional evaluation script ├── utils.py # Utility functions └── requirements.txt # Python dependencies


---

## **Acknowledgments**
This repository is based on the work of **Hessa A. Alawwad et al.** presented in *"Enhancing Textbook Question Answering with Large Language Models and Retrieval-Augmented Generation"*. Special thanks to the Hugging Face, PyTorch, and Pinecone communities.

---


## Citation

If you use this code or the methodology in your work, please cite our paper as follows:
@article{alawwad2025enhancing, title={Enhancing textual textbook question answering with large language models and retrieval augmented generation}, author={Alawwad, Hessa A and Alhothali, Areej and Naseem, Usman and Alkhathlan, Ali and Jamal, Amani}, journal={Pattern Recognition}, pages={111332}, year={2025}, publisher={Elsevier} }

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

