## 🧠 Conversational Chatbot with Seq2Seq & Attention  
**Deep Learning Project – Heilbronn University (SoSe 2024)**  
**⚠️ Note:** *No pre-trained model weights are included in this repository.*

---

### 📌 Project Overview

This project demonstrates a **custom-built conversational chatbot** using deep learning techniques. The architecture is based on **Sequence-to-Sequence (Seq2Seq)** models combined with **Bahdanau Attention**, implemented in **PyTorch**.

The goal of the project is to generate human-like answers to everyday user queries in a natural dialogue setting. The chatbot is trained using a combination of movie dialogues and the **DailyDialog** dataset, both preprocessed and normalized for efficient training.

All explanations and code comments are provided in **German** to align with the language of the course.

---

### 🧰 Technologies & Tools

- **Python**, **PyTorch**, **NumPy**, **Matplotlib**
- **Huggingface Datasets** for loading DailyDialog
- **TensorBoard** for training metrics
- **Jupyter Notebook Widgets** for interactive input
- **Custom preprocessing pipeline** for text normalization
- **Bahdanau Attention mechanism**
- **No use of pre-trained language models (GPT, BERT)**

---

### 🗂 Dataset & Preprocessing

- **Datasets used:**
  - Movie dialogues (from Convokit, converted to `.pkl`)
  - **DailyDialog** dataset from Huggingface

- **Preprocessing steps:**
  - Normalization (lowercase, punctuation removal, ASCII conversion)
  - Filtering long sentences (max length = 10 tokens)
  - Trimming rare words (min. 3 occurrences)
  - Converting sentences to padded tensor sequences for model input

---

### 🧠 Model Architecture

- **Encoder**: LSTM with embedding and dropout  
- **Decoder**: LSTM with **Bahdanau Attention Layer**  
- **Hidden Size**: 500  
- **Layers**: 2 (both encoder and decoder)  
- **Dropout**: 0.1  

The attention mechanism allows the decoder to focus on relevant parts of the input sequence dynamically during generation.

---

### ⚙️ Training Details

- **Loss Function**: Negative Log Likelihood Loss (NLLLoss)  
- **Optimizer**: Adam  
- **Epochs trained**: 1 (for demonstration only)  
- **Batch size**: 256  
- **Gradient Clipping**: 50.0  
- **Logging**: TensorBoard support with timestamped runs  
- **Checkpoints**: Saved every epoch (only locally)

📌 *Note: Pre-trained model weights are **not** provided due to training done externally.*

---

### 🧪 Evaluation

- Evaluation is done using:
  - **Perplexity score** via Huggingface's GPT-2 metric
  - Manual test questions (e.g., "What is your name?", "How are you?")
  - Visual analysis of **attention matrices**
  - Output formatting to improve readability
- Examples show:
  - Short, direct answers for simple questions
  - High perplexity for more abstract or poorly trained queries

---

### 🖼️ Attention Visualization

The notebook includes visualizations of **attention weights** to demonstrate which input words influence the chatbot's output most at each decoding step.

---

### 💬 Interactive Chat Mode

An interactive widget-based interface allows users to chat directly with the model in a Jupyter Notebook cell using `ipywidgets`. Responses are formatted cleanly with proper grammar and punctuation.

---

### 🚀 Future Improvements

- Expand training epochs and dataset size
- Fine-tune with more domain-specific data
- Replace architecture with Transformer-based model (e.g., BERT, GPT)
- Deploy model in a lightweight web interface
