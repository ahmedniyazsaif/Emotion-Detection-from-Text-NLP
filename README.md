# Emotion Detection from Text (NLP Platform)

A customized Natural Language Processing (NLP) platform designed to classify textual emotions into five distinct psychological states: **Anger, Happiness, Love, Sadness, and Worry**. The core classifier features a **Naïve Bayes Classifier built from scratch** using numerical libraries, optimizing raw mathematical text frequencies into scalable, low-latency prediction metrics.

Developed as a core academic project for the **CPCS-331 (Artificial Intelligence)** course at **King Abdulaziz University (KAU)**, Faculty of Computing and Information Technology (FCIT).

---

## 🚀 Key Features

- **Built from Scratch:** Pure algorithmic implementation of the Multinomial Naïve Bayes theorem utilizing **NumPy** for vector operations, eliminating high-level library dependencies during training/inference.
- **Advanced Text Preprocessing Pipeline:** Features lowercasing, dynamic tokenization, custom punctuation stripping, and robust **NLTK-driven stop words extraction** to eliminate noise and increase prediction purity.
- **Dataset Balancing Strategy:** Multi-source dataset consolidation (sourcing from Kaggle and Hugging Face) utilizing targeted truncation and data fusion to mitigate severe structural dataset biases.
- **Additive Smoothing Optimization:** Integrates Laplace/Additive smoothing variants ($1e^{-10}$) within log-space calculations to fully protect inference flows from mathematical exceptions like division-by-zero on unseen vocabularies.

---

## 📐 Mathematical Formulation

To overcome hardware issues associated with underflow from multiplying multiple floating-point probabilities, the system implements text-classification using the log-likelihood sum variant of Bayes' Theorem:

$$P(y \mid x_1, \dots, x_n) \propto \log(P(y)) + \sum_{i=1}^{n} \log P(x_i \mid y)$$

Where $y$ represents target emotional classes and $x_i$ isolated token strings.

---

## 📊 Model Performance & Benchmarks

Following randomized split validation configurations (90/10 split), the mathematical engine achieved a **Global Weighted Accuracy of 89%** across **287,205 test samples**:

| Emotion Class | Precision | Recall | F1-Score | Support Samples |
| :--- | :--- | :--- | :--- | :--- |
| **Anger** | 0.91 | 0.93 | 0.92 | 57,317 |
| **Happiness** | 0.91 | 0.86 | 0.89 | 62,555 |
| **Love** | 0.84 | 0.86 | 0.85 | 54,554 |
| **Sadness** | 0.92 | 0.91 | 0.91 | 60,594 |
| **Worry** | 0.89 | 0.91 | 0.90 | 52,185 |
| **Global Accuracy** | | | **0.89** | **287,205** |

### Top Predictive Features Extracted Per Class:
- **Sadness:** `['bad', 'sad', 'lost', 'left', 'hurt', 'depressed']`
- **Love:** `['loved', 'beloved', 'lovely', 'sweet', 'loving', 'accepted']`
- **Anger:** `['angry', 'frustrated', 'annoyed', 'cold', 'selfish', 'stressed']`
- **Worry:** `['scared', 'anxious', 'afraid', 'nervous', 'worry', 'uncomfortable']`
- **Happiness:** `['happy', 'excited', 'wonderful', 'comfortable', 'confident', 'amazing']`

---

## 📂 Project Structure

```bash
Emotion-Detection-NLP/
│
├── data/
│   └── THE_dataset.csv         # Consolidated dataset matrix
├── src/
│   ├── bayes_naive.py          # Primary Naïve Bayes OOP architecture (From Scratch)
│   ├── helper.py               # Custom text sanitization & NLTK parsing pipeline
│   ├── graph.py                # Matplotlib visualization matrices
│   └── main.py                 # Split execution & testing logic
├── model                       # Serialized trained model snapshot (Pickled binary)
└── README.md                   # Technical documentation
