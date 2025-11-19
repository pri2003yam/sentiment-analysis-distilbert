# Sentiment Analysis with DistilBERT

A lightweight sentiment analysis system built using the pretrained **DistilBERT** model fine-tuned on SST-2 dataset.  
Includes real-time prediction through a **Streamlit UI** and an optional fine-tuning pipeline for custom datasets.

---

## 🚀 Features

- **Pre-trained DistilBERT Model**: Uses `distilbert-base-uncased-finetuned-sst-2-english` for sentiment classification
- **Inference Pipeline**: Reusable `SentimentClassifier` class with tokenization and forward pass
- **Streamlit Web UI**: Real-time sentiment analysis with confidence scores
- **Local Model Caching**: Models downloaded and cached locally for offline use
- **GPU Support**: Automatically uses CUDA if available, falls back to CPU
- **Training Pipeline**: Optional fine-tuning script for custom datasets

---

## 📁 Project Structure

```
sentiment-analysis-distilbert/
│
├── app.py                      # Streamlit web application
├── inference.py                # Core inference class (SentimentClassifier)
├── train.py                    # Fine-tuning script for custom data
│
├── data/
│   └── tweets.csv              # Sample dataset (text, label columns)
│
├── models/
│   └── distilbert_model/       # Downloaded pre-trained model
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── fix_cache.py                # Cache management utility
```

---

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/pri2003yam/sentiment-analysis-distilbert.git
   cd sentiment-analysis-distilbert
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

### Run the Streamlit Web App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8502`

**Usage:**
1. Enter any text in the input field
2. Click "Analyze Sentiment"
3. Get sentiment prediction (positive/negative) with confidence score

---

## 💻 Using the Inference Pipeline

```python
from inference import SentimentClassifier

# Load the classifier
classifier = SentimentClassifier()

# Make predictions
result = classifier.predict("I love this movie!")
print(result)
# Output: {'label': 'positive', 'score': 0.9998}
```

---

## 🔧 Fine-Tuning on Custom Data (Optional)

### Dataset Format

Create `data/tweets.csv` with the following structure:

```csv
text,label
this product is amazing,1
terrible experience,0
i love it,1
waste of money,0
```

### Run Fine-Tuning

```bash
python train.py
```

The fine-tuned model will be saved in the `models/` directory.

---

## 📦 Dependencies

- **transformers** - HuggingFace transformers library
- **torch** - PyTorch deep learning framework
- **streamlit** - Web UI framework
- **huggingface_hub** - Model downloading and management

See `requirements.txt` for exact versions.

---

## 🎯 Model Details

- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Task**: Binary sentiment classification (positive/negative)
- **Training Data**: Stanford Sentiment Treebank (SST-2)
- **Input**: Raw text (automatically tokenized)
- **Output**: Sentiment label + confidence score

---

## 💡 How It Works

1. **Tokenization**: Input text is converted to token IDs using DistilBERT tokenizer
2. **Forward Pass**: Tokens are processed through the model
3. **Softmax**: Output logits are converted to probabilities
4. **Classification**: Highest probability label (positive/negative) is selected

---

## 📜 License

MIT License - See LICENSE file for details.

---

## 👤 Author

[pri2003yam](https://github.com/pri2003yam)

---

**Last Updated**: November 19, 2025
