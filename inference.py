from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
import torch
import os

class SentimentClassifier:
    def __init__(self):
        # Choose device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        # Download to a simple local directory (not using HF's cache system)
        local_model_dir = "models/distilbert_model"
        
        if not os.path.exists(local_model_dir):
            print("Downloading model files...")
            # Download directly to local directory
            snapshot_download(
                model_name,
                repo_type="model",
                local_dir=local_model_dir,
                local_dir_use_symlinks=False  # Use actual files, not symlinks
            )
            print(f"Model downloaded to: {local_model_dir}")
        else:
            print(f"Using cached model from: {local_model_dir}")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

        print("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            local_model_dir
        ).to(self.device)

        self.model.eval()
        self.labels = ["negative", "positive"]

    def predict(self, text: str):
        if not text or not text.strip():
            return {"label": None, "score": 0.0}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        max_idx = probs.argmax()
        return {
            "label": self.labels[max_idx],
            "score": float(probs[max_idx])
        }
