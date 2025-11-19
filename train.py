import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import torch
from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SAVE_DIR = "models/fine_tuned_model"

def load_data(path="data/tweets.csv"):
    df = pd.read_csv(path)

    # Expecting columns: "text", "label"
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    return df

def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=256
    )

def main():
    print("Loading dataset...")
    df = load_data()

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    train_ds = train_ds.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Training settings
    training_args = TrainingArguments(
        output_dir="models/training_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        push_to_hub=False
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print(f"Saving model to {SAVE_DIR}...")
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Training complete!")

if __name__ == "__main__":
    main()
