#!/usr/bin/env python3
"""
Dataset format expected:
- A CSV/JSON file with at least two columns:
  * text_column (default: "summary")
  * label_column (default: "loss_category")
- Labels can be strings; they will be mapped to integers.

For me use case: This has worked
py train_lora_flood_classifier.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --train_file data/train.csv \
  --eval_file data/val.csv \
  --text_column summary \
  --label_column loss_category \
  --output_dir outputs/llama3-flood-lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --load_in_4bit \
  --bf16

Requirements (Pre-execution setup stuff):
pip install -U transformers datasets peft accelerate bitsandbytes scikit-learn pandas
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict

import torch
from torch import nn

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType

try:
    from transformers import BitsAndBytesConfig
    _BITSANDBYTES_AVAILABLE = True
except Exception:
    _BITSANDBYTES_AVAILABLE = False

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune LLM for 4-class flooding loss classification")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model checkpoint")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training file (csv/json/jsonl)")
    parser.add_argument("--eval_file", type=str, required=False, help="Path to eval/val file (csv/json/jsonl)")
    parser.add_argument("--text_column", type=str, default="summary", help="Column containing the text")
    parser.add_argument("--label_column", type=str, default="loss_category", help="Column containing the label")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="", help="Comma-separated module names; leave empty for sensible defaults")

    # QLoRA / quantization
    parser.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit QLoRA via bitsandbytes")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")

    # Training extras
    parser.add_argument("--do_train", action="store_true", help="Run training loop")
    parser.add_argument("--do_eval", action="store_true", help="Run eval loop on eval_file")
    parser.add_argument("--save_weights_only", action="store_true", help="Save only LoRA adapter weights")
    parser.add_argument("--class_weighting", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--label_list", type=str, default="", help="Comma-separated label names to enforce order; else inferred from data")

    return parser.parse_args()


def infer_file_format(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".json") or lower.endswith(".jsonl"):
        return "json"
    raise ValueError(f"Unsupported file format for {path}. Use .csv or .json/.jsonl")


def load_split(file_path: str, text_col: str, label_col: str) -> Dataset:
    fmt = infer_file_format(file_path)
    if fmt == "csv":
        df = pd.read_csv(file_path)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Columns missing. Found columns: {df.columns.tolist()}")
        ds = Dataset.from_pandas(df, preserve_index=False)
    else:
        # json or jsonl
        ds = load_dataset("json", data_files=file_path, split="train")
        if text_col not in ds.column_names or label_col not in ds.column_names:
            raise ValueError(f"Columns missing. Found columns: {ds.column_names}")
    return ds


def get_label_list(train_ds: Dataset, label_col: str, provided: Optional[List[str]]) -> List[str]:
    if provided:
        return provided
    vals = list(set(train_ds[label_col]))
    # stable sort
    vals = sorted(vals, key=lambda x: str(x))
    return vals


def build_label_maps(labels: List[str]) -> (Dict[str, int], Dict[int, str]):
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


def pick_target_modules(model_name: str, override: Optional[List[str]] = None) -> List[str]:
    # Ignore empty string overrides
    if override and any(m.strip() for m in override):
        return [m.strip() for m in override if m.strip()]

    name = model_name.lower()
    # Common sensible defaults for decoder LLMs (LLaMA/Mistral/GPT-NeoX)
    if any(k in name for k in ["llama", "mistral", "yi", "phi", "qwen", "opt", "gpt-neox", "mpt"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # For BERT-like encoders:
    if any(k in name for k in ["bert", "roberta", "electra", "deberta"]):
        return ["query", "key", "value", "output.dense", "intermediate.dense"]
    # For DistilBERT:
    if "distilbert" in name:
        return ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
    # Fallback: try common attn/MLP names
    return ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "fc1", "fc2"]


def make_model_and_tokenizer(args, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]):
    compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[args.bnb_4bit_compute_dtype]

    quant_kwargs = {}
    if args.load_in_4bit:
        if not _BITSANDBYTES_AVAILABLE:
            raise RuntimeError("bitsandbytes not available. Install bitsandbytes to use --load_in_4bit.")
        quant_kwargs["load_in_4bit"] = True
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        device_map = "auto"
    else:
        device_map = None

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        device_map=device_map,
        **quant_kwargs
    )
    return model, tokenizer


def preprocess(tokenizer, text_col: str, max_len: int):
    def _tok(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            padding=False,
            max_length=max_len,
        )
    return _tok


class WeightedTrainer(Trainer):
    """Adds per-class weights for imbalanced datasets by overriding compute_loss."""
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    # you can log confusion matrix later if needed
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": p,
        "recall_macro": r,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Load splits
    train_ds = load_split(args.train_file, args.text_column, args.label_column)
    eval_ds = load_split(args.eval_file, args.text_column, args.label_column) if args.eval_file else None

    # Build label list and maps
    provided_labels = [x.strip() for x in args.label_list.split(",")] if args.label_list else None
    label_list = get_label_list(train_ds, args.label_column, provided_labels)
    label2id, id2label = build_label_maps(label_list)

    # Map labels to ids
    def map_labels(example):
        example["labels"] = label2id[str(example[args.label_column])]
        return example

    train_ds = train_ds.map(map_labels)
    if eval_ds is not None:
        eval_ds = eval_ds.map(map_labels)

    # Tokenizer & model
    model, tokenizer = make_model_and_tokenizer(args, num_labels=len(label_list), id2label=id2label, label2id=label2id)

    # LoRA
    target_modules = pick_target_modules(args.model_name_or_path, override=[m.strip() for m in args.target_modules.split(",")] if args.target_modules else None)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    
    model.print_trainable_parameters()

    # Tokenize
    tok_fn = preprocess(tokenizer, args.text_column, args.max_seq_length)
    cols_to_remove = [c for c in train_ds.column_names if c not in [args.text_column, "labels"]]
    tokenized_train = train_ds.map(tok_fn, batched=True, remove_columns=cols_to_remove)
    tokenized_eval = eval_ds.map(tok_fn, batched=True, remove_columns=cols_to_remove) if eval_ds is not None else None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if (args.bf16 or args.fp16 or args.load_in_4bit) else None)

    # Class weights (optional)
    class_weights = None
    if args.class_weighting:
        # compute inverse frequency weights
        counts = np.zeros(len(label_list), dtype=np.float64)
        for y in train_ds["labels"]:
            counts[y] += 1
        weights = 1.0 / np.clip(counts, 1.0, None)
        weights = weights / weights.sum() * len(label_list)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print("Class weights:", weights, "for labels:", label_list)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if args.do_eval else "no",
        eval_steps=args.eval_steps if args.do_eval else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_pin_memory=False,
        report_to="none",
        load_best_model_at_end=True if args.do_eval else False,
        metric_for_best_model="f1_macro" if args.do_eval else None,
        greater_is_better=True if args.do_eval else None,
    )

    trainer_cls = WeightedTrainer if args.class_weighting else Trainer
    if args.class_weighting:
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=tokenized_train if args.do_train else None,
            eval_dataset=tokenized_eval if (args.do_eval and tokenized_eval is not None) else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn if args.do_eval else None,
            class_weights=class_weights
        )
    else:
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=tokenized_train if args.do_train else None,
            eval_dataset=tokenized_eval if (args.do_eval and tokenized_eval is not None) else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn if args.do_eval else None
        )

    # Save label mapping for inference
    mapping_path = os.path.join(args.output_dir, "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({"label_list": label_list, "label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    # Train/Eval
    if args.do_train:
        trainer.train()
        if args.save_weights_only:
            # Save only adapter weights
            peft_dir = os.path.join(args.output_dir, "adapter")
            os.makedirs(peft_dir, exist_ok=True)
            trainer.model.save_pretrained(peft_dir)
            tokenizer.save_pretrained(args.output_dir)
        else:
            trainer.save_model()  # full model (base+adapters if merged) or adapters if PEFT
            tokenizer.save_pretrained(args.output_dir)

    if args.do_eval and tokenized_eval is not None:
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)

    print("Done. Artifacts saved in:", args.output_dir)


if __name__ == "__main__":
    import transformers
    print("Transformers version:", transformers.__version__)
    print("Transformers path:", transformers.__file__)
    from transformers import TrainingArguments
    print("TrainingArguments class:", TrainingArguments)
    print("TrainingArguments file:", TrainingArguments.__module__)
    main()
