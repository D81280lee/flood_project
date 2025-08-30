#!/usr/bin/env python3
import argparse, json, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--adapter_subdir", default="adapter")
    ap.add_argument("--text", required=True)
    return ap.parse_args()

def load_map(p):
    with open(p) as f: m=json.load(f)
    return {int(k):v for k,v in m["id2label"].items()}

def main():
    a=parse_args()
    id2label=load_map(os.path.join(a.output_dir,"label_mapping.json"))
    tok=AutoTokenizer.from_pretrained(a.base_model, use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token if tok.eos_token else "[PAD]"
    ad=os.path.join(a.output_dir, a.adapter_subdir)
    if os.path.isdir(ad):
        cfg=AutoConfig.from_pretrained(a.base_model, num_labels=len(id2label), id2label=id2label, label2id={v:k for k,v in id2label.items()})
        mdl=AutoModelForSequenceClassification.from_pretrained(a.base_model, config=cfg, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None), device_map="auto")
        mdl=PeftModel.from_pretrained(mdl, ad)
    else:
        cfg=AutoConfig.from_pretrained(a.output_dir, num_labels=len(id2label), id2label=id2label, label2id={v:k for k,v in id2label.items()})
        mdl=AutoModelForSequenceClassification.from_pretrained(a.output_dir, config=cfg, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None), device_map="auto")
    mdl.eval()
    enc=tok(a.text, return_tensors="pt", truncation=True, max_length=512).to(mdl.device)
    with torch.no_grad():
        logits=mdl(**enc).logits
        probs=torch.softmax(logits, dim=-1).squeeze().tolist()
    pred=int(torch.argmax(logits, dim=-1).item())
    print("Prediction:", id2label[pred])
    print("Probabilities:", {id2label[i]: float(p) for i,p in enumerate(probs)})

if __name__=="__main__": main()
