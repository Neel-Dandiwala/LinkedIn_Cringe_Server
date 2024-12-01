from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from textblob import TextBlob
import re
import emoji
from functools import lru_cache

class Extractor:
    def __init__(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.bert_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()

    @torch.no_grad()
    @lru_cache(maxsize=1000)
    def extract_bert_features(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        outputs = self.bert_model(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return pooled.flatten()
    
    # def extract_gpt_features(self, text):
    #     inputs = self.gpt_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
    #     with torch.no_grad():
    #     outputs = self.gpt_model(**inputs)
    #     gpt_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    #     return gpt_features
