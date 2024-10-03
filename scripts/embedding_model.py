import torch
import numpy as np
import warnings
from transformers import logging
logging.set_verbosity_error()

class RuBertEmbedder:
    def __init__(self):
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        self.model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

    def transform(self, texts):
        texts = list(texts)
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=50)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings
    
class universal_sentence_encoder():
    def __init__(self):
        import tensorflow_hub as hub
        import tensorflow_text
        self.model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual/2")
    def transform(self, texts):
        texts = np.array(texts)
        return self.model(texts)