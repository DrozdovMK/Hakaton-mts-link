from transformers import BertTokenizer, BertModel
import torch

class RuBertEmbedder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        self.model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=50)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings