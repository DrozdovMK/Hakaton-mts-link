import tensorflow_hub as hub
import tensorflow_text
class universal_sentence_encoder():
    def __init__(self):
        self.model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual/2")
    def transform(self, texts):
        return self.model(texts)