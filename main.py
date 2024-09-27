from scripts.getting_data import json_getter, func # взятие данных
# from scripts.embedding_model import universal_sentence_encoder # эмбеддинги USE
from scripts.profanity_check import profanity_processing # обработка ненормативной лексики

a = func(5,7)
print(a)
# 12

bad_sentence = "Блять нахуй сука ничего не работает, заебало все, я еблуша"
prof_proc = profanity_processing(ratio_to_keep=0.6)
print(prof_proc.transform(bad_sentence))
# Б**ть н**уй с**а ничего не работает, за***ло все, я е***ша