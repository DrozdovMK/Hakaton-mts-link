from g4f.client import Client
from g4f.Provider import You
import requests
from pymystem3 import Mystem
import nltk
from nltk.corpus import stopwords
from collections import Counter
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class gpt_summarizer():
    """
    Получает ответ по суммаризации от chatGPT (по API). Если нет сети или ответ не пришел, то лемминг
    и вывод двух самых частых фраз.
    offline = False (по умолчанию) пытается достучаться до chatgpt или нет
    count_offline_words = 2 в случае если не получил ответ от chatgpt, то сколько слов вернуть
    """
    def __init__(self, offline = False, count_offline_words = 2, max_gpt_responses = 8):
        self.client = Client()
        self.lemmatizer = Mystem()
        self.stop_words = []
        self.offline = offline
        self.count_offline_words = count_offline_words
        self.max_gpt_responses = max_gpt_responses
        file_with_stopwords = os.path.join(os.path.dirname(__file__), 'stopwords-ru.txt')
        with open(file_with_stopwords, 'r') as f:
            for line in f:
                word = line.strip()
                self.stop_words.append(word)
    def summarize(self, responses, embeddings):
        if self.is_connected() and not self.offline: # если подключение есть и не хотим оффлайн алгоритм
            if len(responses) > self.max_gpt_responses: #проверяем что кол-во ответов в кластере больше чем установленное значение
                embeddings_center = embeddings.mean(axis=0)
                distances = [(self.cosine_distance(point, embeddings_center), phrase) 
                            for (point, phrase) in zip(embeddings, responses)]
                sorted_points = sorted(distances, key=lambda x: x[0], reverse=False)
                ordered_responses = [point for _, point in sorted_points]
                responses_for_gpt = ordered_responses[:self.max_gpt_responses]
            else:
                responses_for_gpt = responses.copy()
            print(responses_for_gpt)
            answer = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": """Длительность предложения строго не более 4 слов, 
                       Выдели главную мысль списка фраз:{} """
                       .format('; '.join(responses_for_gpt))}],)
            if answer.choices[0].message.content != "": # если ответ пришел
                return answer.choices[0].message.content
            else: # вернуть оффлайн рассчитанные значения
                return self.offline_summarize(responses)
        else: # вернуть оффлайн рассчитанные значения
            return self.offline_summarize(responses)
    def offline_summarize(self, phrases):
        all_tokens = []
        for phrase in phrases:
            all_tokens.extend(self.process_text(phrase))
        
        filtered_tokens = [word for word in all_tokens if word not in self.stop_words]
        frequency = Counter(filtered_tokens)
        most_common = frequency.most_common(self.count_offline_words)
        return "Популярные слова: "+ '; '.join([most_common[i][0] + '({})'.format(most_common[i][1])
                                                for i in range(self.count_offline_words )])
    def process_text(self, text):
        tokens = nltk.tokenize.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha()]
        lemmatized = [self.lemmatizer.lemmatize(token)[0] for token in tokens]
        return lemmatized
    
    def is_connected(self):
        """Проверяет подключение к интернету."""
        try:
            requests.get('https://www.google.com', timeout=5)
            return True
        except requests.exceptions.RequestException:
            return False
    def cosine_distance(self, point, center):
        dot_product = np.dot(point, center)
        norm_point = np.linalg.norm(point)
        norm_center = np.linalg.norm(center)
        if norm_point == 0 or norm_center == 0:
            return 1  # защитный случай, если один из векторов нулевой
        return 1 - (dot_product / (norm_point * norm_center))

if __name__ == "__main__":
    responses = [
    "Встал, сделал зарядку, позавтракал и отправился на работу.",
    "Спал до последнего, еле успел собраться на работу.",
    "Как обычно, зарядка, душ, кофе, и на работу.",
    "Встал, сделал зарядку, позавтракал и, с хорошим настроением, поехал на работу.",
    "Встал, попил кофе и пошел на работу.",
    "Я люблю поспать, поэтому утром я обычно сплю до последнего.",
    "Встал, сделал зарядку и позавтракал овсянкой с фруктами.",
    "Утром я люблю готовить себе завтрак, поэтому провел полчаса на кухне, готовя омлет.",
    "Утром я люблю смотреть новости, поэтому встал пораньше и посмотрел утренний выпуск.",
    "Встал, попил кофе и почитал новости.",
    "С утра я люблю заниматься спортом, поэтому сегодня была пробежка в парке.",
    "Утро – это мое любимое время дня, поэтому я люблю вставать рано и наслаждаться тишиной.",
    "Я люблю готовить, поэтому утром я обычно готовлю себе завтрак.",
    "Утром я обычно занимаюсь йогой, это помогает мне начать день с позитива.",
    "Я люблю спать, поэтому утром я обычно сплю до последнего.",
    "Встал, попил кофе и пошел на работу.",
    "Утром я люблю читать книги, поэтому я провел час за чтением.",
    "Я люблю слушать музыку, поэтому утром я включаю любимый плейлист.",
    "Утром я люблю заниматься спортом, поэтому сегодня я пошел в тренажерный зал.",
    "Утром я люблю гулять на свежем воздухе, поэтому я пошел в парк.",
    "Встал, позавтракал и вышел из дома."
]
    sumgpt = gpt_summarizer(offline = True, count_offline_words=3)
    print(sumgpt.summarize(responses))


    

