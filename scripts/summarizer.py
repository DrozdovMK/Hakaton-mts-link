from g4f.client import Client
from g4f.Provider import You
import requests
from pymystem3 import Mystem
import nltk
from nltk.corpus import stopwords
from collections import Counter
import os


class gpt_summarizer():
    """
    Получает ответ по суммаризации от chatGPT (по API). Если нет сети или ответ не пришел, то лемминг
    и вывод двух самых частых фраз.
    """
    def __init__(self):
        self.client = Client()
        self.lemmatizer = Mystem()
        self.stop_words = []
        file_with_stopwords = os.path.join(os.path.dirname(__file__), 'stopwords-ru.txt')
        with open(file_with_stopwords, 'r') as f:
            for line in f:
                word = line.strip()
                self.stop_words.append(word) 
    def summarize(self, responses):
        if self.is_connected(): # если подключение есть
            answer = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": """Длительность предложения строго не более 4 слов, 
                       Напиши заголовок к списку фраз:{} """
                       .format('; '.join(responses))}],)
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
        most_common = frequency.most_common(2)
        return "Популярные слова: "+ most_common[0][0] +' и '+ most_common[1][0]
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
        

if __name__ == "__main__":
    responses = [
    "В целом, работа нравится, но есть один большой минус - задержки с зарплатой. Это очень напрягает, потому что приходится планировать бюджет с учетом неопределенности.",
    "Конечно, я стараюсь выполнять свою работу на отлично, но ощущение, что меня недооценивают, когда зарплату задерживают, сильно демотивирует.",
    "Работа интересная, коллектив хороший, но когда задержки с зарплатой, начинаешь задумываться, а стоит ли оно того.",
    "Хотелось бы, чтобы всё было отлично, но задержки с зарплатой - это проблема, которая портит всё впечатление от работы.",
    "Я очень ценю свою работу, но, к сожалению, не могу не отметить постоянные задержки с зарплатой. Это создаёт серьёзные проблемы.",
    "Я бы хотел получать зарплату вовремя, чтобы не приходилось постоянно нервничать и переживать за свои финансы. В остальном, работа устраивает.",
    "Я работаю здесь уже давно, и за это время у меня сложилось впечатление, что задержки с зарплатой - это норма. Это очень неприятно, так как зарплату я жду как манну небесную.",
    "Мне очень нравится моя работа, но задержки с зарплатой - это просто кошмар! Я не понимаю, почему так происходит.",
    "Я бы сказал, что работа интересная, но задержки с зарплаты - это огромный минус, который сильно омрачает всё остальное.",
    "Работаю с удовольствием, но задержка зарплаты - это постоянный источник стресса. Мне кажется, это не уважение к сотрудникам.",
    "В целом, все хорошо, но задержки с зарплатой - это, пожалуй, единственный минус, который хотелось бы исправить."
]
    sumgpt = gpt_summarizer()
    print(sumgpt.summarize(responses))


    

