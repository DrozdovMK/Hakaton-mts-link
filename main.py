from scripts.getting_data import json_getter, func # взятие данных
# from scripts.embedding_model import universal_sentence_encoder # эмбеддинги USE
from scripts.profanity_check import profanity_processing # обработка ненормативной лексики
from scripts.summarizer import gpt_summarizer # обобщение фраз в кластере
a = func(5,7)
print(a)
# 12

bad_sentence = "Блять нахуй сука ничего не работает, заебало все, я еблуша"
prof_proc = profanity_processing(ratio_to_keep=0.6)
print(prof_proc.transform(bad_sentence))
# Б**ть н**уй с**а ничего не работает, за***ло все, я е***ша

responses = [
    "В целом, блять блять блять работа нравится, но есть один большой минус - задержки с зарплатой. Это очень напрягает, потому что приходится планировать бюджет с учетом неопределенности.",
    "Конечно, я стараюсь выполнять свою работу на отлично, но ощущение, что меня блять блять блять недооценивают, когда зарплату задерживают, сильно демотивирует.",
    "Работа интересная, коллектив хороший, но когда задержки с зарплатой, начинаешь задумываться, а стоит ли оно того.",
    "Хотелось бы, чтобы всё было отлично, но задержки с зарплатой - блять блять блять это проблема, которая портит всё впечатление от работы.",
    "Я очень ценю свою работу, но, к сожалению, не могу не отметить постоянные задержки с зарплатой. Это создаёт серьёзные проблемы.",
    "Я бы хотел получать зарплату вовремя, блять блять блять чтобы не приходилось постоянно нервничать и переживать за свои финансы. В остальном, работа устраивает.",
    "Я работаю здесь уже давно, и блять блять блять за это время у меня сложилось впечатление, что задержки с зарплатой - это норма. Это очень неприятно, так как зарплату я жду как манну небесную.",
    "Мне очень нравится моя работа, но задержки с зарплатой - это просто кошмар! Я не понимаю, почему так происходит.",
    "Я бы сказал, что работа интересная, но задержки с зарплаты - это огромный минус, который сильно омрачает всё остальное.",
    "Работаю с удовольствием, но задержка зарплаты - это постоянный источник стресса. Мне кажется, это не уважение к сотрудникам.",
    "В целом, все хорошо, но задержки с зарплатой - это, пожалуй, блять блять блять единственный минус, который хотелось бы исправить."
]

sumgpt = gpt_summarizer()
print(prof_proc.transform(sumgpt.summarize(responses)))