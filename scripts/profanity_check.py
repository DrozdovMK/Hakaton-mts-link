import sys
import os
from  math import ceil
# Путь к директории, в которой находится этот файл
# Чтобы можно импортировать папки с кодом из директории выше
# например from censure import Censor
current_directory = os.path.dirname(__file__)
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)

from censure import Censor

class profanity_processing():
    def __init__(self, ratio_to_keep = 0.5):
        self.censor_ru = Censor.get(lang = "ru", do_compile=False)
        self.ratoi_to_keep = ratio_to_keep
    def check_for_profanity(self, text):
        line_info = self.censor_ru.clean_line(text)
        _word = line_info[3][0] if line_info[1] else line_info[4][0] if line_info[2] else None
        return not _word is None, _word, line_info
    def replace_profanity(self, text, ratio_to_keep = 0.5):
        check_result = self.check_for_profanity(text)
        if check_result[0]:
            sentense_with_profanity = check_result[2][0]
            stop_words = check_result[2][3]
            for i in range(len(stop_words)):
                length = len(stop_words[i])
                min_chars_to_keep = ceil(length * ratio_to_keep)
                chars_to_replace = length - min_chars_to_keep
                start_replace = (length - chars_to_replace) // 2
                end_replace = start_replace + chars_to_replace
                stop_word_corrected = stop_words[i][:start_replace] + '*' * (end_replace - start_replace) + stop_words[i][end_replace:]
                sentense_with_profanity = sentense_with_profanity.replace("[beep]", stop_word_corrected, 1)
            return sentense_with_profanity
        else:
            return text

if __name__ == "__main__":
    """
    Пример использования класса
    """
    comment = "Сука, как же я блять ненавижу их!"
    profanity_proc = profanity_processing()
    print(profanity_proc.replace_profanity(comment, ratio_to_keep=0.6))
    # Результат: С**а, как же я б**ть ненавижу их!
