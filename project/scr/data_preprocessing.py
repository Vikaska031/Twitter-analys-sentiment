import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from config import emoji_pattern, username_pattern, mystem, stop_words, swear_words

def data_preprocessing(text: str, positive_emoji_placeholder: str = 'отлично', negative_emoji_placeholder: str = 'плохо') -> tuple:
    """Предобработка строки: приведение к нижнему регистру, удаление html-тегов, пунктуации, стоп-слов и замена эмодзи.
       Также вычисляет количество нецензурных слов.

    Аргументы:
        text (str): Входная строка для предобработки
        positive_emoji_placeholder (str): Заполнитель для положительных эмодзи
        negative_emoji_placeholder (str): Заполнитель для отрицательных эмодзи

    Возвращает:
        tuple: Предобработанная строка и количество нецензурных слов
    """

    def replace_emoji(match):
        emoji = match.group()
        if emoji in positive_emojis:
            return f" {positive_emoji_placeholder} "
        elif emoji in negative_emojis:
            return f" {negative_emoji_placeholder} "
        return emoji
    
    # Замена эмодзи
    text = emoji_pattern.sub(replace_emoji, text)

    # Приведение текста к нижнему регистру
    text = text.lower()

    # Удаление HTML-тегов
    text = re.sub(r"<.*?>", "", text)
    
    # Замена сокращений
    text = re.sub(r"\bне\b", "не", text)
    
    # Удаление знаков препинания (оставляем пробелы)
    text = "".join([c if c not in string.punctuation else ' ' for c in text])
    
    # Удаление английских слов
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)

    # Удаление имен/идентификаторов
    text = username_pattern.sub('', text)
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    # Лемматизация
    text = "".join(mystem.lemmatize(text)).strip()
    
    # Удаление стоп-слов
    text = " ".join(word for word in text.split() if word not in stop_words)

    # Подсчет матерных слов
    swear_count = sum(1 for word in text.split() if word in swear_words)
    
    return text, swear_count

# Функция для создания словаря частотных слов
def get_words_by_freq(sorted_words: list, n: int = 10) -> list:
    return list(filter(lambda x: x[1] > n, sorted_words))

# Применение предобработки к DataFrame
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data['len'] = data['text'].str.len()
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    data['cleaned_reviews'], data['contains_swear'] = zip(*data['text'].apply(data_preprocessing))
    
    corpus = [word for text in data['cleaned_reviews'] for word in text.split()]
    count_words = Counter(corpus)
    sorted_words = count_words.most_common()
    
    sorted_words = get_words_by_freq(sorted_words, 10)
    vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}

    data = data.drop(columns=['text'])
    
    # Сохранение словаря в JSON
    with open('vocab_to_int.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_to_int, f, ensure_ascii=False)
    
    return data, vocab_to_int
