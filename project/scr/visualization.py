import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(data):
    """Построение столбчатой диаграммы распределения тональности."""
    data['sentiment'].value_counts().plot.bar(color='pink', figsize=(6, 4))
    plt.title('Distribution of Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

def plot_text_length_distribution(data):
    """Построение гистограммы распределения длины текста для положительных и отрицательных отзывов."""
    positive = data[data['sentiment'] == 'positive']['text'].str.len()
    negative = data[data['sentiment'] == 'negative']['text'].str.len()

    plt.figure(figsize=(6, 4))
    plt.hist(positive, color='pink', alpha=0.5, label='Positive')
    plt.hist(negative, color='blue', alpha=0.5, label='Negative')
    plt.legend()
    plt.xlabel('Length of Text')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Length by Sentiment')
    plt.show()

def plot_text_length_kde(data):
    """Построение KDE-графика распределения длины отзывов в зависимости от тональности."""
    data['len'] = data['text'].str.len()
    positive_reviews = data[data['sentiment'] == 1]
    negative_reviews = data[data['sentiment'] == 0]

    plt.figure(figsize=(10, 6))
    sns.kdeplot(positive_reviews['len'], shade=True, label='Положительные отзывы', color='green')
    sns.kdeplot(negative_reviews['len'], shade=True, label='Негативные отзывы', color='red')
    plt.title('Распределение длины отзывов в зависимости от тональности')
    plt.xlabel('Длина отзыва (символы)')
    plt.ylabel('Плотность')
    plt.legend()
    plt.show()

def plot_emoji_distribution(positive_emojis_count, negative_emojis_count):
    """Построение гистограммы распределения смайликов в положительных и отрицательных твитах."""
    emojis_df = pd.DataFrame({
        'emoji': list(set(positive_emojis_count.keys()).union(set(negative_emojis_count.keys()))),
        'positive': [positive_emojis_count.get(emoji, 0) for emoji in list(set(positive_emojis_count.keys()).union(set(negative_emojis_count.keys())))],
        'negative': [negative_emojis_count.get(emoji, 0) for emoji in list(set(positive_emojis_count.keys()).union(set(negative_emojis_count.keys())))]
    })

    emojis_df['total'] = emojis_df['positive'] + emojis_df['negative']
    emojis_df = emojis_df.sort_values('total', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(emojis_df))

    bars1 = ax.bar(index, emojis_df['positive'], bar_width, label='Positive', color='green')
    bars2 = ax.bar(index + bar_width, emojis_df['negative'], bar_width, label='Negative', color='red')

    ax.set_xlabel('Emoji')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Emojis in Positive and Negative Tweets')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(emojis_df['emoji'])
    ax.legend()

    plt.show()

def plot_swear_words_distribution(data):
    """Построение гистограммы распределения матерных слов в зависимости от тональности твитов."""
    with open('/Users/viktoriasmeleva/Desktop/test_task 2/list.txt', 'r', encoding='utf-8') as f:
        swear_words = f.read().splitlines()

    def contains_swear_words(text):
        words = text.split()
        for word in words:
            if word.lower() in swear_words:
                return True
        return False

    data['contains_swear'] = data['text'].apply(contains_swear_words)
    swear_distribution = data.groupby('sentiment')['contains_swear'].mean().reset_index()

    sns.barplot(x='sentiment', y='contains_swear', data=swear_distribution)
    plt.title('Распределение матерных слов в зависимости от тональности твитов')
    plt.xlabel('Тональность')
    plt.ylabel('Доля твитов с матерными словами')
    plt.show()