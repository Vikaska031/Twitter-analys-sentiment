import re
import sys
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from collections import Counter
from pymystem3 import Mystem
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torchutils as tu
from torchmetrics.classification import BinaryAccuracy
from dataclasses import dataclass
from time import time

stop_words = set(stopwords.words("russian")) - {'не', 'ни'}
mystem = Mystem()

# Регулярное выражение для поиска стандартных эмодзи и текстовых смайликов
emoji_pattern = re.compile(
    u"(["
    u"\U0001F600-\U0001F64F"  # смайлики
    u"\U0001F300-\U0001F5FF"  # символы и пиктограммы
    u"\U0001F680-\U0001F6FF"  # транспортные и маппинг символы
    u"\U0001F1E0-\U0001F1FF"  # флаги
    u"]+)|"
    r"(:\)|:\(|;\)|:\-\)|:\-\(|\(:|\):|:D|:P|:\]|O_O|XD|\^\^|<3|:\*|\(\(|\)\))"  # текстовые смайлики
)

# Регулярное выражение для поиска имен/идентификаторов (буквы и цифры)
username_pattern = re.compile(r'\b[a-zA-Z][a-zA-Z0-9_]{3,}\b')

# Загрузка списка матерных слов
with open('/Users/viktoriasmeleva/Desktop/test_task 2/list.txt', 'r', encoding='utf-8') as f:
    swear_words = set(f.read().splitlines())
# Конфигурация для модели
from dataclasses import dataclass

@dataclass
class ConfigRNN:
    vocab_size: int
    device: str
    n_layers: int
    embedding_dim: int
    hidden_size: int
    seq_len: int
    bidirectional: bool

SEQ_LEN = 110
BATCH_SIZE = 32