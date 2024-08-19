from data_preprocessing import data_preprocessing
from train import train_model
from models.rnn import RNNNet
from models.lstm import LSTMClassifier
from config import ConfigRNN, SEQ_LEN, BATCH_SIZE
from utils import padding
import numpy as np
import pandas as pd

def main():
    # Загрузка и предобработка данных
    data = pd.read_csv('data.csv')
    texts = data['text'].tolist()
    labels = data['label'].tolist()

    preprocessed_data = [data_preprocessing(text)[0] for text in texts]
    
    # Преобразование текста в последовательности чисел (например, используя токенизатор)
    # Для примера считаем, что функция tokenization преобразует текст в последовательность индексов
    X = [tokenization(text) for text in preprocessed_data]
    y = np.array(labels)

    # Паддинг последовательностей
    X = padding(X, SEQ_LEN)
    
    # Разделение данных на обучающую и валидационную выборки
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Конфигурация модели
    config = ConfigRNN(
        vocab_size=len(vocab),  # Вам нужно определить vocab
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_layers=2,
        embedding_dim=128,
        hidden_size=64,
        seq_len=SEQ_LEN,
        bidirectional=True
    )

    # Обучение модели RNN
    print("Training RNN model...")
    train_model(RNNNet, config, X_train, y_train, X_valid, y_valid)
    
    # Обучение модели LSTM
    print("Training LSTM model...")
    train_model(LSTMClassifier, config, X_train, y_train, X_valid, y_valid)

if __name__ == '__main__':
    main()
