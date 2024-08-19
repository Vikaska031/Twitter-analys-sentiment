import torch
import torch.nn as nn
from config import ConfigRNN

class RNNNet(nn.Module):
    def __init__(self, rnn_conf: ConfigRNN) -> None:
        super().__init__()
        self.rnn_conf = rnn_conf
        self.seq_len = rnn_conf.seq_len 
        self.emb_size = rnn_conf.embedding_dim 
        self.hidden_dim = rnn_conf.hidden_size
        self.n_layers = rnn_conf.n_layers
        self.vocab_size = rnn_conf.vocab_size
        self.bidirectional = bool(rnn_conf.bidirectional)

        # Слой эмбеддингов
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        
        # RNN-ячейка
        self.rnn_cell = nn.RNN(
            input_size=self.emb_size, 
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=self.bidirectional,
            num_layers=self.n_layers
        )
        
        # Фактор для двунаправленного RNN
        self.bidirect_factor = 2 if self.bidirectional else 1
        
        # Линейные слои для классификации
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim * self.bidirect_factor, 16),  # Выходная размерность hidden state
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def model_description(self):
        direction = 'bidirect' if self.bidirectional else 'onedirect'
        return f'rnn_{direction}_{self.n_layers}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Пропуск через слой эмбеддингов
        x = self.embedding(x.to(self.rnn_conf.device))
        
        # Пропуск через RNN-ячейку
        output, hidden = self.rnn_cell(x)
        
        # Извлечение последнего скрытого состояния для классификации
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # для двунаправленного RNN объединяем последние слои
        else:
            hidden = hidden[-1,:,:]  # используем только последнее скрытое состояние
            
        # Пропуск через линейный слой
        out = self.linear(hidden)
        
        return out
