import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.rnn import RNNNet
from models.lstm import LSTMClassifier
from utils import padding

def train_model(model_class, config, X_train, y_train, X_valid, y_valid):
    model = model_class(config)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float))
    valid_data = TensorDataset(torch.tensor(X_valid, dtype=torch.long), torch.tensor(y_valid, dtype=torch.float))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

    for epoch in range(10):  # Количество эпох
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

        # Валидация
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            correct = 0
            total = 0
            for batch_x, batch_y in valid_loader:
                outputs = model(batch_x)
                valid_loss += criterion(outputs.squeeze(), batch_y).item()
                preds = torch.round(torch.sigmoid(outputs.squeeze()))
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            print(f'Epoch {epoch+1}, Validation Loss: {valid_loss/total}, Accuracy: {correct/total}')

    torch.save(model.state_dict(), 'model.pth')
