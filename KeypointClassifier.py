import torch
import torch.nn as nn
import torch.optim as optim

class KeypointClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size1=64, hidden_size2=32, output_size=1):
        super(KeypointClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.BCELoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def evaluate(self, X, y):
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y).float().mean()
        return accuracy.item()

    def train_step(self, X, y):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
