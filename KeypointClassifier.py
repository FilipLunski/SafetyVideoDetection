import torch
import torch.nn as nn
import torch.optim as optim


class KeypointClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size1=128, hidden_size2=256, hidden_size3=64, output_size=1):
        super(KeypointClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.BCELoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def trainn(self, train_data, val_data=None, batch_size=32, epochs=10, device='cpu'):
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}", end="")
            self.train_epoch(train_loader, device)
            
            if val_data:
                val_loader = torch.utils.data.DataLoader(val_data)
                val_loss, val_accuracy = self.evaluate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        

    def train_epoch(self, train_loader, device='cpu'):
        """
        Train the model for one epoch
        Args:
            train_loader (DataLoader): Training data loader
            device (str): Compute device ('cpu' or 'cuda')
        Returns:
            float: Average training loss for the epoch
        """
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # print('.', end="")
            data, target = data.to(device), target.to(device).float()
            # Forward pass
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            total_loss += loss.item() * data.size(0)
            predicted = (output >= 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def evaluate(self, test_loader, device='cpu'):
        """
        Evaluate model performance
        Args:
            test_loader (DataLoader): Test/validation data loader
            device (str): Compute device ('cpu' or 'cuda')
        Returns:
            tuple: (loss, accuracy)
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).float()
                output = self(data)
                loss = self.criterion(output, target)

                total_loss += loss.item() * data.size(0)
                predicted = (output >= 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    # def evaluate(self, X, y):
    #     self.eval()
    #     with torch.no_grad():
    #         outputs = self(X)
    #         predicted = (outputs > 0.5).float()
    #         accuracy = (predicted == y).float().mean()
    #     return accuracy.item()

    # def train_step(self, X, y):
    #     self.train()
    #     self.optimizer.zero_grad()
    #     outputs = self(X)
    #     loss = self.criterion(outputs, y)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.item()
    # def evaluate(self, data_loader):
    #     self.eval()  # Set the model to evaluation mode
    #     total_loss = 0
    #     correct = 0
    #     total = 0

    #     with torch.no_grad():
    #         for inputs, labels in data_loader:
    #             outputs = self(inputs)
    #             loss = self.criterion(outputs, labels)
    #             total_loss += loss.item()

    #             predicted = (outputs > 0.5).float()
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #     accuracy = correct / total
    #     average_loss = total_loss / len(data_loader)

    #     return average_loss, accuracy

    # def trainn(self, train_data):
    #     val_data=None
    #     epochs=10
    #     batch_size=32

    #     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #     if val_data:
    #         val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    #     for epoch in range(epochs):
    #         self.train()  # Set the model to training mode
    #         total_loss = 0

    #         for inputs, labels in train_loader:

    #             self.optimizer.zero_grad()
    #             outputs = self(inputs[1].view(-1))
    #             print(outputs, labels[1])
    #             loss = self.criterion(outputs, labels[1])
    #             loss.backward()
    #             self.optimizer.step()

    #             total_loss += loss.item()

    #         average_loss = total_loss / len(train_loader)
    #         print(f"Epoch {epoch+1}/{epochs}, Training Loss: {average_loss:.4f}")

    #         if val_data:
    #             val_loss, val_accuracy = self.evaluate(val_loader)
    #             print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    #     return
