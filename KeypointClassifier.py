import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses):
    """
    Plots training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Auto-save the plot
    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

class KeypointClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_sizes = [ 256, 128, 64], output_size=1, device=None):
        super(KeypointClassifier, self).__init__()
        # self.cuda()
        self.fc_first = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(len(hidden_sizes)-1):
            self.fcs.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.bns.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            
        self.fc_last=nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.BCELoss()
        self.dropout = nn.Dropout(0.5)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and device!="cpu" else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc_first(x)))
        for fc, bn in zip(self.fcs, self.bns):
            x = fc(x)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.sigmoid(self.fc_last(x))
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def trainn(self, train_data, val_data=None, batch_size=32, epochs=10):
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, pin_memory=True)
        t=0
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}", end="")
            start = time.time()
            train_loss,train_accuracy=self.train_epoch(train_loader)
            train_losses.append(train_loss)
            print(f"\tTraining Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}", end="")
            stop = time.time()
            t+=stop-start
            # print(f"\tTime: {stop-start:.2f}s", end="", flush=True)
            if val_data:
                val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, pin_memory=True)
                start = time.time()
                val_loss, val_accuracy = self.evaluate(val_loader)
                val_losses.append(val_loss)
                stop = time.time()
                print(f"\tValidation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}", end="")
                # print(f"\tTime: {stop-start:.2f}s", end="")
        print(f"\nAverage time: {t/epochs:.2f}s", end="", flush=True)
        plot_losses(train_losses, val_losses)

    def train_epoch(self, train_loader):
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
        # with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
        t = 0
        strt = time.time()
        for data, target in train_loader:
            start = time.time()
            # print('.', end="", flush=True)
            data, target = data.to(self.device), target.to(self.device).float()
            # Forward pass
            stop = time.time()
            t += stop-start  
            self.optimizer.zero_grad()

            output = self(data)
            # print(output.shape, target.shape, data.shape, flush=True)
            loss = self.criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            total_loss += loss.item() * data.size(0)
            predicted = (output >= 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
        stp = time.time()
        print(f"\tinner: {t:.2f}s", end="", flush=True)
        print(f"\ttotal: {stp-strt:.2f}s", end="", flush=True)
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def evaluate(self, test_loader):
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

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device).float()
            output = self(data)
            loss = self.criterion(output, target)

            total_loss += loss.item() * data.size(0)
            predicted = (output >= 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
        # print(prof.key_averages().table(sort_by="cuda_time_total"))

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
