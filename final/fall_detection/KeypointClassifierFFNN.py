import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import lightning as L
from torchmetrics.classification import BinaryAccuracy


class KeypointClassifierFFNN(L.LightningModule):
    def __init__(self, layers, activation, dropout=0.3, batch_norm=True, device=None):
        super(KeypointClassifierFFNN, self).__init__()

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.to(device)

        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "prelu": nn.PReLU(device=self.device),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "mish": nn.Mish(),
        }

        self.classifier = nn.Sequential()
        for i in range(len(layers)-1):
            self.classifier.add_module(
                f'layer_{i}', nn.Linear(layers[i], layers[i+1]))
            if batch_norm:
                self.classifier.add_module(
                    f'batch_norm_{i}', nn.BatchNorm1d(layers[i+1]))
            self.classifier.add_module(
                f'activation_{i}', activations[activation])
            if dropout > 0:
                self.classifier.add_module(f'dropout_{i}', nn.Dropout(dropout))
        self.classifier.add_module(
            f'layer_{len(layers)-1}', nn.Linear(layers[-1], 1))

        self.hparams.input_size = layers[0]
        self.hparams.output_size = 1
        self.hparams.dropout = dropout
        self.hparams.device = device
        self.hparams.activation = activation
        self.hparams.layers = layers

        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.layers = layers

        self.sigmoid = nn.Sigmoid()
        self.eval()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, x):
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)

        loss = self.criterion(output, target.float())

        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):

        data, target = batch
        output = self(data)

        loss = self.criterion(output, target.float())

        self.log('val_loss', loss, on_epoch=True, on_step=False)
        output = self.sigmoid(output)
        accuracy = self.accuracy(output, target.int()) * 100
        self.log('val_accuracy', accuracy, on_epoch=True, on_step=False)
        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

        
    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        loss = self.criterion(output, target.float())
        
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        output = self.sigmoid(output)
        accuracy = self.accuracy(output, target.int()) * 100
        self.log('test_accuracy', accuracy, on_epoch=True, on_step=False)

        
    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.sigmoid(self(batch))
