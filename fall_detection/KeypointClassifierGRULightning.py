import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import lightning as L
from torchmetrics.classification import BinaryAccuracy


class KeypointClassifierGRULightning(L.LightningModule):
    def __init__(self, input_size=34, rnn_hidden_size=128, rnn_layers_count=2, fc_size=128, output_size=1, rnn_dropout=0.3, fc_droupout=0.3, device=None):
        super(KeypointClassifierGRULightning, self).__init__()

        self.gru = nn.GRU(
            input_size, rnn_hidden_size, rnn_layers_count, dropout=rnn_dropout, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, fc_size),
            nn.ReLU(),
            nn.Dropout(fc_droupout),
            nn.Linear(fc_size, output_size)
        )

        self.hparams.input_size = input_size
        self.hparams.rnn_hidden_size = rnn_hidden_size
        self.hparams.rnn_layers_count = rnn_layers_count
        self.hparams.output_size = output_size
        self.hparams.rnn_dropout = rnn_dropout
        self.hparams.fc_droupout = fc_droupout
        self.hparams.fc_size = fc_size
        self.hparams.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy(threshold=0.5)

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.to(device)
        self.eval()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, x):
        _, x = self.gru(x)
        x = x[-1]

        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)

        try:
            loss = self.criterion(output, target)
        except RuntimeError as e:
            print("train:\tCaught RuntimeError during loss computation!")
            print("train:\tTarget unique values:", target.unique())
            print("train:\tOutput unique values:", output.unique())
            print("train:\tTarget shape:", target.shape)
            print("train:\tOutput shape:", output.shape)
            print("train:\tError message:", str(e))
            raise

        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):

        data, target = batch
        output = self(data)
        try:
            loss = self.criterion(output, target)
        except RuntimeError as e:
            print("valid:\tCaught RuntimeError during loss computation!")
            print("valid:\tTarget unique values:", target.unique())
            print("valid:\tOutput unique values:", output.unique())
            print("valid:\tTarget shape:", target.shape)
            print("valid:\tOutput shape:", output.shape)
            print("valid:\tError message:", str(e))
            raise

        self.log('val_loss', loss, on_epoch=True, on_step=False)
        accuracy = self.accuracy(output, target.int()) * 100
        self.log('val_accuracy', accuracy, on_epoch=True, on_step=False)
        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
