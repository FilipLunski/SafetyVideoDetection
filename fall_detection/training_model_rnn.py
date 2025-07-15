from pathlib import Path
import h5py
import torch
from KeypointClassifierGRU import KeypointClassifierGRU
from KeypointClassifierLSTM import KeypointClassifierLSTM
from torch.utils.data import TensorDataset, ChainDataset
from torch.nn.utils.rnn import pack_sequence
import json
import lightning as L
import os
from lightning.pytorch.loggers import TensorBoardLogger
import time

CHECKPOINTS_FILE = "fall_detection/checkpoints.json"


class VariableLengthDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)

    # Each seq is already (seq_len, 34), just convert to tensor
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]

    # Pack them directly
    packed_sequences = pack_sequence(sequences, enforce_sorted=False)

    labels = torch.tensor(labels)

    return packed_sequences, labels


def load_dataset(paths, batch_size, timesteps=None):

    data = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            for video in f:
                frames = f[video]['dataset']['keypoints'][()]
                for j in range(1, 2):
                    for i in range(len(frames)):
                        start = 0
                        if timesteps is not None and i > timesteps * j:
                            start = i - timesteps * j + 1

                        k = frames[start: i + 1: j]
                        l = [float(f[video]['dataset']['categories'][i])]
                        data.append((k, l))
                    # print(f[video]['dataset']['keypoints'][()].shape)

    dataset = VariableLengthDataset(data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)
    return loader


def load_checkpoint_map():
    if os.path.exists(CHECKPOINTS_FILE):
        with open(CHECKPOINTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint_path(model_version, checkpoint_path):
    checkpoint_map = load_checkpoint_map()
    checkpoint_map[model_version] = checkpoint_path
    with open(CHECKPOINTS_FILE, 'w') as f:
        json.dump(checkpoint_map, f, indent=4)


def get_checkpoint_path(model_version):
    return load_checkpoint_map().get(model_version, None)


default_train_dataset_paths = [
    r'samples\dataset_cauca_m_train.h5',
    r'samples\dataset_fifty_ways_m_train.h5',
    r'samples\dataset_mcfd_x_train.h5',
    r'samples\dataset_le2i_x_train.h5'
]

default_dev_dataset_paths = [
    r'samples\dataset_cauca_m_validation.h5',
    r'samples\dataset_fifty_ways_m_validation.h5',
    r'samples\dataset_mcfd_x_val.h5',
    r'samples\dataset_le2i_x_val.h5'
]

default_test_dataset_paths = [

    r'samples\dataset_cauca_m_test.h5',
    r'samples\dataset_fifty_ways_m_test.h5',
    r'samples\dataset_mcfd_x_test.h5',
    r'samples\dataset_le2i_x_test.h5'
]


def train(train_dataset_paths=default_train_dataset_paths, dev_dataset_paths=default_dev_dataset_paths, rnn_type="gru", model_path=None, epochs=400, rnn_layers=2, rnn_hidden_size=128, fc_size=128, rnn_dropout=0.4, fc_droupout=0.4,
          save=True, device='cuda', from_checkpoint=True, batch_size=4096, timesteps=None, checkpoint_path=None, test=False, test_dataset_paths=default_test_dataset_paths):

    try:
        # _{batch_size}"
        name = f"{rnn_type}_{timesteps}_{rnn_layers}_{rnn_hidden_size}_{fc_size}_{rnn_dropout}_{fc_droupout}"

        print(
            f"----------------------------------Training {name} model ---------------------------------------")

        if not from_checkpoint:
            checkpoint_path = None
        elif checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(name)

        model = {
            "gru": KeypointClassifierGRU(device=device, rnn_hidden_size=rnn_hidden_size,
                                         rnn_layers_count=rnn_layers, rnn_dropout=rnn_dropout, fc_droupout=fc_droupout),
            "lstm": KeypointClassifierLSTM(device=device, rnn_hidden_size=rnn_hidden_size,
                                           rnn_layers_count=rnn_layers, rnn_dropout=rnn_dropout, fc_droupout=fc_droupout)
        }.get(rnn_type, None)

        if model is None:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        logger = TensorBoardLogger(
            f"logs_{rnn_type}", name=name)
        trainer = L.Trainer(max_epochs=epochs, logger=logger)

        model.hparams.model_path = model_path
        model.hparams.previous_model_path = checkpoint_path
        model.hparams.train_dataset_paths = train_dataset_paths
        model.hparams.dev_dataset_paths = dev_dataset_paths
        model.hparams.save = save
        model.hparams.from_checkpoint = from_checkpoint
        model.hparams.timesteps = timesteps
        model.hparams.batch_size = batch_size
        model.hparams.epochs = epochs
        model.save_hyperparameters()

        if not test:

            model.train()

            train_loader = load_dataset(
                train_dataset_paths, batch_size, timesteps)
            val_loader = load_dataset(dev_dataset_paths, batch_size, timesteps) if len(
                dev_dataset_paths) > 0 else None

            trainer.fit(model=model, train_dataloaders=train_loader,
                        val_dataloaders=val_loader, ckpt_path=checkpoint_path)

            if save:
                if model_path is None:
                    model_path = f"models_fall/model_{name}.pt"
                model.save(model_path)

            save_checkpoint_path(
                name, trainer.checkpoint_callback.best_model_path)
        else:
            if checkpoint_path is None:
                raise ValueError(
                    "Checkpoint path must be provided for testing")
            if test_dataset_paths is None:
                raise ValueError(
                    "Test dataset paths must be provided for testing")
            test_loader = load_dataset(
                test_dataset_paths, batch_size, timesteps)

            model.eval()
            trainer.test(model=model, dataloaders=test_loader,
                         ckpt_path=checkpoint_path)

            if rnn_type == "gru":
                model = KeypointClassifierGRU.load_from_checkpoint(
                    checkpoint_path)
            elif rnn_type == "lstm":
                model = KeypointClassifierLSTM.load_from_checkpoint(
                    checkpoint_path)
            else:
                raise ValueError(f"Unknown RNN type: {rnn_type}")

            model.eval()
            model.to(device)
            time_sum = 0
            sample_count = 0
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                data = inputs.data
                for i in range(data.size(0)):
                    sample = data[i].unsqueeze(0)
                    start_time = time.perf_counter()
                    output = model.predict_step(sample, i)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    time_sum += elapsed_time
                    sample_count += 1
            avg_time = time_sum / sample_count   # Convert to seconds
            print(f"Average time per sample: {avg_time:.6f} seconds")

    except Exception as e:
        print(f"An error occurred during training: {e}")


def main(rnn_type="gru"):

    # train(rnn_type=rnn_type, epochs=350, rnn_layers=1, rnn_hidden_size=64, fc_size=64,
    #       timesteps=timesteps, from_checkpoint=False, rnn_dropout=0.15, test=False)
    # train(rnn_type=rnn_type, epochs=400, rnn_layers=1, rnn_hidden_size=64, fc_size=64,
    #       timesteps=timesteps, from_checkpoint=True, rnn_dropout=0.15, test=False)
    # train(rnn_type=rnn_type, epochs=450, rnn_layers=1, rnn_hidden_size=64, fc_size=64,
    #       timesteps=timesteps, from_checkpoint=True, rnn_dropout=0.15, test=False)
    # train(rnn_type=rnn_type, epochs=500, rnn_layers=1, rnn_hidden_size=64, fc_size=64,
    #       timesteps=timesteps, from_checkpoint=True, rnn_dropout=0.15, test=False)
    # train(rnn_type=rnn_type, epochs=550, rnn_layers=1, rnn_hidden_size=64, fc_size=64,
    #       timesteps=timesteps, from_checkpoint=True, rnn_dropout=0.15, test=False)
    # train(rnn_type=rnn_type, epochs=600, rnn_layers=1, rnn_hidden_size=64, fc_size=64,
    #       timesteps=timesteps, from_checkpoint=True, rnn_dropout=0.15, test=False)
    # train(rnn_type=rnn_type, epochs=650, rnn_layers=1, rnn_hidden_size=64, fc_size=64,
    #       timesteps=timesteps, from_checkpoint=True, rnn_dropout=0.15, test=False)
    # train(rnn_type=rnn_type, epochs=700, rnn_layers=1, rnn_hidden_size=64, fc_size=64,
    #       timesteps=timesteps, from_checkpoint=True, rnn_dropout=0.15, test=False)

    # train(rnn_type=rnn_type, epochs=420, rnn_layers=1, rnn_hidden_size=64, fc_size=64, batch_size=6144,
    #       timesteps=timesteps, from_checkpoint=False, rnn_dropout=0.15, test=False, device="gpu", checkpoint_path="fall_detection/GRU.ckpt")

    # train(rnn_type=rnn_type, epochs=420, rnn_layers=3, rnn_hidden_size=64, fc_size=64, batch_size=6144,
    #       timesteps=timesteps, from_checkpoint=False, rnn_dropout=0.15, test=False, device="gpu", checkpoint_path="fall_detection/GRU.ckpt")

    train(rnn_type=rnn_type, epochs=550, rnn_layers=1, rnn_hidden_size=128, fc_size=128, batch_size=6144,
          timesteps=50, from_checkpoint=False, rnn_dropout=0.15, test=False, device="gpu")

    train(rnn_type=rnn_type, epochs=550, rnn_layers=1, rnn_hidden_size=256, fc_size=64, batch_size=6144,
          timesteps=50, from_checkpoint=False, rnn_dropout=0.15, test=False, device="gpu")

    train(rnn_type=rnn_type, epochs=550, rnn_layers=1, rnn_hidden_size=128, fc_size=64, batch_size=6144,
          timesteps=75, from_checkpoint=False, rnn_dropout=0.15, test=False, device="gpu")


main("gru")
