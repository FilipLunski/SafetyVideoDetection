from pathlib import Path
import h5py
import torch
from KeypointClassifierGRULightning import KeypointClassifierGRULightning
from KeypointClassifierLSTMLightning import KeypointClassifierLSTMLightning
from torch.utils.data import TensorDataset, ChainDataset
from torch.nn.utils.rnn import pack_sequence
from H5PoseDataset import H5PoseDataset
import json
import lightning as L
import os
from lightning.pytorch.loggers import TensorBoardLogger

CHECKPOINTS_FILE = "models_fall/checkpoints.json"

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

                for i in range(len(frames)):
                    start = 0
                    if timesteps is not None and i > timesteps:
                        start = i - timesteps+1

                    k = frames[start:i+1]
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


def train(train_dataset_paths, dev_dataset_paths=[], rnn_type="gru", model_path=None, epochs=400, rnn_layers=2, rnn_hidden_size=128, fc_size=128, rnn_dropout=0.4, fc_droupout=0.4,
          save=True, device='cuda', from_checkpoint=True, batch_size=4096, timesteps=None, checkpoint_path=None):

    try:
        name = f"{rnn_type}_{timesteps}_{rnn_layers}_{rnn_hidden_size}_{fc_size}_{rnn_dropout}_{fc_droupout}"


        print(f"----------------------------------Training {name} model ---------------------------------------")

        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(name)
            
        model = {
            "gru": KeypointClassifierGRULightning(device=device, rnn_hidden_size=rnn_hidden_size,
                                                  rnn_layers_count=rnn_layers, rnn_dropout=rnn_dropout, fc_droupout=fc_droupout),
            "lstm": KeypointClassifierLSTMLightning(device=device, rnn_hidden_size=rnn_hidden_size,
                                                    rnn_layers_count=rnn_layers, rnn_dropout=rnn_dropout, fc_droupout=fc_droupout)
        }.get(rnn_type, None)

        if model is None:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        train_loader = load_dataset(train_dataset_paths, batch_size, timesteps)
        val_loader = load_dataset(dev_dataset_paths, batch_size, timesteps) if len(
            dev_dataset_paths) > 0 else None

        model.train()

        logger = TensorBoardLogger(
            "logs", name=name)
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

        if not from_checkpoint:
            checkpoint_path = None

        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=val_loader, ckpt_path=checkpoint_path)

        if save :
            if model_path is None:
                model_path = f"models_fall/model_{name}.pt"
            model.save(model_path) 

        save_checkpoint_path(name, trainer.checkpoint_callback.best_model_path)

    except Exception as e:
        print(f"An error occurred during training: {e}")


def main(rnn_type="gru", timesteps=None):
    
    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
        rnn_layers=1, rnn_hidden_size=64, fc_size=32, timesteps=timesteps, epochs=200)

    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     timesteps=timesteps)

    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     rnn_dropout=0.3, fc_droupout=0.3, timesteps=timesteps)

    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     rnn_layers=1, rnn_hidden_size=128, fc_size=64, timesteps=timesteps)

    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     rnn_layers=1, rnn_hidden_size=64, fc_size=64, timesteps=timesteps)

    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     rnn_hidden_size=256, fc_size=128 , timesteps=timesteps, epochs=200)
    
    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     rnn_hidden_size=256, fc_size=256, timesteps=timesteps, epochs=200)

    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     rnn_layers=3, rnn_hidden_size=128, fc_size=64, timesteps=timesteps, epochs=200)

    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     rnn_layers=3, rnn_hidden_size=256, fc_size=128, timesteps=timesteps, epochs=200)

    # train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
    #     r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], rnn_type,
    #     rnn_layers=3, rnn_hidden_size=256, fc_size=256, timesteps=timesteps, epochs=200)


main("gru",50)
main("gru",100)

# main("lstm",50)
# main("lstm",100)

# main("gru") 
# main("lstm")