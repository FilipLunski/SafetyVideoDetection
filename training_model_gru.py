from pathlib import Path
import h5py
import torch
from KeypointClassifierGRULightning import KeypointClassifierGRULightning
from torch.utils.data import TensorDataset, ChainDataset
from torch.nn.utils.rnn import pack_sequence
from H5PoseDataset import H5PoseDataset
import json
import lightning as L
import os


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


def train(train_dataset_paths, dev_dataset_paths=[], model_path=None, epochs=400, rnn_layers=2, rnn_hidden_size=128, fc_size=128, rnn_dropout=0.4, fc_droupout=0.4,
          save=True, device='cuda', from_checkpoint=False, batch_size=4096, timesteps=None):

    best_model_path = None
    if from_checkpoint and os.path.exists('path_to_best_checkpoint'):
        with open('path_to_best_checkpoint', 'r') as file:
            # Read the contents of the file
            best_model_path = file.read()

    model = KeypointClassifierGRULightning(device=device, rnn_hidden_size=rnn_hidden_size,
                                            rnn_layers_count=rnn_layers, rnn_dropout=rnn_dropout, fc_droupout=fc_droupout)

    train_loader = load_dataset(train_dataset_paths, batch_size, timesteps)
    val_loader = load_dataset(dev_dataset_paths, batch_size, timesteps) if len(
        dev_dataset_paths) > 0 else None

    model.train()
    trainer = L.Trainer(max_epochs=epochs)

    model.hparams.model_path = model_path
    model.hparams.previous_model_path = best_model_path
    model.hparams.train_dataset_paths = train_dataset_paths
    model.hparams.dev_dataset_paths = dev_dataset_paths
    model.hparams.save = save
    model.hparams.from_checkpoint = from_checkpoint
    model.hparams.timesteps = 30
    model.hparams.batch_size = batch_size
    model.hparams.epochs = epochs
    model.save_hyperparameters()

    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=best_model_path)

    model.save(model_path) if save else None

    with open('path_to_best_checkpoint', 'w') as file:
        # Write the string to the file
        file.write(trainer.checkpoint_callback.best_model_path)


def main(timesteps=None):

    # main([r'samples\dataset_fifty_ways_train.h5'], [r'samples\dataset_fifty_ways_validation.h5'], model_path="model_basic.pt")
    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, timesteps=timesteps)
    
    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, rnn_dropout=0.3, fc_droupout=0.3, timesteps=timesteps)
    
    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, rnn_dropout=0.4, fc_droupout=0.4, timesteps=timesteps)
    
    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, rnn_layers=1, rnn_hidden_size=128, fc_size=64, rnn_dropout=0.2, fc_droupout=0.2, timesteps=timesteps)

    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, rnn_layers=1, rnn_hidden_size=64, fc_size=64, rnn_dropout=0.2, fc_droupout=0.2, timesteps=timesteps)
    
    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, rnn_layers=2, rnn_hidden_size=256, fc_size=256, rnn_dropout=0.3, fc_droupout=0.3, timesteps=timesteps)

    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, rnn_layers=3, rnn_hidden_size=128, fc_size=64, rnn_dropout=0.3, fc_droupout=0.3, timesteps=timesteps)
    
    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, rnn_layers=3, rnn_hidden_size=256, fc_size=128, rnn_dropout=0.3, fc_droupout=0.3, timesteps=timesteps)

    train([r'samples\dataset_cauca_s_train.h5', r'samples\dataset_fifty_ways_s_train.h5'], [
        r'samples\dataset_cauca_s_validation.h5', r'samples\dataset_fifty_ways_s_validation.h5'], model_path="model_gru.pt",
        from_checkpoint=False, rnn_layers=3, rnn_hidden_size=256, fc_size=256, rnn_dropout=0.3, fc_droupout=0.3, timesteps=timesteps)

    # main(r'samples\dataset_cauca_train.h5', r'samples\dataset_cauca_validation.h5', device='cpu')


if __name__ == "__main__":
    main(50)
    main(100)
    main()
