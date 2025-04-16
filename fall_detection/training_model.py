import h5py
import torch
from KeypointClassifier import KeypointClassifier
from torch.utils.data import TensorDataset, ChainDataset
from H5PoseDataset import H5PoseDataset
import json
import lightning as L
import os
from lightning.pytorch.loggers import TensorBoardLogger

CHECKPOINTS_FILE = "models_fall/checkpoints.json"


def load_dataset(paths, batch_size):
    keypoints = []
    labels = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            for video in f:
                keypoints.extend(f[video]['dataset']['keypoints'][()])
                labels.extend(f[video]['dataset']['categories'][()])
    keypoints = torch.tensor(keypoints)
    labels = torch.tensor(labels).unsqueeze(1)
    # print(labels)
    # print (keypoints.shape)
    dataset = TensorDataset(keypoints, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
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
    r'samples\dataset_cauca_s_train.h5',
    r'samples\dataset_fifty_ways_s_train.h5'
]

default_dev_dataset_paths = [
    r'samples\dataset_cauca_s_validation.h5',
    r'samples\dataset_fifty_ways_s_validation.h5'
]


def train(train_dataset_paths = default_train_dataset_paths, dev_dataset_paths=default_dev_dataset_paths, epochs=500, save=True, device='cuda', from_checkpoint=True, checkpoint_path=None, batch_size=4096, layers=[34, 128, 64, 32], activation="relu", dropout=0.4):

    try:

        name = f"ffnn_{layers}_{dropout}_{activation}"
        print(
            f"----------------------------------Training {name} model ---------------------------------------")

        model = KeypointClassifier(
            layers=layers, activation=activation, dropout=dropout, device=device)
        train_loader = load_dataset(train_dataset_paths, batch_size)
        val_loader = load_dataset(dev_dataset_paths, batch_size) if len(
            dev_dataset_paths) > 0 else None

        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(name)

        logger = TensorBoardLogger(
            "logs", name=name)
        trainer = L.Trainer(max_epochs=epochs, logger=logger)

        model.hparams.previous_model_path = checkpoint_path
        model.hparams.train_dataset_paths = train_dataset_paths
        model.hparams.dev_dataset_paths = dev_dataset_paths
        model.hparams.save = save
        model.hparams.from_checkpoint = from_checkpoint
        model.hparams.batch_size = batch_size
        model.hparams.epochs = epochs
        model.save_hyperparameters()

        if not from_checkpoint:
            checkpoint_path = None

        trainer.fit(model=model, train_dataloaders=train_loader,
                    val_dataloaders=val_loader, ckpt_path=checkpoint_path)

        if save:
            if model_path is None:
                model_path = f"models_fall/model_{name}.pt"
            model.save(model_path)
        save_checkpoint_path(name, trainer.checkpoint_callback.best_model_path)

    except Exception as e:
        print(f"Error: {e}")
        return


train(epochs=500,  layers=[34, 64, 32], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 128, 32], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64], activation="relu", dropout=0.3)

train(epochs=500,  layers=[34, 256, 64], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64, 32], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64, 32], activation="relu", dropout=0.3)

train(epochs=500,  layers=[34, 256, 64, 32], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 256, 128, 32], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 256, 128, 32], activation="relu", dropout=0.3)

train(epochs=500,  layers=[34, 256, 128, 64, 32], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 512, 128, 64, 32], activation="relu", dropout=0.4)

train(epochs=500,  layers=[34, 512, 256, 64, 32], activation="relu", dropout=0.4)



train(epochs=500,  layers=[34, 64, 32], activation="prelu", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64], activation="prelu", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64], activation="prelu", dropout=0.3)

train(epochs=500,  layers=[34, 128, 64, 32], activation="prelu", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64, 32], activation="prelu", dropout=0.3)

train(epochs=500,  layers=[34, 256, 128, 32], activation="prelu", dropout=0.4)

train(epochs=500,  layers=[34, 256, 128, 32], activation="prelu", dropout=0.3)




train(epochs=500,  layers=[34, 64, 32], activation="tanh", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64], activation="tanh", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64], activation="tanh", dropout=0.3)

train(epochs=500,  layers=[34, 128, 64, 32], activation="tanh", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64, 32], activation="tanh", dropout=0.3)

train(epochs=500,  layers=[34, 256, 128, 32], activation="tanh", dropout=0.4)

train(epochs=500,  layers=[34, 256, 128, 32], activation="tanh", dropout=0.3)



train(epochs=500,  layers=[34, 64, 32], activation="mish", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64], activation="mish", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64], activation="mish", dropout=0.3)

train(epochs=500,  layers=[34, 128, 64, 32], activation="mish", dropout=0.4)

train(epochs=500,  layers=[34, 128, 64, 32], activation="mish", dropout=0.3)

train(epochs=500,  layers=[34, 256, 128, 32], activation="mish", dropout=0.4)

train(epochs=500,  layers=[34, 256, 128, 32], activation="mish", dropout=0.3)