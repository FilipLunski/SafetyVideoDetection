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

def train(train_dataset_paths, dev_dataset_paths=[], epochs =500, save=True, device='cuda', from_checkpoint=True, checkpoint_path=None, batch_size=4096, layers=[34,128,64,32], dropout=0.4):
    
    try:

        name = f"{layers}_{dropout}"
        print(f"----------------------------------Training {name} model ---------------------------------------")

        model = KeypointClassifier(device=device)
        train_loader = load_dataset(train_dataset_paths)
        val_loader = load_dataset(dev_dataset_paths) if len(dev_dataset_paths)>0 else None

        

        logger = TensorBoardLogger(
            "logs", name=name)
        trainer = L.Trainer(max_epochs=epochs, logger=logger)
    except Exception as e:
        print(f"Error: {e}")
        return


def main(train_dataset_paths, dev_dataset_paths=[], model_path=None, save=True, device='cuda'):
    pass
# main([r'samples\dataset_fifty_ways_train.h5'], [r'samples\dataset_fifty_ways_validation.h5'], model_path="model_basic.pt")
main([r'samples\dataset_cauca_train.h5', r'samples\dataset_fifty_ways_train.h5'], [r'samples\dataset_cauca_validation.h5', r'samples\dataset_fifty_ways_validation.h5'], model_path="model_basic.pt")
# main(r'samples\dataset_cauca_train.h5', r'samples\dataset_cauca_validation.h5', device='cpu')
