import h5py
import torch
from KeypointClassifier import KeypointClassifier
from torch.utils.data import TensorDataset, ChainDataset
from H5PoseDataset import H5PoseDataset
import json


def load_dataset(paths):
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
    return TensorDataset(keypoints, labels)


def main(train_dataset_paths, dev_dataset_paths=[], model_path=None, save=True, device='cuda'):
    model = KeypointClassifier(device=device)
    train_dataset = load_dataset(train_dataset_paths)
    dev_dataset = load_dataset(dev_dataset_paths) if len(dev_dataset_paths)>0 else None

    print(train_dataset,dev_dataset)
    model.trainn(train_dataset, dev_dataset, epochs=200, batch_size=4096)

main([r'samples\dataset_cauca_train.h5'], [r'samples\dataset_cauca_validation.h5'])
# main(r'samples\dataset_cauca_train.h5', r'samples\dataset_cauca_validation.h5', device='cpu')
