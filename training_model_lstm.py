import h5py
import torch
from KeypointClassifierLSTM import KeypointClassifierLSTM
from torch.utils.data import TensorDataset, ChainDataset
from H5PoseDataset import H5PoseDataset
import json


def load_dataset(paths, timesteps=10):
    keypoints = []
    labels = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            for video in f:
                frames = f[video]['dataset']['keypoints'][()]

                for i in range(len(frames)):
                    if i < timesteps:
                        continue
                    keypoints.append(frames[i-timesteps:i])
                    labels.append([f[video]['dataset']['categories'][i]])
    print(len(labels))
    keypoints = torch.tensor(keypoints)
    print (keypoints.shape)
    labels = torch.tensor(labels)

    return TensorDataset(keypoints, labels)


def main(train_dataset_paths, dev_dataset_paths=[], model_path=None, save=True, device='cuda'):
    model = KeypointClassifierLSTM(device=device)
    train_dataset = load_dataset(train_dataset_paths)
    dev_dataset = load_dataset(dev_dataset_paths) if len(dev_dataset_paths)>0 else None

    print(train_dataset,dev_dataset)
    model.trainn(train_dataset, dev_dataset, epochs=500, batch_size=4096)
    model.save(model_path) if save else None

# main([r'samples\dataset_fifty_ways_train.h5'], [r'samples\dataset_fifty_ways_validation.h5'], model_path="model_basic.pt")
main([r'samples\dataset_cauca_train.h5', r'samples\dataset_fifty_ways_train.h5'], [r'samples\dataset_cauca_test.h5', r'samples\dataset_fifty_ways_test.h5'], model_path="model_lstm.pt")
# main(r'samples\dataset_cauca_train.h5', r'samples\dataset_cauca_validation.h5', device='cpu')
