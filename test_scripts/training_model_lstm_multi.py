import h5py
import torch
from KeypointClassifierLSTMLightningMulti import KeypointClassifierLSTMLightningMulti
from torch.utils.data import TensorDataset, ChainDataset
from H5PoseDataset import H5PoseDataset
import json
import lightning as L
import os

def load_dataset(paths, timesteps=30):
    keypoints = []
    labels = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            for video in f:
                frames = f[video]['dataset']['keypoints'][()]
                categories = f[video]['dataset']['categories'][()]
                for i in range(len(frames)):
                    if i < timesteps:
                        continue
                    keypoints.append(frames[i-timesteps:i])
                    labels_array = []
                    for k in range(3):
                        labels_array.append(float(1 if k == categories[i] else 0))
                    labels.append(labels_array)
    print(len(labels))
    keypoints = torch.tensor(keypoints)
    print(keypoints.shape)
    labels = torch.tensor(labels)

    return TensorDataset(keypoints, labels)

def main(train_dataset_paths, dev_dataset_paths=[], model_path=None, save=True, device='cuda', from_checkpoint=False):
    
    best_model_path = None
    if from_checkpoint and os.path.exists('path_to_best_checkpoint'):
        with open('path_to_best_checkpoint', 'r') as file:
            # Read the contents of the file
            best_model_path = file.read()

    model = KeypointClassifierLSTMLightningMulti(device=device)
    train_dataset = load_dataset(train_dataset_paths)
    dev_dataset = load_dataset(dev_dataset_paths) if len(
        dev_dataset_paths) > 0 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4096, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=4096, pin_memory=True)

    
    model.train()
    trainer = L.Trainer(max_epochs=200)
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=best_model_path)

    model.save(model_path) if save else None
    with open('path_to_best_checkpoint', 'w') as file:
        # Write the string to the file
        file.write(trainer.checkpoint_callback.best_model_path)


# main([r'samples\dataset_fifty_ways_train.h5'], [r'samples\dataset_fifty_ways_validation.h5'], model_path="model_basic.pt")
main([r'samples\dataset_cauca_multi_train.h5', r'samples\dataset_fifty_ways_multi_train.h5'], [
     r'samples\dataset_cauca_multi_validation.h5', r'samples\dataset_fifty_ways_multi_validation.h5'], model_path="model_lstm_multi.pt",
     from_checkpoint=False)
# main(r'samples\dataset_cauca_train.h5', r'samples\dataset_cauca_validation.h5', device='cpu')
