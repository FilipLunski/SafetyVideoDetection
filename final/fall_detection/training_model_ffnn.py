import h5py
import torch
from KeypointClassifierFFNN import KeypointClassifierFFNN
from torch.utils.data import TensorDataset, ChainDataset
import json
import lightning as L
import os
from lightning.pytorch.loggers import TensorBoardLogger
import time

CHECKPOINTS_FILE = "models_fall/checkpoints.json"


def load_dataset(paths, batch_size, shuffle=True):
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
        dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle)
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
    r'samples\dataset_fifty_ways_m_train.h5'
]

default_dev_dataset_paths = [
    r'samples\dataset_cauca_m_validation.h5',
    r'samples\dataset_fifty_ways_m_validation.h5'
]

default_test_dataset_paths = [
    
    r'samples\dataset_cauca_m_test.h5',
    r'samples\dataset_fifty_ways_m_test.h5'
]


def main(train_dataset_paths=default_train_dataset_paths, dev_dataset_paths=default_dev_dataset_paths, epochs=500, save=True, device='cuda', from_checkpoint=True,
          checkpoint_path=None, batch_size=512, layers=[34, 128, 64, 32], activation="relu", dropout=0.4, batch_norm=True, test=False, test_dataset_paths=default_test_dataset_paths):

    try:

        name = f"ffnn_{layers}_{dropout}_{activation}_{batch_size}{'_bn' if batch_norm else ''}"
        print(
            f"----------------------------------Training {name} model ---------------------------------------")

        model = KeypointClassifierFFNN(
            layers=layers, activation=activation, dropout=dropout, device=device, batch_norm=batch_norm)

        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(name)

        logger = TensorBoardLogger(
            "logs_fnn", name=name)
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

        if not test:
            model.train()
            train_loader = load_dataset(train_dataset_paths, batch_size)
            val_loader = load_dataset(dev_dataset_paths, batch_size) if len(
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
                test_dataset_paths, batch_size, shuffle=False)  

            trainer.test(model=model, dataloaders=test_loader,
                         ckpt_path=checkpoint_path)
            
            model = KeypointClassifierFFNN.load_from_checkpoint(checkpoint_path)

            model.eval()
            print(model.device)
            model.to(device)
            while(True):    
                time_sum = 0
                sample_count = 0
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    for i in range(inputs.size(0)): 
                        sample = inputs[i].unsqueeze(0)
                        start_time = time.perf_counter()
                        output = model.predict_step(sample)
                        if device != "cpu":
                            torch.cuda.synchronize(device)  # Wait for GPU to finish
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        time_sum += elapsed_time
                        sample_count += 1
                avg_time = time_sum / sample_count   # Convert to seconds
                print(f"Average time per sample: {avg_time:.6f} seconds")

    except Exception as e:
        print(f"Error: {e}")
        return

main(epochs=600,  layers=[34, 128, 64, 32], activation="relu", dropout=0.3, test=True, device="cpu", checkpoint_path="fall_detection/FFNN.ckpt")

# main(epochs=600,  layers=[34, 512, 128, 64, 32], activation="relu", dropout=0.4)
# main(epochs=600,  layers=[34, 256, 128, 64, 32], activation="relu", dropout=0.4)


# main(epochs=600,  layers=[34, 128, 32], activation="relu", dropout=0.2)

# main(epochs=600,  layers=[34, 128, 64], activation="relu", dropout=0.2)


# main(epochs=1000,  layers=[34, 512, 256, 64, 32],
# main activation="relu", dropout=0.2)


# main(epochs=500,  layers=[34, 64, 32], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 32], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64], activation="relu", dropout=0.3)

# main(epochs=500,  layers=[34, 256, 64], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64, 32], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64, 32], activation="relu", dropout=0.3)

# main(epochs=500,  layers=[34, 256, 64, 32], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 256, 128, 32], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 256, 128, 32], activation="relu", dropout=0.3)

# main(epochs=500,  layers=[34, 256, 128, 64, 32], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 512, 128, 64, 32], activation="relu", dropout=0.4)

# main(epochs=500,  layers=[34, 512, 256, 64, 32], activation="relu", dropout=0.4)


# main(epochs=500,  layers=[34, 64, 32], activation="prelu", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64], activation="prelu", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64], activation="prelu", dropout=0.3)

# main(epochs=500,  layers=[34, 128, 64, 32], activation="prelu", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64, 32], activation="prelu", dropout=0.3)

# main(epochs=500,  layers=[34, 256, 128, 32], activation="prelu", dropout=0.4)

# main(epochs=500,  layers=[34, 256, 128, 32], activation="prelu", dropout=0.3)


# main(epochs=500,  layers=[34, 64, 32], activation="tanh", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64], activation="tanh", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64], activation="tanh", dropout=0.3)

# main(epochs=500,  layers=[34, 128, 64, 32], activation="tanh", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64, 32], activation="tanh", dropout=0.3)

# main(epochs=500,  layers=[34, 256, 128, 32], activation="tanh", dropout=0.4)

# main(epochs=500,  layers=[34, 256, 128, 32], activation="tanh", dropout=0.3)


# main(epochs=500,  layers=[34, 64, 32], activation="mish", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64], activation="mish", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64], activation="mish", dropout=0.3)

# main(epochs=500,  layers=[34, 128, 64, 32], activation="mish", dropout=0.4)

# main(epochs=500,  layers=[34, 128, 64, 32], activation="mish", dropout=0.3)

# main(epochs=500,  layers=[34, 256, 128, 32], activation="mish", dropout=0.4)

# main(epochs=500,  layers=[34, 256, 128, 32], activation="mish", dropout=0.3)
