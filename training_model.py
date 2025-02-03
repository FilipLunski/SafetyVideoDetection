
from KeypointClassifier import KeypointClassifier
from H5PoseDataset import H5PoseDataset
import json


def main(train_dataset_path, dev_dataset_path=None, model_path=None, save=True):
    model = KeypointClassifier()
    train_dataset = H5PoseDataset(train_dataset_path)
    dev_dataset = H5PoseDataset(dev_dataset_path) if dev_dataset_path else None
    print(dev_dataset)
    model.trainn(train_dataset, dev_dataset, epochs=20, batch_size=256)


main(r'samples\dataset_cauca_train.h5', r'samples\dataset_cauca_validation.h5')
