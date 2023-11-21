from torchvision.datasets import ImageFolder
import os
import data_transform


def get_data():
    transform = data_transform.TrainTransform()
    # Define the path to the dataset
    dataset_dir = './tiny-imagenet-200'

    # Create training and validation datasets
    train_dataset = ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=transform)
    # val_dataset = ImageFolder(root=os.path.join(dataset_dir, 'test'), transform=transform)
    return train_dataset
