# utils.py
import torch
from torchvision import transforms, datasets

def load_data(data_directory, shuffle=True):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }

    # Set shuffle argument for the validation dataloader
    shuffle_valid = shuffle if 'train' in data_directory else False

    image_datasets = {x: datasets.ImageFolder(f'{data_directory}/{x}', data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=shuffle_valid) for x in ['train', 'valid']}

    return dataloaders, image_datasets
