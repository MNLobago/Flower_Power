# This file contains the definiton of my models network architecture
import torch.nn as nn
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(CustomModel, self).__init__()

        # Load the pretrained ResNet50 model
        resnet_model = models.resnet50(pretrained=True)

        self.features = resnet_model

        # Calculate the number of input features for the custom classifier
        num_features = resnet_model.fc.in_features

        # Create a new classifier using nn.Sequential
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.LogSoftmax(dim=1)
        )

        # Replace the final classification layer in the ResNet model with the new classifier
        self.features.fc = self.classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
