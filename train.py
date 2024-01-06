# train.py
import argparse
import torch
from torch import nn, optim
from model import CustomModel 
from utils import load_data
import json

def train_model(data_dir, save_dir, arch, learning_rate, hidden_sizes, epochs, gpu):
    # Load and preprocess data
    dataloaders, _ = load_data(data_dir, shuffle=True)  # Set shuffle to True for training

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Instantiate the model
    model = CustomModel(input_size=2048, output_size=len(dataloaders['train'].dataset.classes), hidden_sizes=hidden_sizes)

    # Freeze all layers except the classifier
    for param in model.features.parameters():
        param.requires_grad = False

    model.to(device)

    # Define loss function and optimizer
    for param in model.classifier.parameters():
        param.requires_grad = True
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        accuracy = 0.0
        validation_loss = 0.0

        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                validation_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

    # Save the model checkpoint
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # Create a class_to_idx mapping based on the loaded dictionary
    class_to_idx = {class_label: idx for idx, (class_label, class_name) in   enumerate(cat_to_name.items())}
    
    model.class_to_idx = dataloaders['train'].class_to_idx
    # Save entire model (architecture + state dictionary)
    def save_model(model, optimizer, epochs, filepath):
        checkpoint = {
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs
        }
        torch.save(checkpoint, filepath)
def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('data_dir', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet50', help='Architecture (e.g., "resnet50")')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', nargs='+', type=int, default=[1150, 228], help='Hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()
    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

if __name__ == '__main__':
    main()
