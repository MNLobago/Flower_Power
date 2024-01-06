# predict.py
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import CustomModel
import json

def load_model(filepath, optimizer):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    return model, optimizer, epochs

def process_image(image_path):
    # Open the image using PIL
    img = Image.open(image_path)
    
    # Define the image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transformations to the image
    img_tensor = preprocess(img)
    
    # Convert to NumPy array
    img_np = img_tensor.numpy()
    
    return img_np

def predict(image_path, model, topk=5):
    model.eval()
    
    image = process_image(image_path)
    image = torch.from_numpy(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)

    probabilities, classes = torch.exp(output).topk(topk)
    
    return probabilities.numpy()[0], classes.numpy()[0]

def main():
    parser = argparse.ArgumentParser(description='Use a trained network to predict the class for an input image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the trained model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    # Load the model
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    # Process the image and make predictions
    probabilities, classes = predict(args.image_path, model, args.top_k)

    # Convert class indices to class labels using the class_to_idx mapping
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_labels = [idx_to_class[idx] for idx in classes]

    # Load category names mapping if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_labels = [cat_to_name[label] for label in class_labels]

    # Print the results
    for label, prob in zip(class_labels, probabilities):
        print(f"Class: {label}, Probability: {prob:.4f}")

if __name__ == '__main__':
    main()
