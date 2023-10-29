import argparse
import torch
from torch import nn
from torchvision import models, transforms, datasets
from PIL import Image
import json


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    vgg_type = checkpoint['vgg_type']

    if vgg_type == "vgg11":
        model = models.vgg11(pretrained=True)
    elif vgg_type == "vgg13":
        model = models.vgg13(pretrained=True)
    elif vgg_type == "vgg16":
        model = models.vgg16(pretrained=True)
    elif vgg_type == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    image = Image.open(image_path)

    # Define the image transformations (resize, crop, normalize)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image)
    return image


def predict(image_path, model, topk, device, cat_to_name):
    image = process_image(image_path).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities, predicted_classes = output.topk(topk)

    probabilities = probabilities.exp().cpu().numpy()[0]
    predicted_classes = predicted_classes.cpu().numpy()[0]

    class_names = [cat_to_name[str(cls)] for cls in predicted_classes]

    return probabilities, class_names


def print_predictions(args):
    # Load the model from the "models" folder
    model = load_checkpoint("models/checkpoint.pth")

    # Set the device (GPU or CPU)
    # device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model.to(device)

    # Load the category names
    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # Predict the image
    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    print("Predictions:")
    for i, (class_name, probability) in enumerate(zip(top_classes, top_ps), start=1):
        print(f"Top-{i} Class: {class_name}, Probability: {probability:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='image_filepath', help="Path to the image file you want to classify")

    # Add other command-line arguments here if needed

    args = parser.parse_args()

    # Provide the path to the JSON file (category names mapping)
    args.category_names_json_filepath = "cat_to_name.json"

    # Set the number of top classes to return (default is 5)
    args.top_k = 5

    # Set GPU flag (True/False) based on your preference
    args.gpu = True  # Change this to True if you want to use GPU

    print_predictions(args)
