import torch
import torchvision
import argparse

import model_builder

# Creating a parser
parser = argparse.ArgumentParser()

# Get image path
parser.add_argument("--image",
                    help="Target image file to predict on")

# Get model path
parser.add_argument("--model_path",
                    default="models/05_going_modular_tinyvgg_model.pth",
                    type=str,
                    help="Target model to use for prediction")

# Get hidden units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="Number of hidden units used during model training")

# Parse the arguments
args = parser.parse_args()

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device {device}")

# Get the image path
IMG_PATH = args.image
print(f"[INFO] Predicting on image {IMG_PATH}")

# Function to lead the model
def load_model(model_path = args.model_path):
    # Need to use same hyperparameters as saved model
    model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=128,
                                  output_shape=3).to(device)
    
    print(f"[INFO] Loading model from {model_path}")
    # Load in the save model state dictionary from file
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model

# Function to load model + predict on image
def predict(img_path = IMG_PATH, model_path = args.model_path):
    # Load the model
    model = load_model(model_path)

    # Load the iamge and turn it into torch.float32 (same type as model)
    image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

    # Preprocess the image to get it between 0 and 1
    image /= 255

    # Resize the image to be the same as the model
    transfrom = torchvision.transforms.Resize((64, 64))
    image = transfrom(image)

    # Predict on image
    model.eval()
    with torch.inference_mode():
        # Put the image to target device
        image = image.to(device)

        # Get prediction logits
        pred_logits = model(image.unsqueeze(dim=0))# make sure image has batch dimension (shape: [batch_size, height, width, color_channels])

        # Get prediction probs
        pred_prob = torch.softmax(pred_logits, dim=1)

        # Get prediction labels
        pred_label = torch.argmax(pred_prob, dim = 1)
        pred_label_class = class_names[pred_label]

    print(f"[INFO] Pred Class: {pred_label_class}, Pred Porb: {pred_prob.max():.3f}")

if __name__ == "__main__":
    predict()
