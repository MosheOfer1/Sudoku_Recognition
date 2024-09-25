import argparse
import torch

from src.train_model import DigitClassifier, load_svhn_dataset, train_model, load_dynamic_sudoku_dataset, \
    load_mnist_dataset, load_combined_dataset
from src.utils import save_model

"""
For training run:
python3 training.py --dataset combined --epochs 40 --batch_size 64 --model_path ./models/digit_classifier.pth
"""


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sudoku Digit Recognizer and SVHN Training')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the Sudoku image for digit recognition (required for classification)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--model_path', type=str, default='./models/digit_classifier_svhn.pth',
                        help='Path to save/load the trained model')
    args = parser.parse_args()

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using the {device}")


    print("Starting model training...")

    # Initialize the custom digit classifier
    model = DigitClassifier().to(device)

    # Load the selected dataset
    if args.dataset == 'svhn':
        trainloader, testloader = load_svhn_dataset(batch_size=args.batch_size)
    elif args.dataset == 'mnist':
        trainloader, testloader = load_mnist_dataset(batch_size=args.batch_size)
    elif args.dataset == 'sudoku':
        trainloader, testloader = load_dynamic_sudoku_dataset(batch_size=args.batch_size)
    elif args.dataset == 'combined':
        trainloader, testloader = load_combined_dataset(batch_size=args.batch_size)
    else:
        print("Invalid dataset selected.")
        return

    # Train the model
    train_model(model, trainloader, testloader, device, epochs=args.epochs)

    # Save the trained model
    save_model(model, args.model_path)
    print(f"Model trained and saved at {args.model_path}")


if __name__ == "__main__":
    main()
