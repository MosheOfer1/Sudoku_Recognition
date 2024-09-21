import argparse
import torch
from src.train_model import DigitClassifier, load_svhn_dataset, train_model
from src.grid_recognition import extract_sudoku_grid_and_classify
from src.utils import save_model


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sudoku Digit Recognizer and SVHN Training')
    parser.add_argument('--task', type=str, choices=['train', 'classify'], required=True,
                        help='Task to perform: "train" or "classify"')
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

    # Task: Train a custom digit classifier on the SVHN dataset
    if args.task == 'train':
        print("Starting model training...")

        # Initialize the custom digit classifier
        model = DigitClassifier().to(device)

        # Load SVHN dataset
        trainloader, testloader = load_svhn_dataset(batch_size=args.batch_size)

        # Train the model
        train_model(model, trainloader, testloader, device, epochs=args.epochs)

        # Save the trained model
        save_model(model, args.model_path)
        print(f"Model trained and saved at {args.model_path}")

    # Task: Classify digits in a Sudoku image
    elif args.task == 'classify':
        if not args.image_path:
            print("Please provide an image path for Sudoku classification.")
            return

        # Load the trained model
        print(f"Loading model from {args.model_path}...")
        model = DigitClassifier().to(device)
        model.load_state_dict(torch.load(args.model_path))

        # Classify digits in the provided Sudoku image
        print(f"Classifying digits in {args.image_path}...")
        sudoku_grid = extract_sudoku_grid_and_classify(args.image_path, model, device)

        # Output the Sudoku grid
        print("Extracted Sudoku Grid:")
        for row in sudoku_grid:
            print(row)


if __name__ == "__main__":
    main()
