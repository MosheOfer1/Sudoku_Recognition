import argparse
import torch
from transformers import CLIPProcessor, CLIPModel
from src.sudoku_manager import extract_sudoku_grid_and_classify
from src.train_model import load_svhn_dataset, train_clip_svhn
from src.utils import save_model, load_model


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sudoku Digit Recognizer and CLIP Fine-tuning')
    parser.add_argument('--task', type=str, choices=['train', 'classify'], required=True,
                        help='Task to perform: "train" or "classify"')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the Sudoku image for digit recognition (required for classification)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--model_path', type=str, default='./models/fine_tuned_clip_svhn',
                        help='Path to save/load the fine-tuned model')
    args = parser.parse_args()

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    # Task: Train the model on the SVHN dataset
    if args.task == 'train':
        print("Starting model training...")

        # Load the pre-trained CLIP model and replace the output layer for 10 classes
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.visual_projection = torch.nn.Linear(model.visual_projection.in_features, 10)  # For 10 digits
        model.to(device)

        # Load SVHN dataset
        trainloader, _ = load_svhn_dataset(batch_size=args.batch_size)

        # Train the model
        train_clip_svhn(trainloader, model, device, epochs=args.epochs)

        # Save the fine-tuned model
        save_model(model, args.model_path)
        print(f"Model fine-tuned and saved at {args.model_path}")

    # Task: Classify digits in a Sudoku image
    elif args.task == 'classify':
        if not args.image_path:
            print("Please provide an image path for Sudoku classification.")
            return

        print(f"Loading model from {args.model_path}...")
        model = load_model(args.model_path)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Classify digits in the provided Sudoku image
        print(f"Classifying digits in {args.image_path}...")
        sudoku_grid = extract_sudoku_grid_and_classify(args.image_path, model, processor)

        # Output the Sudoku grid
        print("Extracted Sudoku Grid:")
        for row in sudoku_grid:
            print(row)


if __name__ == "__main__":
    main()
