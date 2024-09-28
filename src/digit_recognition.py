import cv2
import numpy as np
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


def is_cell_empty(cell):
    """Detects if a cell is empty by looking only at the center of the black-and-white image."""
    height, width = cell.shape[:2]

    # Define the size of the center region (e.g., 70% of the original cell size)
    center_height = int(height * 0.5)
    center_width = int(width * 0.5)

    # Calculate the starting points to center the 50% region
    start_y = (height - center_height) // 2
    start_x = (width - center_width) // 2

    # Calculate the end points
    end_y = start_y + center_height
    end_x = start_x + center_width

    # Crop the center of the cell
    center_region = cell[start_y:end_y, start_x:end_x]

    # Calculate the percentage of white pixels in the center region (assuming white pixels are 'empty')
    threshold = 0.95  # Threshold for empty cell (tune based on testing)
    num_white_pixels = cv2.countNonZero(center_region)
    total_pixels = center_region.shape[0] * center_region.shape[1]
    white_ratio = 1 - (num_white_pixels / total_pixels)

    return white_ratio > threshold


# Function to preprocess the B&W cell image for digit classification
def preprocess_digit_image(cell_image):
    # Resize the cropped digit to 32x32 (required by your custom model)
    resized_digit = cv2.resize(cell_image, (32, 32))

    # Convert the image to a PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3-channel RGB image
    ])
    digit_tensor = transform(resized_digit).unsqueeze(0)  # Add batch dimension

    return digit_tensor, resized_digit


# Function to predict the digit from a B&W cell image and return probabilities
def predict_digit_with_probs(digit_tensor, model, device):
    # Preprocess the cell image to match the model's input format
    digit_tensor = digit_tensor.to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(digit_tensor)

    # Get the probabilities for each digit (output is a 9-class softmax for digits 1-9)
    probs = F.softmax(outputs, dim=1).cpu().numpy().flatten()

    # Get the predicted digit
    predicted_digit = np.argmax(probs).item()

    return predicted_digit + 1, probs  # Return both the predicted digit and the probabilities


# Function to plot all the digit predictions in a single figure
def plot_all_digit_predictions(predicted_cells, probs_list):
    num_cells = len(predicted_cells)

    # Define the grid size for the plots (adjust based on the number of predictions)
    cols = 4  # Number of columns (you can adjust this)
    rows = (num_cells + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 4))  # Each cell takes 2 subplots

    # Flatten the axes for easy indexing
    axes = axes.flatten()

    for i, (index, (cell_index, bw_cell)) in enumerate(enumerate(predicted_cells)):
        probs = probs_list[i]
        digits = list(range(1, 10))  # Digits 1 to 9 (since we add 1 to the predicted digit)

        # Plot the image of the cell on the left subplot
        ax_img = axes[i * 2]
        ax_img.imshow(bw_cell, cmap='gray')
        ax_img.axis('off')
        ax_img.set_title(f'Cell {cell_index + 1}')

        # Plot the probabilities as a bar chart on the right subplot
        ax_probs = axes[i * 2 + 1]
        ax_probs.bar(digits, probs)
        ax_probs.set_xticks(digits)
        ax_probs.set_ylim([0, 1])  # Probability ranges from 0 to 1
        ax_probs.set_title('Prediction Probabilities')
        ax_probs.set_xlabel('Digits')
        ax_probs.set_ylabel('Probability')

    # Hide any unused axes
    for j in range(i * 2 + 2, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig('digits.png')
