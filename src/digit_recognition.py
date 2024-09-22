import cv2
import numpy as np
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms


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
    # # Threshold the image to binary (black & white)
    # _, thresh = cv2.threshold(cell_image, 128, 255, cv2.THRESH_BINARY_INV)
    #
    # # Find contours to isolate the digit
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # if len(contours) == 0:
    #     # If no contour is found, return a blank tensor
    #     return torch.zeros(1, 3, 32, 32)  # Return a blank tensor for model input
    #
    # # Find the bounding box of the largest contour (assumed to be the digit)
    # largest_contour = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(largest_contour)
    #
    # # Crop the digit with some padding around it (to avoid cutting too closely)
    # padding = 2
    # x1 = max(0, x - padding)
    # y1 = max(0, y - padding)
    # x2 = min(cell_image.shape[1], x + w + padding)
    # y2 = min(cell_image.shape[0], y + h + padding)
    # cell_image = thresh[y1:y2, x1:x2]

    # Resize the cropped digit to 32x32 (required by your custom model)
    resized_digit = cv2.resize(cell_image, (32, 32))

    # # Convert to 3-channel grayscale (model expects 3 channels)
    # resized_digit = cv2.cvtColor(resized_digit, cv2.COLOR_GRAY2RGB)

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
