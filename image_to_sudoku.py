import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load pre-trained CLIP model fine-tuned on SVHN
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Function to check if the central region of the cell contains a digit
def is_cell_empty(cell_image):
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    # Threshold to create a binary image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Define the center region (50% of width and height)
    h, w = thresh.shape
    center_region = thresh[h // 4:h * 3 // 4, w // 4:w * 3 // 4]

    # Count non-zero pixels in the center region
    return cv2.countNonZero(center_region) < 10  # If fewer than 10 non-zero pixels, it's empty


def preprocess_digit_image(cell_image):
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary (black & white)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours to isolate the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # If no contour is found, return a blank image tensor
        return torch.zeros(1, 3, 224, 224)  # Return a blank tensor for model input

    # Find the bounding box of the largest contour (assumed to be the digit)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the digit with some padding around it (to avoid cutting too closely)
    padding = 5
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(cell_image.shape[1], x + w + padding)
    y2 = min(cell_image.shape[0], y + h + padding)
    digit_crop = gray[y1:y2, x1:x2]

    # Resize the cropped digit to 224x224 (required by CLIP model)
    resized_digit = cv2.resize(digit_crop, (224, 224))

    # Convert to RGB (model expects 3 channels)
    digit_rgb = cv2.cvtColor(resized_digit, cv2.COLOR_GRAY2RGB)

    # Debug: Show the resized and processed digit using OpenCV
    cv2.imshow("Processed Digit (Resized to 224x224)", digit_rgb)
    cv2.waitKey(300)  # Show the image for 500 milliseconds

    # Convert the image to a PIL Image
    pil_image = Image.fromarray(digit_rgb)

    return pil_image


# Function to predict the digit using the pre-trained CLIP model
def predict_digit(cell_image, model, processor):
    processed_image = preprocess_digit_image(cell_image)

    # Prepare the image for the model
    inputs = processor(
        text=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        images=processed_image,
        return_tensors="pt",
        padding=True
    )

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted digit
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    predicted_digit = torch.argmax(probs).item() + 1
    return predicted_digit


# Function to detect the Sudoku grid and return the largest contour (the grid)
def detect_sudoku_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which should be the grid
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to get 4 corners (the grid)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    return approx


# Function to warp the perspective of the grid to a straight rectangle
def warp_perspective(image, grid_contour):
    if len(grid_contour) != 4:
        raise ValueError("The Sudoku grid contour should have 4 corners.")

    pts = grid_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Debug: Show the warped grid
    cv2.imshow("Warped Grid", warped)
    cv2.waitKey(700)  # Wait for a key press to close the window

    return warped


# Function to split the warped grid into 81 cells (9x9)
def split_into_cells(warped_image):
    height, width = warped_image.shape[:2]
    cell_height = height // 9
    cell_width = width // 9

    cells = []
    for i in range(9):
        for j in range(9):
            x_start = j * cell_width
            y_start = i * cell_height
            x_end = (j + 1) * cell_width
            y_end = (i + 1) * cell_height
            cell = warped_image[y_start:y_end, x_start:x_end]
            cells.append(cell)

    return cells


# Main function to extract the Sudoku grid, split into cells, and recognize digits
def extract_sudoku_grid_and_classify(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Step 1: Detect the Sudoku grid
    grid_contour = detect_sudoku_grid(image)

    # Step 2: Warp the grid to a straight rectangle
    warped_grid = warp_perspective(image, grid_contour)

    # Step 3: Split the warped grid into 81 equal cells (9x9)
    cells = split_into_cells(warped_grid)

    # Step 4: Classify each cell and create the Sudoku grid
    sudoku_grid = []
    for cell in cells:
        if is_cell_empty(cell):
            sudoku_grid.append(0)
        else:
            digit = predict_digit(cell, model, processor)
            sudoku_grid.append(digit)

    # Reshape the flat list into a 9x9 grid
    sudoku_grid = np.array(sudoku_grid).reshape(9, 9)

    return sudoku_grid


# Example usage:
image_path = 'S5.jpg'  # Replace with the path to your input Sudoku image
sudoku_grid = extract_sudoku_grid_and_classify(image_path)

# Print the extracted Sudoku grid
for row in sudoku_grid:
    print(row)

# Close all OpenCV windows after finishing
cv2.destroyAllWindows()
