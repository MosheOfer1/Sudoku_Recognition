import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


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


# Function to preprocess the cell image for digit classification
def preprocess_digit_image(cell_image):
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary (black & white)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours to isolate the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # If no contour is found, return a blank tensor
        return torch.zeros(1, 3, 32, 32)  # Return a blank tensor for model input

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

    # Resize the cropped digit to 32x32 (required by your custom model)
    resized_digit = cv2.resize(digit_crop, (32, 32))

    # Convert to 3-channel grayscale (model expects 3 channels)
    digit_rgb = cv2.cvtColor(resized_digit, cv2.COLOR_GRAY2RGB)

    # Debug: Show the resized and processed digit using OpenCV
    cv2.imshow("Processed Digit (Resized to 224x224)", digit_rgb)
    cv2.waitKey(300)  # Show the image for 500 milliseconds

    # Convert the image to a PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    digit_tensor = transform(digit_rgb).unsqueeze(0)  # Add batch dimension

    return digit_tensor


# Function to predict the digit using the trained classifier model
def predict_digit(cell_image, model, device):
    # Preprocess the cell image to match the model's input format
    digit_tensor = preprocess_digit_image(cell_image)
    digit_tensor = digit_tensor.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(digit_tensor)

    # Get the predicted digit (output is a 10-class softmax for digits 0-9)
    probs = F.softmax(outputs, dim=1)
    predicted_digit = torch.argmax(probs).item()

    return predicted_digit
