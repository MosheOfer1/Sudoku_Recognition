import cv2
import torch
from PIL import Image


def is_cell_empty(cell_image):
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    center_region = thresh[h // 4:h * 3 // 4, w // 4:w * 3 // 4]
    return cv2.countNonZero(center_region) < 10


def preprocess_digit_image(cell_image):
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return torch.zeros(1, 3, 224, 224)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    padding = 5
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(cell_image.shape[1], x + w + padding)
    y2 = min(cell_image.shape[0], y + h + padding)
    digit_crop = gray[y1:y2, x1:x2]

    resized_digit = cv2.resize(digit_crop, (224, 224))
    digit_rgb = cv2.cvtColor(resized_digit, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(digit_rgb)
    return pil_image


def predict_digit(cell_image, model, processor):
    processed_image = preprocess_digit_image(cell_image)
    inputs = processor(text=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                       images=processed_image,
                       return_tensors="pt",
                       padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    predicted_digit = torch.argmax(probs).item() + 1
    return predicted_digit
