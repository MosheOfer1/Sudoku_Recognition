import cv2
import numpy as np
from src.digit_recognition import is_cell_empty, predict_digit


def detect_sudoku_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    return approx


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

    return warped


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


def extract_sudoku_grid_and_classify(image_path, model, device):
    image = cv2.imread(image_path)
    grid_contour = detect_sudoku_grid(image)
    warped_grid = warp_perspective(image, grid_contour)
    cells = split_into_cells(warped_grid)

    sudoku_grid = []
    for cell in cells:
        if is_cell_empty(cell):
            sudoku_grid.append(0)
        else:
            digit = predict_digit(cell, model, device)
            sudoku_grid.append(digit)

    sudoku_grid = np.array(sudoku_grid).reshape(9, 9)
    return sudoku_grid
