import cv2
import numpy as np
import torch
import heapq
from src.digit_recognition import is_cell_empty, predict_digit_with_probs, preprocess_digit_image
from src.utils import solve_sudoku, is_valid_sudoku


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

    # Get the four corner points
    pts = grid_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Get the width and height of the new perspective
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the perspective transform
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Get the perspective transform matrix and warp the image
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Create a black and white version of the warped image with high contrast
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Apply adaptive threshold to improve the contrast (detect cell edges and empty cells)
    _, bw_warped = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # Debug: Show the warped grid
    # cv2.imshow("Warped Grid", warped)
    # cv2.imshow("Black and White Warped Grid", bw_warped)
    # cv2.waitKey(0)  # Wait for a key press to close the window

    return warped, bw_warped


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


# Best-First Search (BFS) implementation
def best_first_search(probs, keys):
    # Priority queue (min-heap) for Best-First Search
    heap = []

    # Initial state: Start with the highest probability for each index
    initial_state = [p[0][0] for p in probs]  # Highest index for each entry
    initial_prob_product = np.prod([p[0][1] for p in probs])  # Product of highest probabilities

    # Push the initial state into the heap with negative priority (since heapq is min-heap)
    heapq.heappush(heap, (-initial_prob_product, initial_prob_product, initial_state))

    # Visited states to avoid re-exploring
    visited = set()

    while heap:
        # Pop the node with the highest probability product
        _, prob_product, current_state = heapq.heappop(heap)

        # Print the current best state and its probability product
        print(f"Current best state: {[x+ 1 for x in current_state]}, Probability product: {prob_product}")

        grid = np.zeros((9, 9), dtype=int)  # Create a 9x9 NumPy array initialized with zeros

        # Fill the grid with the configuration
        for index_str, digit in zip(keys, current_state):
            index = int(index_str)  # Convert index to integer
            row, col = divmod(index, 9)  # Get row and column from the index
            grid[row][col] = digit + 1  # Place the digit in the correct cell
        yield grid

        # Mark this state as visited
        visited.add(tuple(current_state))

        # Generate child nodes (one index is changed at a time)
        for i in range(len(probs)):
            # Create a new state by changing the i-th index to the j-th option
            new_state = current_state[:]

            # Find the index where the first element is 'current_state[i]'
            j = next(idx for idx, (x, _) in enumerate(probs[i]) if x == current_state[i])

            if j + 1 == len(probs[i]):
                continue

            new_state[i] = probs[i][j + 1][0]  # Update the i-th index

            # If this new state hasn't been visited yet
            if tuple(new_state) not in visited:  # Convert list to tuple for checking
                visited.add(tuple(new_state))  # Add the new state to visited as a tuple
                new_prob = []
                for k in range(len(probs)):
                    # Find the index where the first element is 'current_state[k]'
                    j = next(idx for idx, (x, _) in enumerate(probs[k]) if x == new_state[k])
                    new_prob.append(probs[k][j][1])

                # Calculate the new product of probabilities
                new_prob_product = np.prod(new_prob)

                # Push the new state into the heap
                heapq.heappush(heap, (-new_prob_product, new_prob_product, new_state))


def generate_most_probable_configuration(probs_list):
    # {index in the grid : {digit1 : p, digit2 : p, ...}, ...}
    dict_of_dicts = {
        index: {digit: prob for digit, prob in enumerate(prob_list)}
        for index, prob_list in probs_list
    }

    # Sort all inner dicts by their values
    sorted_dict_of_dicts = {
        key: dict(sorted(inner_dict.items(), reverse=True, key=lambda item: item[1]))  # Sort inner dict by values
        for key, inner_dict in dict_of_dicts.items()
    }

    # Extract both keys (digits) and values (probabilities) from each inner dictionary
    key_value_lists = [
        list(inner_dict.items()) for inner_dict in sorted_dict_of_dicts.values()
    ]
    # Run the Best-First Search
    return best_first_search(key_value_lists, sorted_dict_of_dicts.keys())


def extract_sudoku_grid_and_classify(warped_image, bw_warped_image, model, device, max_prob_threshold=0.2,
                                     entropy_threshold=2):
    """
    Extracts Sudoku grid and classifies each digit using only the black-and-white image,
    and plots the predictions and probabilities for each predicted cell in a single figure.

    Args:
        warped_image: Original Sudoku grid image.
        bw_warped_image: Black-and-white version of the Sudoku grid for empty cell detection.
        model: Trained digit classifier model.
        device: Torch device (CPU or GPU).
        max_prob_threshold: Threshold for the maximum probability to consider a confident prediction.
        entropy_threshold: Threshold for entropy to consider a confident prediction.

    Returns:
        The predicted Sudoku grid and its solution if one is found.
    """
    # Split the black-and-white image into individual cells
    cells = split_into_cells(warped_image)
    bw_cells = split_into_cells(bw_warped_image)

    predicted_cells = []  # Store cells that had confident predictions for plotting
    probs_list = []  # Store the probabilities for each cell

    for index, cell in enumerate(cells):
        if is_cell_empty(bw_cells[index]):  # Use black-and-white image for empty cell detection
            continue
        else:
            digit_tensor, digit_img = preprocess_digit_image(cell)
            digit, probs = predict_digit_with_probs(digit_tensor, model, device)  # Get digit and probabilities

            # Convert probs to a PyTorch tensor if it's not already
            probs = torch.tensor(probs)  # Convert numpy array to tensor

            # Check maximum probability confidence
            max_prob = torch.max(probs).item()

            # Compute the entropy of the probability distribution
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

            # Check if the prediction is confident based on both maximum probability and entropy
            if max_prob > max_prob_threshold and entropy < entropy_threshold:
                # Store the cell and probabilities if the model is confident
                predicted_cells.append((index, digit_img))
                probs_list.append((index, probs))
            else:
                print(f"Removed the {index} image with prob: {max_prob}, entropy: {entropy}")

    # After processing all the cells, plot all the predictions in a single figure
    # plot_all_digit_predictions(predicted_cells, [x[1] for x in probs_list])

    # Generate the most probable Sudoku configuration and attempt to solve it
    limit = 20
    for i, sudoku_grid in enumerate(generate_most_probable_configuration(probs_list)):
        if not is_valid_sudoku(sudoku_grid):
            # print("Not a valid grid")
            continue

        solution = solve_sudoku(sudoku_grid)
        if solution is not None:
            return sudoku_grid, solution
        elif i == limit:
            return None, None

    return None, None

