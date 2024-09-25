import cv2
import torch
import traceback
from src.grid_recognition import detect_sudoku_grid, warp_perspective, extract_sudoku_grid_and_classify
from src.train_model import DigitClassifier

# Path to the image file
image_path = 'sudoku_newspaper.jpg'

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './models/digit_classifier_sudoku_ds.pth'

try:
    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()


def solve_sudoku_from_image(image_path, model, device):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Convert the image to grayscale and detect the Sudoku grid
    try:
        grid_contour = detect_sudoku_grid(image)
        if grid_contour is not None:
            # Warp the perspective to get a top-down view of the Sudoku grid
            colorful_pic, bw_warped_image = warp_perspective(image, grid_contour)

            # Extract and classify the Sudoku grid
            print("Extracting and classifying Sudoku grid...")
            sudoku_grid, solution = extract_sudoku_grid_and_classify(colorful_pic, bw_warped_image, model, device)
            print(f"Extracted Sudoku grid:\n{sudoku_grid}")

            if sudoku_grid is not None:
                print(f"Solved Sudoku grid:\n{solution}")
                draw_solution_on_image(sudoku_grid, solution, colorful_pic)
                cv2.imshow("Solved Sudoku", colorful_pic)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Unable to solve the Sudoku puzzle")
        else:
            print("Failed to detect Sudoku grid in the image.")
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()


def draw_solution_on_image(original_grid, solved_grid, colorful_image):
    """Draw the solved Sudoku grid on the image, only showing the numbers that were not in the original."""
    cell_height, cell_width = colorful_image.shape[:2]
    cell_height //= 9
    cell_width //= 9

    for i in range(9):
        for j in range(9):
            if original_grid[i][j] == 0 and solved_grid[i][j] != 0:  # Only draw if it was empty
                x = j * cell_width + cell_width // 2
                y = i * cell_height + cell_height // 2

                color = (7, 7, 7)

                # Mimic the original font, size, and style
                cv2.putText(colorful_image, str(solved_grid[i][j]), (x - 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


if __name__ == '__main__':
    solve_sudoku_from_image(image_path, model, device)
