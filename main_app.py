import cv2
import torch
from src.grid_recognition import extract_sudoku_grid_and_classify, detect_sudoku_grid, warp_perspective
from src.train_model import DigitClassifier
from src.utils import solve_sudoku


def main():
    # Load the model and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './models/digit_classifier_svhn.pth'
    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load(model_path))

    # Start capturing video feed from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Display the frame
        cv2.imshow('Sudoku Solver', frame)

        # Detect largest contour and mark it on the frame
        try:
            grid_contour = detect_sudoku_grid(frame)
            if grid_contour is not None:
                cv2.drawContours(frame, [grid_contour], -1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error detecting Sudoku grid: {e}")

        # Wait for the user to press 'q' to quit or 'c' to capture and process the frame
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Warp perspective and extract digits
            try:
                warped_grid = warp_perspective(frame, grid_contour)
                sudoku_grid = extract_sudoku_grid_and_classify(warped_grid, model, device)
                print("Recognized Sudoku Grid:\n", sudoku_grid)

                # Solve the puzzle
                solution = solve_sudoku(sudoku_grid)
                print("Sudoku Solution:\n", solution)

                # Write the solution back onto the image
                for i in range(9):
                    for j in range(9):
                        if sudoku_grid[i, j] == 0:
                            cv2.putText(warped_grid, str(solution[i, j]),
                                        (j * 50 + 10, i * 50 + 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Show the solved puzzle
                cv2.imshow('Solved Sudoku', warped_grid)
                cv2.waitKey(0)

            except Exception as e:
                print(f"Error processing Sudoku grid: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
