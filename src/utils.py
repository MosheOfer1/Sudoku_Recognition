import torch
import sys
import numpy as np


# Function to check if a number can be placed in a particular position
def is_valid(grid, row, col, num):
    # Check if the number exists in the row
    if num in grid[row]:
        return False

    # Check if the number exists in the column
    if num in grid[:, col]:
        return False

    # Check if the number exists in the 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    subgrid = grid[start_row:start_row + 3, start_col:start_col + 3]
    if num in subgrid:
        return False

    return True


# Backtracking function to solve the grid
def solve_sudoku(grid: np.ndarray) -> np.ndarray:
    # Find the first empty cell (marked as 0)
    empty = np.where(grid == 0)
    if len(empty[0]) == 0:
        # No empty cells, the grid is solved
        return grid

    row, col = empty[0][0], empty[1][0]

    # Try placing numbers 1-9 in the empty cell
    for num in range(1, 10):
        if is_valid(grid, row, col, num):
            # Place the number and recurse
            grid[row, col] = num
            if solve_sudoku(grid) is not None:
                return grid

            # Backtrack if placing num didn't lead to a solution
            grid[row, col] = 0

    return None


def save_model(model, path='./models/digit_classifier_svhn.pth'):
    # Save the trained model
    torch.save(model.state_dict(), path)


# Progress bar function
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create a terminal progress bar.
    :param iteration: current iteration (int)
    :param total: total iterations (int)
    :param prefix: prefix string (str)
    :param suffix: suffix string (str)
    :param decimals: positive number of decimals in percent complete (int)
    :param length: character length of bar (int)
    :param fill: bar fill character (str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()  # New line on completion


