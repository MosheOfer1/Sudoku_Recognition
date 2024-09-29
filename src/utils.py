import torch
import sys
import numpy as np


def is_valid_sudoku(grid):
    """Check if the Sudoku grid is valid without any conflicts using numpy arrays."""

    # Check each row for duplicates (ignoring zeros)
    for row in grid:
        if not is_unique(row):
            return False

    # Check each column for duplicates (ignoring zeros)
    for col in range(9):
        if not is_unique(grid[:, col]):
            return False

    # Check each 3x3 subgrid (box) for duplicates (ignoring zeros)
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            subgrid = grid[box_row:box_row + 3, box_col:box_col + 3].flatten()
            if not is_unique(subgrid):
                return False

    # If no conflicts were found, return True
    return True


def is_unique(nums):
    """Helper function to check if a numpy array contains unique numbers (ignoring zeros)."""
    nums = nums[nums != 0]  # Remove zeros
    return len(nums) == len(np.unique(nums))  # Check if all elements are unique


# Check if the number n can be placed at grid[row][col]
def is_valid(grid, row, col, num):
    # Check if the number is in the row
    if num in grid[row]:
        return False

    # Check if the number is in the column
    if num in grid[:, col]:
        return False

    # Check if the number is in the 3x3 box
    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if grid[box_row_start + i][box_col_start + j] == num:
                return False

    return True


def solve_sudoku_copy(grid: np.ndarray) -> np.ndarray:
    # Create a copy of the grid to avoid modifying the original
    grid_copy = np.copy(grid)

    if solve_sudoku(grid_copy):
        return grid_copy
    else:
        return None


# Function to solve the Sudoku puzzle
def solve_sudoku(grid: np.ndarray) -> bool:
    # Iterate through the grid to find an empty spot (marked as 0)
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                # Try numbers 1 through 9
                for num in range(1, 10):
                    # Check if placing the number is valid
                    if is_valid(grid, row, col, num):
                        grid[row][col] = num  # Place the number

                        # Recursively call the function to continue solving
                        if solve_sudoku(grid):
                            return True

                        # If it leads to an invalid state, backtrack
                        grid[row][col] = 0

                # If no number is valid, return False to backtrack
                return False

    # If the entire grid is filled, return True (solved)
    return True


def calculate_conflicts(grid):
    conflicts = 0

    # Check rows
    for row in grid:
        conflicts += count_conflicts_in_array(row)

    # Check columns
    for col in grid.T:  # grid.T gives us the transpose of the grid
        conflicts += count_conflicts_in_array(col)

    # Check 3x3 sub-grids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            sub_grid = grid[i:i + 3, j:j + 3].flatten()
            conflicts += count_conflicts_in_array(sub_grid)

    return conflicts


def count_conflicts_in_array(arr):
    # Count occurrences of each number (excluding 0)
    counts = np.bincount(arr[arr != 0])

    # Sum up conflicts (a count of n contributes n-1 conflicts)
    return np.sum(np.maximum(counts - 1, 0))


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


