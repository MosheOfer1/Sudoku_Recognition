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


def solve_sudoku(grid: np.ndarray) -> np.ndarray:
    """
    Solve the Sudoku puzzle and return the solution if there is exactly one unique solution.
    If there are 0 or more than 1 solutions, return None.
    """
    solution_count = [0]  # Track the number of solutions found
    unique_solution = [None]  # Store the unique solution if found

    # Helper function to perform backtracking and count solutions
    def backtrack(grid):
        empty = np.where(grid == 0)
        if len(empty[0]) == 0:
            # No empty cells, we found a solution
            solution_count[0] += 1
            if solution_count[0] == 1:
                # Store the first solution found
                unique_solution[0] = np.copy(grid)
            # Stop further search if more than one solution is found
            if solution_count[0] > 1:
                return False
            return True

        row, col = empty[0][0], empty[1][0]

        # Try placing numbers 1-9 in the empty cell
        for num in range(1, 10):
            if is_valid(grid, row, col, num):
                grid[row, col] = num
                if not backtrack(grid):  # If two solutions are found, stop recursion
                    return False
                grid[row, col] = 0  # Backtrack

        return True

    # Run the backtracking algorithm to search for all solutions
    grid = grid.copy()
    backtrack(grid)

    # Return the unique solution if exactly one was found, otherwise return None
    if solution_count[0] < 3:
        return unique_solution[0]
    else:
        return None


def is_valid(grid: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check if placing num in grid[row, col] is valid according to Sudoku rules."""
    # Check if num is already in the current row, column, or 3x3 subgrid
    if num in grid[row, :] or num in grid[:, col]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in grid[start_row:start_row + 3, start_col:start_col + 3]:
        return False
    return True


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


