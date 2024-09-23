import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from src.grid_recognition import warp_perspective, detect_sudoku_grid


class DynamicSudokuDataset(Dataset):
    def __init__(self, length=30_000, transform=None):
        self.length = length
        self.transform = transform
        self.sudoku_image = None  # Placeholder for the current Sudoku image
        self.labels = None  # Placeholder for the labels
        self.digits = None  # Placeholder for digit cells
        self.current_idx = 0  # Tracks the current index (0 to 80)

    def generate_new_sudoku(self):
        """Generates a new Sudoku grid and splits it into digit cells."""
        # Step 1: Generate a random Sudoku image and its labels
        self.sudoku_image, self.labels = create_random_sudoku_image()

        # Step 2: Warp the perspective (optional)
        grid_contour = detect_sudoku_grid(self.sudoku_image)
        warped_image, _ = warp_perspective(self.sudoku_image, grid_contour)

        # Step 3: Split into 81 digit cells
        self.digits = split_into_cells(warped_image)

    def __len__(self):
        """Returns a large number, mimicking an infinite dataset."""
        return self.length

    def __getitem__(self, idx):
        """Returns a digit image and label."""
        # If we haven't generated a Sudoku yet or after every 81 digits, generate a new one
        if self.current_idx % 81 == 0:
            self.generate_new_sudoku()
            self.current_idx = 0  # Reset index

        # Get the current digit and label
        digit_img = preprocess_digit_image(self.digits[self.current_idx]) if random.random() < 0.5 else self.digits[self.current_idx]
        label = self.labels.flatten()[self.current_idx]

        # Apply any transformations (e.g., resizing, normalization, etc.)
        if self.transform:
            digit_img = self.transform(digit_img)

        # Increment the current index
        self.current_idx += 1

        return digit_img, label


def plot_image(original_image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Sudoku Image')
    plt.axis('off')
    plt.show()


# Function to display the cells with their labels
def plot_cells_with_labels(cells, labels):
    plt.figure(figsize=(9, 9))  # 9x9 grid
    for idx, (cell, label) in enumerate(zip(cells, labels.flatten())):
        # Ensure the pixel values are in the range [0, 255] and cast to uint8 for RGB plotting
        cell = np.clip(cell, 0, 255).astype(np.uint8)

        plt.subplot(9, 9, idx + 1)
        plt.imshow(cell, cmap='gray')
        plt.title(str(label), fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_random_sudoku_image():
    size = random.randint(380, 460)
    grid_size = (size, size)
    # Create a random color for the background
    bg_color = np.random.randint(180, 255, (1, 3)).tolist()[0]

    # Create the background canvas with the random color
    background = np.ones((grid_size[0] + 200, grid_size[1] + 200, 3), dtype=np.uint8) * np.array(bg_color,
                                                                                                 dtype=np.uint8)
    background = cv2.GaussianBlur(background, (51, 51), 0)  # Apply Gaussian blur

    # Create the white grid on top of the background
    image = np.ones((grid_size[0], grid_size[1], 3), dtype=np.uint8) * 255  # White grid
    digits = np.random.randint(1, 10, (9, 9))  # 81 random digits

    # Font, color, thickness variations for the digits
    # List of available fonts in OpenCV
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_ITALIC
    ]

    # Randomly select a font
    font = random.choice(fonts)
    digit_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))  # Black digits
    thickness = random.randint(2, 4)  # Adjust the thickness to fit the font size

    cell_size = grid_size[0] // 9  # Size of each cell in the grid
    font_size = cell_size / 100 * 3  # Set font size based on the cell size

    # Draw the digits on the grid
    for row in range(9):
        for col in range(9):
            digit = str(digits[row, col])

            # Calculate the position for the text
            text_size = cv2.getTextSize(digit, font, font_size, thickness)[0]
            text_x = col * cell_size + (cell_size - text_size[0]) // 2  # Center the digit horizontally
            text_y = row * cell_size + (cell_size + text_size[1]) // 2  # Center the digit vertically

            # Add the digit to the grid
            cv2.putText(image, digit, (text_x, text_y), font, font_size, digit_color, thickness)

    # Add 8 vertical and 8 horizontal lines with random thickness between digits
    line_color = (0, 0, 0)  # Black lines
    for i in range(1, 9):  # 8 lines between cells
        # Random thickness for each line
        line_thickness = random.randint(1, 3)

        # Vertical line
        start_point_v = (i * cell_size, 0)
        end_point_v = (i * cell_size, grid_size[0])
        cv2.line(image, start_point_v, end_point_v, line_color, line_thickness)

        # Horizontal line
        start_point_h = (0, i * cell_size)
        end_point_h = (grid_size[1], i * cell_size)
        cv2.line(image, start_point_h, end_point_h, line_color, line_thickness)

    # Create a brown rectangle around the grid
    brown_color = (42, 42, 165)  # BGR format for brown
    border_thickness = 10
    cv2.rectangle(image, (0, 0), (grid_size[0], grid_size[1]), brown_color, border_thickness)

    # Increase the canvas size before rotating to prevent cutting
    expanded_canvas_size = (grid_size[0] + 200, grid_size[1] + 200)
    expanded_canvas = np.ones((expanded_canvas_size[0], expanded_canvas_size[1], 3), dtype=np.uint8) * np.array(
        bg_color, dtype=np.uint8)

    # Paste the grid in the center of the expanded canvas
    x_offset = (expanded_canvas_size[1] - grid_size[1]) // 2
    y_offset = (expanded_canvas_size[0] - grid_size[0]) // 2
    expanded_canvas[y_offset:y_offset + grid_size[0], x_offset:x_offset + grid_size[1]] = image

    # Randomly rotate the entire canvas
    angle = random.uniform(-15, 15)  # Random angle between -15 and 15 degrees
    image_center = (expanded_canvas_size[1] // 2, expanded_canvas_size[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(expanded_canvas, rotation_matrix, expanded_canvas_size,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    # Paste the rotated Sudoku grid on the background
    bg_x_offset = (background.shape[1] - expanded_canvas_size[1]) // 2
    bg_y_offset = (background.shape[0] - expanded_canvas_size[0]) // 2
    background[bg_y_offset:bg_y_offset + expanded_canvas_size[0],
    bg_x_offset:bg_x_offset + expanded_canvas_size[1]] = rotated_image

    return background, digits


# Split the warped image into 9x9 cells
def split_into_cells(warped_image, cell_size=32):
    cells = []
    h, w = warped_image.shape[:2]
    cell_h, cell_w = h // 9, w // 9
    for i in range(9):
        for j in range(9):
            cell = warped_image[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell_resized = cv2.resize(cell, (cell_size, cell_size))
            cells.append(cell_resized)
    return cells


# Preprocess the digit image
def preprocess_digit_image(cell_image):
    # Convert to grayscale, apply threshold or any additional preprocessing if necessary
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return preprocess_or_convert_to_rgb(thresh)


# Convert to 3-channel (RGB) if needed
def preprocess_or_convert_to_rgb(cell):
    if len(cell.shape) == 2:  # If the cell is grayscale (2D array)
        return cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
    elif len(cell.shape) == 3 and cell.shape[2] == 1:  # If the cell is single-channel
        return np.repeat(cell, 3, axis=2)  # Repeat the single channel into 3 channels
    else:
        return cell  # Already 3-channel RGB


# # Save the dataset in .pt format
# def save_to_ptfile(images, labels, filename='sudoku_dataset.pt'):
#     # Ensure all images are RGB (32, 32, 3)
#     images_rgb = [preprocess_or_convert_to_rgb(img) for img in images]
#
#     # Convert images to tensor format (from lists of numpy arrays)
#     images_tensor = torch.stack(
#         [torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in images_rgb])  # Permute to (C, H, W)
#     labels_tensor = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor
#
#     # Save to .pt file
#     torch.save({'images': images_tensor, 'labels': labels_tensor}, filename)
#
#
# # Function to load the dataset and plot all digits with labels
# def load_and_plot_sudoku_dataset(filename='sudoku_dataset.pt', batch_size=81):
#     # Load the dataset
#     dataset = SudokuDataset(data_path=filename)
#
#     # Create a DataLoader to load one batch (81 images)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#     # Get one batch of images and labels
#     for images, labels in dataloader:
#         # Convert images to numpy arrays for plotting (detach from computation graph)
#         images = images.detach().numpy()
#
#         # Ensure images are reshaped correctly for 3-channel RGB images (batch_size, height, width, channels)
#         images = images.transpose(0, 2, 3, 1)  # Convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
#
#         # Plot the first batch of images and labels
#         plot_cells_with_labels(images, labels.numpy())
#         break  # We only need the first batch


# Modify the main function to include separate visualization
# # Main function to generate the dataset
# def generate_sudoku_dataset(num_samples=1000):
#     all_images = []
#     all_labels = []
#
#     for _ in range(num_samples):
#         # Step 1: Generate a random Sudoku image
#         sudoku_image, labels = create_random_sudoku_image()
#
#         # Step 2: Warp the perspective (optional: define your grid_contour)
#         grid_contour = detect_sudoku_grid(sudoku_image)
#         warped_image, _ = warp_perspective(sudoku_image, grid_contour)
#
#         # Step 3: Split into 81 digit cells
#         cells = split_into_cells(warped_image)
#
#         # Step 4: Preprocess each digit image and append to dataset
#         processed_images = [
#             preprocess_digit_image(cell) if random.random() < 0.5 else cell for cell in cells
#         ]
#         # Collect all images and labels
#         all_images.extend(processed_images)
#         all_labels.extend(labels.flatten())  # Flatten the 9x9 label grid
#
#     # Step 5: Save to .mat file
#     save_to_ptfile(all_images, all_labels)

def generate_and_visualize_sudoku_dataset():
    # Step 1: Generate a random Sudoku image
    sudoku_image, labels = create_random_sudoku_image()

    # Step 2: Warp the perspective (optional: define your grid_contour)
    grid_contour = detect_sudoku_grid(sudoku_image)
    warped_image, _ = warp_perspective(sudoku_image, grid_contour)

    # Step 3: Split into 81 digit cells
    cells = split_into_cells(warped_image)

    # Step 4: Preprocess each digit image
    processed_images = [
        preprocess_digit_image(cell) if random.random() < 0.5 else cell for cell in cells
    ]
    # Plot separately
    plot_image(sudoku_image)
    plot_image(warped_image)
    plot_cells_with_labels(processed_images, labels)


if __name__ == "__main__":
    generate_and_visualize_sudoku_dataset()
    # Initialize the dataset
    sudoku_dataset = DynamicSudokuDataset()

    # Create a DataLoader with batch size 32
    data_loader = DataLoader(sudoku_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Example: Fetch a few batches of data from the infinite stream
    for batch_idx, (digit_images, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx}: Digit Image Shape: {digit_images.shape}, Labels Shape: {labels.shape}")

        # Stop after a few batches for the sake of demonstration
        if batch_idx >= 4:  # Fetch only 5 batches (i.e., 5 * 32 = 160 images)
            break
