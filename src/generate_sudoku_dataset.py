import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from src.down_fonts import download_and_save_fonts
from src.grid_recognition import warp_perspective, detect_sudoku_grid


class DynamicSudokuDataset(Dataset):
    def __init__(self, length=30_000, transform=None):
        self.length = length
        self.transform = transform
        self.sudoku_image = None  # Placeholder for the current Sudoku image
        self.labels = None  # Placeholder for the labels
        self.digits = None  # Placeholder for digit cells
        self.current_idx = 0  # Tracks the current index (0 to 80)
        self.fonts = download_and_save_fonts(150)

    def generate_new_sudoku(self):
        """Generates a new Sudoku grid and splits it into digit cells."""
        # Step 1: Generate a random Sudoku image and its labels
        self.sudoku_image, self.labels = create_random_sudoku_image(self.fonts)

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
            plot_image(self.sudoku_image, idx)
            self.current_idx = 0  # Reset index

        # Get the current digit and label
        digit_img = preprocess_digit_image(self.digits[self.current_idx]) if random.random() < 0.05 else self.digits[
            self.current_idx]
        label = self.labels.flatten()[self.current_idx]

        # Increment the current index
        self.current_idx += 1

        # Convert NumPy array to PIL image
        digit_img = Image.fromarray(digit_img)

        # Apply any transformations (e.g., resizing, normalization, etc.)
        if self.transform:
            digit_img = self.transform(digit_img)

        return digit_img, label


def plot_image(original_image, idx):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Sudoku Image')
    plt.axis('off')
    plt.savefig(f'sudoku_{idx}.png')
    plt.close()


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


# Modified create_random_sudoku_image function
def create_random_sudoku_image(fonts):
    size = random.randint(380, 460)
    grid_size = (size, size)
    bg_color = tuple(np.random.randint(180, 255, 3))

    background = Image.new('RGB', (grid_size[0] + 200, grid_size[1] + 200), bg_color)
    image = Image.new('RGB', grid_size, 'white')
    draw = ImageDraw.Draw(image)

    digits = np.random.randint(1, 10, (9, 9))

    font_path = random.choice(fonts)
    digit_color = tuple(np.random.randint(0, 150, 3))
    cell_size = grid_size[0] // 9
    font_size = int(cell_size * random.uniform(0.5, 0.7))
    font = ImageFont.truetype(font_path, font_size)

    for row in range(9):
        for col in range(9):
            digit = str(digits[row, col])
            text_bbox = draw.textbbox((0, 0), digit, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = col * cell_size + (cell_size - text_width) // 2
            text_y = row * cell_size + (cell_size - text_height) // 2
            draw.text((text_x, text_y), digit, font=font, fill=digit_color)

    for i in range(10):
        line_thickness = random.randint(1, 4)
        draw.line([(i * cell_size, 0), (i * cell_size, grid_size[0])], fill='black', width=line_thickness)
        draw.line([(0, i * cell_size), (grid_size[1], i * cell_size)], fill='black', width=line_thickness)

    brown_color = (42, 42, 165)
    border_thickness = random.randint(6, 16)
    draw.rectangle([(0, 0), (grid_size[0] - 1, grid_size[1] - 1)], outline=brown_color, width=border_thickness)

    angle = random.uniform(-25, 25)
    rotated_image = image.rotate(angle, expand=True, fillcolor=bg_color)

    bg_x_offset = (background.width - rotated_image.width) // 2
    bg_y_offset = (background.height - rotated_image.height) // 2
    background.paste(rotated_image, (bg_x_offset, bg_y_offset))

    return np.array(background), digits


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


def generate_and_visualize_sudoku_dataset(fonts):
    # Step 1: Generate a random Sudoku image
    sudoku_image, labels = create_random_sudoku_image(fonts)

    # Step 2: Warp the perspective (optional: define your grid_contour)
    grid_contour = detect_sudoku_grid(sudoku_image)
    warped_image, _ = warp_perspective(sudoku_image, grid_contour)

    # Step 3: Split into 81 digit cells
    cells = split_into_cells(warped_image)

    # Step 4: Preprocess each digit image
    processed_images = [
        preprocess_digit_image(cell) if random.random() < 0.1 else cell for cell in cells
    ]
    # Plot separately
    plot_image(sudoku_image, 1)
    plot_image(warped_image, 2)
    plot_cells_with_labels(processed_images, labels)


def plot_image_batch(images, labels, batch_idx):
    # Convert images from tensor to numpy and denormalize
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = (images * 0.5 + 0.5).clip(0, 1)

    # Create a grid of subplots
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle(f'Batch {batch_idx}', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < images.shape[0]:
            ax.imshow(images[i])
            ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'batch_{batch_idx}_visualization.png')
    plt.close()


if __name__ == "__main__":

    # Your existing code
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=2, translate=(0.2, 0.2), scale=(0.99, 1.01)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Initialize the dataset
    sudoku_dataset = DynamicSudokuDataset(transform=transform)

    # Create a DataLoader with batch size 32
    data_loader = DataLoader(sudoku_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Fetch a few batches of data and plot them
    for batch_idx, (digit_images, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx}: Digit Image Shape: {digit_images.shape}, Labels Shape: {labels.shape}")

        # Plot the images
        plot_image_batch(digit_images, labels, batch_idx)

        # Stop after a few batches for the sake of demonstration
        if batch_idx >= 4:
            break

    print("Visualization complete. Check the saved PNG files for each batch.")
