import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from torchvision.transforms import RandomApply

from src.utils import print_progress_bar


# Modified CNN architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Increased to 1024
        self.fc2 = nn.Linear(1024, 512)  # Increased hidden units
        self.fc3 = nn.Linear(512, 9)  # Output 9 classes (digits 1-9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.bn1(self.relu(self.conv1(x))))
        x = self.pool(self.bn2(self.relu(self.conv2(x))))
        x = self.pool(self.bn3(self.relu(self.conv3(x))))
        x = self.pool(self.bn4(self.relu(self.conv4(x))))  # Added conv4 block
        x = x.view(-1, 512 * 4 * 4)  # Adjust for new input size
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Function to filter out digit 0
def filter_dataset(dataset):
    indices = []
    for i in range(len(dataset)):
        if dataset[i][1] != 0:  # Exclude label 0
            indices.append(i)
    return Subset(dataset, indices)


# Add a transform to invert the colors (white digits on black background)
def invert_colors(image):
    return 1 - image


# Modified dataset loading function
def load_svhn_dataset(batch_size=32):
    def add_gaussian_noise(image, mean=0, std=0.1):
        """Inject Gaussian noise into the image."""
        noise = torch.randn(image.size()) * std + mean
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)  # Keep pixel values in range [0, 1]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(3),  # Apply small random rotations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine transformations
        transforms.ToTensor(),
        RandomApply([transforms.Lambda(add_gaussian_noise)], p=0.5),  # Apply noise with 50% probability
        # Normalize for 3-channel RGB images
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB images
    ])

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    trainset = filter_dataset(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testset = filter_dataset(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# Modified training function with additional techniques
def train_model(model, trainloader, testloader, device, epochs=20, learning_rate=0.001, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=1e-4)  # Changed to AdamW with weight decay

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = patience

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(trainloader)

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            labels = labels - 1  # Adjust labels to be 0-8 instead of 1-9

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print_progress_bar(i + 1, total_batches, prefix=f'Epoch {epoch + 1}/{epochs}', suffix=f'Loss: {loss:.4f}')

        train_accuracy = 100 * correct / total
        print(
            f'Epoch [{epoch + 1}/{epochs}] Training Loss: {running_loss / total_batches:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        test_loss, test_accuracy = evaluate_model(model, testloader, device, criterion, epoch + 1)
        print(f'Epoch [{epoch + 1}/{epochs}] Test Loss: {test_loss:.2f} Test Accuracy: {test_accuracy:.2f}%')

        scheduler.step(test_loss)

        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{epochs}] Learning Rate: {current_lr:.6f}')

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), './models/best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


# Modified evaluation function
def evaluate_model(model, testloader, device, criterion, epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    total_batches = len(testloader)

    # Prepare to store the first 16 images for debugging
    debug_images = []
    debug_predictions = []
    debug_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            labels = labels - 1  # Adjust labels to be 0-8 instead of 1-9
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect the first 16 images and their predictions
            if len(debug_images) < 16:
                debug_images.extend(images.cpu())
                debug_predictions.extend(predicted.cpu())
                debug_labels.extend(labels.cpu())

            print_progress_bar(i + 1, total_batches, prefix='Evaluating', suffix='')

    accuracy = 100 * correct / total
    plot_debug_images(debug_images[:16], debug_predictions[:16], debug_labels[:16], epoch)  # Plot the first 16 images
    return running_loss / total_batches, accuracy


def plot_debug_images(images, predictions, labels, epoch):
    save_path = f"debug_images_{epoch}.png"
    """Plot the first 16 images with their predicted and actual labels and save as an image."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0) * 0.5 + 0.5  # Invert normalization
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Pred: {predictions[i].item() + 1}, Label: {labels[i].item() + 1}")
        ax.axis('off')

    plt.tight_layout()

    # Save the plot instead of showing it
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory after saving

