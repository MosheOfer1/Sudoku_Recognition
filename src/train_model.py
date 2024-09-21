import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from src.utils import print_progress_bar


# Define a deeper CNN architecture with batch normalization
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # 10 output classes (digits 0-9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.bn1(self.relu(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(self.bn2(self.relu(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(self.bn3(self.relu(self.conv3(x))))  # 8x8 -> 4x4
        x = x.view(-1, 256 * 4 * 4)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Load SVHN dataset with data augmentation
def load_svhn_dataset(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),  # Randomly rotate by up to 10 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly jitter colors
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# Training function with learning rate scheduler and early stopping
def train_model(model, trainloader, testloader, device, epochs=10, learning_rate=0.001, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler (verbose removed, print manually)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = patience  # Patience for early stopping

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(trainloader)

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            print_progress_bar(i + 1, total_batches, prefix=f'Epoch {epoch + 1}/{epochs}', suffix=f'Loss: {loss:.4f}')

        train_accuracy = 100 * correct / total
        print(
            f'Epoch [{epoch + 1}/{epochs}] Training Loss: {running_loss / total_batches:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Evaluate on test data
        test_loss, test_accuracy = evaluate_model(model, testloader, device, criterion)
        print(f'Epoch [{epoch + 1}/{epochs}] Test Accuracy: {test_accuracy:.2f}%')

        # Adjust learning rate based on test loss
        scheduler.step(test_loss)

        # Manually print the learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{epochs}] Learning Rate: {current_lr:.6f}')

        # Early Stopping logic
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), './best_model.pth')  # Save the best model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


# Evaluate the model with progress bar
def evaluate_model(model, testloader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    total_batches = len(testloader)

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar for evaluation
            print_progress_bar(i + 1, total_batches, prefix='Evaluating', suffix='')

    accuracy = 100 * correct / total
    return running_loss / total_batches, accuracy
