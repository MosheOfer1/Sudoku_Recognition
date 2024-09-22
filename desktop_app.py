import sys
import cv2
import torch
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, \
    QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from src.grid_recognition import detect_sudoku_grid, warp_perspective, extract_sudoku_grid_and_classify
from src.train_model import DigitClassifier
from src.utils import solve_sudoku

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './models/digit_classifier_svhn.pth'

try:
    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()


class SudokuSolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sudoku Solver")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.button_layout = QHBoxLayout()
        self.capture_button = QPushButton("Capture")
        self.capture_button.clicked.connect(self.capture_image)
        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.solve_sudoku)
        self.button_layout.addWidget(self.capture_button)
        self.button_layout.addWidget(self.solve_button)
        self.layout.addLayout(self.button_layout)

        self.streaming_mode = True  # Flag to track streaming mode
        self.captured_image = None
        self.warped_image = None
        self.sudoku_grid = None

        try:
            stream_url = 'http://10.100.102.5:8080/video'  # Adjust the IP address
            self.camera = cv2.VideoCapture(stream_url)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.show_error_message(f"Error initializing camera: {e}")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        if self.streaming_mode:  # Update frame only if in streaming mode
            try:
                ret, frame = self.camera.read()
                if not ret:
                    raise Exception("Failed to capture frame")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                grid_contour = detect_sudoku_grid(frame)
                if grid_contour is not None:
                    cv2.drawContours(frame, [grid_contour], -1, (0, 255, 0), 2)

                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
            except Exception as e:
                print(f"Error updating frame: {e}")

    def capture_image(self):
        try:
            if self.streaming_mode:  # If in streaming mode, capture the image and stop streaming
                ret, frame = self.camera.read()
                if not ret:
                    raise Exception("Failed to capture image")

                self.captured_image = frame
                grid_contour = detect_sudoku_grid(frame)
                if grid_contour is not None:
                    colorful_pic, self.warped_image = warp_perspective(frame, grid_contour)

                    h, w, ch = colorful_pic.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(colorful_pic.data, w, h, bytes_per_line, QImage.Format_BGR888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
                    print("Image captured and warped successfully")

                self.streaming_mode = False  # Switch to showing the captured image

            else:  # If already showing the captured image, resume streaming
                self.streaming_mode = True  # Resume streaming mode
                self.timer.start(30)

        except Exception as e:
            print(f"Error capturing image: {e}")
            self.show_error_message(f"Error capturing image: {e}")

    def solve_sudoku(self):
        try:
            if self.warped_image is None:
                raise Exception("Please capture an image first")

            print("Extracting and classifying Sudoku grid...")
            self.sudoku_grid = extract_sudoku_grid_and_classify(self.warped_image, model, device)
            print(f"Extracted Sudoku grid:\n{self.sudoku_grid}")

            print("Solving Sudoku...")
            solved_grid = solve_sudoku(self.sudoku_grid)

            if solved_grid is not None:
                print("Sudoku solved successfully")
                self.draw_solution(solved_grid)
            else:
                raise Exception("Unable to solve the Sudoku puzzle")
        except Exception as e:
            print(f"Error solving Sudoku: {e}")
            self.show_error_message(f"Error solving Sudoku: {e}")

    def draw_solution(self, solved_grid):
        try:
            cell_height, cell_width = self.warped_image.shape[:2]
            cell_height //= 9
            cell_width //= 9

            for i in range(9):
                for j in range(9):
                    if self.sudoku_grid[i][j] == 0:
                        x = j * cell_width + cell_width // 2
                        y = i * cell_height + cell_height // 2
                        cv2.putText(self.warped_image, str(solved_grid[i][j]), (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print("Solution drawn on the image")
        except Exception as e:
            print(f"Error drawing solution: {e}")
            self.show_error_message(f"Error drawing solution: {e}")

    def show_error_message(self, message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setText("Error")
        error_box.setInformativeText(message)
        error_box.setWindowTitle("Error")
        error_box.exec_()

    def closeEvent(self, event):
        print("Closing application...")
        if self.camera.isOpened():
            self.camera.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SudokuSolverApp()
    window.show()
    sys.exit(app.exec_())
