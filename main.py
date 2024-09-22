from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.label import Label

import cv2
import torch
from src.grid_recognition import detect_sudoku_grid, warp_perspective, extract_sudoku_grid_and_classify
from src.train_model import DigitClassifier
from src.utils import solve_sudoku


class SudokuSolverApp(App):
    def build(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = './models/digit_classifier_svhn.pth'
        self.model = DigitClassifier().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))

        layout = BoxLayout(orientation='vertical')
        self.img = Image()
        layout.add_widget(self.img)

        button_layout = BoxLayout(size_hint=(1, 0.1))
        self.capture_button = Button(text="Capture")
        self.capture_button.bind(on_press=self.capture_image)
        self.solve_button = Button(text="Solve")
        self.solve_button.bind(on_press=self.solve_sudoku)
        button_layout.add_widget(self.capture_button)
        button_layout.add_widget(self.solve_button)
        layout.add_widget(button_layout)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)

        self.captured_image = None
        self.warped_image = None
        self.sudoku_grid = None

        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = image_texture

    def capture_image(self, instance):
        ret, frame = self.capture.read()
        if ret:
            self.captured_image = frame
            try:
                grid_contour = detect_sudoku_grid(frame)
                self.warped_image = warp_perspective(frame, grid_contour)
                buf1 = cv2.flip(self.warped_image, 0)
                buf = buf1.tostring()
                texture = Texture.create(size=(self.warped_image.shape[1], self.warped_image.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.img.texture = texture
                print("Image captured and warped successfully")
            except Exception as e:
                self.show_error("Error capturing image", str(e))

    def solve_sudoku(self, instance):
        if self.warped_image is None:
            self.show_error("Error", "Please capture an image first")
            return

        try:
            print("Extracting and classifying Sudoku grid...")
            self.sudoku_grid = extract_sudoku_grid_and_classify(self.warped_image, self.model, self.device)
            print(f"Extracted Sudoku grid:\n{self.sudoku_grid}")

            print("Solving Sudoku...")
            solved_grid = solve_sudoku(self.sudoku_grid)

            if solved_grid is not None:
                print("Sudoku solved successfully")
                self.draw_solution(solved_grid)
            else:
                raise Exception("Unable to solve the Sudoku puzzle")
        except Exception as e:
            self.show_error("Error solving Sudoku", str(e))

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

            buf1 = cv2.flip(self.warped_image, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(self.warped_image.shape[1], self.warped_image.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture
            print("Solution drawn on the image")
        except Exception as e:
            self.show_error("Error drawing solution", str(e))

    def show_error(self, title, message):
        popup = Popup(title=title, content=Label(text=message), size_hint=(0.8, 0.4))
        popup.open()

    def on_stop(self):
        self.capture.release()


if __name__ == '__main__':
    SudokuSolverApp().run()
