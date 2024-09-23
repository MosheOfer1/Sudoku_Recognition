from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2


class QuadrilateralDetectorApp(App):

    def build(self):
        self.image = Image()
        self.button = Button(text="Detect Quadrilateral", size_hint=(1, 0.1))
        self.button.bind(on_press=self.detect_quadrilateral)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image)
        layout.add_widget(self.button)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def detect_quadrilateral(self, instance):
        ret, frame = self.capture.read()
        if ret:
            result = self.find_largest_quadrilateral(frame)
            buf = cv2.flip(result, 0).tostring()
            texture = Texture.create(size=(result.shape[1], result.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def find_largest_quadrilateral(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables
        largest_area = 0
        largest_quad = None

        # Iterate through contours
        for contour in contours:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour has 4 corners
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > largest_area:
                    largest_area = area
                    largest_quad = approx

        # Draw the largest quadrilateral on the image
        if largest_quad is not None:
            cv2.drawContours(image, [largest_quad], 0, (0, 255, 0), 3)

        return image

    def on_stop(self):
        self.capture.release()


if __name__ == '__main__':
    QuadrilateralDetectorApp().run()
