from ultralytics import YOLO
import keyboard
import threading
import bettercam
from pynput.mouse import Button, Controller
import numpy as np
import cv2
import torch
import time

# Initialize the mouse controller
mouse = Controller()

# Create a camera object for capturing screenshots with color output in BGR format
camera = bettercam.create(output_color="BGR")

# Cache for circular movement calculations
cached_cos_sin = {}

def initialize_bomb_bbox(x1, y1, x2, y2):
    """Returns a numpy array for a bomb's bounding box coordinates."""
    return np.array([x1, y1, x2, y2])

def is_within_bomb(fruit_box, bomb_list):
    """Checks if any corner of a fruit's bounding box overlaps with any bomb's bounding box."""
    f_x1, f_y1, f_x2, f_y2 = fruit_box
    return any(
        (b_x1 <= f_x1 <= b_x2 and b_y1 <= f_y1 <= b_y2) or
        (b_x1 <= f_x1 <= b_x2 and b_y1 <= f_y2 <= b_y2) or
        (b_x1 <= f_x2 <= b_x2 and b_y1 <= f_y1 <= b_y2) or
        (b_x1 <= f_x2 <= b_x2 and b_y1 <= f_y2 <= b_y2)
        for b_x1, b_y1, b_x2, b_y2 in bomb_list
    )

def determine_safe_fruits(fruits, bombs):
    """Filters out fruits overlapping with bombs, marking only safe fruits for slicing."""
    return [
        (center_x, center_y)
        for center_x, center_y, width, height in fruits
        if center_y <= 1000 and not is_within_bomb(
            (center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), bombs
        )
    ]

def move_mouse(radius, num_steps):
    """Moves the mouse in a circular pattern around its current position."""
    if radius not in cached_cos_sin:
        angles = np.linspace(0, 2 * np.pi, num_steps)
        cached_cos_sin[radius] = (np.cos(angles) * radius, np.sin(angles) * radius)
    
    cos_vals, sin_vals = cached_cos_sin[radius]
    start_x, start_y = mouse.position

    for dx, dy in zip(cos_vals, sin_vals):
        mouse.position = (start_x + dx, start_y + dy)
        time.sleep(1e-6)  # Minor sleep to ensure smooth movement

def run_bot(safe_fruits):
    """Simulates mouse actions to slice safe fruits."""
    for fruit_x, fruit_y in safe_fruits:
        mouse.position = (fruit_x, fruit_y)
        mouse.press(Button.left)
        move_mouse(50, 50)
        mouse.release(Button.left)

def take_screenshot(stop_event):
    """Continuously captures screenshots and processes them for object detection."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('bestv5.pt', task='detect').to(device).eval()

    while not stop_event.is_set():
        screenshot = camera.grab(region=(0, 0, 1920, 1080))
        if screenshot is None:
            continue

        results = model(source=screenshot, iou=0.25, conf=0.7, imgsz=(640, 640))
        detected_fruits, detected_bombs = [], []

        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy()
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            label = results[0].names[int(cls)]

            if label == "bomb":
                detected_bombs.append(initialize_bomb_bbox(x1, y1, x2, y2))
            elif label == "fruit":
                detected_fruits.append((center_x, center_y, width, height))

        safe_fruits = determine_safe_fruits(detected_fruits, detected_bombs)
        run_bot(safe_fruits)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    """Main function to start the bot and listen for stop command."""
    stop_event = threading.Event()
    screenshot_thread = threading.Thread(target=take_screenshot, args=(stop_event,))
    screenshot_thread.start()

    keyboard.wait("q")  # Press 'q' to quit
    stop_event.set()
    screenshot_thread.join()
    camera.release()

if __name__ == "__main__":
    main()