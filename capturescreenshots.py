import mss
import mss.tools
import time
import os
import re

# Set the time interval between screenshots (in seconds)
INTERVAL = 0.1

# Directory to save screenshots
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainingimgs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_last_screenshot_number(output_dir):
    """Finds the highest numbered screenshot file in the output directory."""
    screenshots = [f for f in os.listdir(output_dir) if re.match(r'screenshot_(\d+)\.png', f)]
    if not screenshots:
        return -1
    return max(int(re.search(r'(\d+)', s).group()) for s in screenshots)

def take_screenshot(sct, monitor, screenshot_count):
    """Takes a screenshot and saves it to the output directory."""
    screenshot = sct.grab(monitor)
    screenshot_path = os.path.join(OUTPUT_DIR, f"screenshot_{screenshot_count}.png")
    mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)

def main():
    """Main function to continuously take screenshots."""
    screenshot_count = get_last_screenshot_number(OUTPUT_DIR) + 1
    print("Starting to take screenshots. Press Ctrl+C to stop.")
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while True:
                take_screenshot(sct, monitor, screenshot_count)
                screenshot_count += 1
                time.sleep(INTERVAL)
    except KeyboardInterrupt:
        new_screenshots = screenshot_count - get_last_screenshot_number(OUTPUT_DIR) - 1
        print(f"Stopped taking screenshots. {new_screenshots} new screenshots taken.")

if __name__ == "__main__":
    main()