from picamera2 import Picamera2
import time
from datetime import datetime
import os

# Ensure the pi-images folder exists
image_folder = "pi-images"
os.makedirs(image_folder, exist_ok=True)

# Initialize the PiCam3
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.start()  # Start the camera stream

index = 0

# Infinite loop to capture images every 30 seconds
while True:
    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(image_folder, f"image_{timestamp}.jpg")

    # Capture the image
    picam2.capture_file(filename)

    print(f"Billede taget: {filename}")
    print("15 sekunder til næste billede ...")
    # Wait 30 seconds before next capture
    index += 1
    time.sleep(15)
    if index == 10:  # Max 10 billeder
        print(f"{index} billeder taget, slukker..")
        break
