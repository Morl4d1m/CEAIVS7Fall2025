from picamera2 import Picamera2
import time
from libcamera import Transform
from datetime import datetime
import os

# Ensure the pi-images folder exists
image_folder = "pi-images"
os.makedirs(image_folder, exist_ok=True)

# Initialize the PiCam3 (config pipeline)
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(
    main={"size": (1920, 1080)},
    transform=Transform(hflip=1, vflip=1))  # Our configuration
picam2.configure(camera_config)
picam2.start()  # Start the camera stream

# picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})  # autofocus
if "AfMode" in picam2.camera_controls:
    print("Autofocus supported. Enabling continuous mode.")
    try:
        # Use integer 2 for Continuous (avoids enum issues in some libcamera versions)
        picam2.set_controls({"AfMode": 2})
        print("Continuous autofocus enabled.")
    except Exception as e:
        print(f"Failed to set autofocus: {e}")
else:
    print("Autofocus not supported/detected. Check camera connection and try manual focus.")

index = 0

# While loop
while True:
    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(image_folder, f"image_{timestamp}.jpg")

    # Capture the image
    picam2.capture_file(filename)

    index += 1
    print(f"Billede nr {index} taget: {filename}")
    print("15 sekunder til næste billede ...")
    time.sleep(15)
    if index == 100:  # Max 10 billeder
        print(f"{index} billeder taget, slukker..")
        break
