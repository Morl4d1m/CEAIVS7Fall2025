import os
from pathlib import Path
from PIL import Image

# Input and output folders
input_dir = Path(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Project\SRImplementation\TSD-SR-main\outputs\test\testReID")
output_dir = input_dir / "256x512Versions"
output_dir.mkdir(exist_ok=True)


os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)

    # Check that image is 512x512
    if img.size != (512, 512):
        print(f"Skipping {filename}: size {img.size} != (512, 512)")
        continue

    # Crop left half: (left, upper, right, lower)
    left_half = img.crop((0, 0, 256, 512))

    # Save result
    save_path = os.path.join(output_dir, filename)
    left_half.save(save_path)

    print(f"Saved {save_path}")

print("Done! Cropped all 512x512 images to left 256x512 halves.")
