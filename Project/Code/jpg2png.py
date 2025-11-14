import os
from pathlib import Path
from PIL import Image, PngImagePlugin
import concurrent.futures

# Input and output folders
input_dir = Path(r"C:\Users\Christian Lykke\Downloads\Market-1501-v15.09.15\query")
output_dir = input_dir / "pngs"
output_dir.mkdir(exist_ok=True)

def convert_to_png(img_path: Path):
    """Convert a single JPG image to PNG while preserving metadata."""
    try:
        with Image.open(img_path) as im:
            # Extract metadata (EXIF + ICC Profile if available)
            exif_data = im.info.get("exif")
            icc_profile = im.info.get("icc_profile")

            # Create output path
            out_path = output_dir / f"{img_path.stem}.png"

            # Prepare PNG metadata container
            pnginfo = PngImagePlugin.PngInfo()

            # If EXIF exists, store it in a standard tEXt chunk
            if exif_data:
                pnginfo.add_text("EXIF", exif_data.hex())

            # Save with optional ICC and metadata
            im.save(out_path, format="PNG", icc_profile=icc_profile, pnginfo=pnginfo)

        print(f"Converted: {img_path.name}")
    except Exception as e:
        print(f"❌ Error converting {img_path.name}: {e}")

def main():
    jpg_files = list(input_dir.glob("*.jpg"))
    print(f"Found {len(jpg_files)} JPG images to convert...")

    # Use all available CPU cores for speed
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(convert_to_png, jpg_files))

    print("Conversion complete!")

if __name__ == "__main__":
    main()
