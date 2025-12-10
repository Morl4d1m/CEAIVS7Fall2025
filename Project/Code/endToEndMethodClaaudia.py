#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from PIL import Image, PngImagePlugin

# ============================================================
# CONFIGURATION UPDATE ONLY THESE PATHS IF NEEDED
# ============================================================
HOME = Path("/ceph/home/student.aau.dk/zy15zz")

TSD_SR_DIR = HOME / "TSD-SR-main"
PRETRAINED_MODEL = HOME / "sd3"
VENV_DIR = HOME / "tsdsr_venv"

# INPUT LOCATION (CHANGE THIS IF YOU WANT)
DATA_ROOT = TSD_SR_DIR / "imgs" / "Market-1501-v15.09.15" / "query"

# OUTPUT SUBFOLDERS
PNG_DIR = DATA_ROOT / "pngs"
FRAMES_DIR = DATA_ROOT / "frames"
FRAMES_SR_DIR = DATA_ROOT / "framesSR"
SR_COMPLETE_DIR = DATA_ROOT / "superresolvedComplete"

LORA_DIR = "checkpoint/tsdsr-mse"
EMBED_DIR = "dataset/default"

# ============================================================
# Helpers
# ============================================================

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def loadImages(inputDir: Path):
    inputDir = Path(inputDir) # Sets the directory - sorry for using identical variablenaming, this I'd like to resolve
    images = [] # Array containing the image-paths

    for file in inputDir.iterdir(): # Loops through the entire directory
        if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]: # We'll only check for the most common imagetypes
            images.append(file) # Adds imagepath to array

    print(f"[INFO] Loaded {len(images)} images from {inputDir}") # Update in terminal/output
    return images # Returns array so that other functions can use it


def convertToPngIfNeeded(imgPath: Path, outputDir: Path):
    outputDir.mkdir(parents=True, exist_ok=True) # Creates the "pngs" directory

    if imgPath.suffix.lower() == ".png": # Checks wether or not the current image is png or not
        print(f"[INFO] Already PNG: {imgPath.name}") # Serial update in terminal/output
        #return imgPath # Returns the original image path, as this file is already a valid PNG

    try:
        with Image.open(imgPath) as im: # Uses lazy loading until a process (such as saving) actually begins
            exifData = im.info.get("exif") # Gets exif data from the metadata if present
            iccProfile = im.info.get("icc_profile") # Gets icc profile from the metadata if present

            outPath = outputDir / f"{imgPath.stem}.png" # Creates file address within directory where the PNG should be saved
            pngInfo = PngImagePlugin.PngInfo() # Creates an "array" that contains data about the image

            if exifData: # If exifdata was present for the image, it can now be added to the new PNG version
                pngInfo.add_text("EXIF", exifData.hex()) # adds the exif data as hexadecimal

            im.save(outPath, format="PNG", icc_profile=iccProfile, pnginfo=pngInfo) # Actually saves the image to the filepath with the image data

        print(f"[INFO] Converted to PNG: {imgPath.name}") # Serial update in terminal/output
        return outPath # Returns the path for the new image to be used in other functions
    except Exception as e:
        print(f"[ERROR] Failed PNG conversion for {imgPath.name}: {e}") # Error update if any image should be unavailable for conversion (haven't tried it yet, therefore nothing has been done other than error message)
        return None


def segmentImage128(img: Image.Image):
    w, h = img.size # Saves the dimensions to w and h for width and height

    if w <= 128 and h <= 128: # Checks if any of the dimensions are larger than the maximum of 128x128 pixels
        return [img] # If nothing is out of scope, the image can be used as is, and does not need to be segmented

    frames = [] # List to keep all of the segmented frames within
    for y in range(0, h, 128): # Creates for loop iterating as long as h is larger than current y value, done in 128 pixel steps
        for x in range(0, w, 128): # Creates for loop iterating as long as h is larger than current x value, done in 128 pixel steps
            frame = img.crop((x, y, x + 128, y + 128)) # Creates segmented frame based on current x and y values, along with their respective values 128 pixels later
            frames.append(frame) # Appends the new frame to the list

    print(f"[INFO] Segmented into {len(frames)} frames") # Terminal/output update about how many frames the image is segmented into
    return frames


def saveSegmentedFrames(frames, baseSaveDir: Path, imageName: str):
    saveFolder = baseSaveDir #/ f"{imageName}_frames"
    saveFolder.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        framePath = saveFolder / f"{imageName}_frame_{i:03d}.png"
        frame.save(framePath)

    print(f"[INFO] Saved {len(frames)} segmented frames in {saveFolder}")
    return saveFolder


def load_tiles(folder: Path):
    return [Image.open(f) for f in sorted(folder.glob("*.png"))]


# ============================================================
# RUN TSD-SR INSIDE SINGULARITY
# ============================================================

def runTsdSr(input_folder: Path, output_folder: Path):
    safe_mkdir(output_folder)
    print(f"[INFO] Running TSD-SR on frames: {input_folder}")

    # Build bash command (MUST USE bash -lc)
    cmd = (
        "bash -lc '"
        f"source /scratch/tsdsr_venv/bin/activate && "
        f"python3 test/test_tsdsr.py "
        f"--pretrained_model_name_or_path {PRETRAINED_MODEL} "
        f"--input_dir {input_folder} "
        f"--output_dir {output_folder} "
        f"--lora_dir {LORA_DIR} "
        f"--embedding_dir {EMBED_DIR} "
        f"--align_method adain "
        f"--device cuda"
        "'"
    )

    try:
        subprocess.run(
            cmd,
            cwd=TSD_SR_DIR,
            shell=True,
            check=True
        )
        print("[INFO] TSD-SR finished.")
        return output_folder

    except subprocess.CalledProcessError as e:
        print("[ERROR] TSD-SR FAILED:")
        print(e)
        return None


# ============================================================
# REASSEMBLE FULL SR IMAGE
# ============================================================

def combineframes(frames, originalSize, index): # Uses the number of frames and original image size (before SR) as inputs to create the "new" image
    W0, H0 = originalSize       # original size
    # Number of frames in each dimension (tiling uses steps of 128)
    framesX = (W0 + 127) // 128
    framesY = (H0 + 127) // 128

    # Determine SR frame size dynamically 
    #frameW_sr, frameH_sr = frames[index].size
    # Determine SR frame size using the FIRST frame, opened safely
    with Image.open(frames[index]) as tmp:
        frameW_sr, frameH_sr = tmp.size

    # Full output dimensions
    outW = framesX * frameW_sr
    outH = framesY * frameH_sr

    out = Image.new("RGB", (outW, outH)) # Creates a new RGB image with the dimensions calculated based on frames

    
    for ty in range(framesY):
        for tx in range(framesX):
            if index >= len(frames):
                break

            x = tx * frameW_sr
            y = ty * frameH_sr

            #out.paste(frames[index], (x, y))
            # Open the needed frame *only right here* and close immediately
            with Image.open(frames[index]) as frame:
                out.paste(frame, (x, y))
            index += 1

    print("[INFO] Superresolved image reconstructed") # Terminal update
    # Now crop to the exact 4x upscaled image size
    finalW = W0 * 4 # Since we know the TSD-SR SR's by 4, the new image must be 4 times as large as the orignial in both directions
    finalH = H0 * 4 # Since we know the TSD-SR SR's by 4, the new image must be 4 times as large as the orignial in both directions

    out = out.crop((0, 0, finalW, finalH)) # Ensures no black background is present

    print("[INFO] Cropped final image to match original aspect (4x)") # Terminal update
    return out, index


# ============================================================
# MAIN PROCESS
# ============================================================

def processDirectory(inputDir, PNGDir, framesDir, framesSRDir, srDir):
    inputDir = Path(inputDir)
    PNGDir = Path(PNGDir)
    framesDir = Path(framesDir)
    framesSRDir = Path(framesSRDir)
    srDir = Path(srDir)
    srDir.mkdir(exist_ok=True)

    images = loadImages(inputDir)
    index=0

    for imgPath in images:

        # Step 2 - Convert to PNG
        pngPath = convertToPngIfNeeded(imgPath, PNGDir)
        if pngPath is None:
            continue

        img = Image.open(pngPath)

        # Step 3 - Segment
        frames = segmentImage128(img)

        # Optional Step - Save segmentation for human inspection
        saveSegmentedFrames(frames, baseSaveDir=framesDir, imageName=pngPath.stem)
        # Step 4 - TSD-SR
        framesFolder = framesDir #/ f"{pngPath.stem}_frames"
        tsdsrOutputFolder = framesSRDir #/ f"{pngPath.stem}_SR_frames"
    
    tsdsrOutput = runTsdSr(input_dir=framesFolder, output_dir=tsdsrOutputFolder)
    srFrames = loadFramesFromFolder(tsdsrOutput)

    for imgPath in images:
        pngPath = PNGDir / f"{imgPath.stem}.png"
        img = Image.open(pngPath)
        
        # Step 7 - Reassemble SR frames
        srImage, index = combineframes(srFrames, img.size, index=index)

        # Save SR image
        savePath = srDir / f"{pngPath.stem}_SR.png"
        srImage.save(savePath)
        print(f"[INFO] Saved superresolved: {savePath}")
        
        print("-" * 50)
    print("Done processing, your images are ready for Re-ID")


def loadFramesFromFolder(folder: Path):
    frames = sorted(folder.glob("*_frame_*.png"))
    print(f"[INFO] Found {len(frames)} SR frame paths")

    return frames
# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    processDirectory(
        DATA_ROOT,
        PNG_DIR,
        FRAMES_DIR,
        FRAMES_SR_DIR,
        SR_COMPLETE_DIR
    )