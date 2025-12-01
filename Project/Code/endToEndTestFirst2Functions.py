import os
import subprocess
import sys
from pathlib import Path
from PIL import Image, PngImagePlugin
import concurrent.futures

inputDir = Path(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Project\SRImplementation\TSD-SR-main\imgs\subFunction1And2Test") # Remember to change this to your personal directory
PNGDir = inputDir / "pngs" # Folder name for PNG versions of images
framesDir = inputDir / "frames" # Folder that holds subfolders with each 128x128 frame
framesSRDir = inputDir / "framesSR" # Contains subfolders with each 512x512 SR'ed frame
SRDir = inputDir / "superresolvedComplete" # Contains re-assembled superresolved images

############################################################
# Function 1 — Load images from directory
############################################################
def loadImages(inputDir: Path):
    inputDir = Path(inputDir) # Sets the directory - sorry for using identical variablenaming, this I'd like to resolve
    images = [] # Array containing the image-paths

    for file in inputDir.iterdir(): # Loops through the entire directory
        if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]: # We'll only check for the most common imagetypes
            images.append(file) # Adds imagepath to array

    print(f"[INFO] Loaded {len(images)} images from {inputDir}") # Update in terminal/output
    return images # Returns array so that other functions can use it


############################################################
# Function 2 — Ensure PNG, otherwise convert
############################################################
def convertToPngIfNeeded(imgPath: Path, outputDir: Path):
    outputDir.mkdir(parents=True, exist_ok=True) # Creates the "pngs" directory

    if imgPath.suffix.lower() == ".png": # Checks wether or not the current image is png or not
        print(f"[INFO] Already PNG: {imgPath.name}") # Serial update in terminal/output
        return imgPath # Returns the original image path, as this file is already a valid PNG

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


############################################################
# Function 3 — Segment image >128px into 128×128 tiles
############################################################
def segmentImage128(img: Image.Image):
    w, h = img.size # Saves the dimensions to w and h for width and height

    if w <= 128 and h <= 128: # Checks if any of the dimensions are larger than the maximum of 128x128 pixels
        return [img] # If nothing is out of scope, the image can be used as is, and does not need to be segmented

    frames = [] # Array to keep all of the segmented frames within
    for y in range(0, h, 128): # Creates for loop iterating as long as h is larger than current y value, done in 128 pixel steps
        for x in range(0, w, 128): # Creates for loop iterating as long as h is larger than current x value, done in 128 pixel steps
            tile = img.crop((x, y, x + 128, y + 128)) # Creates segmented frame based on current x and y values, along with their respective values 128 pixels later
            frames.append(tile) # Appends the new frame to the array

    print(f"[INFO] Segmented into {len(frames)} tiles") # Terminal/output update about how many frames the image is segmented into
    return frames


############################################################
# Function 4 — Call TSD-SR model
# Before running make sure that virtual environment has been started with correct dependencies specified in the original "requirements.txt" file for TSD-SR
############################################################
def runTsdSr(
    input_dir,
    output_dir,
    tsdsr_env="tsdsr",
    tsdsr_main_dir=r"C:/Users/Christian Lykke/Documents/Skole/Aalborg Universitet/CEAIVS7/Project/SRImplementation/TSD-SR-main", # Remember to update to your own directory
    pretrained_model_path=r"C:/Users/Christian Lykke/Documents/Skole/Aalborg Universitet/CEAIVS7/Project/SRImplementation/sd3", # Remember to update to your own directory
    lora_dir="checkpoint/tsdsr",
    embedding_dir="dataset/default",
    python_exe="python",
    device="cpu"
): # Basically sets the variables for all of the parameters used in the TSD-SR model, so that you only need to type input and output directories into the function
    input_dir = Path(input_dir) # makes the input directory useable for the function
    output_dir = Path(output_dir) # Makes the output directory useable for the function
    print(f"[INFO] Running TSD-SR on folder: {input_dir}") # Terminal/output update

    command = [ # Acts as a terminal command, so everything from here is identical to how we'd usually call the TSD-SR model in terminal
        python_exe, # Tells terminal that we want to use python for running the function
        "test/test_tsdsr.py", # The testing file derived for testin the TSD-SR script by original authors
        "--pretrained_model_name_or_path", str(pretrained_model_path),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--lora_dir", lora_dir,
        "--embedding_dir", embedding_dir,
        "--device", device
    ]

    try:
        subprocess.run(
            command,
            cwd=tsdsr_main_dir,
            check=True,
            shell=True
        )
        print("[INFO] TSD-SR completed")
        return output_dir
    
    except subprocess.CalledProcessError as e:
        print("[ERROR] TSD-SR failed:")
        print(e)


############################################################
# Function 5 — Print TSD-SR metrics, which we only can do for SR datasets
############################################################
def printTsdSrMetrics(metrics):
    print("[METRICS] TSD-SR Metrics:", metrics)


############################################################
# Function 6 — ReID model on frames
############################################################
def runReIdModel(frames): # needs implementation
    print("[CALL] Re-ID model on frames...")
    fakeMetrics = {"accuracy": 0.0}  # placeholder
    return fakeMetrics


############################################################
# Function 7 — Reassemble SR tiles into full image
############################################################
def combineTiles(frames, originalSize): # Uses the number of frames and original image size (before SR) as inputs to create the "new" image
    W0, H0 = originalSize       # original size
    # Number of tiles in each dimension (tiling uses steps of 128)
    tilesX = (W0 + 127) // 128
    tilesY = (H0 + 127) // 128

    # Determine SR tile size dynamically 
    tileW_sr, tileH_sr = frames[0].size

    # Full output dimensions
    outW = tilesX * tileW_sr
    outH = tilesY * tileH_sr

    out = Image.new("RGB", (outW, outH)) # Creates a new RGB image with the dimensions calculated based on frames

    index = 0
    for ty in range(tilesY):
        for tx in range(tilesX):
            if index >= len(frames):
                break

            x = tx * tileW_sr
            y = ty * tileH_sr

            out.paste(frames[index], (x, y))
            index += 1

    print("[INFO] Superresolved image reconstructed") # Terminal update
    # Now crop to the exact 4× upscaled image size
    finalW = W0 * 4 # Since we know the TSD-SR SR's by 4, the new image must be 4 times as large as the orignial in both directions
    finalH = H0 * 4 # Since we know the TSD-SR SR's by 4, the new image must be 4 times as large as the orignial in both directions

    out = out.crop((0, 0, finalW, finalH)) # Ensures no black background is present

    print("[INFO] Cropped final image to match original aspect (4×)") # Terminal update
    return out


############################################################
# Function 8 — ReID on full superresolved image
############################################################
def runReIdOnSuperresolved(img): # Needs implementation
    print("[CALL] Re-ID on superresolved image...")
    return {"top1": 0.0}  # placeholder


############################################################
# Function 9 — Print ReID metrics
############################################################
def printReIdMetrics(metrics):
    print("[METRICS] Re-ID Metrics:", metrics)


############################################################
# Optional function - Save segmented frames to folder
############################################################
def saveSegmentedFrames(frames, baseSaveDir: Path, imageName: str):
    saveFolder = baseSaveDir / f"{imageName}_frames"
    saveFolder.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        framePath = saveFolder / f"{imageName}_frame_{i:03d}.png"
        frame.save(framePath)

    print(f"[INFO] Saved {len(frames)} segmented frames in {saveFolder}")
    return saveFolder


############################################################
# Optional function — Save SR frames into folder
############################################################
def saveSuperresolvedFrames(frames, baseSaveDir: Path, imageName: str):
    saveFolder = baseSaveDir / f"{imageName}_SR_frames"
    saveFolder.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        framePath = saveFolder / f"{imageName}_SR_frame_{i:03d}.png"
        frame.save(framePath)

    print(f"[INFO] Saved {len(frames)} SR frames -> {saveFolder}")
    return saveFolder


############################################################
# Helper function - finds all images in subfolders
############################################################
def collectAllImages(rootFolder: Path):
    rootFolder = Path(rootFolder)
    return list(rootFolder.rglob("*.png"))


############################################################
# Full Pipeline
############################################################
def processDirectory(inputDir, PNGDir, framesDir, framesSRDir, srDir):
    inputDir = Path(inputDir)
    PNGDir = Path(PNGDir)
    #framesDir = Path(framesDir)
    #framesSRDir = Path(framesSRDir)
    #srDir = Path(srDir)
    #srDir.mkdir(exist_ok=True)

    images = loadImages(inputDir)

    for imgPath in images:

        # Step 2 — Convert to PNG
        pngPath = convertToPngIfNeeded(imgPath, PNGDir)
        if pngPath is None:
            continue
        """""
        img = Image.open(pngPath)

        # Step 3 — Segment
        frames = segmentImage128(img)

        # Optional Step — Save segmentation for human inspection
        saveSegmentedFrames(frames, baseSaveDir=framesDir, imageName=pngPath.stem)

        # Step 4 — TSD-SR
        framesFolder = framesDir / f"{pngPath.stem}_frames"
        tsdsrOutputFolder = framesSRDir / f"{pngPath.stem}_SR_frames"
        tsdsrOutput = runTsdSr(input_dir=framesFolder, output_dir=tsdsrOutputFolder)
        srFrames = loadFramesFromFolder(tsdsrOutput)

        # Optional Step - Save superresolved frames for inspection
        #saveSuperresolvedFrames(srFrames, baseSaveDir=framesSRDir, imageName=pngPath.stem)

        # Step 5 — SR Metrics
        printTsdSrMetrics(metrics={"dummy": 1.0})

        # Step 6 — ReID on tiles
        reidBefore = runReIdModel(frames)

        # Step 7 — Reassemble SR tiles
        srImage = combineTiles(srFrames, img.size)

        # Save SR image
        savePath = srDir / f"{pngPath.stem}_SR.png"
        srImage.save(savePath)
        print(f"[INFO] Saved superresolved: {savePath}")

        # Step 8 — ReID on full SR image
        reidFinal = runReIdOnSuperresolved(srImage)

        # Step 9 — Print metrics
        printReIdMetrics(reidFinal)
        """
        print("-" * 50)



def loadFramesFromFolder(folder: Path):
    frames = []
    for f in sorted(folder.glob("*.png")):
        frames.append(Image.open(f))
    return frames


############################################################
# Entry point
############################################################
if __name__ == "__main__":
    processDirectory(
        inputDir=inputDir,
        PNGDir=PNGDir,
        framesDir=framesDir,
        framesSRDir=framesSRDir,
        srDir=SRDir
    )
