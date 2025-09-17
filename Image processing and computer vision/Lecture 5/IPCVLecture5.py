import cv2
import numpy as np
import os
import glob

# === SETTINGS ===
dataset_path = r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Image processing and computer vision\Lecture 5\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test016" 
video_path = r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Image processing and computer vision\Lecture 5\Exercise materials-20250915\slow_traffic_small.mp4"
use_resize = True
resize_scale = 2  # Resize for ease of viewing (for us)

# === LOAD IMAGES ===
# Creates a "list" of all the files ending in .tif within the given directory. 
# Glob is a pattern matching technique which locates file or directory based on type. 
# This line also sorts the list, ensuring correct order based on name
image_files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))
# Converts all the files found by glob to cv2/numpy array images of grayscale values in a list
# Also contains a for loop looping through all files f in the "image_files" list
frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files] 
 
if use_resize: # Resizes the image if true
    frames = [cv2.resize(f, (0, 0), fx=resize_scale, fy=resize_scale) for f in frames]

# Convert to numpy array for easier math
frames_np = np.array(frames)

# === METHOD 1: First frame as background ===
background_first = frames[0]

# === METHOD 2: Average background ===
# Takes all images within the dataset and calculates the mean, and axis ensures that it is done on a pixel basis 
# The final "astype" ensures that the datatype is converted to uint8, as np.mean returns a float, which cannot be used by opencv
background_avg = np.mean(frames_np, axis=0).astype(np.uint8)

# === METHOD 3: MOG2 Background Subtraction ===
# Creates an adaptive background using a mix of gaussians (MOG) algorithm
# The history parameter accounts for how many previous images are used to create the MOG background, and varThreshold is used to detect variances in the background
# lower varThreshold values yields a more sensitive algorithm, and higher values yield a more robust algorithm
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)

# === LOOP THROUGH FRAMES ===
for i, frame in enumerate(frames):
    # Method 1
    diff_first = cv2.absdiff(frame, background_first) # Returns the absolute difference between the current frame and the background chosen. 
    _, mask_first = cv2.threshold(diff_first, 30, 255, cv2.THRESH_BINARY) # Takes the difference values from diff_first, and marks all values above 30 with a value corresponding to maxval (255 in this case). Thresh_binary ensures all other values are set to 0

    # Method 2
    diff_avg = cv2.absdiff(frame, background_avg)
    _, mask_avg = cv2.threshold(diff_avg, 30, 255, cv2.THRESH_BINARY)

    # Method 3
    mask_mog2 = fgbg.apply(frame) # Applies the MOG mask for each frame

    # Color output: highlight differences in red
    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame_first = frame_color.copy()
    frame_first[mask_first > 0] = [0, 0, 255] # Sets all values larger than 0 from the mask_first image to be red, matching 255 in BGR formatting

    frame_avg = frame_color.copy()
    frame_avg[mask_avg > 0] = [0, 0, 255] # See line 57

    frame_mog2 = frame_color.copy()
    frame_mog2[mask_mog2 > 0] = [0, 0, 255] # See line 57

    # Show results
    cv2.imshow("Original", frame)
    cv2.imshow("First Frame Background", frame_first)
    cv2.imshow("Average Background", frame_avg)
    cv2.imshow("MOG2", frame_mog2)

    key = cv2.waitKey(50)
    if key == 27:  # ESC to quit
        break

# === OPTICAL FLOW FUNCTIONS ===
def compute_good_features(video_path): # Creates a function that can be used on any future video paths
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    while True:
        ret, frame = cap.read() # Divides the video into individual frames
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts each frame into grayscale

        # Detect good features (Shi-Tomasi corners)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7) # Self-explanatory
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1) # Also self-explanatory when hovering "circle"

        cv2.imshow("Good Features to Track", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release() # Frees computational resources


def compute_lucas_kanade(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    # Params for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Params for Lucas-Kanade optical flow
    # winSize = size of search window at each pyramid level
    # maxLevel = number of pyramid layers
    # criteria = stopping criteria (either 10 iterations or epsilon < 0.03)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read first frame, detect initial points to track
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Mask image for drawing motion vectors (lines)
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw motion vectors: line from old position to new position
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            # Overlay lines (mask) on current frame
            output = cv2.add(frame, mask)
            cv2.imshow("Lucas-Kanade Optical Flow", output)

        # Update reference frame and points for next iteration
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if p1 is not None else None

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()


# === RUN OPTICAL FLOW ANALYSIS ===
compute_good_features(video_path)
compute_lucas_kanade(video_path)

#cv2.destroyAllWindows()
