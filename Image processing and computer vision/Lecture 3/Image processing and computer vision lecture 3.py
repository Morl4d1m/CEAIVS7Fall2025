import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def main(argv):
    #exercise1()
    exercise3()

    return 0


def exercise1():
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    
    """ # Automated version where you should get a prompt in the command line to input your image path
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    # Load the image
    src = cv.imread(argv[0], cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1"""
    
    #Hardcoded image path:
    src = cv.imread(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Image processing and computer vision\Lecture 3\IMG_4978.JPG", cv.IMREAD_COLOR)

    srcGaussed = cv.GaussianBlur(src, (3, 3), 0)
    
    
    gray = cv.cvtColor(srcGaussed, cv.COLOR_BGR2GRAY)
    coloured = srcGaussed.copy()
    
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    
    grad_x_coloured = cv.Sobel(coloured, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(coloured,ddepth,0,1)
    grad_y_coloured = cv.Sobel(coloured, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    abs_grad_x_coloured = cv.convertScaleAbs(grad_x_coloured)
    abs_grad_y_coloured = cv.convertScaleAbs(grad_y_coloured)
    
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    grad_coloured = cv.addWeighted(abs_grad_x_coloured, 0.5, abs_grad_y_coloured, 0.5, 0)
    
    
    #cv.imshow(window_name, grad) #Fullscale image display
    cv.waitKey(0)
    
    #Adjustable image scale display
    plt.figure(figsize=(8, 6))  # size in inches
    plt.imshow(grad, cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(8, 6))  # size in inches
    plt.imshow(cv.cvtColor(grad_coloured, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    srcCannied = cv.Canny(src,50,150)
    plt.figure(figsize=(8, 6))  # size in inches
    plt.imshow(srcCannied, cmap='gray')
    plt.axis('off')
    plt.show()

#Exercise 2
def grassfire(binary, neighbors=[(-1,0), (1,0), (0,-1), (0,1)], pause=0.01):
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    fig, ax = plt.subplots()

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 1 and labels[y, x] == 0:
                # Start a new fire
                current_label += 1
                q = deque()
                q.append((y, x))
                labels[y, x] = current_label
                while q:
                    # Current fire front (all burning this step)
                    fire_front = list(q)
                    next_q = deque()

                    # --- Visualization step ---
                    ax.clear()
                    vis = np.copy(labels)
                    ax.imshow(vis, cmap="tab20", vmin=-1, vmax=20)
                    ax.set_title(f"Blob {current_label}: spreading fire")
                    plt.pause(pause)

                    # Expand fire
                    for (cy, cx) in fire_front:
                        for dy, dx in neighbors:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                # Visualize which pixels we are checking
                                old = vis[ny, nx]
                                vis[ny, nx] = 8
                                ax.imshow(vis, cmap="tab20", vmin=-1, vmax=20)
                                ax.set_title(f"Blob {current_label}: spreading fire")
                                vis[ny, nx] = old
                                plt.pause(pause)
                                if binary[ny, nx] == 1 and labels[ny, nx] == 0:
                                    labels[ny, nx] = current_label
                                    vis[ny, nx] = current_label
                                    next_q.append((ny, nx))
                    q = next_q
    plt.show()
    return labels

#Exercise 3
def exercise3():
    #Hardcoded image path:
    src = cv.imread(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Image processing and computer vision\Lecture 3\shapes.png", cv.IMREAD_COLOR)
    # Show the original
    plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.show()

    # Convert to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Threshold to get binary image (foreground=255, background=0)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # Connected components
    num_labels, labels = cv.connectedComponents(binary)

    print(f"Number of connected components (including background): {num_labels}")

    # Visualize the labeled image
    plt.imshow(labels, cmap="tab20")
    plt.title("Connected Components")
    plt.show()
    



if __name__ == "__main__":
    main(sys.argv[1:])
    binary = np.array([
        [0,0,0,1,1,1],
        [0,0,0,0,1,0],
        [0,0,0,0,1,1],
        [0,0,1,1,0,0],
        [0,0,1,1,0,0],
        [0,0,1,1,0,0]
    ], dtype=np.uint8)

    plt.imshow(binary, cmap="gray")
    plt.title("Input Binary Image")
    plt.show()

    neighbors=[(-1,0), (1,0), (0,-1), (0,1)]
    pause = 0.1
    labels = grassfire(binary, neighbors, pause)

    plt.imshow(labels, cmap="tab20")
    plt.title("Final Connected Components")
    plt.show()
