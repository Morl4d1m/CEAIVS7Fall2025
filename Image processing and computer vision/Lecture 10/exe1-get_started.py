import numpy as np
import cv2
from matplotlib import pyplot as plt
import os.path

if __name__ == "__main__":
    # Read the read and left stereo image pair
    imgR = cv2.imread(os.path.join('tsukuba','scene1.row3.col3.ppm'), cv2.IMREAD_GRAYSCALE)
    imgL = cv2.imread(os.path.join('tsukuba','scene1.row3.col1.ppm'), cv2.IMREAD_GRAYSCALE)

    # Compute disparity using block matching
    stereo1 = cv2.StereoBM.create(numDisparities=16, blockSize=11)
    disparity1 = stereo1.compute(imgL,imgR)
    plt.figure(1)
    plt.title('block matching1')
    plt.imshow(disparity1,'gray')
    #plt.show()
    stereo2 = cv2.StereoBM.create(numDisparities=32, blockSize=11)
    disparity2 = stereo2.compute(imgL,imgR)
    plt.figure(2)
    plt.title('block matching2')
    plt.imshow(disparity2,'gray')
    #plt.show()
    stereo3 = cv2.StereoBM.create(numDisparities=128, blockSize=11)
    disparity3 = stereo3.compute(imgL,imgR)
    plt.figure(3)
    plt.title('block matching3')
    plt.imshow(disparity3,'gray')
    stereo4 = cv2.StereoBM.create(numDisparities=256, blockSize=11)
    disparity4 = stereo4.compute(imgL,imgR)
    plt.figure(4)
    plt.title('block matching4')
    plt.imshow(disparity4,'gray')
    plt.show()

    ## Exercise 1.a)
    # Do the exercise... INSERT YOUR OWN CODE

# 1.A:
# Increases or decreases the resolution of the disparity, with lower values matching higher resolution

# 1.B:
# Seems to remove an amount of pixel columns matching the value

# 1.C:
# See B

# 1.D:
# Then it becomes scrambled and cannot be used for human interpretation