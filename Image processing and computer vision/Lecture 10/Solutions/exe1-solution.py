import numpy as np
import cv2
from matplotlib import pyplot as plt
import os.path

if __name__ == "__main__":
    # Read the read and left stereo image pair
    imgR = cv2.imread(os.path.join('tsukuba','scene1.row3.col3.ppm'), cv2.IMREAD_GRAYSCALE)
    imgL = cv2.imread(os.path.join('tsukuba','scene1.row3.col1.ppm'), cv2.IMREAD_GRAYSCALE)

    ## Exercise 1.a)
    # Try with smaller blocksize
    stereo = cv2.StereoBM.create(numDisparities=32, blockSize=5)
    disparity = stereo.compute(imgL,imgR)
    plt.title('small-block')
    plt.imshow(disparity,'gray')
    plt.show()
    # Smaller blocksize results in more details but also more noise.
    
    # Try with bigger blocksize
    stereo = cv2.StereoBM.create(numDisparities=32, blockSize=21)
    disparity = stereo.compute(imgL,imgR)
    plt.title('big-block')
    plt.imshow(disparity,'gray')
    plt.show()
    # Bigger blocksize results reduces noise at the cost of the finer details
    
    ## Exercise 1.b-c) 
    # Try with disparity level of 16
    stereo = cv2.StereoBM.create(numDisparities=16, blockSize=11)
    disparity = stereo.compute(imgL,imgR)
    plt.title('low-disp')
    plt.imshow(disparity,'gray')
    plt.show()
    # Objects close to the cameras disappear as these objects have a disperity higher than 16.
    # By manual inspection of the x-values of the two images we see that there is a difference
    # of roughly 30 pixels (using MS paint or whatever to get the pixel coords)


    ## Exercise 1.d
    # Calculate disparity map with swapped images
    stereo = cv2.StereoBM.create(numDisparities=32, blockSize=11)
    disparity = stereo.compute(imgR,imgL)
    plt.title('swapped disparity map')
    plt.imshow(disparity,'gray')
    plt.show()
    # The resulting disparity map pure noise. The function always expects
    # the left image as the first argument and the right image as the second one.
    # It is super easy to get them mixed up! (in my experience...)
