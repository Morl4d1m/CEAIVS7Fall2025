import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path


if __name__ == "__main__":
    # Read the images
    left = cv2.imread(os.path.join('sword1','im0.png'), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(os.path.join('sword1','im1.png'), cv2.IMREAD_GRAYSCALE)

    # The intrinsic parameters of the two cameras are provided
    # we assume no lens distortion, i.e. dist coeffs = 0
    
    # focal length for both cameras
    f = 6872.874

    # center x-coordinates
    cx1 = 1329.49 # camera 1
    cx2 = 1623.46 # camera 2

    # center y-coordinates for both cameras
    cy = 954.485
    
    # baseline, i.e. distance between the cameras
    tx = -174.724
    
    # Setup the camera matrix 1
    cam1 = np.array([[f, 0, cx1],
                     [0, f, cy],
                     [0, 0, 1]])
    dist1 = np.array([0,0,0,0,0])

    # Setup the camera matrix 2
    cam2 = np.array([[f, 0, 1623.46],
                     [0, f, cy],
                     [0, 0, 1]])
    dist2 = np.array([0,0,0,0,0])

    # Prepare the extrinsic parameters
    rotationMatrix = np.eye(3) # we assume no rotation
    trans = np.array([tx, 0.0, 0.0])

    ## Exercise 2.a)
    plt.figure(1)
    plt.imshow(left)
    plt.figure(2)
    plt.imshow(right)

    stereo1 = cv2.StereoBM.create(numDisparities=256,blockSize=21)
    disparity1 = stereo1.compute(left=left, right=right)
    plt.figure(3)
    plt.imshow(disparity1,'gray')

    ## Exercise 2.b:
    q=np.array([[1,0,0,-cx1],
                [0,1,0,-cy],
                [0,0,0,f],
                [0,0,-1/tx,(cx1-cx2)/tx]])
    print(q)

    ## Exercise 2.C:
    threeDImg=cv2.reprojectImageTo3D(disparity1,q) # HOW CAN THIS BE A MASTERS EXERCISE!

    ## Exercise 2.D
    depth=threeDImg[:,:,2].astype(np.float32)
    maxDepth=1500
    depth[depth > maxDepth] = 0.0
    depth[depth < 0] = 0.0

    plt.figure(4)
    plt.imshow(depth, 'jet')
    plt.show()
