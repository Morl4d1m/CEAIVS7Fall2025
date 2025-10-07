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
    # Let's try to calculate the disparity and plot it
    stereo = cv2.StereoBM.create(numDisparities=256, blockSize=21)
    disparity = stereo.compute(left, right).astype(np.float32)

    plt.imshow(disparity,'jet')
    plt.title("disparity")
    plt.show()

    ## Exercise 2.b)
    # Let's try to calculate the Q matrix using OpenCV's stereoRectify function
    # Note: this function tries to rectify the image pairs but the the images
    # we are using are already rectifited so we only use it for calculating the Q matrix
    image_size = left.shape[::-1]
    R1, R2, P1, P2, Q_cv, roi1, roi2 = cv2.stereoRectify(cam1, dist1,
                                                         cam2, dist2,
                                                         image_size, rotationMatrix, trans,
                                                         flags=0)
    # note: flags=0 to disable the default behaviour of enforcing cx1 = cx2
    
    # We can also try to directly calculate the Q matrix:
    Q_mat = np.array([[1, 0, 0, -cx1],
                      [0, 1, 0, -cy],
                      [0, 0, 0, f],
                      [0, 0, -(1/tx), (cx1-cx2)/tx]])

    # By printing them we see that they are identical
    print(Q_mat)
    print(Q_cv)

    ## Exercise 2.c)
    # Reproject disparity to 3D points using Q matrix
    points = cv2.reprojectImageTo3D(disparity, Q_mat)

    ## Exercise 2.d)
    # Extract z-values
    depth_map = points[:,:,2].astype(np.float32)

    # Clip the resulting z-values to a reasonably range
    depth_max = 1500.0
    depth_map[depth_map > depth_max] = 0.0
    depth_map[depth_map < 0] = 0.0

    plt.imshow(depth_map,'jet')
    plt.title("depth")
    plt.show()
    # From the depth map is looks like the top of the
    # straw hat is roughly 400 milimeters away from the camera
    # We also observe higher depth values for the background
    # compared to lower depth values for the foreground, as expected!
