import numpy as np
import cv2
import matplotlib.pyplot as plt # just for plotting

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
board_size = (6,9)
pts_obj = np.zeros((board_size[0]*board_size[1],3), np.float32)
pts_obj[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

# Scale the object points to correspond with the actual size of the squares
square_size = 20.0 # i.e. 40 mm
pts_obj *= square_size

# Lets save the points to a npz file
np.savez('obj-pts.npz', points=pts_obj)

# And load them again... to see how it can be done
with np.load('obj-pts.npz') as npz:
    print(list(npz.keys()))
    loaded_pts_obj = npz['points']

    # Let check how it is looking
    plt.scatter(pts_obj[:,0],pts_obj[:,1])
    plt.show()
    print(pts_obj)
    # Each point should correspond to an intersection in the checkerboard

# You do the rest..
# hint: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
