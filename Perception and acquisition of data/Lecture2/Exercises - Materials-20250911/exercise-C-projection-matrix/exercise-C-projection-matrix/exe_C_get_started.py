import numpy as np
import cv2
import glob

# Load camera calibration from earlier
# along with extrinsics for each checkerboard
# i.e. rotation and translation
with np.load('cam-cal.npz') as npz:
    print(list(npz.keys()))
    mtx = npz['projection_mat']
    dists = npz['distortion']
    rvecs = npz['rotations']
    tvecs = npz['translations']

# Load the first checkerboard image
images = glob.glob('images/*.jpeg')
img = cv2.imread(images[0])

# Identify the corresponding extrinsinc params
rotation = rvecs[0]
translation = tvecs[0]

# Project the 3D point (0, 0, 0) from the checkerboard to 2D image coords
pts_world = np.float32([[0.0, 0.0, 0.0]])
pts_img, _ = cv2.projectPoints(pts_world, rotation, translation, mtx, dists)
print(pts_img)

# Draw the projected point
pts_img = pts_img[0][0]
img = cv2.circle(img, (int(pts_img[0]),int(pts_img[1])), 5, (0,255,0), -1)
cv2.imwrite('projected_point.jpeg', img)

# You do the rest..
# hint: https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
