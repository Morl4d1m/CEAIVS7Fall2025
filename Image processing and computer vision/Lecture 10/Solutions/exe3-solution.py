import numpy as np
import cv2
import glob
import os.path
import matplotlib.pyplot as plt

# Helper function - loads all the images in the image_dir and detect checkerboards in each
# returns the object points (3D) and image points (2D) for both the left / right images
# also returns the image size
def get_image_points(image_dir, draw_points=False, board_size = (9,6), square_size=20.0):
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
    pts_obj = np.zeros((board_size[0]*board_size[1],3), np.float32)
    pts_obj[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

    # Scale the object points to correspond with the actual size of the squares
    pts_obj *= square_size

    # Arrays to store object points and image points from all the images.
    all_pts_obj = [] # 3d point in real world space
    all_pts_right_img = [] # 2d points in image plane.
    all_pts_left_img = [] # 2d points in image plane.
    images = glob.glob(image_dir)
    image_names = []

    # Loop through all the images
    for fname in images:

        # Loop image and convert to grayscale
        left_img = cv2.imread(fname)
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        #right_img = cv2.imread(fname.replace('Left','Right'))
        right_img = cv2.imread(fname.replace('left','right'))
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, board_size, None)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, board_size, None)

        # If found in both images, add object points, image points (after refining them)
        if(left_ret and right_ret):
            image_names.append(fname)
            # Refine detected image points
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            left_corners2 = cv2.cornerSubPix(left_gray,left_corners, (11,11), (-1,-1), criteria)
            right_corners2 = cv2.cornerSubPix(right_gray,right_corners, (11,11), (-1,-1), criteria)

            # Store object points and image points for calibration later
            all_pts_obj.append(pts_obj)
            all_pts_left_img.append(left_corners2)
            all_pts_right_img.append(right_corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(left_img, board_size, left_corners2, left_ret)
            cv2.drawChessboardCorners(right_img, board_size, right_corners2, right_ret)
            if(draw_points):
                plt.imshow(np.hstack([left_img, right_img]))
                plt.show()
    return all_pts_obj, all_pts_left_img, all_pts_right_img, left_gray.shape[::-1]


# Helper function for drawing lines in an image
def draw_lines(img, lines, colors):
    r,c,_ = img.shape
    scale = r/100.0 # scale line width based on image size
    for k,r in enumerate(lines):
        color = colors[k]
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color, int(scale))
    return img

# Helper function for drawing points in an image
def draw_points(img, pts, colors):
    r,c,_ = img.shape
    scale = r/100.0 # scale point size based on image size
    for k,pt in enumerate(pts):
        color = colors[k]
        print(pt)
        img = cv2.circle(img,(int(pt[0]),int(pt[1])),int(scale*3),color, int(scale))
    return img


if __name__ == "__main__":
    ## Exercise 3.a)
    # Detect checkerboards in all images and save all the points in the checkerboards
    image_dir = os.path.join('opencv-samples','left','*.jpg')
    pts_obj, pts_left, pts_right, image_size = get_image_points(image_dir, draw_points=False)

    # Calibrate both cameras seperately - often recommended
    ret_left, K_left, dist_left, _, _ = cv2.calibrateCamera(pts_obj, pts_left,
                                                            image_size, None, None)
    ret_right, K_right, dist_right, _, _ = cv2.calibrateCamera(pts_obj, pts_right,
                                                               image_size, None, None)

    # Fix the intrinsic parameters as we have already calibrated them
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER +
                       cv2.TERM_CRITERIA_EPS, 100, 1e-5)


    ret, cam1, dist1, cam2, dist2, rot, trans, E, F = cv2.stereoCalibrate(pts_obj,
                                                                          pts_left,
                                                                          pts_right,
                                                                          K_left,
                                                                          dist_left,
                                                                          K_right,
                                                                          dist_right,
                                                                          image_size,
                                                                          criteria_stereo, flags)

    # Let's try to calculate the length of the baseline
    print("baseline: ", trans)
    print(" - length: ", np.linalg.norm(trans))
    # It appears to be roughly 67 mm

    ## Exercise 3.b)
    # Try rectifying a stereo image pair
    # Start by loading the images
    left_img = cv2.imread('opencv-samples/left/left01.jpg')
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    
    right_img = cv2.imread('opencv-samples/right/right01.jpg')
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Do the actual rectification
    rot1, rot2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cam1, dist1,
                                                          cam2, dist2,
                                                          left_gray.shape[::-1], rot, trans)

    # Remap the left image based on the resulting rotation rot1 and projection matrix P1
    leftmapX, leftmapY = cv2.initUndistortRectifyMap(cam1, dist1, rot1, P1, left_gray.shape[::-1], cv2.CV_32FC1)
    left_remap = cv2.remap(left_gray, leftmapX, leftmapY, cv2.INTER_LANCZOS4)

    # Do the same for the right image
    rightmapX, rightmapY = cv2.initUndistortRectifyMap(cam2, dist2, rot2, P2, right_gray.shape[::-1], cv2.CV_32FC1)
    right_remap = cv2.remap(right_gray, rightmapX, rightmapY, cv2.INTER_LANCZOS4)


    # Plot the two original images side-by-side
    ax = plt.subplot(211)
    ax.grid(axis='y', color='red')
    ax.set_title('original')
    ax.imshow(np.hstack([left_gray,right_gray]),'gray')

    # Plot the two remapped images side-by-side
    ax = plt.subplot(212)
    ax.grid(axis='y', color='red')
    ax.set_title('remapped')
    ax.imshow(np.hstack([left_remap,right_remap]),'gray')
    plt.show()
    # The horizontal lines appears to match much better in the remapped images
    # Just like you would expect. Our calibration seems to be working!

    ## Exercise 3.c)
    # Load the images
    left_img = cv2.imread(os.path.join('opencv-samples','left','left02.jpg'))
    right_img = cv2.imread(os.path.join('opencv-samples','right','right02.jpg'))

    # Carefully (not really) manually selected points from the left image
    left_pts = np.array([[334.0, 327.0],
	                 [381.0, 308.0],
	                 [434.0, 286.0],
	                 [493.0, 262.0]])

    # Find epilines in the right image correponding to these points from the left image
    img_index = 1 # index of the image (1 or 2) containing the points, in this case imgL = 1
    linesR = cv2.computeCorrespondEpilines(left_pts.reshape(-1,1,2), img_index, F)
    linesR = linesR.reshape(-1,3)

    # Plot the points in the left image and the corresponding epilines in the right image
    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in left_pts]
    imgL_points = draw_points(left_img, left_pts, colors)
    imgR_lines = draw_lines(right_img, linesR, colors)

    ax = plt.subplot(121)
    ax.set_title('left image')
    ax.imshow(imgL_points)
    
    ax = plt.subplot(122)
    ax.set_title('right image')
    ax.imshow(imgR_lines)
    plt.show()
    # The epilines in the right image appears to correspond
    # well with the lines in the left image. Not 100% as
    # there is some noise. Perhaps the calibration could
    # improve with some more images
