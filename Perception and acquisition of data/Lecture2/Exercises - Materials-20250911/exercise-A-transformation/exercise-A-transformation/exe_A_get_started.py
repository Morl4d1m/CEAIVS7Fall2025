import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read images
img1 = cv2.imread('images/ref.jpeg')
img2 = cv2.imread('images/rot.jpeg')

# Specify the same 3 points in both images
# use MS paint, GIMP or another program to manually get the pixel positions
# this part is typically automated using image processing
pts1 = np.float32([[1828,428], # upper right corner of the book
                   [1272,770], # the dot over the 'i' in 'Multiple'
                   [810,1948]]) # the dot over the 'i' in 'Richard'
pts2 = np.float32([[2020,1346], # upper right corner of the book
                   [1460,1184], # the dot over the 'i' in 'Multiple'
                   [399,1578]]) # the dot over the 'i' in 'Richard'

# Plot corners as a sanity check
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.scatter(pts1[:,0],pts1[:,1], c='r')
plt.title("img1")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.scatter(pts2[:,0],pts2[:,1], c='r')
plt.title("img2")
plt.show()

# Calculate the affine transform M from pts2 to pts1
M = cv2.getAffineTransform(src=pts2,dst=pts1)
print(M)

# Apply to affine transform to img2 and save the results
cols,rows,_ = img2.shape
img2_t = cv2.warpAffine(img2,M,(cols,rows))
cv2.imwrite('rot_transformed.png', img2_t)
