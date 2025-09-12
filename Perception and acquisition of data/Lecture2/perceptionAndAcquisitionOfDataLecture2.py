import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read images
img1 = cv2.imread(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\perception and acquisition of data\Lecture2\imagesA\ref.jpeg", cv2.IMREAD_COLOR)
img2 = cv2.imread(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\perception and acquisition of data\Lecture2\imagesA\rot.jpeg", cv2.IMREAD_COLOR)
img3 = cv2.imread(r"c:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\perception and acquisition of data\Lecture2\imagesA\rotzoom.jpeg",cv2.IMREAD_COLOR)

# Specify the same 3 points in both images
# use MS paint, GIMP or another program to manually get the pixel positions
# this part is typically automated using image processing
pts1 = np.float32([[1828,428], # upper right corner of the book
                   [1272,770], # the dot over the 'i' in 'Multiple'
                   [810,1948]]) # the dot over the 'i' in 'Richard'
pts2 = np.float32([[2020,1346], # upper right corner of the book
                   [1460,1184], # the dot over the 'i' in 'Multiple'
                   [399,1578]]) # the dot over the 'i' in 'Richard'
pts3 = np.float32([[691,1291],
                   [594,2065],
                   [1319,3351]])


# Plot corners as a sanity check
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.scatter(pts1[:,0],pts1[:,1], c='r')
plt.title("img1")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.scatter(pts2[:,0],pts2[:,1], c='r')
plt.title("img2")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.scatter(pts3[:,0],pts3[:,1], c='r')
plt.title("img3")
plt.show()

### Exercise 1:
# Calculate the affine transform M from pts2 to pts1
M = cv2.getAffineTransform(src=pts3,dst=pts1)
print(M)


### Exercise 2:
# Apply to affine transform to img2 and save the results
cols,rows,_ = img3.shape
img3_t = cv2.warpAffine(img3,M,(cols,rows))
#cv2.imwrite(r"c:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\perception and acquisition of data\Lecture2\imagesA\rotzoom_transformed.png", img3_t)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.title("img3")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img3_t, cv2.COLOR_BGR2RGB))
plt.title("img3_t")
plt.show()


### Exercise 3:
img4 = cv2.imread(r"c:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\perception and acquisition of data\Lecture2\imagesA\persp2.jpeg",cv2.IMREAD_COLOR)
pts4 = np.float32([[2413,2367],
                   [2500,1757],
                   [2118,552]])

M2 = cv2.getAffineTransform(src=pts4,dst=pts1)
print(M2)
cols,rows,_=img4.shape
img4_t=cv2.warpAffine(img4,M2,(cols,rows))
#cv2.imwrite(r"c:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\perception and acquisition of data\Lecture2\imagesA\persp1Transformed.jpeg",cv2.IMREAD_COLOR)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
plt.title("img4")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img4_t, cv2.COLOR_BGR2RGB))
plt.title("img4_t")
plt.show()


### Exercise 4
pts1 = np.append(pts1, [[1854,2174]], axis=0) 
pts2 = np.append(pts4, [[1312,965]], axis=0)

#Opencv requires float32
pts1 = pts1.astype(np.float32)
pts4 = pts4.astype(np.float32)

perspM = cv2.getPerspectiveTransform(src=pts4,dst=pts1)
print("Exercise 4")
print(perspM)
cols,rows,_=img4.shape
perspT = cv2.warpPerspective(img4,perspM,(cols,rows))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
plt.title("img4")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(perspT, cv2.COLOR_BGR2RGB))
plt.title("img4_t")
plt.show()