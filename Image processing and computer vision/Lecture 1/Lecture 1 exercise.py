import cv2
import numpy as np

def convert_image_grayscale(image, mode):
    height, width = image.shape[:2]
    new_image = np.zeros((height, width), dtype=np.uint8)
    if mode == "manual":
        for i, row in enumerate(image):
            for j, col in enumerate(row):
                gray_value = int((int(col[0]) + int(col[1]) + int(col[2])) / 3)
                new_image[i, j] = gray_value
    elif mode == "matrix":
        bgr_weights = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        new_image = np.dot(image[..., :3], bgr_weights).astype(np.uint8)
    elif mode == "opencv":
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return new_image

def thresholding_algorithm(image, threshold):
    height, width = image.shape[:2]
    new_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] >= threshold:
                new_image[i, j] = 255
            else:
                new_image[i, j] = 0
    return new_image

def combine_binary_img(image1, image2, logicop):
    height, width = image1.shape[:2]
    combined_img = np.zeros((height, width), dtype=np.uint8)

    if logicop.lower() == "and":
        for i in range(height):
            for j in range(width):
                combined_img[i, j] = image1[i, j] & image2[i, j]
    elif logicop.lower() == "or":
        for i in range(height):
            for j in range(width):
                combined_img[i, j] = image1[i, j] | image2[i, j]

    return combined_img

def adjust_contrast(image, factor):
    new_image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return new_image

if __name__ == "__main__":
    # exercise 3
    img = cv2.imread(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Image processing and computer vision\Lecture 1\Exercise materials-20250903\Lion.jpg")

    img_manual = convert_image_grayscale(img, mode="manual")
    img_matrix = convert_image_grayscale(img, mode="matrix")
    img_opencv = convert_image_grayscale(img, mode="opencv")

    cv2.imshow("Image", img)
    cv2.imshow("Manual Grayscale", img_manual)
    cv2.imshow("Matrix Grayscale", img_matrix)
    cv2.imshow("OpenCV Grayscale", img_opencv)
    
    # exercise 4.1
    thresholded_img = thresholding_algorithm(img_manual, 128)

    cv2.imshow("Thresholded Image", thresholded_img)
    
    # exercise 4.2
    
    thermal_img1 = cv2.imread(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Image processing and computer vision\Lecture 1\Exercise materials-20250903\thermal1.png")
    thermal_img2 = cv2.imread(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Image processing and computer vision\Lecture 1\Exercise materials-20250903\thermal2.png")

    thermal_img1_gray = convert_image_grayscale(thermal_img1, "opencv")
    thermal_img2_gray = convert_image_grayscale(thermal_img2, "opencv")

    thresholded_thermal_img1 = thresholding_algorithm(thermal_img1_gray, 128)
    thresholded_thermal_img2 = thresholding_algorithm(thermal_img2_gray, 128)

    cv2.imshow("Thresholded Thermal Image 1", thresholded_thermal_img1)
    cv2.imshow("Thresholded Thermal Image 2", thresholded_thermal_img2)

    combined_thermal_img_and = combine_binary_img(thresholded_thermal_img1, thresholded_thermal_img2, "AND")
    combined_thermal_img_or = combine_binary_img(thresholded_thermal_img1, thresholded_thermal_img2, "OR")

    cv2.imshow("Combined AND Thermal Image", combined_thermal_img_and)
    cv2.imshow("Combined OR Thermal Image", combined_thermal_img_or)
    
    # exercise 5
    einstein_image = cv2.imread(r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Image processing and computer vision\Lecture 1\Exercise materials-20250903\Einstein.tif")
    
    einstein_image_contrast_2 = adjust_contrast(einstein_image, 2)
    einstein_image_contrast_5 = adjust_contrast(einstein_image, 5)

    cv2.imshow("Einstein Image", einstein_image)
    cv2.imshow("Einstein Image Contrast (Factor: 2)", einstein_image_contrast_2)
    cv2.imshow("Einstein Image Contrast (Factor: 5)", einstein_image_contrast_5)

    cv2.waitKey(0)
    #cv2.destroyAllWindows()