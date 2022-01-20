import cv2
import matplotlib.pyplot as plt
import numpy as np

def edge_enhancement(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find contours - write black over all small contours
    letter = morph.copy()
    cntrs = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    for c in cntrs:
        area = cv2.contourArea(c)
        if area < 100:
            cv2.drawContours(letter, [c], 0, (0, 0, 0), -1)

    # do canny edge detection
    edges = cv2.Canny(letter, 200, 200)
    """
    edges = cv2.Canny(img, 200, 200)

    cv2.imwrite("K_edges.png", edges)
    cv2.imshow("K_edges", edges)
    cv2.waitKey(0)

def sharpening(img):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  6, -1],
                       [ 0, -1,  0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    cv2.imshow("K_edges", image_sharp)
    cv2.waitKey(0)
    return image_sharp

def contrast_brightness_correction(image):

    alpha = 2.0  # Simple contrast control // 1.0-3.0
    beta = 50  # Simple brightness control  // 0-100
    new_image = np.zeros(image.shape, image.dtype)

    #"""
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                #print("a")
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
    #"""
    #new_image = alpha * image + beta
    print("a")
    cv2.imshow("aaadsasd", new_image)
    cv2.waitKey(0)
    return new_image

# ---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------


def histogram_equlization_rgb(img):
    # Simple preprocessing using histogram equalization
    # https://en.wikipedia.org/wiki/Histogram_equalization

    intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
    img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

    # For Grayscale this would be enough:
    # img = cv2.equalizeHist(img)
    cv2.imwrite("preprocessing/visualization/hit_eq_rgb.png", img)
    #cv2.imshow("hit_eq_rgb", img)
    #cv2.waitKey(0)
    return img

def histogram_equlization_bnw(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.equalizeHist(gray)

    cv2.imwrite("preprocessing/visualization/hit_eq_bnw.png", img)
    #cv2.imshow("hit_eq_rgb", img)
    #cv2.waitKey(0)
    return img

def contrast_brightness_correction_px(image):

    alpha = 2.0  # Simple contrast control // 1.0-3.0
    beta = 50  # Simple brightness control  // 0-100
    new_image = np.zeros(image.shape, image.dtype)

    # """
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                # print("a")
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
    # """
    # new_image = alpha * image + beta
    cv2.imwrite("preprocessing/visualization/cont_bright_corr_px_2_50.png", new_image)
    #     #cv2.imshow("hit_eq_rgb", img)
    #     #cv2.waitKey(0)
    return new_image



def contrast_brightness_correction_csa(image):

    alpha = 3.0  # Simple contrast control // 1.0-3.0
    beta =  0 # Simple brightness control  // 0-100

    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    cv2.imwrite("preprocessing/visualization/cont_bright_corr_csa_3_0.png", adjusted)
    #     #cv2.imshow("hit_eq_rgb", img)
    #     #cv2.waitKey(0)
    return adjusted

def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table

    aa = cv2.LUT(image, table)
    cv2.imwrite("preprocessing/visualization/adj_gamma_2.png", aa)
    #     #cv2.imshow("hit_eq_rgb", img)
    #     #cv2.waitKey(0)
    return aa

def edge_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 100)
    cv2.imwrite("preprocessing/visualization/edge_det_canny_100_100.png", edges)
    #cv2.imshow("edge_det_canny", edges)
    #cv2.waitKey(0)
    return edges


def sharpening(img):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    cv2.imwrite("preprocessing/visualization/sharpening_ker_5.png", image_sharp)

    return image_sharp

img = cv2.imread("data/ears/test/0008.png")
#cv2.imwrite("preprocessing/visualization/original.png", img)
#edge_enhancement(img)
#sharpening(img)
#contrast_brightness_correction(img)

#histogram_equlization_rgb(img) # This one makes VJ worse
#histogram_equlization_bnw(img)
#contrast_brightness_correction_px(img)
#contrast_brightness_correction_csa(img)
#adjust_gamma(img, 2.0)
#edge_detection(img)
#sharpening(img)