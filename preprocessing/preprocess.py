import cv2
import numpy as np


class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)
        print("a")
        return img

    def histogram_equlization_bnw(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.equalizeHist(gray)

        print("a")
        return img

    # Add your own preprocessing techniques here.

    # brightness correction, edge enhancement, etc.

    def contrast_brightness_correction_px(self, image):

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
        print("a")
        return new_image



    def contrast_brightness_correction_csa(self, image):

        alpha = 0.5  # Simple contrast control // 1.0-3.0
        beta =  0 # Simple brightness control  // 0-100

        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        print("a")
        return adjusted

    def adjust_gamma(self, image, gamma):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table

        aa = cv2.LUT(image, table)
        print("a")
        return aa

    def edge_detection(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 100, 100)

        return edges


    def sharpening(self, img):
        kernel = np.array([[ 0, -1,  0],
                           [-1,  6, -1],
                           [ 0, -1,  0]])
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        print("a")
        return image_sharp