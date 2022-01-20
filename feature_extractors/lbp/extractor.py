import cv2, sys
from skimage import feature
import numpy as np
from sklearn.svm import LinearSVC
from imutils import paths
#import argparse
from keras.applications.resnet import ResNet50


class LBP:
	def __init__(self, num_points=8, radius=2, eps=1e-6, resize=100):
		self.num_points = num_points * radius
		self.radius = radius
		self.eps = eps
		self.resize=resize

	def extract(self, img, im_name):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))

		lbp = feature.local_binary_pattern(img, self.num_points, self.radius, method="uniform")
		print(lbp)
		n_bins = int(lbp.max() + 1)
		hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

		# TODO
		print(hist)

		print("LEN----" + str(len(hist)))

		##############?????????????????????????????????????
		# hist = hist.astype("float")
		# hist /= (hist.sum() + 1e-7)
		##############?????????????????????????????????????

		return hist


	def myExtract(self, img, im_name):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))

		#lbp = feature.local_binary_pattern(img, self.num_points, self.radius, method="uniform")


		# TODO
		h, w = img.shape
		lbp = np.zeros((h, w), np.uint8)

		for i in range(1,h-1):
			for j in range(1,w-1):
				centerPX = img[i][j]
				decimal = 0

				if centerPX < img[i-1][j-1]:
					decimal = decimal + 1
				if centerPX < img[i-1][j]:
					decimal = decimal + 2
				if centerPX < img[i-1][j+1]:
					decimal = decimal + 4
				if centerPX < img[i][j+1]:
					decimal = decimal + 8
				if centerPX < img[i+1][j+1]:
					decimal = decimal + 16
				if centerPX < img[i+1][j]:
					decimal = decimal + 32
				if centerPX < img[i+1][j-1]:
					decimal = decimal + 64
				if centerPX < img[i][j-1]:
					decimal = decimal + 128
				"""
				nw = img[i-1][j-1]
				n = img[i-1][j]
				ne = img[i-1][j+1]
				e = img[i][j+1]
				se = img[i+1][j+1]
				s = img[i+1][j]
				sw = img[i+1][j-1]
				w = img[i][j-1]
				"""

				lbp[i,j] = decimal
		#hist = 0

		print(lbp)
		hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
		#n_bins = int(lbp.max() + 1)
		#hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
		print(hist)
		return hist

	"""
	def recognize(self, img, im_name):

		desc = LBP(24, 8)
		data = []
		labels = []

		for imagePath in paths.list_images(args["testing"]):
			# load the image, convert it to grayscale, describe it,
			# and classify it
			image = cv2.imread(imagePath)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			hist = desc.describe(gray)
			prediction = model.predict(hist.reshape(1, -1))

			# display the image and the prediction
			cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
						1.0, (0, 0, 255), 3)
			cv2.imshow("Image", image)
			cv2.waitKey(0)

		# loop over the training images
		for imagePath in paths.list_images(args["training"]):
			# load the image, convert it to grayscale, and describe it
			image = cv2.imread(imagePath)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			hist = desc.describe(gray)
			# extract the label from the image path, then update the
			# label and data lists
			labels.append(imagePath.split(os.path.sep)[-2])
			data.append(hist)
		# train a Linear SVM on the data
		model = LinearSVC(C=100.0, random_state=42)
		model.fit(data, labels)
	"""

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = LBP()
	features = extractor.extract(img)
	print(features)