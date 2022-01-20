import cv2, sys, os
import numpy as np
from PIL import Image


class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!

    #cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml')))
	leftear_cascade = cv2.CascadeClassifier(
		"/Users/zala/PycharmProjects/ibb_ass2_/detectors/cascade_detector/custom_cascades/cascadeleftreal.xml")
	rightear_cascade = cv2.CascadeClassifier(
		"/Users/zala/PycharmProjects/ibb_ass2_/detectors/cascade_detector/custom_cascades/cascaderightreal.xml")

	#rightear_cascade = cv2.CascadeClassifier("/Users/zala/PycharmProjects/ibb_ass2_/detectors/cascade_detector/cascades/haarcascade_mcs_rightear.xml")

	net = cv2.dnn.readNet("/Users/zala/PycharmProjects/ibb_ass2_/yolov3_training_best_1k.weights", "/Users/zala/PycharmProjects/ibb_ass2_/yolov3_testing.cfg")
	classes = ["Ear"]
	#cascade = cv2.CascadeClassifier("/Users/zala/PycharmProjects/ibb_ass2_/detectors/cascade_detector/cascades/haarcascade_mcs_rightear.xml")

	def detect(self, img):
		print("DETECTAM S HAAR")
		# det_list = self.cascade.detectMultiScale(img, 1.05, 1)

		#	"""
		det_list = list()

		left_ear = self.leftear_cascade.detectMultiScale(img, 1.05, 1)
		right_ear = self.rightear_cascade.detectMultiScale(img, 1.05, 1)

		if len(left_ear) != 0:
			det_list = left_ear.tolist()

		if len(right_ear) != 0:
			det_list = det_list + right_ear.tolist()

		#	"""

		print(det_list)
		return det_list

	def detectYOLO(self, img, im_name):
		print("DETECTAM Z YOLO")
		det_list = list()
		layer_names = self.net.getLayerNames()

		output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
		height, width, channels = img.shape
		# Detecting objects
		blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

		self.net.setInput(blob)
		outs = self.net.forward(output_layers)

		class_ids = []
		confidences = []
		boxes = []

		for out in outs:
			count = 0
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]

				if confidence > 0.3:
					# Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					print([x, y, w, h])
					l = len(im_name)
					saveto = im_name[0:l-8] + "cropped/" + im_name[l-8:l-4] + "-" + str(count) + im_name[l-4:l]
					print(saveto)
					count = count + 1
					#cropped = img.crop((x, y, x + w, y + h))
					#cropped.save(saveto)
					if y < 0:
						y = 0
					bab = y+h
					if bab > 359:
						bab = 359
					if x < 0:
						x = 0
					aba = x+w
					if aba > 479:
						aba = 479
					print([x, y, aba, bab])
					crop_img = img[y:bab, x:aba]
					#cv2.imshow("cropped", crop_img)
					status = cv2.imwrite(saveto, crop_img)


					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				det_list.append(boxes[i])
				label = str(self.classes[class_ids[i]])

		return det_list

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = Detector()
	#detected_loc = detector.detect(img)
	detected_loc = detector.detectYOLO(img)

	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)