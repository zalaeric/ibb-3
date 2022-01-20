import urllib.request
import cv2
import numpy as np
import os
import glob
import cvzone
# seed the pseudorandom number generator
from random import seed
from random import randrange


def store_neg_images():
    pic_num = 1
    print("asd")


    for i in glob.glob('data/landscapes' + '/*.jpg', recursive=True):
        try:
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("data/neg2/" + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))


def create_pos_n_neg():
    for type in ['neg']:

        for img in os.listdir("data/" + type):

            if type == 'pos':
                line = type + '/' + img + ' 1 0 0 50 50\n'
                with open('info.dat', 'a') as f:
                    f.write(line)
            elif type == 'neg':
                line = type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)


def superimpose():

    pic_num = 1
    f = open("info.lst", "a")
    for i in glob.glob('data/neg' + '/*.jpg', recursive=True):
        #if pic_num < 2:


        try:
            background = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            overlay = cv2.imread('ear.jpg', cv2.IMREAD_GRAYSCALE)

            size = randrange(10, 60)
            #cv2.imshow("Image", overlay)
            #cv2.waitKey(0)
            resized_image = cv2.resize(overlay, (size, size))
            int1 = randrange(0,100-size)
            int2 = randrange(0,100-size)
            #tableyo = [int1, int2]
            #print(tableyo)
            #added_image = cvzone.overlayPNG(background, overlay, tableyo)
            background[int1:int1+size, int2:int2+size] = resized_image
            cv2.imwrite("data/pos/" + str(pic_num) + ".jpg", background)
            f.write("pos/" + str(pic_num) + ".jpg 1 " + str(int1) + " " + str(int2) + " "  + str(size) + " " + str(size) + "\n")
            pic_num += 1

        except Exception as e:
            print(str(e))

        """
        line = type + '/' + img + '\n'
        with open('bg.txt', 'a') as f:
            f.write(line)
        """
    f.close()

store_neg_images()

#create_pos_n_neg()

#superimpose()

