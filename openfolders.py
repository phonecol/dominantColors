# from Shapedetector import ShapeDetector
from operator import sub
from shapedetector import ShapeDetector
import imutils
import cv2
import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
# Inked90ppmafter2min__LI

#load image
IMAGE_DIRECTORY = 'captured_images'

#initialize the list of images, and its filenames
images = []
ppm_values = []
subfolders = []
image_folders = []
imagesss = []
imagessss= []
imagesssss = []
image_paths= []
files = os.listdir(IMAGE_DIRECTORY)
files = natsorted(files)
# print(files)


#function for getting the image
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    return image


def get_circles(images):

    for i in range(len(images)):
        img = images[i]
        scale_percent = 100 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # load the image
        # img = cv2.imread(sys.argv[1])
        # convert BGR to RGB to be suitable for showing using matplotlib library
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # make a copy of the original image
        cimg = img.copy()
        # convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply a blur using the median filter
        img = cv2.medianBlur(img, 5)
        # finds the circles in the grayscale image using the Hough transform
        circles = cv2.HoughCircles(
            image=img,
            method=cv2.HOUGH_GRADIENT, 
            dp=1,
            minDist=300, 
            param1=40, 
            param2=23,
            minRadius=45, 
            maxRadius=60
        )
        print(circles)
        print("asd")
        print(circles[0,:,0])
        print("asd")
        NUM_ROWS = 3
        #sort circles
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda v: [v[0], v[0]])
        print(circles)

        sorted_cols = []
        for k in range(0, len(circles), NUM_ROWS):
            col = circles[k:k+NUM_ROWS]
            sorted_cols.extend(sorted(col, key=lambda v: v[1]))

        circles = sorted_cols
        print(circles)

        circles = np.uint16(np.around(circles))

        #initialize mask for the paper sensor
        mask = np.zeros(shape=img.shape[0:2],dtype='uint8')

        #create mask for each paper sensor
        for co, i in enumerate(circles, start=1):
            # draw the outer circle
            cv2.circle(mask,(int(i[0]),int(i[1])),int(i[2]),(255,255,255),-1)

        # cv2.imshow('mask',mask)
        # cv2.waitKey(0)

        # plt.imshow(mask, cmap="brg")
        # plt.show()

        #add the mask and the image
        masked =  cv2.bitwise_and(cimg,cimg, mask=mask)

        # cv2.imshow('masked',masked)
        # cv2.waitKey(0)

        for co, i in enumerate(circles, start=1):
            # draw the outer circle
            print(co)
            print(i[0])
            print(i[1])
            print(i[2])
            
            cv2.putText(masked,str(co),(int(i[0]+60),int(i[1])+60),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),2)
            cv2.circle(masked,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
            
            radius = int(math.ceil(i[2]))
            origin_x = int(math.ceil(i[0]) - radius)
            origin_y = int(math.ceil(i[1]) - radius)
            

            cv2.rectangle(cimg,(origin_x-10,origin_y-10),(origin_x+2*radius+10,origin_y+2*radius+10),(200,0,0),2)

            roi=masked[origin_y:origin_y+2*radius,origin_x:origin_x+2*radius]
            roi2=masked[origin_y+20:origin_y+2*radius-20,origin_x+20:origin_x+2*radius-20]
            cv2.imwrite("ROI/"+str(co)+','+ppm_value + '.jpg', roi)
            cv2.imwrite("roi2/"+str(co)+','+ppm_value+ '.jpg', roi2)

            # roi=cimg[y:y+h,x:x+w]
        # print the number of circles detected
        print("Number of circles detected:", co)
        # save the image, convert to BGR to save with proper colors
        # cv2.imwrite("coins_circles_detected.png", cimg)
        # show the image
        plt.imshow(masked)
        plt.show()



for file in files:
    # print(file)
    folder_path = os.path.join(IMAGE_DIRECTORY, file)  
    
    for subfolder in os.listdir(folder_path):
        # print(subfolder)
        subfolder_path = os.path.join(IMAGE_DIRECTORY,file, subfolder)
        # print(subfolder_path)
        for image in os.listdir(subfolder_path): 
        
            # print(image)
            image_number, ppm_value,sec,seconds,second = image.split(',')
            # print(image_number)
            image_path = os.path.join(IMAGE_DIRECTORY,file, subfolder, image)
            # print(image_path)
            if image_number == '1':
            # ppm_value, image_type, = image.split('.')
                images.append(get_image(image_path))
                image_paths.append(image_path)
                print(image_number)
                print(image)
                print("diri2")
        # print(imagesss)
        
    # subfolders.append(subfolder)

get_circles(images)
# print(imagesss)
print("lezgoo")
print(image_paths) 
 
print("lezgoooooooooo")
