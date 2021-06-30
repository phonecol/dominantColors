import argparse
import imutils
import cv2
import numpy as np

# ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image", required=True,
#     help ="path to the input image")
# args = vars(ap.parse_args())
# path = 'shapes_and_colors.jpg'
path = 'papersensor.jpg'
# image = cv2.imread(args[image])
image = cv2.imread(path)

# resize image
scale_percent = 40 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
image = resized

#convert image to grayscale colorspace
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blur the grayscale image
blurred = cv2.GaussianBlur(gray,(5,5),0)

#thresholding separates the paper sensor or ROI to the background
#this will create the mask
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

#the original image will be masked with thresholded image so that the paper sensor will have a black background
masked = cv2.bitwise_and(image,image, mask=thresh)
print(image.shape)

cv2.imshow('Original',image)
cv2.imshow('Resized',resized)
cv2.imshow('Threshold',thresh)
cv2.imshow('masked',masked)

cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#for loop over the contours
for c in cnts:
    #compute the center of the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
    # set values as what you need in the situation
        cX, cY = 0, 0
    # cX = int(M["m10"]/ M["m00"])
    # cY = int(M["m01"]/ M["m00"])

    #draw the contour and center of the shape on the image
    cv2.drawContours(masked, [c], -1, (0,255,0),2)
    cv2.circle(masked, (cX,cY), 7,(255,255,255), -1)
    cv2.putText(masked, "center",(cX-20,cY-20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)

    cv2.imshow("Image",masked)
    cv2.waitKey(0)