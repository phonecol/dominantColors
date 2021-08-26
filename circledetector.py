import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

path = '1,90ppm,26,30,seconds.png'
# path = 'papersensor.jpg'
# path = 'Inked90ppmafter2min__LI.jpg'
img = cv2.imread(path)
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
circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=1,
                            minDist=300, param1=40, param2=23,minRadius=45, maxRadius=60)
print(circles)
print("asd")
# contours = circles[0][:].argsort(axis=0)
contours = np.sort(circles[0][:], axis=0)
# contours = sorted(circles, key = lambda x: x[0][:])
print(contours)

for co, i in enumerate(circles[0, :], start=1):
    # draw the outer circle
    print(co)
    print(i[0])
    print(i[1])
    print(i[2])
    cv2.putText(cimg,str(co),(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
    cv2.circle(cimg,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
    # draw the center of the circle
    # cv2.circle(cimg,(int(i[0]),int(i[1])),2,(0,0,255),3)

    radius = int(math.ceil(i[2]))
    origin_x = int(math.ceil(i[0]) - radius)
    origin_y = int(math.ceil(i[1]) - radius)
    print(origin_x)
    print(origin_y)

    cv2.rectangle(cimg,(origin_x-10,origin_y-10),(origin_x+2*radius+10,origin_y+2*radius+10),(200,0,0),2)

    roi=cimg[origin_y:origin_y+2*radius,origin_x:origin_x+2*radius]
    roi2=cimg[origin_y+20:origin_y+2*radius-20,origin_x+20:origin_x+2*radius-20]
    cv2.imwrite("ROI/"+str(co) + '.jpg', roi)
    cv2.imwrite("ROI/"+'square'+str(co) + '.jpg', roi2)

    # roi=cimg[y:y+h,x:x+w]
# print the number of circles detected
print("Number of circles detected:", co)
# save the image, convert to BGR to save with proper colors
# cv2.imwrite("coins_circles_detected.png", cimg)
# show the image
plt.imshow(cimg)
plt.show()