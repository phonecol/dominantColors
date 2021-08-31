import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

path = '1,90ppm,26,30,seconds.png'
imgNum,ppm_value,x,xx,xxx = path.split(',')
print(ppm_value)
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

cv2.imshow('mask',mask)
cv2.waitKey(0)

plt.imshow(mask, cmap="brg")
plt.show()

#add the mask and the image
masked =  cv2.bitwise_and(cimg,cimg, mask=mask)

cv2.imshow('masked',masked)
cv2.waitKey(0)

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


