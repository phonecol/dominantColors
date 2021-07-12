# from Shapedetector import ShapeDetector
from shapedetector import ShapeDetector
import imutils
import cv2

#load image
path = 'papersensor.jpg'
image = cv2.imread(path)

# resize image
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#convert image to grayscale colorspace
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#blur the grayscale image
blurred = cv2.GaussianBlur(gray, (5,5),0)

#thresholding separates the paper sensor or ROI to the background
#this will create the mask
thresh = cv2.threshold(blurred,200,255, cv2.THRESH_BINARY)[1]

#the original image will be masked with thresholded image so that the paper sensor will have a black background
masked = cv2.bitwise_and(resized,resized, mask=thresh)

#find countours on the binary image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
# return the actual contours array
cnts = imutils.grab_contours(cnts)

#use the ShapeDetector Module
sd = ShapeDetector()

#initialize id for each
idx =0
for c in cnts:
    idx+= 1

    x,y,w,h = cv2.boundingRect(c)
    roi=masked[y:y+h,x:x+w]
    cv2.rectangle(masked,(x-10,y-10),(x+w+10,y+h+10),(200,0,0),2)

    #write the cropped ROI
    cv2.imwrite("ROI/"+str(idx) + '.jpg', roi)

    # show the output image
    cv2.imshow("Image", masked)
    cv2.waitKey(0)

cv2.destroyAllWindows()

