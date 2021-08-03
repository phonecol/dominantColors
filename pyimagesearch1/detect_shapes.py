# from Shapedetector import ShapeDetector
from shapedetector import ShapeDetector
import imutils
import cv2
# Inked90ppmafter2min__LI
#load image
<<<<<<< HEAD
path = 'Image.jpeg'
=======
path = '90ppmafter2minss.jpg'
# path = 'papersensor.jpg'
# path = 'Inked90ppmafter2min__LI.jpg'
>>>>>>> 843faaa0ced70aed5d5c8d2d32d8317a17192d92
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
<<<<<<< HEAD
thresh = cv2.threshold(blurred,150,255, cv2.THRESH_BINARY)[1]
=======
thresh = cv2.threshold(blurred,100,255, cv2.THRESH_BINARY)[1]
>>>>>>> 843faaa0ced70aed5d5c8d2d32d8317a17192d92

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

    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
    # set values as what you need in the situation
        cX, cY = 0, 0
    shape, peri = sd.detect(c)


    if shape == "circle" and peri > 100 and peri <240:
        print(shape)
        print(peri)
        x,y,w,h = cv2.boundingRect(c)
        roi=masked[y:y+h,x:x+w]
        cv2.rectangle(masked,(x-10,y-10),(x+w+10,y+h+10),(200,0,0),2)

        #write the cropped ROI
        cv2.imwrite("ROI/"+str(idx) + '.jpg', roi)

        # show the output image
        cv2.imshow("Image", masked)
        cv2.waitKey(0)
        idx+= 1




cv2.destroyAllWindows()

