# from Shapedetector import ShapeDetector
from shapedetector import ShapeDetector
import imutils
import cv2
import os
from natsort import natsorted
# Inked90ppmafter2min__LI


#function for getting the image
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_countours(images):

    for i in range(len(images)):
        img = images[i]
        # resize image
        scale_percent = 25 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        #convert image to grayscale colorspace
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        #blur the grayscale image
        blurred = cv2.GaussianBlur(gray, (5,5),0)

        #thresholding separates the paper sensor or ROI to the background
        #this will create the mask
        thresh = cv2.threshold(blurred,100,255, cv2.THRESH_BINARY)[1]

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
        get_roi(cnts,sd,masked,idx,i)

def get_roi(cnts,sd,masked,idx,i):
    for c in cnts:

        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
        # set values as what you need in the situation
            cX, cY = 0, 0
        shape, peri = sd.detect(c)


        if shape == "circle" and peri > 90 and peri <250:
            print(shape)
            print(peri)
            x,y,w,h = cv2.boundingRect(c)
            roi=masked[y:y+h,x:x+w]
            cv2.rectangle(masked,(x-10,y-10),(x+w+10,y+h+10),(200,0,0),2)

            #write the cropped ROI
            cv2.imwrite("ROI/"+ppm_values[i] +str(idx) + '.jpg', roi)

            # show the output image
            cv2.imshow("Image", masked)
            cv2.waitKey(0)
            idx+= 1




    cv2.destroyAllWindows()


#load image
IMAGE_DIRECTORY = 'images/100ppm'

#initialize the list of images, and its filenames
images = []
ppm_values = []



files = os.listdir(IMAGE_DIRECTORY)
files = natsorted(files)
# files.sort(key=lambda x:int(x[:2]))
print(files)

#a loop for getting the images in the folder and append them into a list
for file in files:
    print(file)
    images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))
    ppm_value, image_type = file.split('.')
    print(ppm_value)
    ppm_values.append(ppm_value)

get_countours(images)