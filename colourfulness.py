
# import the necessary packages
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


def image_colorfulness(image):
    #split the image into its respective RGB components
    (B,G,R) = cv2.split(image.astype("float"))

    # compute rg = R-G
    rg = np.absolute(R-G)

    #compute yb = 0.5 * (R+G) -B
    yb = np.absolute(0.5* (R+G)-B)

    # compute the mean and standard deviation of both 'rg' and 'yb'
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd**2) + (ybStd**2))
    meanRoot = np.sqrt((rbMean**2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it

    return stdRoot + (0.3 * meanRoot)

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
    help= "path to input directory of images")
args = vars(ap.parse_args())


# initialize the result list
print("[INFO] computing colorfulness metric for dataset...")
results = []

# loop over the image paths

for imagePath in paths.list_images(args["images"]):
    #load the image, resize it ( to speed up computation), and
    # compute the colorfulness metric for the image
    print(imagePath)
    imgAdd,imgNo = imagePath.split('\\')
    print(imgNo)
    image = cv2.imread(imagePath)

    image = imutils.resize(image, width = 250)
    image = image[20:230,20:230]
    
    C = image_colorfulness(image)
    print(C)
    #display the colorfulness score on the image
    cv2.putText(image, "{:.2f}".format(C), (40,40),
        cv2.FONT_HERSHEY_COMPLEX, 1.4, (0,255,0), 3)
    cv2.putText(image, imgNo, (40,90),
        cv2.FONT_HERSHEY_COMPLEX, 1.4, (0,255,0), 3)
    #add the image and colorfulness metric to results list
    results.append((image,C))



    #sort the results with more coorful images at the front of the
    #list, then build the lists of the "most colorful" and "least colorful" images
# print(results)
print("[INFO] displaying results...")
results = sorted(results, key=lambda x: x[1], reverse = True)
mostColor = [r[0] for r in results[:10]]
leastColor = [r[0] for r in results[-10:]][::-1]

# construct the montages for the two sets of images
mostColorMontage = build_montages(mostColor, (128,128), (10,1))
leastColorMontage = build_montages(leastColor, (128,128), (10,1))

cv2.imshow("Most Colorful",mostColorMontage[0])
cv2.imshow("Least Colorful",leastColorMontage[0])
cv2.waitKey(0)