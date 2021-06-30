import cv2
import numpy as np
from matplotlib import pyplot as plt


paperSensors = []
#load and read each photo of the cropped paper sensor images with black background
for i in range(1,3):
   paperSensor = cv2.imread(str(i)+".jpg")
   paperSensors.append(paperSensor)

#initialize the matrix for the average value of the RGB intensity of the image
bAveVals= []  #average value for blue
gAveVals= []  #average value for green
rAveVals= []  #average value for red
x_val = []

#initialize the matrix for the sum of the RGB intensity of the image
iTotVal = []


for j in range(1,3):

   #access the first paper sensor in the matrix
    img = paperSensors[j]

    #sum of the RGB intensity values of the image
    totalPixelValues = cv2.sumElems(img)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,200,255, cv2.THRESH_BINARY)[1]
    nonBlack = cv2.countNonZero(thresh)
    print("Widths",img.shape[1])

    print("Height",img.shape[0])
    print("Nonzero", nonBlack)
    print("Total Blue",totalPixelValues[0])
    print("Total Green",totalPixelValues[1])
    print("Total Red",totalPixelValues[2])
    # print("Orginal BGR:",img[30,30])

    bAveVal = totalPixelValues[0]/nonBlack
    gAveVal = totalPixelValues[1]/nonBlack
    rAveVal = totalPixelValues[2]/nonBlack
    print("Average Value of Blue",bAveVal)
    print("Average Value of Green",gAveVal)
    print("Average Value of Red",rAveVal)

    itot = bAveVal + gAveVal + rAveVal
    bAveVals.append(bAveVal)
    gAveVals.append(gAveVal)
    rAveVals.append(rAveVal)
    iTotVal.append(itot)
    print(iTotVal)
    print(bAveVals)

    print(gAveVals)

    print(rAveVals)

    print("total rbg",itot)




    b,g,r = cv2.split(img)
   #  cv2.imshow(str(j),img)
    cv2.imshow("b",thresh)
    # cv2.imshow("b",gray)

    # cv2.imshow("g",g)
    # cv2.imshow("r",r)
    print(b.ravel().shape)

    print(g.ravel().shape)

    print(r.ravel().shape)



    plt.title(str(j))
    plt.hist(b.ravel(),256,[0,256],facecolor='b')
    plt.hist(g.ravel(),256,[0,256],facecolor='g')
    plt.hist(r.ravel(),256,[0,256],facecolor='r')


    # x_val.append(j)
    # plt.plot(x_val,bAveVals)
    # plt.plot(x_val,gAveVals)
    # plt.plot(x_val,rAveVals)
    plt.show()


    cv2.waitKey(0)
    cv2.destroyAllWindows()



