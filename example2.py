from dominantColors import DominantColors
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#The filenames of the images of the ROI of paper sensor must be the PPM levels.

#the directory of the images of the paper sensor
IMAGE_DIRECTORY = 'images'

#initialize the list of images, and its filenames
images = []
ppm_values = []

#function for getting the image
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


#a loop for getting the images in the folder and append them into a list
for file in os.listdir(IMAGE_DIRECTORY):
    images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))
    ppm_value, image_type = file.split('.')
    print(ppm_value)
    ppm_values.append(ppm_value)

ppm_values = np.array(ppm_values) #convert the list to numpy array

#for KMeans Algorithm
clusters = 2 #number of clusters of colors. Value is at the range of (2-5)
index = 1

#
RGB_KMeans = []
RGB_Means = []
RGB_stds =[]
HSV_Means = []
HSV_stds =[]
Lab_Means = []
Lab_stds =[]
print(len(images))

for i in range(len(images)):
    #initialize the DominantColors class
    dc = DominantColors(images[i],clusters)
    dc.saveHistogram("Histograms/{}Histogram".format(i))
    #call the dominantColors function to get the dominant colors of the image using KMeans Algorithm
    colors = dc.dominantColors()
    print("Dominant Colors: ",colors)


    #call the getAveColor function to get the average RGB pixel intensity and its standard deviation of the paper sensor
    rgb_mean,rgb_std,hsv_mean,hsv_std,lab_mean,lab_std = dc.getAveColor()
    # hsv, lab = dc.cvtColorSpace()
    # cv2.imshow('hsv',hsv)
    # cv2.imshow('lab',lab)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    #append the RGB values into a list
    RGB_KMeans.append(colors)
    RGB_Means.append(rgb_mean)

    RGB_stds.append(rgb_std)
    HSV_Means.append(hsv_mean)

    HSV_stds.append(hsv_std)

    Lab_Means.append(lab_mean)

    Lab_stds.append(lab_std)


#convert the list into numpy array
RGB_KMeans = np.array(RGB_KMeans)
RGB_Means = np.array(RGB_Means)
RGB_stds = np.array(RGB_stds)
HSV_Means = np.array(HSV_Means)
HSV_stds = np.array(HSV_stds)
Lab_Means = np.array(Lab_Means)
Lab_stds = np.array(Lab_stds)
print("RGB KMEANS: ",RGB_KMeans)

print("RGB MEANS: ",RGB_Means)

print("RGB STDS: ",RGB_stds)


print("HSV MEANS: ",HSV_Means)

print("HSV STDS: ",HSV_stds)


print("Lab MEANS: ",Lab_Means)

print("Lab STDS: ",Lab_stds)
red = RGB_Means[:,0]
green = RGB_Means[:,1]
blue = RGB_Means[:,2]


#convert the data type into a string
ppm_values_str = ppm_values.astype(str).T
RGB_Means_str = RGB_Means.astype(str).T
RGB_stds_str = RGB_stds.astype(str).T

#concatinate the ppm values, RGB Means, amd the RGB Standard deviation using vstack
data = np.vstack((ppm_values_str,RGB_Means_str,RGB_stds_str))

#transpose the data array
data = data.T

#initialize the header for the csv file
header = 'PPM Values, R, G, B, R_std, G_std,B_std'

#save the data array in a csv filetype with a filename of "data.csv" with the following header defined above
np.savetxt("data.csv", data, delimiter=",",header= header,fmt='%s')


print(data)


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)

#plot the RGB_Mean Intensity of the paper sensor that was taken
ax1.plot(ppm_values,RGB_Means[:,0],color='red', marker='o', linestyle='dashed')
ax1.plot(ppm_values,RGB_Means[:,1],color='green', marker='o', linestyle='dashed')
ax1.plot(ppm_values,RGB_Means[:,2],color='blue', marker='o', linestyle='dashed')
ax1.set_ylabel('Mean Pixel Intensity')
ax1.set_xlabel('PPM Concentration')

labels = ppm_values_str
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
x =x+1
print('x',x)
print(labels)
# fig, ax1 = plt.subplots()
rects1 = ax2.bar(x + width/2,RGB_Means[:,0],width, label='Red',color='r')
rects2 = ax2.bar(x + 1.5*width, RGB_Means[:,1], width, label='Green',color='g')
rects3 = ax2.bar(x + 2.5*width, RGB_Means[:,2], width, label='Blue',color='b')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax2.set_ylabel('Mean Pixel Intensity')
ax2.set_title("Mean RGB Pixel Intensity of Au-NP's")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlabel('PPM Concentration')
ax2.legend()

ax2.bar_label(rects1, padding=3)
ax2.bar_label(rects2, padding=3)
ax2.bar_label(rects3, padding=3)


rects1 = ax3.barh(x + width/2,RGB_Means[:,0],width, label='Red',color='r')
rects2 = ax3.barh(x + 1.5*width, RGB_Means[:,1], width, label='Green',color='g')
rects3 = ax3.barh(x + 2.5*width, RGB_Means[:,2], width, label='Blue',color='b')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax3.set_xlabel('Mean Pixel Intensity')
ax3.set_title("Mean RGB Pixel Intensity of Au-NP's")
ax3.set_yticks(x)
ax3.set_yticklabels(labels)
ax3.set_ylabel('PPM Concentration')
ax3.legend()

ax3.bar_label(rects1, padding=3)
ax3.bar_label(rects2, padding=3)
ax3.bar_label(rects3, padding=3)


fig.tight_layout()



plt.show()


# plt.show()

