from dominantColors import DominantColors
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from imutils import build_montages


# Function for reading an image file
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image


#The filenames of the images of the ROI of paper sensor must be the PPM levels.
#the directory of the images of the paper sensor


def get_images_from_a_folder(path):
#initialize the list of images, and its filenames
    images = []
    ppm_values = []
    combined = []

    files = os.listdir(path)
    files = natsorted(files)
    print(files)

    #a loop for getting the images in the folder and append them into a list
    for file in files:
        print(file)
        image = get_image(os.path.join(IMAGE_DIRECTORY, file))

        images.append(image)
        ppm_value, image_type = file.split('.')
        print(ppm_value)
        cv2.putText(image, "{}".format(ppm_value), (5,5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1)
        ppm_values.append(ppm_value)
        combined.append((image, ppm_value))

    combined = [r[0] for r in combined[:8]]
    ppm_values = np.array(ppm_values) #convert the list to numpy array
    print(ppm_values)
    # print(combined)
    mostColorMontage = build_montages(combined, (128,128), (2,4))
#     cv2.imshow("Most Colorful",mostColorMontage[0])
#     cv2.waitKey(0)
    return images, ppm_values


def save_data(data):
    print("data",data)
    data = data.T #transpose the data array
    print("data",data)
    header = 'PPM,R,G,B,R_std,G_std,B_std,H,S,V,H_std,S_std,V_std,L,a,b,L_std,a_std,b_std' #initialize the header for the csv file
    np.savetxt("data3.csv", data, delimiter=",",header= header,fmt='%s') #save the data array in a csv filetype with a filename of "data.csv" with the following header defined above



IMAGE_DIRECTORY = 'roi2'
images , ppm_values = get_images_from_a_folder(IMAGE_DIRECTORY)


#for KMeans Algorithm
clusters = 1 #number of clusters of colors. Value is at the range of (2-5)
index = 1
#
RGB_KMeans = []
RGB_Means = []
RGB_stds =[]
HSV_Means = []
HSV_stds =[]
Lab_Means = []
Lab_stds =[]
colorspaces = []
print(len(images))

for i in range(len(images)):

    dc = DominantColors(images[i],images,clusters) #initialize the DominantColors class
    dc.saveHistogram("Histograms/{}Histogram".format(i), True)

    colors = dc.dominantColors()  #call the dominantColors function to get the dominant colors of the image using KMeans Algorithm
    print("Dominant Colors: ",colors)
    print("Dominant Colors sorted: ",colors)
    hsv,lab,gray = dc.cvtColorSpace()
    # dc.plotHistogram()
    rgb_mean,rgb_std,hsv_mean,hsv_std,lab_mean,lab_std = dc.getAveColor()  #call the getAveColor function to get the average RGB pixel intensity and its standard deviation of the paper sensor

    #append the RGB, HSV,Lab Values into a list
    RGB_KMeans.append(colors)
    RGB_Means.append(rgb_mean)
    RGB_stds.append(rgb_std)
    HSV_Means.append(hsv_mean)
    HSV_stds.append(hsv_std)
    Lab_Means.append(lab_mean)
    Lab_stds.append(lab_std)
    colorspaces.append((colors, rgb_mean, rgb_std, hsv_mean, hsv_std, lab_mean, lab_std))

# dc.plotMultipleHistogram(0)
# dc.plotMultipleHistogram(1)
# dc.plotMultipleHistogram(2)

#convert the list into numpy array
RGB_KMeans = np.array(RGB_KMeans)
RGB_Means = np.array(RGB_Means)
RGB_stds = np.array(RGB_stds)
HSV_Means = np.array(HSV_Means)
HSV_stds = np.array(HSV_stds)
Lab_Means = np.array(Lab_Means)
Lab_stds = np.array(Lab_stds)
colorspaces= np.array(colorspaces)
print('Colorspaces',colorspaces)
print("RGB KMEANS: ",RGB_KMeans)
print("RGB MEANS: ",RGB_Means)
print("RGB STDS: ",RGB_stds)
print("HSV MEANS: ",HSV_Means)
print("HSV STDS: ",HSV_stds)
print("Lab MEANS: ",Lab_Means)
print("Lab STDS: ",Lab_stds)



# HSV_Means[:,0] = HSV_Means[:,0]/180*360
# HSV_Means[:,1:] = np.round(HSV_Means[:,1:]/255,8)
# HSV_stds[:,0] = HSV_stds[:,0]/180*360
# HSV_stds[:,1:] = np.round(HSV_stds[:,1:]/255,8)
red = RGB_Means[:,0]
green = RGB_Means[:,1]
blue = RGB_Means[:,2]

print("HSV MEANS: ",HSV_Means)
#convert the data type into a string
ppm_values_str = ppm_values.astype(str).T
RGB_Means_str = RGB_Means.astype(str).T
RGB_stds_str = RGB_stds.astype(str).T
HSV_Means_str = HSV_Means.astype(str).T
HSV_stds_str = HSV_stds.astype(str).T
Lab_Means_str = Lab_Means.astype(str).T
Lab_stds_str = Lab_stds.astype(str).T

print("ppm",ppm_values_str)
print("means",RGB_Means_str)
print("stds",RGB_stds_str)

data = np.vstack((ppm_values_str,RGB_Means_str,RGB_stds_str,HSV_Means_str,HSV_stds_str,Lab_Means_str,Lab_stds_str))
save_data(data)

fig1, (ax1, ax2) = plt.subplots(2, 1)

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

fig1.tight_layout()


#
plt.show()


fig2, (ax21, ax22) = plt.subplots(2, 1)
ax21.plot(ppm_values,HSV_Means[:,0],color='red', marker='o', linestyle='dashed')
ax21.plot(ppm_values,HSV_Means[:,1],color='green', marker='o', linestyle='dashed')
ax21.plot(ppm_values,HSV_Means[:,2],color='blue', marker='o', linestyle='dashed')
ax21.set_ylabel('Mean Pixel Intensity in HSV COLORSPACE')
ax21.set_xlabel('PPM Concentration')
plt.show()


fig3, (ax31, ax32) = plt.subplots(2, 1)
ax31.plot(ppm_values,Lab_Means[:,0],color='red', marker='o', linestyle='dashed')
ax31.plot(ppm_values,Lab_Means[:,1],color='green', marker='o', linestyle='dashed')
ax31.plot(ppm_values,Lab_Means[:,2],color='blue', marker='o', linestyle='dashed')
ax31.set_ylabel('Mean Pixel Intensity in Lab COLORSPACE')
ax31.set_xlabel('PPM Concentration')
plt.show()

scatter_color = RGB_Means/255
print(scatter_color)
area = 500  # 0 to 15 point radii

plt.scatter(ppm_values, RGB_Means[:,0], s=area, c=scatter_color, alpha=0.5)

plt.show()


scatter_hsv = HSV_Means[:,0]
print(scatter_hsv)
area = 500  # 0 to 15 point radii

plt.scatter(ppm_values, HSV_Means[:,0], s=area, c=scatter_hsv, alpha=0.5)
plt.show()
