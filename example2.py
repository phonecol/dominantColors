from dominantColors import DominantColors
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

IMAGE_DIRECTORY = 'images'
COLORS = {
    'GREEN': [0, 128, 0],
    'BLUE': [0, 0, 128],
    'YELLOW': [255, 255, 0]
}
images = []

for file in os.listdir(IMAGE_DIRECTORY):
    if not file.startswith('.'):
        images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))


clusters = 2
index = 1
RGB_KMeans = []
RGB_Means = []
RGB_stds =[]
print(len(images))
for i in range(len(images)):
    dc = DominantColors(images[i],clusters)
    colors = dc.dominantColors()
    print("Dominant Colors: ",colors)
    rgb_mean,rgb_std = dc.getAveColor()

    RGB_KMeans.append(colors)
    RGB_Means.append(rgb_mean)

    RGB_stds.append(rgb_std)
RGB_KMeans = np.array(RGB_KMeans)
RGB_Means = np.array(RGB_Means)
RGB_stds = np.array(RGB_stds)
print("RGB KMEANS: ",RGB_KMeans)

print("RGB MEANS: ",RGB_Means)

print("RGB STDS: ",RGB_stds)
red = RGB_Means[:,0]
green = RGB_Means[:,1]
blue = RGB_Means[:,2]
print(red)
x= np.arange(20)
plt.plot(x,red,color='red', marker='o', linestyle='dashed')
plt.plot(x,blue,color='green', marker='o', linestyle='dashed')
plt.plot(x,green,color='blue', marker='o', linestyle='dashed')
plt.show()