from typing import OrderedDict
from skimage.color.delta_e import deltaE_ciede2000
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
from imutils import build_montages

# image = cv2.imread('sample_image.jpg')
# print("The type of this input is {}".format(type(image)))
# print("Shape: {}".format(image.shape))
# plt.imshow(image)


# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)


# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_image, cmap='gray')

# resized_image = cv2.resize(image, (1200, 600))
# plt.imshow(resized_image)

#COLOR Identification

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 200))
    image = image[20:230, 20:230]
    return image

def get_colors(image, number_of_colors, show_chart):

    modified_image = cv2.resize(image,(600,400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1],3)

    clf = KMeans(n_clusters= number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_

    ordered_colors = [center_colors[i] for i in counts.keys()]
    print(ordered_colors)
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    print(rgb_colors[0])
    if(show_chart):
        plt.figure(figsize=(8,6))
        plt.pie(counts.values(),labels = hex_colors, colors = hex_colors)
        plt.show()
    return rgb_colors



def get_deltaE(image,color, threshold= 60, number_of_colors = 3,de00 = True, de76 = True):

    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        if de76:
            dE_76 = deltaE_cie76(selected_color, curr_color)
        if de00:
            dE_2000 = deltaE_ciede2000(selected_color, curr_color)
        print(i)
        print('deltaE76:',dE_76)
        print('deltaE2000:',dE_2000)


    return  dE_76, dE_2000


def show_selected_images(images,ref_img, color, threshold, colors_to_match):
    index = 1
    dE76 = []
    dE2000 = []
    for i in range(len(images)):
        print('Image #',i)
        dE_76, dE_2000 = get_deltaE(images[i],
                                        color,
                                        threshold,
                                        colors_to_match,True,True)


        cv2.putText(images[i], "{:0.2f}".format(dE_2000[0][0]), (5,100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
        cv2.putText(images[i], "{:0.2f}".format(dE_76[0][0]), (5,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
        dE76.append(dE_76)
        dE2000.append(dE_2000)

    merged = tuple(zip(images,dE76,dE2000))
    merged = [r[0] for r in merged[:8]]
    deltaE_Montage = build_montages(merged, (150,150), (8,1))
    cv2.imshow("Most Colorful",deltaE_Montage[0])
    cv2.imshow("Reference Image", ref_img)
    cv2.waitKey(0)



        # if (selected):
        #     plt.subplot(1,10,index)
        #     plt.imshow(images[i])
        #     index += 1

    dE76 = np.ravel(dE76)
    dE2000 = np.ravel(dE2000)

    print(dE76)
    print(dE2000)
    return dE76, dE2000

ref_img= get_image('roi2/4,90ppm.jpg')
ref_color = get_colors(get_image('roi2/4,90ppm.jpg'),1,True)
print(ref_color)

IMAGE_DIRECTORY = 'roi2'
COLORS = {
    'GREEN': [0,128,0],
    'BLUE': [0,0,128],
    'YELLOW': [255,255,0],
    'REF':ref_color[0],
    # 'REF1': ref_color[1],
    'REF2':[130.90385956, 117.26774399, 135.32659249],
    'REF3':[143.7064375 , 130.29196667, 149.38702917]

}
images = []
combined =[]

for file in os.listdir(IMAGE_DIRECTORY):
    print(file)
    if not file.startswith('.'):
        image = get_image(os.path.join(IMAGE_DIRECTORY, file))

        images.append(image)
        combined_image = image.copy()
        cv2.putText(combined_image, "{}".format(file), (5,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 1)

        combined.append((combined_image, file))

combined = [r[0] for r in combined[:8]]
mostColorMontage = build_montages(combined, (250,250), (8,1))
cv2.imshow("Most Colorful",mostColorMontage[0])
cv2.waitKey(0)

# print(images)

# plt.figure(figsize=(20, 10))
# for i in range(len(images)):
#     plt.subplot(1, len(images), i+1)
#     plt.imshow(images[i])
#     plt.show()

plt.figure(figsize = (20, 10))
print('REF')
dE76, dE2000 =show_selected_images(images,ref_img, COLORS['REF'], 10, 1)
print(dE76, dE2000)



# print('REF1')
# show_selected_images(images, COLORS['REF1'], 10, 1)
# print('REF2')
# show_selected_images(images, COLORS['REF2'], 10, 1)
# print('REF3')
# show_selected_images(images, COLORS['REF3'], 10,1)
print('end')