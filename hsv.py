import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from imutils import build_montages




def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (250, 250))
    image = image[20:230, 20:230]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # gray.reshape(gray.shape[0],gray.shape[1],3)
    last_axis = -1
    dim_to_repeat = 2
    repeats = 3
    grscale_img_3dims = np.expand_dims(gray, last_axis)
    gray = np.repeat(grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')
    print(hsv.shape)
    print(gray.shape)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split (hsv)
    mean1 = h.mean ()
    mean2 = s.mean ()
    mean3 = v.mean ()
    print ('H:% .1f S:% .1f V:% .1f'% (mean1, mean2, mean3))
    return image,hsv,gray


#The filenames of the images of the ROI of paper sensor must be the PPM levels.
#the directory of the images of the paper sensor


def get_images_from_a_folder(path):
#initialize the list of images, and its filenames
    images = []
    hsv_images =[]
    gray_images = []
    ppm_values = []
    results = []
    results_hsv=[]
    results_gray=[]
    files = os.listdir(path)
    files = natsorted(files)
    print(files)

    #a loop for getting the images in the folder and append them into a list
    for file in files:
        print(file)
        image,hsv,gray = get_image(os.path.join(IMAGE_DIRECTORY, file))
        images.append(image)
        hsv_images.append(hsv)
        gray_images.append(gray)
        combined_image = image.copy()
        combined_hsv = hsv.copy()
        combined_gray = gray.copy()
        ppm_value, image_type = file.split('.')
        print(ppm_value)
        ppm_values.append(ppm_value)

        cv2.putText(combined_image, "{}".format(ppm_value), (40,40),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)
        cv2.putText(combined_hsv, "{}".format(ppm_value), (40,40),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)
        cv2.putText(combined_gray, "{}".format(ppm_value), (40,40),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)


        #add the image and colorfulness metric to results list
        results.append((combined_image,ppm_value))
        results_hsv.append((combined_hsv,ppm_value))

        results_gray.append((combined_gray,ppm_value))

        #sort the results with more coorful images at the front of the
        #list, then build the lists of the "most colorful" and "least colorful" images
    # print(results)
    # print("[INFO] displaying results...")
    # results = sorted(results, key=lambda x: x[1], reverse = True)
    # results_hsv = sorted(results_hsv, key=lambda x: x[1], reverse = True)

    # results_gray = sorted(results_gray, key=lambda x: x[1], reverse = True)
    # rgb_images = [r[0] for r in results[:10]]
    # hsv_images = [r[0] for r in results_hsv[:10]]

    # gray_images = [r[0] for r in results_gray[:10]]
    # # construct the montages for the two sets of images
    # rgb_images_Montage = build_montages(rgb_images, (200,200), (10,1))

    # hsv_images_Montage = build_montages(hsv_images, (200,200), (10,1))
    # gray_images_Montage = build_montages(gray_images, (200,200), (10,1))

    # cv2.imshow("RGB",rgb_images_Montage[0])

    # cv2.imshow("HSV",hsv_images_Montage[0])

    # cv2.imshow("Gray",gray_images_Montage[0])
    # cv2.waitKey(0)

    ppm_values = np.array(ppm_values) #convert the list to numpy array
    print(ppm_values)

    return images,hsv_images,gray_images, ppm_values


ppm_concentration_str = ["WFP-AuNP",  "Deionised Water", "0.01 ppm", "0.1 ppm" , "0.5 ppm", "1 ppm", "5 ppm", "10 ppm", "20 ppm", "30 ppm"]
ppm_concentration_int = [-1,0,0.01, 0.1,0.5,1,5,10,20,30]

min = str(10)
IMAGE_DIRECTORY = 'ROI'+min+'min'
images ,hsvs,gray, ppm_values = get_images_from_a_folder(IMAGE_DIRECTORY)

bgr_Means = []
hsv_Means = []
for i in range(len(images)):
    hsv = hsvs[i]
    img = images[i]
    # cv2.imshow("hsv",hsv)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)


    means = []
    stds = []
    meanshsv = []
    stdshsv = []
    for j in range(0,3):
        val = np.reshape(img[:,:,j],-1)
        mean = np.mean(val)
        std = np.std(val)
        means.append(mean)
        stds.append(std)

        valhsv = np.reshape(hsv[:,:,j],-1)
        meanhsv = np.mean(valhsv)
        stdhsv = np.std(valhsv)
        meanshsv.append(meanhsv)
        stdshsv.append(stdhsv)

    print('HSV MEAN',meanshsv)
    hsv_Means.append(meanshsv)
    print('RGB',means)
    bgr_Means.append(means)
        # print(stds)

print(bgr_Means)
print(hsv_Means)


bgr_Means = np.array(bgr_Means)
hsv_Means = np.array(hsv_Means)
print(bgr_Means)
print(hsv_Means)
print(ppm_values)
fig1, (ax1, ax2) = plt.subplots(2, 1)

#plot the RGB_Mean Intensity of the paper sensor that was taken

ax1.set_title('BLUE, GREEN AND RED INTENSITY at '+ min + ' minutes Detection Time')
ax1.plot(ppm_concentration_str,bgr_Means[:,0],color='blue', marker='o', linestyle='dashed', label='Blue')
ax1.plot(ppm_concentration_str,bgr_Means[:,1],color='green', marker='o', linestyle='dashed', label='Green')
ax1.plot(ppm_concentration_str,bgr_Means[:,2],color='red', marker='o', linestyle='dashed', label='Red')
ax1.set_ylabel('Mean Pixel Intensity')
ax1.set_xlabel('Cyanide Concentration (PPM)')
ax1.legend()

ax2.set_title('HUE, SATURATION, AND VALUE at  '+ min + '  minutes Detection Time')

ax2.plot(ppm_concentration_str,hsv_Means[:,0],color='blue', marker='o', linestyle='dashed', label='Hue')
ax2.plot(ppm_concentration_str,hsv_Means[:,1],color='green', marker='o', linestyle='dashed', label='Saturation')
ax2.plot(ppm_concentration_str,hsv_Means[:,2],color='red', marker='o', linestyle='dashed', label='Value')
ax2.set_ylabel('Mean Pixel Intensity')

ax2.set_xlabel('Cyanide Concentration (PPM)')
ax2.legend()

for i, txt in enumerate(hsv_Means[:,0].astype(int)):
    ax2.annotate(txt, (ppm_values[i], hsv_Means[i,0]))
plt.show()