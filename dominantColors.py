import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self,image,images, clusters = 3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        self.IMAGES = images
    def dominantColors(self):

        img = self.IMAGE
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0]*img.shape[1],3))

        #save image after operations
        self.FLAT_IMAGE = img

        #using K-means clust to pixels
        kmeans = KMeans(n_clusters= self.CLUSTERS)
        kmeans.fit(img)

        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        #save labels
        self.LABELS = kmeans.labels_



        #returning after converting to integer from float
        return self.COLORS.astype(int)


    def plotHistogram(self):

        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)

        #create frequency count tables
        (hist,_) = np.histogram(self.LABELS, bins = numLabels)



        hist = hist.astype("float")
        hist /= hist.sum()

        #appending frequencies to cluster centers
        centroids = self.COLORS.astype(float)

        # create empty chart
        chart = np.zeros((50,500,3), dtype=np.uint8)
        start = 0
        colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])

        #display chart
        for (percent, color) in colors:
            print(color, "{:0.2f}%".format(percent * 100))
            end = start + (percent * 500)
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), \
                        color.astype("uint8").tolist(), -1)
            start = end



        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]),int(rgb[1]),int(rgb[2]))

    def plotClusters(self):
        #plotting
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        for label, pix in zip(self.LABELS, self.FLAT_IMAGE):
            ax.scatter(pix[0],pix[1],pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        plt.show()

    def colorPixels(self):
        shape = self.IMAGE.shape

        img = np.zeros((shape[0]*shape[1],3))
        labels = self.LABELS

        for i, color in enumerate(self.COLORS):
            indices = np.where(labels == i)[0]

            for index in indices:
                img[index] = color

        img = img.reshape((shape[0],shape[1],3)).astype(int)
        #display img

        plt.figure()
        plt.axis("off")
        plt.imshow(img)
        plt.show()


    def imageChannelHistogram(self,channel, bins=256):
        img = self.IMAGE
        # hsv, lab = DominantColors.cvtColorSpace(self)
        heights, edges = np.histogram(img[:,:,channel], bins, (0,256))
        return heights, edges

    def plotMultipleHistogram(self, channel,bins=256):
        imgs = self.IMAGES
        heights = []
        edges = []

        for img in imgs:
            height, edge = np.histogram(img[:,:,channel], bins, (0,256))
            # print("height",height)
            # print("edge", edge)
            heights.append(height)
            edges.append(edge)

        heights = np.array(heights)
        edges = np.array(edges)


        histoFig = plt.figure()
        histoAxis = histoFig.add_subplot(111)

        histoAxis.set_facecolor('xkcd:grey')
        histoAxis.set_xlim([0,256])
        histoAxis.set_xticks(np.linspace(0,256,9))
        histoAxis.set_xlabel("Intensity")
        histoAxis.set_ylabel("Counts")
        colors=['black', 'red', 'green', 'blue', 'cyan','yellow','orange','pink']
        for i in range(len(imgs)):
            # print("plot number ",i)
            centers = (edges[i][:-1] + edges[i][1:]) / 2
            #Plots the histograms
            histoPlotBlue = histoAxis.bar(centers, heights[i], align='center', color=colors[i], width=edges[i][1] - edges[i][0], alpha=0.2)


    def saveHistogram(self, path, plotFigure=False):

        # img = self.IMAGE
        #Computing the histograms for each channel
        Rheights, edges = self.imageChannelHistogram(0)
        Gheights, edges = self.imageChannelHistogram(1)
        Bheights, edges = self.imageChannelHistogram(2)

        #Converting the result to strings for saving as a csv
        edges_str = edges.astype(int).astype(str)[:-1]
        Bheights_str = Bheights.astype(int).astype(str)
        Gheights_str = Gheights.astype(int).astype(str)
        Rheights_str = Rheights.astype(int).astype(str)

        #Sticking them together
        combo = np.concatenate(([edges_str], [Rheights_str], [Gheights_str], [Bheights_str]), axis=0).T

        #Saving the csv file
        np.savetxt(path+'.csv', combo, fmt='%s', header='bin, Red Channel, Green Channel, Blue Channel', delimiter=',', comments='')

        ###Also saving plots of the histograms for immediate inspection###

        #Average of successive edges will be the centers of the bins
        centers = (edges[:-1] + edges[1:]) / 2


        #Setup for histogram plotting
        #default value for plotFigure is false
        if plotFigure:
            histoFig = plt.figure()
            histoAxis = histoFig.add_subplot(111)

            histoAxis.set_facecolor('xkcd:grey')
            histoAxis.set_xlim([0,256])
            histoAxis.set_xticks(np.linspace(0,256,9))
            histoAxis.set_xlabel("Intensity")
            histoAxis.set_ylabel("Counts")
            histoAxis.set_title(f"Zone {path.split('_')[-1]} Histogram")


            #Plots the histograms
            histoPlotBlue = histoAxis.bar(centers, Bheights, align='center', color='blue', width=edges[1] - edges[0], alpha=0.6)
            histoPlotGreen = histoAxis.bar(centers, Gheights, align='center', color='green', width=edges[1] - edges[0], alpha=0.6)
            histoPlotRed = histoAxis.bar(centers, Rheights, align='center', color='red', width=edges[1] - edges[0], alpha=0.6)


            #Saving the figures
            histoFig.savefig(path+'.png')


    def getAveColor(self, getHSV=True,getLab=True):
        hsv, lab = DominantColors.cvtColorSpace(self)
        img_MEAN_RGB = []
        img_STD_RGB = []
        img_MEAN_HSV = []
        img_STD_HSV = []
        img_MEAN_Lab = []
        img_STD_Lab = []


        img = self.IMAGE
        for i in range(0,3):
            val = np.reshape(img[:,:,i],-1)
            # masked = np.ma.masked_less(val,20)
            img_mean = np.mean(val)
            img_std = np.std(val)
            img_MEAN_RGB.append(img_mean)
            img_STD_RGB.append(img_std)

            # return img_MEAN_RGB, img_STD_RGB
        if getHSV:
            for i in range(0,3):
                val_hsv = np.reshape(hsv[:,:,i],-1)
                # masked_hsv = np.ma.masked_less(val_hsv,20)
                img_mean_hsv = np.mean(val_hsv)
                img_std_hsv = np.std(val_hsv)
                img_MEAN_HSV.append(img_mean_hsv)
                img_STD_HSV.append(img_std_hsv)

            # return img_MEAN_HSV, img_STD_HSV

        if getLab:
            for i in range(0,3):
                val_lab = np.reshape(lab[:,:,i],-1)
                # masked_lab = np.ma.masked_less(val_lab,20)
                img_mean_lab = np.mean(val_lab)
                img_std_lab = np.std(val_lab)
                img_MEAN_Lab.append(img_mean_lab)
                img_STD_Lab.append(img_std_lab)

            # return img_MEAN_Lab, img_STD_Lab

        return img_MEAN_RGB, img_STD_RGB, img_MEAN_HSV, img_STD_HSV, img_MEAN_Lab, img_STD_Lab

    def cvtColorSpace(self):
        img = self.IMAGE
        hsv= cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)


        return hsv, lab
