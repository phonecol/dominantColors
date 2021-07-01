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

    def __init__(self,image, clusters = 3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):

        #read image
        # img = cv2.imread(self.IMAGE)

        # #convert to rgb from bgr
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


        img = self.IMAGE

        # colorss = ("red", "green", "blue")
        # channel_ids = (0, 1,2)

        # # create the histogram plot, with three lines, one for
        # # each color
        # plt.xlim([0, 256])
        # for channel_id, c in zip(channel_ids, colorss):
        #     histogram, bin_edges = np.histogram(
        #         img[:, :, channel_id], bins=256, range=(0, 256)
        #     )
        #     data = zip(*np.histogram(img[:, :, channel_id], bins=256, range=(0, 256)))
        #     print(data)
        #     # np.savetxt('1.csv', data, delimeter=',')
        #     plt.plot(bin_edges[0:-1], histogram, color=c)

        # plt.xlabel("Color value")
        # plt.ylabel("Pixels")

        # plt.show()



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
        colors = self.COLORS

        #descending order sorting as per frequency count
        # colors = colors[(~hist).argsort()]
        # hist= hist[(~hist).argsort()]

        # create empty chart
        chart = np.zeros((50,500,3), np.uint8)
        start = 0

        #for creating color rectangles
        for i in range(self.CLUSTERS):
            end = start +hist[i]*500

            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            cv2.rectangle(chart, (int(start),0),(int(end),50),(r,g,b), -1)
            start = end
        #display chart
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
        heights, edges = np.histogram(img[:,:,channel], bins, (0,256))
        return heights, edges



    def saveHistogram(self, path):

        img = self.IMAGE
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

        #Uncomment for Photoshop-style color-mixing intersections in the saved histograms
##        histoPlotCyan = histoAxis.bar(centers, np.min([Bheights, Gheights],axis=0), align='center', color='cyan', width=edges[1] - edges[0], alpha=1)
##        histoPlotMagenta = histoAxis.bar(centers, np.min([Bheights, Rheights],axis=0), align='center', color='magenta', width=edges[1] - edges[0], alpha=1)
##        histoPlotYellow = histoAxis.bar(centers, np.min([Rheights, Gheights],axis=0), align='center', color='yellow', width=edges[1] - edges[0], alpha=1)
##
##        histoPlotWhite = histoAxis.bar(centers, np.min([Bheights, Gheights, Rheights],axis=0), align='center', color='white', width=edges[1] - edges[0], alpha=1)

        #Saving the figures
        histoFig.savefig(path+'.png')
##        plt.close()

    # def getAveColor(self):
    #     img = self.IMAGE
    #     imgMasked = np.ma.masked_less(img,20)
    #     avcolor = np.ma.mean(imgMasked, axis=(0,1))
    #     std = np.ma.std(imgMasked, axis=(0,1))
    #     print(imgMasked)
    #     # print("Average Color",avcolor)
    #     # print("Standard Deviation of intensity values",std)

    #     return avcolor, std

    def getAveColor(self):
        img_MEAN_RGB = []
        img_STD_RGB = []
        img = self.IMAGE
        for i in range(0,3):
            val = np.reshape(img[:,:,i],-1)
            masked = np.ma.masked_less(val,20)
            img_mean = np.mean(masked)
            img_std = np.std(masked)
            img_MEAN_RGB.append(img_mean)
            img_STD_RGB.append(img_std)

        print("MEAN RGB: ",img_MEAN_RGB)
        print("Standard Deviation: ",img_STD_RGB)
        return img_MEAN_RGB, img_STD_RGB


