from dominantColors import DominantColors
import cv2


#open image
# img = 'colors.jpeg'
img = '17.jpg'
img = cv2.imread(img)

#convert to RGB from BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#no. of clusters
clusters = 2

#initialize using constructor
dc = DominantColors(img, clusters)

#print dominant colors
colors = dc.dominantColors()
print(colors)
# dc.imageChannelHistogram()

dc.saveHistogram("Histogram")
dc.plotHistogram()
# dc.plotClusters()
dc.getAveColor()
dc.colorPixels()