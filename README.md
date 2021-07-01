# dominantColors

  this repo is about extracting colors from an image using KMeans Algorithm.
  
  "dominantColors.py" is the module for extracting colors from an image using KMeans Algorithm.
  It has a class of DominantColors.
  
  The "example.py" is the script in using the "dominantColors.py" module. It loads an image, and use the dominantColors() function to 
  obtain the most dominant colors of the image. it outputs the [R,G,B] of the most dominant colors depending on the number of clusters.
  saveHistogram() function was also called to show Histogram of the RGB Pixel intensity of the image and generate a csv file and a screenshot
  of the histogram plot.
  plotHistogram() function was called to show the dominant colors present in the image in a plot.
  getAveColor() function is a function to calculate the mean and standard deviation of the RGB pixel intensity of the ROI
  plotCluster() function plots a 3d scatter plot of the dominant colors present in the image.
  pixelColors() function recreates the image using the dominant colors as their pixel intensity.
  
  in the pyimagesearch folder, shapedetector.py is the module for pre proccessing of the images.
  detect_shapes.py program will use segmentation to separate the ROI(paper sensor) from the background to have a black background. and crop the ROI and 
  will create a new image file for the analysis.
