Libraries Used:
Language - Python3
Open CV, Pillow, Numpy, Scikit-Image, SK learn, scipy, Qt(easy GUI)

Structure and Working:
Read the images - EasyGUI's fileopenbox() function is used to allow the user to choose images.



Downscale image - In our downscaleImages() function we extract the width and height of the images and resize them to equal dimensions.

Finding Difference Ratio - Difference ratio is found using Pillow library’s Image Chopping which is present in difference.py script. 

Difference between images - We used SSIM(Structural Similarity Index) approach so that we can visualize the differences between images using openCV2, scikit-image and scipy and then draw a contour around all the differences discovered.

Applying Template Matching - By using template matching we can search to see if patches exist on the second image compared to the first. Thus we were able to isolate only the true differences in the two images.

Getting the best patches - The getBestPatches() function takes in a list of contours and a threshold, and compares the two images using template matching, It then returns all the contours which are smaller than the threshold.

Applying Bounding Boxes - Rectangular boxes are drawn over the identified best patches and images are generated.
