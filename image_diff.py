# python image_diff.py --first images/original_01.png --second images/modified_01.png

from skimage.measure import compare_ssim
from PIL import Image, ImageChops
import argparse
import imutils
import cv2
import requests
import difference


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["first"],1)
imageB = cv2.imread(args["second"],1)

# imageA = image1.resize((100,100))
# imageB = image2.resize((100,100))
########### difference ratio calculation using module difference.py ##########
ratio = difference.diff(args["first"],args["second"])
print("Difference Ratio =",ratio)


######### pil image difference  and chopping

img1 = Image.open(args["first"])
img2 = Image.open(args["second"])

diff_image = ImageChops.difference(img1,img2)
if diff_image.getbbox():
    diff_image.show()

########## PIL second spot the difference 
point_table = ([0] + ([255] * 255))

def black_or_b(a, b):
    diff = ImageChops.difference(a, b)
    diff = diff.convert('L')
    diff = diff.point(point_table)
    new = diff.convert('RGB')
    new.paste(b, mask=diff)
    return new

a = Image.open(args["first"])
b = Image.open(args["second"])
c = black_or_b(a, b)
c.show()
c.save('result.png')



# convert the images to grayscale
# next line make error
# OpenCV Error: Assertion failed (scn == 3 || scn == 4) in cvtColor
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
# contours = contours[1] if imutils.is_cv3() else contours[0]

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
# cv2.imshow("Diff", diff)
# cv2.imshow("Thresh", thresh)
diff_image.save("42_diff.png")
cv2.waitKey(0)
