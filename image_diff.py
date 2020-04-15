#IOT-project - SPOT THE DIFFERENCE(Main Application) #################
# Made by Nischal and Purvi 


from skimage.measure import compare_ssim
from PIL import Image, ImageChops, ImageDraw
from math import atan
import argparse
import imutils
import cv2
import numpy as np
import requests
import difference
import easygui

# scaling function

def downscaleImages(img1, img2):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    maxWidth = 1000.0
    if width1 > maxWidth or width2 > maxWidth:
        if width1 > maxWidth and width1 > width2:
            scale = maxWidth / width1
        else:
            scale = maxWidth / width2

        newImg1 = cv2.resize(src=img1, dsize=(
            int(width1 * scale), int(height1 * scale)), interpolation=cv2.INTER_AREA)
        newImg2 = cv2.resize(src=img2, dsize=(
            int(width2 * scale), int(height2 * scale)), interpolation=cv2.INTER_AREA)
    else:
        newImg1 = img1.copy()
        newImg2 = img2.copy()

    return newImg1, newImg2

# getting Patches of differences
def getAllPatches(mask):
    patches = []

    _, contours, _ = cv2.findContours(
        image=mask.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        arcPercentage = 0.01
        epsilon = cv2.arcLength(curve=contour, closed=True) * arcPercentage
        corners = cv2.approxPolyDP(curve=contour, epsilon=epsilon, closed=True)
        x, y, w, h = cv2.boundingRect(points=corners)
        currentArea = w * h

        # Ignore points
        if currentArea > 1:
            patches.append((x, y, w, h))

    return patches

#Template Matching to get the Best Match
#1 means most similar and 0 means most different 
def getBestMatch(img, patch):
    result = cv2.matchTemplate(
        image=img, templ=patch, method=cv2.TM_CCOEFF_NORMED)

    (_, value, _, (x, y)) = cv2.minMaxLoc(src=result)

    return ((x, y), value)

#among all the the patches selecting the best matching patches with a Threshold of 0.8
def getBestPatches(sourceImg, checkImg, patches, threshold=0.8):
    bestPatches = []
    for (x, y, w, h) in patches:
        patch = sourceImg[y: y + h, x: x + w]
        ((mX, mY), matchValue) = getBestMatch(checkImg, patch)
        if matchValue < threshold:
            bestPatches.append((x, y, w, h))

    return bestPatches


def getBestPatchesAuto(sourceImg, checkImg, patches):
    print('Eliminating false-positives')
    bestPatches = getBestPatches(sourceImg, checkImg, patches)
    return bestPatches


############# Easy GUI for selecting the Files
applicationSwitch = True
file1Ver = False
file2Ver = False
while applicationSwitch:
    title = 'Difference Checker'
    instruction = 'Please load image 1 and 2 then begin.'

    if file1Ver == False or file2Ver == False:
        buttons = ['Load Image 1', 'Load Image 2']
    else:
        buttons = ['Load Image 1', 'Load Image 2', 'Begin Application']

    selection = easygui.indexbox(msg = instruction, title = title, choices = buttons)

    if selection == 0:
        file1 = easygui.fileopenbox()
        imageA = cv2.imread(file1)
        img1 = Image.open(file1)

        if img1 is None:
            easygui.msgbox("Please select image files only!")
        else:
            file1Ver = True
    elif selection == 1:
        file2 = easygui.fileopenbox()
        imageB = cv2.imread(file2)
        img2 = Image.open(file2)
        if img2 is None:
            easygui.msgbox("Please select image files only!")
        else:
            file2Ver = True
    elif selection == 2:

#Commented Code for using application through commandline 
#in case gui is not working in you machine the uncomment this code

# python image_diff.py --first images/original_01.png --second images/modified_01.png
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True,
# 	help="first input image")
# ap.add_argument("-s", "--second", required=True,
# 	help="second")
# args = vars(ap.parse_args())
# # load the two input images
# imageA = cv2.imread(args["first"],1)
# imageB = cv2.imread(args["second"],1)

        imageA, imageB = downscaleImages(imageA, imageB)


########### difference ratio calculation using module difference.py ##########
        ratio = difference.diff(img1,img2)
        print("Difference Ratio =",ratio)


        diff_image = ImageChops.difference(img1,img2)
        bbox = diff_image.getbbox()

        if bbox:
            diff_image.show()


# PIL(Pillow) image difference  and chopping
        point_table = ([0] + ([255] * 255))

        def black_or_b(a, b):
            diff = ImageChops.difference(a, b)
            diff = diff.convert('L')
            diff = diff.point(point_table)
            new = diff.convert('RGB')
            new.paste(b, mask=diff)
            return new

        c = black_or_b(img1, img2)
        c.show()
        c.save('result.png')

# convert the images to grayscale
        
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two images
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        

# mask = getMask(grayA, grayB)
#Finding the Patches among all patches obtained
        patches = getAllPatches(thresh)
        bestPatches = getBestPatchesAuto(grayA, grayB, patches)

# loop over the contours
        for (x, y, w, h) in bestPatches:
    # compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	
	        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
        cv2.imshow("Original", imageA)
        cv2.imshow("Modified", imageB)
        cv2.imshow("Diff", diff)
        cv2.imshow("Thresh", thresh)
        diff_image.save("42_diff.png")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        applicationSwitch = False
    else:
        applicationSwitch = False