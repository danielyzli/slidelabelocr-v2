# Labelocr - v2
# For V* H&E slides (and BT?)
# By Daniel Li, 2019-07-11
# Edited by Samantha Lee, 2019-08-11
import argparse as ap
import numpy as np
from PIL import Image as im, ImageFilter as imf, ImageEnhance as ime
import cv2
# import openslide as osi
import pytesseract
import os, glob, sys
import re
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter

def gPIL2CV(pilImage):
	return np.array(pilImage)

#originalImage = osi.OpenSlide("/home/luckypotato/github/slidelabelocr-v2/test/20190703T172158-747166483.tiff")
#labelImage = originalImage.associated_images['label']
#bwLabelImage = labelImage.convert('L')

originalImage = im.open("C:/Users/Sam/Documents/GSE - machine learning/test2/4.tiff")
#originalImage.show()
bwLabelImage = originalImage.convert('L')

#brighten image and enhance contrast
bwLabelImage = ime.Brightness(bwLabelImage).enhance(0.3)
bwLabelImage = ime.Contrast(bwLabelImage).enhance(15)
bwLabelImage = ime.Brightness(bwLabelImage).enhance(0.3)
bwLabelImage = ime.Contrast(bwLabelImage).enhance(15)


# Binarize
# cvBwLabelImage = gPIL2CV(bwLabelImage)
# cvBinaryLabelImage = cv2.adaptiveThreshold(cvBwLabelImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
											# cv2.THRESH_BINARY, 11, -8)
# binaryLabelImage = im.fromarray(cvBinaryLabelImage)
# binaryLabelImage.show()

bwLabelImage = bwLabelImage.rotate(-90, expand = 1)

# Begin deskewing, based on histogram area minimization of black ink
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

delta = 0.5
limit = 45
angles = np.arange(-limit, limit+delta, delta)
scores = []
for angle in angles:
    hist, score = find_score(bwLabelImage, angle)
    scores.append(score)
#plt.plot(scores, angles)
best_score = max(scores)
print(best_score)
best_angle = angles[scores.index(best_score)]
print(best_angle)
# correct skew
rotatedImageCV = inter.rotate(bwLabelImage, best_angle, reshape=False, order=0)
rotatedImage = im.fromarray(rotatedImageCV)
rotatedImage.show()

# find white area to do a primary boxing of the area based on estimate of top left most white pixel
# this eliminates any irrelevant white pixels
thresh = cv2.threshold(rotatedImageCV, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
whitePixels = np.where(thresh > 0)
leftBuffer = 5
topBuffer = 5
rightEstimate = 560
bottomEstimate = 400
firstCropImage = rotatedImage.crop((min(whitePixels[1]) - leftBuffer, min(whitePixels[0]) - topBuffer, min(whitePixels[1]) + rightEstimate, min(whitePixels[0]) + bottomEstimate))
firstCropImage.show()
# find white area again, to refine the framing based on the actual corner pixels
whitePixels2 = np.where(gPIL2CV(firstCropImage) > 0)
secondCropImage = firstCropImage.crop((min(whitePixels2[1]), min(whitePixels2[0]), max(whitePixels2[1]), max(whitePixels2[0])))
secondCropImage.show()

# Use magic numbers to find area of relevant labels

leftText = 30
topText = 46
rightText = 520
bottomText = 108

relevantTextImage = secondCropImage.crop((leftText, topText, rightText, bottomText))
relevantTextImage.show()


## -- Begin the pampering of pytesseract using multiple methods --
#scaling the image to pamper pytesseract
scale = 4
bigRelevantTextImage = relevantTextImage.resize(((rightText - leftText)*scale, (bottomText - topText)*scale))
# bigRelevantTextImage.show()

#confidences = []
#for blurRadius in np.arange(0, 8, 0.1):

#blurring the image a tad to bridge the broken letters
blurRadiusFactor = 5
bigBlurredRelevantTextImage = bigRelevantTextImage.filter(imf.GaussianBlur(radius=blurRadiusFactor)).point(lambda x: 0 if x < 180 else 255)
# bigBlurredRelevantTextImage.show()

#cut between the sample ID and slide number
cutBetween = 1300 #somewhere in the middle of the blank space
bigFirstPartBlurredRelevantTextImage = bigBlurredRelevantTextImage.crop((0, 0, cutBetween, (bottomText - topText)*scale))
bigSecondPartBlurredRelevantTextImage = bigBlurredRelevantTextImage.crop((cutBetween, 0, (rightText - leftText)*scale, (bottomText - topText)*scale))
bigFirstPartBlurredRelevantTextImage.show()
bigSecondPartBlurredRelevantTextImage.show()
BFPBRTImageCV = gPIL2CV(bigFirstPartBlurredRelevantTextImage)
BSPBRTImageCV = gPIL2CV(bigSecondPartBlurredRelevantTextImage)

#further cleaning of letters with erosion to make the letters crisp
kernel = np.ones((5,5),np.uint8)
erodedImage1 = im.fromarray(cv2.bitwise_not(cv2.erode(cv2.bitwise_not(BFPBRTImageCV),kernel,iterations = 4)))
erodedImage1.show()
erodedImage2 = im.fromarray(cv2.bitwise_not(cv2.erode(cv2.bitwise_not(BSPBRTImageCV),kernel,iterations = 2)))
erodedImage2.show()
biggerErodedImage2CV = cv2.bitwise_not(cv2.copyMakeBorder(cv2.bitwise_not(gPIL2CV(erodedImage2)), 30, 30, 300, 0, cv2.BORDER_CONSTANT))

img = im.fromarray(biggerErodedImage2CV)
img.show()


data1 = pytesseract.image_to_string(erodedImage1, output_type = pytesseract.Output.DICT, config = "-c tessedit_char_whitelist= 1234567890" )
data2 = pytesseract.image_to_string(img, output_type = pytesseract.Output.DICT, config = "-c tessedit_char_whitelist= 1234567890")


print(data1)
print(data2)
