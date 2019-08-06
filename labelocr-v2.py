# Labelocr - v2
# For V* H&E slides (and BT?)
# By Daniel Li, 2019-07-11
import argparse as ap
import numpy as np
from PIL import Image as im, ImageFilter as imf, ImageEnhance as ime
import cv2
import openslide as osi
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

originalImage = im.open("/home/luckypotato/github/slidelabelocr-v2/test2/10.tiff")
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

# Begin deskewing
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

# find white area
thresh = cv2.threshold(rotatedImageCV, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
whitePixels = np.where(thresh > 0)
firstCropImage = rotatedImage.crop((min(whitePixels[1]) - 5, min(whitePixels[0]) - 5, min(whitePixels[1]) + 560, min(whitePixels[0]) + 400))
firstCropImage.show()
# find white area again
whitePixels2 = np.where(gPIL2CV(firstCropImage) > 0)
secondCropImage = firstCropImage.crop((min(whitePixels2[1]), min(whitePixels2[0]), max(whitePixels2[1]), max(whitePixels2[0])))
secondCropImage.show()

# Use magic numbers to find area of relevant labels
relevantTextImage = secondCropImage.crop((30, 46, 520, 108))
relevantTextImage.show()
bigRelevantTextImage = relevantTextImage.resize((490*4, 62*4))
bigRelevantTextImage.show()
bigBlurredRelevantTextImage = bigRelevantTextImage.filter(imf.GaussianBlur(radius=5)).point(lambda x: 0 if x < 180 else 255)
bigBlurredRelevantTextImage.show()
bigFirstPartBlurredRelevantTextImage = bigBlurredRelevantTextImage.crop((0, 0, 1300, 62*4))
bigSecondPartBlurredRelevantTextImage = bigBlurredRelevantTextImage.crop((1300, 0, 490*4, 62*4))
bigFirstPartBlurredRelevantTextImage.show()
bigSecondPartBlurredRelevantTextImage.show()
data2 = pytesseract.image_to_data(bigSecondPartBlurredRelevantTextImage, output_type = pytesseract.Output.DICT)
data = pytesseract.image_to_data(bigFirstPartBlurredRelevantTextImage, output_type = pytesseract.Output.DICT)
print(data)
print(data2)
