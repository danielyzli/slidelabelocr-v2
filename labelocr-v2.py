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

# Function for general PIL to CV conversion
def gPIL2CV(pilImage):
	return np.array(pilImage)

# Function for general CV to PIL conversion
def gCV2PIL(cvImage):
	return im.fromarray(cvImage)

# Function for helping deskew by histogram method
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

# Open image (slides)
#originalImage = osi.OpenSlide("/home/luckypotato/github/slidelabelocr-v2/test/20190703T172158-747166483.tiff")
#labelImage = originalImage.associated_images['label']
#grayImage = labelImage.convert('L')

# Open image (testing)
originalImage = im.open("/home/luckypotato/github/slidelabelocr-v2/test2/9.tiff")
grayImage = originalImage.convert('L')

# Brightness and contrast
grayImage = ime.Brightness(grayImage).enhance(0.3)
grayImage = ime.Contrast(grayImage).enhance(15)
grayImage = ime.Brightness(grayImage).enhance(0.3)
grayImage = ime.Contrast(grayImage).enhance(15)

# Rotate image
grayImage = grayImage.rotate(-90, expand = 1)

# Find angle to deskew
delta = 0.5 # 0.5 degree increments
limit = 45 # Check from -45 to 45 degrees rotation
angles = np.arange(-limit, limit+delta, delta)
scores = []
for angle in angles:
    hist, score = find_score(grayImage, angle)
    scores.append(score)
best_score = max(scores)
best_angle = angles[scores.index(best_score)]

# Rotate image to deskew
dskwImage = inter.rotate(grayImage, best_angle, reshape=False, order=0)

# Crop into (expected, padded) bounding box of label sticker
threshImage = cv2.threshold(dskwImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
whiteMask1 = np.where(threshImage > 0)
cropImage1 = gCV2PIL(dskwImage).crop((min(whiteMask1[1]) - 5, min(whiteMask1[0]) - 5, min(whiteMask1[1]) + 560, min(whiteMask1[0]) + 400))

# Refine crop of label sticker
whiteMask2 = np.where(gPIL2CV(cropImage1) > 0)
cropImage2 = cropImage1.crop((min(whiteMask2[1]), min(whiteMask2[0]), max(whiteMask2[1]), max(whiteMask2[0])))

# Crop into relevant label text on the sticker (coordinates should remain consistent across images)
textImage = cropImage2.crop((30, 46, 520, 108))

# Upscale for pytesseract
upscaledTextImage = textImage.resize((490*4, 62*4))

# Blur + threshold, and erode to connect gaps in letters
blurThreshImage = upscaledTextImage.filter(imf.GaussianBlur(radius = 5)).point(lambda x: 0 if x < 180 else 255)
erosionKernel = np.ones((5, 5), np.uint8)
erodedImage = gCV2PIL(cv2.bitwise_not(cv2.erode(cv2.bitwise_not(gPIL2CV(blurThreshImage)), erosionKernel, iterations = 3)))

# Extract first and second text blocks and add padding
textBlockImage1 = erodedImage.crop((0, 0, 1300, 62*4))
textBlockImage1 = gCV2PIL(cv2.bitwise_not(cv2.copyMakeBorder(cv2.bitwise_not(gPIL2CV(textBlockImage1)), 50, 50, 50, 50, cv2.BORDER_CONSTANT)))
textBlockImage2 = erodedImage.crop((1565, 0, 490*4, 62*4))
textBlockImage2 = gCV2PIL(cv2.bitwise_not(cv2.copyMakeBorder(cv2.bitwise_not(gPIL2CV(textBlockImage2)), 50, 50, 50, 50, cv2.BORDER_CONSTANT)))
textBlockImage2.show()

# Set up tesseract path for pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Run tesseract
patientData = pytesseract.image_to_data(textBlockImage1, output_type = pytesseract.Output.DICT, config='-c tessedit_char_whitelist=0123456789BTACDEFG-')
print(patientData)
blockSliceData = pytesseract.image_to_data(textBlockImage2, output_type = pytesseract.Output.DICT, config='-c tessedit_char_whitelist=0123456789ABCDEFG')
print(blockSliceData)
