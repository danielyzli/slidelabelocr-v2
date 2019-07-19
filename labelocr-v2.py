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

originalImage = osi.OpenSlide("/home/luckypotato/github/slidelabelocr-v2/test/20190703T172158-747166483.tiff")
#originalImage = osi.OpenSlide("/home/luckypotato/github/slidelabelocr-v2/test/20190703T171606-202146481.tiff")
labelImage = originalImage.associated_images['label']
bwLabelImage = labelImage.convert('L')

# Binarize
cvBwLabelImage = gPIL2CV(bwLabelImage)
cvBinaryLabelImage = cv2.adaptiveThreshold(cvBwLabelImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
											cv2.THRESH_BINARY, 11, -8)
binaryLabelImage = im.fromarray(cvBinaryLabelImage)
binaryLabelImage.show()

bwLabelImage = bwLabelImage.rotate(90, expand = 1)
binaryLabelImage = bwLabelImage.point(lambda x: 0 if x < 180 else 255)
data = pytesseract.image_to_data(binaryLabelImage, output_type = pytesseract.Output.DICT)
print(data)

def gPIL2CV(pilImage):
	return np.array(pilImage)
