import os, glob, sys
import openslide as osi
from PIL import Image as im
os.chdir("/media/luckypotato/Seagate_6tb/2019-07-19 YipLuad")
filesNameList = glob.glob("*.svs")
print(filesNameList)
i = 0
for file in filesNameList:
	i = i + 1
	originalImage = osi.OpenSlide(file)
	labelImage = originalImage.associated_images['label']
	labelImage.save("/home/luckypotato/github/slidelabelocr-v2/test2/" + str(i) + ".tiff")
