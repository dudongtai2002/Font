"""
This file is used to generate characters.
"""

__author__ = 'dudongtai'
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
from os.path import basename

for file in os.listdir("Fonts/"):
    switch
    font=ImageFont.truetype("Fonts/"+file,49)

    im=Image.new("L",(49,49),"WHITE")
    dr=ImageDraw.Draw(im)
    dr.text((0,0),chr(0x6c38),font=font)
    name="test/"+file[0:-3]+".png"
    im.save(name)