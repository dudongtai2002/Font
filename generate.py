__author__ = 'dudongtai'

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

wordfile = open('test/ChineseCharacter.txt')
word = wordfile.read()[1:]


for file in os.listdir("Fonts/"):
    if (file.endswith(".ttf")):
        if not os.path.exists("test/"+file[0:-4]+ '/'):
            os.makedirs("test/"+file[0:-4]+ '/')
        for x in word:
            font=ImageFont.truetype("Fonts/"+file,49)
            im=Image.new("L",(49,49),"WHITE")
            dr=ImageDraw.Draw(im)
            dr.text((0,0),x,font=font)
            name="test/"+file[0:-4]+ '/' + x + ".png"
            im.save(name)



"""
Testing
for x in word:
    font = ImageFont.truetype("Fonts/simsun.ttf", 49)
    im = Image.new("L", (49, 49), "WHITE")
    dr = ImageDraw.Draw(im)
    dr.text((0, 0), x, font=font)
    name = "test/" + 'simsun/' + x + ".png"
    im.save(name)
"""