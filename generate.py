__author__ = 'dudongtai'

"""
This file is used to generate characters.
"""

__author__ = 'Dongtai Du & zhifan yin'
import json
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
from os.path import basename
import numpy as np
import random
import theano
import theano.tensor as T

"""
wordfile = open('test/ChineseCharacter.txt')
word = wordfile.read()[1:]


for file in os.listdir("Fonts/"):
    if (file.endswith(".ttf") or file.endswith(".TTF")):

        if not os.path.exists("test/"+file[0:-4]+ '/'):
            os.makedirs("test/"+file[0:-4]+ '/')
        for x in word:
            font=ImageFont.truetype("Fonts/"+file,49)
            im=Image.new("L",(50,50),"WHITE")
            dr=ImageDraw.Draw(im)
            dr.text((0,0),x,font=font)
            name="test/"+file[0:-4]+ '/' + x + ".png"
            im.save(name)

"""



def generatedata(imagesize,trainsame,traindifferent,testnumber):
    traininput1=np.zeros((imagesize*imagesize,trainsame+traindifferent))
    traininput2=np.zeros((imagesize*imagesize,trainsame+traindifferent))
    testinput1=np.zeros((imagesize*imagesize,testnumber))
    testinput2=np.zeros((imagesize*imagesize,testnumber))   #1,2 is are all input, the reason I wrote them in a separate way
    #is that it's convinent for the network to train them, since the first 3 layer of NN should be deal with them respectively
    y_train=np.zeros(trainsame+traindifferent)
    y_test=np.zeros(testnumber)
    f1=open('ChineseCharacter.txt','r')
    letters=f1.readline()
    letters=list(letters)
    random.shuffle(letters)
    fonts=[]
    for file in os.listdir("Fonts/"):
        if (file.endswith(".ttf") or file.endswith(".TTF")):
            fonts.append(file)
    for i in range(0,trainsame):   #generate the same data
        tempfont=fonts[i%len(fonts)]
        font=ImageFont.truetype("Fonts/"+tempfont,36)
        im=Image.new("L",(36,36),"WHITE")
        draw=ImageDraw.Draw(im)
        draw.text((0,0),letters[i%len(letters)],(0),font=font)
           #for testing
        traininput2[:,i]=np.array(im).flatten()
        img=Image.new("L",(36,36),"WHITE")
        draw=ImageDraw.Draw(img)
        draw.text((0,0),"永",(0),font=font)

        traininput1[:,i]=np.array(img).flatten()
        y_train[i]=1
    for i in range(trainsame,trainsame+traindifferent):   #generate the different data
        tempfont=fonts[i%len(fonts)]
        font=ImageFont.truetype("Fonts/"+tempfont,36)

        tempfont1=random.choice(fonts)
        while (tempfont1==tempfont):
            tempfont1=random.choice(fonts)
        font1=ImageFont.truetype("Fonts/"+tempfont1,36)

        im=Image.new("L",(36,36),"WHITE")
        draw=ImageDraw.Draw(im)
        draw.text((0,0),letters[i%len(letters)],(0),font=font)
        traininput2[:,i]=np.array(im).flatten()
        if(i==10500):
            im.save("trainsame.png")
        img=Image.new("L",(36,36),"WHITE")
        draw=ImageDraw.Draw(img)
        draw.text((0,0),"永",(0),font=font1)
        if(i==10500):
            img.save("originaltrain1.png")  #for testing
        traininput1[:,i]=np.array(img).flatten()
        y_train[i]=0
    random.shuffle(letters)
    for i in range(0,testnumber):
        if(random.choice([True,False])):
            font=ImageFont.truetype("Fonts/"+random.choice(fonts),36)
            font1=ImageFont.truetype("Fonts/"+random.choice(fonts),36)
            if(font==font1):
                y_test[i]=1
            else: y_test[i]=0
        else:
            font=ImageFont.truetype("Fonts/"+random.choice(fonts),36)
            font1=font
            y_test[i]=1
        im=Image.new("L",(36,36),"WHITE")
        img=Image.new("L",(36,36),"WHITE")
        draw=ImageDraw.Draw(im)
        draw1=ImageDraw.Draw(img)
        draw.text((0,0),letters[i%len(letters)],(0),font=font)
        draw1.text((0,0),"永",(0),font=font1)
        testinput1[:,i]=np.array(img).flatten()
        testinput2[:,i]=np.array(im).flatten()
    return(traininput1,traininput2,testinput1,testinput2,y_train,y_test)




def shared_dataset(data_x1, data_x2,data_y):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x1 = theano.shared(np.asarray(data_x1, dtype=theano.config.floatX))
    shared_x2 = theano.shared(np.asarray(data_x2, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x1,shared_x2,T.cast(shared_y, 'int32')


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