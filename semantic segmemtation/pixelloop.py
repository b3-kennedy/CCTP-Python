# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:23:55 2022

@author: Ben
"""

import PIL
from PIL import Image, ImageOps
import numpy as np
import cv2 
import os

from random import randrange


def renameFile():
     for file in os.listdir("E:/Users/Ben/Documents/CCTP/dataset/masks"):
        img =  PIL.Image.open("E:/Users/Ben/Documents/CCTP/dataset/masks/"+file)
        # name = file.split('mask')
        # print(name[1])
        img.save("E:/Users/Ben/Documents/CCTP/dataset/roadmasks/"+file+".png")

#used to create masks for road dataset
def processMap():
    fileNumber = 0
    for file in os.listdir("E:/Users/Ben/Documents/CCTP/dataset/predictionimages/"):
        print(file)
        fileNumber += 1
        img =  PIL.Image.open("E:/Users/Ben/Documents/CCTP/dataset/predictionimages/"+file)
        pix = img.load()
        width, height = img.size
        for i in range(width):
            for j in range(height):
                colour = img.getpixel((i,j))
                if(colour != (255,255,255) and colour != (255,242,174) and colour != (255,235,162) and colour != (251,253,252)):
                    new_colour = (0,0,0)
                    img.putpixel((i,j), new_colour)
                if(colour[0] > 249 and colour[1] > 249 and colour[2] > 249):
                    img.putpixel((i,j), (255,255,255))
                if(i < 200 and j > height - 55):
                    img.putpixel((i,j), (0,0,0))
                    
                if(i > 1080 and j > height - 55):
                    img.putpixel((i,j), (0,0,0))
        img = ImageOps.grayscale(img)
        img.save("E:/Users/Ben/Documents/CCTP/dataset/predictionmasks/"+"mask"+str(fileNumber)+file)
    

#converts prediction image to black and white
def processMask(directory, savePath, fileName):
    fileNum = 0
    for file in os.listdir(directory):
        fileNum += 1        
        img = cv2.imread(directory+file)
        img = cv2.resize(img,(250,250))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(savePath+fileName,thresh)
                

#converts tree and grass dataset to black and white
def createMask():
    for file in os.listdir("E:/Users/Ben/Documents/CCTP/Splitaerialimaging/grassaccuracytests/"):
        img =  PIL.Image.open("E:/Users/Ben/Documents/CCTP/Splitaerialimaging/grassaccuracytests/"+file)
        pix = img.load()
        width, height = img.size

        for i in range(width):
            for j in range(height):
                colour = img.getpixel((i,j))
                #print(colour)
                if(colour == (255,255,255,255)):
                    print("white")
                else:
                    img.putpixel((i,j), (0,0,0))
        img = ImageOps.grayscale(img)
        img.save("E:/Users/Ben/Documents/CCTP/Splitaerialimaging/"+"mask"+file)
        
#random number sampling
def treeDistribution():
    for file in os.listdir("Predictions/ProcessedMasks/Trees/"):
        img =  PIL.Image.open("Predictions/ProcessedMasks/Trees/"+file)
        img = img.resize((250,250))
        pix = img.load()
        width, height = img.size
    
        for i in range(width):
            for j in range(height):
                random_num = randrange(100)
                img = img.convert('RGB')
                colour = img.getpixel((i,j))
                if(colour == (255,255,255)):
                    print(colour)
                    if(random_num == 3):
                        img.putpixel((i,j),(255,0,0))
                        print(str(i) + " " + str(j))
                        # with open("trees.txt", "a") as f:
                        #   f.write(str(i)+","+str(j)+"\n")
        #img = ImageOps.grayscale(img)
        img.show()
        img.save("E:/Users/Ben/Documents/CCTP/final/randomtree.png")
        
#used to get the area of grass, trees and roads so that the terrain can be coloured
def colourDistribution(path, fileName, savePath):
    for file in os.listdir(path):
        img =  PIL.Image.open(path+file)
        img = img.resize((250,250))
        pix = img.load()
        width, height = img.size
        for i in range(width):
            for j in range(height):
                colour = img.getpixel((i,j))
                if(colour == 255):
                    with open(savePath + fileName, "a") as f:
                        f.write(str(i)+","+str(j)+"\n")
    
    

#createMask()
#processMask()
#treeDistribution()
#colourDistribution("")
#processMap()
#renameFile()
    






