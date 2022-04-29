# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 22:49:45 2022

@author: Ben
"""




import requests
from io import BytesIO
from PIL import Image


startNumber = 0


        
#used to get ground truth road images for the road dataset
def nonsatellite(coordX, coordY):
        parentDir = "E:/Users/Ben/Documents/CCTP/dataset/predictionimages/"
        url = "https://maps.googleapis.com/maps/api/staticmap?center="+str(coordX)+","+str(coordY)+"&zoom=19&style=feature:all|element:labels|visibility:off&scale=2&size=1920x1080&maptype=terrain&visual_refresh=true&key=APIKEY"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        rgb = image.convert('RGB')
        #image.show()
        rgb.save(parentDir+"Bristolroadnonsat" + str(startNumber) + ".png")


#used to get satellite images to perform predictions on.
def satellite(coordX, coordY):
    parentDir = "E:/Users/Ben/Documents/CCTP/dataset/"
    url = "https://maps.googleapis.com/maps/api/staticmap?center="+str(coordX)+","+str(coordY())+"&zoom=19&style=feature:all|element:labels|visibility:off&scale=2&size=1920x1080&maptype=satellite&visual_refresh=true&key=APIKEY"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    rgb = image.convert('RGB')
    #image.show()
    rgb.save(parentDir+"BristolRoadsat" + str(startNumber) + ".png")

satellite(51.444855333745444, -2.5543053410820438)
nonsatellite(51.444855333745444, -2.5543053410820438)
