# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 19:25:18 2022

@author: Ben
"""

import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi


def getOrientation(pts, img):
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
  cntr = (int(mean[0,0]), int(mean[0,1])) 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
  return np.rad2deg(angle)


def FindBuildings(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (250,250))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thresh)
    
    ROI_number = 0
    cnts = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('image', image)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if(h > 10 and w > 10):
            newx = x
            newy = y
            newwidth = w
            newheight = h
            m = cv2.moments(c)
            centreX = int(m["m10"]/ m["m00"])
            centreY = int(m["m01"] / m["m00"])
            #cv2.rectangle(image, (newx, newy), (newx + newwidth, newy + newheight), (0,0,255), 1)
            rotation = getOrientation(c, image)
            original = image.copy()
            ROI = original[y:y+h, x:x+w]
            cv2.imwrite('Buildings/buildingtest('+str(centreX)+','+str(centreY)+','+str(rotation)+').png'.format(ROI_number), ROI)
            ROI_number += 1



