# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:22:59 2022

@author: Ben
"""

import cv2
import numpy as np
import os



def boundingBoxCorners(path, savePath):
    for file in os.listdir(path):
        img = cv2.imread(path+file)
        dimensions = img.shape
        height = img.shape[0]
        width = img.shape[1]
        corners = []
        corners.append([0,0,0])
        corners.append([width,0,0])
        corners.append([width,0,height])
        corners.append([0,0,height])
        corners.append([0,20,0])
        corners.append([width,20,0])
        corners.append([width,20,height])
        corners.append([0,20,height])
        
        fileName = file.split('.')
        completeName = os.path.join(savePath,fileName[0]+'.txt')
        
        for i in range(len(corners)):
            corner = str(corners[i][0])+','+str(corners[i][1])+','+str(corners[i][2])+'\n'
            with open(completeName, 'a') as f:
                f.write(corner)
                
def boundingBoxArea(path, savePath):
    for file in os.listdir(path):
        img = cv2.imread(path+file)
        dimensions = img.shape
        height = img.shape[0]
        width = img.shape[1]
        
        fileName = file.split('.')
        completeName = os.path.join(savePath,fileName[0]+'.txt')
        
        with open(completeName, 'a') as f:
            f.write(str(width)+'\n')
            f.write(str(height)+'\n')
        
        
def FindCorners(path, savePath):
    for file in os.listdir(path):
        img = cv2.imread(path+file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,5,3,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        
        fileName = file.split('.')
        try:
            corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
            for i in range(1, len(corners)):
                completeName = os.path.join(savePath,fileName[0]+'.txt')
                with open(completeName, 'a') as f:
                    f.write(str(int(corners[i,0])) + ",0," + str(int(corners[i,1]))+'\n')
                    f.write(str(int(corners[i,0])) + ",20," + str(int(corners[i,1]))+'\n')
                #cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (0,255,0), 2)
            img[dst>0.1*dst.max()]=[0,0,255]
        except:
            os.remove(path+file)
            print("failed to produce corners on building, removing from directpry")
        
#findCorners('E:/Users/Ben/Documents/CCTP/AI/semantic segmemtation/Buildings/')