import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
import scipy as sp
from scipy import spatial as sp_spatial
import os


def GetGeometry(path, savePath):
    for file in os.listdir(path):
        vertsArray = []
        openFile = open(path+file)
        lines = openFile.readlines()
        for line in lines:
            splitLine = line.split(',')
            lastElement = splitLine[2].split('\n')
            splitLine[2] = lastElement[0]
            vertsArray.append([int(splitLine[0]),int(splitLine[1]),int(splitLine[2])])
        newArray = np.array(vertsArray)
        try:
            newHull  = sp_spatial.ConvexHull(newArray)
        except:
            print("Geometry cannot be created")
        fileName = file.split('.')
        #print(newIndices)
        completeName = os.path.join(savePath, fileName[0]+'.txt')
        for tri in newHull.simplices:
            string = str(tri).replace('[','')
            string2 = string.replace(']','')
            string3 = string2.replace(' ', '\n')
            with open(completeName, 'a') as f:
                f.write(string3+'\n')
        openFile.close()
        
        
def GetGeometryDelaunay(path, savePath):
    for file in os.listdir(path):
        print(file)
        vertsArray = []
        openFile = open(path+file)
        lines = openFile.readlines()
        for line in lines:
            splitLine = line.split(',')
            lastElement = splitLine[2].split('\n')
            splitLine[2] = lastElement[0]
            vertsArray.append([int(splitLine[0]),int(splitLine[1]),int(splitLine[2])])
        newArray = np.array(vertsArray)
        newHull  = sp_spatial.Delaunay(newArray)
        fileName = file.split('.')
        #print(newIndices)
        completeName = os.path.join(savePath, fileName[0]+'.txt')
        for tri in newHull.simplices:
            string = str(tri).replace('[','')
            string2 = string.replace(']','')
            string3 = string2.replace(' ', '\n')
            with open(completeName, 'a') as f:
                f.write(string3+'\n')
            print(string3)
        openFile.close()





