import makeprediction
import boundingbox
import triangulation3D
import cornerdetection
import datasetloading
import pixelloop
import poisson

import os
import glob






#datasetloading.train()

def Predictions():
    print("Making Predictions...")
    makeprediction.Prediction("Images/", "Models/BuildingDropout18Endfix", "Predictions/Buildings/" + "buildings" + ".png")
    makeprediction.Prediction("Images/", "Models/TreeDropout18End", "Predictions/Trees/" + "trees" + ".png")
    makeprediction.Prediction("Images/", "Models/GrassDropout18End", "Predictions/Grass/" + "grass" + ".png")
    makeprediction.Prediction("Images/", "Models/RoadDropout18End", "Predictions/Roads/" + "roads" + ".png")

    
def ProcessBuildings():
    print("Processing Buildings...")
    boundingbox.FindBuildings("Predictions/Buildings/buildings.png")
    cornerdetection.boundingBoxCorners("Buildings/", "Geometry/boundingverts")
    cornerdetection.boundingBoxArea("Buildings/", "Geometry/boundingarea/")
    cornerdetection.FindCorners("Buildings/", "Geometry/verts")
    triangulation3D.GetGeometry("Geometry/verts/", "Geometry/tris")
    triangulation3D.GetGeometry("Geometry/boundingverts/", "Geometry/boundingtris/")

def ProcessTrees():
    print("Processing Trees...")
    pixelloop.processMask("Predictions/Trees/", "Predictions/ProcessedTreeMask/","treePrediction.png")
    poisson.run("Predictions/ProcessedTreeMask/treePrediction.png")
    pixelloop.colourDistribution("Predictions/ProcessedTreeMask/", "treearea.txt", "Foliage/")


def ProcessGrass():
    print("Processing Grass...")
    pixelloop.processMask("Predictions/Grass/", "Predictions/ProcessedGrassMask/", "grassPrediction.png")
    pixelloop.colourDistribution("Predictions/ProcessedGrassMask/", "grassarea.txt", "Foliage/")


def ProcessRoads():
    print("Processing Roads...")
    pixelloop.processMask("Predictions/Roads/", "Predictions/ProcessedRoadMask/", "roadPrediction.png")
    pixelloop.colourDistribution("Predictions/ProcessedRoadMask/", "roadarea.txt", "Foliage/")
    
    




Predictions()
ProcessBuildings()
ProcessTrees()
ProcessGrass()
ProcessRoads()



# #process roads
# pixelloop.processMask()
# pixelloop.colourDistribution()

# #process grass
# pixelloop.processMask()
# pixelloop.colourDistribution()

# #process trees
# treePositionFile = "treePos"
# treeAreaFile = "treeArea"
# treeMask = fileName
# pixelloop.processMask()
# pixelloop.colourDistribution()
# poisson.run()







