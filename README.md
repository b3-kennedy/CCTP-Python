This repository contains the machine learning and image processing techniques needed to produce a 3D representation of a satellite image. The Buildings, Foliage, Geometry and Predictions folders inside the semantic segmentation folder contain data from running the 'ExtractFeatures.py' file.

To create your own environment:
1. Delete the contents of the 'Buildings' and 'Foliage' folder. You will also need to delete the contents of the folders inside the 'Geometry' and 'Predictions' folders.
2. Swap out the image in the 'Image' folder with your own satellite image.
3. Run the 'ExtractFeatures.py' file, this will produce text files needed within Unity to create an environment.
4. Within Unity create a new scene and drag the 'World' prefab into the scene.
5. Within the Resources folder create a new folder and with the subdirectories 'Buildings' and 'Other'.
6. Drag the text files within the 'Foliage' folder from the python project into the 'Other' folder within Unity.
7. Drag the folders in the 'Geometry' folder from the python project into the 'Buildings' folder within Unity.
8. On the 'World' asset within the scene locat the 'Get Text Files' script and change the paths to correspond with folder you created within the 'Resources folder'.


