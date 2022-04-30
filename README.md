This repository contains the machine learning and image processing techniques needed to produce a 3D representation of a satellite image. The Buildings, Foliage, Geometry and Predictions folders inside the semantic segmentation folder contain data from running the 'ExtractFeatures.py' file.

To create your own environment:

Within the Python Project:
1. Delete the contents of the 'Buildings' and 'Foliage' folder within the Python project. You will also need to delete the contents of the folders inside the 'Geometry' and 'Predictions' folders.
2. Swap out the image in the 'Image' folder with your own satellite image.
3. Run the 'ExtractFeatures.py' file, this will produce text files needed within Unity to create an environment.

Within the Unity Project:
1. Within Unity create a new scene and drag the 'World' prefab into the scene.
2. Within the Resources folder create a new folder and with the subdirectories 'Buildings' and 'Other'.
3. Drag the text files within the 'Foliage' folder from the python project into the 'Other' folder within Unity.
4. Drag the folders in the 'Geometry' folder from the python project into the 'Buildings' folder within Unity.
5. On the 'World' asset within the scene locate the 'Get Text Files' script and change the paths to correspond with folder you created within the 'Resources' folder.

Link to Unity project: https://github.com/b3-kennedy/CCTP-Unity


