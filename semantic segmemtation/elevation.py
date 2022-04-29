import requests
import json

lat = 48.76136813246271
long = -122.48899195925918


latmetre = 0.000009
longmetre = 0.00001

#distance between each sample point, can be used to change width and height of area
latincrement =0.000005
longincremenet = 0.000008

#move area up or down
latOffset = -0.0006
longOffset = 0.0003


length = 250
height = 250

data = [[0 for x in range(length)] for y in range(height)] 

stringlat = str(lat);
splitLat = stringlat.split('.')

def splitWord(word):
    return [char for char in word]

def combineWord(word):
    string = ""
    for x in word:
        string += x
    return string

latDecimals = splitWord(splitLat[1])




latTopLeft = lat + (height/2 * latmetre)
longTopLeft = long - (length/2 * longmetre)


iteration = 0

for i in range (length):
    for j in range (height):
        newLat = str(latTopLeft + latOffset + (j * - latincrement))
        newLong = str(longTopLeft + longOffset + (i * longincremenet))
        iteration += 1
        url = "https://maps.googleapis.com/maps/api/elevation/json?locations="+str(newLat)+"%2C"+str(newLong)+"&key=APIKEY"
        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)
        res = response.json()
        data[i][j] = res['results'][0]['elevation']
        print("elevation: " + str(data[i][j]) + " coordinates: " + str(i) + ", " + str(j))
        with open("elevation.txt", 'a') as file:
            file.write(str(data[i][j]) + '\n')
        iteration+=1
        print(iteration)
        # if((j == 0 and i == 0) or (j == length-1 and i == 0) or (j == 0 and i == height-1) or (i == height-1 and j == length-1)):
        #     print(str(latTopLeft + latOffset + (j * - latincrement)) + ',' + str((longTopLeft + longOffset + (i * longincremenet))))
        
        
        
        
            
            
            
            
            
            
            
            
print(iteration)

