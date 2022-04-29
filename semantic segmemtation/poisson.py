# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:10:40 2022

@author: Ben
"""



import cv2
import math
import random
import PIL
from PIL import Image, ImageOps
from Vector import Vector2

#performs poisson disc sampling
def run(treePath):
    img = PIL.Image.open(treePath)
    img = img.convert('RGBA')
    
    img = img.resize((250,250))
    width, height = img.size
    
    
    screen_offset = 1
    
    r = 7
    k = 30
    
    active = []
    w = r / math.sqrt(2)
    
    
    cols = math.floor(width/w)
    rows = math.floor(height/w)
    
    
    grid = [i for i in range(math.floor(cols * rows))]
    
    for i in range(0, cols*rows):
        grid[i] = None
        
    x = random.randint(0,width-1)
    y = random.randint(0, height-1)
    i = math.floor(x/w)
    j = math.floor(y/w)
    pos = Vector2(x,y)
    
    grid[i+j * cols] = pos
    active.append(pos)
    
    
    def list_splice(target, start, delete_count=None, *items):
        if delete_count == None:
            delete_count = len(target) - start
    
        total = start + delete_count
        removed = target[start:total]
        target[start:total] = items
        return removed
    
    iteration = 0
    while len(active) > 0:
        iteration+=1
        if(iteration > 10 and len(active) == 1):
            break
        index = random.randint(0, len(active)-1)
        current_position = active[index]
        found = False
        for n in range(k):
            offset = Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
            new_magnitude = random.randint(r, r*2)
            offset = offset.set_magnitude(new_magnitude)
            offset.x = offset.x + current_position.x
            offset.y = offset.y + current_position.y
    
            col = math.floor(offset.x/w)
            row = math.floor(offset.y/w)
    
            if row < rows-screen_offset and col < cols-screen_offset and row > screen_offset and col > screen_offset:
                viable = True
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        nindex = math.floor( col + i + (row+j) * cols)
    
                        neighbour = grid[nindex];
                        if neighbour is not None:
                            dist = math.sqrt((offset.x - neighbour.x) ** 2 + (offset.y - neighbour.y) ** 2)
                            if dist < r:
                                viable = False
    
    
                if viable is True:
                    found = True
                    grid[math.floor(col + row * cols)] = Vector2(offset.x, offset.y)
                    active.append(Vector2(offset.x, offset.y))
                    break
        if found is not True:
            list_splice(active, index+1, 1)
            
    
    
            for cell in grid:
                if cell is not None:
                    if(img.getpixel((math.floor(cell.x),math.floor(cell.y))) == (255,255,255,255)):
                        img.putpixel((math.floor(cell.x),math.floor(cell.y)), (255,0,0,255))
                        with open("Foliage/treeposition.txt", "a") as f:
                            f.write(str(math.floor(cell.x)) + ","+str(math.floor(cell.y))+"\n")
        
    
    
    
#run("Predictions/ProcessedMasks/Trees/maskaustin36_4000_4500.png")