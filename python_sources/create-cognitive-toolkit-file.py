# The following script creates two encoded files suitable as input for CNTK (www.cntk.ai).
# The format follows the structure "|labels 0 0 0 0 0 0 |features 0 0 0 0 0 0 0 0 0 0 0"
# where labels is an array of length 35 (this corresponds to the x axis of the grid) where
# all x-coordinates of monsters which where killed are encoded as 1. "Features" is an array of
# length 9800 (35x20x14 - 35 = lenght of x axis, 20 = length of y-axis, 14 = number of different
# objects).

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import sys  
import collections
import json
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

fnew = open("gamedata_training_cntk.txt", "a")
fnew_test = open("gamedata_test_cntk.txt", "a")

with open("../input/gamedata.csv", encoding='utf8') as f:
    
    p=0
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        p=p+1
        line = ""
        try:
            gamemap = json.loads(row[2])
            hit = json.loads(row[4])
        except ValueError:
            continue

        if(hit == "" or len(str(gamemap)) < 10 or len(str(hit)) < 2):
            continue

        foreground = gamemap[0]
        objects = gamemap[2]
 
        guyspositions = []
        for obj in objects:
            x = obj["x"]
            objecttype = obj["type"]
            if(objecttype == "guy"):
                if(obj["species"] == 1 and "a" in hit):
                    guyspositions.append(x)
                elif(obj["species"] == 2 and "b" in hit):
                    guyspositions.append(x)
                elif(obj["species"] == 3 and "c" in hit):
                    guyspositions.append(x)
                elif(obj["species"] == 4 and "d" in hit):
                    guyspositions.append(x)
                elif(obj["species"] == 5 and "e" in hit):
                    guyspositions.append(x)

        line += "|labels "

        for i in range(0, 35):
            if(i in guyspositions):
                line += "1 "
            else:
                line += "0 "


        #shift monsters to end of block
        for obj in objects:
            objecttype = obj["type"]
            if(objecttype == "guy"):
                x = obj["x"]
                y = obj["y"]
                newY = 0
                oSet = False
                for q in range(1,19):
                    if(oSet == True):
                        break
                    for tile in foreground:
                        if(y+q > 19):
                            break
                            oSet = True
                        if(tile["x"] == x and tile["y"] == y+q):
                            newY = y+q-1
                            oSet = True
                            break       
                obj["y"] = newY


        line += "|features "

        for k in range(0, 14):
            for j in range(0, 20):
                for i in range(0, 35):

                    if(k == 0):
                        for tile in foreground: 
                            x = tile["x"]
                            y = tile["y"]
                            objSet = False
                            if(i == x and j == y):
                                line += "1 "
                                objSet = True
                                break

                        if(objSet == False):
                            line += "0 "

                    elif(k == 1):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objecttype = obj["type"]
                            objSet = False
                            if(i == x and j == y and objecttype == "guy"):
                                line += "1 "
                                objSet = True
                                break
                                
                        if(objSet == False):
                            line += "0 "

                    elif(k == 2):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "ball"):
                                    if(obj["rotation"] == "left"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 3):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "ball"):
                                    if(obj["rotation"] == "right"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 4):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "ball"):
                                    if(obj["rotation"] == "none"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 5):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "smallball"):
                                    if(obj["rotation"] == "left"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 6):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "smallball"):
                                    if(obj["rotation"] == "right"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 7):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "smallball"):
                                    if(obj["rotation"] == "none"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 8):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "bigball"):
                                    if(obj["rotation"] == "left"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 9):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "bigball"):
                                    if(obj["rotation"] == "right"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 10):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objSet = False
                            objecttype = obj["type"]
                            try:
                                if(i == x and j == y and objecttype == "bigball"):
                                    if(obj["rotation"] == "none"):
                                        line += "1 "
                                        objSet = True
                                        break
                            except KeyError:
                                break
                        if(objSet == False):
                            line += "0 "

                    elif(k == 11):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objecttype = obj["type"]
                            objSet = False
                            if(i == x and j == y and objecttype == "spin"):
                                line += "1 "
                                objSet = True
                                break

                        if(objSet == False):
                            line += "0 "


                    elif(k == 12):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objecttype = obj["type"]
                            objSet = False
                            if(i == x and j == y and objecttype == "spring"):
                                line += "1 "
                                objSet = True
                                break

                        if(objSet == False):
                            line += "0 "

                    elif(k == 13):
                        for obj in objects:
                            x = obj["x"]
                            y = obj["y"]
                            objecttype = obj["type"]
                            objSet = False
                            if(i == x and j == y and objecttype == "box"):
                                line += "1 "
                                objSet = True
                                break
                                
                        if(objSet == False):
                            line += "0 "



        line = line[:-1] + "\n"

        if(random.random() <= 0.7):
            fnew.write(line)
        else:
            fnew_test.write(line)