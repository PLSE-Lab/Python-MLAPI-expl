#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy import *
from copy import copy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def reset():
    map =  matrix([[1,1,1,1,1,0,0,0,0,1
                   [1,1,1,1,0,0,0,0,0,1],
                   [1,1,0,0,0,0,0,0,0,1],
                   [1,0,0,0,0,0,0,0,0,1],
                   [1,0,0,0,0,0,0,0,0,1],
                   [1,0,0,0,0,0,0,0,0,1],
                   [1,0,0,0,0,0,0,0,0,1],
                   [1,0,0,0,0,0,0,0,0,1],
                   [1,3,0,0,0,0,0,0,1,1],
                   [1,1,1,1,1,1,1,1,1,1]])
    camera = matrix(array(zeros(map.shape), dtype = "object"))
    current = where(map==3)
    sensor = map[current[0][0]-1:current[0][0]+2, current[1][0]-1:current[1][0]+2]
    
    return map, camera, sensor

def detect():
    global sensor, map
    current = where(map==3)
    sensor = map[current[0][0]-1:current[0][0]+2, current[1][0]-1:current[1][0]+2]

def check(loc):
    if loc == 'f': return sensor[0,1]
    elif loc == 'b': return sensor[2,1]
    elif loc == 'l': return sensor[1,0]
    elif loc == 'r': return sensor[1,2]
    elif loc == 'br': return sensor[2,2]
    
def move(action):
    global map, sensor
    current = where(map==3)
    map[current] = 2
    if action == 'l': map[current[0][0], current[1][0]-1] = 3
    elif action == 'r': map[current[0][0], current[1][0]+1] = 3
    elif action == 'f': map[current[0][0]-1, current[1][0]] = 3
    elif action == 'b': map[current[0][0]+1, current[1][0]] = 3
    elif action == 'br': map[current[0][0]+1, current[1][0]+1] = 3
    elif action == 'bl': map[current[0][0]-1, current[1][0]-1] = 3
    elif action == 'fr': map[current[0][0]-1, current[1][0]+1] = 3
    detect()
    
def show(im):
    fig, ax = plt.subplots(figsize=(2, 2))
    img = ax.imshow(im)
    return fig


# In[ ]:


map, camera, sensor = reset()
change, lane = 0, 0

for t in range(100):
    try:
        if check('f') != 1 and change == 0:
            move('f')
        elif check('f') == 1 and lane == 0:
            move('br')
            lane = 1
        elif change < 2:
            if check('b') != 1:
                move('b')
                change = 1
            else:
                change = 2
        else:
            if change == 2:
                move('fr')
                change = 3
            else:
                if check('b')!= 1:
                    move('b')
                else:
                    lane = 0
                    change = 0
        show(map)
    except:
        break

