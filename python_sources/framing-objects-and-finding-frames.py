#!/usr/bin/env python
# coding: utf-8

# ## Framing Objects and Finding Frames
# Many given examples seem to include some concept of 'Object'. This code tries to draw a frame around possible objects and thereby identifying them. The code finds the object by making frames from the backgroundcolor.
# 
# For objects of the same color to be identified seperately they must be one pixel apart. The object frame is also identified outside the picture area, but not drawn. This can give some visual interpretation challenges. 
# 
# Some given examples use a frame to point out features in the picture. Almost the same routine can also be used to find these frames (and some more).
# 
# Again I have to thank @Bo for his extremely nice plotting routines: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines. I have changed the background color to white and introduced black (10) for my own markings. In the examples these frames are marked with back (10).
# 
# With these routines the aim seems to move a litlle closer. To be really useful parameters and perhaps the code will need some tweaking for specific usecases.
# 
# Two errors corrected in ver 6 - see comments in code below
# 

# In[ ]:





# In[ ]:


import numpy as np
from copy import copy, deepcopy

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors


# In[ ]:



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path   = data_path / 'training'
training_tasks_list = sorted(os.listdir(training_path))

def get_matrix(task,i,train_or_test,input_or_output):
    #get matrix from json structur
    return task[train_or_test][i][input_or_output]

def readTasks(files_list,path,exampletype):
    training_tasks   = {}
    for i,taskname in enumerate(files_list):
        training_tasks[i] = {}
        training_tasks[i]["filename"] = taskname
        training_tasks[i]["file"]     = str(path/taskname)

        with open(training_tasks[i]["file"], 'r') as f:
            jtask = json.load(f)
            for kind in ['train']: #,'test']:
                matrix1_list = []
                matrix2_list = []
                num_exam = len(jtask[kind])
                for j in range(num_exam):  
                    matrix1_list.append(np.array(get_matrix(jtask,j,'train','input')))
                    matrix2_list.append(np.array(get_matrix(jtask,j,'train','output')))
                training_tasks[i]["matrix1_list"] = matrix1_list 
                training_tasks[i]["matrix2_list"] = matrix2_list 
    return training_tasks

training_tasks   = readTasks(training_tasks_list,training_path,'training')
                                    


# In[ ]:


cmap = colors.ListedColormap(
    ['#FFFFFF', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#000000'])
norm = colors.Normalize(vmin=0, vmax=10)
# 0:white, 1:blue, 2:red, 3:greed, 4:yellow,
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown, 10,black

def plot_matrix(ax,matrix):
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='grey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    


# ## Identifying and Framing

# In[ ]:


def grow_ones(matrix_in):
    # sums calculated one row at a time from y=0 and downwards
    matrix  = matrix_in.copy()
    for i in range(len(matrix)-2,-1,-1):
        matrix[i] = (matrix[i] + matrix[i+1])*matrix[i] 
    return matrix

def rowIdentifyFrames(irow,matrix,frames,mindim):
    if not 1 in matrix[irow]:
        return False
    if mindim < 3:
        mindim = 3
    dim1 = len(matrix[0])
    j = 0                    # Error corrected ver 6 Tip from Higepon thank you
    while j < dim1-1:
        j += 1

        if not matrix[irow][j] == 1:
            continue
        # go left and right. 
        height = 3    #lowest permissible height
        while height >= 3:
            jleft = j-1
            found = False
            while jleft >= 0:
                if matrix[irow][jleft] == 0:
                    break
                elif matrix[irow][jleft] >= height:
                    found = True
                    break
                else:
                    jleft -=1
            if not found:
                break
            jright = j+1
            found = False
            while jright < dim1:
                if matrix[irow][jright] == 0:
                    break
                elif matrix[irow][jright] >= height:
                    found = True
                    break
                else:
                    jright +=1
            if not found:
                break
            # check can close
            found = True
            for k in range(jleft+1,jright):
                if matrix[irow+height-1][k]== 0:
                    found = False
                    break
            # Are dimensions allowed ?
            if found: 
                if height >= mindim and jright-jleft+1 >= mindim                    and not (irow == 0 and jleft == 0 and irow+height-1 == len(matrix)-1 and jright ==len(matrix[0])-1):
                    frames.append((irow,irow+height-1,jleft,jright))
                for k in range(j+1,dim1-2):   # new code  in ver 6
                    if matrix[irow][k] > 1:
                        jright = k 
                        break
                j = jright
                break
            else:
                if irow+height-1 < len(matrix)-1:
                    height  += 1
                else:
                    height = 0
    return False
                    
def matrixIdentifyFrames(matrix,mindim):
    frames = []
    for i in range(len(matrix)):
        rowIdentifyFrames(i,matrix,frames,mindim)
    return frames

def unpad(matrix, pad):
    nx, ny = matrix.shape                                                                                                                                                                                                                                                         
    return  matrix[pad:nx-pad,pad:ny-pad]  

def matrixFrameObjectsFromBackground(matrix_in,backgroundcolor,mindim=3,maxcount=999):
    pad_size = 1                #automatisk padding, fjernes igen
    matrix_padded = np.pad(matrix_in, ((pad_size,pad_size),(pad_size,pad_size))                               , "constant", constant_values=backgroundcolor)
        
    matrix01 = (matrix_padded == backgroundcolor).astype(int)
    matrix = grow_ones(matrix01)
    frames = matrixIdentifyFrames(matrix,mindim)
    for i in range(len(frames)):
        frames[i] = (frames[i][0]-1,frames[i][1]-1,frames[i][2]-1,frames[i][3]-1)
    return frames

def matrixIdentifyFramesOfColor(matrix_in,color,mindim=3,maxcount=999):
    matrix01 = (matrix_in == color).astype(int)
    matrix = grow_ones(matrix01)
    frames = matrixIdentifyFrames(matrix,mindim)
    return frames


# ### Plotting frame that can be partly outside the picture

# In[ ]:


def plotFrameToMatrix(frame,matrix,color):
#        print("plotting frame",frame,"color",color)
        # 4 sides
        if frame[0] >= 0:
            matrix[frame[0]:frame[0]+1,frame[2]+1:frame[3]] = color
        if frame[1] < len(matrix):
            matrix[frame[1]:frame[1]+1,frame[2]+1:frame[3]] = color
        if frame[2] >= 0:
            matrix[frame[0]+1:frame[1],frame[2]:frame[2]+1] = color
        if frame[3] < len(matrix[0]):
            matrix[frame[0]+1:frame[1],frame[3]:frame[3]+1] = color
        # 4 corners
        if frame[0] >= 0 and  frame[2] >= 0:
            matrix[frame[0]:frame[0]+1,frame[2]:frame[2]+1] = color            
        if frame[1] < len(matrix) and  frame[2] >= 0:
            matrix[frame[1]:frame[1]+1,frame[2]:frame[2]+1] = color
        if frame[0] >= 0 and  frame[3] < len(matrix[0]):
            matrix[frame[0]:frame[0]+1,frame[3]:frame[3]+1] = color          
        if frame[1] < len(matrix) and  frame[3] <len(matrix[0]):
            matrix[frame[1]:frame[1]+1,frame[3]:frame[3]+1] = color
    


# ## A few Test examples
# The plots below has four rows for each task:
# 
# 1. Original matrix
# 2. Objects identified and framed
# 3. Identified frames in the given matrix
# 4. The given solution to evaluate whether the identified stuff could be useful

# In[ ]:


mindim   = 3
maxcount = 100
plotrows = 4
example_text = ""
examples = [28,30,32,37,43,97,99,207]
ex_texts = ["Frames, No Objects","Objects, No Frames","Objects, No Frames","Objects, No Frames","Both Frame and Objects"              ,"Objects, No Frames","Both Frame and Objects","Frames, Two very small Objects"]
example_text = "Frames and Objects"

for itask,task in training_tasks.items():
    if not itask in examples: 
        continue
    else:
        example_text = ex_texts[examples.index(itask)]
        
    plotsetup = False
    matrix1_list = task["matrix1_list"]
    matrix2_list = task["matrix2_list"]
    for i in range(len(matrix1_list)):
        if i > 2:
            break
        matrix1 = matrix1_list[i]
        matrix2 = np.zeros_like(matrix1)
        matrix2 = matrix1.copy()
        matrix3 = np.zeros_like(matrix1)
        matrix3 = matrix3 # + 10
        matrix4 = matrix2_list[i]

        backgroundcolor = 0
        foundframes = False
        if True:
            color = backgroundcolor
            objframes = matrixFrameObjectsFromBackground(matrix1,color,mindim=mindim,maxcount=999)
            if color == backgroundcolor:
                newcolor = 10
            else:
                newcolor = color
            for frame in objframes:
                foundframes = True
                plotFrameToMatrix(frame,matrix2,newcolor)
        
        if True:
            colors   = np.unique(matrix1, return_counts=False)
            for color in colors:
                if color == backgroundcolor:
                    continue
                frames    = matrixIdentifyFramesOfColor(matrix1,color,mindim=mindim,maxcount=999)
                for frame in frames:
                    foundframes = True
                    plotFrameToMatrix(frame,matrix3,color)
        if not foundframes and i == 0:
            break
        if not plotsetup:
            plotsetup = True
            print("Task no",itask,"-- ",example_text," --")
            fig, axs = plt.subplots(plotrows, 3, figsize=(3*3,3*2))
        for k in range(plotrows):
            r = k    #//3
            s = i%3
            if k == 0:
                plot_matrix(axs[r,s],matrix1)
            elif k == 1:
                plot_matrix(axs[r,s],matrix2)
            elif k == 2:
                plot_matrix(axs[r,s],matrix3)
            elif k == 3:
                plot_matrix(axs[r,s],matrix4)
    if plotsetup:
        plt.tight_layout()
        plt.show() 


# In[ ]:




