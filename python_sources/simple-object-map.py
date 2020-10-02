#!/usr/bin/env python
# coding: utf-8

# ### Simple object map
# 
# 
# train 7/400
# 
# evals 0/400
# 
# LB: ????
# 
# based on [this method](https://www.kaggle.com/szabo7zoltan/howtofindmosaics)
# 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pathlib import Path

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)


# In[ ]:


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))
eval_tasks = sorted(os.listdir(evaluation_path))


# In[ ]:


solved_id=set()
solved_eva_id=set()


# In[ ]:


T = training_tasks
Trains = []
for i in range(400):
    task_file = str(training_path / T[i])
    task = json.load(open(task_file, 'r'))
    Trains.append(task)
    
E = eval_tasks
Evals= []
for i in range(400):
    task_file = str(evaluation_path / E[i])
    task = json.load(open(task_file, 'r'))
    Evals.append(task)


# In[ ]:


cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()

def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()


# In[ ]:



def plot_picture(x):
    plt.imshow(np.array(x), cmap = cmap, norm = norm)
    plt.show()
    
def Create(task, task_id = 0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output

def Defensive_Copy(A): 
    if type(A)!=list:
        A=A.tolist()
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()


# Object split

# In[ ]:


BACKGROUND=0
def _get_bound(img0):
    img=np.array(img0)
    h, w = img.shape
    x0 = w - 1
    x1 = 0
    y0 = h - 1
    y1 = 0
    for x in range(w):
        for y in range(h):
            if img[y, x] == BACKGROUND:
                continue
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
    return x0, x1, y0, y1
def get_bound_image(img0):
    x0,x1,y0,y1=_get_bound(img0)
    img=np.array(img0)
    return img[y0:y1+1,x0:x1+1].tolist()
BACKGROUND = 0

_neighbor_offsets = {
    4: [(1, 0), (-1, 0), (0, 1), (0, -1)],
    8: [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
}
def _expand_region_indices(img, i, j, neighbor=4):
    h, w = img.shape
    seed_color = img[i, j]
    idx = np.zeros_like(img, dtype=np.bool)
    region = []
    region.append((i, j))
    while len(region) > 0:
        ii, jj = region.pop()
        if img[ii, jj] != seed_color:
            continue
        idx[ii, jj] = True
        for di, dj in _neighbor_offsets[neighbor]:
            ni, nj = ii + di, jj + dj
            if ni >= 0 and ni < h and nj >= 0 and nj < w                     and not idx[ni, nj]:
                region.append((ni, nj))
    return idx
def _expand_region_indices01(img, i, j, neighbor=4):
    h, w = img.shape
    seed_color = 1
    idx = np.zeros_like(img, dtype=np.bool)
    region = []
    region.append((i, j))
    while len(region) > 0:
        ii, jj = region.pop()
        if img[ii, jj] == 0:
            continue
        idx[ii, jj] = True
        for di, dj in _neighbor_offsets[neighbor]:
            ni, nj = ii + di, jj + dj
            if ni >= 0 and ni < h and nj >= 0 and nj < w                     and not idx[ni, nj]:
                region.append((ni, nj))
    return idx



def _split_object(img0, neighbor=4):
    regions = []
    img=np.array(img0)
    mem = np.zeros_like(img, dtype=np.bool)
    h, w = img.shape
    for j in range(w):
        for i in range(h):
            p = img[i, j]
            if p == BACKGROUND or mem[i, j]:
                continue
            conn_idx = _expand_region_indices(img, i, j, neighbor)
            mem[conn_idx] = True
            splitimage=np.where(conn_idx, img, BACKGROUND)

            (minx,maxx,miny,maxy)=_get_bound(splitimage)
            split_object=(splitimage[miny:maxy+1,minx:maxx+1]).tolist()
            
            
            regions.append({'start': (miny, minx), 'obj':split_object})
    return regions

def _split_object01(img0, neighbor=4):
    regions = []
    img=np.array(img0)
    mem = np.zeros_like(img, dtype=np.bool)
    h, w = img.shape
    for j in range(w):
        for i in range(h):
            p = img[i, j]
            if p == BACKGROUND or mem[i, j]:
                continue
            conn_idx = _expand_region_indices01(img, i, j, neighbor)
            mem[conn_idx] = True
            splitimage=np.where(conn_idx, img, BACKGROUND)

        
            (minx,maxx,miny,maxy)=_get_bound(splitimage)
            split_object=(splitimage[miny:maxy+1,minx:maxx+1]).tolist()
            
            
            regions.append({'start': (miny, minx), 'obj':split_object})
    return regions

    
    
    return color_image
def split_object(img):
    return _split_object(img, neighbor=4)
def split_object8(img):
    return _split_object(img, neighbor=8)
def split_object01(img):
    return _split_object01(img, neighbor=4)
def split_object801(img):
    return _split_object01(img, neighbor=8)


# In[ ]:


def inoutmap3(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    name_dic={}
    obj_dic={}
    for x, y in zip(Input,Output):
        x_array=np.array(x)
        y_array=np.array(y)
        n0,m0=len(x),len(x[0])
        a=split_object801(x)
        if len(a)>10 or len(a)==0:
            return -1
        obj=[]
        for i in range(len(a)):
            obj.append(a[i]["obj"])
        di_obj=[]
        for i in range(len(obj)):
            if obj[i] not in di_obj:
                di_obj.append(obj[i])
        



        for k in range(len(di_obj)):
            example=di_obj[k]
            n,m=len(example),len(example[0])
            for i in range(n0-n+1):
                for j in range(m0-m+1):
                    if x_array[i:i+n,j:j+m].tolist()==example:
                        
                        if i-1>=0 and i+n+1<=n0 and j-1>=0 and j+m+1<=m0:
                            
                            tmp_x=x_array[i-1:i+n+1,j-1:j+m+1]
                            tmp_y=y_array[i-1:i+n+1,j-1:j+m+1]
                            if str(tmp_x) not in obj_dic:
                                name_dic[str(tmp_x)]=tmp_x
                                obj_dic[str(tmp_x)]=tmp_y

    return name_dic,obj_dic


# In[ ]:


def solve_inoutmap3(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    
    if inoutmap3(basic_task)==-1:
        return -1
    name_dic,object_dict=inoutmap3(basic_task)
    
    for x, y in zip(Input,Output):
        
        if len(x)!=len(y) or len(x[0])!=len(y[0]):
            return -1
        x_array=np.array(x)
        x_array_pad=np.pad(x_array,((1,1),(1,1)),'constant',constant_values = (0,0)) 
        x_array_copy=x_array.copy()
        x_pad1=np.pad(x_array_copy,((1,1),(1,1)),'constant',constant_values = (0,0)) 
        y_array=np.array(y)
        n0,m0=len(x),len(x[0])
        
        a=split_object801(x)
        
        if len(a)>10 or len(a)==0:
            return -1
        obj=[]
        for i in range(len(a)):
            obj.append(a[i]["obj"])
        di_obj=[]
        for i in range(len(obj)):
            if obj[i] not in di_obj:
                di_obj.append(obj[i])
                
        for k in range(len(di_obj)):
            example=di_obj[k]
            n,m=len(example),len(example[0])
            for i in range(n0-n+1):
                for j in range(m0-m+1):
                    if x_array_pad[i+1:i+n+1,j+1:j+m+1].tolist()==example:
                        
                        if str(x_array_pad[i:i+n+1+1,j:j+m+1+1]) not in object_dict:
                            return -1
                       

                        try:
                            x_pad1[i:i+n+1+1,j:j+m+1+1]=object_dict[str(x_array_pad[i:i+n+1+1,j:j+m+1+1])]
                            
                        except:
                            return-1
                        
        res=x_pad1[1:-1,1:-1]
        
        
        if not (res==y_array).all():
            return -1
        
    
    Test_Case_array=np.array(Test_Case)
    Test_Case_array_copy=Test_Case_array.copy()
    Test_Case_pad=np.pad(Test_Case_array_copy,((1,1),(1,1)),'constant',constant_values = (0,0)) 
    Test_Case_pad1=np.pad(Test_Case_array_copy,((1,1),(1,1)),'constant',constant_values = (0,0)) 
    n0,m0=len(Test_Case),len(Test_Case[0])
    a=split_object801(Test_Case)
    obj=[]
    
    for i in range(len(a)):
        obj.append(a[i]["obj"])
    di_obj=[]
    for i in range(len(obj)):
        if obj[i] not in di_obj:
            di_obj.append(obj[i])
    for k in range(len(di_obj)):
            example=di_obj[k]
            n,m=len(example),len(example[0])
            for i in range(n0-n+1):
                for j in range(m0-m+1):
                    if  Test_Case_pad[i+1:i+n+1,j+1:j+m+1].tolist()==example:
                        
                        if str(Test_Case_pad[i:i+n+1+1,j:j+m+1+1]) not in object_dict:
                            return -1
                        
                       
                        Test_Case_pad1[i:i+n+1+1,j:j+m+1+1]=object_dict[str(Test_Case_pad[i:i+n+1+1,j:j+m+1+1])]
    return Test_Case_pad1[1:-1,1:-1].tolist()


# ### Result
# 

# In[ ]:


for i in range(400):
    task = Trains[i]
    k = len(task['test'])

    for j in range(k):        
        basic_task = Create(task, j)
        a=solve_inoutmap3(basic_task)
        if a!=-1 :
            print(i,j)
            plot_task(task)
            plot_picture(a)
            solved_id.add(i)


# In[ ]:


list(solved_id)


# It is naive thinking,many place can improve.
# 
# But,it may have potential.
# 
# You can combine it to your DSL.
# 
# This is my first open Notebook in kaggle.Please tell me what can improve.
# 
# 

# In[ ]:




