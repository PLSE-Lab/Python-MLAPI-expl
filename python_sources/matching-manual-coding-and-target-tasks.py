#!/usr/bin/env python
# coding: utf-8

# # What is this notebook?
# Inspired by [@nagiss](https://www.kaggle.com/nagiss)'s [great notebook](https://www.kaggle.com/nagiss/manual-coding-for-the-first-10-tasks), I prepared functions to check if task-solving functions (algorithm) are applicable for target tasks. I modified some codes from [@nagiss](https://www.kaggle.com/nagiss)'s [notebook](https://www.kaggle.com/nagiss/manual-coding-for-the-first-10-tasks) and [@meaninglesslives](https://www.kaggle.com/meaninglesslives)'s [notebook].(https://www.kaggle.com/meaninglesslives/using-decision-trees-for-arc). I also prepared functions (see 2-1. Functions for Pattern-1 tasks) to solve the following tasks as an example.
# 
# * [Training dataset] (7/400): 0520fde7.json, 1b2d62fb.json, 3428a4f5.json, 6430c8c4.json, 99b1bc43.json, ce4f8723.json, f2829549.json
# 
# * [Evaluation dataset] (6/400): 0c9aba6e.json, 195ba7dc.json, 34b99a2b.json, 506d28a5.json, 5d2a5c43.json, e133d23d.json
# 
# ***This notebook is for the [Abstraction and Reasoning Challenge](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview) competition.***

# # 1. Preparation

# In[ ]:


import numpy as np
import pandas as pd
import operator
import os
import json
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    
from pathlib import Path


# In[ ]:


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))
print(len(training_tasks), len(evaluation_tasks), len(test_tasks))


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


def get_data(task_filename):
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task

num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
color2num = {c: n for n, c in enumerate(num2color)}


# In[ ]:


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# In[ ]:


sample_sub = pd.read_csv(data_path/'sample_submission.csv')
sample_sub = sample_sub.set_index('output_id')
sample_sub.head()


# In[ ]:


example_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
sample_sub.output=flattener(example_grid)
sample_sub.head()


# In[ ]:


def check(task, learn_func, pred_func):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig_num = 0
    param_ls=[]#Added
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        t_pred, param = learn_func(t_in,t_out)
        if param==False:
            print('Not suitable')
            plt.close('all')
            return
        param_ls.append(param)
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        axs[2][fig_num].imshow(t_pred, cmap=cmap, norm=norm)
        axs[2][fig_num].set_title(f'Train-{i} pred')
        axs[2][fig_num].set_yticks(list(range(t_pred.shape[0])))
        axs[2][fig_num].set_xticks(list(range(t_pred.shape[1])))
        fig_num += 1
    if len(set(param_ls))==1:
        param=param_ls[0]
        for i, t in enumerate(task["test"]):
            t_in = np.array(t["input"])
            t_pred = pred_func(t_in,param)
            axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
            axs[0][fig_num].set_title(f'Test-{i} in')
            axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
            axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
            axs[1][fig_num].imshow(t_pred, cmap=cmap, norm=norm)
            axs[1][fig_num].set_title(f'Test-{i} pred')
            axs[1][fig_num].set_yticks(list(range(t_pred.shape[0])))
            axs[1][fig_num].set_xticks(list(range(t_pred.shape[1])))
            fig_num += 1
            print(t_pred)
            if mode=='test':
                print('mode test')
                sub_ls=flattener(t_pred.tolist())
                sample_sub.loc[f'{test_tasks[iteration][:-5]}_{i}','output'] = sample_sub.loc[f'{test_tasks[iteration][:-5]}_{i}','output'] +' '+sub_ls
        plt.show()
        
    else:
        print('NOT THIS TYPE')
        return


# ## 1-1. Function to flip and rotate a predicted output to match the true output
# The flip_rot() function will test whether predicted outputs can match the fliped/rotated true output. 

# In[ ]:


def flip_rot(matrix_pred,matrix_out):
    flip_0=np.flip(matrix_pred, 0)
    
    matrix_out=matrix_out.tolist()
    matrix_out=flattener(matrix_out)

    if matrix_out==flattener(matrix_pred.tolist()):
        print('!!!match1!!!')
        return 'y'
    elif matrix_out==flattener((np.rot90(matrix_pred)).tolist()):
        print('!!!match2!!!')
        return 'np.rot90(y)'
    elif matrix_out==flattener((np.rot90(matrix_pred, 2)).tolist()):
        print('!!!match3!!!')
        return 'np.rot90(y, 2)'
    elif matrix_out==flattener((np.rot90(matrix_pred, 3)).tolist()):
        print('!!!match4!!!')
        return 'np.rot90(y, 3)'
    
    elif matrix_out==flattener(flip_0.tolist()):
        print('!!!match5!!!')
        return 'np.flip(y, 0)'
    elif matrix_out==flattener((np.rot90(flip_0)).tolist()):
        print('!!!match6!!!')
        return 'np.rot90(np.flip(y, 0))'
    elif matrix_out==flattener((np.rot90(flip_0, 2)).tolist()):
        print('!!!match7!!!')
        return 'np.rot90(np.flip(y, 0),2)'
    elif matrix_out==flattener((np.rot90(flip_0, 3)).tolist()):
        print('!!!match8!!!')
        return 'np.rot90(np.flip(y, 0),3)'
    else:
        #print('not_match')
        return 'y'


# # 2. Codes to solve some tasks

# # 2-1. Functions for Pattern-1 tasks 
# The functions below can solve the following tasks (let's say Pattern-1 tasks).
# 
# * [Training dataset] (7/400): 0520fde7.json, 1b2d62fb.json, 3428a4f5.json, 6430c8c4.json, 99b1bc43.json, ce4f8723.json, f2829549.json
# 
# * [Evaluation dataset] (6/400): 0c9aba6e.json, 195ba7dc.json, 34b99a2b.json, 506d28a5.json, 5d2a5c43.json, e133d23d.json

# In[ ]:


def pattern_1_learn(x,t_out):    
    def split_by_gray_line(arr):
        H, W = arr.shape
        Y = [-1]
        for y in range(H):
            if (arr[y, :]==color3).all():
                Y.append(y)
                
        Y.append(H)
        X = [-1]
        
        for x in range(W):
            if (arr[:, x]==color3).all():
                X.append(x)
        
        X.append(W)
        res = [[arr[y1+1:y2, x1+1:x2] for x1, x2 in zip(X[:-1], X[1:])] for
                       y1, y2 in zip(Y[:-1], Y[1:])]

        return res
    
    def change_color(arr, d):
        res = arr.copy()
        for k, v in d.items():
            res[arr==k] = v
        return res

    for color2 in range(10): #output color
        for color3 in range(10): #border color
            for color5 in range(10): #input background
                for color6 in range(10): #output background

                    x_split = split_by_gray_line(x)
                    if len(x_split)==1 and len(x_split[0])==2:
                        x1, x2 = x_split[0]
                    elif len(x_split)==2 and len(x_split[0])==1:
                        x1, x2 = x_split
                        x1=x1[0]
                        x2=x2[0]
                    else:
                        continue

                    if x1.shape!=x2.shape:
                        continue
                    
                    rules=['operator.or_(x1,x2)','operator.and_(x1,x2)',
                           'operator.xor(x1,x2)']

                    for rule in rules:

                        zero_one = lambda t: 0 if (t == color5) else 1
                        vfunc = np.vectorize(zero_one) 
                        try:
                            x1=vfunc(x1)
                            x2=vfunc(x2)
                        except ValueError as e:
                            print(e)
                            continue
          
                        try:
                            y = eval(rule)
                        except ValueError as e:
                            continue
                            
                        change_color_v2 = lambda t: color6 if (t == 0) else color2
                        vfunc = np.vectorize(change_color_v2) 
                        y=vfunc(y)
                        
                        #--For rotation and flip, just in case.
                        f_r=flip_rot(y,t_out)
                        y=eval(f_r)
                        #-----

                        if flattener(t_out.tolist())==flattener(y.tolist()):
                            param=(color2,color3,color5,color6,rule,f_r)
                            return y, param
                
    #print('Return False')
    return _, False

def pattern_1_pred(x,param):
    color2,color3,color5,color6,rule,f_r = param
    
    def split_by_gray_line(arr):
        H, W = arr.shape
        Y = [-1]
        for y in range(H):
            if (arr[y, :]==color3).all():
                Y.append(y)
        Y.append(H)
        X = [-1]
        for x in range(W):
            if (arr[:, x]==color3).all():
                X.append(x)
        X.append(W)
        
        res = [[arr[y1+1:y2, x1+1:x2] for x1, x2 in zip(X[:-1], X[1:])] for
                       y1, y2 in zip(Y[:-1], Y[1:])]

        return res
    
    def change_color(arr, d):
        res = arr.copy()
        for k, v in d.items():
            res[arr==k] = v
        return res

    x_split = split_by_gray_line(x)


    if len(x_split)==1 and len(x_split[0])==2:
        x1, x2 = x_split[0]
    elif len(x_split)==2 and len(x_split[0])==1:
        x1, x2 = x_split
        x1=x1[0]
        x2=x2[0]
    else:
        print('errrrrr')

    zero_one = lambda t: 0 if (t == color5) else 1
    vfunc = np.vectorize(zero_one) 
    x1=vfunc(x1)
    x2=vfunc(x2)
    try:
        y = eval(rule)
    except ValueError as e:
        print(e)
    
    change_color_v2 = lambda t: color6 if (t == 0) else color2
    vfunc = np.vectorize(change_color_v2) 
    y=vfunc(y)
    y=eval(f_r)
    return y


# # 2-2. Apply the functions to the 400 training datasets

# In[ ]:


mode='train'
for iteration in range(400):
    task = get_data(str(training_path / training_tasks[iteration]))
    acc = check(task, pattern_1_learn, pattern_1_pred)
    print(f'Task({iteration}):',training_tasks[iteration])


# # 2-3. Apply the functions to the 400 evaluation datasets

# In[ ]:


mode='eval'
for iteration in range(400):
    task = get_data(str(evaluation_path / evaluation_tasks[iteration]))
    acc = check(task, pattern_1_learn, pattern_1_pred)
    print(f'Task({iteration}):',evaluation_tasks[iteration])


# # 2-4. Apply the functions to the 100 test datasets

# In[ ]:


mode='test'
for iteration in range(100):
    task = get_data(str(test_path / test_tasks[iteration]))
    acc = check(task, pattern_1_learn, pattern_1_pred)
    print(f'Task({iteration}):',test_tasks[iteration])


# # 3. Create submission CSV

# In[ ]:


pd.options.display.max_rows = 999
display(sample_sub)
len(sample_sub)
sample_sub.to_csv('submission.csv')

