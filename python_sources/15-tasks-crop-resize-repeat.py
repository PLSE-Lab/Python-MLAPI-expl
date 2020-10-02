#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
import json
from pathlib import Path
from tqdm.notebook import tqdm
import inspect

import matplotlib.pyplot as plt
from matplotlib import colors

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))


# In[ ]:


# from: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_one(ax, i,train_or_test,input_or_output):
   
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):     
        plot_one(axs[0,i],i,'train','input')
        plot_one(axs[1,i],i,'train','output')        
    plt.tight_layout()
    plt.show()        
        
    num_test = len(task['test'])
    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))
    if num_test==1: 
        plot_one(axs[0],0,'test','input')
        plot_one(axs[1],0,'test','output')     
    else:
        for i in range(num_test):      
            plot_one(axs[0,i],i,'test','input')
            plot_one(axs[1,i],i,'test','output')  
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()


# In[ ]:


# from: https://www.kaggle.com/nagiss/manual-coding-for-the-first-10-tasks

def get_data(task_filename):
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task

num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
color2num = {c: n for n, c in enumerate(num2color)}

def check_p(task, pred_func):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        sz_out = t_out.shape
        if len(inspect.getargspec(pred_func).args)==1:
            t_pred = pred_func(t_in)
        else:
            t_pred = pred_func(t_in,sz_out)
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
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        if len(inspect.getargspec(pred_func).args)==1:
            t_pred = pred_func(t_in)
        else:
            t_pred = pred_func(t_in,sz_out)
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        axs[2][fig_num].imshow(t_pred, cmap=cmap, norm=norm)
        axs[2][fig_num].set_title(f'Test-{i} pred')
        axs[2][fig_num].set_yticks(list(range(t_pred.shape[0])))
        axs[2][fig_num].set_xticks(list(range(t_pred.shape[1])))
        fig_num += 1


# In[ ]:


def train(task, pred_func):
    try:
        ok = 0
        for i, t in enumerate(task["train"]):
            t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
            sz_out = t_out.shape
            if len(inspect.getargspec(pred_func).args)==1:
                t_pred = pred_func(t_in)
            else:
                t_pred = pred_func(t_in,sz_out)
            if len(t_out)==len(t_pred):
                if len(t_out)==1:
                    if t_pred==t_out:
                        ok += 1
                elif (t_pred==t_out).all():
                    ok += 1
        t_pred = []
        if ok==len(task["train"]):
            for i, t in enumerate(task["test"]):
                t_in = np.array(t["input"]).astype('uint8')
                if len(inspect.getargspec(pred_func).args)==1:
                    t_pred.append(pred_func(t_in))
                else:
                    t_pred.append(pred_func(t_in, sz_out))
                return t_pred
        else:
            return None
    except:
        return None


# In[ ]:


def flattener(pred):
    str_pred = str([row for row in pred])
    return str_pred.replace(', ', '').replace('[[', '|').replace('][', '|').replace(']]', '|')


# # Some useful basic functions

# In[ ]:


def crop_min(a):
    try:
        b = np.bincount(a.flatten(),minlength=10)
        c = int(np.where(b==np.min(b[np.nonzero(b)]))[0])
        coords = np.argwhere(a==c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return a[x_min:x_max+1, y_min:y_max+1]
    except:
        return a  


# In[ ]:


def crop_max(a):
    try:
        b = np.bincount(a.flatten(),minlength=10)
        b[0] = 255
        c = np.argsort(b)[-2]
        coords = np.argwhere(a==c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return a[x_min:x_max+1, y_min:y_max+1]
    except:
        return a


# In[ ]:


def resize_o(a,s):
    try:
        nx,ny = s[1]//a.shape[1],s[0]//a.shape[0]
        return np.repeat(np.repeat(a, nx, axis=0), ny, axis=1)
    except:
        return a


# In[ ]:


def resize_c(a):
    c = np.count_nonzero(np.bincount(a.flatten(),minlength=10)[1:])
    return np.repeat(np.repeat(a, c, axis=0), c, axis=1)


# In[ ]:


def resize_2(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


# In[ ]:


def repeat_1(a,s):
    try:
        si = a.shape
        nx,ny = s[1]//si[1],s[0]//si[0]
        return np.tile(a,(nx,ny))
    except:
        return a


# In[ ]:


def repeat_2(a):
    return np.tile(a,a.shape)


# ## Some usage examples

# In[ ]:


task = get_data(str(evaluation_path / evaluation_tasks[325]))
check_p(task, resize_c)


# In[ ]:


task = get_data(str(evaluation_path / evaluation_tasks[148]))
check_p(task, resize_2)


# In[ ]:


task = get_data(str(training_path / training_tasks[222]))
check_p(task, resize_o)


# In[ ]:


task = get_data(str(evaluation_path / evaluation_tasks[311]))
check_p(task, repeat_2)


# In[ ]:


task = get_data(str(training_path / training_tasks[268]))
check_p(task, resize_c)


# In[ ]:


task = get_data(str(training_path / training_tasks[288]))
check_p(task, resize_o)


# In[ ]:


task = get_data(str(training_path / training_tasks[13]))
check_p(task, crop_min)


# In[ ]:


task = get_data(str(training_path / training_tasks[30]))
check_p(task, crop_min)


# In[ ]:


task = get_data(str(training_path / training_tasks[35]))
check_p(task, crop_min)


# In[ ]:


task = get_data(str(training_path / training_tasks[48]))
check_p(task, crop_min)


# In[ ]:


task = get_data(str(training_path / training_tasks[309]))
check_p(task, crop_min)


# In[ ]:


task = get_data(str(training_path / training_tasks[383]))
check_p(task, crop_min)


# In[ ]:


task = get_data(str(training_path / training_tasks[262]))
check_p(task, crop_max)


# In[ ]:


task = get_data(str(training_path / training_tasks[299]))
check_p(task, crop_max)


# In[ ]:


task = get_data(str(evaluation_path / evaluation_tasks[251]))
check_p(task, repeat_1)


# In[ ]:


task = get_data(str(training_path / training_tasks[176]))
check_p(task, crop_max)


# In[ ]:


task = get_data(str(training_path / training_tasks[87]))
check_p(task, crop_max)


# In[ ]:


task = get_data(str(evaluation_path / evaluation_tasks[224]))
check_p(task, crop_max)


# ## Test Submission

# In[ ]:


submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')


# In[ ]:


for output_id in tqdm(submission.index):
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = test_path / str(task_id + '.json')
    if f.is_file():
        task = get_data(f)
        data = task['test'][pair_id]['input']   
        pred_1 = flattener(data)
        for oper in ['crop_min','crop_max','resize_o','resize_c','resize_2','repeat_1','repeat_2']:
            data = train(task, globals()[oper])
            if data: 
                pred_1 = flattener(data)
                break
        
    data = task['test'][pair_id]['input']
    pred_2 = flattener(data)
    data = [[5 if i==0 else i for i in j] for j in data]
    pred_3 = flattener(data)
    
    # concatenate and add to the submission output
    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 
    submission.loc[output_id, 'output'] = pred


# In[ ]:


submission.to_csv('submission.csv')

