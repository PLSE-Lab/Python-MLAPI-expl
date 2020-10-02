#!/usr/bin/env python
# coding: utf-8

# ## This shows easy function to find the rule of input and output color change

# In[ ]:


import numpy as np
import pandas as pd

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


# In[ ]:


from pathlib import Path

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

train_tasks = sorted(os.listdir(training_path))
eval_tasks = sorted(os.listdir(evaluation_path))


# In[ ]:


def plot_one(ax, i,train_or_test,input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task, data_set = 'train'):
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
        if data_set == 'train':
            plot_one(axs[1],0,'test','output')     
    else:
        for i in range(num_test):      
            plot_one(axs[0,i],i,'test','input')
            if data_set == 'train':
                plot_one(axs[1,i],i,'test','output')  
    plt.tight_layout()
    plt.show() 


# In[ ]:


train_file = str(training_path / train_tasks[15])

with open(train_file, 'r') as f:
    task = json.load(f)


# In[ ]:


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# In[ ]:


plot_task(task)


# ## Write a function that auto detect color change rule

# In[ ]:


def find_count_tow_list(a,b):
    count = 0
    for i,j in enumerate(a):
        if j == b[i]:
            count+=1
    return  count


# In[ ]:


replace_color = {}
for k in range(len(task['train'])):
    target = flattener(task['train'][k]['output'])
    input_str = flattener(task['train'][k]['input'])
    base_count = find_count_tow_list(input_str,target)
    for i in range(10):
        for j in range(10):
            input_strr = input_str.replace(str(i),str(j))
            new_count = find_count_tow_list(target,input_strr)
            if new_count>base_count:
                replace_color[i]=j
#new_str = ''
data = [[replace_color[i] for i in j] for j in task['test'][0]['input']]


# ### Plot function

# In[ ]:


def plot_one(ax, i,train_or_test,input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task, data_set = 'train'):
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
        if data_set == 'train':
            plot_one(axs[1],0,'test','output')     
    else:
        for i in range(num_test):      
            plot_one(axs[0,i],i,'test','input')
            if data_set == 'train':
                plot_one(axs[1,i],i,'test','output')  
    plt.tight_layout()
    plt.show() 


# ### Training set detect and predict

# In[ ]:


train_correct = len(train_tasks)
for i in range(len(train_tasks)):
    train_file = str(training_path / train_tasks[i])
    
    with open(train_file, 'r') as f:
        task = json.load(f)
    
    # Find the replace color rule
    replace_color = {}
    for k in range(len(task['train'])):
        if len(flattener(task['train'][k]['output'])) == len(flattener(task['train'][k]['input'])):
            target = flattener(task['train'][k]['output'])
            input_str = flattener(task['train'][k]['input'])
            base_count = find_count_tow_list(input_str,target)
            for i in range(10):
                for j in range(10):
                    input_strr = input_str.replace(str(i),str(j))
                    new_count = find_count_tow_list(target,input_strr)
                    if new_count>base_count:
                        replace_color[i]=j
        else:
            break
    data = [[replace_color[i] if i in replace_color else i for i in j] for j in task['test'][0]['input']]
    pred_1 = flattener(data)
    
    if pred_1 == flattener(task['test'][0]['output']):
        print('file name = '+train_tasks[i])
        plot_task(task)
        train_correct-=1
    
    # copy paste first rows 
    for j in task['test'][0]['input']:
        if not np.any(j==0):
            temp_data = j
    
    data = [temp_data for j in range(len(task['test'][0]['input']))]  
    pred_2 = flattener(data)
    if pred_2 == flattener(task['test'][0]['output']):
        print('file name = '+train_tasks[i])
        plot_task(task)
        train_correct-=1                   


# In[ ]:


print ("Training set error rate ==> {}, detect tasks {} / {}".format(train_correct/len(train_tasks),len(train_tasks) - train_correct,len(train_tasks)))


# ### Evaluate set detect and predict

# In[ ]:


eval_correct = len(eval_tasks)
for i in range(len(eval_tasks)):
    eval_file = str(evaluation_path / eval_tasks[i])
    
    with open(eval_file, 'r') as f:
        task = json.load(f)
        
    # Find the replace color rule    
    replace_color = {}
    for k in range(len(task['train'])):
        if len(flattener(task['train'][k]['output'])) == len(flattener(task['train'][k]['input'])):
            target = flattener(task['train'][k]['output'])
            input_str = flattener(task['train'][k]['input'])
            base_count = find_count_tow_list(input_str,target)
            for i in range(10):
                for j in range(10):
                    input_strr = input_str.replace(str(i),str(j))
                    new_count = find_count_tow_list(target,input_strr)
                    if new_count>base_count:
                        replace_color[i]=j
        else:
            break
    data = [[replace_color[i] if i in replace_color else i for i in j] for j in task['test'][0]['input']]
    pred_1 = flattener(data)
    if pred_1 == flattener(task['test'][0]['output']):
        print('file name = '+train_tasks[i])
        plot_task(task)
        eval_correct-=1
        
    # copy paste first rows 
    for j in task['test'][0]['input']:
        if not np.any(j==0):
            temp_data = j
    data = [temp_data for j in range(len(task['test'][0]['input']))]  
    pred_2 = flattener(data)
    if pred_2 == flattener(task['test'][0]['output']):
        print('file name = '+train_tasks[i])
        plot_task(task)
        eval_correct-=1                   


# In[ ]:


print ("Evaluate set error rate ==> {}, detect tasks {} / {}".format(eval_correct/len(eval_tasks),len(eval_tasks) - eval_correct,len(eval_tasks)))


# ## Test set Submit

# In[ ]:


submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')
display(submission.head())


# In[ ]:


count = 0
for output_id in submission.index:
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))
    with open(f, 'r') as read_file:
        task = json.load(read_file)
    
    # Find the replace color rule 
    replace_color = {}
    for k in range(len(task['train'])):
        if len(flattener(task['train'][k]['output'])) == len(flattener(task['train'][k]['input'])):
            target = flattener(task['train'][k]['output'])
            input_str = flattener(task['train'][k]['input'])
            base_count = find_count_tow_list(input_str,target)
            for i in range(10):
                for j in range(10):
                    input_strr = input_str.replace(str(i),str(j))
                    new_count = find_count_tow_list(target,input_strr)
                    if new_count>base_count:
                        replace_color[i]=j
        else:
            break
    data = [[replace_color[i] if i in replace_color else i for i in j] for j in task['test'][pair_id]['input']]
    pred_1 = flattener(data)
    
    # copy paste first rows 
    for j in task['test'][0]['input']:
        if not np.any(j==0):
            temp_data = j
    data = [temp_data for j in range(len(task['test'][0]['input']))]  
    pred_2 = flattener(data)

    data = [[0 for i in j] for j in data]
    pred_3 = flattener(data)
    
    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 
    submission.loc[output_id, 'output'] = pred
    

submission.to_csv('submission.csv')

