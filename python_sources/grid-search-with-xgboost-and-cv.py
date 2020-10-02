#!/usr/bin/env python
# coding: utf-8

# # Credit to Siddharth.

# # Loading Libraries

# In[ ]:


# *- encoding: utf-8 -*-
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd

import os
import json
from pathlib import Path
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.model_selection import GridSearchCV

import pdb


# # Set Paths

# In[ ]:


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'


# # Plotting functions

# In[ ]:


def plot_result(test_input, test_prediction,
                input_shape):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, 2, figsize=(15,15))
    test_input = test_input.reshape(input_shape[0],input_shape[1])
    axs[0].imshow(test_input, cmap=cmap, norm=norm)
    axs[0].axis('off')
    axs[0].set_title('Actual Target')
    test_prediction = test_prediction.reshape(input_shape[0],input_shape[1])
    axs[1].imshow(test_prediction, cmap=cmap, norm=norm)
    axs[1].axis('off')
    axs[1].set_title('Model Prediction')
    plt.tight_layout()
    plt.show()
    
def plot_test(test_prediction, task_name):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, 1, figsize=(15,15))
    axs.imshow(test_prediction, cmap=cmap, norm=norm)
    axs.axis('off')
    axs.set_title(f'Test Prediction {task_name}')
    plt.tight_layout()
    plt.show()


# # For flattening 2D numpy arrays

# In[ ]:


# https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook
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


# # Extract neighbourhood Features

# In[ ]:


def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):

    if cur_row<=0: top = -1
    else: top = color[cur_row-1][cur_col]
        
    if cur_row>=nrows-1: bottom = -1
    else: bottom = color[cur_row+1][cur_col]
        
    if cur_col<=0: left = -1
    else: left = color[cur_row][cur_col-1]
        
    if cur_col>=ncols-1: right = -1
    else: right = color[cur_row][cur_col+1]
        
    return top, bottom, left, right

def get_tl_tr(color, cur_row, cur_col, nrows, ncols):
        
    if cur_row==0:
        top_left = -1
        top_right = -1
    else:
        if cur_col==0: top_left=-1
        else: top_left = color[cur_row-1][cur_col-1]
        if cur_col==ncols-1: top_right=-1
        else: top_right = color[cur_row-1][cur_col+1]   
        
    return top_left, top_right


# # Make features for each train sample

# In[ ]:


def features(task, mode='train'):
    cur_idx = 0
    num_train_pairs = len(task[mode])
    total_inputs = sum([len(task[mode][i]['input'])*len(task[mode][i]['input'][0]) for i in range(num_train_pairs)])
    feat = np.zeros((total_inputs,nfeat))
    target = np.zeros((total_inputs,), dtype=np.int)
    
    global local_neighb
    for task_num in range(num_train_pairs):
        input_color = np.array(task[mode][task_num]['input'])
        target_color = task[mode][task_num]['output']
        nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

        target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])
        
        if (target_rows!=nrows) or (target_cols!=ncols):
            print('Number of input rows:',nrows,'cols:',ncols)
            print('Number of target rows:',target_rows,'cols:',target_cols)
            not_valid=1
            return None, None, 1

        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx,0] = i
                feat[cur_idx,1] = j
                feat[cur_idx,2] = input_color[i][j]
                feat[cur_idx,3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
                feat[cur_idx,7:9] = get_tl_tr(input_color, i, j, nrows, ncols)
                feat[cur_idx,9] = len(np.unique(input_color[i,:]))
                feat[cur_idx,10] = len(np.unique(input_color[:,j]))
                feat[cur_idx,11] = (i+j)
                feat[cur_idx,12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                             j-local_neighb:j+local_neighb]))
        
                target[cur_idx] = target_color[i][j]
                cur_idx += 1
            
    return feat, target, 0


# # Training and Prediction

# In[ ]:


param_grid = {
    "xgb__n_estimators": [10],
    "xgb__learning_rate": [0.1],
    "xgb__early_stopping_rounds": np.array((50, 1000))
}


# In[ ]:


all_task_ids = sorted(os.listdir(test_path))

nfeat = 13
local_neighb = 5
valid_scores = {}
for task_id in all_task_ids:

    task_file = str(test_path / task_id)
    with open(task_file, 'r') as f:
        task = json.load(f)

    feat, target, not_valid = features(task)
    if not_valid:
        print('ignoring task', task_file)
        print()
        not_valid = 0
        continue

    nrows, ncols = len(task['train'][-1]['input']
                       ), len(task['train'][-1]['input'][0])
    # use the last train sample for validation
    val_idx = len(feat) - nrows*ncols

    train_feat = feat[:val_idx]
    val_feat = feat[val_idx:, :]

    train_target = target[:val_idx]
    val_target = target[val_idx:]

    #     check if validation set has a new color
    #     if so make the mapping color independant
    if len(set(val_target) - set(train_target)):
        print('set(val_target)', set(val_target))
        print('set(train_target)', set(train_target))
        print('Number of colors are not same')
        print('cant handle new colors. skipping')
        continue

    xgb = XGBClassifier(n_estimators=100, n_jobs=-1)
   # hgb_pipe = make_pipeline([('xgb', xgb)])


    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1001)
    hgb_grid = GridSearchCV(xgb, param_grid, n_jobs=8, 
         cv=skf, verbose=2, refit=True)
    hgb_grid.fit(feat, target)
#     training on input pairs is done.
#     test predictions begins here

    num_test_pairs = len(task['test'])
    for task_num in range(num_test_pairs):
        cur_idx = 0
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(task['test'][task_num]['input']), len(
            task['test'][task_num]['input'][0])
        feat = np.zeros((nrows*ncols, nfeat))
        unique_col = {col: i for i, col in enumerate(
            sorted(np.unique(input_color)))}

        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx, 0] = i
                feat[cur_idx, 1] = j
                feat[cur_idx, 2] = input_color[i][j]
                feat[cur_idx, 3:7] = get_moore_neighbours(
                    input_color, i, j, nrows, ncols)
                feat[cur_idx, 7:9] = get_tl_tr(
                    input_color, i, j, nrows, ncols)
                feat[cur_idx, 9] = len(np.unique(input_color[i, :]))
                feat[cur_idx, 10] = len(np.unique(input_color[:, j]))
                feat[cur_idx, 11] = (i+j)
                feat[cur_idx, 12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                              j-local_neighb:j+local_neighb]))

                cur_idx += 1

        print('Made predictions for ', task_id[:-5])
        preds = hgb_grid.predict(feat).reshape(nrows, ncols)
        preds = preds.astype(int).tolist()
        plot_test(preds, task_id)
        sample_sub.loc[f'{task_id[:-5]}_{task_num}',
                       'output'] = flattener(preds)


# In[ ]:


print('\n Best hyperparameters:')
print(hgb_grid.best_params_)


# In[ ]:


sample_sub.head()


# In[ ]:


sample_sub.to_csv('submission.csv')

