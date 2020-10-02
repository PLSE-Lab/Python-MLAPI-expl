#!/usr/bin/env python
# coding: utf-8

# ## Simple Color Map Program
# I'm not sure if anyone had tried this yet, just wanted to share my results. I wrote a couple functions to check if the solution involves using a color map, then to determine that map and apply it. There are **three examples** of this task in the training set, however **none are present in the evaluation set or private LB test set**.

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import colors


# In[ ]:


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(glob(str(training_path / '*')))
evaluation_tasks = sorted(glob(str(evaluation_path / '*')))


# In[ ]:


cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plotTest(t_in, t_out, t_pred, title=''):
    fig, axs = plt.subplots(3, 1, figsize=(4,12), dpi=50)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(title, fontsize=16)
    axs[0].imshow(t_in, cmap=cmap, norm=norm)
    axs[0].set_title('Test in')
    axs[0].set_yticks(list(range(t_in.shape[0])))
    axs[0].set_xticks(list(range(t_in.shape[1])))
    axs[1].imshow(t_out, cmap=cmap, norm=norm)
    axs[1].set_title('Test out')
    axs[1].set_yticks(list(range(t_out.shape[0])))
    axs[1].set_xticks(list(range(t_out.shape[1])))
    axs[2].imshow(t_pred, cmap=cmap, norm=norm)
    axs[2].set_title('Test pred')
    axs[2].set_yticks(list(range(t_pred.shape[0])))
    axs[2].set_xticks(list(range(t_pred.shape[1])))


# In[ ]:


def getData(task_filename):
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task

def getObjectHash(pixmap):
    flat = pixmap.flatten().astype(np.bool)
    mult = np.array([2 ** x for x in range(len(flat))])
    return np.sum(flat * mult)

def groupByColor(pixmap):
    nb_colors = int(pixmap.max()) + 1
    splited = [(pixmap == i) * i for i in range(1, nb_colors)]
    return [x for x in splited if np.any(x)]

def checkColorMap(task):
    c = 1
    for example in task['train']:
        inp = np.array(example['input'])
        out = np.array(example['output'])
        inp_hashes = sorted([getObjectHash(pm) for pm in groupByColor(inp)])
        out_hashes = sorted([getObjectHash(pm) for pm in groupByColor(out)])
        c *= inp_hashes == out_hashes
    return bool(c)

def findColorMap(task):
    colormap = {}
    for example in task['train']:
        inp = np.array(example['input']).flatten()
        out = np.array(example['output']).flatten()
        for col, idx in zip(*np.unique(inp,return_index=True)):
            if col in colormap.keys(): continue
            colormap[col] = out[idx]
    return colormap

def applyColorMap(pixmap, colormap):
    return np.vectorize(colormap.__getitem__)(pixmap)


# ### Testing on the training set

# In[ ]:


for task_id in training_tasks:
    task = getData(task_id)
    if checkColorMap(task):
        colormap = findColorMap(task)
        for example in task['test']:
            test_in = np.array(example['input'])
            test_out = np.array(example['output'])
            pred = applyColorMap(test_in,colormap)
            correct = np.array_equal(pred,test_out)
            if correct:
                plotTest(test_in,test_out,pred,title=task_id)


# ### Now create submission on test set

# In[ ]:


def flattenPred(pred):
    return str(pred.astype(np.int))                    .replace(' ','')                    .replace('[[','|')                    .replace(']\n[','|')                    .replace(']]','|')


# In[ ]:


submission = pd.read_csv(data_path / 'sample_submission.csv')
test_tasks = submission['output_id'].values

predictions = []
for output_id in test_tasks:
    task_id, grid_id = output_id.split('_')
    task_id = test_path / ('%s.json' % task_id)
    task = getData(task_id)
    if checkColorMap(task):
        colormap = findColorMap(task)
        test_in = np.array(task['test'][output_id]['input'])
        pred = applyColorMap(test_in,colormap)
    else:
        pred = np.zeros((1,1))
    predictions.append(flattenPred(pred))

submission["output"] = predictions
submission.to_csv('submission.csv', index=False)

