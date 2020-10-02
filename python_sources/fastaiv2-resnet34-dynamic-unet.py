#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# First attempt at an end to end submission!
from datetime import datetime
import os
import pandas as pd


# In[ ]:


TIME_EVENTS = [('Book Start', datetime.now())]
print(f'Start of Book: {TIME_EVENTS[0][1]}')
def runtime(desc):
    global TIME_EVENTS
    now = datetime.now()
    TIME_EVENTS.append((desc, now))
    print(f'Now: {desc}: {now}')
    print(f'Time Since First Event, {TIME_EVENTS[0][0]}: {TIME_EVENTS[-1][1]-TIME_EVENTS[0][1]})')
    print(f'Time Since Last Event, {TIME_EVENTS[-2][0]}: {TIME_EVENTS[-1][1]-TIME_EVENTS[-2][1]}')


# In[ ]:


if os.path.exists('/kaggle/'):
    get_ipython().system('pip install ../input/fastai2-wheels/fastscript-0.1.4-py3-none-any.whl > /dev/null')
    get_ipython().system('pip install ../input/fastai2-wheels/kornia-0.2.0-py2.py3-none-any.whl > /dev/null')
    get_ipython().system('pip install ../input/fastai2-wheels/nbdev-0.2.12-py3-none-any.whl > /dev/null')
    get_ipython().system('pip install ../input/fastai2-wheels/fastprogress-0.2.3-py3-none-any.whl > /dev/null')
    get_ipython().system('pip install ../input/fastai2-wheels/fastcore-0.1.16-py3-none-any.whl > /dev/null')
    get_ipython().system('pip install ../input/fastai2-wheels/fastai2-0.0.16-py3-none-any.whl > /dev/null')
    
    get_ipython().system('mkdir -p /root/.cache/torch/checkpoints/')
    get_ipython().system("cp '../input/pytorch-pretrained-models/resnet34-333f7ec4.pth' '/root/.cache/torch/checkpoints/resnet34-333f7ec4.pth'")


# In[ ]:


# Bug in fastai2-0.0.16: https://github.com/fastai/fastai2/issues/331
import subprocess
c = subprocess.call(
    ['sed',
     's/bias_std=0)/bias_std=0.01)/',
     '-i',
     '/opt/conda/lib/python3.7/site-packages/fastai2/layers.py'])


# In[ ]:


# Approximately 3 minutes 45 seconds
runtime('Pip Installs')


# In[ ]:


import openslide

from PIL import Image
import pandas as pd
import numpy as np
import torch
from fastai2.vision.all import *
torch.cuda.set_device(0)


# In[ ]:


SEED=182


# TRAIN_DIR = 'test_set_2'
# TRAIN_DIR = '/opt/mount/train_images'
TRAIN_DIR = '../input/prostate-cancer-grade-assessment/train_images'
# MASK_DIR = 'test_set_2_masks'
# MASK_DIR = '/opt/mount/train_label_masks'
MASK_DIR = '../input/prostate-cancer-grade-assessment/train_label_masks'
TEST_DIR = '../input/prostate-cancer-grade-assessment/test_images'

_MD = set([y.replace('_mask', '') for y in os.listdir(MASK_DIR)])
TRAIN_IDS = [os.path.join(TRAIN_DIR, x) for x in os.listdir(TRAIN_DIR) if x in _MD]
print(len(TRAIN_IDS))
TARGET_DIM = 128

RAD_CODES = [
    'background', 'healthy stroma', 'healthy epithelium',
    'gleason level 3', 'gleason level 4', 'gleason level 5']


# In[ ]:


def _open_and_resize(path):
    slide = openslide.OpenSlide(path)
    l2_dims = slide.level_dimensions[2][:2]

    return np.array(slide.read_region(
        (0,0), 2, l2_dims).resize(
            (TARGET_DIM,TARGET_DIM)))[:,:,:3]


def get_x(path):
    return _open_and_resize(path)


def get_y(path):
    name = os.path.basename(path)
    path = os.path.join(MASK_DIR, name).replace('.tiff', '_mask.tiff')
    return _open_and_resize(path)


def create_datablock(bs=16):
    # Rad / downsampled only for the time being

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=['bg', 'good s', 'good e', 'g3', 'g4', 'g5'])),
        splitter=RandomSplitter(0.2, seed=SEED),
        get_x=get_x,
        get_y=get_y,
    )
    res = DataLoaders.from_dblock(dblock, TRAIN_IDS, bs=bs)
    return dblock, res


def isup_grade(result, bincount=None):
    if bincount is None:
        bincount = np.bincount(result[0].flatten())
    if len(bincount) <= 3:
        return 0

    bincount[0], bincount[1], bincount[2] = 0, 0, 0
    maximum = 0
    first = 0
    second = 0

    for i, count in enumerate(bincount):
        if count > maximum:
            first = i
            maximum = count

    if not first:
        return 0

    maximum = 0
    bincount[first] = 0
    for i, count in enumerate(bincount):
        if count > maximum:
            second = i
            maximum = count

    if not second:
        second = first
    isup = {
        '3+3': 1,
        '3+4': 2,
        '4+3': 3,
        '4+4': 4,
        '3+5': 4,
        '5+3': 4,
        '4+5': 4,
        '5+4': 5,
        '5+5': 5
    }
    return isup[f'{first}+{second}']


# In[ ]:


dblock, dls = create_datablock(bs=8)
dls.show_batch(max_n=6)


# In[ ]:


learn = unet_learner(dls, resnet34)


# In[ ]:


learn.model = learn.model.to('cuda')


# In[ ]:


# Approximately 15 seconds
runtime('Create model and push to GPU')


# In[ ]:


# skip
# results = learn.lr_find(start_lr=1e-7, end_lr=10, num_it=1000)
# results


# In[ ]:


# Approximately 3.5 minutes
# learning_rate = results.lr_min
learning_rate = 0.00010568174766376615


# In[ ]:


runtime('Determine learning rate')


# In[ ]:


EPOCHS = 5
learn.fit(EPOCHS, learning_rate)


# In[ ]:


runtime(f'Train {EPOCHS} Epochs')


# In[ ]:


import os
TEST_PATH = '../input/prostate-cancer-grade-assessment/test_images'
TRAIN_PATH = '../input/prostate-cancer-grade-assessment/train_images'


def submit(path, fallback):
    target = './submission.csv'
    
    submission = []

    # Try to run on the real images, otherwise just use the test set
    try:
        images = os.listdir(path)
    except FileNotFoundError:
        path = fallback
        images = os.listdir(path)[:1100]

    # Predict for each file
    for i in images:
        file_path = os.path.join(path, i)
        _id = i.replace('.tiff', '')
    
        with learn.no_bar():
            grade = isup_grade(learn.predict(file_path))

        submission.append((str(_id), str(grade)))

    # Write out the data
    with open('./submission.csv', 'wb') as f:
        f.write(bytes('image_id,isup_grade\n', encoding='utf8'))
        for image in submission:
            f.write(bytes(f'{image[0]},{image[1]}\n', encoding='utf8'))
                
submit(TEST_PATH, TRAIN_PATH)


# In[ ]:


get_ipython().system('cat ./submission.csv | head -n 10')


# In[ ]:


runtime(f'Submit some results!')

