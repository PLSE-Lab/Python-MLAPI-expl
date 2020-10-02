# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train_labels.csv')
train_zip = '../input/train/'
test_zip = '../input/test/'

def read(path):
    bgr_img = cv2.imread(path)
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img
#Any results you write to the current directory are saved as output.
fig,ax = plt.subplots(2,5,figsize=(20,8))
fig.suptitle('Scans')
for i ,idx in enumerate(train[train['label'] == 0]['id'][:5]):
    path = os.path.join(train_zip,idx)
    ax[0,i].imshow(read(path+'.tif'))
    box = patches.Rectangle((32,32),32,32,linewidth='2',edgecolor = 'r',facecolor = 'none',linestyle =':')
    ax[0,i].add_patch(box)
for i ,idx in enumerate(train[train['label']== 1]['id'][:5]):
    path = os.path.join(train_zip,idx)
    ax[1,i].imshow(read(path+'.tif'))
    box = patches.Rectangle((32,32),32,32,linewidth='2',edgecolor ='b',facecolor = 'none',linestyle =':')
    ax[1,i].add_patch(box)    