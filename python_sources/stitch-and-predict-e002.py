#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://www.kaggle.com/the1owl/draper-satellite-image-chronology/stitch-and-predict/run/233527
#original by the1owl:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob

sample_sub = pd.read_csv('../input/sample_submission.csv')
train_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/train_sm/*.jpeg")])
train_files.columns = ['path', 'group', 'pic_no']
test_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/test_sm/*.jpeg")])
test_files.columns = ['path', 'group', 'pic_no']
print(len(train_files),len(test_files),len(sample_sub))
train_images = train_files[train_files["group"]=='set107']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1])
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
a = []
for l in train_images.path:
    im = cv2.imread(l)
    plt.subplot(5, 2, i_+1).set_title(l)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(5, 2, i_+2).set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)


# In[ ]:


im = Image.open(train_images.path.iloc[0])
im2 = Image.open(train_images.path.iloc[1])
im2 = im2.rotate(-15)
im2 = im2.resize((3000, 2000), Image.ANTIALIAS)
r,g,b = im2.split()
mask = Image.merge("L", (b,))
im.paste(im2, (0,0,3000,2000), mask)
plt.imshow(im); plt.axis('off')


# In[ ]:


train_images = train_files[train_files["group"]=='set4']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1])
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
a = []
for l in train_images.path:
    im = cv2.imread(l)
    plt.subplot(5, 2, i_+1).set_title(l)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(5, 2, i_+2).set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)


# In[ ]:


import time; start_time = time.time()
import warnings; warnings.filterwarnings('ignore');
import multiprocessing
from sklearn import ensemble
from sklearn import pipeline, grid_search
from sklearn.metrics import label_ranking_average_precision_score as lraps

def image_features(path, tt, group, pic_no):
    im = cv2.imread(path)
    me_ = cv2.mean(im)
    s=[path, tt, group, pic_no, im.mean(), me_[2], me_[1], me_[0]]
    f = open("data.csv","a")
    f.write((',').join(map(str, s)) + '\n')
    f.close()
    return

f = open("data.csv","w");
col = ['path','tt', 'group', 'pic_no', 'individual_im_mean','rm','bm','gm']
f.write((',').join(map(str,col)) + '\n')
f.close()

if __name__ == '__main__':
    cpu = multiprocessing.cpu_count(); print (cpu);
    
    j = []
    for s_ in range(0,len(train_files),cpu):     #train
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(train_files):
                if i_ % 100 == 0:
                    print("train ", i_)
                filename = train_files.path[i_]
                
                p = multiprocessing.Process(target=image_features, args=(filename,'train', train_files["group"][i_], train_files["pic_no"][i_],))
                j.append(p)
                p.start()
    j = []
    for s_ in range(0,len(test_files),cpu):     #test
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(test_files):
                if i_ % 100 == 0:
                    print("test ", i_)
                filename = test_files.path[i_]
                p = multiprocessing.Process(target=image_features, args=(filename,'test', test_files["group"][i_], test_files["pic_no"][i_],))
                j.append(p)
                p.start()
    
    while len(j) > 0: #end all jobs
        j = [x for x in j if x.is_alive()]
        time.sleep(1)
    


# In[ ]:




