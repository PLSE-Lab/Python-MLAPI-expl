#!/usr/bin/env python
# coding: utf-8

# # This Kernel is not allowed to use as a final submission. Use at your own risk.
# # This Kernel was forked from the original Imitation Game.
# # The only change it the blending alpha weight.
# 

# In[ ]:


from multiprocessing import Pool, cpu_count
import glob, zipfile, os, itertools
from PIL import Image, ImageStat
from sklearn import *
import pandas as pd
import numpy as np

def get_features(path):
    try:
        st = []
        img = Image.open(path)
        im_stats_ = ImageStat.Stat(img)
        st += im_stats_.sum
        st += im_stats_.mean
        st += im_stats_.rms
        st += im_stats_.var
        st += im_stats_.stddev
    except:
        print(path)
    return [path, st]

def normalize_img(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_features, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    return pd.DataFrame(fdata)

dog_bytes = pd.DataFrame(glob.glob('../input/all-dogs/all-dogs/**'), columns=['Path'])
dog_bytes = pd.concat((dog_bytes, normalize_img(dog_bytes.Path.values)), axis=1)
dog_bytes.head()


# In[ ]:


dog_bytes['Group'] = cluster.KMeans(n_clusters=100, random_state=3, n_jobs=-1).fit_predict(dog_bytes[list(range(15))])
dog_bytes['Group'].value_counts()[:5]


# Inspiration
# ===========

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(8, 80))
samples = []
for i in range(100):
    g = dog_bytes[dog_bytes['Group'] == i]
    if len(g) >= 5:
        samples += list(g['Path'].values[:5])

for i in range(len(samples))[:50]:
    ax = fig.add_subplot(len(samples)/5, 5, i+1, xticks=[], yticks=[])
    img = Image.open(samples[i])
    img = img.resize((100,int(img.size[1]/(img.size[0]/100))), Image.ANTIALIAS)
    img = img.crop((0, 0, 64, 64))
    plt.imshow(img)


# Motivation
# ==============

# In[ ]:


def sim_img(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((100,int(img.size[1]/(img.size[0]/100))), Image.ANTIALIAS)
    img = img.crop((0, 0, 64, 64))
    return img

samples = []
for i in range(100):
    g = dog_bytes[dog_bytes['Group'] == i]
    if len(g) >= 23:
        s = g['Path'].values[:23]
        s = list([p for p,_ in itertools.groupby(sorted([sorted(p) for p in list(itertools.permutations(s, 2))]))])
        samples += s
print(len(samples))


# Submission
# =============

# In[ ]:


z = zipfile.PyZipFile('images.zip', mode='w')
for i in range(10_000):
    p1, p2 = samples[i]
    try:
        im = Image.blend(sim_img(p1), sim_img(p2), alpha=0.20)
        f = str(i)+'.png'
        im.save(f,'PNG'); z.write(f); os.remove(f)
        if i % 500==0:
            print(i)
    except:
        print(p1, p2)

print (len(z.namelist()))
z.close()

