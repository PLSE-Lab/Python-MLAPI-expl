#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from multiprocessing import Pool, cpu_count
import glob, zipfile, os, itertools
from PIL import Image, ImageStat
from sklearn import *
import pandas as pd
import numpy as np

# Statistics
def get_features(path):
    try:
        st = []
        # Image pixel obtained
        img = Image.open(path)
        # Start the statistical result of RGB by image data
        im_stats_ = ImageStat.Stat(img)
        # total
        st += im_stats_.sum
        # Average value
        st += im_stats_.mean
        # Root mean square
        st += im_stats_.rms
        # dispersion
        st += im_stats_.var
        # standard deviation
        st += im_stats_.stddev
    except:
        print(path)
    return [path, st]

# Parallel processing
def normalize_img(paths):
    imf_d = {}
    p = Pool(cpu_count())
    # get_features(Parallel processing of functions)
    ret = p.map(get_features, paths)
    # Arrange the result of parallel processing
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    return pd.DataFrame(fdata)

# Load image data path
dog_bytes = pd.DataFrame(glob.glob('../input/all-dogs/all-dogs/**'), columns=['Path'])
# Get statistical data of pixcel data for each image
dog_bytes = pd.concat((dog_bytes, normalize_img(dog_bytes.Path.values)), axis=1)
dog_bytes.head()


# In[ ]:


# Divide image data into 100 classifications by KMeans method
dog_bytes['Group'] = cluster.KMeans(n_clusters=100, random_state=3, n_jobs=-1).fit_predict(dog_bytes[list(range(15))])
# Get 5 large categories from 100 categories (displayed)
dog_bytes['Group'].value_counts()[:5]


# Inspiration
# ===========

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Create a window to display the image
# Unit is in inches
fig = plt.figure(figsize=(8, 80))
samples = []
# Get 5 samples from image data of a specific category
for i in range(100):
    # Acquire image data of a specific classification
    g = dog_bytes[dog_bytes['Group'] == i]
    if len(g) >= 5:
        # Get 5 samples from image data of specific classification
        samples += list(g['Path'].values[:5])

# Display images for each category
for i in range(len(samples))[:50]:
    # Get one of the 5 rows and 5 columns of windows
    ax = fig.add_subplot(len(samples)/5, 5, i+1, xticks=[], yticks=[])
    # Get image data
    img = Image.open(samples[i])
    # Resize image data
    # Unit is pixel
    # Resolution (dpi) = pixel / inch
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
        # Create a complete set of two sets of image data in the same classification
        s = list([p for p,_ in itertools.groupby(sorted([sorted(p) for p in list(itertools.permutations(s, 2))]))])
        samples += s
print(len(samples))


# Submission
# =============

# In[ ]:


z = zipfile.PyZipFile('images.zip', mode='w')
for i in range(10000):
    p1, p2 = samples[i]
    try:
        # Mix two images in the same classification to create a new image
        # out = p1 * (1 - 0.4) + p2 * 0.4
        im = Image.blend(sim_img(p1), sim_img(p2), alpha=0.4)
        f = str(i)+'.png'
        im.save(f,'PNG'); z.write(f); os.remove(f)
        if i % 1000==0:
            print(i)
    except:
        print(p1, p2)

print (len(z.namelist()))
z.close()


# In[ ]:




