#!/usr/bin/env python
# coding: utf-8

# **Extract NDVIs then dump to csv for further offline merging**
# -----------------
# 
# take codes from
# 1) https://www.kaggle.com/zhugds/fractals-of-nature-blend-v2
# 2) https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050
# 
# update to train_v2.csv and submission_v2.csv

# In[ ]:


from multiprocessing import Pool, cpu_count
from skimage import io
import pandas as pd
import numpy as np
import glob, cv2
import random
import scipy

random.seed(1)
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')

bins_nir = 64
bins_ndvi = 20

def get_features(path):
    try:
        st = []
        try:
            #skimage tif
            imgr = io.imread(path)
            imgr = cv2.resize(imgr, (256, 256))
            tf = imgr[:, :, 3]
            st += list(cv2.calcHist([tf], [0], None, [bins_nir],[0, 65536]).flatten()) #near ifrared
            ndvi = ((imgr[:, :, 3] - imgr[:, :, 0]) / (imgr[:, :, 3] + imgr[:, :, 0])) #water ~ -1.0, barren area ~ 0.0, shrub/grass ~ 0.2-0.4, forest ~ 1.0
            st += list(np.histogram(ndvi, bins=bins_ndvi, range=(-1, 1))[0])
            ndvi = ((imgr[:, :, 3] - imgr[:, :, 1]) / (imgr[:, :, 3] + imgr[:, :, 1]))
            st += list(np.histogram(ndvi, bins=bins_ndvi, range=(-1, 1))[0])
            ndvi = ((imgr[:, :, 3] - imgr[:, :, 2]) / (imgr[:, :, 3] + imgr[:, :, 2]))
            st += list(np.histogram(ndvi, bins=bins_ndvi, range=(-1, 1))[0])
        except:
            st += [-1 for i in range(256)]
            st += [-2 for i in range(60)]
            print('err', path.replace('jpg','tif'))
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
    return fdata

in_path = '../input/'
train = pd.read_csv(in_path + 'train_v2.csv')[:]
train['path'] = train['image_name'].map(lambda x: in_path + 'train-tif-v2/' + x + '.tif')
xtrain = normalize_img(train['path']); print('train...')
xtrain = np.array(xtrain, dtype=np.float32)
print(xtrain.shape)
for i in range(xtrain.shape[1]):
    if i < bins_nir:
        train['hist_nir_{:04d}'.format(i)] = xtrain[:, i]
    elif i < bins_nir + bins_ndvi:
        train['hist_ndvi_0_{:04d}'.format(i)] = xtrain[:, i]
    elif i < bins_nir + 2 * bins_ndvi:
        train['hist_ndvi_1_{:04d}'.format(i)] = xtrain[:, i]
    elif i < bins_nir + 3 * bins_ndvi:
        train['hist_ndvi_2_{:04d}'.format(i)] = xtrain[:, i]

train.to_csv('train_ndvi.csv', index=False)

test= pd.read_csv(in_path + 'sample_submission_v2.csv')[:]
test['path'] = test['image_name'].map(lambda x: in_path + 'test-tif-v2/' + x + '.tif')
xtest = normalize_img(test['path']); print('test...')
xtest = np.array(xtest, dtype=np.float32)
print(xtest.shape)
for i in range(xtest.shape[1]):
    if i < bins_nir:
        test['hist_nir_{:04d}'.format(i)] = xtest[:, i]
    elif i < bins_nir + bins_ndvi:
        test['hist_ndvi_0_{:04d}'.format(i)] = xtest[:, i]
    elif i < bins_nir + 2 * bins_ndvi:
        test['hist_ndvi_1_{:04d}'.format(i)] = xtest[:, i]
    elif i < bins_nir + 3 * bins_ndvi:
        test['hist_ndvi_2_{:04d}'.format(i)] = xtest[:, i]

test.to_csv('test_ndvi.csv', index=False)

