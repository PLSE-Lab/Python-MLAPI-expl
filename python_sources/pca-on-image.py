#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn
import glob, os
get_ipython().run_line_magic('matplotlib', 'inline')
import skimage
from skimage.feature import greycomatrix, greycoprops,corner_harris
from skimage.filters import sobel,gaussian
from skimage.color import rgb2gray
from skimage.transform import resize 
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit


# In[ ]:


drivers = pd.read_csv('../input/driver_imgs_list.csv')
train_files = [f for f in glob.glob("../input/train/*/*.jpg")]
test_files = ["../input/test/" + f for f in os.listdir("../input/test/")]
print(train_files[:10])
print(test_files[:10])


# In[ ]:


# read training images
img_all = []
img_all_avg = []
img_all_gs = []
data_all = []

img_all_rz = []
img_all_avg_rz = []
img_all_gs_rz = []
data_all_rz = []
target_all = []
subject_all = []
for i in range(0,len(train_files)):
    if (i%5000==0):
        print(str(i) + ' images read')
    path = train_files[i]
    #print path
    im_read = plt.imread(path)
    im_read_avg = im_read[:,:,0]+im_read[:,:,1]+im_read[:,:,2]
    img_gray = rgb2gray(im_read)
    dims = np.shape(img_gray)
    img_data= np.reshape(img_gray, (dims[0] * dims[1], 1))
    
    img_gray_rz = gaussian(resize(img_gray,(48,64)),sigma=1)
    im_read_rz = resize(im_read,(48,64))
    im_read_avg_rz = im_read_rz[:,:,0]+im_read_rz[:,:,1]+im_read_rz[:,:,2]
    dims = np.shape(img_gray_rz)
    img_data_rz= np.reshape(img_gray_rz, (dims[0] * dims[1], 1))
    data_all_rz.append(img_data_rz)
    target_all.append(drivers.loc[i]['classname'])
    subject_all.append(drivers.loc[i]['subject'])


# In[ ]:


## Converting data to NP-array
data_all_model = np.asarray(data_all_rz)
target_all = np.asarray(target_all)
subject_all =np.asarray(subject_all)
data_all_model = data_all_model[:,:,0]


# In[ ]:


n_components = 200
print("Extracting the top %d PCs from %d images"
      % (n_components, data_all_model.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data_all_model)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


path = train_files[0]
img = plt.imread(path)
img_rz = resize(img,(48,64))
img_rz_gs = rgb2gray(img_rz)
plt.figure(figsize=(15,20))
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(img_rz_gs,cmap='gray')


# In[ ]:


print(np.shape(img_rz_gs))
img_rz_gs_v = np.reshape(img_rz_gs,(1,3072))
img_PC = pca.transform(img_rz_gs_v)
img_PC2 = pca.inverse_transform(img_PC)

plt.figure(figsize=(5,7))
plt.imshow(np.reshape(img_PC2,(48,64)),cmap='gray')


# In[ ]:


target_sub = [target_all[i]+'-'+subject_all[i] for i in range(0,np.shape(data_all_model)[0])]
sss = StratifiedShuffleSplit(target_sub, 1, test_size=0.4, random_state=42)


# In[ ]:


for train_index,test_index in sss:
    X_train = data_all_model[train_index]
    X_test  = data_all_model[test_index]
    y_train = target_all[train_index]
    y_test = target_all[test_index]


print("Projecting the input data on the PC's orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


print("Training SVC model")
t0 = time()
svm_model = SVC(kernel='rbf', class_weight='balanced', C=1000,gamma=0.01, probability=True)
svm_model = svm_model.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting class on the test set")
t0 = time()
y_pred = svm_model.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:




