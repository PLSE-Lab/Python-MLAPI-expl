#!/usr/bin/env python
# coding: utf-8

# # Inaccurate Labels in Kaggle Stanford Cars Data Set?
# This notebook documents apparent problems in the labels of the data set at https://www.kaggle.com/jessicali9530/stanford-cars-dataset/version/2

# In[ ]:


import pandas as pd
from scipy.io import loadmat 
from pathlib import Path

datapath = Path('../input/')


# 
# The data comes in as a .mat file from matlab, and the structure is not well documented.  My interpretation is that the first column in annotations appears to be the file name.  The next 4 columns represent the car bbox, while the 6th column apparently represents the car class index.  The 7th column apparently represents whether the file is in the train or test data set.
# 

# In[ ]:


MAT = loadmat(datapath/'cars_annos.mat')

print("Annotations")
print(MAT["annotations"][0,:5])
print("Class Names")
print(MAT["class_names"][0][:5])


# The get_labels functions creates a dataframe with the filnames and meta-data for easier interpretation.

# In[ ]:


def get_labels():
    MAT = loadmat(datapath/'cars_annos.mat')
    annotations = MAT["annotations"][0,:]
    nclasses = len(MAT["class_names"][0])
    class_names = dict(zip(range(1,nclasses),[c[0] for c in MAT["class_names"][0]]))
    
    labelled_images = {}
    dataset = []
    for arr in annotations:
        # the first entry in the row is the image name
        # The rest is the data, first bbox, then classid then a boolean for whether in train or test set
        dataset.append([arr[0][0].replace('car_ims/','')] + [y[0][0] for y in arr][1:])
    # Convert to a DataFrame, and specify the column names
    DF = pd.DataFrame(dataset, 
                      columns =['filename',"BBOX_Y2","BBOX_X1","BBOX_Y1","BBOX_X2","ClassID","TestSet"])

    DF = DF.assign(ClassName=DF.ClassID.map(dict(class_names)))
    return DF

DF = get_labels()
DF.head()


# According to this table, the class ID for image 000001.jpg is 1 which should be a Hummer.
# Lets take a look at the Test set 00001.jpg file and can see its actually an Audi sedan.

# In[ ]:


from pylab import imread,subplot,imshow,show
import matplotlib.pyplot as plt

image = imread(datapath/'cars_train/cars_train/00001.jpg')  
plt.imshow(image)


# Juat in case, also checking in the test data set, which is also not a hummer.
# 

# In[ ]:


image = imread(datapath/'cars_test/cars_test/00001.jpg')  #// choose image location

plt.imshow(image)


# It may be possible that I am misinterpreting the data format, or that there is something else going on here.  In any case, be warned before you try to use this data set that it is either corrupt, or not easy to parse correctly.  :-)  If you know what the trick is, let me know!
