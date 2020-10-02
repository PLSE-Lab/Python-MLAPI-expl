#!/usr/bin/env python
# coding: utf-8

# # This is an Analytical Library used for converting Image data to csv or rather 2 D frames so that dimensionality Reduction and Machine Learning Algorithms can be applied to it 

# In[ ]:


import numpy as np
from os import listdir
import pandas as pd
import os
os.listdir("../input/plantdisease/plantvillage/PlantVillage")


# In[ ]:


list_of_dir = os.listdir("../input/plantdisease/plantvillage/PlantVillage")


# # Our Target is Only Tomato dataset
# 

# In[ ]:


Req = []
for i in list_of_dir:
    if i.split("_")[0] == 'Tomato' : Req.append(i)
Req


# In[ ]:


def create(location , var):
    import cv2
    import glob
    X_data = []
    files = glob.glob("../input/plantdisease/plantvillage/PlantVillage/"+location+"/*.JPG")
    for myFile in files:
        #print(myFile)
        image = cv2.imread (myFile)
        image = cv2.resize(image, (64,64)) 
        X_data.append (image)
    print('X_data shape:', np.array(X_data).shape)
    numpy_entry = np.array(X_data).reshape(np.array(X_data).shape[0] , np.array(X_data).shape[1]*np.array(X_data).shape[2]*np.array(X_data).shape[3])
    print('numpy_entry shape:', numpy_entry.shape)
    df=pd.DataFrame(data=numpy_entry[0:,0:],index=[i for i in range(numpy_entry.shape[0])],columns=['Pixel '+str(i) for i in range(numpy_entry.shape[1])])
    df['Category'] = var
    return df


# In[ ]:


master_df=pd.DataFrame(columns=['Pixel '+str(i) for i in range(64*64*3)])
var = 0
for i in Req:
    df = create(i,var)
    frames=[master_df , df]
    master_df = pd.concat(frames)
    print(" Done For ",i," With category Value ",var)
    print("Master Data Frame   ",master_df.shape)
    var = var + 1


# # So Thus the tough part of data creation is done let us now see how the data looks and have some basic analytics about it

# In[ ]:


master_df.head()


# Lets see the shape of the dataset

# In[ ]:


master_df.shape


# In[ ]:


master_df.to_csv('Tomato_Pixel_DataSet.csv') 


# In[ ]:


master_df.columns


# In[ ]:


master_df['Category'].unique()


# Kaggle donot allow any more data storage on sinle file so do view my page and find for Machine learning for Data Reduction - II for further usage
