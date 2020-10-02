#!/usr/bin/env python
# coding: utf-8

# ### Resize iWildCam 2019 and iNat Idaho data
# 
# Train models on complete image data (iWildCam 2019 + supplemental iNat Idaho)
# 
# * iWildCam2019: https://www.kaggle.com/c/iwildcam-2019-fgvc6/data
# * Supplemental iNat Idaho data: https://github.com/visipedia/iwildcam_comp
# * Original idea of resizing imgs from https://www.kaggle.com/xhlulu/reducing-image-sizes-to-32x32
# 

# **How to use this kernel**
# * Fork kernel
# * Modify *USR_IMG_SIZE* to the desired image dimensions 
# * Commit kernel
# * In a new kernel, click *+ add dataset* -> *Kernel output files* -> *Your Work* -> *iWildCam 2019 + iNat Idaho*
# 
# You will now have access to x_train, y_train and y_test in /kaggle/input to start building models!

# In[ ]:


# modify as needed
USR_IMG_SIZE = 32 # random sample images will be saved as 32x32


# ### Implementation
# 
# Below implementation can remain untouched unless you want to add extra preprocessing to images.
# 
# *USR_IMG_SIZE* is the only main user modifications

# In[ ]:


import cv2
import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import StratifiedShuffleSplit


# **iWildCam 2019 (Kaggle input data)**

# In[ ]:


test_df = pd.read_csv('/kaggle/input/sample_submission.csv')
test_df['file_path'] = test_df['Id'].apply(lambda x: f'/kaggle/input/test_images/{x}.jpg')
test_df.drop(columns=['Predicted','Id'],inplace=True)
test_df.head()


# In[ ]:


train_df = pd.read_csv('/kaggle/input/train.csv')
train_df['file_path'] = train_df['id'].apply(lambda x: f'/kaggle/input/train_images/{x}.jpg')
train_df = train_df[['file_path','category_id']]
train_df['is_supp'] = False
train_df.head()


# **Supplemental iNat Idaho**

# Extract

# In[ ]:


get_ipython().run_cell_magic('time', '', '!wget https://wildcamdrop.blob.core.windows.net/wildcamdropcontainer/iWildCam_2019_iNat_Idaho.tar.gz -P /kaggle/supplemental')


# In[ ]:


get_ipython().run_cell_magic('time', '', '!tar xvf /kaggle/supplemental/iWildCam_2019_iNat_Idaho.tar.gz -C /kaggle/supplemental/ >> /kaggle/working/log.txt\n\n# images now in /kaggle/supplemental/iWildCam_2019_iNat_Idaho/train_val2017/ and /kaggle/supplemental/iWildCam_2019_iNat_Idaho/train_val2018/\n\n!rm /kaggle/supplemental/iWildCam_2019_iNat_Idaho.tar.gz\n!rm /kaggle/working/log.txt')


# Get a dataframe

# In[ ]:


with open('/kaggle/supplemental/iWildCam_2019_iNat_Idaho/iWildCam_2019_iNat_Idaho.json') as json_data:
    supp_data = json.load(json_data)


# In[ ]:


image_df = pd.DataFrame.from_dict(supp_data['images'])
image_df = image_df[['file_name','id']]
image_df.rename(columns={'id':'image_id'}, inplace=True)
image_df.head()


# In[ ]:


annotation_df = pd.DataFrame.from_dict(supp_data['annotations'])
annotation_df = annotation_df[['category_id','image_id']]
annotation_df.head()


# In[ ]:


supp_df = pd.merge(image_df, annotation_df, on='image_id', how='inner')

supp_df['is_supp'] = True
supp_df['file_path'] = supp_df['file_name'].apply(lambda x: f'/kaggle/supplemental/iWildCam_2019_iNat_Idaho/{x}')
supp_df.drop(columns=['image_id','file_name'],inplace=True)

supp_df.head()


# ** Combine kaggle data with supplemental **

# In[ ]:


complete_df = pd.concat([train_df,supp_df], ignore_index = True)
complete_df.head()


# **Resize all images**

# In[ ]:


# Image transformations and more data prep can be done in a child kernel
def resize(image_path, desired_size):
    img = cv2.imread(image_path)
    return cv2.resize(img, (desired_size,)*2).astype('uint8')


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_resized_imgs = [resize(file_path,USR_IMG_SIZE) for file_path in complete_df['file_path']]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_resized_imgs = [resize(file_path,USR_IMG_SIZE) for file_path in test_df['file_path']]")


# In[ ]:


X_train = np.stack(train_resized_imgs)
X_test = np.stack(test_resized_imgs)
y_train = pd.get_dummies(complete_df['category_id']).values

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)


# **Saving**

# In[ ]:


# No need to save the IDs of X_test, since they are in the same order as the ID column in sample_submission.csv
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)

# save the complete_df for future reference
complete_df.to_pickle("/kaggle/working/complete_df.pkl")

