#!/usr/bin/env python
# coding: utf-8

# ## How to set up directory for keras flow_from_directory function

# We present in this notebook various utility functions to create proper setup for the flow_from_directory function of the keras.preprocessing.image library. The basic idea is to create an ImageDataGenerator and then call the function flow_from_directory. But we need to have the folder organizedwith  one subdirectory per class. Then, any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator and classified with the label given by the name of the subfolder.
# 
# Hence it is usefull to create a folder hierarchy that creaes these subfolder. We present here the corresponding helper functions
# 
# 

# We first load up our pandas dataframe to get the directory information

# In[8]:


import pandas as pd
import numpy as np
import os
import shutil

dtypes = {
        'id': 'str',
        'url': 'str',
        'landmark_id': 'uint32',
}
train_df = pd.read_csv("../input/train.csv", dtype = dtypes)


# Function to create the folders

# In[4]:


'''
Create the subfolders with the given labels from the data frame
'''
def create_folders():
    categories = np.unique(train_df.landmark_id.values)
    for i in categories:
        folder = r"../input/train/" + str(i)
        if not os.path.exists(folder):
            print('created folder', i)
            os.mkdir(folder)
        else:
            print( str(i), ' exists!')


# In[5]:


'''
function to move the images to their corresponding folder
'''
def move_files():
    failed = 0
    for i, row in train_df.iterrows():
        filename = r"../input/train/{}/{}.jpg".format(row.landmark_id, 
                                      row.id )
        oldfile = r"../input/train/{}.jpg".format(row.id )
        if not os.path.exists(filename):
            try:
                os.rename(oldfile, filename)
                print('moved {}.jpg to {}'.format(row.id, row.landmark_id))
            except:
                failed +=1
        else:
            print('{}.jpg is in {}'.format(row.id, row.landmark_id))
    
    print('failed on {} files'.format(failed))


# If we need some validation folder for our keras CNN, we need also to organize the folders accordingly

# In[6]:


'''
function to create the training validation folder
'''
def create_test_folder():
    failed = 0
    val_size = int(len(train_df)*0.1)
    val_folder = r"../input/train_val"
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)
        
    for i, row in train_df.iloc[:val_size].iterrows():
        filename = r"../input/train/{}/{}.jpg".format(row.landmark_id, 
                                      row.id )
        newFile = r"../input/train_val/{}/{}.jpg".format(row.landmark_id, 
                                      row.id )
        folder = r"../input/train_val/{}".format(row.landmark_id )
        print('testing {}.jpg in {}'.format(row.id, row.landmark_id))
        if not os.path.exists(newFile):
            if not os.path.exists(folder):
                os.mkdir(folder)
                print('created folder', folder)
            try:
                shutil.copy2(filename, newFile)
                print('copied {}.jpg to train_val/{}'.format(row.id, row.landmark_id))
            except:
                failed +=1
        else:
            print('{}.jpg is in {}'.format(row.id, row.landmark_id))
    print('failed on {} files'.format(failed))


# Last but not least, one needs also to set the test folder

# In[7]:


'''
For the test folder, we need to create a dummy subfolder. In this case, we created 0
'''
def create_test():
    count = 0
    folder = r"../input/test"    
    
    if not os.path.exists(folder + r"/0"):
            os.mkdir(folder + r"/0")
    
    for root, dirs, files in os.walk(folder):
        if '0' in dirs:
            for f in files:
                oldfile  = os.path.join(root, f)    
                newfile = '{}/0/{}'.format(root,f)
                os.rename(oldfile,newfile)
                count +=1
                if count % 20 == 0:
                    print('moved {} to {}'.format(oldfile,newfile))


# Then call the various functions accordingly.

# exit()
