#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import numpy as np
import os
import pandas as pd
import shutil
from pathlib import Path
from shutil import unpack_archive
from subprocess import check_output
print(os.listdir("../input/recursion-cellular-image-classification"))


# ### Unpack zip files

# In[ ]:


path = '../input/recursion-cellular-image-classification' # specify directory where the competition zip files were downloaded
fileList = os.listdir(path) # get file list in the path directory
# list files
print(fileList)


# In[ ]:


# unpack zip files in path directory
for i in fileList:
    if "zip" in i:
        unpack_archive(path + i, path, 'zip')


# In[ ]:


fileList = os.listdir(path) # get updated file list in path directory
# list files/folders
print(fileList)


# In[ ]:


# populate train.csv into a pandas DataFrame
train = pd.read_csv(path + '/train.csv')
print(train.shape) # print DataFrame shape
train.head() # print first 5 records of the DataFrame


# In[ ]:


# populate test.csv into a pandas DataFrame
test = pd.read_csv(path + '/test.csv')
print(test.shape) # print DataFrame shape
test.head() # print first 5 records of the DataFrame


# ### Update directory structore

# In[ ]:


# create DataFrame containing a sorted list of sirna from the train DataFrame
imgClasses = train.sirna.sort_values(axis=0)
# create Dataframe containing a unique list of sirna from the train DataFrame
imgClassesUnique = imgClasses.unique()
print(imgClassesUnique,len(imgClassesUnique))


# In[ ]:


"""shutil.rmtree(work_path + "train")
shutil.rmtree(work_path + "valid")
shutil.rmtree(work_path + "test")"""


# In[ ]:


# create train, valid, and test directories
# create train, valid, and test directories
work_path = '../working/'
shutil.os.mkdir(work_path + "train")
shutil.os.mkdir(work_path + "valid")
shutil.os.mkdir(work_path + "test")


# In[ ]:


print(os.listdir(work_path+'train'))


# In[ ]:


# create directories for each unique species in the leafClassesUnique Dataframe in the train and valid directories previous created
for i in imgClassesUnique:
    shutil.os.mkdir(work_path + 'train/' + str(i))
    shutil.os.mkdir(work_path + 'valid/' + str(i))


# In[ ]:


print(os.listdir(path))


# In[ ]:


# Create the modified Test csv file appropriate for the work
dftest = pd.read_csv(path+'/test.csv');
dftest = dftest['id_code'];
dictest = {};
for fold1 in os.listdir(path+'/test'):
    for fold2 in os.listdir(path+'/test/'+fold1):
        for image in os.listdir(path+'/test/'+fold1+'/'+fold2):
            dictest[str(fold1)+'_'+fold2[5:]+'_'+image[0:3]] = str(fold1)+'/'+fold2+'/'+image;
testData = pd.DataFrame(list(dictest.items()), columns=['id_code','foldPath']);
testData.to_csv(r'../working/testData.csv', index = None, header=True);
#print(testData.tail())


# In[ ]:





# In[ ]:


# move the test images from the images directory to the test directory
for i in testData.iloc[:,1]: # test image labels are identified in the id column of the test DataFrame
    shutil.copy(path + "/test/"+ i, work_path + "test")
    data_file = Path(work_path + 'test/'+i.split('/')[-1])
    data_file.rename(work_path + 'test/'+ i.split('/')[0]+'_'+str(i.split('/')[1][5:])+'_'+i.split('/')[-1][0:3] + '.png')


# In[ ]:


#print(os.listdir(work_path+'/test'))


# In[ ]:


# Create the modified Train csv file appropriate for the work
path = '../input/recursion-cellular-image-classification'
dftrain = pd.read_csv(path+'/train.csv');
dftrain = dftrain[['id_code','sirna']];
dic = {};
for fold1 in os.listdir(path+'/train'):
    for fold2 in os.listdir(path+'/train/'+fold1):
        for image in os.listdir(path+'/train/'+fold1+'/'+fold2):
            dic[str(fold1)+'_'+fold2[5:]+'_'+image[0:3]] = str(fold1)+'/'+fold2+'/'+image;
df = pd.DataFrame(list(dic.items()), columns=['id_code','Item']);
dftraindf = pd.merge(df, dftrain);
trainData = dftraindf[['Item','sirna']];
trainData.to_csv(r'../working/trainData.csv', index = None, header=True);
print(trainData.head())


# In[ ]:


for index, row in trainData.iterrows(): # test image labels are identified in the id column of the test DataFrame
    leave_path = path + '/train/'+ row[0]
    dest_path = work_path + 'train/' + str(row[1])
    #print(leave_path, dest_path)
    shutil.copy(leave_path, dest_path)
    data_file = Path(work_path + 'train/' +str(row[1])+'/'+row[0].split('/')[-1])
    #work_path + 'train/' +str(row[1])+'/'+row[0].split('/')[-1]
    data_file.rename(work_path + 'train/' +str(row[1])+'/'+row[0].split('/')[0]+'_'+str(row[0].split('/')[1][5:])+'_'+row[0].split('/')[-1][0:3] + '.png')


# In[ ]:


#print(os.listdir(work_path+'/train/11'))


# In[ ]:


"""for name in dir():
    if not name.startswith('_'):
        del globals()[name]"""

