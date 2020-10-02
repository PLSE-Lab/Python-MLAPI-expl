#!/usr/bin/env python
# coding: utf-8

# **Reorganize Data with Symbolic Links**

# In computing, a symbolic link (also symlink or soft link) is a term for any file that contains a reference to another file or directory in the form of an absolute or relative path and that affects pathname resolution (Source: https://en.wikipedia.org/wiki/Symbolic_link).  Here we will use symbolic links to reorganize our data.
# 
# Originally the "Dogs vs Cats" images were organized inside of two folders titled "Train" and "Test" and the images of dogs and cats were easily identifiable based off of the file name for each image.  We want to reorganize the images such that now they will be organized inside of two folders titled "Train" and "Valid" and such that the images of dogs and cats will be easily identifiable based off of whether they are in a subfolder that is titled "Dog" or a subfolder that is titled "Cat".  This is the [format](http://wiki.fast.ai/index.php/Lesson_1_Notes#Data_Structure) that is required for completing exercises in the 2018 [FAST.AI](http://www.fast.ai/) Deep Learning course.
# 

# *Step 1: Describe Original Data Organization*

# In[17]:


import os
print(os.listdir("../input"))


# In[18]:


print(os.listdir("../input/train"))


# In[19]:


from sklearn.model_selection import train_test_split
PATH = "../input/"
root_prefix = PATH
train_filenames = os.listdir('%s/train/' % (root_prefix))
print("Sample of Training Data:", train_filenames[0:10])
test_filenames  = os.listdir('%s/test/'  % (root_prefix))
print("\nSample of Testing Data:", test_filenames[0:10])


# In[20]:


my_train = train_filenames
my_train, my_cv = train_test_split(train_filenames, test_size=0.1, random_state=0)
print("Number of Training Images:",len(my_train))
print("Number of Testing Images:", len(my_cv))


# *Step 2: Reorganize Data Using Symbolic Links*

# In[21]:


import shutil
from pathlib import Path
# Make symlinks
get_ipython().system('cp -as "$(pwd)/../input/" "$(pwd)/COPY"')
root_prefix = 'COPY'

def remove_and_create_class(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.mkdir(dirname+'/cat')
    os.mkdir(dirname+'/dog')

remove_and_create_class('%s/train' % (root_prefix))
remove_and_create_class('%s/valid' % (root_prefix))

for filename in filter(lambda x: x.split(".")[0] == "cat", my_train):
    os.symlink('%s/train/' % (root_prefix)+filename, '%s/train/cat/' % (root_prefix)+filename)
for filename in filter(lambda x: x.split(".")[0] == "dog", my_train):
    os.symlink('%s/train/' % (root_prefix)+filename, '%s/train/dog/' % (root_prefix)+filename)
for filename in filter(lambda x: x.split(".")[0] == "cat", my_cv):
    os.symlink('%s/train/' % (root_prefix)+filename, '%s/valid/cat/' % (root_prefix)+filename)
for filename in filter(lambda x: x.split(".")[0] == "dog", my_cv):
    os.symlink('%s/train/' % (root_prefix)+filename, '%s/valid/dog/' % (root_prefix)+filename)


# *Step 3: Describe New Data Organization*

# In[22]:


PATH = 'COPY'
print(os.listdir('COPY/train'))
print(os.listdir('COPY/valid'))


# In[23]:


print(os.listdir('COPY/valid/cat'))


# In[24]:


# Remove symlinks before committing
get_ipython().system('rm -rf "$(pwd)/COPY"')


# *Step 4: Proceed with analysis using your newly reorganized data*
