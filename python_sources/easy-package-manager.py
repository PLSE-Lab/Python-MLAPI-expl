#!/usr/bin/env python
# coding: utf-8

# This is an easy way to get your requirements.txt into a dataset so your other notebooks can install all the packages without internet. This notebook needs internet to download the packages and zip them. 

# Instructions: 
# 
# 1. Clone this notebook and enable internet
# 2. Find the requirements section and paste your requirements.txt (pip freeze > requirements.txt) 
# 3. Commit the cloned notebook 
# 4. At the very bottom after the commit finished running, click new dataset, which will create a new dataset from the output
# 5. Add the dataset to your other notebook 
# 6. replace datasetname with your dataset's name and run !pip install  -r /kaggle/input/datasetname/packages/requirements.txt --no-index --find-links file:///kaggle/input/datasetname/packages
# 7. If you are having issues, run the walk command and it will print the paths (be careful not to print the paths to the training/test data). 
# 

# In[ ]:


import sys 
import platform
import os 
import zipfile 

def get_env_info(): 
    
    print(sys.platform)
    print(platform.python_implementation())
    print(sys.version)
    
get_env_info()


# In[ ]:


os.mkdir('packages')


# This is the requirements section. 

# In[ ]:


#copy paste your requirements file contents to here 

open('packages/requirements.txt', 'w').write('''
numpy>=1.16.0
scipy>=1.4.1
opencv-python
efficientnet
''')


# In[ ]:


#download packages to directory 
get_ipython().system('pip download -d packages -r packages/requirements.txt')


# In[ ]:


# #zip packages so you can create a dataset from them easily
# #actually don't this is dumb. just create a new dataset from the output files
# import shutil
# shutil.make_archive('packages.zip', 'zip', 'packages')


# In[ ]:


# shutil.rmtree('packages')


# In[ ]:


#extract again using below 
# shutil.unpack_archive('packages.zip.zip', 'packages')

