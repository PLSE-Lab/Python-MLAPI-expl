#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pip install -q kaggle 


# In[ ]:


from google.colab import files 
uploaded = files.upload( ) 


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('mkdir ~/.kaggle')


# In[ ]:


get_ipython().system(' touch /root/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system(' chmod 600 /root/.kaggle/kaggle.json')


# In[ ]:


#give your user name in quotes and copy key from json file which can be downloaded manually from kaggle website- My account-create API token, 
#Ensure kaggle.json is in the location ~/.kaggle/kaggle.json to use the API.
get_ipython().system(' echo \'{"username":"","key":""}\' >> /root/.kaggle/kaggle.json')


# In[ ]:


#in place of competition name give your desired competition, here ieee-... competition is used as example
get_ipython().system('kaggle competitions download -c ieee-fraud-detection')


# In[ ]:


#u will see list of all files 
get_ipython().system('ls')


# In[ ]:


get_ipython().system('unzip filename')


# In[ ]:


#after preparing your final submit file load it into submission.csv as shown below
submission.to_csv('submission.csv')


# In[ ]:


#use following command to submit with your own competition name, here champs-scalar-coupling competition is used
get_ipython().system('kaggle competitions submit -c champs-scalar-coupling -f submission.csv -m "Message"')

