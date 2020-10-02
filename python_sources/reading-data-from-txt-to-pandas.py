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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import apache_beam


# **

# **Crap! I prefer apache beam for making data pipe anyways lets install it
# **

# In[ ]:


get_ipython().system('pip install apache_beam')


# **Just amazing now we have to use pandas OK lets import OS and pandas modules **

# In[ ]:


import os
import pandas as pd


# *Lets make a dataframe to store the data*
# 1. First column - Text file name
# 2. Second column - Date
# 3****. Third Column -Content of the text file

# In[ ]:


df = pd.DataFrame(columns=['file_name','Date','observation'])

path = "../input/cityofla/CityofLA/Job Bulletins"


# 

# **Time to read the text files and append them to dataframe**

# In[ ]:







for filename in os.listdir(path):
    with open(os.path.join(path, filename),errors='ignore') as f:
        observation = f.read()

        if "Open Date:" in observation:
            job_bulletin_date = observation.split("Open Date:")[1].split("(")[0].strip()
            current_df = pd.DataFrame({'file_name':filename,'Date':job_bulletin_date,'observation': [observation]})
        df = df.append(current_df, ignore_index=True)


# You have all the data for this competition in one pandas dataframe now 

# In[ ]:


df


# # GOOD LUCK 
# ## Play with the data,wrangle it and dont forget to have fun
