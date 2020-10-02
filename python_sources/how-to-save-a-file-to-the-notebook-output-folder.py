#!/usr/bin/env python
# coding: utf-8

# # How to save a .CSV file to the output folder of your Kaggle notebook

# *Step 1: Save the data to the working directory*

# In[ ]:


import pandas as pd
PHD_STIPENDS = pd.read_csv('/kaggle/input/phd-stipends/csv') # load from notebook input
PHD_STIPENDS.to_csv('/kaggle/working/phd_stipends.csv',index=False) # save to notebook output
PHD_STIPENDS.head(10)


# *Step 2: Download the data from the output folder of your Kaggle notebook*
# * Make sure you have committed your notebook 
# * Open the notebook viewer by removing "/edit" from the URL
# * Click on the "output" tab and download your file

# ![](https://i.imgur.com/3xdTfz8.png)

# In[ ]:




