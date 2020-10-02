#!/usr/bin/env python
# coding: utf-8

# Hi! I want to share with you same small datasaving tricks. I suppose it could help somebody
# 1. When generating of new features takes a lot of time you can save results in additional kernel and use it in your main kernel. It will help you save a lot of time when you reopen your kernel or something goes wrong. 
# 
# You just need add new dataset in the kernel interface
# 
#  <img src="https://i.ibb.co/fYvt6xx/2019-03-29-19-22-29-2.png" style="float: left; width: 30%; margin-right: 5%; margin-bottom: 0.5em;">
#  <img src="https://i.ibb.co/rQD0tF2/2019-03-29-17-30-531.png" style="float: left; width: 64%; margin-right: 1%; margin-bottom: 0.5em;">
#  
#  
# 
# And write new ways to your data. Notice, the path to primary data will be changed

# In[ ]:


import pandas as pd
import os
os.listdir('../input/')


# In[ ]:


PATH_TO_PRIMARY_DATA = '../input/mlcourse-dota2-win-prediction'
PATH_TO_ADDITIONAL_DATA = '../input/source-kernel-example'
df_train = pd.read_csv(os.path.join(PATH_TO_PRIMARY_DATA, 'train_features.csv'), index_col='match_id_hash')
df_train.head(2)


# In[ ]:


df_example= pd.read_csv(os.path.join(PATH_TO_ADDITIONAL_DATA, 'output.csv'), index_col='Unnamed: 0')
df_example.head(2)


# 2. You can save data from the kernel without commiting. Just use this code

# In[ ]:


from IPython.display import HTML
def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)


# In[ ]:


df_example.to_csv('example.csv')
create_download_link(filename='example.csv')


# Good luck!
