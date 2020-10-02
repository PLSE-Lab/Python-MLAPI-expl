#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# This kernel is dedicated to extensive EDA and exploration of this Data Science for Good competition.
# 
# Work is in progress.

# In[ ]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# import datetime
# from scipy import stats
# from scipy.sparse import hstack, csr_matrix
# from sklearn.model_selection import train_test_split

plt.style.use('seaborn-notebook')

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools

init_notebook_mode(connected=True)


# ## Working with data structure

# We have unusually sophisticated data structure here.

# In[ ]:


files = os.listdir("../input/cpe-data/")
print(files)


# At first we have separate folders for each of 6 departments. Let's choose one of them for now.

# In[ ]:


files = os.listdir("../input/cpe-data/Dept_11-00091/")
files


# Here we have one folder for shape files, aka geodata and one folder for tabular data.

# In[ ]:


files = os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/")
files


# Then we have 5 folders for various data.

# In[ ]:


files = os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing")
files


# And at last we have the data itself and metadata.

# In[ ]:


owner_data_11 = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing/ACS_16_5YR_S2502_with_ann.csv')


# In[ ]:


owner_data_11.head()


#     Hm. It seems that the first row contains column description.

# In[ ]:


plt.title(owner_data_11['HC01_EST_VC01'][0]);
plt.hist(owner_data_11['HC01_EST_VC01'][1:].astype(int));


# But plotting data from single files isn't very interesting, let's try combining data from all departments.

# In[ ]:


owner_data = pd.DataFrame()
for i, path in enumerate(glob.glob("../input/cpe-data/D*")):
    dep_num = path.split('_')[1]
    df = pd.read_csv(path + '/' + dep_num + '_ACS_data/' + dep_num + '_ACS_owner-occupied-housing/ACS_16_5YR_S2502_with_ann.csv')
    df['dep_num'] = dep_num
    if i == 0:
        owner_data = df.iloc[1:, :]
        col_names = {k: v for k, v in list(zip(df.columns, df.iloc[0, :].values))}
    else:
        owner_data = owner_data.append(df.iloc[1:, :])


# In[ ]:


owner_data.head()


# In[ ]:


data = []
for i in owner_data['dep_num'].unique():
    trace = go.Histogram(x=owner_data.loc[owner_data['dep_num'] == i, 'HC01_EST_VC01'], name=i)
    data.append(trace)
    
layout = dict(title=f"{col_names['HC01_EST_VC01']} by department", margin=dict(l=200), width=800, height=400)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Now we can compare values across departments!

# In[ ]:




