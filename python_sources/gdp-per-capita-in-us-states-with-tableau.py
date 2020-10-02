#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.imgur.com/roB1UmS.jpg" width="800">

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('../input/gdp-per-capita-in-us-states/bea-gdp-by-state.csv')


# In[ ]:


data.shape


# In[ ]:


data.head(60)


# In[ ]:


data.describe()


# **GDP per capita in the US (average)**

# <img src="https://i.imgur.com/QpFjGEB.png" width="1000">

# **GDP per capita in 2017 (descending order)**

# <img src="https://i.imgur.com/rmH2gtH.png" width="1000">

# **Median GDP per capita in 2013**

# <img src="https://i.imgur.com/LJpq03O.png" width="1000">

# **GDP per capita in 2017**

# <img src="https://i.imgur.com/rMefihP.png" width="1000">

# **The overall growth of GDP per capita in the US (2013-2017)**

# <img src="https://i.imgur.com/Ah6nsvs.png" width="1000">

# **GDP per capita in the Southwest area**

# <img src="https://i.imgur.com/S3A4O8d.png" width="1000">

# **GDP per capita by state (MAX) in 2016**

# <img src="https://i.imgur.com/MYGuvD4.png" width="1000">

# **GDP per capita in 2013 with the Average Line**

# <img src="https://i.imgur.com/ECso5uI.png" width="1000">

# **GDP per capita (2013) compared between regions**

# <img src="https://i.imgur.com/wxKyRbZ.png" width="1000">

# **GDP per capita (2014) compared between regions**

# <img src="https://i.imgur.com/EYCZ6KF.png" width="1000">

# **Average GDP per capita compared between regions in 2015**

# <img src="https://i.imgur.com/cF1itu9.png" width="1000">

# **GDP per capita compared between regions in 2016**

# <img src="https://i.imgur.com/XNWdPJI.png" width="1000">

# **Information about GDP per capita by region in 2017**

# <img src="https://i.imgur.com/EDUTRrJ.png" width="1000">

# **GDP per capita in some states (2017)** 

# <img src="https://i.imgur.com/c5hBYVC.png" width="1000">

# **GDP per capita (MIN) in Arizona, Mississippi, Florida, Georgia and Texas in 2013-2017**

# <img src="https://i.imgur.com/LCfTewe.png" width="1000">
