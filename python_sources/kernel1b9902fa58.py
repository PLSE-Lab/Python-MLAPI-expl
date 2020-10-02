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


# India's export and import visualization

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.formula.api as ols
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stat


# In[ ]:


import pandas as pd
export = pd.read_csv("../input/india-trade-data/2018-2010_export.csv")
Import = pd.read_csv("../input/india-trade-data/2018-2010_import.csv")


# In[ ]:


# Export data analysis 
# Indias most exported commodities 
# between 2010 - 2014
plt.figure(figsize = (12,8))
ex_com = export.groupby("Commodity")["value"].mean().sort_values(ascending = False).head(5).plot(kind = "bar", alpha = 0.5, color = ["r", "y", "b", "g", "m"], edgecolor = "b")


# In[ ]:


ex_co = export.groupby("Commodity").agg({"value":"mean"})
sns.barplot(x = ex_co.index, y = ex_co.value)


# In[ ]:



# Trade between top 5 countries 
plt.figure(figsize = (12,8))
ex_con = export.groupby("country")["value"].mean().sort_values(ascending = False).head(5).plot(kind = "bar", alpha = 0.5, color = ["r", "y", "b", "g", "m"], edgecolor = "b")


# In[ ]:


# growth of trade of each top countries over years
plt.figure(figsize = (12,8))
USA = export.query("country == 'U S A'").groupby("year")["value"].mean().plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "m")


# In[ ]:


plt.figure(figsize = (12,8))
UAE = export.query("country == 'U ARAB EMTS'").groupby("year")["value"].mean().plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "g")


# In[ ]:


plt.figure(figsize = (12,8))
H_KONG = export.query("country == 'HONG KONG'").groupby("year")["value"].mean().plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "y")


# In[ ]:


plt.figure(figsize = (12,8))
CHINA = export.query("country == 'CHINA P RP'").groupby("year")["value"].mean().plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "r")


# In[ ]:


plt.figure(figsize = (12,8))
SINGAPORE = export.query("country == 'SINGAPORE'").groupby("year")["value"].mean().plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


# Top country wise export
EX_USA = export.query("country == 'U S A'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
EX_USA.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


EX_UAE = export.query("country == 'U ARAB EMTS'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
EX_UAE .head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


EX_H_KONG = export.query("country == 'HONG KONG'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
EX_H_KONG.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


EX_CHINA = export.query("country == 'CHINA P RP'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
EX_CHINA.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


EX_SINGAPORE = export.query("country == 'SINGAPORE'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
EX_SINGAPORE.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


# Import data analysis 
Import.head()


# In[ ]:


Import.shape


# In[ ]:



# Indias most imported commodities 
im_com = Import.groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
im_com.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:



# Import trade between top 5 countries 
im_con = Import.groupby("country")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
im_con.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


# Growth of import from each top countries over years
plt.figure(figsize = (12,8))
Iraq = Import.query("country == 'IRAQ'").groupby("year")["value"].mean().plot(kind = "line",  color = "r", alpha = 0.5)


# In[ ]:


plt.figure(figsize = (12,8))
China = Import.query("country == 'CHINA P RP'").groupby("year")["value"].mean().plot(kind = "line",  color = "y", alpha = 0.5)


# In[ ]:


plt.figure(figsize = (12,8))
Saudi = Import.query("country == 'SAUDI ARAB'").groupby("year")["value"].mean().plot(kind = "line",  color = "b", alpha = 0.5)


# In[ ]:


plt.figure(figsize = (12,8))
Venezuela = Import.query("country == 'VENEZUELA'").groupby("year")["value"].mean().plot(kind = "line",  color =  "g", alpha = 0.5)


# In[ ]:


plt.figure(figsize = (12,8))
Angola = Import.query("country == 'ANGOLA'").groupby("year")["value"].mean().plot(kind = "line",  color = "m", alpha = 0.5)


# In[ ]:


plt.figure(figsize = (12,8))
Usa = Import.query("country == 'U S A'").groupby("year")["value"].mean().plot(kind = "line",  color = "Black", alpha = 0.5)


# In[ ]:


# Top country wise import
IM_Iraq =Import.query("country == 'U S A'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
IM_Iraq.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


IM_China = Import.query("country == 'CHINA P RP'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
IM_China .head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


IM_H_Saudi = Import.query("country == 'SAUDI ARAB'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
IM_H_Saudi.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


IM_Venezuela = Import.query("country == 'VENEZUELA'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
IM_Venezuela.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


IM_Angola = Import.query("country == 'ANGOLA'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
IM_Angola.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:


IM_Usa = Import.query("country == 'U S A'").groupby("Commodity")["value"].mean().sort_values(ascending = False)
plt.figure(figsize = (12,8))
IM_Usa.head(5).plot(kind = "bar", color = ["r", "y", "b", "g", "m"], alpha = 0.5, edgecolor = "b")


# In[ ]:



# TRADE SLIP

ex = export["value"].mean()

im = Import["value"].mean()

deficit = ex - im

