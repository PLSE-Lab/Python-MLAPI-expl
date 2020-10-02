#!/usr/bin/env python
# coding: utf-8

# # Startup Story of India
# 
# ![](https://indianaffiliateprograms.com/wp-content/uploads/2016/01/rise-of-india-as-a-startup-nation.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use("fivethirtyeight")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from IPython.display import IFrame

# Any results you write to the current directory are saved as output.


# # Which city has maximum startups ?
# #### Startup Count by City

# In[ ]:


IFrame('https://public.tableau.com/views/StartupIndiaMapbyCity/Dashboard1?:embed=y&:showVizHome=no', width=1150, height=925)


# In[ ]:


# null_counter = startup.isnull().sum().to_frame()
# null_counter.columns = ["Values"]
# null_counter.sort_values(by="Values").plot(kind="barh",figsize = (20,8))
# plt.xlabel("Missing Values")
# plt.ylabel("Column Name")
# plt.title("Missing Value By Column")


# In[ ]:


# startup.dtypes.value_counts().plot(kind="barh",figsize = (20,8))
# plt.xlabel("Number of Columns")
# plt.ylabel("data Types")
# plt.title("Column By Datatype")


# In[ ]:


# for i in startup.select_dtypes(['object']):
#     print(i,"null value:",startup[i].isnull().sum())


# In[ ]:


# for i in startup.select_dtypes(['int64']):
#     startup[i] = startup[i].fillna(startup[i].mean())


# In[ ]:


# for i in startup.select_dtypes(['object']):
#     startup[i] = startup[i].fillna('other')


# In[ ]:


# startup.to_csv("startup_fill.csv",index =False)


# # Startup Funding By City

# In[ ]:


IFrame('https://public.tableau.com/views/StartupFundingMapbyCity/StartupFundingByCity?:embed=y&:showVizHome=no', width=1150, height=925)
# https://public.tableau.com/views/StartupFundingMapbyCity/StartupFundingByCity?:embed=y&:display_count=yes&publish=yes


# ## How much funds does startups generally get in India ?

# In[ ]:


IFrame('https://public.tableau.com/views/HowmuchfundsdoesstartupsgenerallygetinIndia/HowmuchfundsdoesstartupsgenerallygetinIndia?:embed=y&:showVizHome=no', width=1150, height=925)
# https://public.tableau.com/views/HowmuchfundsdoesstartupsgenerallygetinIndia/Dashboard3?:embed=y&:display_count=yes&publish=yes

# https://public.tableau.com/views/HowmuchfundsdoesstartupsgenerallygetinIndia/HowmuchfundsdoesstartupsgenerallygetinIndia?:embed=y&:display_count=yes&publish=yes


# # Major Investor in India(Who is the important investors in the Indian Ecosystem?)

# In[ ]:


IFrame('https://public.tableau.com/views/MajorInvestorinIndia/MajorInvestor?:embed=y&:showVizHome=no', width=1150, height=600)

# https://public.tableau.com/views/MajorInvestorinIndia/MajorInvestor?:embed=y&:display_count=yes&publish=yes


# # Investment type count(What are different types of funding for startups ?)

# In[ ]:


IFrame('https://public.tableau.com/views/Investmenttypecount/InvestmentType?:embed=y&:showVizHome=no', width=1150, height=230)
# https://public.tableau.com/views/Investmenttypecount/InvestmentType?:embed=y&:display_count=yes&publish=yes


# # Year and Quarterly Funding

# In[ ]:


IFrame('https://public.tableau.com/views/QuarterlyFundinginIndia/QuarterlyFundinginIndia?:embed=y&:showVizHome=no', width=1150, height=600)
# https://public.tableau.com/views/QuarterlyFundinginIndia/QuarterlyFundinginIndia?:embed=y&:display_count=yes&publish=yes


# # Most Funding Company

# In[ ]:


IFrame('https://public.tableau.com/views/MostFundingCompany/MostFundingCompany?:embed=y&:showVizHome=no', width=1150, height=800)
# https://public.tableau.com/views/MostFundingCompany/MostFundingCompany?:embed=y&:display_count=yes&publish=yes


# # Industry wise sub vertical plateform

# In[ ]:


IFrame('https://public.tableau.com/views/Industrywisesubverticalplateform/Industrywisesubverticalplateform?:embed=y&:showVizHome=no', width=1150, height=800)


# # Number of Investor by Company

# In[ ]:


IFrame('https://public.tableau.com/views/NumberofInvestornamebyCompany/NumberofInvestornamebyCompany?:embed=y&:showVizHome=no', width=1150, height=800)

# https://public.tableau.com/views/NumberofInvestornamebyCompany/NumberofInvestornamebyCompany?:embed=y&:display_count=yes&publish=yes


# # Funding type by City

# In[ ]:


IFrame('https://public.tableau.com/views/FundingtypesbyCity/FundingtypesbyCity?:embed=y&:showVizHome=no', width=1150, height=800)
# https://public.tableau.com/views/FundingtypesbyCity/FundingtypesbyCity?:embed=y&:display_count=yes&publish=yes


# ## Stay Tuned for more Update!!!

# In[ ]:




