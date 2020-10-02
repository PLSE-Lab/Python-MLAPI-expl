# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



# coding: utf-8

# # Introduction to Visualization in Python - Part 2

# ## 2.Seaborn

# ### Benefits 
# ##### 1. Visualizing information with matrices and dataframes
# ##### 2. Attractive statistical plots

# ### Import and Read Data

# In[51]:

### 
import pandas as pd
import seaborn as sns
sns.set(style="white", color_codes=True)


# In[52]:

# Read Dataset
import os 
#os.chdir("Datasets")
os.getcwd()


# In[53]:

irisdf = pd.read_csv('../input/Iris.csv',header=0,sep=',')
irisdf.head()


# In[54]:

irisdf.describe()


# ### Histogram

# In[55]:

## Plot 
sns.distplot(irisdf.SepalLengthCm,kde=False,rug=True)


# ### Boxplots

# In[56]:

sns.boxplot(data=irisdf.SepalLengthCm)


# ### Univariate & Bivariate on whole dataset

# In[61]:

sns.pairplot(irisdf[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']],hue='Species')


# ### Visualizing linear relationships among variables

# In[58]:

sns.lmplot(x = 'SepalLengthCm',y = 'PetalLengthCm',data =irisdf)


# In[59]:

sns.lmplot(x = 'SepalLengthCm',y = 'SepalWidthCm',data =irisdf)

