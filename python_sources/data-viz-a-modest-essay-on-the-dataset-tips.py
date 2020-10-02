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


# **1. Load the data**

# In[ ]:


tips = pd.read_csv("/kaggle/input/seaborn-tips-dataset/tips.csv")
tips.head()


# **2. Get the shape of the dataset**

# In[ ]:


tips.shape


# **3. Get types of data in the dataset**

# In[ ]:


tips.info()


# **4. Describe the data (numbers) in the dataset**

# In[ ]:


tips.describe()


# **5. Describe all the data in the dataset**

# In[ ]:


tips.describe(include="all")


# **6. cheking missing values (isna method) in the dataset**

# In[ ]:


tips.isna().sum()


# **6. cheking missing values (isnull method) in the dataset**

# In[ ]:


tips.isnull().sum()


# **7. Get correclation matrix of the dataset**

# In[ ]:


corr = tips.corr()
corr


# **8. Importing the matplotlib and seaborn librairies**

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# **9. Heatmap of the correlation matrix**

# In[ ]:


plt.figure(figsize=(8,8))
sns.heatmap(corr, annot=True, )
plt.title("HeatMap of correlation matrix of tips dataset")

plt.show()


# **10. Barplot in differents ways**

# In[ ]:


plt.figure(figsize=(16,14))

plt.subplot(321)
sns.barplot(tips.sex, tips.total_bill)
plt.title("Barplot of Total Bill Vs Sexe")

plt.subplot(322)
sns.barplot(tips.smoker, tips.total_bill)
plt.title("Barplot of Total Bill Vs Smoker")

plt.subplot(323)
sns.barplot(tips.sex, tips.total_bill, hue=tips.smoker)
plt.title("Barplot of Total Bill Vs Sexe and Smoker")

plt.subplot(324)
sns.barplot(tips.day, tips.total_bill, order=["Thur", "Fri", "Sat", "Sun"])
plt.title("Barplot of Total Bill Vs Days")

plt.subplot(313)
sns.barplot(tips.day, tips.total_bill, order=["Thur", "Fri", "Sat", "Sun"], hue=tips.sex)
plt.title("Barplot of Total Bill Vs Days")

plt.tight_layout()
plt.show()


# **11. Distplot in differents ways**

# In[ ]:


plt.figure(figsize=(16,6))

plt.subplot(121)
sns.distplot(tips.total_bill, kde=True)
plt.title("Distplot Total Bill (kde=True)")

plt.subplot(122)
sns.distplot(tips.total_bill, kde=False)
plt.title("Distplot Total Bill (kde=False)")


plt.show()


# **11. Joinplot in differents ways**

# In[ ]:


sns.jointplot(data=tips, x="total_bill", y="tip", height=7)

plt.show()


# In[ ]:


sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg", height=7)

plt.show()


# In[ ]:


sns.jointplot(data=tips, x="total_bill", y="tip", kind="resid", height=7)

plt.show()


# In[ ]:


sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde", height=7)

plt.show()


# In[ ]:


sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex", height=7)

plt.show()


# **12. Pairplot in differents ways**

# In[ ]:


sns.pairplot(tips)

plt.show()


# In[ ]:


sns.pairplot(tips, hue="sex")

plt.show()


# In[ ]:


sns.pairplot(tips, hue="smoker")

plt.show()


# In[ ]:


sns.pairplot(tips, hue="day", hue_order=["Thur", "Fri", "Sat", "Sun"])

plt.show()


# **13. Clustermap in differents ways**

# In[ ]:


num_tips = tips.select_dtypes(exclude="object").copy()
num_tips.head()


# In[ ]:


sns.clustermap(num_tips, metric="correlation")

plt.show()


# In[ ]:


sns.clustermap(num_tips, standard_scale=1)

plt.show()


# **14. Dendogram in differents ways**

# In[ ]:


from scipy.cluster.hierarchy import dendrogram, linkage
plt.figure(figsize=(15,10))

plt.subplot(211)
dendrogram(linkage(num_tips, method='ward'))

plt.subplot(212)
dendrogram(linkage(num_tips, method='single'))

plt.tight_layout()
plt.show()

