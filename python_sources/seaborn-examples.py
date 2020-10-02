#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings  
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/Iris.csv')
df.head()


# In[ ]:


ax = sns.scatterplot(x="PetalLengthCm", y="SepalLengthCm", hue="Species",data=df)
sns.set(rc={'figure.figsize':(9,8)})


# In[ ]:


ax = sns.barplot(x="PetalWidthCm", y="SepalWidthCm", data=df)
sns.set(rc={'figure.figsize':(9,8)})


# In[ ]:


ax = sns.lineplot(x="PetalWidthCm", y="SepalWidthCm", hue="Species",data=df)
sns.set(rc={'figure.figsize':(9,8)})


# In[ ]:


ax = sns.boxplot(x="Species", y="SepalLengthCm",data=df)
sns.set(rc={'figure.figsize':(9,8)})


# In[ ]:


sns.set(style="darkgrid")
iris = pd.read_csv('../input/Iris.csv')


# Subset the iris dataset by species
setosa = iris.query("Species == 'Iris-setosa'")
virginica = iris.query("Species == 'Iris-virginica'")

# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot( setosa.SepalLengthCm,setosa.SepalWidthCm,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.SepalLengthCm,virginica.SepalWidthCm, 
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]


# In[ ]:


sns.set(style="darkgrid")

g=sns.jointplot("SepalWidthCm", "SepalLengthCm", data=df, kind="reg",
                  xlim=(0, 6), ylim=(3, 8), color="m", height=6)


# In[ ]:


sns.pairplot(df, hue="Species")


# In[ ]:


sns.distplot(df.SepalLengthCm, bins=20)
sns.set(rc={'figure.figsize':(6,9)})

