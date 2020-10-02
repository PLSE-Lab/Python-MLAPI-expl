#!/usr/bin/env python
# coding: utf-8

# # Goal: Practice different types of plots and scenarios using Python - Seaborn
# 
# 
# 1. [Histogram with NAs dropped](#optparam1)
# 2. [ Changing figure size using matplotlib.pyplot](#optparam2)
# 3.  [Setting the number of bins for histogram and not displaying Kernel Density Estimation curve](#optparam3)
# 4.  [Plotting just the KDE without histogram](#optparam4)
# 5.  [Setting the color for histogram](#optparam5)
# 6.  [Just the estimated density function with Shading. No bar histogram](#optparam6)
# 7.  [7. Multiple density plots on the same graph ](#optparam7)
# 8.  [Comparing two CONTINUOUS features using *jointplot()*](#optparam8)
# 9.  [Comparing all continuous features using pairplot()](#optparam9)
# 10.  [Controlling axes LIMITS in jointplot() if features are at a different scale](#optparam10)
# 11.  [Representing a THIRD DIMENSION - COLOR in *pairplot()*](#optparam11)
# 12.  [Linear regression line in pairplot()](#optparam12)
# 13. [In pairplot(), representing KDE plots on the diagonal instead of HISTOGRAM](#optparam13)
# 14. [Correlation Plot - Heatmap](#optparam14)
#  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


olympics = pd.read_csv("../input/athlete_events.csv")
olympics.head()


# In[ ]:


olympics.shape


# <a id="optparam1"></a>
# 
# ## 1. Histogram with NAs dropped

# In[ ]:


sns.distplot(olympics['Height'].dropna())


# <a id="optparam2"></a>
# 
# ## 2. Changing figure size using matplotlib.pyplot

# In[ ]:


# Changing the figure size using pyplot
f, ax = plt.subplots(figsize=(15,5))
sns.distplot(olympics['Weight'].dropna())


# <a id="optparam3"></a>
# 
# ## 3. Setting the number of bins for histogram and not displaying Kernel Density Estimation curve

# In[ ]:


# Changing the figure size using pyplot
f, ax = plt.subplots(figsize=(15,5))
# specifying bins and removing KDE
sns.distplot(olympics['Weight'].dropna(), bins=75, kde=False)


# <a id="optparam4"></a>
# 
# ## 4. Plotting just the KDE without histogram

# In[ ]:


# Just showing KDE and not the histogram
f, ax = plt.subplots(figsize=(15,5))
sns.distplot(olympics['Weight'].dropna(), hist=False, kde=True)


# <a id="optparam5"></a>
# 
# ## 5. Setting the color for histogram

# In[ ]:


f, ax = plt.subplots(figsize=(15,5))
# Setting a color for plot
sns.distplot(olympics['Weight'].dropna(), bins=50, kde=False, color="g")


# <a id="optparam6"></a>
# 
# ## 6. Just the estimated density function with Shading. No bar histogram

# In[ ]:


f, ax = plt.subplots(figsize=(15,5))
# Setting a color for KDE plot
sns.kdeplot(olympics['Weight'].dropna(), shade=True, color="r")


# <a id="optparam7"></a>
# 
# ## 7. Multiple density plots on the same graph 

# In[ ]:


f, ax = plt.subplots(figsize=(15,5))
# Multiple KDE plots on one graph
sns.kdeplot(olympics['Weight'].dropna(), color="r", label="Weight")
sns.kdeplot(olympics['Height'].dropna(), color="g", label="Height")


# <a id="optparam8"></a>
# 
# ## 8. Comparing two CONTINUOUS features using *jointplot()*

# In[ ]:


# A different kind of plot to compare two continuous variables
sns.jointplot(x="Weight", y="Height", data=olympics)


# <a id="optparam9"></a>
# 
# ## 9. Comparing all continuous features using*pairplot()

# In[ ]:


# Pairplot
sns.pairplot(olympics.dropna(), size=4)


# <a id="optparam10"></a>
# 
# ## 10. Controlling axes LIMITS in *jointplot()* if features are at a different scale

# In[ ]:


# Joint plot with controlled limits
sns.jointplot(x="Weight", y="Height", data=olympics, xlim=(25,175), ylim=(140,200))


# <a id="optparam11"></a>
# 
# ## 11. Representing a THIRD DIMENSION - COLOR in *pairplot()*

# In[ ]:


# Representing a third dimension color in a pairplot
sns.pairplot(olympics.dropna(), hue="Medal")


# <a id="optparam12"></a>
# 
# ## 12. Linear regression line in *pairplot()*

# In[ ]:


# Representing a regression line in the bivariate relationships in a pairplot
sns.pairplot(olympics[['Height', 'Weight', 'Age']].dropna(), kind="reg")


# <a id="optparam13"></a>
# 
# ## 13. In *pairplot()*, representing KDE plots on the diagonal instead of HISTOGRAM

# In[ ]:


# Representing KDE plots instead of histograms on the diagonal
sns.pairplot(olympics[['Height', 'Weight', 'Age']].dropna(), diag_kind="kde")


# <a id="optparam14"></a>
# 
# ## 14. CORRELATION PLOT - Heatmap

# In[ ]:


# Representing correlations between various features in the data as a heatmap
corrmat = olympics.dropna().corr()
f, ax = plt.subplots(figsize=(10,10))
# annot controls annotations, square=True outputs squares as correlation representing figures, cmap represents color map
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt=".2f", cmap="summer")

