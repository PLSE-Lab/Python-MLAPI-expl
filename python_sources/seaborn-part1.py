#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


from sklearn import datasets
iris = datasets.load_iris()


# In[ ]:


Y = iris.target
X = iris.data
data=pd.DataFrame(X,columns=['sepal length in cm\n' , 'sepal width in cm\n','petal length in cm\n','petal width in cm\n'  ])
data['target']=Y


# In[ ]:


data.head()


# # Matplotlib
#  -> providwa the raw building blocks for Seaborn's visualization
# 

#  -> It can also be used on its own to plot data

# In[ ]:


fig, ax = plt.subplots()
ax.hist(data['sepal length in cm\n'])


# # Pandas
#  -> is a foundational library for analyzing data
#  

#  -> It also supports basic plotting capability

# In[ ]:


data['sepal length in cm\n'].plot.hist()


# # Seaborn
#  -> seaborn supports complex visualizations of data

#  -> It is built on matplotlib and works best with pandas DataFrame

# In[ ]:


import seaborn as sns


# # Distplot
#  -> The distplot is similar to the histogram shown in previous example. By default, generates a Gaussian Kernel Density Estimate(KDE)

# In[ ]:


sns.distplot(data['sepal length in cm\n'])


# # Histogram vs Distplot

# **Pandas Histogram**
# 1.   Actual frequency of observation
# 2.   No automatic labels
# 3.   Wide bines
# 
# 
# 
# 
# 

# In[ ]:


data['sepal width in cm\n'].plot.hist()


# **seaborn distplot**
# 
# 1.   Automatic label on X axis
# 2.   Muted color palette
# 3.   KDE plot
# 4.   Narrow bins
# 

# In[ ]:


sns.distplot(data['sepal width in cm\n'])


# # Creating a **histogram**
# 
# 
# *   Distplot function has multiple optional arguments
# *   In order to plot a simple histogram, you can disable the **KDE** and specify the number of bins to use
# 
# 

# In[ ]:


sns.distplot(data['petal length in cm\n'],kde=False,bins=10)


# # Alternative data distributions
# 
# 
# *   A **Rug Plot** is an alternative way to view the distribution of data
# *   A **KDE curve** and rug plot can be combined
# 
# 

# In[ ]:


sns.distplot(data['petal length in cm\n'], hist=False, rug=True)


# # Further Customizations
# 
# *   The **distplot** function uses several functions including **kdeplot** and **rugplot**
# *   It is possible to further customize a plot bt passing arguments to the **bold** function
# 
# 

# In[ ]:


sns.distplot(data['petal length in cm\n'], hist=False, rug=True, kde_kws={'shade':True})


# #  Regression Plots in Seaborn

# # Regplot
#  
# 
# *   The **regplot** function generates a scatter plot with a regression line
# *   Usage is similar to the **distplot**
# *   The **data** and **x** annd **y** variables must be defined     
# 
# 

# In[ ]:


sns.regplot(x='petal length in cm\n',y='petal width in cm\n', data=data)


# In seaborn, their are more then one way to plot a same(or similar plot) 
# *  like: **regplot** -low level and **lmplot** -high level(but lmplot is much more powerful)    

# In[ ]:


sns.lmplot(x='petal length in cm\n',y='petal width in cm\n', data=data)


# # **lmplot faceting**
# 1.  Organize data by colors(***hue***)

# In[ ]:


sns.lmplot(x='petal length in cm\n',y='petal width in cm\n', data=data, hue="target")


# 2. Organize data by columns(***col***)

# In[ ]:


sns.lmplot(x='petal length in cm\n',y='petal width in cm\n', data=data, col="target")


# # Seaborn Styles
# 
# *   Seaborn has default configurations that can be applied with ***sns.set()***
# *   These styles can override matplotlib and pandas plots as well
# 
# 

# In[ ]:


data['petal width in cm\n'].plot.hist() #without sns.set()


# In[ ]:


sns.set() #with sns.set()
data['petal width in cm\n'].plot.hist()
#darkgrid theme


# There are several theme in seaborn and can be applyed by set_style()
# 
# 
# *   Example
# 
# 

# In[ ]:


for style in ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
  sns.set_style(style)
  sns.distplot(data['petal width in cm\n'])
  plt.show()


# # Removing axes with despine()
# 
# *   Sometimes plots are improved by removing elements
# *   Seaborn contains a shortcut for removing the spines of a plot
# 
# 

# In[ ]:


sns.set_style('white')
sns.distplot(data['petal length in cm\n'])
sns.despine(left=True)


# # Colours in seaborn
# 
# 
# *   Seaborn supports assigning colors to plot using **matplotlib** color codes
# 
# 
# 
# 

# In[ ]:


sns.set(color_codes=True)
sns.distplot(data['petal length in cm\n'],color ='g')


# # Palettes
# 
# 
# *   Seaborn uses the **set_palette()** function to define a palette
# 
# 

# In[ ]:


for p in sns.palettes.SEABORN_PALETTES:
  sns.set_palette(p)
  sns.distplot(data['sepal width in cm\n'])
  plt.title(p)
  plt.show()


# # Displaying Palettes
# 
# *  **sns.palplot()** *function displays a palette*
# *   **sns.color_palette()** *returms the current palette*
# 
# 

# In[ ]:


for p in sns.palettes.SEABORN_PALETTES:
  sns.set_palette(p)
  sns.palplot(sns.color_palette())
  plt.title(p)
  plt.show()


# # Defining Custom Plattes
#  1.  **Circular colors** = when the data is not ordered

# In[ ]:


sns.palplot(sns.color_palette("Paired", 12))


# 2. **Sequential colour** = when the data has a consistent range from high to low

# In[ ]:


sns.palplot(sns.color_palette("Blues", 12))


# 3.  **Diverging colors** = when both the low and high values are interesting

# In[ ]:


sns.palplot(sns.color_palette("BrBG", 12))


# # Custimizing with matplotlib
# **Matplotlib Axes**
# * Most customization available through *matplotlib axes* object
# * *Axes* can be passed to seaborn functions

# In[ ]:


fig, ax =plt.subplots()
sns.distplot(data['sepal width in cm\n'], ax=ax)
ax.set(xlabel="sepal width")


# **Further Customizations**
# * The axes object supports many common customizations

# In[ ]:


fig, ax =plt.subplots()
sns.distplot(data['sepal width in cm\n'], ax=ax)
ax.set(xlabel = "sepal width",
      ylabel = 'Distribution', xlim=(0,6),
title='visual of sepal width')


# # Combining Plots
# * It is possible to combine and configure multiple plots

# In[ ]:


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
sns.distplot(data['sepal width in cm\n'], ax=ax0)
sns.distplot(data['sepal length in cm\n'], ax=ax1)
ax1.axvline(x=6,label='six',linestyle='--')
ax1.legend()

