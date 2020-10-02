#!/usr/bin/env python
# coding: utf-8

# In[61]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 0.9.0 version
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
sns.set(style='darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


# Load data into a pandas dataframe
iris = pd.read_csv("../input/Iris.csv")

# See first 5 entries of dataframe
iris.head()


# There are two ways you can create a scatter plot with seaborn. The first is using *relplot* which was recently introduced in the 0.9.0 version. The second is using *lmplot*. Both methods are shown below.

# In[63]:


# Create a scatter plot with relplot
sns.relplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)


# In[64]:


# Create a scatter plot with lmtest
sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)


# Note that when we use *lmplot*, the function automatically plots a linear regression line and confidence intervals. We can easily remove this with the key argument *fit_reg=False*.

# In[65]:


# Scatter plot without regression line
sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', fit_reg=False, data=iris)


# Before we continue with other types of visualizations, let's learn how to label our plot as well as change its axis ranges. To do this, we will use matplotlib.pyplot commands.

# In[66]:


# Create figure
sns.relplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)

# Add title to figure
plt.title('Scatter Plot')

# Change x-axis label
plt.xlabel('Sepal Length (cm)')

# Change y-axis label
plt.ylabel('Sepal Width (cm)')

# Change x-axis range
plt.xlim(5,7)

# Change y-axis range
plt.ylim(2,4);


# We can also adjust the height and width of our figure using the inputs *height* and *aspect* respectively.

# In[67]:


sns.relplot(x='SepalLengthCm', y='SepalWidthCm', height=7, aspect=1.5, data=iris)

plt.title('Scatter Plot')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Condensed way of setting axis limits
plt.axis([5,7,2,4])


# Now let's look at further features we can add to our scatter plots to visualize the iris dataset better. The first thing we can do is change the color of the points based on which species it represents. This can be done by including the input *hue*. We set this equal to the *Species* variable, so each color corresponds to a different group.

# In[68]:


sns.relplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)


# We can do other fun things to our data points such as changing their shape! To change the shape of the points, we set *style* equal to a categorical variable which is *Species* in our case.

# In[69]:


sns.relplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', style='Species', data=iris)


# Instead of just creating one plot at a time, we can also create multiple subplots using *relplot* for each species of flower. We do this with the *col* input. As we see below, we create 3 separate plots for each species that compare *SepalLengthCm* and *SepalWidthCm*.

# In[70]:


sns.relplot(x='SepalLengthCm', y='SepalWidthCm', col='Species', data=iris)


# We can further explore the variable *Species* using categorical boxplots using *sns.catplot*. This compares each level in the category with a continuous variable.

# In[71]:


sns.catplot(x='Species', y='SepalLengthCm', data=iris)


# The categorical scatter plot automatically applies some "jitter", so the points are not in a straight line. If we want to remove the jitter, we set *jitter=False*.

# In[72]:


sns.catplot(x='Species', y='SepalLengthCm', jitter=False, data=iris)


# We can also the define the order of the groups on the x-axis using *order* which is set equal to a list of the names of each level in the desired order.

# In[73]:


sns.catplot(x='Species', y='SepalLengthCm', jitter=False, 
            order=['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'], data=iris)


# Another way to compare a categorical and continuous variable is with a boxplot. To do this, we continue using the *sns.catplot* function but set *kind="box"*.

# In[74]:


sns.catplot(x="Species", y="SepalLengthCm", kind="box", data=iris)


# Similar to a boxplot, a violin plot adds kernel density plot on each side. We can create this in seaborn by setting *kind="violin"*.

# In[75]:


sns.catplot(x="Species", y="SepalLengthCm", kind="violin", data=iris)


# We can also visualize the data points on top of the violin plot with *sns.swarmplot* which creates a categorical scatterplot, most appropiately used with a violin or boxplot.

# In[76]:


# First define the violin plot with the insides set to be empty
sns.catplot(x="Species", y="SepalLengthCm", kind="violin", inner=None, data=iris)

# Add scatterplot
sns.swarmplot(x="Species", y="SepalLengthCm", color="k", size=3, data=iris)


# To visualize the counts of each species in our dataset, we can use a bar graph. This can be created in two different ways. The first is using *sns.catplot* and setting *kind="count"*. The second way is to use *sns.countplot*.

# In[77]:


# Creating a bar graph using sns.catplot
sns.catplot(x='Species', kind='count', data=iris)


# In[78]:


# Creating a bar graph using sns.countplot
sns.countplot(x="Species", data=iris)


# Next, we can focus on looking at univariate distributions of our continuous variables. One of the eaiest ways to visualize an univariate distribution in seaborn is using the *distplot* function which creates both a histogram and kernel density estimator.

# In[79]:


# Note that the input is no longer x = "" but just the actual dataframe column
sns.distplot(iris['SepalLengthCm'])


# Instead of just plotting both a histogram and kernel density estimator at the same time, we can also do this separately. We use the same function *distplot* but set either *hist=False* or *kde=False* depending on what we want.

# In[80]:


# Create a histogram
sns.distplot(iris['SepalLengthCm'], kde=False)


# We can also add small vertical ticks at each observation by setting *rug=True*.

# In[81]:


# Add tick marks
sns.distplot(iris['SepalLengthCm'], kde=False, rug=True)


# The function *distplot* automatically chooses the number of bins by default for the histogram, but you can choose the number of bins by setting *bins* equal to whatever number you want.

# In[82]:


# Set number of bins equal to 20
sns.distplot(iris['SepalLengthCm'], bins=20)


# In[83]:


# Create a kernel density estimator: each observation is replaced with a Gaussian curve centered at its value 
# and the curves are summed to find the density value at the point; the resulting curve is normalized to have an area of 1
sns.distplot(iris['SepalLengthCm'], hist=False, rug=True)


# We can also create a KDE using *sns.kdeplot* which is generally easier to use when we just want a KDE plot.

# In[84]:


sns.kdeplot(iris['SepalLengthCm'], shade=True)


# Instead of just looking at univariate distributions, we can visualize the bivariate distribution of two variables using *jointplot*. We have already seen an example of a bivariate distribution with a scatter plot but now we can combine both a scatter plot and histogram.

# In[85]:


sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)


# We can also create a contour plot and KDE together too.

# In[86]:


# Set kind='kde'
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', kind='kde', data=iris)


# Lastly, we can visualize pairwise bivariate distributions in our dataset using just the *pairplot* function. On the diagonal, the univariate distribution of each variable is shown.

# In[87]:


# Input is name of dataframe
sns.pairplot(iris, hue='Species')


# 
