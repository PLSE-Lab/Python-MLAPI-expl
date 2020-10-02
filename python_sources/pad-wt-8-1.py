#!/usr/bin/env python
# coding: utf-8

# ## Walkthrough 8-1: Plotting multiple relationships with Seaborn and Matplotlib

# ## Plot multiple graphs onto one figure

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[ ]:


# Set global styles for Matplotlib
sns.set_style('ticks')

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.spines.top': False,
         'axes.spines.right': False}

mpl.rcParams.update(params)

# Full reference of other possible parameters to set: https://matplotlib.org/users/customizing.html


# In[ ]:


# Load Motor Trend Car Road Tests data


'''
mpg -  Miles/(US) gallon
cyl -  Number of cylinders
disp - Displacement (cu.in.)
hp -   Gross horsepower
drat - Rear axle ratio
wt -   Weight (1000 lbs)
qsec - 1/4 mile time
vs -   V/S
am -   Transmission (0 = automatic, 1 = manual)
gear - Number of forward gears
carb - Number of carburetors
'''


# In[ ]:





# In[ ]:


## Create a figure with 4 rows of plots


## Graph distributions for mpg, disp, hp, and cyl on each axis


# Add labels
axes[0].set(xlabel='mpg',ylabel='frequency', title='Distribution: mpg');
axes[1].set(xlabel='disp',ylabel='frequency', title='Distribution: disp');
axes[2].set(xlabel='hp',ylabel='frequency', title='Distribution: hp');
axes[3].set(xlabel='cyl',ylabel='frequency', title='Distribution: cyl');

## Automatically set proper margins for each plot


# In[ ]:


# Create a figure with 2 rows and 2 columns of plots


# Graph distributions for mpg, disp, hp, and cyl on each axis


# Set labels
axes[0][0].set(xlabel='mpg',ylabel='frequency', title='Distribution: mpg');
axes[0][1].set(xlabel='disp',ylabel='frequency', title='Distribution: disp');
axes[1][0].set(xlabel='hp',ylabel='frequency', title='Distribution: hp');
axes[1][1].set(xlabel='cyl',ylabel='frequency', title='Distribution: cyl');


# Automatically set proper margins for each plot
plt.tight_layout()


# ## Graph extra dimensions

# In[ ]:


## Graph relationship between hp and disp


# In[ ]:


## Add categorical variable of cylinders to further understand relationship of hp, disp, and cylinders


# In[ ]:


## Graph the average displacement for each cylinder class by transmission type


# By default, statistical estimator is the mean for barplot


# In[ ]:


## Use facets to divide up graphs into smaller subsets. This graphing concept is called "small multiples" or 
## "trellis plots"




grid.add_legend()

fig, axes = grid.fig, grid.axes
fig.set_size_inches(15,6)
fig.suptitle('Average Displacement',y=1,fontsize=17)
plt.subplots_adjust(top=.88)


# ## Matrix graphs
# 
# Matrix graphs is an application of "small multiple" plots. Matrix graphs are often used to show all possible relationships between multiple variables. Matrix graphs are a key tool to efficiently conduct multivariate statistics, especially for machine learning.

# In[ ]:


# Scatterplot matrix

'''Use Seaborn PairGrid function to plot pairwise relationships in a dataset. Refer to documentation for more examples and use cases:
https://seaborn.pydata.org/generated/seaborn.PairGrid.html#seaborn.PairGrid
'''

# grid = sns.PairGrid(dfCars,vars=['mpg','cyl','disp','hp','carb','qsec'])
# grid.map(plt.scatter)

## Plot distribution on the diagonals and scatter plots off the diagonals



# In[ ]:


## Plot same scatterplot matrix using Pandas scatter_matrix function


# In[ ]:


# Correlation matrix

# Demo

'''Correlation matrix is a great tool that allows you to efficiently look for variables that correlate to one another.
This matrix helps answer questions that revolve around relationships between variables.
'''

# Create correlation matrix using Pearson's method
corr = cars_df.corr()

# Create mask to only show values in lower left triangle of matrix
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, center=0, square=True,
            ax=ax, fmt='.2f', linewidths=.5,cbar_kws={"shrink": .5});


# ## High density scatter plots

# In[ ]:


# Hexbin

# Create data points
rs = np.random.RandomState(5)
n = 1000

x = rs.gamma(2, size=n)
y = -0.5 * x + rs.normal(size=n)

## Plot hexbin plot using Seaborn jointplot


# In[ ]:


# Kernel Density Estimator (Contour plot)

## Plot kde plot using Seaborn jointplot


#grid.ax_joint.collections[0].set_alpha(0)




## Add scatter plot on top of the contour plot

#grid.plot_joint(plt.scatter,marker='+',color='gold',linewidth=1)


# In[ ]:


## Plot kde plot using Seaborn kdeplot


# In[ ]:


# More references can be found here to easily plot categorical data: 
# https://seaborn.pydata.org/tutorial/categorical.html

