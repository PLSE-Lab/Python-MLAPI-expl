#!/usr/bin/env python
# coding: utf-8

# ## This notebook demos Python data visualizations on the Iris dataset
# 
# Dataset contains 50 sample observations of Three Iris flower species (Setosa, Versicolor, Virginica).
# 
# Each observation contains Four key dimensions (SepalLength, SepalWidth, PetalLength, PetalWidth) in Cm.
# 
# Three python libraries are primarily used for this tutorial: [pandas](http://pandas.pydata.org/), [matplotlib](http://matplotlib.org/), and [seaborn](http://stanford.edu/~mwaskom/software/seaborn/).
# 
# Press "Fork" at the top-right of this screen to run this notebook yourself and build each of the examples.

# In[ ]:


# First, import pandas, a data processing and CSV file I/O library
import pandas as pd

# Then import matplotlib, a Python 2D plotting library
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Also import seaborn, a Python graphing library
import seaborn as sns
sns.set(style="white", color_codes=True)

# Enable seaborn settings to ignore initial warnings, if any
import warnings
warnings.filterwarnings("ignore")

# Press shift+enter to execute the code in a cell


# In[ ]:


# Next, load the Iris flower dataset, which is in the "../input/" directory
iris = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame

# Let's see what's in the iris data
iris.head()


# In[ ]:


# Let's see how many examples we have of each species
iris["Species"].value_counts()


# ### Some of the plots we can make using Pandas and Matplotlib

# In[ ]:


# Make a boxplot with Pandas on each feature split out by species
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(15, 8))


# In[ ]:


# Use .plot extension to make a scatterplot of all the Iris features ('SepalLengthCm', 'SepalWidthCm').
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


# In[ ]:


# Use .add_subplot extension to produce Scatter plot on Sepal and Petal Dimensions, side by side.
x1 = iris['SepalLengthCm']
y1 = iris['SepalWidthCm']
x2 = iris['PetalLengthCm']
y2 = iris['PetalWidthCm']

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('SepalLengthCm')
ax1.set_ylabel('SepalWidthCm')
ax1.set_title("Scatter Plot on Sepal Dimensions")
ax1.scatter(x1,y1)

ax2 = fig.add_subplot(122)
ax2.set_xlabel('PetalLengthCm')
ax2.set_ylabel('PetalWidthCm')
ax2.set_title("Scatter Plot on Petal Dimensions")
ax2.scatter(x2,y2)


# In[ ]:


# One more sophisticated technique available in pandas is called Andrews Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")


# In[ ]:


# One more multivariate visualization technique available in pandas is parallel_coordinates
# Parallel coordinates plots each feature on a separate column & then draws lines
# connecting the features for each data sample
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")


# In[ ]:


# "radviz" is another multivariate visualization technique in pandas, which takes each feature
# as a point on a 2D plane, and then simulates having each sample attached to those points through
# a spring weighted by the relative value for that feature.
from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")


# ### Some of the plots we can make using Pandas and Seaborn

# In[ ]:


# Use seaborn's .FacetGrid extension to make similar scatterplot by species
sns.FacetGrid(iris, size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")


# In[ ]:


# One piece of information missing in the plot above is what species each flower is
# Use add_legends() and hue="Species" to color the scatterplot by species
sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()


# In[ ]:


# Seaborn's .jointplot extension shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)


# In[ ]:


# Seaborn's Hexbin plot showing above bivariate scatterplots and univariate histograms in the same figure
sns.set(style="ticks")
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", kind="hex", data=iris, color="#4CB391")


# In[ ]:


# Use seaborn's .boxplot extension to look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)


# In[ ]:


# Extend the above boxplot by adding a layer of individual points on top of it through "sns.stripplot".
# Use "jitter=True" to ensure all the points don't fall in single vertical lines above the species.
# Save the resulting axes as "ax" so that the resulting plot is shown on top of the previous axes.
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")


# In[ ]:


# A Seaborn's violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)


# In[ ]:


# One of the seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(iris, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()


# In[ ]:


# An useful seaborn pairplot, which shows the bivariate relation between each pair of features.
# In pairplot, we can see that the Iris-setosa species is separataed from the other two across all
# feature combinations.
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)


# In[ ]:


# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")


# # Wrapping Up
# 
# I hope you enjoyed this quick introduction to some of the simple data visualizations you can create with pandas, seaborn, and matplotlib in Python!
# 
# I encourage you to run through these examples yourself, tweaking them and seeing what happens. From there, you can try applying these methods to a new dataset and incorprating them into your own workflow!
