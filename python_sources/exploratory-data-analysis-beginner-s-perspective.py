#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis (EDA) a beginner's perspective.

# ## Iris Flower dataset

# #### We perform EDA to get insights to the data and how it could be relevant to our task. How to analyize data in a attractive and understandable manner.

# Toy  Dataset: Iris Dataset: [https://en.wikipedia.org/wiki/Iris_flower_data_set]
# * A simple dataset to learn the basics.
# * 3 flowers of Iris species. [see images on wikipedia link above]
# * 1936 by Ronald Fisher.
# * Petal and Sepal: http://terpconnect.umd.edu/~petersd/666/html/iris_with_labels.jpg
# *  Objective: What colud be the important features to classify a new flower as belonging to one of the 3 classes given the features.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Load Iris.csv into a pandas dataFrame.
iris = pd.read_csv("/kaggle/input/iris.csv")

iris.head()


# In[ ]:


# (Q) how many data-points and features?
print (iris.shape)


# In[ ]:


#(Q) What are the column names in our dataset?
print (iris.columns)


# In[ ]:


#(Q) How many data points for each class are present? 
#(or) How many flowers for each species are present?

iris["species"].value_counts()
# balanced-dataset vs imbalanced datasets
#Iris is a balanced dataset as the number of data points for every class is 50.


# # 2-D Scatter Plot

# In[ ]:


#2-D scatter plot:
#ALWAYS understand the axis: labels and scale.

iris.plot(kind='scatter', x='sepal_length', y='sepal_width') ;
plt.show()

#cannot make much sense out it. 
#What if we color the points by thier class-label/flower-type.


# In[ ]:


# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="species", size=4)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend();
plt.show();

# Notice that the blue points can be easily seperated 
# from red and green by drawing a line. 
# But red and green data points cannot be easily seperated.
# Can we draw multiple 2-D scatter plots for each combination of features?
# How many cobinations exist? 4C2 = 6.


# **Observation(s):**
# 1. Using sepal_length and sepal_width features, we can distinguish Setosa flowers from others.
# 2. Seperating Versicolor from Viginica is much harder as they have considerable overlap.

# ## 3D Scatter plot
# 
# To be very truthful you generally can't get much insisghts by using 3-D plots nor can you place it on a paper report. Also as the dimensions increse it get's harder to imagine the relative difference in distance among the points so we will skip it here, but we can do something very interesting and that is representing data of dimension 3 and 4 ona a 2-D representation which I willl coven in my next visualization and it will even more practical.
# 
# https://plot.ly/pandas/3d-scatter-plots/
# 
# Needs a lot to mouse interaction to interpret data.
# 
# What about 4-D, 5-D or n-D scatter plot?

# # Pair-plot

# In[ ]:


# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Can be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
plt.close();
sns.set_style("whitegrid");
sns.pairplot(iris, hue="species", height=3);
plt.show()
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.


# **Observations**
# 1. petal_length and petal_width are the most useful features to identify various flower types.
# 2. While Setosa can be easily identified (linearly seperable), Virnica and Versicolor have some overlap (almost linearly seperable).
# 3. We can find "lines" and "if-else" conditions to build a simple model to classify the flower types.

# # Histogram, PDF, CDF

# In[ ]:


# What about 1-D scatter plot using just one feature?
#1-D scatter plot of petal-length
import numpy as np
iris_setosa = iris.loc[iris["species"] == "setosa"];
iris_virginica = iris.loc[iris["species"] == "virginica"];
iris_versicolor = iris.loc[iris["species"] == "versicolor"];
#print(iris_setosa["petal_length"])
plt.plot(iris_setosa["petal_length"], np.zeros_like(iris_setosa['petal_length']), 'o')
plt.plot(iris_versicolor["petal_length"], np.zeros_like(iris_versicolor['petal_length']), 'o')
plt.plot(iris_virginica["petal_length"], np.zeros_like(iris_virginica['petal_length']), 'o')

plt.show()
#Disadvantages of 1-D scatter plot: Very hard to make sense as points 
#are overlapping a lot.
#Are there better ways of visualizing 1-D scatter plots?


# In[ ]:


np.zeros_like([[1,2,3],[1,2,3]]).shape


# In[ ]:


sns.FacetGrid(iris, hue="species", size=5)    .map(sns.distplot, "petal_length")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(iris, hue="species", size=5)    .map(sns.distplot, "petal_width")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(iris, hue="species", size=5)    .map(sns.distplot, "sepal_length")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(iris, hue="species", size=5)    .map(sns.distplot, "sepal_width")    .add_legend();
plt.show();


# In[ ]:


# Histograms and Probability Density Functions (PDF) using KDE
# How to compute PDFs using counts/frequencies of data points in each window.
# How window width effects the PDF plot.


# Interpreting a PDF:
## why is it called a density plot?
## Why is it called a probability plot?
## for each value of petal_length, what does the value on y-axis mean?
# Notice that we can write a simple if..else condition as if(petal_length) < 2.5 then flower type is setosa.
# Using just one feature, we can build a simple "model" suing if..else... statements.

# Disadv of PDF: Can we say what percentage of versicolor points have a petal_length of less than 5?

# Do some of these plots look like a bell-curve you studied in under-grad?
# Gaussian/Normal distribution.
# What is "normal" about normal distribution?
# e.g: Hieghts of male students in a class.
# One of the most frequent distributions in nature.


# In[ ]:



# Need for Cumulative Distribution Function (CDF)
# We can visually see what percentage of versicolor flowers have a 
# petal_length of less than 5?
# How to construct a CDF?
# How to read a CDF?

#Plot CDF of petal_length

counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(counts)
print(pdf);
print(bin_edges);
print(bin_edges.shape)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show();
print(cdf)


# In[ ]:


# Plots of CDF of petal_length for various types of flowers.

# Misclassification error if you use petal_length only.

counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# virginica
counts, bin_edges = np.histogram(iris_virginica['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#versicolor
counts, bin_edges = np.histogram(iris_versicolor['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


plt.show();


# #  Mean, Variance and Std-dev

# In[ ]:


get_ipython().run_line_magic('pinfo', 'np.append')


# In[ ]:


#Mean, Variance, Std-deviation,  
print("Means:")
print(np.mean(iris_setosa["petal_length"]))
#Mean with an outlier.
print(np.mean(np.append(iris_setosa["petal_length"],50)));
print(np.mean(iris_virginica["petal_length"]))
print(np.mean(iris_versicolor["petal_length"]))

print("\nStd-dev:");
print(np.std(iris_setosa["petal_length"]))
print(np.std(iris_virginica["petal_length"]))
print(np.std(iris_versicolor["petal_length"]))


# # Median, Percentile, Quantile, IQR, MAD

# In[ ]:


#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(iris_setosa["petal_length"]))
#Median with an outlier
print(np.median(np.append(iris_setosa["petal_length"],50)));
print(np.median(iris_virginica["petal_length"]))
print(np.median(iris_versicolor["petal_length"]))


print("\nQuantiles:")
print(np.percentile(iris_setosa["petal_length"],np.arange(0, 100, 25)))
print(np.percentile(iris_virginica["petal_length"],np.arange(0, 100, 25)))
print(np.percentile(iris_versicolor["petal_length"], np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(iris_setosa["petal_length"],90))
print(np.percentile(iris_virginica["petal_length"],90))
print(np.percentile(iris_versicolor["petal_length"], 90))

from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(iris_setosa["petal_length"]))
print(robust.mad(iris_virginica["petal_length"]))
print(robust.mad(iris_versicolor["petal_length"]))


# # Box plot and Whiskers

# In[ ]:



#Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitivey.
# The Concept of median, percentile, quantile.
# How to draw the box in the box-plot?
# How to draw whiskers: [no standard way] Could use min and max or use other complex statistical techniques.
# IQR like idea.

#NOTE: IN the plot below, a technique call inter-quartile range is used in plotting the whiskers. 
#Whiskers in the plot below donot correposnd to the min and max values.

#Box-plot can be visualized as a PDF on the side-ways.

sns.boxplot(x='species',y='petal_length', data=iris)
plt.show()


# # Violin plots

# In[ ]:


# A violin plot combines the benefits of the previous two plots 
#and simplifies them

# Denser regions of the data are fatter, and sparser ones thinner 
#in a violin plot

sns.violinplot(x="species", y="petal_length", data=iris, size=8)
plt.show()


# # Summarizing plots in english
# 
# * Remembering why we did EDA we see that now is the time to comelete our primary objective of exaplain our findings/conclusions in plain language.
# * Never forget your objective (the probelm you are solving) . Perform all of your EDA aligned with your objectives.
# 

# # Multivariate probability density, contour plot.
# 

# In[ ]:


sns.jointplot(x="petal_length", y="petal_width", data=iris_setosa, kind="kde");
plt.show();


# ### Hope you liked it and found useful so pls. Upvote.
