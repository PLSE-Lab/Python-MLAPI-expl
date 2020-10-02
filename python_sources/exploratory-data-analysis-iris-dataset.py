#!/usr/bin/env python
# coding: utf-8

# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.  <br>
# <br>The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other. 
# 
# 

# In[ ]:


import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

#Load Iris.csv into a pandas dataFrame.
iris = pd.read_csv("../input/iris-dataset/iris.csv")


# Pandas is mainly used for machine learning in form of dataframes. Pandas allow importing data of various file formats such as csv, excel etc. Pandas allows various data manipulation operations such as groupby, join, merge, melt, concatenation as well as data cleaning features such as filling, replacing or imputing null values.

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


# ## 2-D Scatter Plot

# In[ ]:


#2-D scatter plot:
#ALWAYS understand the axis: labels and scale.
#Here we are plotting sepal length aganist sepal width

iris.plot(kind='scatter', x='sepal_length', y='sepal_width');
plt.show()

#cannot make much sense out it. 
#What if we color the points by thier class-label/flower-type.


# In[ ]:


# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="species", height=4)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend();
plt.show();

# Notice that the blue points can be easily seperated 
# from red and green by drawing a line. 
# But red and green data points cannot be easily seperated.
# Can we draw multiple 2-D scatter plots for each combination of features?
# How many cobinations exist? 4C2 = 6.


# **Observation(s):**
# 1. Using sepal_length and sepal_width features, we can distinguish Setosa flowers from others. It is linearly separable.
# 2. Seperating Versicolor from Viginica is much harder as they have considerable overlap.

# ## 3D Scatter plot
# 
# Needs a lot to mouse interaction to interpret data. So 3-D plot is not commonly used.
# 
# What about 4-D, 5-D or n-D scatter plot?
# Humans are not evolved to visualize n-D space where n>3. We use math to see 4-D spaces.

# In[ ]:


df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                    color='petal_length', symbol='species')
fig.show()


# ## Pair-plot

# In[ ]:


#Hack used to visualize data in 4-D spaces
# pairwise scatter plot: Pair-Plot
#4C2 pair plots are possible
# Dis-advantages: Not practical if dimentions are very high
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
plt.close();
sns.set_style("whitegrid");
sns.pairplot(iris, hue="species", height=2);
plt.show()
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.


# **Observations**
# 1. petal_length and petal_width are the most useful features to identify various flower types.
# 2. While Setosa can be easily identified (linearly seperable), Virnica and Versicolor have some overlap (almost linearly seperable).
# 3. We can find "lines" and "if-else" conditions to build a simple model to classify the flower types.

# ## Histogram, PDF, CDF

# Univariate analysis: only one variable is consideredd for analysis.<br>
# Gives the number of points that exist in each of these windows. Aslo called density plots<br>
# x-axis: petal length<br>
# y-axis: Number of points<br>
# Lines: probability Density functions(PDF). It is a smoothed histogram. Further the seperation amoung PDF's, easier it will be to distinguish.<br>
# To distinguish between versicolor and verginica, intersection of their PDF's is the best point.<br>

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
#Disadvantages of 1-D scatter plot: Very hard to count points 
#are overlapping a lot.
#Are there better ways of visualizing 1-D scatter plots?


# In[ ]:


sns.FacetGrid(iris, hue="species", height=5)    .map(sns.distplot, "petal_length")    .add_legend();
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

# Gaussian/Normal distribution.
# What is "normal" about normal distribution?
# e.g: Hieghts of male students in a class.
# One of the most frequent distributions in nature.


# In[ ]:



# Need for Cumulative Distribution Function (CDF)
# We can visually see what percentage of setosa flowers have a petal_length of less than or equal to 1.6?
# How to construct a CDF?
# How to read a CDF?

#Plot CDF of petal_length

counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();


#  CDF's adds all values got at each points in PDF's upto that point<br>
#  CDF at a point is area under the curve of of PDF till that point <br>
#  Diffrentiation on CDF gives PDF<br>
#  Integration on PDF gives CDF<br>

# In[ ]:


# Need for Cumulative Distribution Function (CDF)
# We can visually see what percentage of versicolor flowers have a petal_length of less than 1.6?
# CDF's adds all values got at each points in PDF's upto that point
# CDF at a point is area under the curve of of PDF till that point 

# How to construct a CDF?
# How to read a CDF?

#Plot CDF of petal_length

counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();


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


# PDF's are used to find intersection points and CDF's are used to find the accuracy given point

# ## Mean, Variance and Std-dev

# In[ ]:


#Mean, Variance, Std-deviation,  
#Mean measures central tendancy.
print("Means:")
print(np.mean(iris_setosa["petal_length"]))
#Mean with an outlier (Mistake) Becoz of one outlier, Mean can vary drastically.
print(np.mean(np.append(iris_setosa["petal_length"],50)));
print(np.mean(iris_virginica["petal_length"]))
print(np.mean(iris_versicolor["petal_length"]))

#Varience is the spread/length of range
#Std-Deviation is the square root of varience
print("\nStd-dev:");
print(np.std(iris_setosa["petal_length"]))
print(np.std(iris_virginica["petal_length"]))
print(np.std(iris_versicolor["petal_length"]))

#All the three are currupted by outliers


# ## Median, Percentile, Quantile, IQR, MAD

# In[ ]:


#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(iris_setosa["petal_length"]))
#Median with an outlier. Not corrupted by outliers(except 50% of the values are corrupted)
print(np.median(np.append(iris_setosa["petal_length"],50)));
print(np.median(iris_virginica["petal_length"]))
print(np.median(iris_versicolor["petal_length"]))

#0th, 25th, 50th, 75th, 100th percentiles are called quantiles
print("\nQuantiles:")
print(np.percentile(iris_setosa["petal_length"],np.arange(0, 100, 25)))
print(np.percentile(iris_virginica["petal_length"],np.arange(0, 100, 25)))
print(np.percentile(iris_versicolor["petal_length"], np.arange(0, 100, 25)))

#90% of the values will be less than 90th percentile
print("\n90th Percentiles:")
print(np.percentile(iris_setosa["petal_length"],90))
print(np.percentile(iris_virginica["petal_length"],90))
print(np.percentile(iris_versicolor["petal_length"], 90))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(iris_setosa["petal_length"]))
print(robust.mad(iris_virginica["petal_length"]))
print(robust.mad(iris_versicolor["petal_length"]))


# ## Box plot and Whiskers

# In[ ]:



#Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitivey.
# The Concept of median, percentile, quantile.
# How to draw the box in the box-plot?
# How to draw whiskers: [no standard way] Could use min and max or use other complex statistical techniques. Wiskers are Lines.
# seaborn users Whiskers = 1.5 x IQR (IQR is width of the box)
# IQR like idea.

#NOTE: IN the plot below, a technique call inter-quartile range is used in plotting the whiskers. 
#Whiskers in the plot below donot correposnd to the min and max values.

#Box-plot can be visualized as a PDF on the side-ways. Each line in the box corresponds to 25th, 50th, 75th percentile.

sns.boxplot(x='species',y='petal_length', data=iris)
plt.show()


# Advantages of Box plots:- Error detection is very easy. If I put a threshold at 5, I can know how many verginica flowers will be classified wrongly.

# ## Violin plots

# In[ ]:


# A violin plot combines the benefits of the previous two plots 
#and simplifies them

# Denser regions of the data are fatter, and sparser ones thinner. Center part shows the box plot and bulgings shows PDF's
#in a violin plot

sns.violinplot(x="species", y="petal_length", data=iris, size=8)
plt.show()


# ## Univariate, bivariate and multivariate analysis

# Univariate: histogram, Box-Plots, Violin plots etc<br>
# Bivariate: pair-plots, scatter plots<br>
# Multivariate analysis: 3-D scatter plot<br>
# 

# ## Multivariate probability density, contour plot.
# 

# In[ ]:


#2D Density plot, contors-plot
#Points on the circles have same height
sns.jointplot(x="petal_length", y="petal_width", data=iris_setosa, kind="kde");
plt.show();


# In[ ]:




