#!/usr/bin/env python
# coding: utf-8

# <h1>Data Visualization for Machine Learning Introduction</h1>
# In this kernel we are learning how to use Matplotlib, Pandas and Seaborn to plot basic graphs like scatter plots and histograms and after he got the basics down we will look at more advanced graphs and graphing technics.
# 
# 1. Importing Datasets  
# 2. Matplotlib Introduction  
#     2.1 Import Matplotlib  
#     2.2 Basic plots  
# 3. Pandas Visualization  
#     3.1 Basic Plots  
# 4. Seaborn  
#     4.1 Importing Libary  
#     4.2 Basic Plots  
# 5. More advanced graphs  
#     5.1 Box and Violin Plots  
#     5.2 Heatmap  
#     5.3 Faceting  

# <h2>1. Importing Datasets</h2>
# For this Kernel we will use the [Iris](https://archive.ics.uci.edu/ml/datasets/iris) and the [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews) datasets.

# In[ ]:


import pandas as pd # data processing


# In[ ]:


iris = pd.read_csv('../input/iris/Iris.csv')
iris.head()


# In[ ]:


wine = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
wine.head()


# <h2>2. Matplotlib Introduction</h2>
# Matplotlib is the main plotting libary of the Python programing language. All of the other libarys are build on Matplotlib.

# <h4>2.1 Import Matplotlib</h4>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# <h4>2.2 Basic plots</h4>

# In[ ]:


fig, ax = plt.subplots()

ax.scatter(iris['SepalLengthCm'], iris['SepalWidthCm'])
ax.set_title('Iris dataset')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')


# We can color each point by its class

# In[ ]:


colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
fig, ax = plt.subplots()
for i in range(len(iris['SepalLengthCm'])):
    ax.scatter(iris['SepalLengthCm'][i], iris['SepalWidthCm'][i], color=colors[iris['Species'][i]])
ax.set_title('Iris dataset')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')


# In[ ]:


columns = iris.columns.drop(['Species', 'Id'])
fig, ax = plt.subplots()
for column in columns:
    ax.plot(iris[column])
ax.set_title('Iris Dataset')
ax.legend()


# In[ ]:


fig, ax = plt.subplots()
ax.hist(wine['points'])
ax.set_title('Wine Review Scores')
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')


# In[ ]:


fig, ax = plt.subplots()
data = wine['points'].value_counts()
ax.bar(data.index, data.values)


# <h2>3. Pandas Visualization</h2>
# Pandas uses Matplotlib and makes it easy to plot data stored in a pandas series or dataframe
# <h4>3.1 Basic Plots</h4>

# In[ ]:


iris.plot.scatter(x='SepalLengthCm', y='SepalWidthCm', title='Iris Dataset')


# In[ ]:


iris.drop(['Species', 'Id'], axis=1).plot.line(title='Iris Dataset')


# In[ ]:


wine['points'].plot.hist()


# In[ ]:


wine['points'].value_counts().sort_index().plot.bar()


# <h2>4. Seaborn</h2>
# Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

# <h4>4.1 Importing Libary</h4>

# In[ ]:


import seaborn as sns
sns.__version__


# <h4>4.2 Basic Plots</h4>

# In[ ]:


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)


# Highlighting the classes is significally easier than using Matplotlib. We only need to specify the hue parameter:

# In[ ]:


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)


# In[ ]:


sns.lineplot(data=iris.drop(['Species', 'Id'], axis=1))


# In[ ]:


sns.distplot(wine['points'], bins=10, kde=False)


# In[ ]:


sns.countplot(wine['points'])


# <h2>5. More advanced graphs</h2>
# Now that you have a basic understanding about the syntax of Matplotlib, Pandas Visualization and Seaborn I want to show you a few other graph types usefull for Data Science and Machine Learning. For most of them Seaborn is the go to Libary because you can use it to make complicated graphs with almost no lines of code.

# <h4>5.1 Box and Violin Plots</h4>
# Box and Violin Plots are useful to show distributions.

# In[ ]:


df = wine[(wine['points']>=95) & (wine['price']<1000)]
sns.boxplot('points', 'price', data=df)


# In[ ]:


sns.violinplot('points', 'price', data=df)


# <h4>5.2 Heatmap</h4>
# Heatmaps are perfect for exploring the correlation of the features

# In[ ]:


sns.heatmap(iris.corr(), annot=True)


# We can make plots bigger by using the figsize parameter from matplotlib. To use it with Seaborn we need to pass the seaborn function we are using the matplotlib axis as an argument.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(iris.corr(), ax=ax, annot=True)


# <h4>5.3 Faceting </h4>
# 
# Faceting is the act of breaking data variables up across multiple subplots, and combining those subplots into a single figure.

# <h4>5.3.1 FacetGrid (Seaborn)</h4>

# In[ ]:


g = sns.FacetGrid(iris, col='Species')
g = g.map(sns.kdeplot, 'SepalLengthCm')


# In[ ]:


g = sns.FacetGrid(iris, col='Species')
g = g.map(sns.kdeplot, 'SepalLengthCm', 'SepalWidthCm')


# In[ ]:


g = sns.FacetGrid(iris, col='Species')
g = g.map(plt.hist, 'PetalLengthCm', bins=10)


# <h4>5.3.2 Pairplot (Seaborn)</h4>

# In[ ]:


sns.pairplot(iris.drop(['Id'], axis=1), hue='Species')


# <h4>5.3.3 Scatter Matrix (Pandas)</h4>

# In[ ]:


from pandas.plotting import scatter_matrix

scatter_matrix(iris.drop(['Id', 'Species'], axis=1), diagonal='kde', figsize=(10, 10))


# That's all from this kernel I hope I could help you getting started with data visualization.
