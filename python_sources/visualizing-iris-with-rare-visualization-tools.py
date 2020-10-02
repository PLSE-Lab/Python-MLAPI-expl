#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# * In this kernel, we will learn Rare Visualization Tools.
# 
# <br>Content:
# * [Matrix and Box Plots (Missingno)](#1)
# * [Parallel Plots (Pandas)](#2)
# * [Network Charts (Networkx)](#3)
# * [Venn (Matplotlib)](#4)
# * [Donut (Matplotlib)](#5)
# * [Spider Chart (Matplotlib)](#6) 
# * [Inset Plots (Plotly)](#7) 
# * [Basic 3D Scatter Plot (Plotly)](#8) 
# 
# * [Conclusion](#9) 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_venn as venn
from pandas.tools.plotting import parallel_coordinates
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


iris.info()


# In[ ]:


iris = iris.drop(['Id'],axis=1)


# In[ ]:


new_iris = iris.iloc[:,:3]
new_iris.SepalLengthCm[np.arange(1,150,10)] = np.nan
new_iris.PetalLengthCm[np.arange(25,120,7)] = np.nan
new_iris


# <a id="1"></a> <br>
# ## Matrix and Bar Plots (Missingno)
# * In data analysis, one of the first steps is cleaning messy datasets and missing values.
# * In order to explore whether data has missign value or not, I always use .info() method from pandas library. It gives a lot of information.
# * Visualization is always attractive for people. Therefore, if we can visualize missign values, it can be better understandable.
# * In order to visualize missign data, we can use missigno package.
# * Lets create pandas dataframe that includes missign values (NaN) and visualize it.
#     * Dictionary: One of the methods of creating data frame is first creating dictionary then put it into pd.DataFrame
#     * data_missingno: Data frame that we will use in this example
#     * import missingno as msno: import missingno library and define as msno (shortcut)
#     * matrix(): Create matrix. Number of rows is number of sample and number of columns is number of features(column1, column2, column3) in data_missingno.
#     * show(): shows the plot
# * The sparkline at right summarizes the general shape of the data completeness and points out the maximum and minimum rows.
# * Missign values are white and non missign values are black in plot.
# * It can be seen from plot column1 does not have missign value. Column2 has five missign values and column3 has three missign values.

# In[ ]:


# import missingno library
import missingno as msno

msno.matrix(new_iris)
plt.show()


# <a id="2"></a> <br>
# ## Parallel Plots (Pandas)
# * In order to learn parallel plots, we will use famous iris data set from sklearn library
# * Parallel plot allow to compare the feature of several individual observations on a set of numerical variables.
# * Each vertical bar represents a feature(column or variable) like petal length (cm).
# * Values are then plotted as series of lines connected across each axis.
# * Different colors shows different classes like setosa.
# * Parallel plot allow to detect interesting patterns. For example as you can see we can easily classify *setosa* according to *petal width (cm)* feature.
# * Lets look at code.
#     * Load iris data into data frame
#     * parallel_coordinates: we import parallel_coordinates from pandas library
#     * colormap: color map that paints classes with different colors

# In[ ]:


# Make the plot
plt.figure(figsize=(15,10))
parallel_coordinates(iris, 'Species', colormap=plt.get_cmap("Set2"))
plt.title("Iris data class visualization according to features (setosa, versicolor, virginica)")
plt.xlabel("Features of data set")
plt.ylabel("cm")
plt.savefig('graph.png')
plt.show()


# <a id="3"></a> <br>
# ## Network Charts (Networkx)
# * We will use iris data that we import at previous part( parallel plot)
# * Network charts are related with correlation network.
# * It can be used instead of heatmaps in seaborn library.
# * At first look at correlation between features.
#     * corr(): gives correlation between features
#     * As you can see from table below, petal length is positively correlated with sepal length and petal width. Also, petal length is negatively correlated with sepal width.
# * We have 4 individuals(features), and know how close they are related to each other (above correlation table).
# * It is possible to represent these relationships in a network
# * Each individual called as a node. If 2 individuals(features like sepal length and sepal width) are close enough (threshold), then they are linked by a line.
#     * threshold: threshold of the correlation. For example, if we say that threshold = 0.5, network will be established between the nodes that have higher correlation than 0.5
# * I will put -1 that is default threshold value. min(cor(A,B))= -1 so all nodes are connected with each other.
# * You can try threshold = 0.9, you will see that petal length and width are connected with each other.
# * It is alternative to heatmap.
# * As a final words of network charts, they can be used in data sets that are related with populations and their habits. Maybe,we can observe populations are clearly split in X groups according to their habits.
# * Now lets look at our code with iris datasets.

# In[ ]:


# Display positive and negative correlation between columns
iris.corr()


# In[ ]:


#sorts all correlations with ascending sort.
iris.corr().unstack().sort_values().drop_duplicates()


# In[ ]:


iris.corr().stack().reset_index()


# In[ ]:


# import networkx library
import networkx as nx

# Transform it in a links data frame (3 columns only):
links = iris.corr().stack().reset_index()
links.columns = ['var1', 'var2','value']

# correlation
threshold = -1

# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered=links.loc[ (links['value'] >= threshold ) & (links['var1'] != links['var2']) ]
 
# Build your graph
G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
 
# Plot the network
nx.draw_circular(G, with_labels=True, node_color='green', node_size=1000, edge_color='cyan', linewidths=3, font_size=12)


# <a id="4"></a> <br>
# ## Venn (Matplotlib)
# * You can use venn diagram to visualize the size of groups and their intersection.

# In[ ]:


# venn2
from matplotlib_venn import venn2
data_1 = len(iris.SepalLengthCm)
data_2 = len(iris.SepalWidthCm)
data_3 = len(iris[(iris.SepalLengthCm==iris.SepalWidthCm)]) # =0

# First way to call the 2 group Venn diagram
venn2(subsets = (data_1, data_2, data_3), set_labels = ('SepalLengthCm', 'SepalWidthCm'))
plt.show()


#  <a id="5"></a> <br>
# ## Donut (Matplotlib)
# * A donut chart is a pie chart with an area of the center cut out. 

# In[ ]:


# donut plot
feature_names = "sepal_length","sepal_width","petal_length","petal_width"
feature_size = [len(iris.SepalLengthCm),len(iris.SepalWidthCm),len(iris.PetalLengthCm),len(iris.PetalWidthCm)]
# create a circle for the center of plot
circle = plt.Circle((0,0),0.5,color = "white") #(0,0) coordinate
plt.pie(feature_size, labels = feature_names, colors = ["black","green","blue","cyan"] )
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("Number of Each Features")
plt.show()


#  <a id="6"></a> <br>
# ## Spider Chart (Matplotlib)
# * A spider(radar) plot  is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables.

# In[ ]:


# spider graph
categories = list(iris)[:4]
N = len(categories)
angles = [ n / float(N)*2*pi for n in range(N)]
angles = angles + angles[:1]
plt.figure(figsize = (10,10))
ax = plt.subplot(111,polar = True)
ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1],categories)
ax.set_rlabel_position(0)
plt.yticks([0,2,4,6],["0","2","4","6"],color= "red", size = 7)
plt.ylim(0,6)

values = iris.loc[0].drop("Species").values.flatten().tolist()
values = values + values[:1]
ax.plot(angles,values,linewidth = 1,linestyle="solid",label ="setosa" )
ax.fill(angles,values,"b",alpha=0.1)

values = iris.loc[1].drop("Species").values.flatten().tolist()
values = values + values[:1]
ax.plot(angles,values,linewidth = 1,linestyle="solid",label ="versicolor" )
ax.fill(angles,values,"orange",alpha=0.1)
plt.legend(loc = "upper left",bbox_to_anchor = (0.1,0.1))
plt.show()


# <a id="7"></a> <br>
# ## Inset Plots (Plotly)
# * If you do not understand the code check my plotly tutorial.

# In[ ]:


# trace1 is line plot
# go: graph object
trace1 = go.Scatter(
    x=iris.index,
    y=iris.SepalLengthCm,
    mode = "markers",
    xaxis='x2',
    yaxis='y2',
    name = "SepalLengthCm",
    marker = dict(color = 'rgba(76, 120, 213, 0.8)'),
)

# trace2 is histogram
trace2 = go.Histogram(
    x=iris.SepalLengthCm,
    opacity=0.75,
    name = "Sepal Length(Cm)",
    marker=dict(color='rgba(120, 5, 125, 0.6)'))

# add trace1 and trace2
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.7, 1],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = ' Sepal Length(Cm) Histogram and Scatter Plot'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="8"></a> <br>
# ## Basic 3D Scatter Plot (Plotly)
# * import data again to avoid confusion
# * go.Scatter3d: 3D scatter
# * We will plot iris setosa and iris virginica classes according to their Sepal Length(x), Sepal Width(y), and Petal Length(z).

# In[ ]:


# import data again
iris = pd.read_csv('../input/Iris.csv')
# data of iris setosa
iris_setosa = iris[iris.Species == "Iris-setosa"]
# data of iris virginica
iris_virginica = iris[iris.Species == "Iris-virginica"]
# data of iris virginica
iris_versicolor = iris[iris.Species == "Iris-versicolor"]

# trace1 =  iris setosa
trace1 = go.Scatter3d(
    x=iris_setosa.SepalLengthCm,
    y=iris_setosa.SepalWidthCm,
    z=iris_setosa.PetalLengthCm,
    mode='markers',
    name = "iris_setosa",
    marker=dict(
        color='rgb(217, 100, 100)',
        size=12,
        line=dict(
            color='rgb(0, 0, 0)',
            width=0.1
        )
    )
)
# trace2 =  iris virginica
trace2 = go.Scatter3d(
    x=iris_virginica.SepalLengthCm,
    y=iris_virginica.SepalWidthCm,
    z=iris_virginica.PetalLengthCm,
    mode='markers',
    name = "iris_virginica",
    marker=dict(
        color='rgb(54, 170, 127)',
        size=12,
        line=dict(
            color='rgb(0, 0, 0)',
            width=0.1
        )
    )
)
# trace3 =  iris versicolor
trace3 = go.Scatter3d(
    x=iris_versicolor.SepalLengthCm,
    y=iris_versicolor.SepalWidthCm,
    z=iris_versicolor.PetalLengthCm,
    mode='markers',
    name = "iris_setosa",
    marker=dict(
        color='rgb(100, 150, 145)',
        size=12,
        line=dict(
            color='rgb(0, 0, 0)',
            width=0.1
        )
    )
)
data = [trace1, trace2, trace3]
layout = go.Layout(
    title = ' 3D iris_setosa and iris_virginica',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="9"></a> <br>
# # Conclusion
# * If you like it, thank you for you upvotes.
# * If you have any question, I will happy to hear it
