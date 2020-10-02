#!/usr/bin/env python
# coding: utf-8

# # Data Visualization

# **Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.
# In this notebook I have used plotly and seaborn for plotting various features against target variable.**
# 
# 

# **This notebook will brief you about the following:** 
# 1. Histogram 
# 2. Barplot 
# 3. Piechart
# 4. Pairplot
# 5. Heatmap
# 6. Lineplot

# ****If you find this notebook helpful then consider upvoting it.****

# # Important useful modules
# 
# **Numpy** - NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
# 
# **Pandas** - Pandas is a software library written for the Python programming language for data manipulation and analysis.
# 
# **Matplotlib** - Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy
# 
# **Seaborn** - Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
# 
# **Plotly** - The plotly Python library (plotly.py) is an interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


train_data = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.shape


# Checking null values in the data

# In[ ]:


train_data.isnull().sum()


# Filling null values in the age columns with mean of the columns.
# 

# In[ ]:


mean = train_data['Age'].mean()
train_data['Age'] = train_data['Age'].fillna(mean)


# In[ ]:


train_data = train_data.drop(['Ticket'],axis = 1)


# In[ ]:


train_data = train_data.drop(['PassengerId'],axis = 1)


# # Heatmap
# A heat map is a two-dimensional representation of information with the help of colors. Heat maps can help the user visualize simple or complex information. Heat maps are used in many areas such as defense, marketing and understanding consumer behavior.It is helful in finding patterns and gives a perspective of depth.

# In[ ]:


sea.heatmap(train_data.corr(),annot = True)


# # Pair Plor

# Pair plot is used to understand the best set of features to explain a relationship between two variables or to form the most separated clusters. It also helps to form some simple classification models by drawing some simple lines or make linear separation in our dataset.

# In[ ]:


sea.pairplot(train_data)
sea.set(style="ticks", color_codes=True)


# # Countplot
# Countplot shows the counts of observations in each categorical bin using bars. A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable.

# In[ ]:


sea.countplot(train_data['Survived'])


# Passengers who survived are very less as compared to passengers who died

# In[ ]:


#plotting the graph sex vs survived
plot_1 = train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).sum().sort_values(by='Survived',ascending=False)
fig = px.bar(plot_1,x = plot_1['Sex'],y = plot_1['Survived'],color = 'Sex',text = 'Survived')
fig_1 = px.pie(plot_1,names = 'Sex',values = 'Survived',color = 'Sex')
fig_1.update_layout(width = 500,height = 500)
fig.update_layout(width = 500,height = 500)
fig_1.update_traces(pull = 0.05)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig_1.show()
fig.show()


# Most of the passengers who survived were women

# # Pie Chart
# Pie charts are generally used to show percentage or proportional data and usually the percentage represented by each category is provided next to the corresponding slice of pie. Pie charts are good for displaying data for around 6 categories or fewer.
# 
# # Barplot
# A bar chart uses bars to show comparisons between categories of data. These bars can be displayed horizontally or vertically. A bar graph will always have two axis. One axis will generally have numerical values, and the other will describe the types of categories being compared.

# In[ ]:


plot_2 = train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).sum().sort_values(by='Survived',ascending=False)
fig = px.bar(plot_2,x = 'Pclass',y = 'Survived',color = 'Pclass',text = 'Survived')
fig_1 = px.pie(plot_2,names = 'Pclass',values = 'Survived',color = 'Pclass')
fig.update_layout(width = 500,height = 500)
fig_1.update_layout(width = 500,height = 500)
fig_1.update_traces(pull = 0.05)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig_1.show()
fig.show()


# Survival rate in Pclass as follows:
# Pclass1 > Pclass3 > Pclass2

# # Histogram
# The purpose of a histogram (Chambers) is to graphically summarize the distribution of a univariate data set.Histogram is only used to plot the frequency of score occurrences in a continuous data set that has been divided into classes, called bins. Bar charts, on the other hand, can be used for a great deal of other types of variables including ordinal and nominal data set

# In[ ]:


g = sea.FacetGrid(train_data,col = 'Survived')
g.map(plt.hist,'Age', bins = 20)


# Most of the passengers who survived and died were of age between 20 and 40

# In[ ]:


g = sea.FacetGrid(train_data,col = 'Survived',row = 'Sex')
g.map(plt.hist,'Age', bins = 20)


# Most of the passengers who died were of age group 20-40 and were male

# In[ ]:


plot_4 = train_data[['Embarked','Survived']].groupby(['Embarked'],as_index=False).sum().sort_values(by='Survived',ascending=False)
fig = px.bar(plot_4,x = 'Embarked',y = 'Survived',color = 'Embarked',text = 'Survived')
fig_1 = px.pie(plot_4,names = 'Embarked',values = 'Survived',color = 'Embarked')
fig.update_layout(width = 500,height = 500)
fig_1.update_layout(width = 500,height = 500)
fig_1.update_traces(pull = 0.05)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig_1.show()
fig.show()


# From graph its clear that most of the passengers who survived is of embarked S

# # Lineplot
#  A line plot is a graph that shows frequency of data along a number line. It is best to use a line plot when comparing fewer than 25 numbers. It is a quick, simple way to organize data.

# In[ ]:


grid2 = sea.FacetGrid(train_data,row='Embarked')
grid2.map(sea.lineplot,'Pclass','Survived','Sex')


# Completing embarked column

# In[ ]:


freq_port = train_data.Embarked.dropna().mode()[0]
train_data['Embarked'] = train_data['Embarked'].fillna(freq_port)


# In[ ]:


grid = sea.FacetGrid(train_data, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sea.barplot, 'Sex', 'Fare',alpha = 1)


# From graph we can see that survival rate of passengers with low fare and embarked Q is least
# And survival rate of passengers with high fare and embarked C is very high

# In[ ]:


plot_4 = train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).sum().sort_values(by='Survived',ascending=False)
fig = px.line(plot_4,x = 'SibSp',y = 'Survived')
fig_1 = px.pie(plot_4,names = 'SibSp',values = 'Survived',color = 'SibSp')
fig.update_layout(width = 500,height = 500)
fig_1.update_layout(width = 500,height = 500)
fig_1.update_traces(pull = 0.05)
#fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig_1.show()
fig.show()

