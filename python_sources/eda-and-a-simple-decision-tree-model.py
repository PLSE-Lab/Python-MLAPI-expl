#!/usr/bin/env python
# coding: utf-8

# # Student Performance Report
# 
# This is an evaluation based on marks secured by the students in high school Students from the United States, whose aim is to analysis how 5 factors affect the students' performance.

# ## Data Preprocessing
# 
# First, we need to import dependencies and introduce data itself. We import numpy and pandas to manipulate data; sci-kit learn for simple machine learning and matplotlib for visualization.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scipy.stats import gaussian_kde

plt.style.use('ggplot')


data = pd.read_csv('../input/StudentsPerformance.csv')


# Then, we need to do data cleaning. It is a neccessary process, if we have missing values, inapproporate data type, duplicated entry, wrong calculation, abnormal or extreme vaule, any of those can affect our result. In our case, we only need to concern about missing values or duplicated entry based on the natrue of our data.

# In[ ]:


data.isnull().sum()


# In[ ]:


data.duplicated().sum()


# As seen above, there are no missing or duplicated values in our dataframe, which is perfect and ideal to move onto next stage. We would like to take a deeper glance into the dataset to gain more insights.

# In[ ]:


data.head()


# In[ ]:


data.shape


# We have 1000 entries in total representing 1000 students and there are eight elements for each individual, three of which are test scores and five of which are certain demographical features may or may not relate to their test scores.
# Let's look into details of these features.

# In[ ]:


info_dict = {}
for i,item in enumerate(data.columns):
    if i < 5:
        info_list = list(set(list(data[item])))
        info_list.sort()
        info_dict[item]=info_list
info_dict


# ## Exploratory Data Analysis
# 
# With clean dataset on hand, we are able to step further to explore more.
# 
# ### General proporties of scores distribution and visualization
# 
# We run a fast check of distribution proporties to be set as a base line in later evaluation.

# In[ ]:


data_summary = data.describe()
data_summary.loc['skewness'] = data.skew()
data_summary.loc['kurtosis'] = data.kurtosis()
data_summary


# We could conclude that these three distributions of scores are negative skewed, which means having right-leaning curve. Moreover, math score distribution has observable positive kurtosis, implying a fatter-tail compared to a normal distribution with that same mean and std.
# To visualize, histrograms and their respective kernel density estimation is plotted below.

# In[ ]:


xs = np.linspace(0,100,100)

fig = plt.figure(figsize=[8, 13.2])
fig.suptitle('Score Histograms and Kernel Density Estimations',fontsize=14,fontweight='bold')

ax1 = fig.add_subplot(311)
fig.subplots_adjust(top=0.945)
ax1.set_title('Math Score',fontsize=12)
ax1.set_ylabel('Number of Students')

density = gaussian_kde(data['math score'])
density.covariance_factor = lambda : .25
density._compute_covariance()
ax1.plot(xs,density(xs)*1000,color = '#4B4B4B')
ax1.hist(data['math score'],xs,color = '#FF3366')

ax2 = fig.add_subplot(312)
ax2.set_title('Reading Score',fontsize=12)
ax2.set_ylabel('Number of Students')

density = gaussian_kde(data['reading score'])
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs)*1000,color = '#4B4B4B')
ax2.hist(data['reading score'],xs,color = '#6666FF')

ax3 = fig.add_subplot(313)
ax3.set_title('Writing Score',fontsize=12)
ax3.set_ylabel('Number of Students')

density = gaussian_kde(data['writing score'])
density.covariance_factor = lambda : .25
density._compute_covariance()
ax3.plot(xs,density(xs)*1000,color = '#4B4B4B')
ax3.hist(data['writing score'],xs,color = '#FFFF33')

plt.show()


# ### Relevance bewteen features and scores
# 
# With five features and three scores on hand, it's natural to ask if there are connections among them. Such connection could be observed by a shift in the probability distribution. In this subsection, we will take a closer look at each feature.

# In[ ]:


df_math = pd.DataFrame()
for item in info_dict.keys():
    for features in info_dict[item]:
        df_math[features]=data.loc[data[item]==features].describe()['math score']
df_math.loc['skewness'] = df_math.skew()
df_math.loc['kurtosis'] = df_math.kurtosis()
df_math['Total'] = data_summary['math score']
df_math


# In[ ]:


fig = plt.figure(figsize=[8, 20])
fig.suptitle('Kernel Density Estimations of Math Score for each Feature',fontsize=14,fontweight='bold')
fig.subplots_adjust(top=0.95)

color_list = ['#0000FF','#FF0000','#00FFFF','#FF00FF','#FFFF00','#00FF00']

for i,item in enumerate(info_dict.keys()):
    ax = fig.add_subplot(6,1,i+1)
    ax.set_title(item,fontsize=12)
    ax.set_ylabel('Probability Density')
    for ii,features in enumerate(info_dict[item]): 
        density = gaussian_kde(data.loc[data[item]==features]['math score'])
        density.covariance_factor = lambda : .3
        density._compute_covariance()
        ax.plot(xs,density(xs),color = color_list[ii])
    ax.legend(labels = info_dict[item], loc = 'best')


# In[ ]:


df_reading = pd.DataFrame()
for item in info_dict.keys():
    for features in info_dict[item]:
        df_reading[features]=data.loc[data[item]==features].describe()['reading score']
df_reading.loc['skewness'] = df_reading.skew()
df_reading.loc['kurtosis'] = df_reading.kurtosis()
df_reading['Total'] = data_summary['reading score']
df_reading


# In[ ]:


fig = plt.figure(figsize=[8, 20])
fig.suptitle('Kernel Density Estimations of Reading Score for each Feature',fontsize=14,fontweight='bold')
fig.subplots_adjust(top=0.95)

color_list = ['#0000FF','#00FF00','#FF0000','#00FFFF','#FF00FF','#FFFF00']

for i,item in enumerate(info_dict.keys()):
    ax = fig.add_subplot(6,1,i+1)
    ax.set_title(item,fontsize=12)
    ax.set_ylabel('Probability Density')
    for ii,features in enumerate(info_dict[item]): 
        density = gaussian_kde(data.loc[data[item]==features]['reading score'])
        density.covariance_factor = lambda : .3
        density._compute_covariance()
        ax.plot(xs,density(xs),color = color_list[ii])
    ax.legend(labels = info_dict[item], loc = 'best')


# In[ ]:


df_writing = pd.DataFrame()
for item in info_dict.keys():
    for features in info_dict[item]:
        df_writing[features]=data.loc[data[item]==features].describe()['writing score']
df_writing.loc['skewness'] = df_writing.skew()
df_writing.loc['kurtosis'] = df_writing.kurtosis()
df_writing['Total'] = data_summary['writing score']
df_writing


# In[ ]:


fig = plt.figure(figsize=[8, 20])
fig.suptitle('Kernel Density Estimations of Writing Score for each Feature',fontsize=14,fontweight='bold')
fig.subplots_adjust(top=0.95)

color_list = ['#0000FF','#00FF00','#FF0000','#00FFFF','#FF00FF','#FFFF00']

for i,item in enumerate(info_dict.keys()):
    ax = fig.add_subplot(6,1,i+1)
    ax.set_title(item,fontsize=12)
    ax.set_ylabel('Probability Density')
    for ii,features in enumerate(info_dict[item]): 
        density = gaussian_kde(data.loc[data[item]==features]['writing score'])
        density.covariance_factor = lambda : .3
        density._compute_covariance()
        ax.plot(xs,density(xs),color = color_list[ii])
    ax.legend(labels = info_dict[item], loc = 'best')


# Though we can tell that there is certain difference among each groups in a certain feature, those figures still looks massy because of there are too many lines. The reason why we need so many of them is that human lives in a three dimensional world, it's easy for us to understand 2D image, still reasonable to recognize 3D object, but it become nonsense with 4D or higher. To deal with such high-dimensional infomation, rather than putting then into tons of graphs, another technique is introduced: t-Distributed Stochastic Neighbor Embedding (t-SNE).
# 
# Specifically, t-SNE models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.
# 
# In our case, we use gender for the first example. 

# In[ ]:


X = data[['math score','reading score','writing score']].values.tolist()
Y = data[['gender']].values.tolist()

X_embedded = TSNE(n_components=2).fit_transform(X)
x_min, x_max = np.min(X, 0), np.max(X, 0)
X_normalized = (X - x_min) / (x_max - x_min) 

plt.figure(figsize=[8,7])

blue_dot_x = []
blue_dot_y = []
red_dot_x = []
red_dot_y = []

for i in range(X_normalized.shape[0]):
    if Y[i] == ['male']:
        blue_dot_x.append(X_normalized[i,0])
        blue_dot_y.append(X_normalized[i,1])
    else:
        red_dot_x.append(X_normalized[i,0])
        red_dot_y.append(X_normalized[i,1])
plt.scatter(blue_dot_x, blue_dot_y, color = '#0000FF', label = 'male')
plt.scatter(red_dot_x, red_dot_y, color = '#FF0000',  label = 'female')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title('t_SNE plot with perplexity 30')
plt.legend()
plt.show()


# Compare to the three figures used before, the t_SNE plot is more friendly to read. It is obvious to see that we can tell that it is possible to distinguish from genders by barely look at grades even though there is an overlapping area. 
# 
# Given such a differnce, we can use a simple decision tree model to try making prediction whether it is a male or female student given his/her grades.
# 
# We first separate or dataset into training set and testing set, then use ski-kit learn to bulid and optimize our model, and eventually apply our model onto the test set to check accuracy. 

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                        test_size=0.1, random_state=42)

clf = tree.DecisionTreeClassifier(max_depth = 6)
clf = clf.fit(X_train,Y_train)

print('The accuracy on training set is %6.4f' % clf.score(X_train,Y_train))
print('The accuracy on testing set is %6.4f' %clf.score(X_test, Y_test))


# It's quite amazing to have such a high accuracy rate, what is better is that decision tree is easy to visualize. I'm gonna skip this visulization part since it requires two more libraries, pydot and GraphViz (software also required). 

# In[ ]:




