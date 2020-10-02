#!/usr/bin/env python
# coding: utf-8

# **based on work of https://www.kaggle.com/suneelpatel**

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# for path
import os
print(os.listdir("../input"))


# ### Import necessary libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[ ]:


# For interactive visualizations
import seaborn as sns
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff


# ### Load Dataset

# In[ ]:


# importing the dataset
data = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data.shape


# ### Review Dataset

# In[ ]:


# Let's see top 5 dataset
data.head()


# In[ ]:


# Let's see last 5 datasets
data.tail()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# ### Data Visualization

# In[ ]:


plt.subplot(1, 2, 1)
sns.distplot(data['Annual Income (k$)'],color='orange')
plt.title('Distribution of Annual Income ', fontsize = 15)
plt.xlabel('Annual Income(in k$)')
plt.ylabel('Prob.')


plt.subplot(1, 2, 2)
sns.distplot(data['Age'])
plt.title('Distribution of Age', fontsize = 15)
plt.xlabel('Range of Age')
plt.ylabel('prob.')

plt.show()


# Here, In the above Plots we can see the Distribution pattern of Annual Income and Age, By looking at the plots,
# 
# we can infer one thing that There are few people who earn more than 100k US Dollars. Most of the people have an earning of around 50k-75k US Dollars. Also, we can say that the least Income is around 20k US Dollars.
# 
# Taking inferences about the Customers.
# 
# The most regular customers for the Mall has age around 30-35 years of age. Whereas the the senior citizens age group is the least frequent visitor in the Mall. Youngsters are lesser in umber as compared to the Middle aged people.

# In[ ]:


labels = ['Female', 'Male']
size = data['Gender'].value_counts()
colors = ['blue', 'orange']

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, labels = labels,autopct = '%.2f%%')
plt.title('Gender', fontsize = 15)
plt.legend(fontsize=12)
plt.show()


# By looking at the above pie chart which explains about the distribution of Gender in the Mall
# 
# Interestingly, The Females are in the lead with a share of 56% whereas the Males have a share of 44%, that's a huge gap specially when the population of Males is comparatively higher than Females.

# In[ ]:


plt.rcParams['figure.figsize'] = (14, 8)
sns.countplot(data['Age'])
plt.title('Distribution of Age', fontsize = 15)
plt.show()


# This Graph shows a more Interactive Chart about the distribution of each Age Group in the Mall for more clariy about the Visitor's Age Group in the Mall.
# 
# By looking at the above graph-, It can be seen that the Ages from 27 to 39 are very much frequent but there is no clear pattern, we can only find some group wise patterns such as the the older age groups are lesser frequent in comparison. Interesting Fact, There are equal no. of Visitors in the Mall for the Agee 18 and 67. People of Age 55, 56, 69, 64 are very less frequent in the Malls. People at Age 32 are the Most Frequent Visitors in the Mall.

# In[ ]:


plt.rcParams['figure.figsize'] = (14, 8)
sns.countplot(data['Annual Income (k$)'], palette = 'rainbow')
plt.title('Distribution of Annual Income', fontsize = 15)
plt.show()


# Again, This is also a chart to better explain the Distribution of Each Income level, Interesting there are customers in the mall with a very much comparable freqyuency with their Annual Income ranging from 15 US Dollars to 137K US Dollars. There are more Customers in the Mall whoc have their Annual Income as 54k US Dollars or 78 US Dollars.

# In[ ]:


plt.subplot(1,2,1)
sns.countplot(data['Spending Score (1-100)'])
plt.title('Distribution Grpah of Spending Score', fontsize = 15)

plt.subplot(1,2,2)
sns.distplot(data['Spending Score (1-100)'])


# This is the Most Important Chart in the perspective of Mall, as It is very Important to have some intuition and idea about the Spending Score of the Customers Visiting the Mall.
# 
# On a general level, we may conclude that most of the Customers have their Spending Score in the range of 35-60. Interesting there are customers having I spending score also, and 99 Spending score also, Which shows that the mall caters to the variety of Customers with Varying needs and requirements available in the Mall.

# In[ ]:


sns.pairplot(data)
plt.title('Pairplot for the Data', fontsize = 15)
plt.show()


# In[ ]:


data.corr()


# In[ ]:


## Correlation coeffecients heatmap
sns.heatmap(data.corr(), annot=True).set_title('Correlation Factors Heat Map', size='15')


# The Above Graph for Showing the correlation between the different attributes of the Mall Customer Segementation Dataset, This Heat map reflects the most correlated features with Orange Color and least correlated features with yellow color.
# 
# We can clearly see that these attributes do not have good correlation among them, that's why we will proceed with all of the features.

# In[ ]:


plt.rcParams['figure.figsize'] = (16, 8)
sns.stripplot(data['Gender'], data['Age'], size = 10)
plt.title('Gender vs Spending Score', fontsize = 15)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (16,8)
sns.violinplot(data['Gender'], data['Annual Income (k$)'])
plt.title('Gender vs Spending Score', fontsize = 15)
plt.show()


# A Bivariate Analysis between the Gender and the Annual Income, to better visualize the Income of the different Genders.
# 
# There are more number of males who get paid more than females. But, The number of males and females are equal in number when it comes to low annual income.

# ## Clustering Analysis

# In[ ]:


data.head()


# In[ ]:


x = data[['Annual Income (k$)','Spending Score (1-100)']]

# let's check the shape of x
print(x.shape)


# In[ ]:


x


# ### Kmeans Algorithm
# 
# **The Elbow Method to find the No. of Optimal Clusters**

# In[ ]:


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km= KMeans(i)
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# **Let's Visualizaing the Clusters**

# In[ ]:


km2=KMeans(5)
km2.fit(x)
data_clustered=data.copy()
data_clustered['Cluster']=km2.fit_predict(x)
data_clustered.head()


# In[ ]:


plt.rcParams['figure.figsize'] = (10,5)
plt.scatter(data_clustered['Annual Income (k$)'],data_clustered['Spending Score (1-100)'],c=data_clustered['Cluster'],cmap='rainbow')
plt.title('K Means Clustering', fontsize = 15)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# This Clustering Analysis gives us a very clear insight about the different segments of the customers in the Mall. There are clearly Five segments of Customers namely Miser, General, Target, Spendthrift, Careful based on their Annual Income and Spending Score which are reportedly the best factors/attributes to determine the segments of a customer in a Mall.

# **Clusters of Customers Based on their Ages**

# In[ ]:


x = data.iloc[:, [2, 4]].values
x.shape


# ### K-means Algorithm

# In[ ]:


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.rcParams['figure.figsize'] = (16, 8)
plt.plot(wcss)
plt.title('K-Means Clustering(The Elbow Method)', fontsize = 20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid()
plt.show()


# In[ ]:


kmeans = KMeans(4)
ymeans = kmeans.fit_predict(x)

plt.rcParams['figure.figsize'] = (16, 8)
plt.title('Cluster of Ages', fontsize = 30)

plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'green', label = 'Usual Customers' )
plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')
plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'cyan', label = 'Target Customers(Young)')
plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'blue')

plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()


# According to my own intuition by looking at the above clustering plot between the age of the customers and their corresponding spending scores, I have aggregated them into 4 different categories namely Usual Customers, Priority Customers, Senior Citizen Target Customers, Young Target Customers. Then after getting the results we can accordingly make different marketing strategies and policies to optimize the spending scores of the customer in the Mall.

# In[ ]:




