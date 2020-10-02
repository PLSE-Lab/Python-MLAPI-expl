#!/usr/bin/env python
# coding: utf-8

# __Content__
# 
# You are owing a supermarket mall and through membership cards , you have some basic data about your customers like Customer ID, age, gender, annual income and spending score. Spending Score is something you assign to the customer based on your defined parameters like customer behavior and purchasing data.
# 
# Problem Statement You own the mall and want to understand the customers like who can be easily converge [Target Customers] so that the sense can be given to marketing team and plan the strategy accordingly.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data = pd.read_csv('../input/Mall_Customers.csv')


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.shape


# ### __Visualizing some features__

# In[ ]:


#Count of males & females
plt.title('Count of Males & Females')
sns.set(style="darkgrid")
sns.countplot(x = 'Gender', data = data)


# We can see that the male customers registered for membership card is about 90, while female customers are about 110. The distribution of gender for membership card is slightly unbalanced.

# In[ ]:


#Exploring Customer Spending Section
plt.title('Customer Spending Score distribution')
sns.distplot(data['Spending Score (1-100)'], color="g", kde = False)


# Most of the customer spending score is around 40, and the distribution of spending score is roughly symmetric.

# In[ ]:


#Exploring Customer Income Section
plt.title('Customer Income distribution')
sns.distplot(data['Annual Income (k$)'], color="b", kde = True)


# As we can see from the income section: the customer income is not normally distributed - The graph is skewed to the left. We can try to apply transformation to the income below: 

# In[ ]:


plt.title('Customer logged Income distribution')
sns.distplot(np.log(data['Annual Income (k$)']), color="b", kde = True)


# Tring out one more transformation technique

# In[ ]:


sqrt_tran = data.apply(lambda x: x['Annual Income (k$)'] ** 0.5, axis=1)
plt.title('Customer Squared Root Transformed Income distribution')
sns.distplot(sqrt_tran, color="b", kde = True)


# It seems that square root transformation do the best job normalizing the customer annual income column
# 
# ### __2d relationship between features__

# In[ ]:


#Customer age distribution in terms of gender
plt.title('Customer age distribution in terms of gender')
ax = sns.violinplot(x="Gender", y="Age",
                   data=data, palette="Blues", split=True)


# From the violin plot we can see that most of the male customers are around 33~34 while most of the female customers are around 31~33. 

# In[ ]:


#Customer Income distribution in terms of gender
plt.title('Customer income distribution in terms of gender')
ax = sns.violinplot(x="Gender", y="Annual Income (k$)",
                   data=data, palette="GnBu_d", split=True)


# Most male customers have income around 75k which is about the same with female customers.

# In[ ]:


#Customer Spending Score distribution in terms of gender
plt.title('Customer spending distribution in terms of gender')
ax = sns.violinplot(x="Gender", y="Spending Score (1-100)",
                   data=data, palette="coolwarm", split=True)


# The overall spending interquartile range 25% to 75% is slightly higher for female customers
# 
# while the mean spending of male is slightly higher than female

# In[ ]:


sns.jointplot("Annual Income (k$)", "Spending Score (1-100)", data=data, color="m")


# This plot is particularly interesting, it seems that it is composed of 5 different segment alreay:
# - Annual Income between 20 to 40 k upper part, spending score decreasing with the increase of income
# - Annual Income between 20 to 40 k lower part, spending score increase with the increase of income
# - Annual income between 40 to 60 k in the middle, the spending score scattered randomly in between 40 to 60
# - Annual Income between 70 to 140 k upper part, spending score decrease with the increase of income
# - Annual Income between 70 to 140 k lower part, spending score increase with the increase of income
# 
# This might enables us to select k = 5 as 5 clusters (each with a centroid) in latter part of the analysis

# In[ ]:


sns.jointplot("Age", "Spending Score (1-100)", data=data, color="b")


# In[ ]:


sns.jointplot("Age", "Annual Income (k$)", data=data, color="r")


# In[ ]:


#Regression plot from Kushal
#https://www.kaggle.com/kushal1996/customer-segmentation-k-means-analysis

plt.figure(1 , figsize = (15 , 7))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = data)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()


# From the data in this particular dataset, we can see neither of these 3 factors: annual income, age and spending score seems to correlate each other much. We can further demonstrate this correlation with heatmap. 

# ### __Heat Map__

# In[ ]:


#Dummy encoding the gender variable
one_hot = pd.get_dummies(data['Gender'])
data_2 = data.join(one_hot)
data_2 = data_2.drop(['Gender', 'Male'], axis = 1)
data_2.head()


# In[ ]:


data_without_id = data_2.drop('CustomerID', axis = 1)
data_without_id.rename(columns={'Female':'Is_Female', 'Spending Score (1-100)': 'Spending_Score',
                               'Annual Income (k$)':'Annual_Income'}, inplace=True)
colormap = plt.cm.inferno
plt.figure(figsize=(8,6))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data_without_id.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# Indeed, the highesr correlation in the heatmap is less tha 0.07. So we can say that these features are independent of each other

# ### __K-Means Clustering__

# - Although we have visualize the number of ks that we are going to use from previous graphs, it would be wise to use the ELBOW method to give us the k that has the smallest __sum of squared errors (SSE)__.
# 
# 
# - I think we do not need PCA before doing clustering, since we have relatively lower dimensions for this dataset. In addition, they are relatively independent of each other.
# 
# 
# - Lastly, it is a good idea to standardize the data before we do k-mean clustering. For this particular dataset, the range of each column of data is roughly set in between 1 to 100. So I will skip the standardization part. You could use standardscaler in sklearn lib to do it.

# In[ ]:


from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

# k means determine k
def PlotElbow(X):
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.figure(figsize = (10, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# In[ ]:


#We started off by clustering age vs spending score
data_age = data_without_id[['Age' , 'Spending_Score']].iloc[: , :].values

PlotElbow(data_age)


# We will choose the point where our improvement stops declining rapidly creating this elbow shaped graph as our optimal value for k, in this case, k = 4 would be a good choice

# In[ ]:


#The follow plot is from Kushal: referring to
#https://www.kaggle.com/kushal1996/customer-segmentation-k-means-analysis
#Fit model and plot the graph
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(data_age)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_


# In[ ]:


h = 0.02
x_min, x_max = data_age[:, 0].min() - 1, data_age[:, 0].max() + 1
y_min, y_max = data_age[:, 1].min() - 1, data_age[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 


# In[ ]:


plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = data , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
plt.show()


# In[ ]:


#Then we do spending score vs annual income
data_income = data_without_id[['Annual_Income' , 'Spending_Score']].iloc[: , :].values

PlotElbow(data_income)


# In[ ]:


#As we expected, k = 5 is the right cluster where the elbow forms
#Fit model and plot the graph
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(data_income)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

h = 0.02
x_min, x_max = data_income[:, 0].min() - 1, data_income[:, 0].max() + 1
y_min, y_max = data_income[:, 1].min() - 1, data_income[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = data , c = labels2 , 
            s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Anuual Income')
plt.show()


# In[ ]:


data_all = data_without_id[['Is_Female','Age', 'Annual_Income' , 'Spending_Score']].iloc[: , :].values

PlotElbow(data_all)


# In[ ]:


algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(data_all)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_


# In[ ]:


import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected = True)


# In[ ]:


data['label3'] =  labels3
trace1 = go.Scatter3d(
    x= data['Age'],
    y= data['Spending Score (1-100)'],
    z= data['Annual Income (k$)'],
    mode='markers',
     marker=dict(
        color = data['label3'], 
        size= 20,
        line=dict(
            color= data['label3'],
            width= 12
        ),
        opacity=0.8
     )
)
new_data = [trace1]
layout = go.Layout(
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)
fig = go.Figure(data=new_data, layout=layout)
py.offline.iplot(fig)


# In[ ]:





# In[ ]:





# In[ ]:




