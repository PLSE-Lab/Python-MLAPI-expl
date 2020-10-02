#!/usr/bin/env python
# coding: utf-8

# Market Basket Analysis is a modelling technique based upon the theory that if you buy a certain group of items, you are more (or less) likely to buy another group of items.
# In this case we use the Mall Customer Segmentation dataset that has parameters such as Age, Gender, Income and Spending Score.
# The Spending score is based on parameters like customer behavior and purchasing data.

# In[ ]:


#load the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.color_palette("Paired")

import warnings
warnings.filterwarnings("ignore")


# 1) Loading the Dataset and renaming the columns for the ease of use.

# In[ ]:


data = pd.read_csv('../input/Mall_Customers.csv')
data.head()


# In[ ]:


data = data.rename(index=str,columns = {"CustomerID":"id","Annual Income (k$)":"income","Spending Score (1-100)":"score"})


# In[ ]:


print(data.columns)
print(type(data))


# Simple Analysis of the Dataset to understand the parameters and what each is trying to convey.

# In[ ]:


print(data.describe())
print(data.isnull().sum())


# Looking athe Description we can tell that 200 people between the Age of 18 and 70 are considered in this test with an Average age of 38.85~ 39 Years is taken. Similarly most of the People have their income around  60K dollar  with the Heighest being 137K dollar and lowest being $ 15K dollar. On average people have a credit score of 25.82 with the heighest rewarded for the best customer,ie:99 and lowest being 1 for the worst.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Gender'] = le.fit_transform(data.Gender)
print(le.classes_)
#Male is encoded as 1
#Female is encoded as 0.
print(data.head(5))


# Here we calculate the percentage of Male and Female participants who were selected for the Analysis.

# In[ ]:


m,f=data['Gender'].value_counts()[1],data['Gender'].value_counts()[0]
gen = [m,f]
lab = ['Male','Female']
plt.pie(gen,labels=lab, shadow=True,autopct='%1.0f%%', startangle=140)


# Here we generate the PDFs and the CDFs for Age, Income and Score. We that most of the Participants were in their 30s. They had an income of around 75K Dollars and mostly a score between 40 and 60.

# In[ ]:



fig = plt.figure()
fig.suptitle('PDF',fontsize=20)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(2,len(data.columns)):
    ax = fig.add_subplot(1, 3, i-1)
    fig.set_figheight(5)
    fig.set_figwidth(20)
    sns.distplot(data[str(data.columns[i])],hist=False,kde_kws={"shade": True},color='orange')
    
    ax.title.set_text(data.columns[i])
    

plt.show()
fig = plt.figure()
fig.suptitle('CDF',fontsize=20)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(2,len(data.columns)):
    ax = fig.add_subplot(1, 3, i-1)
    fig.set_figheight(5)
    fig.set_figwidth(20)
    sns.distplot(data[str(data.columns[i])],hist=False,kde_kws={"shade": True,"cumulative":True},color='g')
    ax.title.set_text(data.columns[i])
    

plt.show()


# Joint Plot

# In[ ]:


fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(2,len(data.columns)-1):
    sns.jointplot(str(data.columns[i]),'score',data, kind="kde",space=0, color="#4CB391")


# On Violin Plot we observe that most of the women were of the age 30 whereas the age group is more uniformally distribute in case of Men.
# It can be seen that the highest income for men was slightly higher than that of Female however the Spending score for minimum spendind score for men was lesser than that of women. At the same time the spendng score for women is much better.

# In[ ]:


fig = plt.figure()
fig.suptitle('Violin Plots',fontsize=20)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
cols = ['Age', 'income', 'score']
for i in range(len(cols)):
    ax = fig.add_subplot(1, 3, i+1)
    fig.set_figheight(5)
    fig.set_figwidth(20)
    sns.violinplot(x = 'Gender' , y = cols[i], data = data )
    
    


# In[ ]:


fig = plt.figure()
fig.suptitle('Violin Plots',fontsize=20)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
cols = ['Age', 'income', 'score']
for i in range(len(cols)):
    ax = fig.add_subplot(1, 3, i+1)
    fig.set_figheight(5)
    fig.set_figwidth(20)
    sns.violinplot(x = 'Gender' , y = cols[i], data = data )
    


# Here we generate all the possible scatter plots for all parameters.

# In[ ]:


import itertools
col_list = ['Age','score','income','Gender']
combinations =list(itertools.permutations(col_list, 3))
print(combinations)
fig = plt.figure()
fig.suptitle('Scatter Plots',fontsize=20)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(0,len(combinations)):
    try:
        ax = fig.add_subplot(4, 4, i)
        fig.set_figheight(20)
        fig.set_figwidth(20)
        sns.scatterplot(x=str(combinations[i][0]),y=str(combinations[i][1]),hue=str(combinations[i][2]),data=data)
    except:
        pass


# Elbow curve generation for creating a 3d clustering graph taking age,income and score into consideration.

# In[ ]:


print(data.columns)
attr,label = data.iloc[:,1:-1],data.iloc[:,-1]
sse = {}
from sklearn.cluster import KMeans
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(attr)
    
    sse[k] = kmeans.inertia_ 
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=4).fit(attr)
label = kmeans.labels_
centers = kmeans.cluster_centers_


# In[ ]:


import plotly as py
import plotly.graph_objs as go
import plotly.offline as py_of
py_of.init_notebook_mode(connected=True)
data['labels3'] =  label
trace1 = go.Scatter3d(
    x= data['Age'],
    y= data['score'],
    z= data['income'],
    mode='markers',
     marker=dict(
        color = data['labels3'], 
        size= 20,
        line=dict(
            color= data['labels3'],
            width= 12
        ),
    
        
     )
)
data_2 = [trace1]
layout = go.Layout(
    
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)
layout.autosize= True
fig = go.Figure(data=data_2, layout=layout)
py_of.iplot(fig)


# In[ ]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(attr,label)


# Clustering based on Age and Spending Score

# In[ ]:


data_temp = data.iloc[:,[2,4]]
sse = {}
from sklearn.cluster import KMeans
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data_temp)
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=4,random_state=2).fit(data_temp)
label = kmeans.labels_
centers = kmeans.cluster_centers_
print(data_temp.head())


# The following can be concluded from the clustering.
# 4 clusters are generated:
# 
# Cluster number 0 suggesting the group of people who are 'Young and have a normal Spending Score'.
# 
# Cluster 1 are group of people who have a high Spending score and are young as well. These can be depicted as the target customers.
# 
# Cluster 2 are group of People who are old and have an Average Spending score.
# 
# Cluster 3 are the group of people who have a very low score. 

# In[ ]:


sns.scatterplot(x='score',y='Age',hue = label,data = data_temp,palette="Set2")


# Clustering on the basis of Income and Score

# In[ ]:


data_temp = data.iloc[:,[3,4]]
sse = {}
from sklearn.cluster import KMeans
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data_temp)
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=1000,random_state=2).fit(data_temp)
label = kmeans.labels_
centers = kmeans.cluster_centers_


# Here 5 clusters are generated 
# 
# Cluster 0 are the group of people who have an average Income and an Average Score.
# 
# Cluster 1 are people who have really high Income but don't really care about incomes.
# 
# Cluster 2 are people who have low income and low scores.
# 
# Cluster 3 are are people who have a really high income and a really High Score.
# 
# Cluster 4 shows the group of people who have low incomes but tend to have really Hight scores.
# 
# 
# 

# In[ ]:


sns.scatterplot(x='score',y='income',data=data_temp,hue=label)


# I hope you liked my work. Please suggest what more I can do and do upvote if you like it. Thank You!

# In[ ]:




