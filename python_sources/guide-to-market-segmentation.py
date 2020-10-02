#!/usr/bin/env python
# coding: utf-8

# # Mall Customer Segmentation Data
# ###  Market Analysis
# ![](https://media.giphy.com/media/3o6Zt3LmfCJKgpXDb2/giphy.gif)
# ## Aim
# ### To perform clustering by dividing potential markets or consumers into specific groups based on similarities.
# 

# Import the required libraries

# In[ ]:


import numpy as np #linear algebra
import pandas as pd #data processing
import seaborn as sns #data visualization
import matplotlib.pyplot as plt


# Let us read the data from the csv file

# In[ ]:


df=pd.read_csv('../input/Mall_Customers.csv')


# |Performing Inferential Statisitcs on data

# In[ ]:


df.info()


# We will print top few rows to understand about the various data columns

# In[ ]:


df.head()


# Let us understand about the basic information of the data, like min, max, mean and standard deviation etc

# In[ ]:


df.describe()


# Check if there is any null value in the data

# In[ ]:


df.isnull().sum()


# Deleting CustomerID as it is not currently required.

# In[ ]:


del df['CustomerID']


# In[ ]:


print("Mean of Annual Income (k$) of Female:",df['Annual Income (k$)'].loc[df['Gender'] == 'Female'].mean())
print("Mean of Annual Income (k$) of Male:",df['Annual Income (k$)'].loc[df['Gender'] == 'Male'].mean())


# In[ ]:


print("Mean of Spending Score (1-100) of Female:",df['Spending Score (1-100)'].loc[df['Gender'] == 'Female'].mean())
print("Mean of Spending Score (1-100) of Male:",df['Spending Score (1-100)'].loc[df['Gender'] == 'Male'].mean())


# #### Mean Annual Income(Male) >Mean Annual Income(Female) but mean of spending score(Male) < mean of spending score(Female)
# #### Inference obtained is that females are slightly more inclined towards shopping

# Let us plot the Correlation between different columns

# In[ ]:


df.corr()
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[ ]:


df.query('Gender == "Male"').Gender.count()


# In[ ]:


df.query('Gender == "Female"').Gender.count()


# Now performe Data Visualization by using Matplotlib and Seaborn Libraries

# In[ ]:



labels = ['Male','Female']
sizes = [df.query('Gender == "Male"').Gender.count(),df.query('Gender == "Female"').Gender.count()]
#colors
colors = ['#ffdaB9','#66b3ff']
#explsion
explode = (0.05,0.05)
plt.figure(figsize=(8,8)) 
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85,explode=explode)
p=plt.gcf()
plt.axis('equal')
p.gca().add_artist(my_circle)
plt.show()


# In[ ]:


p1=sns.kdeplot(df['Annual Income (k$)'].loc[df['Gender'] == 'Male'],label='Income Male', shade=True, color="r")
p1=sns.kdeplot(df['Annual Income (k$)'].loc[df['Gender'] == 'Female'],label='Income Female', shade=True, color="b")
plt.xlabel('Annual Income (k$)')
plt.show()


# In[ ]:


df.sort_values(['Age'])
plt.figure(figsize=(10,8))
plt.bar( df['Age'],df['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:


sns.lmplot(x='Age', y='Spending Score (1-100)', data=df, fit_reg=True, hue='Gender')
plt.show()


# There is a very little correlation between Annual Income and Spending Score but still we will plot

# In[ ]:


sns.lmplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, fit_reg=True, hue='Gender')
plt.show()


# In[ ]:


p1=sns.kdeplot(df['Spending Score (1-100)'].loc[df['Gender'] == 'Male'],label='Density Male',bw=2, shade=True, color="r")
p1=sns.kdeplot(df['Spending Score (1-100)'].loc[df['Gender'] == 'Female'],label='Density Female',bw=2, shade=True, color="b")
plt.xlabel('Spending Score')
plt.show()


# ## KMeans Clustering
# Now Let us perform clustering on the given data
# We will use Kmeans Clustering which  is one of  the simplest unsupervised  learning  algorithms  that  solve  the well  known clustering problem.
# * The procedure follows a simple and  easy  way  to classify a given data set  through a certain number of  clusters (assume k clusters)
# * The  main  idea  is to define k centers, one for each cluster. These centers  should  be placed in a cunning  way  because of  different  location  causes different  result.
# *  So, the better  choice  is  to place them  as  much as possible  far away from each other
# * The  next  step is to take each point belonging  to a  given data set and associate it to the nearest center
# *  After we have these k new centroids, a new binding has to be done  between  the same data set points  and  the nearest new center.
# *  A loop has been generated. As a result of  this loop we  may  notice that the k centers change their location step by step until no more changes  are done.

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


Y = df[['Spending Score (1-100)']].values
X = df[['Annual Income (k$)']].values
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[ ]:


km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:,1:])

df["label"] = clusters

from mpl_toolkits.mplot3d import Axes3D
 

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='purple', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


# ### Thankyou for your watching my kernel
