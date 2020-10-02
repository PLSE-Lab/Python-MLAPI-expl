#!/usr/bin/env python
# coding: utf-8

# ![](http://columbiareviewmag.com/wp-content/uploads/2015/04/1-1.jpg)

# > **Importing of Libraries** 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# > ### **Importing the data **

# In[ ]:


dataset = pd.read_csv('../input/Mall_Customers.csv')


# **Head method to quickly check the top 5 rows and columns of the dataset which is being loaded.**
# 
# 

# In[ ]:


dataset.head()


# > **Renaming the Columns:**
# 
# Lets rename the columns from Annual Income (k$) to Annual_Income and Spending Score (1-100) to Spending_Score. This will help in performingoperations on the columns in more efficient manner further in the analysis. We will use the rename method to do so. 

# In[ ]:


dataset.rename(columns={'Annual Income (k$)':'Annual_Income','Spending Score (1-100)':'Spending_Score'},inplace=True)


# > **Checking the columns and rows of the dataset using shape.**

# In[ ]:


dataset.shape


# Quickly we will go through the dataset which has been loaded to check the data containing inside it after renaming the columns.

# In[ ]:


dataset.head()


# > ### **Summary statistics of the data to find out the distribution of the data.**

# In[ ]:


dataset.describe()


# In[ ]:


dataset.info()


# > **Unique Value in each feature of the data**
# 
# Using nunique we will check how many unique values are present in the dataset. 

# In[ ]:


dataset.nunique()


# > ### **EDA for the Age Distribution,Annual Income and Spending Score**
# 
# We plot the Age,Annual Income and Spending Score using seaborn to understand distribution of both the columns. This will help us in understanding the distribution range.
# 
# 

# **Age Distribution using seaborn**

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(dataset['Age'])
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()


# **From the above plot of age distribution we can observe:**  
# 
#     * Youngster are in lesser number in visiting the mall. 
#     * Age group from 27-40 years of age is more frequent in visiting the mall. But clear pattern is still not found. 
#     * As the age increases the visitors starts decreasing.
#     * Surprisingly age group of 32 is more frequent whereas age group of 33 is one of the least in the chart.
#     * Ages group of 18,24,28,54,59 and 67 years have equal number of visitors in between them.
#     * Similarly age group of 55,56,64 and 69 have less number of visitors     
#     
#     
#     
# 

# #### **Annual Income Distribution using seaborn**
# 
# Let's now check for the Annual Income Distribution
# 
#     

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(dataset['Annual_Income'])
plt.title('Annual Income')
plt.xlabel('Annual Income($)')
plt.show()


# **Annual Income Observation:**
# 
# 
#     *  Salary range starts from $15K to $137K.
#     *  Customers who have annual income of $54k and $78k have more customers visiting the mall.
#     *  There are less customers who have less annual income from 15k-18k and so are in the range of 131k-137k
# 

# #### **Data Visualization of Spending Score Distribution**
# 
# Important for any mall is to analyze to spending score, so we will plot the spending score in order to understand how it is distributed.

# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(dataset['Spending_Score'])
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.axis()
plt.show()


# **From the above plot of spending score we find the below observation:**
# 
#     * Interestingly mall has customer group of sepending score as low as 1 to 99.
#     * Spending Score group of 35-60 has more number of customers 
#     

# #### **Distribution Plot for age and Annual Income**
# 
# 

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(dataset['Age'])
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.distplot(dataset['Annual_Income'],color='pink')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income')
plt.ylabel('Count')


# From the above chart its clear for the age distribution and annual income which we inferred in our bar chart earlier.

# > #### **Gender Distribution**
# 
# We will check now how the mall data is distributed in terms of gender. We will use the pie chart to understand the data distribution.

# In[ ]:


plt.figure(figsize=(8,8))

colors = ['LightBlue','Lightgreen']
explode = [0,0.1]
plt.pie(dataset['Gender'].value_counts(),explode=explode,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)
plt.legend(labels=['Female','Male'])
plt.title('Male v/s Female Distribution')
plt.axis('off')


# In[ ]:


dataset['Gender'].value_counts()


# **Gender Observation**
# 
# Clearly from the above charts, Female percentage which accounts to 56% is more than the male percentage of 44%. This infers that the female visiting the mall is more than male

# > #### **Heat map for Co related features**
# 
# We will generate heat map to understand the co relation between features. The color bar and the annot will help us in understanding the co relation between features.

# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(dataset.corr(),annot=True)
plt.show()


# From the above it is clear that these features don't have good co relation among them.

# #### **Plotting of Gender v/s Spending score:**
# 
# We will see how each gender has a spending score range. We will use both stripplot and box plot to understand the distribution.
# 

# In[ ]:


plt.figure(figsize=(15,5))
sns.stripplot(dataset['Gender'], dataset['Spending_Score'])
plt.title('Strip plot for Gender vs Spending Score')
plt.show()

plt.figure(figsize=(15,5))
sns.boxplot(dataset['Gender'], dataset['Spending_Score'])
plt.title('Box plot for Gender vs Spending Score')
plt.show()

plt.figure(figsize=(15,5))
sns.violinplot(dataset['Gender'],dataset['Spending_Score'])
plt.title('Gender Wise Spending Score')
plt.show()


# **From the above plot it is clear :**
# 
#     * Male Spending score ranges more from 25-70
#     * Female Spending score ranges from 35-75.
# 
# So **Female customers** are leading with the Spending Score here as well.
# 

# #### **Gender v/s Annual Income**
# 
# Lets see Gender and its Annual income distribution. We will use Violin and Box plot for inferring the result.

# In[ ]:


plt.figure(figsize=(15,5))
sns.violinplot(dataset['Gender'],dataset['Annual_Income'])
plt.title('Gender wise Annual Income Distribution')
plt.show()

plt.figure(figsize=(15,5))
sns.boxplot(dataset['Gender'],dataset['Annual_Income'])
plt.title('Gender wise Annual Income Distribution')
plt.show()


# **So above plot states:**
# 
#     * Male customers has more Annual_Income in comparison to Female customers.
#     * Female Customers and male customers has almost same number of low Annual_Income.    
# 

# > **Cluster Analysis**
# 
# We will now Segementation using **Age and Annual_Income**

# In[ ]:


x = dataset.loc[:,['Age' , 'Annual_Income']].values


# > #### **K Means Clustering :**
# 
# We will use KMeans Clustering. At first we will find the optimal clusters based on inertia and using elbow method. The distance between the centroids and the data points should be less.

# In[ ]:


from sklearn.cluster import KMeans

wcss =[]
for n in range(1,11):
    kmeans=KMeans(n_clusters=n,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.grid()
plt.show()


# So from the above plot it is clear that **cluster with 4** is the optimal.
# 
# 

# In[ ]:


#Applying K Means 

kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0,algorithm='elkan')
y_means = kmeans.fit_predict(x)


# In[ ]:


#Visualizing
plt.figure(figsize=(15,10))
plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='yellow',label='cluster2')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='green',label='cluster3')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='blue',label='cluster4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='centroids')
plt.title('Age wise Annual Income Cluster')
plt.xlabel('Age')
plt.ylabel('Annual Income($)')
plt.legend()
plt.show()


# So we have 4 clusters based on Age and Annual Income and the mall can have the best strategy to target consumers based on it.

# In[ ]:


x1 = dataset.loc[:,['Age','Spending_Score']].values


# In[ ]:


from sklearn.cluster import KMeans

wcss =[]
for n in range(1,11):
    kmeans=KMeans(n_clusters=n,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x1)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# > **K Means Clustering for Age V/S Spending Score**
# 
# In earlier plot we saw the Age v/s Annual Income, now we will check for the Age V/S Spending Score. This provides an insight of age and its spending score for mall customers. 
# 
# We will use the same strategy to find the optimal clusters that is through elbow method and then applying the optimal clusters to Kmeans algorithm as we performed in the earlier section. 

# In[ ]:


#Applying K Means 

kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means = kmeans.fit_predict(x1)


# In[ ]:


#Visualizing
plt.figure(figsize=(10,10))
plt.scatter(x1[y_means==0,0],x1[y_means==0,1],s=100,c='red',label='cluster1')
plt.scatter(x1[y_means==1,0],x1[y_means==1,1],s=100,c='yellow',label='cluster2')
plt.scatter(x1[y_means==2,0],x1[y_means==2,1],s=100,c='green',label='cluster3')
plt.scatter(x1[y_means==3,0],x1[y_means==3,1],s=100,c='blue',label='cluster4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='centroids')
plt.title('Age wise Spending Score(0-100)')
plt.xlabel('Age')
plt.ylabel('Spending Score(0-100)')
plt.legend()


# So looking through the above clustering plot mall can target customers based on the age group to 4 clusters for spending score. 
# 

# > #### Clustering based on Annual Income v/s Spending Score 
# 
# So we have clusters based on annual income and spending score based on Age. So now we will proceed with the most important cluster that is Annual Income v/s Spending Score. 
# 
# We will use the same strategy to get the optimum 

# In[ ]:


#Feature Selection for Annual Income and Spending Score
x2 = dataset.loc[:,['Annual_Income','Spending_Score']].values


# In[ ]:


from sklearn.cluster import KMeans

wcss =[]
for n in range(1,11):
    kmeans=KMeans(n_clusters=n,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x2)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()


# In[ ]:


# If we zoom the plot above , optimal clusters comes to 5, so we will take it n_clusters as 5 here.
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(x2)


# In[ ]:


#Visualizing all the clusters 

plt.figure(figsize=(15,8))
plt.scatter(x2[y_kmeans == 0, 0], x2[y_kmeans == 0, 1], s = 100, c = 'Orange', label = 'Pinch Penny')
plt.scatter(x2[y_kmeans == 1, 0], x2[y_kmeans == 1, 1], s = 100, c = 'Pink', label = 'Normal Customer')
plt.scatter(x2[y_kmeans == 2, 0], x2[y_kmeans == 2, 1], s = 100, c = 'yellow', label = 'Target Customer')
plt.scatter(x2[y_kmeans == 3, 0], x2[y_kmeans == 3, 1], s = 100, c = 'magenta', label = 'Spender')
plt.scatter(x2[y_kmeans == 4, 0], x2[y_kmeans == 4, 1], s = 100, c = 'Red', label = 'Balanced Customer')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'Black', label = 'Centroids')
plt.title('Annual Income v/s Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# > **Conclusion**
# 
# So we have 5 clusters divided based on Annual Income and Spending Score.
# 
# **Pinch Penny**     -- Earn More Spend Less
# 
# **Normal Customer** -- An Average consumer in terms of spending and Annual Income
# 
# **Target Customer**--  Annual Income High as well as Spending Score is high, so a target consumer.
# 
# **Spender** --         Annual Income is less but spending high, so can also be treated as **potential** target customer.
# 
# **Balanced**--         Earns less as well as spends less
# 
# 
# 
# The dataset does not contain the detailed of product information which would have helped in deeper analysis for the products being targeted by age,the annual income and its spending score and their preference.
# 
# 
#  

# > #                     Thank you and please Upvote if you like my Kernel
# 

# In[ ]:




