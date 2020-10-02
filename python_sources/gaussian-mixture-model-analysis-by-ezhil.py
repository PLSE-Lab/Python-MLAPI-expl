#!/usr/bin/env python
# coding: utf-8

# ### Created By : Ezhilarasan Kannaiyan
# To analyse the Mall Customer Segmentation Dataset and perform clustering using **Gaussian Mixture Model**

# <u>Import appropriate python libraries</u>

# In[ ]:


get_ipython().run_line_magic('reset', '-f')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import re  #regular expression
from sklearn.preprocessing import StandardScaler
from pandas.plotting import andrews_curves
from mpl_toolkits.mplot3d import Axes3D


# <u>Settings for Display & Running Mode</u>

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# <u>Read the datset</u>

# In[ ]:


customer = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
customer.shape
customer.head()


# <u>Remove Special characters in Column names </u>

# In[ ]:


new_columns = {x : re.sub('[^A-Za-z]+','',x) for x in customer.columns.values}
new_columns
customer.rename(columns = new_columns,inplace=True)
customer.rename(columns = {"AnnualIncomek": "AnnualIncome"},inplace=True)


# <u>Create Gender column as numerical categorical type</u> 

# In[ ]:


customer["Gender"].value_counts()
customer["GenderCode"] = customer["Gender"].map({"Female" : 0, "Male" : 1})


# Drop CustomerID and old Gender column

# In[ ]:


customer.drop(columns=["CustomerID","Gender"], inplace=True)


# In[ ]:


customer.head()


# Verify whether any column has null value

# In[ ]:


customer.info()


# Analyse statistical data

# In[ ]:


customer.describe()


# Annual income starts from 15K \\$ to max of 137K \\$. <br/>
# Spending score starts from 1 to max upto 99 <br>
# Mean and Median are almost same for both Annual Income and Spending Score

# <u>Count plot to see the Gender count</u>

# In[ ]:


values = customer["GenderCode"].value_counts()
ax = sns.countplot(customer["GenderCode"])
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1, values[i],ha="center")


# <u>Andrew Curve :</u> Gender Code 0 and 1 both mixed up in the dataset

# In[ ]:


andrews_curves(customer, "GenderCode")


# <u> Box plot of all columns with respect to Gender</u>

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.boxplot(data=customer, x="GenderCode",y="Age")
ax=plt.subplot(1,3,2)
sns.boxplot(data=customer, x="GenderCode",y="AnnualIncome")
ax=plt.subplot(1,3,3)
sns.boxplot(data=customer, x="GenderCode",y="SpendingScore")


# Based on the box plots, Good Annual income and Spending scores in both Male & Female 

# <b>Some other visualization plots to analyse</b>

# <u>Strip Plot</u>

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.stripplot(data=customer, x="GenderCode",y="Age")
ax=plt.subplot(1,3,2)
sns.stripplot(data=customer, x="GenderCode",y="AnnualIncome")
ax=plt.subplot(1,3,3)
sns.stripplot(data=customer, x="GenderCode",y="SpendingScore")


# <u>Swarm Plot</u>

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.swarmplot(data=customer, x="GenderCode",y="Age")
ax=plt.subplot(1,3,2)
sns.swarmplot(data=customer, x="GenderCode",y="AnnualIncome")
ax=plt.subplot(1,3,3)
sns.swarmplot(data=customer, x="GenderCode",y="SpendingScore")


# <u> Distribution Plots </u>

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.distplot(customer.Age, rug=True)
ax=plt.subplot(1,3,2)
sns.distplot(customer.AnnualIncome, rug=True)
ax=plt.subplot(1,3,3)
sns.distplot(customer.SpendingScore, rug=True)


# <u> Pair Plots </u>

# In[ ]:


sns.pairplot(customer, vars=["Age","AnnualIncome","SpendingScore"], diag_kind="kde"
             , kind="reg", hue="GenderCode", markers=["o","D"],palette="husl")


# <u> Scatter Plots </u>

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.scatterplot(data=customer, x="Age",y="AnnualIncome")
ax=plt.subplot(1,3,2)
sns.scatterplot(data=customer, x="Age",y="SpendingScore")
ax=plt.subplot(1,3,3)
sns.scatterplot(data=customer, x="AnnualIncome",y="SpendingScore")


# <u> Regression Plots </u>

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.regplot(data=customer, x="Age",y="AnnualIncome")
ax=plt.subplot(1,3,2)
sns.regplot(data=customer, x="Age",y="SpendingScore")
ax=plt.subplot(1,3,3)
sns.regplot(data=customer, x="AnnualIncome",y="SpendingScore")


# From all the above plots,we could see the data is distributed well in the given data set.<br/>

# <u>Animation Graph </u>

# In[ ]:


px.scatter(customer.sort_values(by="Age"),
          x = "AnnualIncome",
          y = "SpendingScore",
          #size = "GenderCode",
          range_x=[0,140],
          range_y=[0,100] ,
          animation_frame = "Age", 
          animation_group = "GenderCode", 
          color = "GenderCode" 
          )


# **Some 3D graphs**

# In[ ]:


fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection='3d')
ax.scatter3D(customer['Age'], customer['AnnualIncome'], customer['SpendingScore']
             , c=customer['GenderCode'], cmap='RdBu');
ax.set_xlabel('Age')
ax.set_ylabel('AnnualIncome')
ax.set_zlabel('SpendingScore')


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection='3d')
ax.plot(customer['Age'], customer['AnnualIncome'], customer['SpendingScore']);
ax.set_xlabel('Age')
ax.set_ylabel('AnnualIncome')
ax.set_zlabel('SpendingScore')


# <hr/>
# Now we will see clustering information

# <u>Scaling of the dataset</u>

# In[ ]:


ss= StandardScaler()
ss.fit(customer)
X = ss.transform(customer)
X.shape


# Scree plot using K-Means Algorithm

# In[ ]:


sse = []
for k in range(1,10):
    km = KMeans(n_clusters = k)
    km.fit(X)
    sse.append(km.inertia_)
plt.plot(range(1,10), sse, marker='*')


# Scree plot using Gaussian Mixture Algorithm

# In[ ]:


bic = []
aic = []
for i in range(8):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm.fit(X)
    bic.append(gm.bic(X))
    aic.append(gm.aic(X))

fig = plt.figure()
plt.plot([1,2,3,4,5,6,7,8], aic)
plt.plot([1,2,3,4,5,6,7,8], bic)
plt.show()


# Based on Scree plot using Gaussian Mixture Algorithm, we could finalize 2 cluster groups

# <u>Clustering visualization using K-Means Algorithm</u>

# In[ ]:


kmeans_bad = KMeans(n_clusters=2,
                    n_init =10,
                    max_iter = 800)
kmeans_bad.fit(X)

centroids=kmeans_bad.cluster_centers_

fig = plt.figure()
plt.scatter(X[:, 1], X[:, 2],
            c=kmeans_bad.labels_,
            s=2)
plt.scatter(centroids[:, 1], centroids[:, 2],
            marker='x',
            s=100,               # marker size
            linewidths=150,      # linewidth of marker edges
            color='red'
            )
plt.show()


# Clustering Visualization using Gaussian Mixture Algorithm

# In[ ]:


gm = GaussianMixture(
                     n_components = 2,
                     n_init = 10,
                     max_iter = 100)
gm.fit(X)
#gm.means_
#gm.converged_
#gm.n_iter_
#gm.predict(X)
#gm.weights_
#np.unique(gm.predict(X), return_counts = True)[1]/len(X)
#gm.sample()
fig = plt.figure()

plt.scatter(X[:, 1], X[:, 2],
            c=gm.predict(X),
            s=5)
plt.scatter(gm.means_[:, 1], gm.means_[:, 2],
            marker='v',
            s=10,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()


# TSNE Visualization using Gaussian Mixture Algorithm

# In[ ]:


gm = GaussianMixture(
                     n_components = 2,
                     n_init = 10,
                     max_iter = 100)
gm.fit(X)

tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(X)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='x',
            s=50,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm.predict(X)   # Colour as per gmm
            )


# Anamalies

# In[ ]:


densities = gm.score_samples(X)
densities

density_threshold = np.percentile(densities,5)
density_threshold

anomalies = X[densities < density_threshold]
anomalies
anomalies.shape



fig = plt.figure()
plt.scatter(X[:, 1], X[:, 2], c = gm.predict(X))
plt.scatter(anomalies[:, 0], anomalies[:, 1],
            marker='x',
            s=50,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )


# In[ ]:


unanomalies = X[densities >= density_threshold]
unanomalies.shape   

df_anomalies = pd.DataFrame(anomalies[:,[1,2]], columns=['salary','spendingscore'])
df_anomalies['type'] = 'anomalous'   # Create a IIIrd constant column
df_normal = pd.DataFrame(unanomalies[:,[1,2]], columns=['salary','spendingscore'])
df_normal['type'] = 'unanomalous'    # Create a IIIrd constant column


# In[ ]:


df_anomalies.head()
df_normal.head()


# In[ ]:



# 7.3 Let us see density plots
sns.distplot(df_anomalies['salary'], color='orange')
sns.distplot(df_normal['salary'], color='blue')


# In[ ]:


sns.distplot(df_anomalies['spendingscore'], color='orange')
sns.distplot(df_normal['spendingscore'], color='blue')


# In[ ]:



df = pd.concat([df_anomalies,df_normal])
df_anomalies.shape
df_normal.shape
df.shape


# In[ ]:


sns.boxplot(x = df['type'], y = df['salary'])


# In[ ]:


sns.boxplot(x = df['type'], y = df['spendingscore'])


# I just want to see the clustering without Gender columns <br>
# 
# <u>Clustering without Gender field</u>

# In[ ]:


customer_NoGender = customer.copy() #Deep Copy
customer_NoGender.drop(columns=["GenderCode"], inplace = True)
#customer.head()
customer_NoGender.head()


# In[ ]:


ss= StandardScaler()
ss.fit(customer_NoGender)
X = ss.transform(customer_NoGender)


# In[ ]:


bic = []
aic = []
for i in range(8):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm.fit(X)
    bic.append(gm.bic(X))
    aic.append(gm.aic(X))

fig = plt.figure()
plt.plot([1,2,3,4,5,6,7,8], aic)
plt.plot([1,2,3,4,5,6,7,8], bic)
plt.show()


# Without Gender field, Scree plot shows that we will have 5 clusters

# In[ ]:


gm = GaussianMixture(
                     n_components = 5,
                     n_init = 10,
                     max_iter = 100)
gm.fit(X)
#gm.means_
#gm.converged_
#gm.n_iter_
#gm.predict(X)
#gm.weights_
#np.unique(gm.predict(X), return_counts = True)[1]/len(X)
#gm.sample()
fig = plt.figure()

plt.scatter(X[:, 1], X[:, 2],
            c=gm.predict(X),
            s=5)
plt.scatter(gm.means_[:, 1], gm.means_[:, 2],
            marker='v',
            s=10,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )


# In[ ]:


gm = GaussianMixture(
                     n_components = 5,
                     n_init = 10,
                     max_iter = 100)
gm.fit(X)


# TSNE Visualization for 5 Clusters (without Gender field)

# In[ ]:


tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(X)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='x',
            s=50,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm.predict(X)   # Colour as per gmm
            )


# **End:**
# 
# 1. With Gender Column (2 clusters)
# 2. Without Gender Column (5 clusters)
# 
