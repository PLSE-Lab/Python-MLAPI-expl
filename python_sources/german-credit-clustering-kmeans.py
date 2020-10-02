#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
print(os.listdir("../input"))
df=pd.read_csv("../input/german_credit_data.csv")
#df=pd.read_csv("german_credit_data.csv")
df.head(10)


# checking for missing data in dataframe

# In[2]:


df.isnull().sum()


# In[3]:


df.drop("Checking account",axis="columns",inplace=True)


# In[4]:


#lets work on the missing values
#we use fwd fill and back fill
df["Saving accounts"].fillna(method="bfill",inplace=True)


# In[5]:


df.head()


# In[6]:


#dropping the first column as its of no use
df.drop(columns='Unnamed: 0',axis="columns",inplace=True)
df.head()


# In[7]:


df.corrwith(df["Credit amount"],axis=0) #data corellation wih each other


# Taking care of categorical values

# In[8]:


# we have categorical values like Sex, Job Housing and Purpose.
# Lets skip purpose from further analysis as its nota good parameter for the analysis as per my consideration.

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfe=df.copy() # lets take a copy of dataframe for analysis
dfe.Sex=le.fit_transform(dfe.Sex)
dfe.Housing=le.fit_transform(dfe.Housing)
dfe["Saving accounts"]=le.fit_transform(dfe["Saving accounts"])


# In[9]:


dfe.head(20)


# In[10]:


dfe1 =df.copy()
dfe.drop("Purpose",axis="columns",inplace=True)


# In[11]:


df.corrwith(df["Credit amount"],axis=0) #data corellation wih each other


# # Visualisations

# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,5))
plt.bar(df["Age"],df["Credit amount"],color="green")
#plt.scatter(df["Purpose"],df["Credit amount"],color="g")
plt.xlabel("Age")
plt.ylabel("credit amount")
plt.xlim(18,75)
plt.title("customers age and credit amount")
plt.show()


# In[13]:


import seaborn as sns
sns.distplot(df["Age"],bins=10,kde=True)


# In[14]:


#count plot shows that distribution of loan purpose with Gender. like men took more car loans than women.
plt.figure(figsize=(10,5))
sns.countplot(x="Purpose",data=df,hue="Sex")


# In[15]:


sns.distplot(df["Duration"]) 
# distribution of loans based on duration. Max loans are high duration one.


# In[16]:


sns.countplot(x="Housing",data=df,hue="Sex") # maximum loan applicants are male with on housing


# In[17]:


#categorical plotting #tip: put x axis as categorical value
#jobs and loan stats
#Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
sns.catplot(x="Job",y="Credit amount",data=df)
sns.catplot(x="Job",y="Age",data=df)
#Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)


# In[18]:


sns.catplot(x="Housing",y="Credit amount",data=dfe)


# In[19]:


sns.pairplot(dfe)


# In[20]:


dfe.corrwith(dfe["Credit amount"],axis=0) #data corellation wih each other


# In[21]:


dfe.head()


# In[22]:


df.columns


# In[23]:


dfe1 = df.copy()
dfe1.drop('Saving accounts', axis = 1, inplace = True)
dfe1.Job = dfe1.Job.astype(str)
dfe1 = dfe1[[ 'Job','Sex','Age','Credit amount', 'Duration','Housing', 'Purpose']]
X1 = pd.get_dummies(dfe1)
X1.head()


# In[24]:


# #Machine learning process started
# X=dfe[[ 'Job','Sex','Age','Credit amount', 'Duration']]
# X


# Scaling the data to remove disparity

# In[25]:


#scaling the features for the model.
#minmax scaler gave score as -54 
# from sklearn.preprocessing import MinMaxScaler
# mm=MinMaxScaler()
# X=mm.fit_transform(X1)
#normalize gave score -192
from sklearn.preprocessing import normalize
X=normalize(X1)

#satandard scaler was worst
# from sklearn.preprocessing import StandardScaler
# sk=StandardScaler()
# X=sk.fit_transform(X)


# In[26]:


#elbow plot to find then_clusters valuefor KMEANS algorithm
#to find SSE or sum square error
distorsions = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(1,20), distorsions)
plt.grid(True)
plt.title('Elbow curve')


# In[27]:


#ML model KMeans
from sklearn.cluster import KMeans
km=KMeans(n_clusters=5, max_iter=10000,random_state=None)
km.fit_transform(X)


# The k-means score is an indication of how far the points are from the centroids. In scikit learn, the score is better the closer to zero it is.
# 
# Bad scores will return a large negative number, whereas good scores return close to zero. Generally, you will want to take the absolute value of the output from the scores method for better visualization.

# In[28]:


km.score(X)# negetive score does not mean bad model


# The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
# 
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.

# In[29]:


from sklearn.metrics import silhouette_score
silhouette_score(X, km.labels_)


# In[ ]:




