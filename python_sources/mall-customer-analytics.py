#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ***
# The Mall customers dataset contains information about people visiting the mall. The dataset has gender, customer id, age, annual income, and spending score. It collects insights from the data and group customers based on their behaviors.
# 
# Dataset: https://www.kaggle.com/shwetabh123/mall-customers
# 
# ## Problem
# ***
# Segment the customers based on the age, gender, interest. Customer segmentation is an important practise of dividing customers base into individual groups that are similar. It is useful in customised marketing.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


filepath="../input/mall-customers/Mall_Customers.csv"
df=pd.read_csv(filepath)
df.head()


# In[ ]:


df.rename(columns={"Genre":"Gender"}, inplace = True)
df


# In[ ]:


df.describe()


# # Customer gender visualization
# ***

# In[ ]:


df_gen = df["Gender"]
df_gen.value_counts().plot(kind="bar", legend =1)


# In[ ]:


df_gen.value_counts().plot(kind="pie",legend =1)


# ### With these charts, we can conclude that majority of the mall customers are female.

# ## Age Distribution
# ***

# In[ ]:


df_age = df["Age"]
df_age.describe()


# In[ ]:


df_age.hist(grid = 0)


# ### From the histrogram, we can conclude that people of age between 18 and 35 are frequently visiting the mall.

# # Annual Income Analysis

# In[ ]:


df.head()


# In[ ]:


df.rename(columns={"Annual Income (k$)":"Salary"}, inplace=1)
df.rename(columns={"Spending Score (1-100)":"Score"}, inplace=1)
df.head()


# In[ ]:


df["Salary"].describe()


# In[ ]:


h0=df["Salary"].hist(grid=0)
h0.set_title("Histogram of Annual Income of Customers")
h0.set_xlabel("Dollars(in Thousands)")
h0.set_ylabel("Frequency")


# In[ ]:


sns.distplot(df['Salary'])


# ### With the above graphs, we can see that most of the people have annual income between 50,000 and 75,000 Dollars.

# # Analyzing Spending score 
# ***

# In[ ]:


df.head()


# In[ ]:


sps=sns.boxplot(df["Score"], color="grey")
sps.set_title("Distribution of Spending Scores")


# In[ ]:


h1=df['Score'].hist(grid = 0, color= "grey")
h1.set_title("Histogram of Spending Scores")
h1.set_xlabel("Spending Scores")


# ### We can see that most of the people have scores between 35 to 75.

# # K-Means Clustering

# In[ ]:





# In[ ]:


df.head()
#df.drop(index="CustomerID")


# In[ ]:


plt.scatter(df["Salary"],df["Score"])


# ### Finding the number of clusters with the "elbow " method

# In[ ]:


k_range=[1,2,3,4,5,6,7,8,9,10]
sse=[]
for k in k_range:
    km=KMeans(n_clusters=k)
    km.fit(df[["Age","Salary","Score"]])
    sse.append(km.inertia_)


# In[ ]:


sse


# In[ ]:


plt.xlabel('k')
plt.ylabel('sum of squared error')
plt.plot(k_range,sse)


# ### The elbow point can be found at k=4 or 5. In other words, there are 4 or 5 clusters in this dataset.

# In[ ]:


km= KMeans(n_clusters=4)
km


# In[ ]:


y_pred=km.fit_predict(df[["Salary","Score"]])
y_pred


# In[ ]:


df['cluster']=y_pred
df.tail()


# In[ ]:


df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
df3=df[df.cluster==3]

plt.scatter( df0.Salary, df0.Score,color="grey")
plt.scatter( df1.Salary, df1.Score,color="red")
plt.scatter( df2.Salary, df2.Score,color="green")
plt.scatter( df3.Salary, df3.Score,color="blue")

plt.legend()


# ### There seems to be 5 clusters...

# In[ ]:


km1=KMeans(n_clusters=5)
y_pred1=km.fit_predict(df[["Salary","Score"]])
y_pred1


# In[ ]:


df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
df3=df[df.cluster==3]
df4=df[df.cluster==4]

plt.scatter( df0.Salary, df0.Score,color="grey")
plt.scatter( df1.Salary, df1.Score,color="red")
plt.scatter( df2.Salary, df2.Score,color="green")
plt.scatter( df3.Salary, df3.Score,color="blue")
plt.scatter( df4.Salary, df4.Score,color="black")


# ## It worked! Now, lets scale the graph for more accuracy

# In[ ]:


scaler= MinMaxScaler()
scaler.fit(df[['Salary']])
df["Salary"]= scaler.transform(df[["Salary"]])
df


# In[ ]:


scaler= MinMaxScaler()
scaler.fit(df[['Score']])
df["Score"]= scaler.transform(df[["Score"]])
df


# In[ ]:


km = KMeans(n_clusters=5)
y_pred=km.fit_predict(df[["Salary", "Score"]])
y_pred


# In[ ]:


df.head()


# In[ ]:


df['cluster']=y_pred
df.head()


# In[ ]:


df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
df3=df[df.cluster==3]
df4=df[df.cluster==4]

plt.scatter( df0.Salary, df0.Score,color="grey")
plt.scatter( df1.Salary, df1.Score,color="red")
plt.scatter( df2.Salary, df2.Score,color="green")
plt.scatter( df3.Salary, df3.Score,color="blue")
plt.scatter( df4.Salary, df4.Score,color="black")
plt.legend()


# ### With this scatter plot, we can deduce that customers can be segmented into 5 groups with varying characteristics. The businesses in the mall can be better serve their customer base by further studying their interests, which invlove what they buy, when they buy. how much they buy.

# In[ ]:




