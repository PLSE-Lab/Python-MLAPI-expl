#!/usr/bin/env python
# coding: utf-8

# **This notebook presents a simple yet interesting classification of customers in a shopping mall by using the KMeans method; the data we have at our disposal is income, spending score and some demographics.**

# In[ ]:




import numpy as np 
import pandas as pd 



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn import preprocessing

import matplotlib.pyplot as plt
db = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
db.head()


# As it can be seen, most of the variables, with the exception of gender, are numeric, so first things first, I'll convert the categorical gender into numeric.

# In[ ]:


#encode gender
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
gender_cat = db[['Gender']] 
gender_cat_encoded = ordinal_encoder.fit_transform(gender_cat)

#make it a dataframe
gender_new = pd.DataFrame(gender_cat_encoded)
gender_new.columns = ['gender_new']

#add it to the original database
db = db.merge(gender_new,  how = 'inner', left_index = True, right_index = True)

#remove the categorical Gender variable because we don't need it anymore
db = db.drop('Gender', axis = 1)
db.head()


# In[ ]:


db.info()

So there are 200 non-null observations and no categorical variables anymore.
All good, now what I want is to cluster the customers, and for this I'll firstly choose to keep a database that has only the variables that are the most important for clustering at the moment: income and spendings.
# In[ ]:


db2 = db.iloc[:,[2,3]]
db2.head()


# The next step is to determine how many clusters are optimal; for this I'll be using the Elbow method, testing number of clusters between 1 and 10 (remember we have only 200 observations, so I can't choose an unrealistic number of clusters to test).

# In[ ]:


from sklearn.cluster import KMeans


cl = []
list_k = list(range(1, 10))

for k in list_k:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(db2)
    cl.append(kmeans.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, cl, '-o')
plt.xlabel(r'Clusters number')
plt.ylabel('Sum of squared distance')


# The elbow of the curve seems to be at 5 so I'll go with 5 clusters.

# In[ ]:


kmeans = KMeans(n_clusters=5) 
kmeans.fit(db2)

y_km = kmeans.fit_predict(db2)

db2 = np.array(db2)

plt.title("Clusters", fontsize=20)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.scatter(db2[y_km ==0,0], db2[y_km == 0,1], c='red')
plt.scatter(db2[y_km ==1,0], db2[y_km == 1,1], c='black')
plt.scatter(db2[y_km ==2,0], db2[y_km == 2,1], s=100, c='blue')
plt.scatter(db2[y_km ==3,0], db2[y_km == 3,1], s=100, c='orange')
plt.scatter(db2[y_km ==4,0], db2[y_km == 4,1], s=100, c='yellow')


# So we can clearly identify our 5 clusters; according to them, we can say that the clients can be classified as follows: people that don't earn too much are two types: some that keep their spendings low (orange) and those that spend a lot(blue). The ones with high incomes can be split in a similar way: the red cluster are the people that keep their spendings at lower levels, while the yellow ones spend a lot. There is also a medium cateogory: people that have medium level incomes and keep their spendings at medium levels.

# For a better understanding, we could also have a look at how people spend by age category; for this I'll make a database with only age and spendings.

# In[ ]:


db3 = db.iloc[:,[1,3]]
db3.head()


# Let's see how many clusters we can get:

# In[ ]:


cl = []
list_k = list(range(1, 10))

for k in list_k:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(db3)
    cl.append(kmeans.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, cl, '-o')
plt.xlabel(r'Clusters number')
plt.ylabel('Sum of squared distance')


# Four clusters seems just right.

# In[ ]:


kmeans = KMeans(n_clusters=4) 
kmeans.fit(db3)

y_km = kmeans.fit_predict(db3)

db3 = np.array(db3)

plt.title("Clusters", fontsize=20)
plt.xlabel("Age")
plt.ylabel("Spending Score")

plt.scatter(db3[y_km ==0,0], db3[y_km == 0,1], c='red')
plt.scatter(db3[y_km ==1,0], db3[y_km == 1,1], c='black')
plt.scatter(db3[y_km ==2,0], db3[y_km == 2,1], s=100, c='blue')
plt.scatter(db3[y_km ==3,0], db3[y_km == 3,1], s=100, c='orange')


# Ok, so here we can clearly identify a group of people that keep their spendings and a low level, msot of them seem to be above 30 years old; the ones below 30 years old are two types: people that keep their spending at a medium level a people that spend a lot. As seen previously, this last group is made of people of opposite level of income, low and high. The ones that spend at a medium level, in red and orange, are the the group we've seen above in the black cluster, people that also have medium incomes,
