#!/usr/bin/env python
# coding: utf-8

# **Clustering** is a Machine Learning technique that involves the grouping of data points. Given a set of data points, we can use a clustering algorithm to classify each data point into a specific group. In theory, data points that are in the same group should have similar properties and/or features, while data points in different groups should have highly dissimilar properties and/or features. Clustering is a method of unsupervised learning and is a common technique for statistical data analysis used in many fields.
# 
# We can use clustering analysis to gain some valuable insights from our data by seeing what groups the data points fall into when we apply a clustering algorithm.

# In[ ]:


import numpy as np 
import pandas as pd 


# **Let's Load the data**

# In[ ]:


titanic_data = pd.read_csv('../input/train.csv')
titanic_data.head()


# Here we will be looking for patterns with which we can group datapoints so all the features/columns are not important .
# 
# Some columns are passenger specific such as passengerId, Ticket, etc we can drop them .

# In[ ]:


titanic_data.drop(['PassengerId','Name','Ticket','Cabin'],'columns',inplace=True)
titanic_data.head()


# **Let's understand the data** :-
# 
# 1) Survived column tells us whether the Passenger survived the sinking of titanic or not. 
#     **0 - did not survive, 1 - survived**
#     
# 2) Pclass- is which class the passenger was travelling ,i.e 1st ,2nd or 3rd.
# 
# 3) Sex- male or female
# 
# 4) Age- How old the passenger is .
# 
# 5) SibSp and Parch - The number of siblings/parents aboard the titanic.
# 
# 6) Fare - the price of ticket
# 
# 7) Embarked - tells where the passenger boarded the ship . (C - Cherbourg, Q - Queenstown,S= Southampton )

# Lets convert the *Gender* column having the *categorical value in numeric form* .

# In[ ]:


from sklearn import preprocessing

le=preprocessing.LabelEncoder()
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'].astype(str))
titanic_data.head()


# Now for *Embarked column* we will be using *One Hot Encoding*

# In[ ]:


titanic_data = pd.get_dummies(titanic_data,columns=['Embarked'])
titanic_data.head()


# **Lets check for missing data values**

# In[ ]:


titanic_data[titanic_data.isnull().any(axis=1)]


# 177 rows have missing values . We will *drop them*.

# In[ ]:


titanic_data = titanic_data.dropna()


# **MeanShift**

# **Mean shift clustering is a sliding-window-based algorithm that attempts to find dense areas of data points**. It is a centroid-based algorithm meaning that the goal is to locate the center points of each group/class, which works by updating candidates for center points to be the mean of the points within the sliding-window. These candidate windows are then filtered in a post-processing stage to eliminate near-duplicates, forming the final set of center points and their corresponding groups. Check out the graphic below for an illustration.
# ![](http://cdn-images-1.medium.com/max/1600/1*bkFlVrrm4HACGfUzeBnErw.gif)
# 

# *1)* To explain mean-shift we will consider a set of points in two-dimensional space like the above illustration. We begin with a circular sliding window centered at a point C (randomly selected) and having radius r as the kernel. Mean shift is a hill climbing algorithm which involves shifting this kernel iteratively to a higher density region on each step until convergence.
# 
# *2)* At every iteration the sliding window is shifted towards regions of higher density by shifting the center point to the mean of the points within the window (hence the name). The density within the sliding window is proportional to the number of points inside it. Naturally, by shifting to the mean of the points in the window it will gradually move towards areas of higher point density.
# 
# *3)* We continue shifting the sliding window according to the mean until there is no direction at which a shift can accommodate more points inside the kernel. Check out the graphic above; we keep moving the circle until we no longer are increasing the density (i.e number of points in the window).
# 
# *4)* This process of steps 1 to 3 is done with many sliding windows until all points lie within a window. When multiple sliding windows overlap the window containing the most points is preserved. The data points are then clustered according to the sliding window in which they reside.
# 
# **An illustration of the entire process from end-to-end with all of the sliding windows is show below**. Each black dot represents the centroid of a sliding window and each gray dot is a data point.
# 
# ![](http://cdn-images-1.medium.com/max/1600/1*vyz94J_76dsVToaa4VG1Zg.gif)

# In[ ]:


from sklearn.cluster import MeanShift

analyzer = MeanShift(bandwidth=30) #We will provide only bandwith in hyperparameter . The smaller values of bandwith result in tall skinny kernels & larger values result in short fat kernels.
#We found the bandwith using the estimate_bandiwth function mentioned in below cell.
analyzer.fit(titanic_data)


# In[ ]:


#Below is a helper function to help estimate a good value for bandwith based on the data.
"""from sklearn.cluster import estimate_bandwith
estimate_bandwith(titanic_data)"""   #This runs in quadratic time hence take a long time


# In[ ]:


labels = analyzer.labels_


# In[ ]:


np.unique(labels)


# *Thus a bandwith of 30 produces 5 clusters - every point is assigned to one of these clusters.*

# In[ ]:


#We will add a new column in dataset which shows the cluster the data of a particular row belongs to.
titanic_data['cluster_group'] = np.nan
data_length=len(titanic_data)
for i in range(data_length):
    titanic_data.iloc[i,titanic_data.columns.get_loc('cluster_group')] = labels[i]
titanic_data.head()


# Lets do an **EDA on entire dataset** 

# In[ ]:


titanic_data.describe()


# We can see
# 
# 1) Total passenger information we have is 715.
# 
# 2) Out of these 715 passenger only 40% survived.
# 
# 3) PClass mean= 2.237 which shows there are more people in lower class cabins
# 
# 4) Mean of Sex column is 0.63 where 0 is female and 1 is male which means there were more males in ship than females.
# 
# 5) The average fare of passenger in this dataset is $35

# Lets do **EDA using the Clusters** now .

# In[ ]:


#Grouping passengers by Cluster
titanic_cluster_data = titanic_data.groupby(['cluster_group']).mean()
#Count of passengers in each cluster
titanic_cluster_data['Counts'] = pd.Series(titanic_data.groupby(['cluster_group']).size())
titanic_cluster_data


# Cluster 0 i.e the 1st Cluster
# * Have 558 passengers
# 
# * Survival rate is 33%(very low) means most of them didn't survive
# 
# * They belong to the lower classes 2nd and 3rd class mostly and are mostly male .
# 
# * The average fare paid is $15
# 
# Cluster 1 i.e the 2nd Cluster
# * Have 108 passengers
# 
# * Survival rate is 61% means a little more than half of them survived
# 
# * They are mostly from 1st and 2nd class
# 
# * The average fare paid is $65
# 
# Cluster 2 i.e the 3rd Cluster
# * Have 30 passengers
# 
# * Survival rate is 73% means most of them survived
# 
# * They are mostly from 1st class 
# 
# * The average fare paid is $131 (high fare)
# 
# Cluster 3 i.e the 4th Cluster
# * Have 15 passengers
# 
# * Survival rate is 73% means most of them survived
# 
# * They are mostly from 1st class and are mostly female
# 
# * The average fare paid is $239 (which is far higher than the 1st cluster average fare)
# 
# 
# The last cluster has just 3 datapoints so it is not that significant hence we can ignore for data analysis

# **Thanks.**

# *References :-*
# 
# *towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68*
