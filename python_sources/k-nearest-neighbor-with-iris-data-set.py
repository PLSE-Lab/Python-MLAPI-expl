#!/usr/bin/env python
# coding: utf-8

# ### Machine Learning with python using IRIS data set ( Example of Classification - Supervised learning) 

# In[2]:


## import standard libraries for our work
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ####  importing the 'sklearn' library that is used for machine learning 
# 

# In[3]:


import sklearn


# #####  iris is botanical data set  that classifies the flowers into a specific species based on some measurementsts. Using 'load_iris' built in fuction loading this data into  a Bunch object

# In[4]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[5]:


type(iris)


# #### Convert this Bunch data set to Dataframe 

# In[6]:


data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


# In[7]:


data1.shape  #There are 5 features ( columns ) and 150 rows, observations


# In[8]:


data1.head()


# In[9]:


data1['target'].value_counts()
### There are basically 3 types of categories 0 means setosa, 1 means versicolor, and 2 means virginica


# #### Add a column 'Species' to the Dataset with this classification 

# In[10]:


def categorize(a):
    if a == 0.0:
        return('setosa')
    if a == 1.0:
        return('versicolor')
    return('virginica')
data1['Species'] = data1['target'].apply(categorize)
    


# In[11]:


data1.head()


# ### Plot the data and classfications to see data has a classification

# In[12]:


plt.figure(figsize=[18,8])
plt.scatter(data1['Species'], data1['sepal length (cm)'],  marker= 'o')
plt.scatter(data1['Species'], data1['sepal width (cm)'], marker= 'x')
plt.scatter(data1['Species'], data1['petal width (cm)'], marker= '*')
plt.scatter(data1['Species'], data1['petal length (cm)'], marker= ',')
plt.ylabel('Length in cm')
plt.legend()
plt.xlabel('Species Name')
plt.show()


# In[13]:


plt.figure(figsize=[18,8])
plt.plot(data1['sepal length (cm)'], marker= 'o')
plt.plot(data1['sepal width (cm)'], marker= 'x')
plt.plot(data1['petal length (cm)'], marker= '*')
plt.plot(data1['petal width (cm)'], marker= ',')
plt.ylabel('Length in cm')
plt.legend()
plt.show()


# In[27]:


sns.jointplot(data1['sepal length (cm)'], data1['sepal width (cm)'], size= 13, kind = 'kde')


# In[28]:


sns.jointplot(data1['petal length (cm)'], data1['petal width (cm)'], size= 13, kind = 'kde')


# #### From the above plots , there appears a  grouping trend of data elements. 

# ### Objective of this machine learning exercise : 
# ####    The  flower based on its physical measurements,  is classfied into a specifc species .  It means, there is relation ship between the physical measurements and the species. We need to establish a model / Method through which for a given measurements we should be able to clasify the species. From the given dataset, machine learning happens to define the relationship and a model is built with which we can predict the species.
# 
# #### Steps Involved
#       #1. Split the given data into two sets - Train Data, Test Data
#       #2. Plot the data for visual inspections.
#       #3. Building K- Nearest neighbour classifier model
#   
# 

# #### For Machine learning purpose the data is split into two portions Train data, Test Data

# In[29]:


### It is a standard convention to name X_train in capital X and y_train in small letters. 
###  All the measurements (features) are considered as X and the Species is considered as y


# In[30]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data1[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], data1['Species'], random_state=0 )


# In[31]:


X_train.head()


# In[32]:


y_train.head()


# In[33]:


X_test.head()


# In[34]:


y_test.head()


# #### K-Nearest Neighbours Model
#     ### for a test data set ( In this case the measurements of the flower 4 values ) classifying the data to nearest data point       and identify the species 
#     

# In[35]:


from sklearn.neighbors import KNeighborsClassifier


# In[36]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[37]:


knn.fit(X_train, y_train) # This is fitting the model  with the training data. 


# In[38]:


prediction = knn.predict(X_test) # By supplying the test data now predicting the  Y (Species values)


# In[39]:


prediction


# In[40]:


y_test + "  " +  prediction #
#Comparision of the predicted data from the Test sent and the y_test data
# Predicted data and the y_test data are same. This gives the highest confidence level on the model built


# In[41]:


### Now we can test the model using any data and it would be accurate 


# #### Testing the model with some test data

# In[42]:


X_new = np.array([[5, 2.9, 1, 0.2]])


# In[43]:


predection1 = knn.predict(X_new)


# In[44]:


predection1


# In[ ]:





# In[ ]:




