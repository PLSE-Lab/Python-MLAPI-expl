#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors Project 
# 
# ## Import Libraries
# **Import pandas,seaborn, and the usual libraries.**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# In[ ]:


dataframe = pd.read_csv("../input/knn-data1/KNN_Project_Data")


# **Check the head of the dataframe.**

# In[ ]:


dataframe.head()


# # EDA
# 
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[ ]:


sns.pairplot(dataframe, hue="TARGET CLASS")


# # Standardize the Variables
# 
# Time to standardize the variables.
# 
# ** Import StandardScaler from Scikit learn.**

# In[ ]:


from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**

# In[ ]:


scaler = StandardScaler()


# ** Fit scaler to the features.**

# In[ ]:


scaler.fit(dataframe.drop("TARGET CLASS", axis=1))


# **Use the .transform() method to transform the features to a scaled version.**

# In[ ]:


scaled_feat = scaler.transform(dataframe.drop("TARGET CLASS", axis=1))
scaled_feat


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[ ]:


df_feat = pd.DataFrame(scaled_feat, columns=dataframe.columns[:-1])
df_feat.head()


# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_feat
y = dataframe["TARGET CLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# # Using KNN
# 
# **Import KNeighborsClassifier from scikit learn.**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# **Create a KNN model instance with n_neighbors=1**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# **Fit this KNN model to the training data.**

# In[ ]:


knn.fit(X_train, y_train)


# # Predictions and Evaluations
# Let's evaluate our KNN model!

# **Use the predict method to predict values using your KNN model and X_test.**

# In[ ]:


predictions = knn.predict(X_test)


# ** Create a confusion matrix and classification report.**

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


print(confusion_matrix(y_test, predictions))


# In[ ]:


print(classification_report(y_test, predictions))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

# In[ ]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_new = knn.predict(X_test)
    error_rate.append(np.mean(pred_new != y_test))


# **Now create the following plot using the information from your for loop.**

# In[ ]:


plt.figure(figsize=(15,4))
plt.plot(range(1,40), error_rate)


# ## Retrain with new K Value
# 
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
pred_new = knn.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

