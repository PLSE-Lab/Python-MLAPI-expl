#!/usr/bin/env python
# coding: utf-8

# # **K Nearest Neighbors Project**

# Import required libraries

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Get the data

# In[ ]:


df = pd.read_csv("../input/knn-data1/KNN_Project_Data")


# In[ ]:


df.head()


# # **Exploratory Data Analysis**

# In[ ]:


sns.pairplot(df,hue="TARGET CLASS",palette="coolwarm")


# # **Standardize the Variables**

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


scaler.fit(df.drop('TARGET CLASS', axis=1))


# In[ ]:


scaled_features=scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[ ]:


df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()


# # **Train Test Split**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_feat
y = df['TARGET CLASS']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # **Using KNN**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train, y_train)


# # **Predictions and Evaluations**
# Let's evaluate our KNN Model

# In[ ]:


pred = knn.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# # **Choosing a K Value**
# Let's go ahead and pick up a good K value using the elbow method

# In[ ]:


error_rate = []
for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue',linestyle="--",marker="o", markerfacecolor='red',markersize=10)
plt.title("Error Rate Vs K")
plt.xlabel("K")
plt.ylabel("Error Rate")


# # **Retrain with new K Value**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:




