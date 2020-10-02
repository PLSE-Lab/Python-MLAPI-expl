#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


df.info()


# In[ ]:


df.head()


# # EDA

# In[ ]:


# Use seaborn on the dataframe to create a pairplot with the hue indicated by the Outcome column.
# It is very large plot
sns.pairplot(df,hue='Outcome',palette='coolwarm');


# In[ ]:


sns.pairplot(df)


# # Standardize the Variables

# In[ ]:


# import the main KNN ilibrary
from sklearn.preprocessing import StandardScaler


# In[ ]:


# StandardScaler() object called scaler
scaler = StandardScaler()


# In[ ]:


# Fit the scaler to the features
scaler.fit(df.drop('Outcome',axis=1))


# In[ ]:


# Transform the features to a scaled version 
scaled_features = scaler.transform(df.drop('Outcome',axis=1))


# In[ ]:


# Convert the scaled features to a dataframe 
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[ ]:


# Train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Outcome'],
                                                    test_size=0.30)


# # Using KNN

# In[ ]:


# Create a KNN model instance with n_neighbors=1# 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[ ]:


# Fit this KNN model to the training data

pred = knn.predict(X_test)


# # Predictions

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# # How to choosing a K Value

# In[ ]:


error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


#After that we choose some K Value for available algorihmas value
# Retrain with new K Value
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

