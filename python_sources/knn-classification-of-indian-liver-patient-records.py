#!/usr/bin/env python
# coding: utf-8

# IMPORTING THE NECESSARY LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# READING THE DATA FILE IN A DATAFRAME USING PANDAS

# In[ ]:


df = pd.read_csv('../input/ILPD.csv')


# VIEWING THE HEAD OF THE DATASET

# In[ ]:


df.head()


# REMOVING AGE AND GENDER

# In[ ]:


df.drop('age',axis=1,inplace=True)


# In[ ]:


df.drop('gender',axis=1,inplace=True)


# In[ ]:


WATCHING THE HEAD OF THE DATA


# 

# In[ ]:


df.head()


# STANDARDIZATION OF THE DATA WITH MEAN = 0 S.D = 1

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaled_values = scaler.fit(df.drop('is_patient',axis=1))
scaled_values =scaler.transform(df.drop('is_patient',axis=1))


# CREATING DATA FRAME FROM THE SCALED VALUES

# In[ ]:


df2 = pd.DataFrame(scaled_values,columns=df.columns[:-1])


# In[ ]:


df2.head()


# PAIRPLOT OF THE INITIAL DATASET

# In[ ]:


sns.pairplot(data=df,hue='is_patient')


# IMPORTING THE KNN CLASSIFIER CLASS

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier()


# IMPORTING TRAIN_TEST_SPLIT FOR SPLITTING DATASET

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X= df2
y=df['is_patient']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# CHECKING FOR K=1 TO 40

# In[ ]:


error_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    error_rate.append(np.mean(y_test != pred))


# PLOTTING THE ERROR GRAPH FOR K= 1 TO 40

# In[ ]:


plt.figure(figsize=(10,6))
sns.set(style="whitegrid")
plt.plot(range(1,40),error_rate,color='green', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# CHECKING THE ACCURACY USING CLASSIFICATION REPORT

# In[ ]:


#at K = 33
from sklearn.metrics import classification_report


# In[ ]:


knn2= KNeighborsClassifier(n_neighbors=33)
knn2.fit(X_train,y_train)
pred2 = knn2.predict(X_test)
print(classification_report(y_test,pred))

