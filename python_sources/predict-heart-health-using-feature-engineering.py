#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


heart = pd.read_csv("../input/heart.csv")
heart.head()


# In[ ]:


print(heart.dtypes)


# In[ ]:


sns.pairplot(heart[['age','sex','cp','chol','slope']], hue='slope', palette='afmhot',size=1.4)


# In[ ]:


heart.target.value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
        heart.drop('target', 1), 
        heart['target'], 
        test_size = 0.3, 
        random_state=10
        ) 


# In[ ]:


X_train.shape                        
                      
                     
                      


# In[ ]:


X_test.shape 


# In[ ]:


y_test.shape  


# In[ ]:


y_train.shape 


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import log_loss


# In[ ]:


y_test_pred = clf.predict_proba(X_test)
log_loss(y_test, y_test_pred)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# In[ ]:


rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(X_train, y_train)
heart_preds = rf.predict(X_test)
print(mean_absolute_error(y_test, heart_preds))


# In[ ]:


clf.feature_importances_ 
clf.feature_importances_.size


# In[ ]:


(heart_preds == y_test).sum()/y_test.size  # Check the accuracy


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale


# In[ ]:


############################ Feature creation using kmeans ####################
# Create a StandardScaler instance
se = StandardScaler()


# In[ ]:


#fit() and transform() in one step
heart = se.fit_transform(heart)


# In[ ]:


heart.shape


# In[ ]:


#  Perform kmeans using 13 features.
#     No of centroids is no of classes in the 'target'
centers = y_train.nunique()  
centers       # 2


# In[ ]:


from sklearn.cluster import KMeans  


# In[ ]:


# Begin clustering
#First create object to perform clustering
kmeans = KMeans(n_clusters=centers, # How many
                n_jobs = 5)         # Parallel jobs for n_init


# In[ ]:


kmeans.fit(heart[:, : 13])
kmeans.labels_
kmeans.labels_.size


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


# Create an instance of OneHotEncoder class
ohe = OneHotEncoder(sparse = False)


# In[ ]:


ohe.fit(kmeans.labels_.reshape(-1,1)) 


# In[ ]:


# Transform data now
dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))
dummy_clusterlabels
dummy_clusterlabels.shape 


# In[ ]:


#  We will use the following as names of new two columns
#      We need them at the end of this code

k_means_names = ["k" + str(i) for i in range(2)]
k_means_names

