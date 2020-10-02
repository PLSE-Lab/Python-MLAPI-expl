#!/usr/bin/env python
# coding: utf-8

# # EEG brain wave for confusion
# 
# # <u>Table of Contents</u>
# 
# ## <a href='#setup'>Setup</a>
# 
# ## <a href='#visualization'>Visualization</a>
# 
# ###  <a href='#dataset'>Dataset</a>
# ###  <a href='#cfmd'>Check for missing data</a>
# ###  <a href='#corr'>Correlations</a>
# 
# 
# ## <a href='#method1'>Method 1: Logistic Regression<a/>
# 
# ## <a href='#method2'>Method 2: Neural Network<a/>
# 
# ## <a href='#method3'>Method 3: KNN<a/>

# # <a id=setup><u>Setup</u></a>
# 
# This section will be dedicated to importing data and packages necessary for this project

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


labels = ['Subject ID', 'Video ID', 'Attention', 'Mediation', 'Raw', 
          'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 
          'Gamma1', 'Gamma2', 'predefined_label', 'user-defined_label']


# In[ ]:


data = pd.read_csv("../input/EEG data.csv", header = None, names = labels)


# In[ ]:


#Create X and y 
X = data.ix[:,:'Gamma2']
y = data.ix[:,'user-defined_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)


#Create Scaled X
scaler = StandardScaler()
scaler.fit(data.drop('user-defined_label',axis=1))
scaled_features = scaler.transform(data.drop('user-defined_label',axis=1))
data_scaled = pd.DataFrame(scaled_features,columns=data.columns[:-1])

X_scaled = data_scaled.ix[:,:'predefined_label']


X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=14)


# # Visualization

# ### Dataset
# 
# **Let's take a look at our dataset.**

# In[ ]:


data.head(5)


# ### Check for Missing data
# 
# **Let's make a heat map of all of the data that is null to check for missing data.**
# 

# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False, cmap='viridis')


# ### Correlations
# 
# Let's see if there are any correlations between our data fields

# In[ ]:


sns.heatmap(data.corr())


# **"Past research has indicated that Theta signal is correlated with confusion level."**

# In[ ]:


sns.heatmap(data[['predefined_label','user-defined_label', 'Theta']].corr(),cmap='coolwarm',
            annot=True,square=True,lw=1,linecolor='black')


# #KNN
# 
# ## Training
# 
# #### Find the k with lowest error rate

# In[ ]:


error_rate = []

for i in range(1,10):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled,y_train)
    pred_i = knn.predict(X_test_scaled)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(20,5))
sns.plt.plot(range(1,10), error_rate)


# ### Train with the most optimal k
# 
# **It looks like taking 3 nearest neighbors yields the lowest error rate, so let's try that.**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


knn.fit(X_train_scaled,y_train)


# ## Testing

# In[ ]:


knn_predictions = knn.predict(X_test_scaled)


# In[ ]:


print(classification_report(y_test,knn_predictions))
confusion_matrix(y_test, knn_predictions)


# ##Conclusion
# 
# 
# The knn method seems to be the best solution.  However, I am very skeptical because the accuracy is very high compared to other works. I would assume that I am making some kind of error an will look into this.  
# 
# The reason for such a high accuracy might be due to the fact that I am using all data insluding the predefined label
