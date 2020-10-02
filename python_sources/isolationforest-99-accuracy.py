#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("../input/creditcard.csv")


# In[3]:


print(data.columns)


# In[4]:


print(data.shape)


# In[5]:


data.describe()


# In[6]:


data = data.sample(frac = 0.1, random_state = 1)


# In[7]:


print(data.shape)


# In[8]:


data.hist(figsize = (20,20))
plt.show


# In[9]:


fraud = data[data['Class']==1]
valid = data[data['Class']==0]

outlier_fraction = len(fraud)/float(len(valid))

print('fraud calss: {}'.format(len(fraud)))
print('valid class: {}'.format(len(valid)))


# In[10]:


corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax = 0.8, square = True)
plt.show()


# In[11]:


columns = data.columns.tolist()

columns = [c for c in columns if c not in ['Class']]

target = 'Class'

X = data[columns]
Y = data[target]

print(X.shape)
print(Y.shape)


# In[12]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
state = 1

classifier = {
    "Isolation Forest": IsolationForest(max_samples = len(X), contamination = outlier_fraction, random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
                            
}


# In[13]:


n_outliers = len(fraud)


for i, (clf_name, clf) in enumerate(classifier.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:





# In[ ]:




