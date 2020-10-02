#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load data into dataframe

# In[ ]:


# Load Data
df = pd.read_csv('/kaggle/input/nasa-asteroids-classification/nasa.csv')
df.head()


# ## Understanding the dependent variable

# In[ ]:


df['Hazardous'].value_counts()


# In[ ]:


df.corr()['Hazardous']


# ** Orbit uncertainty, absolute mangitude, and minimum orbit intersection show strongest correlation. **

# # Construct and test models

# ## Construct feature set

# In[ ]:


# Show independent variable names
df.columns


# In[ ]:


# Discard columns with identical values, identifiers, etc.
X = df[['Absolute Magnitude', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Epoch Date Close Approach', 'Relative Velocity km per hr', 'Miles per hour','Miss Dist.(miles)','Orbit Uncertainity','Minimum Orbit Intersection', 'Jupiter Tisserand Invariant','Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination','Asc Node Longitude', 'Orbital Period', 'Perihelion Distance','Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly','Mean Motion']]
X[0:5]


# In[ ]:


X.info()


# ## Normalize feature set

# In[ ]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[ ]:


y = df['Hazardous'].values
y[0:5]


# ## Split data into test and train set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# ## Train, test, evaluate 3-NN Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
neighbors = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
neighbors


# In[ ]:


# Predict using 3-NN Classifier
y_hat = neighbors.predict(X_test)
y_hat[0:5]


# In[ ]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neighbors.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_hat))


# ## Train, test, evaluate 5-NN Classifier

# In[ ]:


neighbors_1 = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

y_hat_1 = neighbors_1.predict(X_test)

print(neighbors_1)
print(metrics.accuracy_score(y_train, neighbors_1.predict(X_train)))
print(metrics.accuracy_score(y_test, y_hat_1))


# ## Determine best value of K

# In[ ]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neighbors = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    y_hat=neighbors.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_hat)

    
    std_acc[n-1]=np.std(y_hat==y_test)/np.sqrt(y_hat.shape[0])

mean_acc


# In[ ]:


print( "The best accuracy of", mean_acc.max(), "was measured with k=", mean_acc.argmax()+1) 


# # Conclusion
# Using 5 nearest-neighbors, we can train a classification model that performs at 89.8% accuracy predicting asteroid hazardousnous.

# In[ ]:


print('prediction set:', y_hat_1[0:5])
print('test set:      ', y_test[0:5])


# ## Next Steps
# 1. Use smaller feature set to avoid overfitting - strongly-correlated values (orbit_uncertainty, absolute_mangitude, minimum_orbit_intersection)
# 2. Better evaluation of model with multiple folds/sizes of test/train sets
# 

# In[ ]:




