#!/usr/bin/env python
# coding: utf-8

# # **BREAST CANCER CLASSIFICATION**

# ## STEP #1: PROBLEM STATEMENT

# - Predicting if the cancer diagnosis is benign or malignant based on several observations/features
# - 30 features are used, examples:
#     - radius (mean of distances from center to points on the perimeter)
#     - texture (standard deviation of grey-scale values)
#     - perimeter
#     - area
#     - smoothness (local variation in radius lengths)
#     - compactness (perimeter^2 / area - 1.0)
#     - concavity (severity of concave portions of the contour)
#     - symmetry
#     - fractal dimension ("coastline approximation" - 1)
# - Datasets are linear separable using 30 input features
# - Number of instances: 569
# - Class Distribution: 212 Malignant, 357 Benign
# - Target Class:
#     - Malignant
#     - Benign

# ## STEP #2: IMPORTING DATA

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.datasets import load_breast_cancer


# In[ ]:


cancer = load_breast_cancer()
cancer


# In[ ]:


cancer.keys()


# In[ ]:


print(cancer['DESCR'])


# In[ ]:


print(cancer['target'])


# In[ ]:


print(cancer['target_names'])


# In[ ]:


print(cancer['feature_names'])


# In[ ]:


cancer['data'].shape


# In[ ]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()


# ## STEP #3: VISUALIZING THE DATA

# In[ ]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])


# In[ ]:


sns.countplot(df_cancer['target'])


# In[ ]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[ ]:


plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot = True)


# ## STEP #4: MODEL TRAINING

# In[ ]:


X = df_cancer.drop(['target'], axis = 1)
X.head()


# In[ ]:


y = df_cancer['target']
y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)


# In[ ]:


print('Shape of X_train: ', X_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of y_test: ', y_test.shape)


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


svc_model = SVC()
svc_model.fit(X_train, y_train)


# ## STEP #5: MODEL EVALUTION

# In[ ]:


y_predict = svc_model.predict(X_test)
y_predict


# In[ ]:


cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)


# ## STEP #6: IMPROVING THE MODEL

# In[ ]:


min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train


# In[ ]:


sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[ ]:


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# In[ ]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[ ]:


svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)


# In[ ]:


cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)


# In[ ]:


print(classification_report(y_test, y_predict))


# ### IMPROVING THE MODEL - PART 2

# In[ ]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit = True,verbose=4)
grid.fit(X_train_scaled,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid_predictions = grid.predict(X_test_scaled)


# In[ ]:


cm = confusion_matrix(y_test, grid_predictions)

