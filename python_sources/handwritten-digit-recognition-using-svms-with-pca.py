#!/usr/bin/env python
# coding: utf-8

# ## Handwritten Digit Recognition
# 
# #### Problem Statement
# A classic problem in the field of pattern recognition is that of handwritten digit recognition. Suppose that you have images of handwritten digits ranging from 0-9 written by various people in boxes of a specific size - similar to the application forms in banks and universities.
# 
#  
# 
# The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 
# 
# #### Objective
# You are required to develop a model using Support Vector Machine which should correctly classify the handwritten digits from 0-9 based on the pixel values given as features. Thus, this is a 10-class classification problem. 
# 
#  
# 
# #### Data Description
# For this problem, we use the MNIST data which is a large database of handwritten digits. The 'pixel values' of each digit (image) comprise the features, and the actual number between 0-9 is the label. 
# 
#  
# 
# Since each image is of 28 x 28 pixels, and each pixel forms a feature, there are 784 features. MNIST digit recognition is a well-studied problem in the ML community, and people have trained numerous models (Neural Networks, SVMs, boosted trees etc.) achieving error rates as low as 0.23% (i.e. accuracy = 99.77%, with a convolutional neural network).
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, IncrementalPCA


# In[ ]:


sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
training_data = pd.read_csv("../input/digit-recognizer/train.csv")


# In[ ]:


training_data.shape


# It has 42000 rows and 785 columns (features)

# In[ ]:


training_data.info()


# In[ ]:


training_data.head()


# In[ ]:


training_data.max().sort_values()


# In[ ]:


training_data.isna().sum().sort_values(ascending=False)


# In[ ]:


training_data.duplicated().sum()


# There are no duplicated rows in the dataframe

# In[ ]:


training_data.columns


# In[ ]:


count_table = training_data.label.value_counts()
count_table = count_table.reset_index().sort_values(by='index')
count_table


# In[ ]:


plt.figure(figsize=(10, 5))
sns.barplot(x='index', y='label', data=count_table)


# In[ ]:


digit_means = training_data.groupby('label').mean()
digit_means.head()
plt.figure(figsize=(18, 10))
sns.heatmap(digit_means)


# In[ ]:


# average feature values
round(training_data.drop('label', axis=1).mean(), 2).sort_values()


# In[ ]:


# splitting into X and y
X = training_data.drop("label", axis = 1)
y = training_data['label']


# In[ ]:


# scaling the features
X_scaled = scale(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)


# In[ ]:


# applying PCA to find number of Principal components to use
pca = PCA(svd_solver='randomized', random_state=42)
pca.fit(X_train)


# In[ ]:


#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[ ]:


pca = IncrementalPCA(n_components=400)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[ ]:


X_train.shape


# ### Building and Evaluating the Final Model
# 
# Let's now build and evaluate the final model, i.e. the model with highest test accuracy.

# In[ ]:


# model with optimal hyperparameters

# model
model = SVC(C=10, gamma = 0.001, kernel="rbf")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")


# In[ ]:


# scaling test data
# splitting into X and y
X_test_data = test_data
X_test_data = scale(X_test_data)
X_test_data = pca.transform(X_test_data)
y_test_pred = model.predict(X_test_data)
y_test_pred


# In[ ]:


output = pd.DataFrame({"ImageId": i+1 , "Label": y_test_pred[i]} for i in range(0, X_test_data.shape[0]))
output.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




