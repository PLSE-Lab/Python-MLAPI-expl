#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Import our Libraries 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
# Any results you write to the current directory are saved as output.


# ## Import Data
# ### We first see Acceleration in the X, Y, and Z plane

# In[ ]:


swing = pd.read_csv("../input/swingClassification.csv")
swing.head()


# ## Stastical descripition

# In[ ]:


swing['Ax'].describe()


# ## Drop Unecessary Columns

# In[ ]:


acceleration_data = swing.drop(columns=['Swing'])
acceleration_data.head()


# ## Basic Line Visualization

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt


acceleration_data.plot(y='Az')


# ## Predict Swings with Machine Learning
# ![Ken Griffey Swing](https://media.giphy.com/media/3o7TKQVvClrMdt7mSs/giphy.gif)
# This will be a basic tutorial over using machine for sports performance purpose. We will be leveraging sensor data that is classified based on when someon is swinging a baseball bat, and when they are not swinging a baseball bat. This is one of the first steps when we are looking to optimize our performance, and hit homeruns like Ken Griffey Junior! 
# 
# Once we have our swings tracked automatically we can then figure out what are the flaws in our swing to be on track for greater performance. Let's roll tide

# # Import Data

# In[ ]:


swing_prediction = pd.read_csv("../input/swingClassification.csv")
swing_prediction.head()


# # Import Machine Learning Libraries SciKit Learn
# ### We are going to use the model of KNN nearest neighbor model to identifty class classification of models 
# 
# The end goal demonstration of this project will be to model scatter and group different swing types based on color. 
# Our Classification will only have two states if someone is swinging or not.
# 
# ![Image Classification Model](https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png)

# In[ ]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# # Establish Training Data
# We want to estable some training data so that our model will know how to perform.

# In[ ]:


feature_cols = ['Ax','Ay','Az']

#Input Data
X = swing_prediction[feature_cols]

#Output Data
y = swing_prediction['Swing']


# # Setup Knn Model Classifier

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)


# # Predict the Outcome 

# In[ ]:


y_predict = knn.predict(X)
print(metrics.accuracy_score(Y, y_predict))


# # Print Results
# 

# In[ ]:


for swing in y_predict:
    if swing: print("Swinging")
    else: print("Not Swinging")


# # Give Your Machine Learning a Color Map 

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

n_neighbors = 1
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', ])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', ])

for weights in ['uniform', 'distance']:
    #Create a matrix
    X_mat = X[['Ax', 'Ay']].as_matrix()
    y_mat = Y.as_matrix()
    
    # we create an instance of Neighbours Classifier and fit the data.
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    h = .02  # step size in the mesh
    
    #Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    vrb = np.c_[xx.ravel(), yy.ravel()]
    print(vrb.shape)
    print(vrb)
    Z = clf.predict(vrb)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))


plt.show()


# In[ ]:


x_min, x_max = X.min() - 1, X.max() + 1
y_min, y_max = Y.min() - 1, Y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
print(xx)
print(yy)
np.meshgrid(x_min)

clf.predict(xx,yy)
#X.min()['Ax']
#clf.predict(np.c_[xx.ravel(), yy.ravel()])


# In[ ]:




