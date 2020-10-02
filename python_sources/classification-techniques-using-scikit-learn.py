#!/usr/bin/env python
# coding: utf-8

# Created by: Sangwook Cheon
# 
# Date: Dec 23, 2018
# 
# This is step-by-step guide to Classification using scikit-learn, which I created for reference. I added some useful notes along the way to clarify things. I am excited to move onto more advanced concepts, such as deep learning, using frameworks like Keras and Tensorflow.
# This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of classification.
# 
# # Content:
# 
# ### 1. Logistic Regression
# ### 2. K-nearest Neighbors (KNN)
# ### 3. Support Vector Machine (SVM)
# ### 4. Kernel SVM
# ### 5. Naive Bayes algorithm
# ### 6. Decision Tree Classification
# ### 7. Random Forest Classification
# ### 8. Evaluating Classification models
# 
# __________
# _________

# # Logistic Regression 
# Use the same library that was used in Regression notebook.
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
# Y is not scaled, because it contians categorical values.


# In[ ]:


from sklearn.linear_model import LogisticRegression #Logistic regression  is still a linear model

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
y_pred, Y_test


# In[ ]:


#Making the Confusion Matrix --> to assess accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm


# In[ ]:


#65 + 24 = 89 correct results, 3+8 = 11 incorrect results.

#Visualizing the Training Set results
plt.figure(1)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted')
plt.legend()
plt.show()


# Prediction boundary determines the boundary between categories. 
# As above model was a linear classifier, the prediction boundary can only be a straight line. Non-linear boundaries can be created as well.

# # K-nearest Neighbor (KNN)
# 
# ![i2](https://i.imgur.com/mA3sq9N.png) 
# 
# In above case, 5 neighbors were chosen.
# 
# Step 1: Choose the number K of neighbors
# Step 2: Take the K nearest neighbors of the new data point, according to the Euclidean distance
# Step 3: Among these K neighbors, count the number of data points in each category
# Step 4: Assign the new data point to the category where I counted the most neighbors.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #n_neighbors parameter can be tuned, but will learn later.
#minkowsky - 2 --> Euclidean distance
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


# In[ ]:


#visualizing the result

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (KNN)')
plt.legend()
plt.show()


# # Support Vector Machine (SVM)
# 
# ![i3](https://i.imgur.com/HzHqaGA.png)
# 
# Support vectores are those who do not have the clearest characteristics of each category. This makes SVM special. For example, support vectors might be an orange that looks like an apple, or an apple that resembles an orange, if the problem was to classify between oranges and apples.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.svm import SVC #Support vecture classifier
classifier = SVC(kernel = 'linear', random_state = 0) #other options for kernel: 'poly', 'rbf' (gaussean), 'sigmoid', 'precomputed'
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm


# This is a linear classification. Kernel SVM should be used for better fitting.
# 
# # Kernel SVM
# ### Mapping data to a higher dimension to make non-linearly-separable to separable
# This is possible, by using certain mapping functions, such as 1d line to 2d parabola, but it requries a lot of computing and processing power. Therefore, a different approach should be explored: 
# 
# ### Gaussian RBF Kernel
# This allows transformation of each data point without actually mapping it into a higher dimension. Using the specific formula, computer only computes values while not creating an entirely new higher-dimensional array, which saves processing power. A landmark should be placed within the data, and distance from each point to the landmark is used to determine its position. If it is close to the landmark, it has the value greater than 0. The maximum value is 1, which would be the landmark itself. 
# 
# Formula: ![i4](https://i.imgur.com/Moe3GSG.png) 
# 
# #### Different types of kernels
# * Gaussian RBF Kernel
# * Sigmoid Kernel
# * Polynomial Kernel --> popular
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (rbf SVM Kernel)')
plt.legend()
plt.show()


# # Naive Bayes
# 
# ## Bayes Theorem
# Terms:
# 
# P(Something | condition) = 50% --> Probability of something happening given a condition is 50%
# ![i5](https://i.imgur.com/UX47tMA.png)

# # Naive Bayes Classification algorithm

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (Naive Bayes)')
plt.legend()
plt.show()


# # Decision Tree Classification

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (DecisionTree Classification)')
plt.legend()
plt.show()


# ### Interpreting the result
# 
# This seems to have the characteristics of overfitting. 

# # Random Forest Classification
# 
# This is a team of tree models, so this will be more powerful.
# Emsemble learning: combining many machine learning models to produce one better result. 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (Random Forest Classification)')
plt.legend()
plt.show()


# Testing the classfier on test set is needed to detect overfitting.
# 
# # Evaluating Classification models 
# False Negative Errors (Type II error) : When the model predicted a value that is rounded to be a higher category, but when the result is actually lower category. 
# False Positive Errors (Type I error): When the model predicted a value that is rounded to be a lower category, but when the result is actually the higher category. #When the model says it will happen, but it actually doesn't happen.
# 
# ## Confusion matrix
# 

# In[ ]:


cm


# Here, 63 and 28 are correct values. 5 is the false positive values, and 4 is the false nagative values.
# 
# Accuracy: (91/100) = 0.91
# 
# ## Cumulative Accuracy Profile (CAP)
# this is used to analyze the performance of the model
# ![i6](https://i.imgur.com/PFXbIfS.png)
# If the area between the performance curve and a straight line (generated by a random sample) is higher, it shows that the model is better. If it approaches the perfect line (ideal line), it is better.
# 
# #### How to quantify CAP
# First option: Area between the trained model and the random model line/Area between the random model line and perfect line. Closer to 1, better the model
# 
# Second option: take the halfway point of the x-axis, then see what that value projects to (the corresponding y_value). If the y_value is:
# 
# * < 60% ---> Rubbish
# * 60% < y < 70% --> Poor
# * 70% < y < 80% ---> good
# * 80% < y < 90% ---> Very good
# * y > 90% --> too good
