#!/usr/bin/env python
# coding: utf-8

# # Part-1 Applying Different Estimators For Simple Classification Problem

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/iris-dataset/iris.data.csv', header=None)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


y = df.iloc[:,4].values


# In[ ]:


np.unique(y)


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[ ]:


le = le.fit(y)


# In[ ]:


y = le.transform(y)


# In[ ]:


y


# In[ ]:


le.classes_


# In[ ]:


X = df.iloc[:,[2,3]].values


# In[ ]:


X


# In[ ]:


X.shape, y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# In[ ]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[ ]:


X_train_std


# In[ ]:


X_test_std


# # Training a perceptron via scikit-learn

# In[ ]:





# In[ ]:


from sklearn.linear_model import LogisticRegression
clt = LogisticRegression(random_state=1).fit(X_train_std, y_train)


# In[ ]:


from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(X_train_std, y_train)


# In[ ]:


from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)


# In[ ]:


y_pred = clf.predict(X_test_std)


# In[ ]:


y_pred


# In[ ]:


print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[ ]:


from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[ ]:


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')


# In[ ]:


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# In[ ]:


X_combined_std.shape, y_combined.shape


# In[ ]:


y_train.shape, y_test.shape


# In[ ]:


plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


x1_min, x1_max = X_combined_std[:, 0].min() - 1, X_combined_std[:, 0].max() + 1
x2_min, x2_max = X_combined_std[:, 1].min() - 1, X_combined_std[:, 1].max() + 1


# In[ ]:


x1_min, x1_max, x2_min, x2_max


# In[ ]:


xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),np.arange(x2_min, x2_max, 0.02))


# In[ ]:


xx1.shape, xx2.shape


# In[ ]:


nx, ny = (3, 3)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)


# In[ ]:


x


# In[ ]:


y


# In[ ]:


xv, yv = np.meshgrid(x, y)


# In[ ]:


xv


# In[ ]:


yv


# In[ ]:


xv.ravel()


# In[ ]:


(np.array([xx1.ravel(), xx2.ravel()]).T).shape


# # Task 1 Use Classifiers: Logistic Regression, SVM and Decision Tree from sklearn

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Part-2 Data Manipulation

# In[ ]:


from io import StringIO
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
df


# # Eliminating samples or features with missing values

# In[ ]:


# remove rows that contain missing values

df.dropna(axis=0)


# In[ ]:


# remove columns that contain missing values

df.dropna(axis=1)


# In[ ]:


# only drop rows where all columns are NaN

df.dropna(how='all')


# In[ ]:


# drop rows that have less than 3 real values 

df.dropna(thresh=4)


# In[ ]:


# only drop rows where NaN appear in specific columns (here: 'C')

df.dropna(subset=['C'])


# # Imputing missing values

# In[ ]:


# impute missing values via the column mean

from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data


# # Encoding (One Hot Encoding)

# In[ ]:


import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']
df


# In[ ]:


from sklearn.preprocessing import LabelEncoder
X = df[['color', 'size', 'price']].values

color_le_1 = LabelEncoder()
X[:, 0] = color_le_1.fit_transform(X[:, 0])
X


# In[ ]:


color_le_2 = LabelEncoder()
X[:, 1] = color_le_2.fit_transform(X[:, 1])
X


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse=False), [0, 1]),], remainder='passthrough')


# In[ ]:


ct.fit_transform(X)


# # Wine Dataset

# In[ ]:


df_wine = pd.read_csv('../input/wine-quality/winequalityN.csv')


# In[ ]:


df_wine


# # Task 2 Apply different kind of filtering to clean the data

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


np.unique(df_wine['type'])


# # Assignment - Implement a perceptron from scratch and apply it to iris dataset given here.

# In[ ]:


from IPython.display import Image
Image("../input/percetron-image/Perceptron.PNG")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




