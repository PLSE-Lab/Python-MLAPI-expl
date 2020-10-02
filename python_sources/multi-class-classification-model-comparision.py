#!/usr/bin/env python
# coding: utf-8

# ### Tags
# 
# - Multiclass Classification
# - Numeric Data analysis
# - Seaborn Basics
# - Decision Tree, Random Forest Tree, SVM
# - Convolution matrix to undertand the result of classification
# - Accuracy, Precision, Recall and F1 Score
# - Tensorflow - Keras - Multiclass classification
# - Validation graph vs traning graph
# - Simple data normalization and manipulation using Sklearn

# In[ ]:


import pandas as pd
import pandas_profiling as pp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import tkinter
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


pp.ProfileReport(data)


# In[ ]:


data.quality.unique()


# ### Data Initial Understadning
# 
# - all the features are numeric value.
# - since features are numeric , they can be normalized. (Normalized data converge faster and better)
# - number of examples are not very large, only 1599 rows. (Simple machine learning algo can be enough)
# - number of features are not very large, so need for dimensionality resunction. (No need of PCA kind of data conversion)
# - no null values are present , so no pre processing required for filling null values.
# - based on the correlation diagram 
#     - sulphate, alcohol, and citric acid most closely related to wine quality in negative relation
#     - volatile acidity, chloride, total sulphur dioxide, density related to wine quality in positive relation.
#     

# ### Data Visualization

# In[ ]:


# Distribution of wine quality
# inline is required for placing the graph in motebook itself
get_ipython().run_line_magic('matplotlib', 'inline')

sns.countplot(x="quality", data=data)


# ### Label Engineering Analysis
# - there are 6 lable in total
# - value mostly concentrated for 5 and 6
# - since the number of examples are very less, and quality value varies from 1 - 10. we will convert quality in different bucket.
#     - 1-4 : bucket 0 [bad]
#     - 5-7 : bucket 1 [ok]
#     - above 7 : bucket 2 [good]

# In[ ]:


# alchol value - wine quality
sns.violinplot(x=data["alcohol"])


# In[ ]:


sns.violinplot(x=data["sulphates"])


# In[ ]:


sns.violinplot(x=data["volatile_acidity"])


# In[ ]:


sns.catplot(x="quality",y="alcohol",data=data)


# In[ ]:


sns.catplot(x="quality",y="sulphates",data=data)


# In[ ]:


sns.catplot(x="quality",y="volatile_acidity",data=data)


# ### Feature Engineering
# - shuffle the data 
# - bucketize the label into new buckets.
# - finalized the features columns and labels.
# - normalize the feature columns
# - deivide the dataset in training and validation set.
# - analyse the input feature dimension.

# In[ ]:


bins=[0,4,6,10]
labels=[0,1,2]
data['wine_quality']=pd.cut(data['quality'],bins=bins,labels=labels)


# In[ ]:


sns.countplot(x="wine_quality", data=data)


# In[ ]:


data.columns


# In[ ]:


# Bucketization of labels
features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol'] 
label = ['wine_quality']


# In[ ]:


data = shuffle(data)
X = data[features]
y = data[label]


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


print('Training Feature Shape', X_train.shape)
print('Validation Feature Shape', X_val.shape)

print('Label Training Shape', y_train.shape)
print('Label Validation Shape', y_val.shape)


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# In[ ]:


print('Training Feature Shape', X_train.shape)
print('Validation Feature Shape', X_val.shape)


# ### Machine Learning Alorithm Comparision
# - State Vector Machine
# - Decision Tree
# - Random Forest Tree
# - Simple Neural Network Using Tensorflow and Keras. [Deep Learning]

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 


# In[ ]:


model = SVC()
param = {
    'kernel':['rbf'],
    'C':[1,5,10,15, 20],
    'coef0':[0.001, 0.01,0.1, 0.5, 1]
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "gsc = GridSearchCV(\n        estimator=model,\n        param_grid=param,\n        cv=5, scoring='accuracy', verbose=0, n_jobs=-1)\ngrid_result = gsc.fit(X_train, y_train)\nprint('Best Param', grid_result.best_params_)")


# In[ ]:


y_pred = grid_result.best_estimator_.predict(X_val)


# In[ ]:


print(y_pred.shape)
print('Accuracy', accuracy_score(y_val, y_pred))
print("classification Report:\n",classification_report(y_val,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_val, y_pred))


# In[ ]:


model = DecisionTreeClassifier()
param = {
    'max_depth':[2,4,6,9,10,15],
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "gsc = GridSearchCV(\n        estimator=model,\n        param_grid=param,\n        cv=5, scoring='accuracy', verbose=0, n_jobs=-1)\ngrid_result = gsc.fit(X_train, y_train)\nprint('Best Param', grid_result.best_params_)")


# In[ ]:


y_pred = grid_result.best_estimator_.predict(X_val)


# In[ ]:


print(y_pred.shape)
print('Accuracy', accuracy_score(y_val, y_pred))
print("classification Report:\n",classification_report(y_val,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_val, y_pred))


# In[ ]:


model = RandomForestClassifier()
param = {
    'n_estimators':[2,4,6,9,10,15,20],
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "gsc = GridSearchCV(\n        estimator=model,\n        param_grid=param,\n        cv=5, scoring='accuracy', verbose=0, n_jobs=-1)\ngrid_result = gsc.fit(X_train, y_train)\nprint('Best Param', grid_result.best_params_)")


# In[ ]:


y_pred = grid_result.best_estimator_.predict(X_val)


# In[ ]:


print(y_pred.shape)
print('Accuracy', accuracy_score(y_val, y_pred))
print("classification Report:\n",classification_report(y_val,y_pred))
print('Confusion Matrix \n' , confusion_matrix(y_val, y_pred))


# In[ ]:


## Neural Network Using Tensorflow - Keras
### Sequential Model 

input_dimension = X_train.shape[1] # this represent number of features

### hyper parameters
epochs = 20
batch_size = 100

### model
model = Sequential()
model.add(Dense(12, input_shape=(input_dimension,), activation='relu', kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(X_train, y_train.values, epochs=epochs, batch_size=batch_size,
          validation_data=(X_val, y_val.values))


# In[ ]:


print('\nhistory dict:', history.history)


# In[ ]:


plt.plot( history.history['accuracy'], color='skyblue', linewidth=2, label='training acc')
plt.plot( history.history['val_accuracy'], color='green', linewidth=2, label='val acc')

plt.plot( history.history['loss'], color='skyblue', linewidth=2, linestyle='dashed', label="training loss")
plt.plot( history.history['val_loss'], color='green', linewidth=2, linestyle='dashed', label="val loss")
plt.legend()

