#!/usr/bin/env python
# coding: utf-8

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


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


object_cols_train = train_df.select_dtypes("object").columns
print(object_cols_train)


# In[ ]:


test_df.info()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


object_cols_test = test_df.select_dtypes("object").columns
print(object_cols_test)


# In[ ]:


import seaborn as sns #data visualization is more grafical and user friendly
import matplotlib.pyplot as plt #data visualization

#see if the our target is balances
sns.countplot(x="Survived", data=train_df)


# In[ ]:


sns.countplot(x="Survived", data=train_df, hue="Sex")


# In[ ]:


#Correlation of all features
plt.figure(figsize=(12, 8))
plt.title('Titanic Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.corr().abs(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)


# In[ ]:


#correlation: if the correlation of 2 features are too high it means it is redundant
train_df.corr().abs()


# In[ ]:


#correlation based on one parameter only
train_df.corr().Survived.abs().sort_values()


# In[ ]:


#distriuation of all columns based on each other
sns.pairplot(train_df)


# In[ ]:


#distribution plot
sns.distplot(train_df['Age'], bins=24, color='b')


# In[ ]:


#Because Cabin has alot of null drop it
#Name PassengerId are non related so drop it
#default is axis = 0 which mean row wise but we want column wise
train_df_v2 = train_df.drop(['PassengerId', 'Name', 'Cabin'], axis = 1)

#fillna: We can also propagate non-null values forward or backward.
train_df_v2.Embarked.fillna(method='ffill', inplace=True)
train_df_v2.isnull().sum()


# In[ ]:


#Age and Pclass are correlated so we estimate the missing Age based on Pclass
def predict_age(row_age_pclass):
    age = row_age_pclass[0]
    pclass = row_age_pclass[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 29
        else:
            return 25
    else:
        return age
    
#apply get a column and apply a change row by row
train_df_v3 = train_df_v2.copy()
train_df_v3['Age'] = train_df_v3[['Age', 'Pclass']].apply(predict_age, axis=1) 
train_df_v3.isnull().sum()


# In[ ]:


#converting categorical features to numerical
#Encode target labels with value between 0 and (n_classes-1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
object_cols_train = train_df_v3.select_dtypes('object').columns
train_df_v4 = train_df_v3.copy()
train_df_v4[object_cols_train] = train_df_v4[object_cols_train].apply(le.fit_transform) 
train_df_v4.info()


# In[ ]:


train_df_v4.corr().Survived.abs().sort_values()


# **Cleaning the Test Data**

# In[ ]:


test_df_v2 = test_df.drop(['Name', 'Cabin'], axis = 1)

#fillna: We can also propagate non-null values forward or backward.
test_df_v2.Fare.fillna(method='ffill', inplace=True)
#apply get a column and apply a change row by row
test_df_v3 = test_df_v2.copy()
test_df_v3['Age'] = test_df_v3[['Age', 'Pclass']].apply(predict_age, axis=1) 
test_df_v3.isnull().sum()


# In[ ]:


#converting categorical features to numerical
#Encode target labels with value between 0 and (n_classes-1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
object_cols_test = test_df_v3.select_dtypes('object').columns
test_df_v4 = test_df_v3.copy()
test_df_v4[object_cols_test] = test_df_v4[object_cols_test].apply(le.fit_transform) 
test_df_v4.info()


# In[ ]:


X = train_df_v4.drop('Survived', axis=1)
y = train_df_v4.Survived
#Scaling: standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)


# **Split the data**

# In[ ]:


from sklearn.model_selection import train_test_split
#split the training data t0 70%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# **Logistic Regression**

# In[ ]:


# Disabling warnings
import warnings
warnings.simplefilter("ignore")

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
#train
logmodel.fit(X_train, y_train)
#predict
logmodel_predict = logmodel.predict(X_test)
#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, logmodel_predict))
print(confusion_matrix(y_test, logmodel_predict))


# In[ ]:


#cross validation
from sklearn.model_selection import cross_val_score
CVS = cross_val_score(logmodel, X, y, scoring='accuracy', cv=5)
print(CVS)
print("\nMean accuracy of cross-validation: ", CVS.mean())


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#train
randmodel = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)
randmodel.fit(X_train, y_train)
#predict
randmodel_predict = randmodel.predict(X_test)
#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, randmodel_predict))
print(confusion_matrix(y_test, randmodel_predict))


# In[ ]:


#cross validation
from sklearn.model_selection import cross_val_score
CVS = cross_val_score(randmodel, X, y, scoring='accuracy', cv=5)
print(CVS)
print("\nMean accuracy of cross-validation: ", CVS.mean())


# **Neural Network**

# In[ ]:


X_train_nn = X_train.values
y_train_nn = y_train.values
X_test_nn = X_test.values
y_test_nn = y_test.values


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
nn_model = Sequential()
#add layer
nn_model.add(Dense(units=8, activation='relu')) #unit is the # nodes which is roughly # of features
nn_model.add(Dropout(0.5)) #prevent NN from overfitting by disabling 50% of the activation nodes
#add layer
nn_model.add(Dense(units=4, activation='relu'))
nn_model.add(Dropout(0.5)) 
#add layer (final layer)
nn_model.add(Dense(units=1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam')
#if model not working stop before the iterations (epochs)
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#train
nn_model.fit(x=X_train_nn, y=y_train_nn, epochs=200, validation_data=(X_test_nn, y_test_nn), verbose=1, callbacks=[early_stop])


# In[ ]:


nn_model_loss = pd.DataFrame(nn_model.history.history)
nn_model_loss.plot()


# In[ ]:


#predict
nn_model_predict = nn_model.predict_classes(X_test_nn)
#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test_nn, nn_model_predict))
print(confusion_matrix(y_test_nn, nn_model_predict))


# **SVC**

# In[ ]:


from sklearn.svm import SVC
svc_model = SVC()
#train
svc_model.fit(X_train, y_train)
#predict
svc_predict = svc_model.predict(X_test)
#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, svc_predict))
print(confusion_matrix(y_test, svc_predict))


# **Pipeline**
