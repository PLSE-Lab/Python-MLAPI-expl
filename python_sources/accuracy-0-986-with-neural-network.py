#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import and Preprocess data
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df = pd.DataFrame(pd.read_csv('../input/data.csv'))
df['y'] = lb.fit_transform(df['diagnosis'])
df.drop(columns='Unnamed: 32',inplace=True)
df.head(5)


# In[ ]:


#df.drop(columns='Unnamed: 32',inplace=True)
df.head(5)


# In[ ]:


#Descritive statistics
df[df.columns[2:len(df.columns)-1]].describe()


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
def confusionMatrix(y_pred, y_true):
    return sns.heatmap(confusion_matrix(y_pred=y_pred,y_true=y_true),annot =True,fmt="d",cmap='GnBu')#GnBu


# In[ ]:


#import modules
from sklearn.model_selection import KFold,cross_val_predict,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier,kneighbors_graph
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split,BaseCrossValidator,cross_val_score,cross_val_predict

gb = GradientBoostingClassifier()
nb = GaussianNB()
kn = KNeighborsClassifier()
model = ExtraTreeClassifier()
pca = PCA()
svd = TruncatedSVD()
mm = MinMaxScaler()
kf = KFold(5)


X = df[df.columns[2:len(df.columns)-1]].values
y = df.y.values

X = mm.fit_transform(X)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

result = cross_val_predict(kn, X, y, cv=5)

print(accuracy_score(result, y))
confusionMatrix(result, y)


# In[ ]:


cross_val_predict(cv=5,X=X_test,y=y_test,estimator=kn)


# In[ ]:


print(accuracy_score(cross_val_predict(cv=5,X=X_test,y=y_test,estimator=kn), y_test))
confusionMatrix(cross_val_predict(cv=5,X=X_test,y=y_test,estimator=kn), y_test)


# In[ ]:


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# In[ ]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[ ]:


X = df[df.columns[2:len(df.columns)-1]].values
y = df.y.values

X = mm.fit_transform(X)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=30, activation='tanh'))
#	model.add(Dense(9, activation='tanh'))
	model.add(Dense(1, activation='softsign'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(5, shuffle=True)

results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



# In[ ]:


X = df[df.columns[2:len(df.columns)-1]].values
y = df.y.values

X = mm.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# define baseline model

# create model
model = Sequential()
model.add(Dense(8, input_dim=30, activation='relu'))
#model.add(Dense(16, activation='tanh'))
model.add(Dense(1, activation='tanh'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


estimator = KerasClassifier(build_fn=model, epochs=100, batch_size=5, verbose=0)

#cross_val_score(estimator, X_train, y_train, cv=5)
#estimator.(X_train,y_train)


#print (accuracy_score(model.predict(X_train),y_train))

model.fit(X_train,y_train,epochs=50)

history = pd.DataFrame(model.history.history)
#history.head(4)
history.plot()

print(accuracy_score(model.predict_classes(X_train),y_train))
print(accuracy_score(model.predict_classes(X_test),y_test))


# In[ ]:


history = pd.DataFrame(model.history.history)
history.head(4)
history.plot()
#help(estimator)


# In[ ]:


help(train_test_split)


# In[ ]:


result = cross_val_predict(estimator, X, y, cv=5)

print(accuracy_score(result, y))
confusionMatrix(result, y)


# In[ ]:


print(accuracy_score(cross_val_predict(cv=5,X=X_test,y=y_test,estimator=estimator), y_test))
confusionMatrix(cross_val_predict(cv=5,X=X_test,y=y_test,estimator=estimator), y_test)


# In[ ]:





# In[ ]:




