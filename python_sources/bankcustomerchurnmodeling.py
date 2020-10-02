#!/usr/bin/env python
# coding: utf-8

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


bankdata = pd.read_csv("../input/Churn_Modelling.csv")


# In[ ]:


bankdata.head(10)


# In[ ]:


bankdata.isna().sum()


# In[ ]:


bankdata.CustomerId.unique().size


# In[ ]:


bankdata.info()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bankdata.Exited.value_counts()
plt.title = "Exited Class Histogram"
plt.xlabel = "Exited"
plt.ylabel = "Frequency"
pd.value_counts(bankdata['Exited']).plot.bar()


#         Data Looks clean, but there is a class imbalance. 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bankdata.Geography.value_counts()
plt.title = "Exited Class Histogram"
plt.xlabel = "Geography"
plt.ylabel = "Frequency"
pd.value_counts(bankdata['Geography']).plot.bar()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bankdata.Gender.value_counts()
plt.title = "Gender Class Histogram"
plt.xlabel = "Gender"
plt.ylabel = "Frequency"
pd.value_counts(bankdata['Gender']).plot.bar()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bankdata.IsActiveMember.value_counts()
plt.title = "IsActiveMember Class Histogram"
plt.xlabel = "IsActiveMember"
plt.ylabel = "Frequency"
pd.value_counts(bankdata['IsActiveMember']).plot.bar()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bankdata.HasCrCard.value_counts()
plt.title = "HasCrCard Histogram"
plt.xlabel = "HasCrCard"
plt.ylabel = "Frequency"
pd.value_counts(bankdata['HasCrCard']).plot.bar()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bankdata.Surname.value_counts()
plt.title = "Surname Histogram"
plt.xlabel = "Surname"
plt.ylabel = "Frequency"
pd.value_counts(bankdata['Surname']).plot.bar()


# In[ ]:


bankdata.describe()


# In[ ]:


bankdata.corr()


# In[ ]:


import seaborn as sn
correlationmat = bankdata.corr(method='pearson')
f, ax = plt.subplots(figsize = (10,10))
sn.heatmap(correlationmat, vmax=0.8, square=True, annot=True)


# In[ ]:


bankdata.drop(columns=['RowNumber','CustomerId', 'Surname'], axis= 1, inplace= True)
bankdata.IsActiveMember.value_counts()


# In[ ]:


plot_data = bankdata[['CreditScore', 'Age', 'Tenure',
                     'Balance','NumOfProducts','EstimatedSalary']]
grid = sn.pairplot(data = plot_data, size = 3)


# There is no correlation between features and target

# In[ ]:


# box and whisker plots to check outliers
bankdata.plot(kind='box', subplots=True, layout=(4,4), fontsize=8, figsize=(14,14))
plt.show()


# In[ ]:


print("Before",bankdata.shape)
def outlier(col): 
            q3 = bankdata[col].quantile(0.75) 
            q1 = bankdata[col].quantile(0.25) 
            iqr = q3 - q1 
            lowval = q1 - 1.5* iqr 
            highval = q3 + 1.5 * iqr 
            loc_ret = bankdata.loc[(bankdata[col] > lowval) & (bankdata[col] < highval)] 
            return loc_ret      
numeric_subset = bankdata.select_dtypes('number') 

bankdata = outlier('Age')
bankdata = outlier('CreditScore')
bankdata = outlier('NumOfProducts')
print("After",bankdata.shape)


# In[ ]:


bankdata.head(5)


# In[ ]:


Y=bankdata[['Exited']]
X= bankdata.drop('Exited', axis=1)


#  Normalize the train and test data (2.5 points)

# In[ ]:


from sklearn.preprocessing import LabelEncoder
lencoder = LabelEncoder()
for i in range(0,X.shape[1]):
    if X.dtypes[i]=='object':
        X[X.columns[i]] = lencoder.fit_transform(X[X.columns[i]])


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Scaled_X= pd.DataFrame(ss.fit_transform(X), columns=X.columns)


# In[ ]:


Scaled_X.head(5)


# Divide the data set into Train and test sets, Apply smote to resolve the issue of class imbalnce

# In[ ]:


from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Scaled_X, Y, test_size=0.2, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[ ]:


#print("Before OverSampling, counts of label '1': {}".format(sum(y_train[:1]==1)))
#print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


#  Initialize & build the model (10 points)

# In[ ]:


#y_train_res1.shape
X_train_res1 = pd.DataFrame(X_train_res)
y_train_res1 = pd.DataFrame(y_train_res)


# In[ ]:


y_train_res1.head(5)


# In[ ]:


import keras 
model = keras.models.Sequential()
model.add(keras.layers.Dense(5, input_dim=10, activation='relu'))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], batch_size=20, show_accuracy=True, validation_split=0.2, verbose = 2)


# In[ ]:


model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train_res1, y_train_res1, epochs=20, batch_size=10)


# In[ ]:


predict = model.predict(X_test)
predict = predict> 0.5


# In[ ]:


from sklearn.metrics import confusion_matrix
matrix1 =confusion_matrix(y_test, predict)


# In[ ]:


matrix1


# In[ ]:


accuracy = (matrix1[0,0]+ matrix1[1,1])/(matrix1[0,0]+ matrix1[1,1] + matrix1[0,1]+ matrix1[1,0])
accuracy


# In[ ]:


from keras.optimizers import SGD
model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(5, input_dim=10, activation='relu'))
model2.add(keras.layers.BatchNormalization())
model2.add(keras.layers.Dense(10, activation='relu'))
model2.add(keras.layers.BatchNormalization())
model2.add(keras.layers.Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum = 0.9)
model2.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model2.fit(X_train_res1, y_train_res1, epochs=10, batch_size=10)


# In[ ]:


predict1 = model2.predict(X_test)
predict1 = predict1> 0.5


# In[ ]:


from sklearn.metrics import confusion_matrix
matrix2 =confusion_matrix(y_test, predict1)


# In[ ]:


matrix2


# In[ ]:


accuracy = (matrix2[0,0]+ matrix2[1,1])/(matrix2[0,0]+ matrix2[1,1] + matrix2[0,1]+ matrix2[1,0])
accuracy


# In[ ]:





# In[ ]:





# In[ ]:




