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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[ ]:


#importing the ILPD data
liverdata = pd.read_csv('../input/indian_liver_patient.csv')


# In[ ]:


datatype=liverdata.dtypes.index
datatype


# In[ ]:


liverdata.shape
liverdata


# In[ ]:


liverdata.info()


# In[ ]:


liverdata.isnull().values.any()


# In[ ]:


liverdata.isnull().sum()


# In[ ]:


#albumin_and_globulin_ratio has null values
liverdata['Albumin_and_Globulin_Ratio'].isnull().sum()


# In[ ]:


liverdata['Albumin_and_Globulin_Ratio'].mean()


# In[ ]:


liverdata["Albumin_and_Globulin_Ratio"].fillna(liverdata['Albumin_and_Globulin_Ratio'].mean(), inplace = True)


# In[ ]:


liverdata.isnull().sum()


# In[ ]:


plt.subplots(figsize=(10,10))
sns.heatmap(liverdata.corr(),linewidths=0.5,linecolor='black',vmax=1.0,square=True,cmap="YlGnBu",annot=True)
plt.title('Correlation of Liver Disease Features')
plt.show()


# In[ ]:


## output value has '1' for liver disease and '2' for no liver disease so making it 0 for no disease

def partition(x):
    if x == 2:
        return 0
    return 1

liverdata['Dataset'] = liverdata['Dataset'].map(partition)


liverdata.head(250)


# In[ ]:


## output value has '1' for liver disease and '2' for no liver disease so making it 0 for no disease 

def partition(x):
    if x =='Male':
        return 0
    return 1

liverdata['Gender'] = liverdata['Gender'].map(partition)


liverdata.head(250)


# In[ ]:


count_class_0, count_class_1 = liverdata['Dataset'].value_counts()

# Divide by class
data_class0 = liverdata[liverdata['Dataset'] == 0]
data_class1 = liverdata[liverdata['Dataset'] == 1]

liverdata = pd.concat([data_class0,data_class1.head(250)], axis=0)
liverdata


# In[ ]:


liverdata.shape


# In[ ]:


X = liverdata.drop('Dataset',axis=1)
Y = liverdata['Dataset']


# In[ ]:


sns.countplot(data=liverdata, x = 'Dataset', label='Count')

LD,NLD = liverdata['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)


# In[ ]:


liverdata.hist(bins=11,figsize=(12,12))
plt.show()


# In[ ]:


# split data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 99)


#scaling of the data using min-max scaler
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


#Libraries to Build Ensemble Model : Random Forest Classifier 
# Create the parameter grid based on the results of random search 

params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

# Performing CV to tune parameters for best SVM fit 
svm_model = GridSearchCV(SVC(kernel="rbf", probability=True, C=0.1, gamma=0.001), params_grid, cv=5)
svm_model.fit(X_train, Y_train)

svm_model.fit(X_train,Y_train)


# In[ ]:


Y_predict = svm_model.predict(X_test)
cm = confusion_matrix(Y_test,Y_predict)


# In[ ]:


cm = np.array(confusion_matrix(Y_test,Y_predict,labels=[1,0]))

confusion=pd.DataFrame(cm, index=['is_Liver_Disease', 'No_Liver_Disease'], columns=['Prediction_of_Disease', 'Prediction_of_Healthy']) 
confusion


# In[ ]:


print(classification_report(Y_test,Y_predict))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
accuracy = accuracy_score(Y_test,Y_predict)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.convolutional import Convolution2D
from keras.utils import to_categorical

# Initialising the ANN
myliverclassifier = Sequential() 
myliverclassifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
#hidden layer
myliverclassifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
myliverclassifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#output layer
myliverclassifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compile ANN
myliverclassifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the dat
history =myliverclassifier.fit(X_train, Y_train, batch_size = 20, epochs =17 )


# In[ ]:


y_predict = myliverclassifier.predict(X_test)

y_predict = [ 1 if y>=0.5 else 0 for y in y_predict ]


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
accuracy = accuracy_score(Y_test,Y_predict)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


liverdata=pd.read_csv('../input/indian_liver_patient.csv')


# In[ ]:


datatype=liverdata.dtypes.index
datatype


# In[ ]:


liverdata.shape
liverdata


# In[ ]:


liverdata["Albumin_and_Globulin_Ratio"].fillna("0.6", inplace = True)
liverdata['Gender']=liverdata.Gender.map(dict(Female=0,Male=1))
liverdata


# In[ ]:


sns.countplot(data=liverdata, x = 'Dataset', label='Count')

LD,NLD = liverdata['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)


# In[ ]:


liverdata_sex = pd.get_dummies(liverdata['Gender'])
liverdata_new = pd.concat([liverdata, liverdata_sex], axis=1)
liverdata_gender = liverdata_new.drop(labels=['Gender' ],axis=1 )
liverdata_gender.columns = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','1','0','Dataset']

X = liverdata_gender.drop('Dataset',axis=1)
Y = liverdata_gender['Dataset']


# split data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 99)


#scaling of the data using min-max scaler
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


#Libraries to Build Ensemble Model : Random Forest Classifier 
# Create the parameter grid based on the results of random search 

params_grid = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
               {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

# Performing CV to tune parameters for best SVM fit 
svm_model = GridSearchCV(SVC(kernel="rbf", probability=True, C=0.1, gamma=0.001), params_grid, cv=5)
svm_model.fit(X_train, Y_train)


#model = SVC(kernel="rbf", probability=True, C=0.1, gamma=0.001, max_iter=10000)
#svm_model = GridSearchCV(model, parameters, n_jobs=-1, cv=3)

svm_model.fit(X_train,Y_train)


# In[ ]:


Y_predict = svm_model.predict(X_test)
cm = confusion_matrix(Y_test,Y_predict)

cm = np.array(confusion_matrix(Y_test,Y_predict,labels=[1,0]))

confusion=pd.DataFrame(cm, index=['is_Liver_Disease', 'No_Liver_Disease'], columns=['Prediction_of_Disease', 'Prediction_of_Healthy']) 
confusion


# In[ ]:


print(classification_report(Y_test,Y_predict))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
accuracy = accuracy_score(Y_test,Y_predict)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


myliverclassifier = Sequential() # Initialising the ANN
#input layer
myliverclassifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#hidden layer
myliverclassifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
myliverclassifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#output layer
myliverclassifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compile ANN
myliverclassifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the dat
history =myliverclassifier.fit(X_train, Y_train, batch_size = 20, epochs = 16)


# In[ ]:


plt.style.use("ggplot")
plt.figure()
N = 16
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")


# In[ ]:


y_predict = myliverclassifier.predict(X_test)
y_predict = [ 1 if y>=0.5 else 0 for y in y_predict ]


# In[ ]:


print(classification_report(Y_test, y_predict))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
accuracy = accuracy_score(Y_test,y_predict)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:




