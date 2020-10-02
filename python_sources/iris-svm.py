#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import csv as csv 

from sklearn import svm
import warnings 
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

csv_file_object = csv.reader(open('../input/Iris.csv'))
header = next(csv_file_object)

data=[] 
for row in csv_file_object:
    data.append(row) 
    
data = np.array(data)

print(header)
print(data[0])


# In[ ]:


data[0::,5:6] = (data[0::,5:6]=='Iris-setosa') + 2*(data[0::,5:6]=='Iris-versicolor')+3*(data[0::,5:6]=='Iris-virginica')
data= data[0::,1:6]

data = data.astype(np.float32, copy=False)


# In[ ]:


X_train=np.zeros((90,4))
Y_train=np.zeros((90,1))

X_CV=np.zeros((30,4))
Y_CV=np.zeros((30,1))

X_test=np.zeros((30,4))
Y_test=np.zeros((30,1))


# In[ ]:


X_train[0:30,0::]=data[0:30,0:4]
X_train[30:60,0::]=data[50:80,0:4]
X_train[60:90,0::]=data[100:130,0:4]

Y_train[0:30]=data[0:30,4:5]
Y_train[30:60]=data[50:80,4:5]
Y_train[60:90]=data[100:130,4:5]

randomize = np.arange(len(Y_train))
np.random.shuffle(randomize)
X_train = X_train[randomize]
Y_train = Y_train[randomize]
print('Training Data Size')
print(X_train.shape)
print(Y_train.shape)


# In[ ]:


X_CV[0:10,0::]=data[30:40,0:4]
X_CV[10:20,0::]=data[80:90,0:4]
X_CV[20:30,0::]=data[130:140,0:4]

Y_CV[0:10]=data[30:40,4:5]
Y_CV[10:20]=data[80:90,4:5]
Y_CV[20:30]=data[130:140,4:5]

randomize = np.arange(len(Y_CV))
np.random.shuffle(randomize)
X_CV = X_CV[randomize]
Y_CV = Y_CV[randomize]
print('CV Data Size')
print(X_CV.shape)
print(Y_CV.shape)


# In[ ]:


X_test[0:10,0::]=data[40:50,0:4]
X_test[10:20,0::]=data[90:100,0:4]
X_test[20:30,0::]=data[140:150,0:4]

Y_test[0:10]=data[40:50,4:5]
Y_test[10:20]=data[90:100,4:5]
Y_test[20:30]=data[140:150,4:5]

randomize = np.arange(len(Y_test))
np.random.shuffle(randomize)
X_test = X_test[randomize]
Y_test = Y_test[randomize]
print('Test Data Size')
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


C=np.array([0.001, 0.003, 0.005, 0.01 ,0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50])
accuracy=0.0
acc_max=0.0
C_optimal=0.0

print('Finding Optimal C......')
for i in range(15):
    clf = svm.LinearSVC(C=C[i])
    clf.fit(X_train, Y_train)  
    accuracy= (np.sum((clf.predict(X_CV))==Y_CV.T))*10/3
    print('C :%f ->Accuracy :%f'%(C[i],accuracy))
    
    if accuracy>acc_max:
        acc_max=accuracy
        C_optimal=C[i]
  
print('Optimal C =%f'%C_optimal)


# In[ ]:


clf = svm.LinearSVC(C=1)
clf.fit(X_train, Y_train)
accuracy= (np.sum((clf.predict(X_test))==Y_test.T))*10/3
print('Iris-setosa:1 ,Iris-versicolor:2 ,Iris-virginica:3')
print('Prediction:')
print(clf.predict(X_test))
print('Expected:')
print(Y_test.T)
print('Accuracy=%f'%accuracy)

