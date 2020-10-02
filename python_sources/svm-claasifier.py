#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:



data = pd.read_csv("../input/diabetes.csv")
data.head()


# In[ ]:


data.info()


# We will read the csv file through pd.read_csv.And through head() we can see top 5 rows. There are some factors where the values cannot be zero. For example Glucose value cannot be 0 for a human. Similary BloodPressure,SkinThickness,Insulin and BMI cannot be zero for a human.

# In[ ]:


non_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for coloumn in non_zero:
    data[coloumn] = data[coloumn].replace(0,np.NaN)
    mean = int(data[coloumn].mean(skipna = True))
    data[coloumn] = data[coloumn].replace(np.NaN,mean)
    print(data[coloumn])


# In[ ]:


from sklearn.model_selection import train_test_split
X =data.iloc[:,0:8]
y =data.iloc[:,8]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0, stratify=y)
X.head()


# For data X we are taking all the rows of coloumn ranging from 0 to 7. Similary for y we are taking all the rows for the 8th coloumn.
# 
# We have train_test_split which we had imported during the start of the program and we have defined test size as 0.2 which implies out of all the data 20% will be kept aside to test the data at a later stage.

# In[ ]:


#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


from sklearn import svm
svm1 = svm.SVC(kernel='linear', C = 0.01)
svm1.fit(X_test,y_test)


# In[ ]:


y_train_pred = svm1.predict(X_train)
y_test_pred = svm1.predict(X_test)


# In[ ]:



y_test_pred


# We have araay of data but we need to evalute our model to check the accuracy. Lets start it with confusion matrix

# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix


# Lets check the confusion matrix

# In[ ]:


confusion_matrix(y_test,y_test_pred)


# We have the confusion matrix where the diagnol with 118 and 36 shows the correct value and 0,0 shows the prediction that we missed.
# 
# We will check the accuracy score

# In[ ]:


accuracy_score(y_test,y_test_pred)


# We have  accuracy score of 0.78
# 
# 

# In[ ]:


df=pd.DataFrame({'Actual':y_test, 'Predicted':y_test_pred})
df


# We created our linear model with C as 0.01. But how to ensure its the best value. Once option is to change is manually. We can assign different values and run the code one by one.but this process is very lenghty and time consuming.

# We will use grid search where we will assign different values of C and from the dictionary of value our model will tell use which is the best value for C as per the model. To do so we need to import GridsearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param = {'C':(0,0.01,0.5,0.1,1,2,5,10,50,100,500,1000)}


# In[ ]:


svm1 = svm.SVC(kernel = 'linear')


# In[ ]:


svm.grid = GridSearchCV(svm1,param,n_jobs=1,cv=10,verbose=1,scoring='accuracy')


# cv represnet cross validation. verbose is 1: represnt the amount of message we will be getting. 

# In[ ]:


svm.grid.fit(X_train,y_train)


# In[ ]:


svm.grid.best_params_


# This will give us the result of the best C value for the model

# In[ ]:


linsvm_clf = svm.grid.best_estimator_


# In[ ]:


accuracy_score(y_test,linsvm_clf.predict(X_test))


# This is the best accuracy we can get out of the above C values.

# In the similar way we can try for Kernel ='poly'. But for rbf we need to define gaama values as well.
# param = {'C':(0,0.01,0.5,0.1,1,2,5,10,50,100,500,1000)}, 'gamma':(0,0.1,0.2,2,10)
#     and with normal one value of C 
#     from sklearn import svm
# svm1 = svm.SVC(kernel='rbf',gamma=0.5, C = 0.01)
# svm1.fit(X_test,y_test)

# In[ ]:




