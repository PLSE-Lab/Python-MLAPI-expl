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


data = pd.read_csv("../input/hivprogression/training_data.csv")


# In[ ]:


data.head()


# In[ ]:


list(data)


# In[ ]:


data = data[['PatientID', 'Resp', 'VL-t0', 'CD4-t0']]


# In[ ]:


data.shape


# In[ ]:


data.isna().sum()


# No missing Values 

# In[ ]:


data.head()


# In[ ]:


data['Resp'].value_counts()


# We can see there is class imbalance in data.

# In[ ]:


data['Resp'].nunique()


# Check number of unique entries and what are the entries.

# In[ ]:


data['Resp'].unique()


# Matrix plot of data excluding the Patient Id column to check the correlation between the variables. 

# In[ ]:


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
scatter_matrix(data.drop(['PatientID'],axis = 1))
plt.show()


# There is no relation between the variables.

# In[ ]:


plt.plot(data['Resp'],data['CD4-t0'],'bo')
plt.xlabel('Resp')
plt.ylabel('CD4-t0')
plt.show()


# In[ ]:


data.boxplot(by = 'Resp',column = ['CD4-t0'],grid = False)


# Some outliers are present in the data. Patients which are in class 0 are having high value of CD4-t0 and those are in class 1 are having low values of CD4-t0.

# In[ ]:


plt.plot(data['Resp'],data['VL-t0'],'bo')
plt.xlabel('Resp')
plt.ylabel('VL-t0')
plt.show()


# In[ ]:


data.boxplot(by = 'Resp',column = ['VL-t0'],grid = False)


# One outliers is present in the data. Patients which are in class 0 are having low value of VL-t0 and those are in class 1 are having high values of VL-t0.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X = data[['VL-t0','CD4-t0']].values
Y = data[['Resp']].values


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state =2 )


#  **Logistic Regression**

# In[ ]:


model = LogisticRegression()
model.fit(X_train, Y_train)


# In[ ]:


prediction = model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report 
print(classification_report(Y_test, prediction))


# We can see that though accuracy of the model is good which is 82% but our model is not prforming well. As precision and recall for class 1 is very poor.

# **K Nearest Neighbour**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[ ]:


print(classification_report(Y_test, y_pred))


# We can see that though accuracy of the model is good which is 82% but our model is not prforming well. As precision and recall for class 1 is very poor.

# Precision Recall Curve

# In[ ]:


y_score_rf = model.predict_proba(X_test)[:,-1]


# In[ ]:


from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve
average_precision = average_precision_score(Y_test, y_score_rf)

print('Average precision-recall score RF: {}'.format(average_precision))


# Average precision-recall score is 0.38 which is very low.

# In[ ]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(Y_test, y_score_rf)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))


# As there is class imbalance in the data we will use SMOTE (Synthetic Minority Oversampling Technique) and ADASYN (Adaptive synthetic sampling approach) to balanced the classes.

# In[ ]:


# importing SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter
# applying SMOTE to our data and checking the class counts
X_resampled, y_resampled = SMOTE().fit_resample(X, Y)
print(sorted(Counter(y_resampled).items()))


# In[ ]:


X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_resampled,y_resampled,test_size = 0.2, random_state =2 )


# In[ ]:


model.fit(X1_train, Y1_train)


# In[ ]:


pred1 = model.predict(X1_test)


# In[ ]:


print(classification_report(Y1_test, pred1))


# We can see that now models accuracy and recall both are good. Our model is performing bettter.

# In[ ]:


from imblearn.over_sampling import ADASYN
from collections import Counter
# applying SMOTE to our data and checking the class counts
X_resampled1, y_resampled1 = ADASYN().fit_resample(X, Y)
print(sorted(Counter(y_resampled1).items()))


# In[ ]:


X2_train,X2_test,Y2_train,Y2_test = train_test_split(X_resampled1,y_resampled1,test_size = 0.2, random_state =2 )


# In[ ]:


model.fit(X2_train, Y2_train)


# In[ ]:


pred2 = model.predict(X2_test)


# In[ ]:


print(classification_report(Y2_test, pred2))


# We can see that now models accuracy and recall both are good. Our model is performing bettter.

# In[ ]:


y_score_rf2 = model.predict_proba(X2_test)[:,-1]
average_precision1 = average_precision_score(Y2_test, y_score_rf2)

print('Average precision-recall score RF: {}'.format(average_precision1))


# In[ ]:


precision, recall, _ = precision_recall_curve(Y2_test, y_score_rf2)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision1))


# In[ ]:




