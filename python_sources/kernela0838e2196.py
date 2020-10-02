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


dataset=pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


sns.countplot(dataset['target_class'],label='count')


# In[ ]:


dataset.hist(bins=10,figsize=(20,15))
plt.show()


# In[ ]:


dataset.corr()


# In[ ]:


sns.pairplot(data=dataset,
             palette="husl",
             hue="target_class",
             vars=[" Mean of the integrated profile",
                   " Excess kurtosis of the integrated profile",
                   " Skewness of the integrated profile",
                   " Mean of the DM-SNR curve",
                   " Excess kurtosis of the DM-SNR curve",
                   " Skewness of the DM-SNR curve"])

plt.suptitle("PairPlot of Data Without Std. Dev. Fields",fontsize=18)

plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(20,16))
sns.heatmap(data=dataset.corr(),annot=True)
plt.title('Co-Relation Mattrix')
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix,classification_report


# In[ ]:


X=dataset.drop('target_class',axis=1)
y=dataset['target_class']


# In[ ]:


model=DecisionTreeClassifier()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


pred=model.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


r2_score(y_test,pred)


# In[ ]:


accuracy_score(y_test,pred)*100


# In[ ]:


df=pd.DataFrame({'Actual Pred':y_test,'Predicted ':pred})
df1=df.head(25)
print(df1)


# In[ ]:


df1.plot(kind='bar',figsize=(20,5))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


print('Mean absolute Error',metrics.mean_absolute_error(y_test,pred))
print('Mean squared Error',metrics.mean_squared_error(y_test,pred))
print('Mean squared Error',np.sqrt(metrics.mean_absolute_error(y_test,pred)))


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model2=LogisticRegression()
model2.fit(X_train,y_train)


# In[ ]:


y_pred=model2.predict(X_test)
print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


accuracy_score(y_test,y_pred)*100


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model3=LinearRegression()
model3.fit(X_train,y_train)


# In[ ]:


lin_pred=model3.predict(X_test)


# In[ ]:


r2_score(y_test,lin_pred)


# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


model3=LinearSVC(C=1000)


# In[ ]:


model3.fit(X_train,y_train)
prediction=model3.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,prediction))


# In[ ]:


print(accuracy_score(y_test,prediction)*100)


# In[ ]:


print(classification_report(y_test,prediction))


# In[ ]:




