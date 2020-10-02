#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


df.head()


# In[ ]:


df.notnull().sum()


# In[ ]:


df.info()


# In[ ]:


sns.pairplot(df)


# In[ ]:


plt.pyplot.figure(figsize=(16, 16))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


sns.boxplot(x='Outcome', y='Glucose', data=df)


# In[ ]:


X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model = svm.SVC(C=2, gamma='scale', kernel='linear')
model.fit(X_train, y_train)
prediction = model.predict(X_test)


# In[ ]:


print(metrics.accuracy_score(prediction, y_test) * 100)


# In[ ]:


print(confusion_matrix(y_test, prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[ ]:




