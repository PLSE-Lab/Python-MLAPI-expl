#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


df = pd.read_csv('/kaggle/input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv')
df.head()


# In[ ]:


df.head()


# In[ ]:


df_d = df[['male', 'age','currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose','TenYearCHD']]


# In[ ]:


df_d.head()


# Checking for any coorelation with the target varibale "TenYearCHD".From the heat map below we are not able find any variable that has a strong correlation with traget varibale.

# In[ ]:


sns.heatmap(df_d.corr())


# Here I am coverting few variables to categorical varibales using dummies method.And also dropping the null values.

# In[ ]:


df_data = pd.get_dummies(df_d,columns = ['currentSmoker','prevalentHyp','diabetes'])
df_data.dropna(inplace = True)


# In[ ]:


df_data.head()


# This MaxStandr mathod stnadardizes the values using Max Values.

# In[ ]:


def MaxStandr (col_list):
    for col in col_list :
        df_data[col] = df_data[col]/df_data[col].max()
        


# In[ ]:


MaxStandr(['cigsPerDay','totChol','sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])


# In[ ]:


X = df_data[['cigsPerDay','totChol',
       'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'currentSmoker_0',
       'currentSmoker_1', 'prevalentHyp_0', 'prevalentHyp_1', 'diabetes_0',
       'diabetes_1']]
y = df_data['TenYearCHD']


# Train and Test data splitting

# In[ ]:


total = X['BMI'].count()
train_count = int(round(total*0.8))
test_count = total-train_count 


# In[ ]:


X_train = X[:train_count]
Y_train = y[:train_count]
X_test = X[train_count:]
Y_test = np.array(y[train_count:])


# In[ ]:


clf = LogisticRegression(solver='saga').fit(X_train, Y_train)
y_hat = clf.predict(X_test)


# In[ ]:


cnf_matrix = metrics.confusion_matrix(Y_test,y_hat)
cnf_matrix


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# From the confusion matrix above we can see that the model is able to predict very accurately for False cases and very badly for True cases.

# In[ ]:


print("Accuracy:",metrics.accuracy_score(Y_test,y_hat))
print("Precision:",metrics.precision_score(Y_test,y_hat))
print("Recall:",metrics.recall_score(Y_test,y_hat))


# Even though the accuracy is high model is not that much perfect for classifying heart attacks.
