#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/UCI_Credit_Card.csv")


# In[ ]:


df.shape


# In[ ]:


df.head(4)


# In[ ]:


sns.countplot(x='SEX',data=df)


# In[ ]:


sns.countplot(x='EDUCATION',data=df)


# In[ ]:


sns.countplot(x='MARRIAGE',data=df)


# In[ ]:


sns.countplot(x='AGE',data=df)


# In[ ]:


sns.distplot(df['AGE'])


# In[ ]:


sns.factorplot(x='AGE',y='LIMIT_BAL',data=df)


# In[ ]:


sns.boxplot(x='AGE',y='LIMIT_BAL',data=df)


# In[ ]:


df.groupby('AGE')['AGE'].count()


# In[ ]:


df.groupby('MARRIAGE')['MARRIAGE'].count()


# In[ ]:


sns.countplot(x='PAY_0',data=df)


# In[ ]:


sns.distplot(df['PAY_0'])


# In[ ]:


df['PAY_0'].describe()


# In[ ]:


sns.barplot(x='SEX', y = 'PAY_0', hue = 'MARRIAGE', data = df)


# In[ ]:


df.head(3)


# In[ ]:


corr=df.corr()
corr = (corr)
plt.figure(figsize=(18,18))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')


# In[ ]:


df['DUE_1']=df['BILL_AMT1']-df['PAY_AMT1']
df['DUE_2']=df['BILL_AMT2']-df['PAY_AMT2']
df['DUE_3']=df['BILL_AMT3']-df['PAY_AMT3']
df['DUE_4']=df['BILL_AMT4']-df['PAY_AMT4']
df['DUE_5']=df['BILL_AMT5']-df['PAY_AMT5']
df['DUE_6']=df['BILL_AMT6']-df['PAY_AMT6']


# In[ ]:


corr=df.corr()
corr = (corr)
plt.figure(figsize=(18,18))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')


# In[ ]:


data=df[['ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','DUE_1']]
lebel=df['default.payment.next.month']


# In[ ]:


sns.distplot(data['DUE_1'])


# In[ ]:


data_np=np.array(data)
data.DUE_1=np.log(data.DUE_1)
data.LIMIT_BAL=np.log(data.LIMIT_BAL)


# In[ ]:


data.DUE_1=np.nan_to_num(data.DUE_1)


# In[ ]:


data.head(2)
data=data.drop(['ID'],axis=1)


# In[ ]:


sns.distplot(data['LIMIT_BAL'])


# In[ ]:


sns.regplot(x='LIMIT_BAL',y='DUE_1',data=data)


# In[ ]:


#Train-Test split
from sklearn.model_selection import train_test_split
#label = df.pop('Class')
data_train, data_test, label_train, label_test = train_test_split(data, lebel, test_size = 0.3, random_state = 42)


# In[ ]:


#Apply Machine learning model

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(data_train, label_train)
logis_score_train = logis.score(data_train, label_train)
print("Training score: ",logis_score_train)
logis_score_test = logis.score(data_test, label_test)
print("Testing score: ",logis_score_test)


# In[ ]:


data.DUE_1=df['DUE_1']
data.LIMIT_BAL=df['LIMIT_BAL']
data.DUE_1=np.log(data.DUE_1)
data.LIMIT_BAL=np.log(data.LIMIT_BAL)
data.DUE_1=np.nan_to_num(data.DUE_1)


# In[ ]:


#decision tree
from sklearn.ensemble import RandomForestClassifier
rm = RandomForestClassifier()
rm.fit(data_train, label_train)
rm_score_train = rm.score(data_train, label_train)
print("Training score: ",rm_score_train)
rm_score_test = rm.score(data_test, label_test)
print("Testing score: ",rm_score_test)


# In[ ]:


data_train.head(3)
data.DUE_1=np.nan_to_num(data.DUE_1)


# In[ ]:


#kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(data_train, label_train)
knn_score_train = knn.score(data_train, label_train)
print("Training score: ",knn_score_train)
knn_score_test = knn.score(data_test, label_test)
print("Testing score: ",knn_score_test)


# In[ ]:


#decision tree
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(data_train, label_train)
dt_score_train = dt.score(data_train, label_train)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(data_test, label_test)
print("Testing score: ",dt_score_test)


# In[ ]:


#Model comparison
models = pd.DataFrame({
        'Model'          : ['Logistic Regression', 'Random Forest','kNN', 'Decision Tree'],
        'Training_Score' : [logis_score_train,rm_score_test, knn_score_train, dt_score_train],
        'Testing_Score'  : [logis_score_test,rm_score_test, knn_score_test, dt_score_test]
    })
models.sort_values(by='Testing_Score', ascending=False)


# In[ ]:


sns.barplot(x='Model',y='Testing_Score',data=models)

