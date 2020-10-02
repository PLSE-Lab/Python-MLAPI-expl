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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
da=pd.read_csv("../input/indian-liver-patient-records/indian_liver_patient.csv")
da.head()


# In[ ]:


da.sample(10)


# In[ ]:


da.info()
da.head(5)


# In[ ]:


#Describe gives statistical information about NUMERICAL columns in the dataset
da.describe(include='all')


# In[ ]:


da.columns
#showing columns


# In[ ]:



da.isnull().sum()
#finding null values


# In[ ]:


#finding frequency of items in a column mssm
sns.countplot(x='Dataset',data=da,label="count",palette="Set3")
#number of patients with liver disease is 1#n


# In[ ]:





# In[ ]:


da['Gender'].value_counts()


# In[ ]:


#number of patients
sns.countplot(x='Gender',data=da,label='count',palette="Set3")


# In[ ]:


da[['Gender','Age','Dataset']].groupby(['Dataset','Gender']).count()


# In[ ]:


da[['Gender','Age','Dataset']].groupby(['Dataset','Gender']).mean()


# In[ ]:


sns.pairplot(da,hue="Dataset",height=2.5)


# In[ ]:


p=sns.FacetGrid(da,row="Dataset",col="Gender",margin_titles=True)
p.map(plt.hist,"Age",color="Green")


# DISEASE PREDICTION USING MACHINE LEARNING

# In[ ]:


#using decisiontree

 


# In[ ]:


da.head()


# In[ ]:





# In[ ]:


plt.hist(da['Alkaline_Phosphotase'])

#label encoding before scaling to convert categotrical  to numerical values Here is GEnder column

#gender is two value binary we use Label Binarizer
from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()

da['Gender']=lb.fit_transform(da['Gender'])


da.head()


# In[ ]:


#input and output dataset

from sklearn.preprocessing import MinMaxScaler,StandardScaler

X=da.drop('Dataset',axis=1)
y_output=da['Dataset']







# In[ ]:



avgAGR=X['Albumin_and_Globulin_Ratio'].mean()


#filling na value with mean
X['Albumin_and_Globulin_Ratio'].fillna(avgAGR,inplace=True)
X['Albumin_and_Globulin_Ratio'].isnull().any()
avgAGR


# In[ ]:


#standard scalar

ss=StandardScaler()
df=pd.DataFrame(ss.fit_transform(X),columns=X.columns,index=X.index)
df.head()
df.loc[2]


# In[ ]:


#MACHINE LEARNING USING RANDOM TREE
tree=DecisionTreeClassifier()
tree.fit(df,y_output)
#print(df.shape)
#print(pd.DataFrame(df.iloc[1,:]).shape)
predict=tree.predict(df)

print(classification_report(y_output,predict))


# In[ ]:


#machine learning using random forest

