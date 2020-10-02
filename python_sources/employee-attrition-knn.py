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
import numpy as np
import matplotlib.pyplot as  plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import pandas_profiling


# In[ ]:


df = pd.read_csv("../input/HR-Employee-Attrition.csv")


# In[ ]:


df.head()


# In[ ]:



pandas_profiling.ProfileReport(df)


# In[ ]:


num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns


# In[ ]:


df[cat_cols].apply(lambda x:print(x.value_counts()))


# In[ ]:


df.OverTime.replace({"Yes":1,"No":0},inplace = True)


# In[ ]:


df.Over18.replace({"Y":1},inplace = True)


# In[ ]:


df_onehot = pd.get_dummies(df[cat_cols.drop(["Attrition","Over18","OverTime"])])


# In[ ]:


df_onehot.head()


# In[ ]:


df_final = pd.concat([df_onehot,df[num_cols],df["Attrition"],df["Over18"],df["OverTime"]],axis = 1)


# In[ ]:


df_final.head()


# In[ ]:


X = df_final.drop(columns=['Attrition'])
X.head()


# In[ ]:


Y = df_final[['Attrition']]


# In[ ]:


Y.head()


# In[ ]:


from sklearn import preprocessing


# In[ ]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))


# In[ ]:


X


# In[ ]:


from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state= 42)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
from math import sqrt
from sklearn import metrics


# In[ ]:


length = round(sqrt(df.shape[0])) 
length


# In[ ]:


accuracy_dict = {}
accuracy_list = []
for k in range(1,length+1):
    model = KNeighborsClassifier(n_neighbors = k,weights='uniform', algorithm='auto').fit(X_train,Y_train)
    Y_predict = model.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test,Y_predict)
    accuracy_dict.update({k:accuracy})
    accuracy_list.append(accuracy)
    print("Accuracy ---> k = {} is {}" .format(k,accuracy))


# In[ ]:


key_max = max(accuracy_dict.keys(), key=(lambda k: accuracy_dict[k]))

print( "The Accuracy value is ",accuracy_dict[key_max], "with k= ", key_max)


# In[ ]:


elbow_curve = pd.DataFrame(accuracy_list,columns = ['accuracy'])


# In[ ]:


elbow_curve.plot()

