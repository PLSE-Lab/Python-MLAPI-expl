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


df= pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR ', 'CGPA', 'Research', 'Chance of Admit ']


# We are going to find correlation using pair plot

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df[columns])
plt.show()


# In[ ]:


df[columns].corr()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(df[columns])
z=pd.DataFrame(sc.transform(df[columns]))
z.info()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(z.iloc[:,0:6], z.iloc[:,7],test_size=0.3)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=200)
RFR.fit(x_train,y_train)
y_predict_std = RFR.predict(x_test)   


# In[ ]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_predict_std))


# In[ ]:


plt.plot(range(len(y_test)),y_test)
plt.plot(range(len(y_predict_std)),y_predict_std)


# Being a beginner,this is the best i could do, suggestions are highly appreciated,Thank you.
