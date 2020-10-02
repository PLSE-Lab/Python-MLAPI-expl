#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/column_2C_weka.csv")


# In[ ]:


data.head()


# In[ ]:


data["class"].unique()


# In[ ]:


data["class"].value_counts()


# In[ ]:


data.info()


# In[ ]:





# In[ ]:


Abnormal =data[data["class"] == 'Abnormal']
Normal = data[data["class"] == 'Normal']


# In[ ]:


Abnormal.drop(["class"],axis = 1,inplace = True)
Normal.drop(["class"],axis = 1,inplace = True)


# In[ ]:


Abnormal = (Abnormal-np.min(Abnormal))/(np.max(Abnormal)-np.min(Abnormal))
Normal = (Normal-np.min(Normal))/(np.max(Normal)-np.min(Normal))


# # Scatter Plot

# In[ ]:


plt.figure(figsize=(15,10))
plt.scatter(Abnormal.pelvic_incidence,Abnormal["pelvic_tilt numeric"],color = 'red',label = 'Abnormal')
plt.scatter(Normal.pelvic_incidence,Normal["pelvic_tilt numeric"],color = 'green',label = 'Normal')
plt.xlabel('pelvic_incidence')
plt.ylabel('pelvic_tilt numeric')
plt.legend()
plt.show()


# In[ ]:


#data["class"] = [0 if i == "Abnormal" else 1 for i in data["class"]]
data["class"].replace(["Abnormal","Normal"],[0,1],inplace = True)


# In[ ]:


y = data["class"].values
x_data = data.drop(["class"],axis = 1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# # Split data

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)


# In[ ]:


# KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)


# In[ ]:


print("{} knn score: {}".format(22,knn.score(x_test,y_test)))


# In[ ]:


score =[]
for i in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score.append(knn2.score(x_test,y_test))
plt.figure(figsize=(15,10))
plt.plot(range(1,30),score)
plt.xlabel("knn value")
plt.ylabel("Score")
plt.show()


# # Conclusion

# The highest knn score is obtained when the n neighbor value was 22
