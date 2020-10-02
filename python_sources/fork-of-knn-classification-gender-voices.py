#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# read  and customize  dataset
data = pd.read_csv("../input/voice.csv")
data.label = [1 if each=="male" else 0 for each  in data.label ]
y = data.label.values
x_data= data.drop(["label"],axis=1).values
#normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train and test split
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#KNN model
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors =13)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
# finding best key value
score_list=[]
for each in range(1,30):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,30),score_list)
plt.xlabel("k values")
plt.ylabel=("accuracy")
plt.show()
print("{} nn score:{}".format(13,knn.score(x_test,y_test)))

