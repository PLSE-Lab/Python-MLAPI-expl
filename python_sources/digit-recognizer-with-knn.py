#!/usr/bin/env python
# coding: utf-8

# # digit recognizer with KNN _ He Li

# # import lib

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# # import train data and test data

# In[ ]:


def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
    X_train = train.values[:train_row, 1:]
    y_train = train.values[:train_row, 0]
    
    Pred_test = pd.read_csv(data_dir + "test.csv").values
    Pred_test = Pred_test[:train_row]
    
    return X_train, y_train, Pred_test

train_row = 5000
data_dir = "../input/"
Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)
print(Origin_X_train.shape, Origin_y_train.shape, Origin_X_test.shape)


# # show the train in pic

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt

row = 7
print (Origin_y_train[row])
plt.imshow(Origin_X_train[row].reshape(28, 28))
plt.show()


# # split the train to train and vali with test_size = 0.3

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train, Origin_y_train, test_size = 0.3, random_state = 0)
print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)


# # find the best k from (1, 8)

# In[ ]:


import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0

k_range = range(1, 8)
scores = []

for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_vali)
    accuracy = accuracy_score(y_vali,y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, y_pred))  
    print(confusion_matrix(y_vali, y_pred))  
    
    print("Complete time: " + str(end-start) + " Secs.")


# # print accuracy and find the best k

# In[ ]:


print (scores)
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()


# # predict the test
# 

# In[ ]:


k = 3

model = KNeighborsClassifier(n_neighbors = k)
model.fit(Origin_X_train,Origin_y_train)
y_pred = model.predict(Origin_X_test[:300])


# # check the result

# In[ ]:


row = 250
print (y_pred[row])
plt.imshow(Origin_X_test[row].reshape((28, 28)))
plt.show()


# 
# * ImageId	Label
# * 1	2
# * 2	0
# * 3	9
# * 4	9

# In[ ]:


print(len(y_pred))

# save submission to csv
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recognicer_Result.csv', index=False,header=True)


# In[ ]:




