#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
dataset = pd.read_csv("../input/nba-prediction-for-naive-bayes/nba_longevity.csv")


# In[ ]:



X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0,test_size=0.2)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(f"--------------features----------\n {X}")
print(f"---------------Y--------------- \n{Y}")
print(f"--------------REAL--------------\n {Y_test}")
print(f"-------------PREDICTED----------\n {Y_pred}")
print(f"---------CONFUSION MATRIX-------\n {cm}")

