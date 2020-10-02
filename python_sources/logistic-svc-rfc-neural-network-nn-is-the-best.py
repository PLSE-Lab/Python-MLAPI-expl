#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# 1) age
# 
# 2) sex
# 
# 3) chest pain type (4 values)
# 
# 4) resting blood pressure
# 
# 5) serum cholestoral in mg/dl
# 
# 6) fasting blood sugar > 120 mg/dl
# 
# 7) resting electrocardiographic results (values 0,1,2)
# 
# 8) maximum heart rate achieved
# 
# 9) exercise induced angina
# 
# 10) oldpeak = ST depression induced by exercise relative to rest
# 
# 11) the slope of the peak exercise ST segment
# 
# 12) number of major vessels (0-3) colored by flourosopy
# 
# 13) thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
# 
# 14) target: 0= less chance of heart attack 1= more chance of heart attack

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv")
df.head()


# In[ ]:


df["target"].value_counts() #165 positive and 138 negative


# In[ ]:


sns.heatmap(df.corr(),cmap="YlGn")
# As we can see, the cp, thalach, slope are more correlated.


# In[ ]:


from sklearn.model_selection import train_test_split
y,x = df["target"], df.drop("target", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# Now we are going to use different models for classification
# Before we really start, we can predict the performance by heuristic first : 
# 1. Logistic Regression can be a classic model for comparison.
# 2. SVM (Or SVC) is another important choice since we have a lot of important features.
# 3. Ensembled Trees is another choice: since we have a lot of noise in the data, random forest is a better choice compared to boosted trees. 
# 4. At last, we can try the neural network 

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve


# ## 1. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
# We have to determine several hypers first :
# Since data is noisy, we should use regularization. C=0.1 can enlarge the regularization.
# However, setting a grid would be nice too. 
# Panelty is default to l2, a strong regularization.
# Solver should be "liblinar" since it is a small dataset and we only have two classes
for c in [0.8,0.5,0.2,0.1,0.05,0.02,0.01]:
    logit_model = LogisticRegression(C=c, solver="liblinear")
    logit_model.fit(x_train, y_train)
    y_pred = logit_model.predict(x_test)
    print("Accuracy for c = %f is %f" %(c, accuracy_score(y_test, y_pred)))
#We get the best result for c=0.1/0.2/0.5 #We will choose 0.2 here


# ## 2. Support Vector Classifier with GridSearch

# In[ ]:


#For SVM, we should scale the data.
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
#x_train_scaled = preprocessing.StandardScaler().fit(x_train)
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf','sigmoid'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]
clf = GridSearchCV(SVC(), tuned_parameters, scoring="precision")
clf.fit(x_train, y_train)
print("Best Parameters Are")
print(clf.best_params_)


# In[ ]:


#Based on the best parameters we can get the result
y_pred = clf.predict(x_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("Accuracy score for SVC is %f" %(accuracy_score(y_test, y_pred)))
#We can an accuracy of 86.88% which is slightly better than logistic regression.


# ## 3. Random Forest

# In[ ]:


#Agian we will use grid search here
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
n_estimators = [100, 300, 500, 700, 900]
max_depth = [5, 8, 10, 15, 25, 30]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 5] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(rfc, hyperF,  verbose = 1, n_jobs = -1)
cv_rfc = gridF.fit(x_train, y_train)
cv_rfc.best_params_


# In[ ]:


rfc = RandomForestClassifier(random_state=42,max_depth=5,min_samples_leaf=5,min_samples_split=2,n_estimators=100)
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy score for RFC is %f" %(accuracy_score(y_test, y_pred)))
#Agian we get the same results as SVC
#However, for medical purpose, this result is better since FN value is smaller.


# ## 4. Neural Network

# In[ ]:


#Since we dont have much data, the key point here is to avoid overfitting the training set. 

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], activation='relu',kernel_initializer='random_normal'))
model.add(Dense(25,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(25,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))
model.compile(loss='binary_crossentropy', 
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics =['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
    patience=30, verbose=1, mode='auto', restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)


# The NN models shows better results with an accuracy of 86.89%, highest result. 
# 
# I got the result of 88.52% when using the Jupyter notebook.
