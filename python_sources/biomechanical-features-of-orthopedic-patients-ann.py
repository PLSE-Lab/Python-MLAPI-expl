#!/usr/bin/env python
# coding: utf-8

# ### I imported the necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense , Dropout
from sklearn.metrics import confusion_matrix , roc_auc_score, roc_curve

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### I installed the data that has two different labels

# In[ ]:


two_label_dataset=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")


# ### I observed the data

# In[ ]:


two_label_dataset.head()


# ### I converted the class features to number because the machine can learn by numbers

# In[ ]:


two_label_dataset["class"] = [1 if i == "Abnormal" else 0 for i in two_label_dataset["class"]


# ### I observed the data againly

# In[ ]:


two_label_dataset.head()


# ### I had knowledge about the count of class

# In[ ]:


sns.countplot(two_label_dataset["class"])
plt.xlabel("class")
plt.ylabel("count")
plt.show()


# ### I divided the data as dependent variable and independent varieables

# In[ ]:


x=two_label_dataset.iloc[:,:-1].values
y=two_label_dataset.iloc[:,-1:].values
print("x shape :",x.shape)
print("y shape :",y.shape)


# ### I divided the data as train and test

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print("x train shape :",x_train.shape)
print("x test shape :",x_test.shape)
print("y train shape :",y_train.shape)
print("y test shape :",y_test.shape)


# ### The formula : The standard score of a sample x is calculated as:
# 
# z = (x - u) / s
# 
# where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

# In[ ]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# ### I created my model

# #                             Artificial Neural Network

# In[ ]:


model=Sequential()

model.add(Dense(26,activation="relu",input_dim=6))
model.add(Dropout(0.5))

model.add(Dense(52,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(104,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1,activation="sigmoid"))


# In[ ]:


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# ### I determined the some parameters

# In[ ]:


epochs=200
batch_size=250


# ### I started to work my model

# In[ ]:


hist=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,
              validation_data=(x_test,y_test))


# # Model Analyzing and Evaluation Error

# In[ ]:


print(hist.history.keys())


# In[ ]:


plt.plot(hist.history["loss"],color="red",label="Train Loss")
plt.plot(hist.history["val_loss"],color="green",label="Test Loss")
plt.legend()
plt.title("Loss Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Values")
plt.show()


# In[ ]:


plt.plot(hist.history["accuracy"],color="red",label="Train Accuracy")
plt.plot(hist.history["val_accuracy"],color="green",label="Test Accuracy")
plt.legend()
plt.title("Acuracy Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Values")
plt.show()


# In[ ]:


y_predict=model.predict(x_test)
for i in range(y_predict.shape[0]):
    if y_predict[i]>0.5:
        y_predict[i] = 1
    else:
        y_predict[i] = 0


# In[ ]:


cfm=confusion_matrix(y_test,y_predict)
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=1,linecolor="black",fmt=".1f",ax=ax)
plt.xlabel("True Labels")
plt.ylabel("Predict Labels")
plt.show()


# In[ ]:


fpr,tpr,threshold=roc_curve(y_test,model.predict(x_test),pos_label=1)


# In[ ]:


print("fpr shape :",fpr.shape)
print("tpr shape :",tpr.shape)
print("threshold :",threshold.shape)


# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(fpr,color="red",label="FPR")
plt.plot(tpr,color="green",label="TPR")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
plt.show()


# In[ ]:


score=roc_auc_score(y_test,model.predict(x_test))
score


# # I applied same things

# In[ ]:


three_label_dataset=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")


# In[ ]:


three_label_dataset.head()


# In[ ]:


three_label_dataset["class"].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
three_label_dataset["class"]=le.fit_transform(three_label_dataset["class"])
"""
hernia = 0
spondylolisthesis = 1
normal = 2
"""


# In[ ]:


three_label_dataset.head()


# In[ ]:


x=three_label_dataset.iloc[:,:-1].values
y=three_label_dataset.iloc[:,-1:].values
print("x shape :",x.shape)
print("y shape :",y.shape)


# In[ ]:


from keras.utils import to_categorical
y = to_categorical(y,num_classes=3)
print("New y shape :",y.shape)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print("x train shape :",x_train.shape)
print("x test shape :",x_test.shape)
print("y train shape :",y_train.shape)
print("y test shape :",y_test.shape)


# In[ ]:


x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[ ]:


model = Sequential()

model.add(Dense(26,activation="relu",input_dim=6))
model.add(Dropout(0.4))

model.add(Dense(52,activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(104,activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(3,activation="softmax"))


# In[ ]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


hist1=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))


# In[ ]:


plt.plot(hist1.history["loss"],color="red",label="Train Loss")
plt.plot(hist1.history["val_loss"],color="green",label="Test Loss")
plt.legend()
plt.title("Loss Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Values")
plt.show()


# In[ ]:


plt.plot(hist.history["accuracy"],color="red",label="Train Accuracy")
plt.plot(hist.history["val_accuracy"],color="green",label="Test Accuracy")
plt.legend()
plt.title("Acuracy Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Values")
plt.show()


# In[ ]:


y_predict=model.predict(x_test)
y_predict_classes=np.argmax(y_predict,axis=1)
y_true=np.argmax(y_test,axis=1)
print("y predict classes shape :",y_predict_classes.shape)
print("y true shape :",y_true.shape)


# In[ ]:


from sklearn.metrics import confusion_matrix
cfm=confusion_matrix(y_true,y_predict_classes)
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=1,linecolor="black",fmt=".1f",ax=ax)
plt.xlabel("True Labels")
plt.ylabel("Predict Labels")
plt.show()


# In[ ]:


fpr,tpr,threshold=roc_curve(y_test.reshape(y_test.shape[0]*y_test.shape[1]),y_predict.reshape(y_predict.shape[0]*y_predict.shape[1]))


# In[ ]:


print("fpr shape :",fpr.shape)
print("tpr shape :",tpr.shape)
print("threshold :",threshold.shape)


# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(fpr,color="red",label="FPR")
plt.plot(tpr,color="green",label="TPR")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
plt.show()


# In[ ]:


score=roc_auc_score(y_test,model.predict(x_test))
score

