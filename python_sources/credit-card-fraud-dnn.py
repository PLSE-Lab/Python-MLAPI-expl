#!/usr/bin/env python
# coding: utf-8

# # Credit card fraud detection 
# In this kernel I'm trying to build an algorythm that detect if a transaction made with a credit card is fraudulent or not. For this purpose, I'm using a dataset of about 300.000 transactions with credit card. The data has been anonymized to protect the privacy of the customers.
# 
# # Table of Contents:
# * [1-Exploratory Analysis](#exploratory)
# * [2- Preprocessing the data](#preprocessing)
# * [3- Deep Neural Network](#DNN)
# * [4- Training the DNN](#training)
# * [5- Evaluation of the DNN](#evaluation)
# * [6- Undersampling the dataset](#undersampling)
# * [7- SMOTE](#smote)
# * [8- Conclusions](#conclusions)
# 
# 
# 

# In[ ]:


#importing libraries
import pandas as pd 
import numpy as np
import keras
import tensorflow
import seaborn as sns
import matplotlib.pyplot as plt


# ## Exploratory Analysis <a class="anchor" id="exploratory"></a>

# In[ ]:


#loading our dataset
df=pd.read_csv("../input/creditcard.csv")
df.tail()


# In[ ]:


#Heatmap to see the correlations between the variables
plt.figure(figsize=(15,10))
sns.heatmap(df.corr())
print("there is no correlation between the variables")


# ## Preprocessing the data <a class="anchor" id="preprocessing"></a>

# In[ ]:


from sklearn.preprocessing import StandardScaler

#We are going to standarize the column Amount due the range of values it has.
df['normAmount']= StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))


# In[ ]:


df= df.drop(['Amount'],axis=1)
df= df.drop(['Time'],axis=1)


# In[ ]:


#Splitting the dataset into X and Y
X= df.iloc[:,df.columns != 'Class']
Y= df.iloc[:,df.columns == 'Class']


# In[ ]:


#Splitting the dataset into the train set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=2019)


# In[ ]:


X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test =np.array(y_test)


# In[ ]:


X_train.shape


# # Deep Neural Network(DNN)<a class='anchor' id='DNN'></a>

# In[ ]:


#importing the libraries of the DNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[ ]:


#Defining the neural network
model = Sequential([
    Dense(units=16,input_dim=29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(units=20,activation='relu'),
    Dense(units=24,activation='relu'),
    Dense(units=1,activation='sigmoid'),
])


# In[ ]:


model.summary()


# # Training the DNN <a class='anchor' id='training' ></a>

# In[ ]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[ ]:


score= model.evaluate(X_test,y_test)
print(score)


# In[ ]:


y_pred= model.predict(X_test)
y_test=pd.DataFrame(y_test)


# In[ ]:


#Defining the confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# # Evaluation of the DNN <a class='anchor' id=evaluation></a>

# In[ ]:


#Confusion matrix of our Test set
c_mat=confusion_matrix(y_test,y_pred.round())
plot_confusion_matrix(c_mat,classes=[0,1])


# In[ ]:


#Confusion matrix of the dataset
y_pred=model.predict(X)
y_expected=pd.DataFrame(Y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# # Undersampling the dataset <a class='anchor' id='undersampling' ></a>

# In[ ]:


#creating an index for the fraudulent and normal transactions
fraud_index=np.array(df[df.Class==1].index)
normal_index=df[df.Class == 0].index
count_fraud=len(fraud_index)


# In[ ]:


rand_normal_index = np.random.choice(normal_index,count_fraud,replace=False)
rand_normal_index = np.array(rand_normal_index)


# In[ ]:


undersample_index=np.concatenate([fraud_index,rand_normal_index])
print(len(undersample_index))


# In[ ]:


#undersampling the dataset
under_sample_data = df.iloc[undersample_index,:]
X_undersample = under_sample_data.iloc[:,under_sample_data.columns !='Class']
Y_undersample = under_sample_data.iloc[:,under_sample_data.columns =='Class']


# In[ ]:


#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_undersample,Y_undersample,test_size=0.3,random_state=2019)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test =np.array(y_test)


# In[ ]:


#training the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[ ]:


#confusion matrix of the test (for the undersampling)
y_pred =model.predict(X_test)
y_expected=pd.DataFrame(y_test)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# In[ ]:


#confusion matrix for the dataset (for the undersampling)
y_pred=model.predict(X)
y_expected=pd.DataFrame(Y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# # SMOTE <a class='anchor' id='smote'></a>

# In[ ]:


#creating the oversample
from imblearn.over_sampling import SMOTE
X_resample, y_resample = SMOTE().fit_sample(X,Y.values.ravel())
X_resample = pd.DataFrame(X_resample)
y_resample = pd.DataFrame(y_resample)


# In[ ]:


#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3,random_state=1492)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test =np.array(y_test)


# In[ ]:


#training the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[ ]:


#confusion matrix of the test (oversampling)
y_pred =model.predict(X_test)
y_expected=pd.DataFrame(y_test)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# In[ ]:


#confusion matrix of the dataset (oversampling)
y_pred =model.predict(X)
y_expected=pd.DataFrame(Y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# # Conclusions <a class='anchor' id='conclusions' ></a>

# ### I have created a Deep Neural Network (DNN) that is able to detect the 99% of the fraudulent transactions. However, it detects some of the normal transactions as fraudulents. Nonetheless, the amount of work of the fraud detection department has been significantly reduced. 

# In[ ]:




