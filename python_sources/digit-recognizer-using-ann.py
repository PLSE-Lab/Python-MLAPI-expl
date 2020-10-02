#!/usr/bin/env python
# coding: utf-8

# # **Handwritten Digit Prediction** 
# ## My First Kaggle notebook
# ### Digits are predicted by using simple ANN model.
# ### Please give a upvote if you like my work.

# **Importing the neccessary libraries** 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Reading the data**

# In[ ]:


train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
digits=pd.concat([train,test],axis=0)
train.shape,test.shape


# In[ ]:


#printing the mergerd data
print(digits.shape)
digits.head()


# ### Splitting the class label from the dataset

# In[ ]:


target=train.label
train.drop(columns=['label'],inplace=True)


# ### Converting target/class label into np array

# In[ ]:


y_train=np.asarray(target)
y_train


# ### Also converting features of train and test datasets into array

# In[ ]:


X_train=np.asarray(train)
X_test=np.asarray(test)
X_train.shape,X_test.shape


#     Normalizing the features into 0 to 1 scale
#     Pixels values ranges from 0 to 255

# In[ ]:


X_train=X_train/255
X_test=X_test/255


# ### reshaping the array to make it into matrix form
# 

# In[ ]:


X_train=X_train.reshape(-1,28,28)
X_test=X_test.reshape(-1,28,28)
X_train.shape,X_test.shape


# now there are 42000 rows, each row of 28*28 dimensions

# ### Vizualizing the array of matrix form
#     some samples drawn from training an testing arrays

# In[ ]:


plt.matshow(X_train[0])
plt.show()


# In[ ]:


plt.matshow(X_train[1])
plt.show()


# In[ ]:


plt.matshow(X_train[2])
plt.show()


# In[ ]:


#Class labels of above image pixels of training data
y_train[0:3]


# ### We have to predict the following pixel images of test data

# In[ ]:


# test image pixel
plt.matshow(X_test[0])
plt.show()


# In[ ]:


# test image pixel
plt.matshow(X_test[5])
plt.show()


# In[ ]:


#class labeles present in datasets
class_labels=list(set(y_train))
class_labels


# ## **Model Building using ANN**
# 

# In[ ]:


ann=Sequential()


# In[ ]:


# input layers of size of 28*28
ann.add(Flatten(input_shape=[28,28]))

# 3 hidden layers containing 100 neurons
ann.add(Dense(512,activation='relu'))
ann.add(Dense(256,activation='relu'))
ann.add(Dense(128,activation='relu'))

#output layers containing 10 neurons to predict each of digit
ann.add(Dense(10,activation='softmax'))


# In[ ]:


ann.summary()


# In[ ]:


ann.compile(loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
            


# In[ ]:


ann.fit(X_train,y_train,batch_size=32,epochs=15)


# In[ ]:


detail=ann.evaluate(X_train,y_train)
print('loss:',detail[0])
print('accuracy achieved:',round(detail[1]*100,4))


# ### Predicting the class labels for X_test/test data

# In[ ]:


y_pred=ann.predict(X_test)


# In[ ]:


#predicted digits are output of test data
predicted_digits=[class_labels[np.argmax(y_pred[i])] for i in range(len(y_pred))]
print("first ten outputs of test data:",*predicted_digits[:10])
                  


# In[ ]:


#checking our output for 2nd digit in test data
plt.matshow(X_test[1])
plt.show()


# In[ ]:


#above pixel image is 0 and prediction also showing 0
predicted_digits[1]


# ## **Prediction on some random samples of Test data using ANN model**

# In[ ]:


images=np.random.choice(len(X_test),size=12)
print("")
fig=plt.figure(figsize=(15,9))
fig.suptitle("predicted outputs of random handwritten digits".upper(),fontsize=18)
fig.subplots_adjust(hspace=0.5,wspace=0.5)

for i,num in zip(images,range(1,13)):
    label=class_labels[np.argmax(y_pred[i])]
    ax=fig.add_subplot(3,4,num)
    ax.matshow(X_test[i])
    ax.set_xlabel("Prediction-->{}".format(label),fontsize=16)


# ### Again comparing my the model output with test pixel images

# In[ ]:


images=np.random.choice(len(X_test),size=12)
print("")
fig=plt.figure(figsize=(15,9))
fig.suptitle("predicted outputs of random handwritten digits".upper(),fontsize=18)
fig.subplots_adjust(hspace=0.5,wspace=0.5)

for i,num in zip(images,range(1,13)):
    label=class_labels[np.argmax(y_pred[i])]
    ax=fig.add_subplot(3,4,num)
    ax.matshow(X_test[i])
    ax.set_xlabel("Prediction-->{}".format(label),fontsize=16)


# In[ ]:


y_train_pred=ann.predict(X_train)


# In[ ]:


y_train_p=[class_labels[np.argmax(y_train_pred[i])] for i in range(len(y_train_pred))]
y_train_p[0:10]


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
con_mat=confusion_matrix(y_train,y_train_p)
print("Confusion Matrix")
pd.DataFrame(con_mat,columns=class_labels,index=class_labels)


# ## **My above model has achieved the accuracy of 99.695**
# ## The model truly predicted some random samples of test data (checked by running that code multiple times)
# ### **If you find any mistake in above code please feel free to comment and give me advices**
# ### **I am a beginner in deep learning and please appreciate my work if you like it. **
# ##     ** Happy Learning**

# ### submission

# In[ ]:


my_submission=pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
my_submission.head()


# In[ ]:


my_submission['Label']=predicted_digits
my_submission.head(10)


# In[ ]:


my_submission.to_csv("digits_submission2.csv",index=False)


# In[ ]:


pd.read_csv("digits_submission2.csv").head(10)


# In[ ]:




