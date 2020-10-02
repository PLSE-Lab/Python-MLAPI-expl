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


input_df=pd.read_csv(r'../input/train.csv')


# In[ ]:


input_df.head()


# In[ ]:


X=input_df.drop('label',axis=1).values.reshape(-1,28,28,1)
Y=input_df['label']


# In[ ]:


input_df.describe()


# In[ ]:


X=X/255


# In[ ]:


X.shape


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


b=np.arange(0,11)
c=b-0.5
plt.figure(figsize=(8,8))
plt.xlabel('labels')
plt.ylabel('count')
plt.title('label vs count')
plt.xticks(b)
plt.hist(Y,bins=c,rwidth=0.9)
plt.show()


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Input


# In[ ]:


model=Sequential([Conv2D(6,padding='same',kernel_size=(5,5),activation='relu',input_shape=(28,28,1)),MaxPooling2D(),Conv2D(16,padding='valid',kernel_size=(5,5),activation='relu'),MaxPooling2D(),Flatten(),Dense(120,activation='relu'),Dense(84,activation='relu'),Dense(10,activation='softmax')])


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy')


# In[ ]:


model.fit(X,Y,batch_size=64,epochs=150)


# In[ ]:


def get_predictions(md,x_test):
    preds = md.predict(x_test)
    y_pred = [np.argmax(i, axis=0) for i in preds]
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape((-1, 1))
    return y_pred


# In[ ]:


test_df=pd.read_csv(r'../input/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


X_rtest=test_df.values.reshape(-1,28,28,1)


# In[ ]:


predictions=get_predictions(model,X_rtest)


# In[ ]:


submission=pd.DataFrame(predictions)


# In[ ]:


submission=submission.reset_index()


# In[ ]:


submission.head()


# In[ ]:


submission.columns=['ImageId','Label']


# In[ ]:


submission['ImageId']=submission['ImageId']+1
submission.head()


# In[ ]:


submission.to_csv('submissions.csv',index=False)


# In[ ]:




