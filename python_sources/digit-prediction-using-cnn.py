#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np


# In[32]:


test=pd.read_csv('../input/test.csv')


# In[33]:


train=pd.read_csv('../input/train.csv')


# In[34]:


Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 


# In[35]:


# Check the missing data
X_train.isnull().any().describe()


# In[36]:


test.isnull().any().describe()


# In[37]:


# Normalize the data
X_train = X_train / 255.0
test = test / 255.0


# In[38]:


# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# In[40]:


from tensorflow import keras as tfk


# In[41]:


Y_train = tfk.utils.to_categorical(Y_train) 


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train,X_val,Y_train,Y_val=train_test_split(X_train, Y_train, test_size = 0.1, random_state=0)


# In[44]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[45]:


tb=tfk.callbacks.TensorBoard()


# In[46]:



# CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[47]:


model.compile(optimizer='RMSprop', loss=tfk.losses.categorical_crossentropy, metrics=["acc"])


# In[48]:


model_history = model.fit(X_train, Y_train, batch_size=600, epochs=5, validation_split=0.2, callbacks=[tb])


# In[49]:


model.summary()


# In[51]:


model.evaluate(X_val, Y_val, batch_size=600)


# In[52]:


y_test_model = model.predict(X_val, batch_size=600)


# In[53]:


y_test_model[0]


# In[55]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


image_index = 4144
plt.imshow(X_val[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(X_val[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())


# here it is trying to predict 7 with the accuracy of 98% which we got

# In[58]:


y_test_model = np.argmax(y_test_model, axis=1)


# In[59]:


y_test_original = np.argmax(Y_val, axis=1)


# In[60]:


from sklearn.metrics import confusion_matrix


# In[63]:


cm=confusion_matrix(y_test_model,y_test_original)


# In[62]:


import seaborn as sns


# In[74]:


plt.figure(figsize=(20, 20))
sns.heatmap(cm,annot=True,square=True,cmap="Reds")
plt.show()


# In[75]:


from sklearn.metrics import classification_report


# In[76]:


classification_report(y_test_model,y_test_original)


# In[77]:


model_history.history.keys()


# In[78]:


model_history.params


# In[79]:


plt.plot(model_history.history["val_acc"], label="Validation Acc")
plt.plot(model_history.history["acc"], label="Training Accuracy")
plt.legend()


# In[80]:


plt.plot(model_history.history.get("loss") ,label="Losses")
plt.plot(model_history.history.get("val_loss"), label="Validation Loss")
plt.legend()


# In[98]:


results=model.predict(test)
results=np.argmax(results,axis=1)
results = pd.Series(results,name="Label")
submission = pd.DataFrame([pd.Series(range(1,28001),name = "ImageId"),results])
submission.to_csv('submission.csv',header=True,index=False)


# In[99]:


submission.shape


# In[100]:


submission.head()


# In[ ]:




