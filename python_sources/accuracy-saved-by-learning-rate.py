#!/usr/bin/env python
# coding: utf-8

# # Problem Statement.
# - Handwritten Digits recognizer.
# - Train dataset consists of 42k rows with 785 cols.
# - Test dataset consists of 28k rows with 784 cols.
# - hello world competition of Computer Vision. 

# # Steps:
# 1. Import libraries.
# 2. Get data into pandas dataframe
#     - 2.1 drop label from train dataset.
# 3. Pre-processing
#     - 3.1 Normalize data for better performance (Convert pixels values from [0:255] to [0:1])
#     - 3.2 Reshape data for keras convention (sample size,width,height,color) -- syntax
#     - 3.3 convert labels into one-hot-encode and also view OHE labels for understanding.
#     
# 4. Visualize data with matplotlib
# 5. Split train data for validation.
# 6. Build model CNN using keras
#    - 6.1 Set parameters and stack layers
#    - 6.2 Get summary.
#    - 6.3 Define loss function and metrics 
#    - 6.4 Train model with parameters.
#    - 6.5 Set callbacks to monitor val_acc and for Learning rate to stop when there won't increases in val_accuracy.
# 7. Predict and submit.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1. Import Libraries

# In[ ]:


import keras # Neural nets API
import numpy as np # Linear algebra
import pandas as pd # Data manipulation.


# In[ ]:


# Load data into train and test pandas dataframe
train_df=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_df=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


# view top 5 rows. 
train_df.head()


# - Label variable states the digit of each col

# In[ ]:


test_df.head() # view top 5 rows of test data.


# - No label we need to predict .

# In[ ]:


# shape of both train and test dataset.
train_df.shape ,test_df.shape


# In[ ]:


# drop target (label) into new one
target=train_df["label"]
train_df.drop("label",axis=1,inplace=True)


# In[ ]:


train_df.head()


# - Label dropped

# # 2. Normalize and Reshape data to visualize images format.

# In[ ]:


train_df=train_df/255 # normalize will work better with cnn
test_df=test_df/255 # from [0:255] to [0:1]


# In[ ]:


X_train=train_df.values.reshape(-1,28,28,1) # reshaping to keras convention (sample,height,width,color)
test=test_df.values.reshape(-1,28,28,1)


# In[ ]:


from keras.utils.np_utils import to_categorical
y_train=to_categorical(target,num_classes=10) # one hot encoding


# In[ ]:


y_train[0] # view first label after OHE.


# # 3. visualize by reshaping data.

# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

for i in range(30):
    plt.subplot(3,10,i+1)
    plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis("off")
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()


# # 4. Split train data for validation

# In[ ]:


# train test split data one for training one for vaildation.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.10,random_state=42)


# In[ ]:


plt.imshow(X_train[0].reshape((28,28))) # plot


# In[ ]:


y_train[0] # result for above plot.


# # 5. Model building.

# 
# ### Network Parameters:
# 
# > - Batch Size - Number of rows from the input data to use it one iteratation from the training purpose  
# > - Num Classes - Total number of possible classes in the target variable  
# > - Epochs - Total number of iterations for which cnn model will run.

# In[ ]:


batch_size=128
num_classes=10
epochs=20
inputshape=(28,28,1)


# In[ ]:


from keras.models import Sequential # import sequential convention so we can add layer after other.
import keras
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,BatchNormalization
model=Sequential()

# add first convolutional layer.
model.add(Conv2D(32,kernel_size=(5,5),activation="relu",input_shape=inputshape))
# add second convolutional layer
model.add(Conv2D(64,(3,3),activation="relu"))
          
# add maxpooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(5,5),activation="relu"))
# add second convolutional layer
model.add(Conv2D(128,(3,3),activation="relu"))

# add one drop layer
model.add(Dropout(0.25))

# add flatten layer
model.add(Flatten())

# add dense layer
model.add(Dense(256,activation="relu"))
model.add(Dense(128,activation="relu"))
          
# add another dropout layer
model.add(Dropout(0.5))

# add dense layer
model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


# complile the model and view its architecture
model.compile(loss="categorical_crossentropy",  optimizer="Adam", metrics=['accuracy'])
model.summary()


# In[ ]:


# callbacks
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.3, min_lr = 0.00001)
checkpoint = ModelCheckpoint('save_weights.h5', monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 10, verbose = 1, restore_best_weights = True)

callbacks = [reduce_learning_rate, checkpoint, early_stopping]


# In[ ]:


# train model
model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test),callbacks=callbacks)
accuracy=model.evaluate(X_test,y_test)


# In[ ]:


pred = model.predict_classes(test)
res = pd.DataFrame({"ImageId":list(range(1,28001)),"Label":pred})
res.to_csv("output.csv", index = False)

