#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from keras.models import Sequential


# In[ ]:


train_data_path="../input/intel-image-classification/seg_train/seg_train"
test_data_path="../input/intel-image-classification/seg_test/seg_test"


# In[ ]:


train_datagen=ImageDataGenerator(rotation_range=40,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,vertical_flip=True,rescale=1./255)


# In[ ]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[ ]:


train_data=train_datagen.flow_from_directory(train_data_path,target_size=(150,150),class_mode="categorical",batch_size=70)


# In[ ]:


test_data=test_datagen.flow_from_directory(test_data_path,target_size=(150,150),class_mode="categorical",batch_size=30)


# In[ ]:


model=Sequential()


# In[ ]:


model.add(Conv2D(190, (3, 3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPool2D(strides=(2,2)))
model.add(Conv2D(180,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(165,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(100,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(70,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))

model.add(Flatten())


# In[ ]:


model.add(Dense(16,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(6,activation="softmax"))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["acc"])


# In[ ]:


history=model.fit_generator(train_data,epochs=50,steps_per_epoch=100,validation_data=test_data,validation_steps=50)


# In[ ]:


import matplotlib.pyplot as plt
train_acc=history.history['acc']
validation_acc=history.history['val_acc']
train_loss=history.history['loss']
validation_loss=history.history['val_loss']


# In[ ]:


plt.plot(train_loss)
plt.plot(validation_loss)
plt.title("Train Loss Vs. Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# In[ ]:


plt.plot(train_acc)
plt.plot(validation_acc)
plt.title("Train acc Vs. Validation acc")
plt.xlabel("Epochs")
plt.ylabel("Accuarcy")
plt.show()


# **Checking the Model on new data**

# In[ ]:


from keras.preprocessing import image


# In[ ]:


img=image.load_img("../input/intel-image-classification/seg_pred/seg_pred/10059.jpg",target_size=(150,150))


# In[ ]:


img


# In[ ]:


img_tensor=image.img_to_array(img)


# In[ ]:


import numpy as np
img_tensor=np.expand_dims(img_tensor,axis=0)


# In[ ]:


img_tensor.shape


# In[ ]:


img_tensor=img_tensor/255.0


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.imshow(img_tensor[0])


# In[ ]:


classes=model.predict(img_tensor)


# In[ ]:


import pandas as pd
def pred_class(classes):
  classes=list(classes)
  mx_class_val=0
  mx_class=0
  for i in range(0,len(classes)):
    if(classes[i]>mx_class_val):
      mx_class=i
      mx_class_val=classes[i]
  return mx_class+1

    


# In[ ]:


if(pred_class(classes[0])==1):
  print("Building")

if(pred_class(classes[0])==2):
  print("Forest")

if(pred_class(classes[0])==3):
  print("Glacier")
if(pred_class(classes[0])==4):
  print("Mountain")
if(pred_class(classes[0])==5):
  print("Sea")
if(pred_class(classes[0])==6):
  print("Street")

