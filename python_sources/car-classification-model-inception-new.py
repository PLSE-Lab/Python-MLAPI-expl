#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This model was trained on google colab so the cell outputs are not visible here.


#Go to https://colab.research.google.com/drive/1qGw8XtxRPir7stp-awqHBbd2XEpzC3AY?usp=sharing



import pandas as pd
import numpy as np
import os


# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')


# In[ ]:


import zipfile
import io
zf = zipfile.ZipFile(io.BytesIO(uploaded['Train.zip']), "r")
zf.extractall()


# In[ ]:


train_dir='/content/Train/Train'
val_dir='/content/Train/val'


# In[ ]:


print(len(os.listdir(train_dir)))
print(len(os.listdir(val_dir)))


# In[ ]:


from tensorflow.keras.applications.inception_v3 import InceptionV3


# In[ ]:


model=InceptionV3(include_top=False,weights='imagenet',input_shape=(224,224,3),classes=45)


# In[ ]:


for layer in model.layers[0:7]:
  layer.trainable = False
for layer in model.layers[8:]:
  layer.trainable=True


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.1,
                                   shear_range=0.1,
                                   horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255)


# In[ ]:


train_generator=train_datagen.flow_from_directory(train_dir,
                                                  class_mode='categorical',
                                                  target_size=(224,224),
                                                  batch_size=25,
                                                  color_mode='rgb',
                                                  shuffle = True,
                                                  seed=42)
val_generator=val_datagen.flow_from_directory(val_dir,
                                              class_mode='categorical',
                                              target_size=(224,224),
                                              batch_size=10,
                                              color_mode='rgb',
                                              shuffle = True,
                                              seed=42
                                              )


# In[ ]:


steps_train=train_generator.n//train_generator.batch_size
steps_val=val_generator.n//val_generator.batch_size


# In[ ]:


x=Flatten()(model.output)
x=Dense(2048,activation='relu')(x)
x=Dropout(0.5)(x)
pred=Dense(45,activation='softmax')(x)
model_tr = Model(model.input,pred)


# In[ ]:


model_tr.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import Callback


# In[ ]:


class myCallback(Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_accuracy') > 0.93 ):    
          self.model.stop_training = True
callbacks = myCallback()


# In[ ]:


history=model_tr.fit_generator(
    train_generator,
    steps_per_epoch=steps_train,
    validation_data=val_generator,
    validation_steps=steps_val,
    epochs=50,
    callbacks=[callbacks]
)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


test_dir='/content/drive/My Drive/car_classification/Test/Test1'


# In[ ]:


from tensorflow.keras.preprocessing.image import load_img,img_to_array


# In[ ]:


image=load_img(os.path.join(test_dir,'image'+str(20)+'.jpg'),
                   grayscale=False,
                   color_mode="rgb",
                   target_size=(224,224))
image


# In[ ]:


input_arr = img_to_array(image)
input_arr = np.array([input_arr],np.float32)/255


# In[ ]:


np.argmax(model_tr.predict(input_arr))


# In[ ]:


pred1=[]
for i in range(0,450):
  image=load_img(os.path.join(test_dir,'image'+str(i+1)+'.jpg'),
                   grayscale=False,
                   color_mode="rgb",
                   target_size=(224,224))
  input_arr = img_to_array(image)
  input_arr = np.array([input_arr],np.float32)/255
  pred1.append(np.argmax(model_tr.predict(input_arr)))


# In[ ]:


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions1 = [labels[k] for k in pred1]


# In[ ]:


for i in range(0,450):
  print(predictions1[i],i+1)


# In[ ]:


df=pd.read_csv('/content/drive/My Drive/car_classification/sample_submission.csv')


# In[ ]:


df.head()


# In[ ]:


df.drop('predictions',axis=1)


# In[ ]:


df['predictions']=pred1


# In[ ]:


df.head()


# In[ ]:


df.to_csv("results_new1.csv",index=False)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


model_tr.save('Inception_model.h5')


# In[ ]:




