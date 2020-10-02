#!/usr/bin/env python
# coding: utf-8

# # Facial Expression Recognition (Emotion Detection)

# ## Dataset : FER2013 (unknown/black/comic pictures removed)
# ![fer2013](https://miro.medium.com/max/602/1*slyZ64ftG12VU4VTEmSfBQ.png)

# In[ ]:


import os
print(os.listdir('../input/fer2013clean/fer2013-clean'))


# In[ ]:


print(os.listdir('../input/fer2013clean/fer2013-clean/Training'))


# In[ ]:


import matplotlib.pyplot as plt
def plot_imgs(item_dir, top=10):
    all_item_dirs = os.listdir(item_dir)
    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:5]
  
    plt.figure(figsize=(10, 10))
  
    for idx, img_path in enumerate(item_files):
        plt.subplot(5, 5, idx+1)
    
        img = plt.imread(img_path)
        plt.tight_layout()         
        plt.imshow(img, cmap='gray') 


# In[ ]:


data_path = '../input/fer2013clean/fer2013-clean/Training'


# In[ ]:


plot_imgs(data_path+'/Angry')


# In[ ]:


plot_imgs(data_path+'/Disgust')


# In[ ]:


plot_imgs(data_path+'/Fear')


# In[ ]:


plot_imgs(data_path+'/Happy')


# In[ ]:


plot_imgs(data_path+'/Neutral')


# In[ ]:


plot_imgs(data_path+'/Sad')


# In[ ]:


plot_imgs(data_path+'/Surprise')


# In[ ]:


from matplotlib import pyplot
from math import sqrt 
import numpy as np 
from PIL import Image
from IPython.display import display 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


num_classes = 7
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
target_size = (48, 48)


# ## Data Generator

# In[ ]:


train_dir = '../input/fer2013clean/fer2013-clean/Training'
val_dir   = '../input/fer2013clean/fer2013-clean/PrivateTest'
test_dir  = '../input/fer2013clean/fer2013-clean/PublicTest'


# In[ ]:


batch_size = 64

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')


# ## Build Model

# In[ ]:


# Build Model
model = Sequential()
# 1st Conv layer
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 2nd Conv layer
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 3nd Conv layer
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# FC layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# summary layers
model.summary()


# In[ ]:


# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


# ## Train Model

# In[ ]:


num_epochs = 100
step_size_train=train_generator.n//train_generator.batch_size
step_size_val =val_generator.n//val_generator.batch_size
step_size_test =test_generator.n//test_generator.batch_size


# In[ ]:


# Train Model
history = model.fit_generator(train_generator, 
                    steps_per_epoch=step_size_train, 
                    epochs=num_epochs,  
                    verbose=1,
                    validation_data=val_generator,  
                    validation_steps=step_size_val) 


# ## Save Model

# In[ ]:


# Save Model
model.save('fer2013_cnn.h5') 


# ## Plot Training History

# In[ ]:


print(history.history.keys())


# In[ ]:


# visualizing losses and accuracy
get_ipython().run_line_magic('matplotlib', 'inline')

train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

epochs = range(len(train_acc))

plt.plot(epochs,train_loss,'r', label='train_loss')
plt.plot(epochs,val_loss,'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.figure()

plt.plot(epochs,train_acc,'r', label='train_acc')
plt.plot(epochs,val_acc,'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.figure()


# ## Evaluate Model

# In[ ]:


# Evaluate Model
result = model.evaluate_generator(test_generator, steps=step_size_test) 
print("Test Loss: " + str(result[0]))
print("Test Accuracy: " + str(result[1]))


# In[ ]:




