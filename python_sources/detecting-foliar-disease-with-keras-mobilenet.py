#!/usr/bin/env python
# coding: utf-8

# # Plant Pathology 2020 - FGVC7
# Identify the category of foliar diseases in apple trees
# 
# Kaggle competition - https://www.kaggle.com/c/plant-pathology-2020-fgvc7/submit

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import os
import shutil
import matplotlib.pyplot as plt


# # Loading Data and Preprocessing
# 
# Here we load the data and take a look at what we're dealing with.

# In[ ]:


train = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')

target = train[['healthy', 'multiple_diseases', 'rust', 'scab']]
test_ids = test['image_id']

train_len = train.shape[0]
test_len = test.shape[0]

train.describe()


# Ah, we see the multiple_diseases label has drastically less images than the rest of the labels. Once we load the images in raw data form, we'll use scikitlearn to randomly over sample so we can fix this class imbalance.
# 
# Now let's load the image data.

# In[ ]:


print("Shape of train data: " + str(train.shape))
print("Shape of test data: " + str(test.shape))


# In[ ]:


train_len = train.shape[0]
test_len = test.shape[0]


# In[ ]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tqdm.notebook import tqdm

path = '../input/plant-pathology-2020-fgvc7/images/'
size = 224

train_images = np.ndarray(shape=(train_len, size, size, 3))
for i in tqdm(range(train_len)):
  img = load_img(path + f'Train_{i}.jpg', target_size=(size, size))
  train_images[i] = np.uint8(img_to_array(img))

test_images = np.ndarray(shape=(test_len, size, size, 3))
for i in tqdm(range(test_len)):
  img = load_img(path + f'Test_{i}.jpg', target_size=(size, size))
  test_images[i] = np.uint8(img_to_array(img))

train_images.shape, test_images.shape


# Let's take a look at what the images look like.

# In[ ]:


for i in range(4):
	plt.subplot(220 + 1 + i)
	plt.title(train['image_id'][i])
	plt.imshow(np.uint8(train_images[i]), interpolation = 'nearest', aspect='auto')
plt.show()
plt.savefig('train_images.png')


# In[ ]:


for i in range(4):
	plt.subplot(220 + 1 + i)
	plt.title(test['image_id'][i])
	plt.imshow(np.uint8(test_images[i]), interpolation = 'nearest', aspect='auto')
plt.show()
plt.savefig('test_images.png')


# Let's split out data into train and test sets for the model.

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_images, target.to_numpy(), test_size=0.1, random_state=289) 

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# Now use RandomOverSampler to fix our class imbalance in the multiple diseases class.

# In[ ]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=289)

x_train, y_train = ros.fit_resample(x_train.reshape((-1, size * size * 3)), y_train)
x_train = x_train.reshape((-1, size, size, 3))
x_train.shape, y_train.shape


# In[ ]:


import gc

del train_images
gc.collect()


# Now we prepare the data for going into a Keras deep learning model. Here I use the ImageDataGenerator to also give us more images by using the parameters to rotate, horizontally flip, and vertically flip. Also the image is samplewise standard normalized the raw data so that the activation functions work properly.

# In[ ]:


from keras_preprocessing.image import ImageDataGenerator

batch_size = 8

train_datagen = ImageDataGenerator(samplewise_center = True,
                                   samplewise_std_normalization = True,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rotation_range=70)

train_generator = train_datagen.flow(
    x = x_train, 
    y = y_train,
    batch_size = batch_size)

validation_datagen = ImageDataGenerator(samplewise_center = True,
                                        samplewise_std_normalization = True)

validation_generator = validation_datagen.flow(
    x = x_test, 
    y = y_test,
    batch_size = batch_size)


# Let's see what the images look like after processing and what they look like going into the model.

# In[ ]:


idx = np.random.randint(8)
x, y = train_generator.__getitem__(idx)
plt.title(y[idx])
plt.imshow(x[idx])


# # Keras Model
# Here we build the model. I will use a pre-trained MobileNet for deep CNN which will then be fed into a dense layer to predict 4 classes, since the original MobileNet predicts 1000. It will compile using the loss function KL Divergence, Adam optimizer, and accuracy metric.

# In[ ]:


def create_model():
    pre_trained = tf.keras.applications.MobileNet(input_shape=(size, size, 3), weights='imagenet', include_top=False)
    for layer in pre_trained.layers:
      layer.trainable = False
    
    #pretrained_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(SIZE,SIZE,3), include_top=False)
    model = tf.keras.Sequential([
      pre_trained,
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(4, activation='softmax')
      ])
    model.compile(
        loss = 'kullback_leibler_divergence', 
        optimizer = 'adam', 
        metrics = ['accuracy'])
    return model

model = create_model()

model.summary()


# Now define some model parameters and set up some callbacks.

# In[ ]:


epochs = 150
steps_per_epoch = x_train.shape[0] // batch_size
validation_steps = x_test.shape[0] // batch_size
print(steps_per_epoch)


# In[ ]:


es = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1)
mc = tf.keras.callbacks.ModelCheckpoint('model.hdf5', save_best_only=True, verbose=0)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1)

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005
rampup_epochs = 40
sustain_epochs = 20
exp_decay = .8

def lrfn(epoch):
  if epoch < rampup_epochs:
    return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
  elif epoch < rampup_epochs + sustain_epochs:
    return max_lr
  else:
    return min_lr
    
lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

rang = np.arange(epochs)
y = [lrfn(x) for x in rang]
plt.plot(rang, y)
print('Learning rate per epoch:')


# In[ ]:


history = model.fit(
    x = train_generator,  
    validation_data = validation_generator,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    verbose=1,
    callbacks=[es, lr, mc, rlr])


# In[ ]:


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[ ]:


train_err = (1-history.history['accuracy'][-1])*100
validation_err = (1-history.history['val_accuracy'][-1])*100
print("Train set error " + str(train_err))
print("Validation set error " + str(validation_err))


# In[ ]:


test_datagen = ImageDataGenerator(samplewise_center = True,
                                 samplewise_std_normalization = True)

test_generator = test_datagen.flow(
    x = test_images,
    shuffle = False)


# In[ ]:


probabilities = model.predict(test_generator, steps = len(test_generator))
print(probabilities)
print(probabilities[:,0].mean()*100)
print(probabilities[:,1].mean()*100)
print(probabilities[:,2].mean()*100)
print(probabilities[:,3].mean()*100)


# In[ ]:


res = pd.DataFrame()
res['image_id'] = test['image_id']
res['healthy'] = probabilities[:, 0]
res['multiple_diseases'] = probabilities[:, 1]
res['rust'] = probabilities[:, 2]
res['scab'] = probabilities[:, 3]


# In[ ]:


res.to_csv('submission.csv', index=False)


# In[ ]:


valid_probabilities = model.predict(validation_generator, steps = len(validation_generator))
print(valid_probabilities[:,0].mean()*100)
print(valid_probabilities[:,1].mean()*100)
print(valid_probabilities[:,2].mean()*100)
print(valid_probabilities[:,3].mean()*100)


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, valid_probabilities)


# In[ ]:




