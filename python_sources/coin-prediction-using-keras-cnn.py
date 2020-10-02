#!/usr/bin/env python
# coding: utf-8

# **Load libraries**

# In[ ]:


import numpy as np # linear algebra
import os
from time import time
from keras.preprocessing.image import ImageDataGenerator, img_to_array, image
from keras.utils import np_utils
import json
from PIL import Image
import os
import tensorflow as tf


# **Specify location of our data**

# In[ ]:


data_dir = "../input/coin-images/coins/data"

data_train_path =  data_dir + '/train'
data_valid_path = data_dir + '/validation'
data_test_path =  data_dir + '/test'

print(os.listdir("../input/coin-images/coins/data"))


# **Load the json that maps the folder number to the coin name**

# In[ ]:


with open('../input/coin-images/cat_to_name.json', 'r') as json_file:
    cat_2_name = json.load(json_file)

print(cat_2_name['200'])


# **Create generators to apply transformations to the images during training**

# In[ ]:


batch_size=120

# Transforms
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,  # randomly shift images horizontally 
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True,
    featurewise_std_normalization=True, # Normalize images
    samplewise_std_normalization=True)

datagen_valid = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True)

datagen_test = ImageDataGenerator(
    rescale=1./255,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True)


# **Load the data using the generators**

# In[ ]:



train_generator = datagen_train.flow_from_directory(
        data_train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

valid_generator = datagen_valid.flow_from_directory(
        data_valid_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen_test.flow_from_directory(
        data_test_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')


# **Plot some coins to see the transformations**

# In[ ]:


import matplotlib.pyplot as plt


# Lets have a look at some of our images
images, labels = train_generator.next()

fig = plt.figure(figsize=(20,10))
fig.subplots_adjust(wspace=0.2, hspace=0.4)

# Lets show the first 32 images of a batch
for i, img in enumerate(images[:32]):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    image_idx = np.argmax(labels[i])


# **Keras maps each folder (class) to a number. Create a dictionary that maps the number assigned by keras to our folder real number**

# In[ ]:


int_to_dir = {v: k for k, v in train_generator.class_indices.items()}


# **Create the model using a pre-trained ResNet50. I add only the fully connected layers at the end.**

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.models import Model



base_model = ResNet50(
    include_top=False,
    weights="imagenet"
)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='elu')(x)
x = Dropout(0.95)(x)
# and a logistic layer
predictions = Dense(211, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = True


# **Specify the optimizer**

# In[ ]:


from keras.optimizers import Adam

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
             metrics=['accuracy'])


# **Specify how I want to train the model and train the model. How to save the model, when to stop training etc.**

# In[ ]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

num_train = len(train_generator.filenames)
num_valid = len(valid_generator.filenames)
num_test = len(train_generator.filenames)


# When to save the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

# Reduce learning rate when loss doesn't improve after n epochs
scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-8, verbose=1)

# Stop early if model doesn't improve after n epochs
early_stopper = EarlyStopping(monitor='val_loss', patience=10,
                              verbose=0, restore_best_weights=True)

# Train the model
history = model.fit_generator(train_generator,
                    steps_per_epoch=num_train//batch_size,
                    epochs=40,
                    verbose=1,
                    callbacks=[checkpointer, scheduler, early_stopper],
                    validation_data=valid_generator,
                    validation_steps=num_valid//batch_size)


# **Save the model. That way we can load it elsewhere**

# In[ ]:


model.save('model.h5')


# **Load the best saved weights into the model with the best scores**

# In[ ]:


model.load_weights('../input/weightskerascoincnn/model.weights.best.hdf5')


# **Evaluate our model**

# In[ ]:


score = model.evaluate_generator(test_generator, steps=num_test//1, verbose=1)
print('\n', 'Test accuracy:', score[1])


# **Sanity check to make sure nothing crazy is happening**

# In[ ]:


def get_prediction(img, real_label):
    img = image.img_to_array(img)/255
    
    #normalise image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean)/std
    
    img_expand = np.expand_dims(img, axis=0)

    prediction = model.predict(img_expand)
    prediction_int = np.argmax(prediction)

    dir_int = int_to_dir[prediction_int]
    label_name = cat_2_name[str(dir_int)]
    
    print("Predicted: {}\nReal:      {}".format(label_name, cat_2_name[str(real_label)]))
    print()


for i in range(10):
    random_index = np.random.randint(0, len(test_generator.filenames))
    
    img = test_generator.filenames[random_index]
    img = image.load_img("../input/coin-images/coins/data/test/"+img, target_size=(224,224))
    real_label = test_generator.filenames[random_index].split("/")[0]

    get_prediction(img, real_label)


# **The end**
