#!/usr/bin/env python
# coding: utf-8

# # Fitting a Model With Data Augmentation

# In[ ]:


import os
print(os.listdir("../input/sohan-dataset-new/augmented asl 3/Augmented ASL 3"))
    
    


# In[ ]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 64

data_generator_with_aug = ImageDataGenerator(
                                    samplewise_center=True, 
                                    samplewise_std_normalization=True,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   validation_split = 0.1, rotation_range = 10,
                                   zoom_range = 0.1)


train_generator = data_generator_with_aug.flow_from_directory(
        '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train',
        target_size=(image_size, image_size),
        batch_size=64,
        class_mode='categorical', subset = 'training')

validation_generator = data_generator_with_aug.flow_from_directory(
        '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train',
        target_size=(image_size, image_size),
        class_mode='categorical', subset = 'validation')


# In[ ]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 29
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=6,
        validation_data=validation_generator, validation_steps = 20)


# In[ ]:


my_new_model.save("trial.hdf5")


# In[ ]:




