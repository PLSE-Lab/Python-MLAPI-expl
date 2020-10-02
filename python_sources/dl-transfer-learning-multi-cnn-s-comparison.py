#!/usr/bin/env python
# coding: utf-8

# **This is just a quick comparasion between pre built and trained CNN's **
# 
# 
# There is a Lot of enhancment that can be Implement even on Data like cropping, and Lable checking ...etc
# as well as one the networks it selfs like , number of freezed and trained layers, number of added layers, number of neurons on each added layers, the activation functions, the optimizers, and much more.
# 
# this is just how to import and use these CNN's (**Demonstrating Transfer Learning only**)

# **Import needed library and frameworks**

# In[ ]:


import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
import matplotlib.pyplot as plt


# Call the InceptionV3 Structured Network without flatten and softmax layers

# In[ ]:


InceptionV3_pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False)

# uncomment the follwing code to use presaved weights
#pre_trained_model.load_weights(local_weights_file)
for layer in InceptionV3_pre_trained_model.layers:
    layer.trainable = False


# Call the ResNet152V2 Structured Network without flatten and softmax layers

# In[ ]:


ResNet152V2_pre_trained_model = ResNet152V2(input_shape = (150, 150, 3), 
                                include_top = False)
# uncomment the follwing code to use presaved weights
#pre_trained_model.load_weights(local_weights_file)
for layer in ResNet152V2_pre_trained_model.layers:
    layer.trainable = False


# Call the VGG19 Structured Network without flatten and softmax layers

# In[ ]:


VGG19_pre_trained_model = VGG19(input_shape = (150, 150, 3), 
                                include_top = False)
# uncomment the follwing code to use presaved weights
#pre_trained_model.load_weights(local_weights_file)
for layer in VGG19_pre_trained_model.layers:
    layer.trainable = False


# Call the VGG16 Structured Network without flatten and softmax layers

# In[ ]:


VGG16_pre_trained_model = VGG16(input_shape = (150, 150, 3), 
                                include_top = False)
# uncomment the follwing code to use presaved weights
#pre_trained_model.load_weights(local_weights_file)
for layer in VGG16_pre_trained_model.layers:
    layer.trainable = False


# view InceptionV3 network summary and select the layers that you want to use as last layer to build on it

# In[ ]:


InceptionV3_pre_trained_model.summary()
last_layer_InceptionV3 = InceptionV3_pre_trained_model.get_layer('mixed7')
print('InceptionV3 last layer output shape: ', last_layer_InceptionV3.output_shape)
InceptionV3_last_output = last_layer_InceptionV3.output


# view ResNet152V2 network summary and select the layers that you want to use as last layer to build on it

# In[ ]:


ResNet152V2_pre_trained_model.summary()
last_layer_ResNet152V2 = ResNet152V2_pre_trained_model.get_layer('conv4_block36_1_relu')
print('ResNet152V2 last layer output shape: ', last_layer_ResNet152V2.output_shape)
ResNet152V2_last_output = last_layer_ResNet152V2.output


# view VGG19 network summary and select the layers that you want to use as last layer to build on it

# In[ ]:


VGG19_pre_trained_model.summary()
last_layer_VGG19 = VGG19_pre_trained_model.get_layer('block5_conv4')
print('VGG19 last layer output shape: ', last_layer_VGG19.output_shape)
VGG19_last_output = last_layer_VGG19.output


# view VGG16 network summary and select the layers that you want to use as last layer to build on it

# In[ ]:


VGG16_pre_trained_model.summary()
last_layer_VGG16 = VGG16_pre_trained_model.get_layer('block5_conv3')
print('VGG16 last layer output shape: ', last_layer_VGG16.output_shape)
VGG16_last_output = last_layer_VGG16.output


# Import tnsorflow Adam Optimizor, this is one of the research area as importing and using different optimizor would help

# In[ ]:


from tensorflow.keras.optimizers import Adam
#########################################################
number_of_used_breed = 5
#########################################################
# number_of_used_breed is used here in Kaggle to reduce number of classes to be able to run all CNN's without timeout failing
# it can be increased to 120 to have all data if there is a suffecient machine


# Add the flatten layer,Dense layer and softmax layer one the top of pretrained InceptionV3 network
# 
# Dropout layer as added for regularaization

# In[ ]:


# Flatten the output layer to 1 dimension
x_InceptionV3 = layers.Flatten()(InceptionV3_last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x_InceptionV3 = layers.Dense(512, activation='relu')(x_InceptionV3)
# Add a dropout rate of 0.2
x_InceptionV3 = layers.Dropout(0.2)(x_InceptionV3)                  
# Add a final sigmoid layer for classification
x_InceptionV3 = layers.Dense  (number_of_used_breed, activation='softmax')(x_InceptionV3)           
InceptionV3_model = Model( InceptionV3_pre_trained_model.input, x_InceptionV3) 
InceptionV3_model.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])


# Add the flatten layer,Dense layer and softmax layer one the top of pretrained ResNet152V2 network
# 
# Dropout layer as added for regularaization

# In[ ]:


# Flatten the output layer to 1 dimension
x_ResNet152V2 = layers.Flatten()(ResNet152V2_last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x_ResNet152V2 = layers.Dense(512, activation='relu')(x_ResNet152V2)
# Add a dropout rate of 0.2
x_ResNet152V2 = layers.Dropout(0.2)(x_ResNet152V2)                  
# Add a final sigmoid layer for classification
x_ResNet152V2 = layers.Dense  (number_of_used_breed, activation='softmax')(x_ResNet152V2)           
ResNet152V2_model = Model( ResNet152V2_pre_trained_model.input, x_ResNet152V2) 
ResNet152V2_model.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])


# Add the flatten layer,Dense layer and softmax layer one the top of pretrained VGG19 network
# 
# Dropout layer as added for regularaization

# In[ ]:


# Flatten the output layer to 1 dimension
x_VGG19 = layers.Flatten()(VGG19_last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x_VGG19 = layers.Dense(1024, activation='relu')(x_VGG19)
# Add a dropout rate of 0.2
x_VGG19 = layers.Dropout(0.2)(x_VGG19)                  
# Add a final sigmoid layer for classification
x_VGG19 = layers.Dense  (number_of_used_breed, activation='softmax')(x_VGG19)           
VGG19_model = Model( VGG19_pre_trained_model.input, x_VGG19) 
VGG19_model.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])


# Add the flatten layer,Dense layer and softmax layer one the top of pretrained VGG16 network
# 
# Dropout layer as added for regularaization

# In[ ]:


# Flatten the output layer to 1 dimension
x_VGG16 = layers.Flatten()(VGG16_last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x_VGG16 = layers.Dense(1024, activation='relu')(x_VGG16)
# Add a dropout rate of 0.2
x_VGG16 = layers.Dropout(0.2)(x_VGG16)                  
# Add a final sigmoid layer for classification
x_VGG16 = layers.Dense  (number_of_used_breed, activation='softmax')(x_VGG16)           
VGG16_model = Model( VGG16_pre_trained_model.input, x_VGG16) 
VGG16_model.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])


# **the following callback class have been added to stop training at specific accuracy amount**

# In[ ]:


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.90):
            print('/nTraining will stop as we have reached 90% of acc')
            self.model.stop_training=True
        
callback=myCallback()


# In[ ]:


###################################
####### Import dataset here #######
###################################


# Creat list of all dogs breed that we have

# In[ ]:


breed_list = os.listdir("../input/stanford-dogs-dataset/images/Images/")

num_classes = len(breed_list)
print("{} breeds".format(num_classes))

n_total_images = 0
for breed in breed_list:
    n_total_images += len(os.listdir("../input/stanford-dogs-dataset/images/Images/{}".format(breed)))
print("{} images".format(n_total_images))


# create new directories for training and validation datasets and copy the image data there

# In[ ]:


from PIL import Image
breed_in_use = 1

os.makedirs('data',exist_ok= True)
os.makedirs('data/Training',exist_ok= True)
os.makedirs('data/Validation',exist_ok= True)
for breed in breed_list:
    os.makedirs('data/Training/' + breed,exist_ok= True)
    os.makedirs('data/Validation/' + breed,exist_ok= True)
    if breed_in_use == number_of_used_breed:
        break
    breed_in_use = breed_in_use+1
print('Created {} folders to store Training images of the different breeds.'.format(len(os.listdir('data/Training'))))
print('Created {} folders to store Validation images of the different breeds.'.format(len(os.listdir('data/Validation'))))

validation_to_training_ratio = .1
breed_in_use = 1
for breed in os.listdir('data/Training'):
    cpt = sum([len(files) for r, d, files in os.walk('../input/stanford-dogs-dataset/images/Images/{}/'.format(breed))])
    validation = (int)(cpt * validation_to_training_ratio)
    index = 0
    for file in os.listdir('../input/stanford-dogs-dataset/annotations/Annotation/{}'.format(breed)):
        img = Image.open('../input/stanford-dogs-dataset/images/Images/{}/{}.jpg'.format(breed, file))
        img = img.convert('RGB')        
        if index<validation:
            img.save('data/Validation/' + breed + '/' + file + '.jpg')
        else:
            img.save('data/Training/' + breed + '/' + file + '.jpg')
        index = index +1
    if breed_in_use == number_of_used_breed:
        break    
    breed_in_use = breed_in_use+1


# create data augmentation data flow to add more data based on the image data that we have

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = 'data/Training'
validation_dir = 'data/Validation'
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 50 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 50,
                                                    class_mode = 'categorical',
                                                    target_size = (150, 150))     

# Flow validation images in batches of 10 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 10,
                                                          class_mode  = 'categorical',
                                                          target_size = (150, 150))


# fit the data into the model using fit_generator to the InceptionV3 compiled network

# In[ ]:


history_InceptionV3 = InceptionV3_model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 16,
            epochs = 20,
            validation_steps = 9,
            verbose = 2,
            callbacks=[callback]
)


# plot the results of Training accuracy and Validation accuracy 

# In[ ]:


acc = history_InceptionV3.history['acc']
val_acc = history_InceptionV3.history['val_acc']
loss = history_InceptionV3.history['loss']
val_loss = history_InceptionV3.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


# fit the data into the model using fit_generator to the VGG19 compiled network

# other network has been comminted because of an issue in kaggle committing
# 
# JUST UNCOMMENT THE REST NETWORK TO TEST THEM
# 

# In[ ]:


#history_VGG19 = VGG19_model.fit_generator(
#            train_generator,
#            validation_data = validation_generator,
#            steps_per_epoch = 16,
#            epochs = 20,
##            validation_steps = 9,
#            verbose = 1,
#            callbacks=[callback]
#)


# plot the results of Training accuracy and Validation accuracy 

# In[ ]:


#acc = history_VGG19.history['acc']
#val_acc = history_VGG19.history['val_acc']
#loss = history_VGG19.history['loss']
#val_loss = history_VGG19.history['val_loss']

#epochs = range(len(acc))

#plt.plot(epochs, acc, 'r', label='Training accuracy')
#plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy')
#plt.legend(loc=0)
#plt.figure()

#plt.show()


# fit the data into the model using fit_generator to the VGG16 compiled network

# In[ ]:


#history_VGG16 = VGG16_model.fit_generator(
#            train_generator,
##            validation_data = validation_generator,
#            steps_per_epoch = 16,
#            epochs = 20,
#            validation_steps = 9,
##            verbose = 1,
#            callbacks=[callback]
#)


# plot the results of Training accuracy and Validation accuracy 

# In[ ]:


#acc = history_VGG16.history['acc']
#val_acc = history_VGG16.history['val_acc']
#loss = history_VGG16.history['loss']
#val_loss = history_VGG16.history['val_loss']

#epochs = range(len(acc))

#plt.plot(epochs, acc, 'r', label='Training accuracy')
#plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy')
#plt.legend(loc=0)
#plt.figure()

#plt.show()


# fit the data into the model using fit_generator to the ResNet152V2 compiled network

# In[ ]:


#history_ResNet152V2 = ResNet152V2_model.fit_generator(
#            train_generator,
#            validation_data = validation_generator,
#            steps_per_epoch = 16,
#            epochs = 20,
#            validation_steps = 9,
#            verbose = 1,
#            callbacks=[callback]
#)


# plot the results of Training accuracy and Validation accuracy 

# In[ ]:


#acc = history_ResNet152V2.history['acc']
#val_acc = history_ResNet152V2.history['val_acc']
#loss = history_ResNet152V2.history['loss']
#val_loss = history_ResNet152V2.history['val_loss']
#
#epochs = range(len(acc))
#
#plt.plot(epochs, acc, 'r', label='Training accuracy')
#plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy')
#plt.legend(loc=0)
#plt.figure()
#
#plt.show()


# In[ ]:


import shutil
shutil.rmtree('data', ignore_errors=False, onerror=None)

