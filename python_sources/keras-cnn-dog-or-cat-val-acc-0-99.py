#!/usr/bin/env python
# coding: utf-8

# The notebook will take you through the use of transfer learning to obtain a validation accuracy of 0.99 +. The code can be used as a basis for other implementions too with a few minor tweaks. This dataset is a good introduction to Deep learning with CNN and Transfer learning. Hopefully i have explained things well as we go on

# # Import Library and check data locations

# In[ ]:


import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("../input"))


# # Prepare Data

# This section we use the zipfile to unzip both the test and train files, these will get stored in the \kaggle\working\ folders under there respective names

# In[ ]:


import zipfile


zip_files = ['test1', 'train']
# Will unzip the files so that you can see them..
for zip_file in zip_files:
    with zipfile.ZipFile("../input/{}.zip".format(zip_file),"r") as z:
        z.extractall(".")
        print("{} unzipped".format(zip_file))


# split the data in the training folder to give a category of cat (0) or dog (1) based on the image file name. this creates a list of 1's and 0's based on the image file name as shown in print categories

# In[ ]:


filenames = os.listdir("../working/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# Lets look at the top and bottom of our dataframe, as you can see the category is 1 or 0 based on the image name

# In[ ]:


df.head()


# In[ ]:


df.tail()


# ### See total amount of each in the training data, the data is nicely balanced so there is no inherent bias in the model

# In[ ]:


df['category'].value_counts().plot.bar()


# From our data we have 12000 cats and 12000 dogs

# # See sample image

# lets look at a random example of an image, if you run this code a number of times you will see that the images vary in size a lot something that will be dealt with later on

# In[ ]:


sample = random.choice(filenames)
image = load_img("../working/train/"+sample)
plt.imshow(image)


# # Transfer Learning
# 
# 

# We will be using the Inception ResNet V2 pretrained model
# 
# We will initialise the model for an image shape of 256,256 with 3 channels and initialise the weights based of its weights for the imagenet challege.
# 

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Dropout, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import inception_resnet_v2

IMG_SHAPE = (256,256, 3)


base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# Now we allow the base model to be trainable from the 700th layer as there is 780 layers in the base model.
# 
# We allow some layers to be trained as the model will be pretrain to classify in image net, hence opening up a few layers will result in the model being able to adapt to the new task better, and the weights already being a fairly close match to the task.
# 
# We freeze all the layers below 700 and then flatten the out of the 780 layer called 'conv_7b_ac'. from this we add a dense layer of 500 and an l2 norm regularization as this layer has about 25 million parameters so we want to not over fit. We then add drop out layer before narrowing the model to 10 neurons before the final neuron where we pass a single neuron with a sigmoid function that helps classify the problem.
# 
# The number of neurons we added is a randomly chosen number and can be seen as a hyperparameter we can chose to explore later along with dropout rate. 
# 
# We never want to intentially create bottlenecks in data only when we want to reduce the dimensionallity of the data.

# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint 

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))
# Fine-tune from this layer onwards
fine_tune_at = 700

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

last_layer = base_model.get_layer('conv_7b_ac')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 500 hidden units and ReLU activation
x = layers.Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.1))(x)
# Add a dropout rate of 0.3
x = layers.Dropout(0.3)(x)   
x = layers.Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.1))(x)          
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)

model = Model( base_model.input, x) 
model.summary()


# # Callbacks

# Implement a number of call backs to aid in the training process, early stopping will stop the training if the validation accuracy doesnt improve after 5 epochs. Reduce on Plateau reduces the learning rate by 0.2 when the validation loss doesnt decrease after 3 epoch. we set a minimum learning rate of 1 * 10^-8

# In[ ]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[ ]:


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', 
    verbose=1,
    patience=5,
    restore_best_weights=True)


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                             patience=3, min_lr=0.00000001,verbose = 1)


# 

# In[ ]:


callbacks = [early_stopping, reduce_lr]


# # Prepare data

# Replace labels with string, and split the training data into 90% training and 10% validation and create a seperate dataframe for each and then count number in each.
# Batch size to be based through the image generators will be 32 at a time, this is an arbitory number and is a hyper parameter we can play with.

# In[ ]:


df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 


# In[ ]:


train_df, validate_df = train_test_split(df, test_size=0.10, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[ ]:


train_df['category'].value_counts().plot.bar()


# In[ ]:


validate_df['category'].value_counts().plot.bar()


# In[ ]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=256


# * # Traning Generator
# 
# we will create and image generator that will cause random augmentations to occur to the data to allow the model to be invariant to change and positioning of the object (cat or dog). Image augmentation has been shown to aid in perfomance of training however we have to be careful to not augment the image too much from its original form. With the augmentation we are trying to create new data each time to the image is passed over to create the invariance. We do not apply augmentation to the validation set as this is the image we want to test the accuracy on.

# In[ ]:


train_datagen = ImageDataGenerator(
    rotation_range=35,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../working/train", 
    x_col='filename',
    y_col='category',
    target_size=(256,256),
    class_mode='binary',
    batch_size=batch_size
)


# ### Validation Generator

# In[ ]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../working/train/", 
    x_col='filename',
    y_col='category',
    target_size=(256,256),
    class_mode='binary',
    batch_size=batch_size
)


# In[ ]:


from tensorflow.keras.optimizers import Adam
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(optimizer = opt, 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])


# As seen above we will use the Adam optimisation, you can change the optimisation to SGD ect but i just saw that Adam worked best for this particular task. AmsGrad is a useful tool to use. Details of different learning algorithms can be found on the keras documentation page.

# # Example of generator
# 
# In this section we will look at an example of the image generator and how the image augmentation works. We input the image and thus get out 15 fairly different images. Hence this is wha the model will see if we train it on 15 epochs where 1 epoch it goes through the data once. We can therefor manually increase the dataset size by specifiying a batch size less than the number of examples and having the steps per epoch higher than is required to go through one epoch. Hence if we set the steps per epoch to be twice the size of length of training set // batch size, in one epoch the training set will be run through twice with a different image augmentation used for each image. The system never stores the image it flows the image from the location through the augmentation and then the model, so its a computationally efficent process. Making it far easier than hand creating new data.

# In[ ]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "../working/train", 
    x_col='filename',
    y_col='category',
    target_size=(256,256),
    class_mode='categorical'
)


# In[ ]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# As we can see the image looks very diffrent in all 15 instances

# # Fit Model
# 
# now we fit the model using 10 epochs or this can be changed to whatever you want.
# 

# In[ ]:


epochs=3
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=22500//256,
    steps_per_epoch=2500//256
)


# # Save Model

# In[ ]:


model.save_weights("model.h5")


# # Virtualize Training

# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# In[ ]:




