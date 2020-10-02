# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))


# Any results you write to the current directory are saved as output.
### Implement this before using GPU!! 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras import optimizers

from keras.callbacks import TensorBoard
from keras.models import load_model

#%%
###
Image Augmentation --> instantiate generators of augmented image batches

# image augmentation for training set
train_datagen = image.ImageDataGenerator(horizontal_flip = True,
                                         rescale = 1. / 255,
                                         shear_range = 0.2,
                                         zoom_range = 0.2)

# image augmentation for training set
test_datagen = image.ImageDataGenerator(rescale = 1. / 255)

#------------------------------------------------------------------------------
training_batch_size = 8
validation_batch_size = 8

train_generator = train_datagen.flow_from_directory(
        r'C:\Users\Admond\Desktop\Shopee_IET_ML_Challenge_2018\Data (FULL)\Training Sets (Updated)',
        target_size = (299, 299),
        batch_size = training_batch_size,
        class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
        r'C:\Users\Admond\Desktop\Shopee_IET_ML_Challenge_2018\Data (FULL)\Validation Sets (Updated)',
        target_size = (299, 299),
        batch_size = validation_batch_size,
        class_mode = 'categorical')

#%%
''' Fine tune the InceptionV3 (total number of layers = 48) '''
 
image_input = Input(shape=(299, 299, 3)) # (299 x 299) for InceptionV3

''' Create a pre-trained model without output(fully connected) layer '''
# create the base pre-trained model
# Don't include the fully-connected layer at the top of the network
base_model = InceptionV3(weights = 'imagenet', include_top = False)
base_model.summary()

last_layer = base_model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)

''' Create a new output (fully connected) layer '''
# add fully-connected & dropout layer
x = Dense(256, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
#x = Dense(256, activation='relu',name='fc-2')(x)
#x = Dropout(0.5)(x)

# add a softmax layer for 18 classes (fully connected classifier)
out = Dense(18, activation='softmax',name='output_layer')(x)

''' 
1. Combine the base_model input layers and the new output layer 
2. custom_v3_model = base_model + out (new output layer)
'''
# this is the model we will train
# Model class API --> include all layers required in the computation of outputs given inputs
custom_v3_model = Model(inputs = base_model.input, outputs = out)
custom_v3_model.summary()

#%%
#''' We use ADAM with epsilon set to 0.1 '''
## epsilon for Inception network on ImageNet can be tuned to either 1 or 0.1 
## suggested by: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#adam = optimizers.Adam(epsilon = 0.1) 

'''
Why SGD?:
    1. Fine tune in a very slow learning rate
    2. This is to make sure that the magnitude of the updates stays very small, 
       so as not to wreck the previously learned features
    
Links:
------
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
'''
sgd = optimizers.SGD(lr=0.0001, momentum=0.9)
#%%
'''
AIM:
-----
1. Freeze all the layers (except output layer) of the base model
2. Train the top level output layer (top level classifier)
3. Then start fine-tuning convolutional weights alongside it

WHY:
-----
1. Because the large gradient updates triggered by the randomly initialized weights
   would wreck the learned weights in the convolutional base

Link: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

'''

''' Freeze all the layers (except output layer) of the base model '''
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    # By default, layer.trainable = True (we must set to False to freeze the weights)
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
custom_v3_model.compile(loss = 'categorical_crossentropy',
                        optimizer = sgd, 
                        metrics = ['accuracy'])

''' 
1. Since we have already fixed all the weights for the input layers of the custom_v3_model...
2. Now, we need to train only the top (output) layers (for a few epochs)
3. Once the top layers are well trained, we can start fine tuning the inner layers 
'''
# train the model on the new data for a few epochs
start = time.time()

hist_1 = custom_v3_model.fit_generator(
            train_generator,
            # use floor devision (get integer for the final value)
            steps_per_epoch = train_generator.samples // training_batch_size,
            epochs = 5,
            validation_data = validation_generator,
            validation_steps = validation_generator.samples // validation_batch_size)

end = time.time()
# check execution time to run V3 model fitting
print('\nTime taken to fit top layer with training data: {} mins\n'.format((end - start)/60))  
#%%
'''
--> Save the model (trained output layers) into a single HDF5 file
---> Load this pre-trained model next time before going to the steps below!!!
'''

custom_v3_model.save(r'C:\Users\Admond\Desktop\Shopee_IET_ML_Challenge_2018\InceptionV3_Trained_Output_Layers.h5')

custom_v3_model = load_model(r'C:\Users\Admond\Desktop\Shopee_IET_ML_Challenge_2018\InceptionV3_Trained_Output_Layers.h5')
#%%

'''
Now, the top layers are well trained. We will freeze the bottom N layers and 
train the remaining top layers
'''

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

'''
We chose to train the top 2 inception blocks, i.e. we will freeze the first 
249 layers and unfreeze the rest:
'''
for layer in custom_v3_model.layers[:249]:  # freeze the first 249 layers
   layer.trainable = False
for layer in custom_v3_model.layers[249:]:  # unfreeze the rest of the layers
   layer.trainable = True

'''
We need to recompile the custom_v3_model for these modifications to take effect
'''
custom_v3_model.compile(loss = 'categorical_crossentropy',
                        optimizer = sgd,
                        metrics = ['accuracy'])
#%%
''' 
Use Tensorboard to monitor the training process 

Executing the following command in a separate terminal
--> tensorboard --logdir=C:\Users\Admond\Desktop\Shopee_IET_ML_Challenge_2018\logs_InceptionV3

1. Notice that the logdir setting is pointing to the root of your log directory.
2. I told the tensorboard callback to write to a subfolder based on a timestamp.
3. Writing to a separate folder for each run is necessary, so you can compare different runs.
4. When you point Tensorboard to the root of the log folder, it will automaticall pick up all the
   runs you perform.
5. You can then select the runs you want to see on the website

''' 

tensorboard = TensorBoard(log_dir=r'C:\Users\Admond\Desktop\Shopee_IET_ML_Challenge_2018\logs_InceptionV3\{}'.format(time.time()),
                          batch_size = 8)   # batch size same as training and validation images

#%%
'''
We train our model again (this time fine-tuning the top 2 inception blocks
alongside the top Dense layers
'''
start = time.time()

hist = custom_v3_model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples / training_batch_size,
        epochs = 5,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples / validation_batch_size,
        callbacks = [tensorboard]) 
end = time.time()
# check execution time to run V3 model fitting
print('\nTime taken to fit InceptionV3 model with training data: {} mins\n'.format((end - start)/60))  

#%%
# Evaluates the model on a data generator
# gives out a measure of performance
(loss, accuracy) = custom_v3_model.evaluate_generator(validation_generator, steps = validation_generator.samples // validation_batch_size)

print("[INFO] loss={:.5f}, accuracy: {:.5f}%".format(loss,accuracy * 100))

# save the model into a single HDF5 file
custom_v3_model.save(r'C:\Users\Admond\Desktop\Shopee_IET_ML_Challenge_2018\InceptionV3.h5')

#%%
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#%%
#Output result
test_generator = test_datagen.flow_from_directory(
        r'C:\Users\Admond\Desktop\Shopee_IET_ML_Challenge_2018\Data (FULL)\Testing Sets',
        target_size = (299, 299),
        batch_size = 1,
        class_mode = 'categorical',
        shuffle = False)

output1 = custom_v3_model.predict_generator(test_generator,
                                            steps = 16111,
                                            max_queue_size = 1)


j = []
for i in range(len(test_generator.filenames)):
    j.append(test_generator.filenames[i][test_generator.filenames[i].find("_")+1:test_generator.filenames[i].find(".")])
j= list(map(int, j))

thefile = open('list.txt', 'w')
for item in j:
  thefile.write("%s\n" % item)
  

#predictions
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model
model = load_model(os.getcwd()+'/InceptionV3_15.h5')
model.summary()

import numpy as np
import pandas as pd
datagen = ImageDataGenerator(rescale=1./255)
gen = datagen.flow_from_directory(r'C:\Users\Admin\Desktop\shopee\testing',target_size=(299, 299),shuffle=0,batch_size=8)
filenames = gen.filenames
nb_test_output = len(filenames)
pred_class=np.argmax(model.predict_generator(gen, 2014),axis=1)
filenames=[filenames[i].split("_")[1].split(".")[0] for i in range(nb_test_output)]
data_output=pd.DataFrame()
data_output['id'] = filenames
data_output['category']=pred_class
data_output=data_output.sort_values(by=["id"])
data_output.to_csv("InceptionV3_15epoch_sub.csv",index=0)

