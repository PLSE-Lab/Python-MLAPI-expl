#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input/btvdd/BTVDD/Validation"))


# In[1]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Lambda
from keras.layers import Permute, TimeDistributed, Bidirectional, GRU, GaussianNoise
from keras.layers import AveragePooling2D, SpatialDropout2D, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model
from IPython.display import Image, display
from learntools.deep_learning.decode_predictions import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib.pylab import rcParams
from os.path import join
from IPython.display import HTML
from skimage import io
import warnings
import os
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('env', 'JOBLIB_TEMP_FOLDER=/tmp')


# In[10]:


# define the CNN
img_size = 224                  # H/V pixels
num_categories = 6             # Categories for classification
num_channels = 1               # 3 for RGB, 1 for mono
pad = 'same'
drop = 0.3 # dropout rate when training

# Assemble CNN
model = Sequential()

# add noise on input
#model.add(GaussianNoise(0.1, input_shape=(img_size,img_size,num_channels)))

# input layer
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', 
                 input_shape=(img_size, img_size, num_channels), padding=pad))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding=pad))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

## perform batch normalization on the convolution outputs before feeding it to MLP architecture
model.add(BatchNormalization())
model.add(Flatten())

## Dense layers : 256 -> num_categories: dropouts to avoid overfitting / regularization
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(drop))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=num_categories,activation='softmax'))

model.summary()


# In[ ]:


# plot model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
Image("model.png")


# In[11]:


# Setup optimiser, callback and complile
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-07, decay=0.0)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
# learning rate callback
learning_rate = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=4, #4
                                            verbose=1, 
                                            factor=0.3,
                                            min_lr=0.00001)

## Alternative
#optimizer='sgd'

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[12]:


#Prepare generators
augment = False
BatchSize = 50

train_dir = "../input/btvdd/BTVDD/Training"
val_dir = "../input/btvdd/BTVDD/Validation"
test_dir = "../input/btvdd/BTVDD/Test"

TotalTrainSamples = len(os.listdir(train_dir+'/Nominal'))
TotalValSamples = len(os.listdir(val_dir+'/Nominal'))

if augment:
    myepochs = 10
    aug_data_generator = ImageDataGenerator(preprocess_input, 
                                            rescale=1./255,
                                            horizontal_flip=False,
                                            width_shift_range = 0.01,
                                            height_shift_range = 0.01,
                                            rotation_range = 1,
                                            #shear_range = 0,
                                           )
else:
    myepochs = 10
    aug_data_generator = ImageDataGenerator(preprocess_input,
                                            rescale=1./255,
                                           )

data_generator = ImageDataGenerator(preprocess_input,
                                   rescale=1./255,
                                   )

train_generator = aug_data_generator.flow_from_directory(
                                            directory=train_dir,
                                            shuffle=True,
                                            color_mode = 'grayscale',
                                            target_size=(img_size, img_size),
                                            batch_size=BatchSize,
                                            class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
                                            directory=val_dir,
                                            shuffle=True,
                                            color_mode = 'grayscale',
                                            target_size=(img_size, img_size),
                                            batch_size=BatchSize,
                                            class_mode='categorical')


# In[13]:


SetsPerEpoch = 1

fit_stats = model.fit_generator(train_generator,
                                epochs = myepochs, 
                                validation_data = validation_generator,
                                verbose = 1, 
                                steps_per_epoch = SetsPerEpoch*TotalTrainSamples//BatchSize,
                                callbacks=[learning_rate],
                                validation_steps=TotalValSamples//BatchSize)

vindices = list(validation_generator.class_indices.keys())


# In[14]:


# plot training loss/accuracy
plt.rcParams['axes.grid'] = True
fig, ax = plt.subplots(2,1,figsize=(10,8))
ax[0].plot(fit_stats.history['loss'], color='b', label="Training loss")
ax[0].plot(fit_stats.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
#ax[0].set_ylim([0,0.1])
ax[1].plot(fit_stats.history['acc'], color='b', label="Training accuracy")
ax[1].plot(fit_stats.history['val_acc'], color='r',label="Validation accuracy")
#ax[1].set_ylim([0.975,1.0])
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


#test on mixture of previously unseen images
file_list = os.listdir(test_dir)
shuffle(file_list)

img_paths = [join(test_dir,filename) for filename in file_list]

def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]    
    img_array = np.array([img_to_array(img) for img in imgs])
    img_array_gray = np.zeros((img_array.shape[0],img_array.shape[1],img_array.shape[2],1))
    # next lines for grayscale
    for i, img in enumerate(img_array):
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        img_gray = (R * 299. / 1000 + G * 587. / 1000 + B * 114. / 1000) / 255
        img_array_gray[i,:,:,0] = img_gray
    #output = preprocess_input(img_array_gray)
    output = img_array_gray
    return(output)

def my_decode_preds(preds, vindices):
    out = []
    for i in range(len(preds)):
        qualif = "For sure "
        bestidx = np.argmax(preds[i])
        if preds[i][bestidx] < 0.6:
            qualif = "Possibly "
        elif preds[i][bestidx] < 0.9:
            qualif = "Probably "
        out.append(((qualif + vindices[bestidx] + " " + str(round(preds[i][bestidx],2)))))
    return(out)

test_data = read_and_prep_images(img_paths)
print(test_data.shape)
preds = model.predict(test_data)

most_likely_labels = my_decode_preds(preds, vindices)
print(validation_generator.class_indices)

for i in range(0,len(img_paths)):
    display(Image(img_paths[i], width = 100), HTML(str(most_likely_labels[i])))
    input()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


print(preds[:,0])

