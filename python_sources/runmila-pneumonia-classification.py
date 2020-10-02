#!/usr/bin/env python
# coding: utf-8

# **Using a Convolutional Neural Network to Diagnose Pneumonia (Radiology)**
# 
# 
# This session on Computer Vision covers the training of a Convolutional Neural Network for Pneumonia X-Ray Image Classification. 
# By: Runmila AI Institute (https://runmilainstitute.com)
# 
# 
# 
# 
# **TensorFlow 2**
# ![https://storage.ning.com/topology/rest/1.0/file/get/3628709184?profile=RESIZE_710x](https://storage.ning.com/topology/rest/1.0/file/get/3628709184?profile=RESIZE_710x)

# **Multilayer Perceptrons**
# ![https://miro.medium.com/max/3000/1*BIpRgx5FsEMhr1k2EqBKFg.gif](https://miro.medium.com/max/3000/1*BIpRgx5FsEMhr1k2EqBKFg.gif)
# 
# 
# 
# 
# 
# 
# 
# **Convolutional Neural Networks**
# 
# They use filters to be able to successfully capture the Spatial and Temporal dependencies in an image. 
# 
# 
# ![https://cdn-images-1.medium.com/max/1600/1*ZCjPUFrB6eHPRi4eyP6aaA.gif](https://cdn-images-1.medium.com/max/1600/1*ZCjPUFrB6eHPRi4eyP6aaA.gif)
# 
# 
# ![http://karpathy.github.io/assets/selfie/gif2.gif](http://karpathy.github.io/assets/selfie/gif2.gif)
# 
# 
# **Convolutional Layers**
# 
# keras.layers.Conv2D(number_of_filters, filter_size, activation=None)
# 
# 
# **Filters/Kernels**
# 
# The feature capturing fields/windows. During training, your ConvNet would learn the best filters necessary for accomplishing its objective.
# 
# Not much different from the IG or other photo editing filters.  
# 
# 
# 
# ![https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/28011851/conv.gif](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/28011851/conv.gif)
# 
# 
# ![https://mlnotebook.github.io/img/CNN/convSobel.gif](https://mlnotebook.github.io/img/CNN/convSobel.gif)
# 
# 
# ![https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=364](https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=364)
# 
# 
# 
# 

# 

# 
# 
# 
# 
# **Pooling Layer**
# 
# Reduces the spatial size of Captured Features (*Feature Maps*) by extracting dominant features. 
# 
# 
# **Max Pooling** returns the maximum value from the portion of the image covered by the Filter.
# 
# 
# 
# ![https://developers.google.com/machine-learning/practica/image-classification/images/maxpool_animation.gif](https://developers.google.com/machine-learning/practica/image-classification/images/maxpool_animation.gif)

# 

# ![https://i0.wp.com/vinodsblog.com/wp-content/uploads/2018/10/CNN-2.png?resize=1300%2C479&ssl=1](https://i0.wp.com/vinodsblog.com/wp-content/uploads/2018/10/CNN-2.png?resize=1300%2C479&ssl=1)

# Sample Images From Dataset

# In[ ]:


pneumonia_path = '../input/chest_xray/chest_xray/test/PNEUMONIA/person103_bacteria_490.jpeg'
normal_path = '../input/chest_xray/chest_xray/test/NORMAL/IM-0005-0001.jpeg'


# Image of Pneumonia X-Ray

# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

img = mpimg.imread(pneumonia_path)
#imgplot = plt.imshow(img, cmap = 'gray')
imgplot = plt.imshow(img)


# Image of Normal X-Ray

# In[ ]:


img = mpimg.imread(normal_path)
imgplot = plt.imshow(img)


# Peprocessing and Augmenting Dataset Images 

# In[ ]:


#import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (64, 64)
datagen = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)


# In[ ]:



train_generator = datagen.flow_from_directory(
        '../input/chest_xray/chest_xray/train',
        target_size=IMG_SIZE,
        color_mode = 'grayscale',
        batch_size=32,
        class_mode='binary')

x_val, y_val = next(datagen.flow_from_directory(
        '../input/chest_xray/chest_xray/val',
        target_size=IMG_SIZE,
        color_mode = 'grayscale',
        batch_size=32,
        class_mode='binary')) # one big batch

x_test, y_test = next(datagen.flow_from_directory(
        '../input/chest_xray/chest_xray/test',
        target_size=IMG_SIZE,
        color_mode = 'grayscale',
        batch_size=180,
        class_mode='binary')) # one big batch


# Convolutional Neural Network Architecture

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=x_test.shape[1:]))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
model.summary()


# ModelCheckpoint To Save Best Weights and EarlyStopping to Stop Training When Training Stops Improving

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('pneumonia_cnn')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5)
callbacks_list = [checkpoint, early]


# Training The Convolutional Neural Network with The Training Data (First Round)

# In[ ]:


#First Round 
model.fit_generator(train_generator, 
                    steps_per_epoch=100, 
                    validation_data = (x_val, y_val), 
                    epochs = 1, 
                    callbacks = callbacks_list)

   
# Save the entire model as a SavedModel
model.save('pneumonia_cnn') 


# Evaluating The Convolutional Neural Network With Validation Data 

# In[ ]:


scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("val_loss:", scores[0])
print("val_mean_absolute_error:", scores[2])


# Training The Convolutional Neural Network with The Training Data (Continuation)

# In[ ]:


# Continued Training
model.fit_generator(train_generator, 
                    steps_per_epoch=100, 
                    validation_data = (x_val, y_val), 
                    epochs = 11, 
                    callbacks = callbacks_list)


# Load the Best Weights and Evaluate The Trained Convolutional Neural Network

# In[ ]:


model.load_weights(weight_path)
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("val_loss:", scores[0])
print("val_mean_absolute_error:", scores[2])


# Use The Trained Convolutional Neural Network for Clasification

# In[ ]:


pred_Y = model.predict(x_test, batch_size = 32, verbose = True)
print(pred_Y[:15])


# In[ ]:


print(y_test[:15])


# ROC Curve and Area Under ROC Curve for Evaluating The Trained Convolutional Neural Network

# In[ ]:


from sklearn.metrics import roc_curve, auc
# Compute ROC curve and ROC area for each class

num_classes = 0

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, pred_Y)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(11,8))
lw = 2
plt.plot(fpr, tpr, color='darkorange', 
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc2.png')

