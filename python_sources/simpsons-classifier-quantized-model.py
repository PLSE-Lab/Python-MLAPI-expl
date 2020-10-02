#!/usr/bin/env python
# coding: utf-8

# # Simpsons Classifier - Building a Quantized Model
# ##### Author : Anand
# 
# This is a fun kernel to give a brief idea about post training model quantization and the major use of it in real life
# 
# A model is as good as the real world value it can provide.So we need to find a way to show case our model to the Users.Since ML models are very resource demanding deploying our model to cloud sounds like a great idea, but this comes with a fair share of cons as well like
# * Security Concerns
# * Cloud cost for model deployment
# * Need of strong connection 
# ..etc
# 
# Another way is to deploy our model on resource constrained edge devices (tiny IOT devices,android,IOS devices ..etc).So we need a way to minimize the size our models so that it could run smoothly on all these resource constrained devices.
# 
# So this kernel is on how to build a keras image classifier then quantize this model so that this model could be deployed on smaller devices.
# 
# #### Model Quantization
# 
# Lets discuss on the topic model quantization, its basically a a way of conversion which could drastically a reduce the size of model without creating much degradation on the model accuracy.Creating an android app with a more than 200MB size model would be really bad idea, so post model quantization would really helpful in such situations.Model quantization process can reduce the model size 4x and can considerable increase the inference speed which are really useful in real life applications.
# 
# ## TF-Lite
# Tensorflow lite is framework for quantizing the model and on device inference.So we can convert any tensorflow or keras model to lite model so that there will considerable decrease in the model size and we could use the tf lite interpreter for on device inference.The working TF-lite is as follows
# 
# ![tf](https://sdtimes.com/wp-content/uploads/2017/11/tensorflow-490x464.jpg)
# sdtimes.com/embedded-devices/google-previews-tensorflow-lite/
# 
# For details
# https://www.tensorflow.org/lite
#  
# Our kernel plan of action
# * Build a simpsons image classifier using keras
# * Perform post model quantization
# * Create a .tflite model
# * Using TF-lite interpreter for inference

# In[ ]:


#imports
import os,random
from shutil import copyfile,copytree,rmtree
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import numpy as np

#visualization
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print(tf.__version__)
tf.test.is_gpu_available()


# In[ ]:


#lets set the train & validation path
image_dir = "/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset/"
test_dir = "/kaggle/input/the-simpsons-characters-dataset/kaggle_simpson_testset/"


# In[ ]:


#I noticed that simpson_dataset is already inside simpsons folder which should be removed
#Image should be copied to working to have delete permission
copytree(image_dir,"/kaggle/working/simpsons")


# In[ ]:


train_dir = "/kaggle/working/simpsons/"
rmtree('/kaggle/working/simpsons/simpsons_dataset/')


# In[ ]:


#Lets go with 32x32 pixels for faster training
IMG_SHAPE = (32,32,3)
num_classes = 42


# In[ ]:


#Now lets augement our image data
datagen_train = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   horizontal_flip=True,fill_mode='nearest')
datagen_test = ImageDataGenerator(rescale=1./255)


# In[ ]:


#We are converting all our training data to size of 32x32 pixels for reducing training time 
#Test set are allowed to have 224x224 for easy visualization and we can convert to 32x32 during predictions
train_generator = datagen_train.flow_from_directory(train_dir,target_size=(32,32))
test_generator = datagen_test.flow_from_directory(test_dir,target_size=(224,224))
class_names = {v:k for k,v in train_generator.class_indices.items()}


# In[ ]:


#Now lets take a sample from the train set
X_train,y_train = next(train_generator)
for i in range(5):
    plt.imshow(X_train[i])
    plt.show()
    print("Label:",class_names[np.argmax(y_train[i])])


# ### CNN Model Architecture
# 
# We are going to go with a model simpler yet very similar architecture to that of famous VGG16 as shown
# 
# ![CNN](https://www.researchgate.net/profile/Saikat_Roy9/publication/322787849/figure/fig1/AS:588338117488642@1517282146532/VGG16-Architecture-with-Softmax-layer-replaced-used-as-the-base-model-for-each-classifier.png)
# image copyright : www.researchgate.net/profile/Saikat_Roy9
# 
# The model is very similar to that of VGG with two convolution followed by maxpooling. So tweaks in the model
# * Added batch normalization for better convergence and reduced overfitting
# * Model is made simpler by not adding three conv is in the final convolution layers

# In[ ]:


#model building
model = tf.keras.models.Sequential()

#Conv 1
model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',input_shape = IMG_SHAPE,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
#Conv 2
model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
#Conv 3
model.add(tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
#Conv 4
model.add(tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
#Conv 5
model.add(tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
#Conv 6
model.add(tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

plot_model(model,show_shapes=True)


# In[ ]:


#lets compile the model
model.compile(loss='categorical_crossentropy',
             metrics=['acc'],
             optimizer=tf.keras.optimizers.Adam(lr = 0.001))


# In[ ]:


#We are not going for model checkpoints/earlystopping since squeezing out the best model is not the scope of this kernel
#So we haven't made any train/validation splits here as well
#lets train the model
epochs = 30
batch_size = 32
model.fit_generator(train_generator,epochs=epochs)


# In[ ]:


#Lets save the model fro later use
model.save('simpsons_model.h5')


# In[ ]:


#Now model building is completed.
#So lets see how good our model preforms on the test data
img,label = next(test_generator)
for i in range(5):
    #reshaping the image as per the model input shape
    pred_img = cv2.resize(img[i],(32,32))
    res = model.predict(np.expand_dims(pred_img,axis=0))
    plt.imshow(img[i])
    plt.show()
    print("Predicted :",class_names[np.argmax(res)])            


# #### Over predictions looks almost great :) 
# ![](https://media2.giphy.com/media/A6aHBCFqlE0Rq/giphy.gif?cid=790b761197a14552e72bd33d3a1bd7f479b2129158ff652d&rid=giphy.gif)
# credits:https://giphy.com
# 
# Since our model is ready and seems to work just fine its time to see how to quanitize it and make it ready for resource constrained devices.
# After saving the model the save of .h5 file is around 25MB which is not the big given the model architecure and image sizes we can still decrease and size without having that much of degradation on the accuracy
# 

# In[ ]:


#Save the model
export_dir = 'simpson_saved_model'
tf.saved_model.save(model,export_dir)


# ### Optimization
# 
# TF lite provide several optimization strategy like
# * DEFAULT
# * OPTIMIZE_FOR_LATENCY
# * OPTIMIZE_FOR_SIZE
#  In which has its own behaviour and we will be using OPTIMIZE_FOR_LATENCY as we give more importance to ion device inference accuracy and time
#  
#  Check
#  https://www.tensorflow.org/lite/performance/model_optimization
#  https://www.tensorflow.org/api_docs/python/tf/lite/Optimize
#  

# In[ ]:


#Now lets choose the optimzation strategy
optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY


# ### Converting Model to Lite
# Now we have our everything we need to convert our model and build the tflite model

# In[ ]:


#Generate the tflite model
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [optimization]
tflite_model = converter.convert()

#Now lets save the TFlite model
with open('simpsons.tflite','wb') as f:
    f.write(tflite_model)


# ### Lite model Interpretation
# 
#   We have checked our model efficiency on non quantized model and were very satisfied with result.But we are not sure the our lite model is up to the mark of its parent model. Thanks to tensorflow which provides something called an interpreter to see the efficiency  of our lite model.If the lite model is not up to the mark there is no point of deploying to any devices. So lets see how it goes
# 

# In[ ]:


#Time to test the TFlite mode
interpreter = tf.lite.Interpreter(model_content = tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


# In[ ]:


#let us take an image from the test set
rand_item = random.randint(0,len(img))
pred_img = cv2.resize(img[rand_item],(32,32))

interpreter.set_tensor(input_index, pred_img.reshape(-1,32,32,3))
interpreter.invoke()
res = interpreter.get_tensor(output_index)
print("Predicted :",class_names[np.argmax(res)])  
plt.imshow(img[rand_item])


# ## Hurrayyy :)
# 
# ### The lite model is working as expected
# ![](https://media1.giphy.com/media/3orieRXneF4pvFAoAo/giphy.gif?cid=790b76113276d6395f379cd8dfc912b44d042a0f32fedaba&rid=giphy.gif)
# 
# 

# ### On device inference
# 
# Now since our lite model is ready lets see how it actually performed on an android device
# I took the inference kotlin scripts from tensorflow github
# 
# ![inference](https://i.imgur.com/ybboFlP.png?1)
# 
# The predictions are okay but it may vary from device to device based on its resources(CPU,camera,memory)
# 
# Credits
# 
# https://www.tensorflow.org/lite
# 
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite
# 
# For more info 
# 
# https://github.com/anandsm7/simpsons_classifier_tflite
# 
# #### I know its a bit unconventional kernel from other data science kernel but i clearly hate seeing great models sitting on local machines.If there any mistakes/doubts please feel free to comment.
# 
# #### If you find this kernel useful please upvote 
# 
# 
