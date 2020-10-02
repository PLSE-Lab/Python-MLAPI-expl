#!/usr/bin/env python
# coding: utf-8

# # Image Classification - MobileNet v2 Transfer Learning

# In[ ]:


# Import Libraries
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from IPython.display import Image


# In[ ]:


DLDir = "../input/testdata1/"    #directory of downloaded files
fruit1 = 'Granny_Smith'  #keyword to be used in train model
fruit2 = 'banana'  #keyword to be used in train model


# ## Load Model (Keras built-in)

# In[ ]:


def prepare_image(filepath):
   img = image.load_img(filepath, target_size=(224, 224))
   img_array = image.img_to_array(img)
   img_array_expanded_dims = np.expand_dims(img_array, axis=0)
   return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# In[ ]:


import os
def chkFolder(path):
    try:
        os.listdir(path)
        return True
    except:
        return False
        
def validateImageFile(key , removeErr=True, prn = False) :
    #global DLDir #dd = '/kaggle/input/testdata1/applefood/'
    dd = DLDir + key
    fnames =[]
    errFName = []
    if(chkFolder(dd)):
        for fname in os.listdir(dd):
            _fname = dd  +'/' +fname
            try:
                Image(filename=_fname)
                if(prn):print('ok',_fname )
                fnames.append(_fname)
            except:
                if(removeErr):os.remove(_fname )
                print('error read file  fn', _fname)
    print ('done to validateImageFile',key )
    return fnames,errFName


# In[ ]:


#show image
import numpy as np # linear algebra
from PIL import Image as im
import matplotlib.pyplot as plt
def showImage(fname):
    img_array = np.array(im.open(fname))
    plt.imshow(img_array)


# In[ ]:


# Load Mobile
#@@ As creating model, one could meet error sometimes. it may be busy in server or slow in connecting speed
model = keras.applications.mobilenet_v2.MobileNetV2()


# ## Test Model

# In[ ]:


img_file='../input/testdata/Granny_Smith.jpg'
Image(filename=img_file)


# In[ ]:


# check model prediction
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
results = decode_predictions(predictions)
print(results[0])


# ### *Confirm the model recognize German Shepherd*

# In[ ]:


img_file='../input/images1/banana.jpg'
Image(filename=img_file)


# In[ ]:


# check model prediction
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
results = decode_predictions(predictions)
print(results)


# ## Download Images from Internet

# ### Install google_images_download

# In[ ]:


def funDownLoadImage( key , print_urls = False , print_size = False, print_paths = False):
    get_ipython().system('pip install google_images_download')
    from google_images_download import google_images_download
    global DLDir #dd = '/kaggle/input/testdata1/applefood/'
    dd = DLDir 
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":key,"limit":30,
                 "print_urls":print_urls,"print_size":print_size,"print_paths":print_paths,"format":"jpg", "size":">400*300",
                 "output_directory" : dd
                }
    paths = response.download(arguments)
    return paths


# ### Download fruit1 images

# In[ ]:


#first check if file exist already before downloading
fnames,errFName = validateImageFile(fruit1 )
if(len(fnames)<=0) : 
    paths = funDownLoadImage(fruit1)
    validateImageFile(fruit1 , False ) #validate again


# ### Download fruit2 food images

# In[ ]:


#downloading #2
fnames,errFName = validateImageFile(fruit2 )
if(len(fnames)<=0) : 
    paths = funDownLoadImage(fruit2)
    validateImageFile(fruit2 , False ) #validate again


# In[ ]:


#R


# In[ ]:


#DLDir = "../input/testdata1/"    #directory of downloaded files

#!ls -l ../input/testdata1/Granny_Smith


# In[ ]:


# remove files of URLError / Wrong image format
'''
!rm ./downloads/apple/15.*
!rm ./downloads/apple/2.*
!rm ./downloads/apple/8.*
!rm ./downloads/apple/4.*
!rm ./downloads/apple/14.*
!rm ./downloads/apple/13.*
!rm ./downloads/apple/16.*
!rm ./downloads/apple/17.*
!rm ./downloads/apple/19.*
!rm ./downloads/apple/20.*
# rename files with special characters
#!mv ./downloads/"apple food/2.* ./downloads/"apple"/2.blue-tit.jpg
#!mv ./downloads/apple/18.* ./downloads/"apple"/18.blue-tit.jpg
'''


# [](http://)[](http://)### Download Water pear Images

# In[ ]:



'''
'# remove files of URLError / Wrong image format
!rm ./downloads/banana/10.*.jpg
!rm ./downloads/banana/14.*.jpg
!rm ./downloads/banana/15.*.jpg
# rename files with special characters
!mv ./downloads/banana/12.* ./downloads/banana/13.Waterpear.jpg
'''


# In[ ]:


# Data Generator
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory(DLDir,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=True)


# In[ ]:


num_classes = 2
prediction_dict = {0: fruit1, 1: fruit2}
#prediction_dict = {1: "apple food", 0: "Water pear"}


# In[ ]:


# Load Model (MobieNet V2)
base_model=keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3),weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

# Add Extra Layers to Model
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(num_classes,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)


# In[ ]:


# Check layers no. & name
for i,layer in enumerate(model.layers):
    print(i,layer.name)


# In[ ]:


# set extra layers to trainable (layer #155~159)
for layer in model.layers[:155]:
    layer.trainable=False
for layer in model.layers[155:]:
    layer.trainable=True

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


# Train Model (target is loss <0.01)
num_epochs = 20
step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=num_epochs)


# In[ ]:


#showImage('../input/testdata/Granny_Smith.jpg')


# > ## Test New Model

# In[ ]:


#img_file='../input/images22/apple.jpg'
img_file='../input/testdata/Granny_Smith.jpg'
Image(filename=img_file)


# In[ ]:


# Test the new model
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex],prediction_dict[maxindex])


# In[ ]:


img_file='../input/images2/applef.jpg'
Image(filename=img_file)


# In[ ]:


# Test the new model
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex],prediction_dict[maxindex])


# In[ ]:


img_file='../input/images/German_Shepherd.jpg'
Image(filename=img_file)


# In[ ]:


# Test the new model
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
print(predictions[0])


# In[ ]:


R


# In[ ]:


# remove downloaded images
get_ipython().system('rm -rf ../input/testdata1/')

