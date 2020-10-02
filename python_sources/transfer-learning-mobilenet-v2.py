#!/usr/bin/env python
# coding: utf-8

# # Image Classification - MobileNet v2 Transfer Learning

# In[ ]:


# Immport Libraries
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from IPython.display import Image


# ## Load Model (Keras built-in)

# In[ ]:


# Load Mobile
model = keras.applications.mobilenet_v2.MobileNetV2()


# In[ ]:


def prepare_image(filepath):
   img = image.load_img(filepath, target_size=(224, 224))
   img_array = image.img_to_array(img)
   img_array_expanded_dims = np.expand_dims(img_array, axis=0)
   return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# ## Test Model

# In[ ]:


img_file='../input/xi-pooh/pooh.jpg'
Image(filename=img_file)


# In[ ]:


# check model prediction
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
results = decode_predictions(predictions)
print(results)


# ### *Confirm the model recognize German Shepherd*

# In[ ]:


img_file='../input/xi-pooh/xi.jpg'
Image(filename=img_file)


# In[ ]:


# check model prediction
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
results = decode_predictions(predictions)
print(results)


# ### *Confirm the model don't recognize Blue Tit*

# ## Download Images from Internet

# ### Install google_images_download

# In[ ]:


get_ipython().system('pip install google_images_download')


# In[ ]:


# import google_images_download
from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()


# ### Download Pooh images

# In[ ]:


arguments = {"keywords":"pooh","limit":20,"print_urls":False,"format":"jpg", "size":">400*300"}
paths = response.download(arguments)


# In[ ]:


# remove files of URLError / Wrong image format
get_ipython().system('rm ./downloads/"pooh"/8.*')
get_ipython().system('rm ./downloads/"pooh"/15.*')
get_ipython().system('rm ./downloads/"pooh"/19.*')
get_ipython().system('rm ./downloads/"pooh"/17.*')
get_ipython().system('rm ./downloads/"pooh"/4.*')
get_ipython().system('rm ./downloads/"pooh"/5.*')
# rename files with special characters
get_ipython().system('mv ./downloads/"pooh"/2.* ./downloads/"pooh"/2.pooh.jpg')
get_ipython().system('mv ./downloads/"pooh"/18.* ./downloads/"pooh"/18.pooh.jpg')


# ### Download Xi Images

# In[ ]:


arguments = {"keywords":"xi","limit":20,"print_urls":False, "format":"jpg", "size":">400*300"}
paths = response.download(arguments)


# In[ ]:


# remove files of URLError / Wrong image format
get_ipython().system('rm ./downloads/xi/2.*.jpg')
get_ipython().system('rm ./downloads/xi/18.*.jpg')
get_ipython().system('rm ./downloads/xi/17.*.jpg')
get_ipython().system('rm ./downloads/xi/20.*.jpg')
# rename files with special characters
get_ipython().system('mv ./downloads/xi/1.* ./downloads/xi/1.xi.jpg')


# In[ ]:


# Data Generator
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory('./downloads',
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=True)


# In[ ]:


num_classes = 2
prediction_dict = {0: "pooh", 1: "xi"}


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
num_epochs = 10
step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=num_epochs)


# > ## Test New Model

# In[ ]:


img_file='../input/xi-pooh/pooh.jpg'
Image(filename=img_file)


# In[ ]:


# Test the new model
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex],prediction_dict[maxindex])


# In[ ]:


img_file='../input/xi-pooh/xi.jpg'
Image(filename=img_file)


# In[ ]:


# Test the new model
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex],prediction_dict[maxindex])


# In[ ]:


img_file='../input/xi-pooh/pooh.jpg'
Image(filename=img_file)


# In[ ]:


# Test the new model
preprocessed_image = prepare_image(img_file)
predictions = model.predict(preprocessed_image)
print(predictions[0])


# In[ ]:


# remove downloaded images
get_ipython().system('rm -rf ./downloads')

