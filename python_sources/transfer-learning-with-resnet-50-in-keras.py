#!/usr/bin/env python
# coding: utf-8

# For the general context, see  also:
# 
# * A deepsense.ai blog post [Keras vs. PyTorch - Alien vs. Predator recognition with transfer learning](https://deepsense.ai/keras-vs-pytorch-avp-transfer-learning) in which we compare and contrast Keras and PyTorch approaches.
# * Repo with code: [github.com/deepsense-ai/Keras-PyTorch-AvP-transfer-learning](https://github.com/deepsense-ai/Keras-PyTorch-AvP-transfer-learning).
# * Free event: [upcoming webinar (10 Oct 2018)](https://www.crowdcast.io/e/KerasVersusPyTorch/register), in which we walk trough the code (and you will be able to ask questions).
# 
# ### 1. Import dependencies

# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import Image


# In[ ]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json


# In[ ]:


keras.__version__  # should be 2.2.2


# In[ ]:


import tensorflow as tf
tf.__version__  # should be 1.10.x


# In[ ]:


import PIL
PIL.__version__  # should be 5.2.0


# In[ ]:


# path for Kaggle kernels
input_path = "../input/alien_vs_predator_thumbnails/data/"


# ### 2. Create Keras data generators 

# In[ ]:


train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    input_path + 'train',
    batch_size=32,
    class_mode='binary',
    target_size=(224,224))

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    input_path + 'validation',
    shuffle=False,
    class_mode='binary',
    target_size=(224,224))


# ### 3. Create the network

# In[ ]:


conv_base = ResNet50(
    include_top=False,
    weights='imagenet')

for layer in conv_base.layers:
    layer.trainable = False


# In[ ]:




x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x) 
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)


# Note:  there was an error with the above on Kaggle (even though it works on my computer, same versions of Keras and TF):
# 
# > AttributeError: 'Node' object has no attribute 'output_masks'
# 
# See [this issue](https://github.com/keras-team/keras/issues/10907).
# After reinstalling TensorFlow in Kaggle (packages -> tensorflow), no error.

# In[ ]:


optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# ### 4. Train the model

# In[ ]:


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=347 // 32,  # added in Kaggle
                              epochs=3,
                              validation_data=validation_generator,
                              validation_steps=10  # added in Kaggle
                             )


# ### 5. Save and load the model
# 
# Note: this is for demonstration. You don't need to to so, if you intend to run predictions within this notebook.

# In[ ]:


get_ipython().system('mkdir models')
get_ipython().system('mkdir models/keras')


# #### A. Architecture and weights in HDF5

# In[ ]:


# save
model.save('models/keras/model.h5')


# In[ ]:


# load
model = load_model('models/keras/model.h5')


# #### B. Architecture in JSON,  weights in HDF5

# In[ ]:


# save
model.save_weights('models/keras/weights.h5')
with open('models/keras/architecture.json', 'w') as f:
        f.write(model.to_json())


# In[ ]:


# load
with open('models/keras/architecture.json') as f:
    model = model_from_json(f.read())
model.load_weights('models/keras/weights.h5')


# ### 6. Make predictions on sample test images

# In[ ]:


validation_img_paths = ["validation/alien/11.jpg",
                        "validation/alien/22.jpg",
                        "validation/predator/33.jpg"]
img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]


# In[ ]:


validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                             for img in img_list])


# In[ ]:


pred_probs = model.predict(validation_batch)
pred_probs


# In[ ]:


fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
    ax.imshow(img)


# In[ ]:




