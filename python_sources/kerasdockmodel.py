#!/usr/bin/env python
# coding: utf-8

# OpenSprayer 
#  Keras mobilnet model
#  

# In[ ]:


import matplotlib.pyplot as plt
import os
import inspect
import numpy as np
import pandas as pd


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model,model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils as kutils


# from keras.applications.mobilenet import MobileNet

# base_model = MobileNet(input_shape=(224,224,3), 
#                     alpha=1.0, 
#                     depth_multiplier=1, 
#                     dropout=1e-3, 
#                     include_top=True, 
#                     weights='imagenet', 
#                     input_tensor= None, 
#                     pooling=None)

# model = MobileNet(  input_shape=(256,256,3), 
#                     alpha=1.0, 
#                     depth_multiplier=1, 
#                     dropout=1e-3, 
#                     include_top=True, 
#                     weights=None, #'imagenet', 
#                     input_tensor= None, 
#                     pooling=None, classes=2)

# Rather than build from a topless version of the base_model, I just transfer relevant weights.
# 
# Inspecting them shows the last two are for 1000 classes instead of my 2, so trim them:

# base_weights = base_model.get_weights()
# my_weights = model.get_weights()
# base_weights[-2] = base_weights[-2][:,:,:,0:2]
# base_weights[-1] = my_weights[-1]

# model.set_weights(base_weights)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# In[ ]:


model = load_model("../input/models-trained-on-docknet-data/MobXDock3.hd5")


# In[ ]:


docnet_train_path = "../input/open-sprayer-images/docknet/Docknet/train"
docnet_valid_path = "../input/open-sprayer-images/docknet/Docknet/valid"


# In[ ]:


docks = [docnet_train_path + "/docks/" + fn for fn in os.listdir(docnet_train_path + "/docks")]
docks += [docnet_valid_path + "/docks/" + fn for fn in os.listdir(docnet_valid_path + "/docks")]
not_docks = [docnet_train_path + "/notdocks/" + fn for fn in os.listdir(docnet_train_path + "/notdocks")]
not_docks += [docnet_valid_path + "/notdocks/" + fn for fn in os.listdir(docnet_valid_path + "/notdocks")]


# In[ ]:


dock_df = pd.DataFrame()
dock_df['image_path'] = docks + not_docks


# In[ ]:


dock_df['weed'] = ['no' if nd else 'yes' for nd in dock_df['image_path'].str.contains('notdocks')]


# There was one oversized image in the not_docks directory. I have not excluded it here, 
# (the keras pre-processor resizes it, I think),
# but it ought to be.

# In[ ]:


not_docks = dock_df[dock_df['weed'].str.contains('no')]
docks = dock_df[dock_df['weed'].str.contains('yes')]
print(len(not_docks), len(docks))


# Grab a sampling of the not_docks to get a ballanced set for fitting

# In[ ]:


fit_these = not_docks.sample(len(docks))
fit_these = fit_these.append(docks)
len(fit_these)


# Checkout the ImageDataGenerator before using it! I had been using the brightness range,
# 
# and getting just black images. Here I apply it to 8 samples to see what it does.
# 
# I think a cropped version would be better. Maybe later.

# In[ ]:


datagen=ImageDataGenerator(
    rescale=1./255.,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 5,
    shear_range = 5,
    zoom_range = (0.90,1.10),
    fill_mode = "constant",
    cval = 0,
    validation_split = 0.0
    )


# In[ ]:


for image_path in dock_df.sample(8).image_path:
    ima = plt.imread(image_path)
    txfm = datagen.get_random_transform(np.shape(ima))
    imt = datagen.apply_transform(ima,txfm)
    plt.subplot(121)
    plt.imshow(ima)
    plt.subplot(122)
    plt.imshow(imt)
    plt.show()
    


# In[ ]:


train_generator = datagen.flow_from_dataframe(
    fit_these,
    x_col = 'image_path',
    y_col = 'weed',
    target_size = (256,256),
    color_mode="rgb")


# For anything with more classes, keep track of them. Write this dictionary along with the model. 

# In[ ]:


cd = train_generator.class_indices
ivd = {v: k for k, v in cd.items()}
cddf = pd.DataFrame.from_dict(cd,orient='index')
cd


# Fitting runs my 1080ti to almost 100%. I5 CPU isn't too stressed tho.
# 
# So 10 epochs took about 5 min.
# 
# MobXDock3 trained on three 10 epoch runs with different samples of the non_dock data,
# to get an accuracy of 99, with no false-negatives.

# In[ ]:


history = model.fit_generator(train_generator, epochs=1,
    use_multiprocessing = False,
    verbose=1,shuffle=True
    )


# In[ ]:


#plt.plot(history.history['accuracy'])
#plt.show()


# In[ ]:


#model.save("MobXDockDemo.hd5")


# In[ ]:




