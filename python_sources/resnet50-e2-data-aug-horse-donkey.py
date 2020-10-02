#!/usr/bin/env python
# coding: utf-8

# # Using Data Augmentation, transfer-learning Resnet50 on small datasets to classify horse vs donkey images.
# 
# **Inspired by Lesson 5 in the [Deep Learning](https://www.kaggle.com/learn/deep-learning) track**  
# This shows how to build an image classification model really quickly by using transfer-learning from Resnet50. With a very small image dataset (hundreds of samples), we use data augmentation. The model trains extremely quickly, enabling building better classifiers than just reusing an imagenet-trained model. Remember that top-1 accuracy is 0.79 for Resnet50. By building a custom layer on top of Resnet50, we can get much better accuracy, up to .96.
# 
# Previously, this approach had very good results on differentiating rural/urban images: 95% val acc with just 100 images and 2 epochs of training. 
# 
# But the rural/urbal classification is quite a coarse classification task. Let's try differentiating between more fine-grained objects: horses vs donkeys. This is a more challenging task, similar to classifying dog breeds.
# 
# ## How to create a dataset using Google Images
# The necessary horse/donkey image dataset is already attached to this notebook. Here's how it was created. Google Images is a convenient source of images.
# 
# ### Download image URLs from Google Images
# Use Advanced Image Search, search for minimum 400x300 images, Photos only, JPEG only, wide aspect. Use this js snippet to get their URLs (must be copy pasted in the Dev Tools Console)
# 
# ```
# var cont=document.getElementsByTagName("body")[0];
# var imgs=document.getElementsByTagName("a");
# var i=0;var divv= document.createElement("div");
# var aray=new Array();var j=-1;
# while(++i<imgs.length){
#     if(imgs[i].href.indexOf("/imgres?imgurl=http")>0){
#       divv.appendChild(document.createElement("br"));
#       aray[++j]=decodeURIComponent(imgs[i].href).split(/=|%|&/)[1].split("?imgref")[0];
#       divv.appendChild(document.createTextNode(aray[j]));
#     }
#  }
# cont.insertBefore(divv,cont.childNodes[0]);
# ```
# 
# Now copy-paste the URLs in a file named horses_urls. Download the images using `wget -i horses_urls`. Now, filter and delete bad images (paintings, drawings, large groups, close-up portraits). This step is optional. Repeat for the other class, donkey instead of horse and we'll have our dataset: 400 images of horse and donkey each.
# 
# To resize and crop the downloaded images to a square format:
# 
# ```
# # To rename files sequentially, as downloaded files tend to have messy names
# find . -maxdepth 1 -type f | cat -n | while read n f; do mv "$f" `printf "PREFIX%04d.jpg" $n`; done
# # To find non-jpeg files
# find . -maxdepth 1 -type f | cat -n | while read n f; do file $f; done | grep -v "JPEG image"
# # And to delete them
# find . -maxdepth 1 -type f | cat -n | while read n f; do file $f; done | grep -v "JPEG image" | grep -o -E "\w+[0-9]{3}\.jpg" | xargs -I _ rm _
# magick mogrify -resize "480x480^" *
# mogrify -crop 480x480+0+0 -gravity center *
# 
# ### To move 45 random images to the validation dataset
# mkdir val  # make sure to create the dir, otherwise the mv will overwrite the files!
# find . -maxdepth 1 -type f | gsort -R | tail -50 | xargs -I _ mv _ val
# ```
# 
# ### Train / validation split
# I have kept 360 images in the training dataset and 40 in the validation dataset.
# 
# Update: I have also created an unfiltered dataset. It is created with the same method as above, but I did not clean manually the "bad" images. However, I did use type: Photo, format: JPG and size: at least 400x300 as options for Google Advanced Image Search. This removes many of the bad images right from the start. I split the data set into 800 training images and 200 validation. I did make sure the validation images are clean.
# 
# ### TODO try it with squeezenet
# 
# ## Setting up the necessary imports and the model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from tensorflow.python.keras.applications import ResNet50\nfrom tensorflow.python.keras.models import Sequential\nfrom tensorflow.python.keras.layers import Dense, Flatten, Dropout\n\nnum_classes = 2\n# This is the same as using "weights=\'imagenet\'" when calling ResNet50()\nresnet_weights_path = \'../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5\'')


# In[ ]:


get_ipython().run_cell_magic('time', '', "my_new_model = Sequential()\n\nbase_model = ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)\nmy_new_model.add(base_model)\n\n# Some more fully-connected layers help, but too many will stop the network fom converging.\n#my_new_model.add(Dense(64, activation='relu'))\n#my_new_model.add(Dropout(0.66))\n#my_new_model.add(Dense(16, activation='relu'))\n#my_new_model.add(Dropout(0.66))\nmy_new_model.add(Dense(num_classes, activation='softmax'))\n\n# Do not train first layer (ResNet) model. It is already trained.\nmy_new_model.layers[0].trainable = False\n\nmy_new_model.summary()")


# We can see that we have a quite small number of trainable parameters, 4096. This makes training possible on our small dataset and using a CPU. Of course, our model contains a Resnet50, with its 23 million already-trained parameters.

# In[ ]:


from tensorflow.python.keras.optimizers import Adam

# Adam optimizer works even better than rmsprop
my_new_model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


# ### Fitting a Model With Data Augmentation
# We'll use the Keras ImageDataGenerator to apply data augmentation: this will increase model performance because it increases the number of training images, by synthetizing new training images from the existing ones. It does this by applying transformations (horizontal flips, shifts) that leave the label invariant, unchanged.

# In[ ]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

train_generator = data_generator_with_aug.flow_from_directory(
        '../input/horse_donkey_photos/horse_donkey_photos/horse_donkey_photos/train',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical')

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = data_generator_no_aug.flow_from_directory(
        '../input/horse_donkey_photos/horse_donkey_photos/horse_donkey_photos/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

hist = my_new_model.fit_generator(
        train_generator,
        #steps_per_epoch=3,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=40/32)


# This task gets a lower (~82%) accuracy after just two-epochs than the rural/urban classification. This is expected, given the tasks more fine-grained nature. It can be argued that all our donkey and horse images are a subset of the earlier rural images concept. Naturally, learning subconcepts is more work.

# In[ ]:


hist = my_new_model.fit_generator(
        train_generator,
        #steps_per_epoch=3,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=2)


# We go back to 95% accuracy if we train for one more epoch. If you see a lower accuracy, that's the luck of the draw with the random parameter initialization and the random mini-batch selection. Here, the fourth epoch overfits and the val accuracy lowers.
# 
# ## Let's try some finetuning.
# This will take a longer time, however, as we increase the number of trainable parameters! Better to do this on a GPU. Still, I've seen 8 min per epoch on a VCPU.

# In[ ]:


my_new_model.layers[0].trainable = True
for layer in base_model.layers:
   layer.trainable = False
# Except the very last few layers
for layer in base_model.layers[-5:]:
   layer.trainable = True
my_new_model.summary()
my_new_model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

hist = my_new_model.fit_generator(
        train_generator,
        #steps_per_epoch=3,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=2)


# Note the increase in "Trainable params".

# In[ ]:


hist = my_new_model.fit_generator(
        train_generator,
        #steps_per_epoch=3,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=1)


# ### Finetuning success!
# This brings us to .9688, a very nice result, after 4 epochs.
# 
# We can compare to related work such as https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html, which achieved .94 after finetuning Inception for 50 epochs. We can attribute this to our choice of finetuning Resnet50, a better solution to the imagenet task.

# ## Unfiltered dataset
# As explained above, the horse_donkey_unfiltered_photos has been created with an improved method, but only the validation set is filtered to remove obviously bad images.
# 
# This will test if CNNs are robust to "dirty" training datasets. If yes, this opens the very interesting possibility of automatically building image datasets using Google Image Search (Advanced).

# In[ ]:


get_ipython().run_cell_magic('time', '', "my_new_model = Sequential()\n\nbase_model = ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)\nmy_new_model.add(base_model)\n\n# Some more fully-connected layers help, but too many will stop the network fom converging.\n#my_new_model.add(Dense(64, activation='relu'))\n#my_new_model.add(Dropout(0.66))\n#my_new_model.add(Dense(16, activation='relu'))\n#my_new_model.add(Dropout(0.66))\nmy_new_model.add(Dense(num_classes, activation='softmax'))\n\n# Do not train first layer (ResNet) model. It is already trained.\nmy_new_model.layers[0].trainable = False\n\nmy_new_model.summary()")


# In[ ]:


from tensorflow.python.keras.optimizers import Adam

# Adam optimizer works even better than rmsprop
my_new_model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

train_generator = data_generator_with_aug.flow_from_directory(
        '../input/horses-vs-donkeys-unfiltered/horse_donkey_unfiltered_photos/horse_donkey_unfiltered_photos/train',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical')

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = data_generator_no_aug.flow_from_directory(
        '../input/horses-vs-donkeys-unfiltered/horse_donkey_unfiltered_photos/horse_donkey_unfiltered_photos/val',
        target_size=(image_size, image_size),
        class_mode='categorical')


# In[ ]:


hist = my_new_model.fit_generator(
        train_generator,
        #steps_per_epoch=3,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=200/32)


# In[ ]:


hist = my_new_model.fit_generator(
        train_generator,
        #steps_per_epoch=3,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=200/32)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'hist = my_new_model.fit_generator(\n        train_generator,\n        #steps_per_epoch=3,\n        epochs=2,\n        validation_data=validation_generator,\n        validation_steps=200/32)')


# ### Conclusion: great success. Good validation accuracy, .94
# But with a twice as big training set (812) and a 6x bigger validation set (200) and 3 epochs of training. Starts to overfit after the 3rd epoch. 
# 
# The hypothesis that CNNs are robust regarding a dirty dataset is supported. There is a small loss of accuracy, about 1-2%.
# 
