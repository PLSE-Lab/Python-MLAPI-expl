#!/usr/bin/env python
# coding: utf-8

# ## Notes on transfer learning
# 
# **Transfer learning** is a technique that enables training powerful machine learning models on marginal amounts of data and compute by reusing weights learned by an existing complex model. Despite the scary-sounding name, technically speaking it really is that simple.
# 
# There's a lot of flexibility in how transfer learning can be applied. We may use an entire already-trained model as a subnet in our own neural network architecture. Or we can pick just a few layers from that model and use those. We may choose to make the model layers trainable or not trainable. If the model layers are marked untrainable we will not backpropogate on them, and so we will avoid a lot of additional computational work.
# 
# Transfer learning is a sort-of example of a pretraining technique: a way of determining relatively good weights for a model without performing the full computations from a random start. Other pretraining techniques include using [autoencoders](https://www.kaggle.com/residentmario/autoencoders) and [Restricted Boltzmann Machines](https://www.kaggle.com/residentmario/restricted-boltzmann-machines-and-pretraining). The difference is (basically) that transfer learning uses results that are already known, which can mean reusing Google-designed models or whatever that took thousands of dollars of compute to train...so it's something that's done quite often, in contrast with RBMs and autoencoders, which have fallen out of favor.
# 
# ### Demo
# 
# The following code is based on [this blog post](https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e) on transfer learning. No model is actually trained here; I'm just defining one to see how it works. Note that this code cell fails because Kaggle blocks Internet access by default and we need to download MobileNet before using it, but it will work locally and it will work if we use an Internet-enabled kernel.

# In[ ]:


import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


# imports the MobileNet model and discards the 1000-category output layer
# the output layer is not useful to us unless we happen to have the exact same 1000 classes!
base_model=MobileNet(weights='imagenet', include_top=False)


# define our own fully connected layers
# these will stack after the convolutional layers, which are borrowed from MobileNet
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
out=Dense(120,activation='softmax')(x)


# now define our model by chaining our custom output layers on the MobileNet convolutional core
model=Model(inputs=base_model.input, outputs=out)


# optionally set the first 20 layers of the network (the MobileNet component) to be non-trainable
# this means that we will use (the convolutional part of) MobileNet exactly
# which will speed up training. The alternative would be to continue to optimize these layers
for layer in model.layers[:20]:
    layer.trainable=False
    

# finally, compile the model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# now we are ready for fitting!


# ### How to make it work
# This "fine-tuning" strategy is presented by Francois Chollet in his blog post "[Building Powerful Image Classifiers with Very Little Data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)". That post does two things: it shows an alternative approach to doing the above, and shows how you do the above even further with "fine-tuning".
# 
# Instead of stacking the models and turning off learning on some layers, Chollet runs model data through the pretrained model's convolutional layers, then uses the raw outputs as raw inputs into the next model layers. He calls this approach "bottleneck features". I don't see this being notably advantageous, except in that it allows something more advanced: "fine-tuning".
# 
# Fine-tuning means unfreezing one (or potentially more) of the layers of convolutional part of the network and retraining those alongside the re-emplaced dense layers. To fine-tune properly:
# 
# * Use a slowly-converging algorithm and probably not an adaptive learning one.
# * Pre-train the topmost layer (by using the "bottleneck features" approach).
# 
# More details in the blog post.
# 
# TODO: apply transfer learning to something not Cat/Dogs myself.
