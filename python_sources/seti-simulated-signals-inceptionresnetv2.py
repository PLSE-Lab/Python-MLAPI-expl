#!/usr/bin/env python
# coding: utf-8

# If you haven't already, you might want to check out [SETI Simulated Signals - Data Exploration](https://www.kaggle.com/tentotheminus9/seti-simulated-signals-data-exploration) first.

# # AI, Meet ET

# SETI have been getting serious about AI [recently](http://seti.berkeley.edu/frb-machine/). After all, what better solution to the problem of too much data?
# 
# This kernel takes a Keras pre-trained model (**InceptionResNetV2**) and trains it on the PNGs produced from the SETI simulated 'primary small' dataset.
# 
# Let's start by loading the required modules,

# In[ ]:


import os
from keras import applications
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing import image
import numpy as np
import math
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')


# Create a few constants,

# In[ ]:


train_dir = '../input/seti-data/primary_small/train/'
validation_dir = '../input/seti-data/primary_small/valid/'
test_dir = '../input/seti-data/primary_small/test/'

img_dim  = 197


# Now, let's use Keras to train the model. The first step is to create some data generators to flow the data into the training process. I'm going to use some **data augmentation** to expand the dataset. Note ... some of the signal types look very similar to each other. For example, 'narrowbanddrd' is basically just a slightly curved version of 'narrowband'. I suspect that some augmentation transformations such as skewing could make one type look like another, so beware!

# In[ ]:


#Generators
train_datagen = ImageDataGenerator(
  rotation_range = 180,
  horizontal_flip = True,
  vertical_flip = True,
  fill_mode = "reflect")

# Note that the validation data shouldn't be augmented!
validation_datagen = ImageDataGenerator()  
test_datagen = ImageDataGenerator()  


# In[ ]:


training_batch_size = 64
validation_batch_size = 64

train_generator = train_datagen.flow_from_directory(
  train_dir,                                                  
  classes = ('noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 
            'squigglesquarepulsednarrowband', 'brightpixel'),
  target_size = (img_dim, img_dim),            
  batch_size = training_batch_size,
  class_mode = "categorical",
  shuffle = True,
  seed = 123)


# In[ ]:


validation_generator = validation_datagen.flow_from_directory(
  validation_dir,
  classes = ('noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 
            'squigglesquarepulsednarrowband', 'brightpixel'),
  target_size = (img_dim, img_dim),
  batch_size = validation_batch_size,
  class_mode = "categorical",
  shuffle = True,
  seed = 123)


# In[ ]:


test_size = 700
test_batch_size = 1

test_generator = test_datagen.flow_from_directory(
  test_dir,
  classes = ('noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 
            'squigglesquarepulsednarrowband', 'brightpixel'),
  target_size = (img_dim, img_dim),
  batch_size = test_batch_size,
  class_mode = "categorical",
  shuffle = False)


# Next, load the [InceptionResNetV2 model from keras](https://keras.io/applications/). Note that you have to import the weights as as separate 'dataset' in Kaggle,

# In[ ]:


base = InceptionResNetV2(
  weights = '../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
  include_top = False,
  input_shape = (img_dim, img_dim, 3)
)


# I need to tweak this model to make predictions for the 7 signal categories. Notice that I didn't include the 'top' in the above step, the 'top' being the output layers in the original model. Below adds a bespoke top. Notice the 7-output **softmax** layer,

# In[ ]:


x = base.output
x = Flatten(input_shape=base.output_shape[1:])(x)
x = Dense(img_dim, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(7, activation="softmax")(x)


# Now, let's merge the 'base' with the custom 'top',

# In[ ]:


model = Model(inputs=base.input, outputs=x)


# I've played around with **transfer learning** using [ImageNet](http://image-net.org/) weights from various depths of the model, but performance wasn't great. Let's just allow the whole thing to be trained,

# In[ ]:


for layer in model.layers:
   layer.trainable = True


# Let's compile the model. I've not played around a great deal with the optimiser choice or hyperparameters, but this seems to work reasonable well,

# In[ ]:


model.compile(loss = "binary_crossentropy", optimizer = optimizers.rmsprop(lr=1e-4), metrics=["accuracy"])


# We can now train the model,

# In[ ]:


#Train

training_step_size = 64
validation_step_size = 32

history = model.fit_generator(
  train_generator,
  steps_per_epoch = training_step_size,
  epochs = 50,
  validation_data = validation_generator,
  validation_steps = validation_step_size,
  verbose = 0,
)


# Plot the performance,

# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Predict on the test set

# I held back some files to create a test set. Let's see how well the model does on it. The first step is to run the test set through the model,

# In[ ]:


predictions = model.predict_generator(test_generator, steps = test_size, verbose = 1)


# Then do some basic tidying of the predictions,

# In[ ]:


df = pd.DataFrame(predictions)


# In[ ]:


df['filename'] = test_generator.filenames


# In[ ]:


df['truth'] = ''
df['truth'] = df['filename'].str.split('/', 1, expand = True)


# In[ ]:


df['prediction_index'] = df[[0,1,2,3,4,5,6]].idxmax(axis=1)


# In[ ]:


df['prediction'] = ''
df['prediction'][df['prediction_index'] == 0] = 'noise'
df['prediction'][df['prediction_index'] == 1] = 'squiggle'
df['prediction'][df['prediction_index'] == 2] = 'narrowband'
df['prediction'][df['prediction_index'] == 3] = 'narrowbanddrd'
df['prediction'][df['prediction_index'] == 4] = 'squarepulsednarrowband'
df['prediction'][df['prediction_index'] == 5] = 'squigglesquarepulsednarrowband'
df['prediction'][df['prediction_index'] == 6] = 'brightpixel'


# In[ ]:


df.head()


# In order to see how well it's done, we can use a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix),

# In[ ]:


cm = confusion_matrix(df['truth'], df['prediction'])
cm


# In[ ]:


cm_df = pd.DataFrame(cm)


# In[ ]:


cm_df.columns = ['noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 
            'squigglesquarepulsednarrowband', 'brightpixel']


# In[ ]:


cm_df['signal'] = ('noise', 'squiggle', 'narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 
            'squigglesquarepulsednarrowband', 'brightpixel')


# In[ ]:


cm_df


# Finally, here is the overall accuracy,

# In[ ]:


accuracy = accuracy_score(df['truth'], df['prediction'])
accuracy


# The next obvious step is to improve this accuracy. More training data will be useful here along with changes to the model and hyperparameters. Deploying the consequent model on real-world SETI data would be then great to see. I hope to add such data soon.
