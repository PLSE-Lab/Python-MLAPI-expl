#!/usr/bin/env python
# coding: utf-8

# **Tensorflow is one of the top deep learning libraries by google. Originally I used Pytorch as my major 
# but recently I decided to explore tensorflow because it's abit smooth to deploy trained models to mobile.
# It relies on Keras as its high level API. In this code walkthrough am going to use [TFHub](https://www.tensorflow.org/tutorials/images/hub_with_keras) with tranfer learning to classify fruits.**
# 
# I am going to use [Fruit dataset](https://www.kaggle.com/moltean/fruits).
# 
# **Lets dive in **

# In[ ]:


# Import Important Libraries
import os # To investigate the data

#Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pathlib
#plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Mathematical operation
import numpy as np


# In[ ]:


# data directories
train_root = "../input/fruits-360_dataset/fruits-360/Training"
test_root = "../input/fruits-360_dataset/fruits-360/Test"


# In[ ]:


# Investigate our data to acetain that number of classes in training ser equal testing set
len(os.listdir(train_root)),len(os.listdir(test_root))


# In[ ]:


# load a sample image
image_path = train_root + "/Apple Braeburn/0_100.jpg"
def image_load(image_path):
    loaded_image = image.load_img(image_path)
    image_rel = pathlib.Path(image_path).relative_to(train_root)
    print(image_rel)
    return loaded_image


# In[ ]:


image_load(image_path)


# In[ ]:


# Loading the data into our model
train_generator = ImageDataGenerator(rescale=1/255) # Training set
test_generator = ImageDataGenerator(rescale=1/255) # Testing set

train_image_data = train_generator.flow_from_directory(str(train_root),target_size=(224,224))
test_image_data = test_generator.flow_from_directory(str(test_root), target_size=(224,224))


# TensorFlow Hub also distributes models without the top classification layer, we wil be using
# "feature_extractor" for our transfer learning.

# In[ ]:


# Model url
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"


# In[ ]:


# Create the model and check expected Image size
def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
IMAGE_SIZE


# In[ ]:


# Check the images shape for train set
for image_batch, label_batch in train_image_data:
    print("Image batch shape:",image_batch.shape)
    print("Label batch shape:",label_batch.shape)
    break


# In[ ]:


# Check the images shape for testing set
for test_image_batch, test_label_batch in test_image_data:
    print("Image batch shape:",test_image_batch.shape)
    print("Label batch shape:",test_label_batch.shape)
    break


# In[ ]:


# Wrap the the module in a keras layer
feature_extractor_layer = layers.Lambda(feature_extractor,input_shape=IMAGE_SIZE+[3])


# In[ ]:


# Freeze the variables in the feature extractor so that training only modifies the new classifier layer
feature_extractor_layer.trainable = False


# In[ ]:


# Attach a classification head
model = Sequential([
    feature_extractor_layer,
    layers.Dense(train_image_data.num_classes, activation = "softmax")
    ])
model.summary()


# In[ ]:


# Initialize the TFHub module
sess = K.get_session()
init = tf.global_variables_initializer()

sess.run(init)


# In[ ]:


# Test a siingle batch to see that the result comes back with the expected shape
result = model.predict(image_batch)
result.shape


# # Training the model

# In[ ]:


# Compile the model with an optimizer
model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = "categorical_crossentropy",
    metrics = ['accuracy']
    )


# In[ ]:


# create a custom callback to visualize the training progress during every epoch
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []
    
  def on_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    
# Implementing early stopping to stop the training if the loss starts to increase and also avoid overvitting
es = EarlyStopping(patience=2,monitor="val_loss")


# In[ ]:


# Calculate appropriate steps per epoch
steps_per_epoch = train_image_data.samples//train_image_data.batch_size
steps_per_epoch


# In[ ]:


# Using CallBacks to record accuracy and loss
batch_stats = CollectBatchStats()
# fit model
model.fit((item for item in train_image_data), epochs = 3,
         steps_per_epoch=1528,
         callbacks = [batch_stats, es],validation_data=test_image_data)


# # Wow we achieving a very high accuracy

# In[ ]:


# Visualize the results
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.plot(batch_stats.batch_losses)
plt.savefig("Model Loss")

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.plot(batch_stats.batch_acc)
plt.show()


# # Cheking predictions

# In[ ]:


# Get the ordered list of labels
label_names = sorted(train_image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
label_names


# In[ ]:


# Run predictions for the test patch
result_batch = model.predict(test_image_batch)

labels_batch = label_names[np.argmax(result_batch, axis=-1)]
labels_batch


# In[ ]:


# Show predicted results
plt.figure(figsize=(13,10))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(test_image_batch[n])
  plt.title(labels_batch[n])
  plt.axis('off')
  plt.suptitle("Model predictions")


# In[ ]:


# Save model for later 
model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
print(model_path)


# **Be sure to commit your code to save outputs**

# # *What next*
# 
# **Pick an image dataset and try it out**
# 
# # **Good Luck**
