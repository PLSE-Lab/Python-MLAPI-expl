#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U tensorflow_hub')
get_ipython().system('pip install -U tensorflow_datasets')


# In[ ]:


import time
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from tensorflow.keras import layers


# In[ ]:


(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)
def format_image(image, label):
  # `hub` image modules exepct their data normalized to the [0,1] range.
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return  image, label

num_examples = info.splits['train'].num_examples

BATCH_SIZE = 32
IMAGE_RES = 224

train_batches      = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)


# In[ ]:


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))
feature_extractor.trainable = False

model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(2)
])
model.compile(
  optimizer='adam', 
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

EPOCHS = 3
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# Now that we've trained the model, we can save it as an HDF5 file, which is the format used by Keras. Our HDF5 file will have the extension '.h5', and it's name will correpond to the current time stamp.

# In[ ]:


t = time.time()

export_path_keras = "./{}.h5".format(int(t))
model.save(export_path_keras)


# # Load the Keras .h5 Model
# We will now load the model we just saved into a new model called reloaded. We will need to provide the file path and the custom_objects parameter. This parameter tells keras how to load the hub.KerasLayer from the feature_extractor we used for transfer learning.

# In[ ]:


reloaded = tf.keras.models.load_model(
  export_path_keras, 
  custom_objects={'KerasLayer': hub.KerasLayer})

reloaded.summary()


# In[ ]:


EPOCHS = 3
history = reloaded.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# # Export as SavedModel

# In[ ]:


t = time.time()
export_path_sm = "./{}".format(int(t))
tf.saved_model.save(model, export_path_sm)


# In[ ]:


reloaded_sm = tf.saved_model.load(export_path_sm)


# In[ ]:


t = time.time()

export_path_sm = "./{}".format(int(t))
tf.saved_model.save(model, export_path_sm)


# In[ ]:


reload_sm_keras = tf.keras.models.load_model(
  export_path_sm,
  custom_objects={'KerasLayer': hub.KerasLayer})

reload_sm_keras.summary()


# In[ ]:


image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
result_batch = model.predict(image_batch)
reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)
(abs(result_batch - reload_sm_keras_result_batch)).max()


# In[ ]:


get_ipython().system('zip -r model.zip {export_path_sm}')
try:
  from google.colab import files
  files.download('./model.zip')
except ImportError:
  pass

