#!/usr/bin/env python
# coding: utf-8

# **[Deep Learning Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
# ---
# 

# Submission to the [**Petals to the Metal**](https://www.kaggle.com/c/tpu-getting-started) competition.  
# 

# In[ ]:


get_ipython().system('pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git')
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.ex_tpu import *
step_1.check()


# 
# ## Load Helper Functions ##

# In[ ]:


from petal_helper import *


# ## Create Distribution Strategy ##

# In[ ]:


# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)


# ## Loading the Competition Data ##

# In[ ]:


ds_train = get_training_dataset()
ds_valid = get_validation_dataset()
ds_test = get_test_dataset()

print("Training:", ds_train)
print ("Validation:", ds_valid)
print("Test:", ds_test)
ds_train


# ## Explore the Data ##
# 
# Try using some of the helper functions described in the **Getting Started** tutorial to explore the dataset.

# In[ ]:


print("Number of classes: {}".format(len(CLASSES)))

print("First 15 classes, sorted alphabetically:")
for name in sorted(CLASSES)[:15]:
    print(name)

print ("Number of training images: {}".format(NUM_TRAINING_IMAGES))


# Examine the shape of the data.

# In[ ]:


print("Training data shapes:")
for image, label in ds_train.take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())


# In[ ]:


print("Test data shapes:")
for image, idnum in ds_test.take(3):
    print(image.numpy().shape, idnum.numpy().shape)
print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string


# Peek at training data.

# In[ ]:


one_batch = next(iter(ds_train.unbatch().batch(20)))
display_batch_of_images(one_batch)


# ## Define Model #

# In[ ]:


with strategy.scope():
    pretrained_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False ,
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = False
    
    model = tf.keras.Sequential([
        # To a base pretrained on ImageNet to extract features from images...
        pretrained_model,
        # ... attach a new head to act as a classifier.
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512,activation = "relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

model.summary()


# ## Train Model ##

# In[ ]:


# Define the batch size. This will be 16 with TPU off and 128 with TPU on
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

# Define training epochs for committing/submitting. (TPU on)
EPOCHS = 14
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
)


# Plot of training curves...

# In[ ]:


display_training_curves(
    history.history['loss'],
    history.history['val_loss'],
    'loss',
    211,
)
display_training_curves(
    history.history['sparse_categorical_accuracy'],
    history.history['val_sparse_categorical_accuracy'],
    'accuracy',
    212,
)


# ## Validation ##
# 
# Create a confusion matrix.

# In[ ]:


cmdataset = get_validation_dataset(ordered=True)
images_ds = cmdataset.map(lambda image, label: image)
labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()
cm_probabilities = model.predict(images_ds)
cm_predictions = np.argmax(cm_probabilities, axis=-1)

labels = range(len(CLASSES))
cmat = confusion_matrix(
    cm_correct_labels,
    cm_predictions,
    labels=labels,
)
cmat = (cmat.T / cmat.sum(axis=1)).T # normalize


# In[ ]:


score = f1_score(
    cm_correct_labels,
    cm_predictions,
    labels=labels,
    average='macro',
)
precision = precision_score(
    cm_correct_labels,
    cm_predictions,
    labels=labels,
    average='macro',
)
recall = recall_score(
    cm_correct_labels,
    cm_predictions,
    labels=labels,
    average='macro',
)
display_confusion_matrix(cmat, score, precision, recall)


# Look at examples from the dataset, with true and predicted classes.

# In[ ]:


dataset = get_validation_dataset()
dataset = dataset.unbatch().batch(20)
batch = iter(dataset)


# In[ ]:


images, labels = next(batch)
probabilities = model.predict(images)
predictions = np.argmax(probabilities, axis=-1)
display_batch_of_images((images, labels), predictions)


# ## Test Predictions ##
# 
# Create predictions to submit to the competition.

# In[ ]:


test_ds = get_test_dataset(ordered=True)

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)


# In[ ]:


print('Generating submission.csv file...')

# Get image ids from test set and convert to integers
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')

# Write the submission file
np.savetxt(
    'submission.csv',
    np.rec.fromarrays([test_ids, predictions]),
    fmt=['%s', '%d'],
    delimiter=',',
    header='id,label',
    comments='',
)

# Look at the first few predictions
get_ipython().system('head submission.csv')


# 
# 
