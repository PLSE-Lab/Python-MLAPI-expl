#!/usr/bin/env python
# coding: utf-8

# **[Deep Learning Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
# ---
# 

# In this exercise, you'll make your first submission to the [**Petals to the Metal**](https://www.kaggle.com/c/tpu-getting-started) competition.  You'll learn how to accept the competition rules, run a notebook on Kaggle that uses (free!) TPUs, and how to submit your results to the leaderboard.
# 
# We won't cover the code in detail here, but if you'd like to dive into the details, you're encouraged to check out the [tutorial notebook](https://www.kaggle.com/ryanholbrook/create-your-first-submission).
# 
# Begin by running the next code cell to set up the notebook.

# In[ ]:


get_ipython().system('pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git')
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.ex_tpu import *
step_1.check()


# If the code cell returns `Setup complete.`, then you're ready to continue.
# 
# # Join the competition #
# 
# Begin by joining the competition. Open a new window with the **[competition page](https://www.kaggle.com/c/tpu-getting-started)**, and click on the **"Rules"** tab.
# 
# This takes you to the rules acceptance page. You must accept the competition rules in order to participate. These rules govern how many submissions you can make per day, the maximum team size, and other competition-specific details. Click on **"I Understand and Accept"** to indicate that you will abide by the competition rules.
# 
# # Commit your Notebook #
# 
# **Committing** your notebook will run a fresh copy of the notebook start to finish, saving a copy of the `submission.csv` file as output.
# 
# First, click on the **Save Version** button in the upper right.
# 
# <figure>
# <img src="https://i.imgur.com/ebMUMSq.png" alt="The blue Save Version button." width=300>
# </figure>
# 
# Choose **Advanced Settings**.
# 
# <figure>
# <img src="https://i.imgur.com/sx9l1fL.png" alt="Advanced Settings in the Version menu." width=600>
# </figure>
# 
# Select **Run with TPU for this session** from the dropdown menu and click the blue **Save** button.
# 
# <figure>
# <img src="https://i.imgur.com/1cB5ykf.png" alt="The Accelerator dropdown menu." width=600>
# </figure>
# 
# Select **Save & Run All (Commit)** and click the blue **Save** button.
# 
# <figure>
# <img src="https://i.imgur.com/YeJLsNG.png" alt="The Save Version menu." width=600>
# </figure>
# 
# The commit may take a while to finish (about 10-15 min), but there's no harm in doing something else while it's running and coming back later.
# 
# This generates a window in the bottom left corner of the notebook. After it has finished running, click on the number to the right of the **Save Version** button. This pulls up a list of versions on the right of the screen. Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**. This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 
# # Make a Submission #
# 
# Now you're ready to make a submission! Click on the **Output** heading in the menu to the right of the notebook.
# 
# <figure>
# <img src="https://i.imgur.com/thKwt1q.png" alt="The Output heading." width=300>
# </figure>
# 
# And finally you'll submit the predictions! Just look for the blue **Submit** button. After clicking it, you should shortly be on the leaderboard!
# 
# <figure>
# <img src="https://i.imgur.com/j00mDeI.png" alt="The Save Version menu." width=600>
# </figure>
# 
# 

# # Code #
# 
# The code reproduces the code we covered together in **[the tutorial](https://www.kaggle.com/ryanholbrook/create-your-first-submission)**.  If you commit the notebook by following the instructions above, then the code is run for you.
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


# ## Explore the Data ##
# 
# Try using some of the helper functions described in the **Getting Started** tutorial to explore the dataset.

# In[ ]:


print("Number of classes: {}".format(len(CLASSES)))

print("First five classes, sorted alphabetically:")
for name in sorted(CLASSES)[:5]:
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
    pretrained_model = tf.keras.applications.ResNet50(
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
EPOCHS = 12
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
)


# Examine training curves.

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


# # Going Further #
# 
# Now that you've joined the **Petals to the Metal** competition, why not try your hand at improving the model and see if you can climb the ranks! If you're looking for ideas, the *original* flower competition, [Flower Classification with TPUs](https://www.kaggle.com/c/flower-classification-with-tpus), has a wealth of information in its notebooks and discussion forum. Check it out!

# ---
# **[Deep Learning Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
