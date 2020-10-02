#!/usr/bin/env python
# coding: utf-8

# # Hello World Deep Learning in Medical Imaging
# Adapted from Lakhani et al. "Hello World Deep Learning in Medical Imaging." Journal of Digital Imaging. 2018 Jun;31(3):283-289 ([link to article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5959832/)). Data obtained and notebook adapted from Paras Lakhani's GitHub repository ([link to repo](https://github.com/paras42/Hello_World_Deep_Learning/tree/9921a12c905c00a88898121d5dc538e3b524e520)) by Walter Wiggins (2018-2019). Code has been modified to match the format and directory structure from the article. Additional modifications were made with the intent to increase readability and facilitate understanding for participants at all levels of coding skill and experience.
# 
# To explore this model and data set, please **Fork** the notebook in the menu above (i.e. create a copy under your Kaggle profile). 
# 
# >When you get into the draft environment, please ensure that you see **"Internet connected"** and **"GPU on"** under **Settings** so you can download the weights for the model, below, and utilize the cloud GPU for faster model training. You will have to go through two-factor authentication (verify a code sent to you via text) when you connect for the first time.
# 
# In this Notebook editing environment, each block of text is referred to as a **cell**.  Cells containing formatted text are **Markdown** cells, as they use the *Markdown* formatting language. Similarly, cells containing code are **code** cells.
# 
# Clicking within a cell will allow you to edit the content of that cell (a.k.a. enter **edit mode**). You can also navigate between cells with the arrow keys. Note that the appearance of Markdown cells will change when you enter edit mode.
# 
# You can **run code cells** (and format Markdown cells) as you go along by clicking within the cell and then clicking the **blue button with *one* arrow** next to the cell or at the bottom of the window. You can also use the keyboard shortcut <kbd>SHIFT</kbd> + <kbd>ENTER</kbd> (press both keys at the same time).
# 
# `Hello, world!` is a time-honored tradition in learning a new programming language. It refers to the initial exercise in any book or tutorial, where one is taught the **syntax** for printing the message "Hello, world!" to the **programming environment**'s *output display*. The syntax for executing this task in Python is included below.

# In[ ]:


print("")    # insert the text "Hello, world!" between the quotes, then run the cell
"Hello world"


# ## 1. Loading Python modules
# For this experiment, we're using the [**Python** programming language](https://www.python.org/) with the [**TensorFlow**](https://www.tensorflow.org/) *backend* for model computation and the [**Keras**](https://keras.io/) **framework** for coding the model **architecture** (i.e. the layers and connections of the network). **Keras** can use multiple backends, so we first need to set an **environment variable** to declare that we will be using **TensorFlow**, then run a quick **Python** command to ensure the proper module is imported. This syntax is specific to utilizing the [**Jupyter**](https://jupyter.org/)-style **notebook** environment simulated here to run a **shell** command within the Kaggle kernel.

# In[ ]:


# This notebook is built around using tensorflow as the backend for keras
get_ipython().system('KERAS_BACKEND=tensorflow python -c "from keras import backend"')


# Next, we use the *standard Python syntax* to import the Keras **modules** we'll need to design our model and run our experiment. In Python lingo, a `module` is a bundled subset of related `functions` within a `package`. Here, `keras` is the package and `applications`, `optimizers`, etc. are the modules.
# 
# >Import the following modules from **Keras**:
# - `applications` -> contains the **InceptionV3** network architecture that we will use as the **base** or **backbone** of our model
# - `ImageDataGenerator` ->  generators are a handy tool for generating data in a memory-efficient manner, which can be used by the model from image files in a data directory...they also facilitate random **data augmentation**, which we'll discuss in more detail later
# - `Adam optimizer` ->  this is the algorithm that we will use to optimize the **weights** in the neural network model during model **training**
# - `Sequential Model`
# - `Model layer` types:
#     - `Dense`
#     - `Flatten`
#     - `GlobalAveragePooling2D`
#     - `Dropout`

# In[ ]:


# Import the appropriate Keras modules
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Taking a look at the data
# I've preloaded the data that we'll be using into this kernel. Let's take a look at the directory structure and contents, then create some variables to help us as we proceed.

# In[ ]:


import os

# Set path variable to the directory where the data is located
# hard-code path if adding Keras pre-trained models dataset (avoids error due to multiple datasets in 'input/')
path = ('../input/hello-world-deep-learning-siim/data/')
print(path)
# Command line "magic" command to show directory contents
#!ls {path}/*/*


# As you can see, the `data` directory contains subdirectories `train` and `val`, which contain the *training* and *validation* data for our experiment. Each of these directories contains subdirectories `abd` and `chst` containing abdominal and chest radiographs for each data set. There are 65 training images and 10 validation images with *balanced distributions* over our *target classes* (i.e. approximately equal numbers of abdominal and chest radiographs in each data set).

# In[ ]:


# set variables for paths to directories for training & validation data
# adjust join() call due to path changes:
train_dir = os.path.join(path + 'train')
val_dir = os.path.join(path + 'val')

# set variables for number of samples in each data set
num_train = 65
num_val = 10

# we'll need to import additional modules to look at an example image
import numpy as np    # this is a standard convention
from keras.preprocessing import image
import matplotlib.pyplot as plt    # also by convention

# these are the dimensions of our images
img_width, img_height = 299, 299

# set the path to a chest radiograph, then load it and show
img_path = os.path.join(train_dir, 'chst/chst_train_001.png')
img = image.load_img(img_path, target_size=(img_width, img_height))
plt.imshow(img)
plt.title('Example chest radiograph')
plt.show()

# set the path to an abdominal radiograph, then load it and show
img2_path = os.path.join(train_dir, 'abd/abd_train_001.png')
img2 = image.load_img(img2_path, target_size=(img_width, img_height))
plt.imshow(img2)
plt.title("Example abdominal radiograph")
plt.show()


# ## 3. Setting up our generators & selecting hyperparameters
# A neural network model consists of:
# - An **architecture** defining the structure of the connections between **layers** of **artificial *neurons*** in the network
#     - *Within a layer*, neurons are NOT connected to each other (with some exceptions, not relevant to this module)
#     - The connections that define the architecture are between neurons in *different layers* of the network
# - The **weights** (*or* parameters) that determine the strength of those connections.
# 
# When training a model, we choose from a wide array of different **hyperparameters** that govern different aspects of the training process. Choosing these well is important for avoiding the phenomenon of **overfitting**, where the network is optimized to features in the training data set that are unique to those data, but not representative of the broader phenomenon you're trying to capture with your network. These features can be thought of as confounding variables. See [this article](https://towardsdatascience.com/overfitting-vs-underfitting-a-conceptual-explanation-d94ee20ca7f9) for a more detailed explanation of overfitting (and it's counterpart, underfitting).
# 
# >Common hyperparameters include:
# - Number of training **epochs**: in each epoch of training, the model is exposed to each training sample exactly once
#     - The more a network is exposed to each training sample, the higher the likelihood it will overfit
#     - However, it is unlikely that you will achieve optimal performance after only one epoch of training
# - Optimization algorithm: typically, a variation of stochastic gradient descent
#     - Learning rate
#     - Batch size (i.e. number of samples to evaluate for each update of the model's weights)
# - Loss function: the measure of network performance used to adjust the model's weights in each training step
#     - Regularization: a method of penalizing large weights to avoid overfitting
# - Performance metric: the measure of network performance used to evaluate the network after each epoch of training is complete
# - Data augmentation: a method of transforming the data in a way that helps prevent overfitting, but preserves the information contained in the data
#     - Types of transformations
#     - Magnitude of a given transformation
#     - Probability of any given transformation occurring for a given sample
# 
# One of the more challenging aspects of training a network is selected the optimal hyperparameters. Here, each participant will train the network with different hyperparameters, so that we can compare results and try to get a sense of the effects they have on training results. Some will experiment with **batch size**. Others will experiment with **image augmentation**.

# In[ ]:


# randomize to groups 1 and 2
group = np.random.randint(1, 3)
print("You have been assigned to Group", group)


# If you've been assigned to Group 1, please leave the following code cell alone and run it in it's current state.
# 
# If you've been assigned to Group 2, please remove image augmentation in the code cell below the example by deleting all of the parameters between the parentheses, except `rescale=1./255`. Your `train_datagen` code should look like this
# ```python
# # EXAMPLE CODE -> NOT a functional code cell
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0,
#         zoom_range=0,
#         rotation_range=0,
#         width_shift_range=0,
#         height_shift_range=0,
#         horizontal_flip=False
# )
# ```
# 
# Make any edits in the code cell below this line.

# In[ ]:


# set the batch size for each training step
batch_size = 8

# create training data generator object
# initialize values for image augmentation
# rescaling is done to normalize the image pixel values into the [0, 1] range
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
)

print('Success!')


# Once you've made your edits and see "Success!" as your output, you may run the code cell below.

# In[ ]:


# finalize training generator
print("Training generator: ", end="")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# create validation data generator object
# no image augmentation here, as we are not training our model on this data
val_datagen = ImageDataGenerator(rescale=1./255)
print("Validation generator: ", end="")
val_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)


# ## 4. Creating the model with transfer learning
# We'll employ a technique called **transfer learning** with the `InceptionV3` model, using weights obtained by **pretraining** the model on the **ImageNet** data set.
# > If you receive an **Error message** on this step, it is most likely due to a faulty connection *between the Kaggle kernel and the Internet*. You should try **restarting the kernel** by clicking the **blue button with two arrows** at the bottom of this window, next to <kbd>Console</kbd>. Please also ensure that you are **connected the Internet** in the <kbd>Settings</kbd> panel to your right. (Note: you will need to rerun all previous code cells prior to running this one again.) If this *does NOT* solve your problem, then you can change `weights='imagenet', ` in the code cell below to `weights=None, `. This should allow you to train the model "from scratch" with randomly initialized weights. However, your performance will *likely suffer from the lack of pretraining*, among other things.
# 
# Alternatively, you can add a dataset to your draft environment and change the filepaths to load keras-pretrained-model weights:

# In[ ]:


# changed weights to dir for Keras-pretrained-models dataset 
#(1. click 'add data' in draft environment right panel, 
# 2. search for keras pretrained weights, 
# 3. copy file path for inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 in place of 'imagenet')
backbone = applications.InceptionV3(weights='../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the pretrained weights in the model backbone
for layer in backbone.layers:
    layer.trainable = False

print('Model backbone initialized!')    # this will print when this step is complete


# Here, we add custom top layers for our task to get the desired model output.
# 
# Adding a `GlobalAveragePooling2D` layer after the final **convolutional** layer in the `backbone` and a `Dropout` layer prior to the final layer are two additional methods employed to improve model performance and reduce overfitting. **Dropout** refers to the random elimination of the inputs from some proportion of the neurons in this layer.
# 
# After the architecture is established, the model must be **compiled** with the desired optimizer, loss function, and performance metric before training is initiated. Here, we use the Adam optimizer, binary cross-entropy as our loss function, and accuracy as our performance metric.
# 
# Some participants will be randomized to run their experiments **without** the `Dropout` layer, so that we can see the effect of dropout on our results.

# In[ ]:


# randomize to subgroups A & B
subgroup = np.random.randint(1, 3)
print("You have been assigned to Subgroup", "A" if subgroup == 1 else "B")


# If you've been assigned to Subgroup A, then please leave the following code cell alone and run it in its current state.
# 
# If you've been assigned to Subgroup B, then please **comment out** the line of code containing `Dropout`. This is done by placing a `#` prior to the line of code, such that it now should look like this. Alternatively, you can click anywhere on the line and use the keyboard shortcut <kbd>Ctrl</kbd> (<kbd>Cmd</kbd> on Mac) + <kbd>/</kbd> to *comment out the line*.
# 
# ```python
# # model_top.add(Dropout(0.5))
# ```
# 
# This will prevent the `Dropout` layer from being added to the model so that we may evaluate the effect of dropout on overfitting.

# In[ ]:


# create the top layers of the model
model_top = Sequential()    # initialize as a Sequential model (i.e. no recurrent layers)
model_top.add(GlobalAveragePooling2D(input_shape=backbone.output_shape[1:], data_format=None)) 
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(1, activation='sigmoid')) # the output will be a probability

# connect "model_top" to "backbone"
model = Model(inputs=backbone.input, outputs=model_top(backbone.output))

# compile the model with the Adam optimizer, binary cross-entropy loss, and accuracy
model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

print('Setup Complete!')


# ## 5. Training the model
# Now, we will **train the model**. Since this will take several minutes and all of the code needs to be run again when the kernel is *committed* (see the blue <kbd>Commit</kbd> button near the top right of your screen), we'll start running the cell to see what the output looks like as the model is training, and then move on to reviewing the few remaining cells before committing the notebook to return later after our model is finished.
# 
# We start by training for **a few epochs** with the backbone *frozen* to avoid propagating large losses through the pretrained layers. If we didn't do this, we could potentially **degrade the higher-level features** in the early layers of the model that were so carefully pretrained on ImageNet...and we wouldn't want to do that.
# 
# >If you receive an **Error message** on this step, it is likely that the GPU functionality of the kernel has failed. Please **restart the kernel** by clicking the **blue button with 2 arrows** at the bottom of this screen next to <kbd>Console</kbd>. (Note: you will have to rerun all previous code cells prior to running this one again.) If this *does NOT* solve your problem, then you should set **GPU off** in the <kbd>Settings</kbd> panel on the right-hand side of this screen and try again. This will force the kernel to use its CPU, which will be slower but will still train the model.

# In[ ]:


# set the number of epochs for training
epochs = 5

# train the model and save the training/validation results for each epoch to "history"
history = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=num_val // batch_size)

print('Training complete!')


# Now we will plot and evaluate the **training curves** for training the custom top layers of our model.

# In[ ]:


print(history.history.keys())

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['acc'], 'orange', label='Training accuracy')
ax[0].plot(history.history['val_acc'], 'blue', label='Validation accuracy')
ax[1].plot(history.history['loss'], 'red', label='Training loss')
ax[1].plot(history.history['val_loss'], 'green', label='Validation loss')
ax[0].legend()
ax[1].legend()
plt.show()


# You should see in the plot above that our training and validation **losses** went *down* over the 5 epochs of training, while our training and validation **accuracy** went *up*.
# 
# Now, we can **unfreeze the model backbone** and train a little more to see if we can improve our performance.

# In[ ]:


# Unfreeze the model backbone before we train a little more
for layer in model.layers:
    layer.trainable = True

# When you make a change to the model, you have to compile it again prior to training
model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

print('Setup Complete!')


# This should converge quickly, so we'll only train for **3 epochs** this time.

# In[ ]:


# set the number of epochs for training
epochs = 3

# train the model and save the training/validation results for each epoch to "history"
history = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=num_val // batch_size)

print('Training complete!')


# In[ ]:


print(history.history.keys())

plt.figure()
plt.plot(history.history['acc'], 'orange', label='Training accuracy')
plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')
plt.plot(history.history['loss'], 'red', label='Training loss')
plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
plt.show()


# ## 6. Evaluating training results and model performance
# 
# Once training is finished, **evaluating the training curves** can help us **tune our hyperparameters** in subsequent experiments by telling us if we're **underfitting** or by showing us when the model begins to **overfit**.
# 
# Finally, we will demonstrate the model output on examples from the validation data set. 
# 
# >In reality, at the end of your experiments, you would want to evaluate network performance on an **independent, *held-out* test data set**, as the model will be *indirectly* exposed to the validation data set over successive experiments, potentially **biasing the model hyperparameters** toward overfitting to the validation data.

# In[ ]:


# load example chest and abdominal radiographs from validation data set 
img_path = os.path.join(val_dir, 'chst/chst_val_001.png')
img_path2 = os.path.join(val_dir, 'abd/abd_val_001.png')
img = image.load_img(img_path, target_size=(img_width, img_height))
img2 = image.load_img(img_path2, target_size=(img_width, img_height))

# show the chest radiograph
plt.imshow(img)
plt.show()

# evaluate the chest radiograph with the model, then print the model's prediction
img = image.img_to_array(img)
x = np.expand_dims(img, axis=0) * 1./255
score = model.predict(x)
print('Predicted:', score[0][0], 'Chest radiograph' if score > 0.5 else 'Abdominal radiograph')

# show the abdominal radiograph
plt.imshow(img2)
plt.show()

# evaluate the abdominal radiograph with the model, then print the model's prediction
img2 = image.img_to_array(img2)
x = np.expand_dims(img2, axis=0) * 1./255
score2 = model.predict(x)
print('Predicted:', score2[0][0], 'Chest radiograph' if score2 > 0.5 else 'Abdominal radiograph')


# ## Congrats on completing this demo! 
# I hope it has fostered some interest in AI and Machine Learning for Radiology. At the very least, you hopefully know a little more about Machine Learning in Radiology than you did before.
# 
# If you're interested, feel free to come back to this kernel and play around with other hyperparameters to gain a little more insight into this topic.
# 
# If you really want to get involved in the Radiology AI/ML community, I encourage you to join the [Society for Imaging Informatics in Medicine](https://siim.org/page/rfds_community_inter) (SIIM) Resident, Fellow & Doctoral Student (RFDS) Community and maybe even participate in future **RSNA Kaggle Challenges**. You can even download the data from past challenges to try your hand at training an algorithm. If you're not sure where to begin, the winners' solutions are posted in the **Discussion** tab of the Challenge page.
# 
# For an excellent one-week introduction to Imaging Informatics, consider taking the [**RSNA-SIIM** National Imaging Informatics Course](https://sites.google.com/view/imaging-informatics-course/home) offered in October and January of each year.
# 
# Thanks for participating! **Finally, if you enjoyed this demo, please up-vote the original kernel on [this page](https://www.kaggle.com/wfwiggins203/hello-world-for-deep-learning-siim).**
# 
# Cheers,
# 
# Walter Wiggins, MD, PhD<sup>1,2</sup><br/>
# M. Travis Caton, MD<sup>1,2</sup><br/>
# Kirti Magudia, MD, PhD<sup>1,2</sup><br/>
# Radiology Residents, BWH-CCDS Data Science Pathway<br/>
# 
# Faculty Advisors:<br/>
# Kathy Andriole, PhD<sup>1,2</sup> <br/>
# Michael Rosenthal, MD, PhD<sup>1,3</sup>
# 
# <sup>1</sup>Brigham & Women's Hospital/Harvard Medical School<br/>
# <sup>2</sup>MGH-BWH Center for Clinical Data Science<br/>
# <sup>3</sup>Dana Farber Cancer Institute
# 
# Contact: Walter Wiggins<br/>
# Email: [wwiggins@bwh.harvard.edu](mailto:wwiggins@bwh.harvard.edu)<br/>
# Twitter: [@walterfwiggins](https://www.twitter.com/walterfwiggins)

# In[ ]:




