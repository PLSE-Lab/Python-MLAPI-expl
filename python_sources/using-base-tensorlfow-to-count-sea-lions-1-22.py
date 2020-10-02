#!/usr/bin/env python
# coding: utf-8

# ## Welcome to the base Tensorflow kernel ##
# 
# 
# ----------
# Lets get right into it:
# 
#  1. First, we will be getting the coordinates with the method used in [this kernel][1]
#  2. We need to then crop the images so that they are classified and only 32 x 32 per lion
#  3. a set of 32 x 32 images **per sea-lion**, **Per: image_id**
#  4. One hot encode these
#  5. Use Tensorflow to build a model
#  6. Train the model
#  7. Plot the results
# 
#   [1]: https://www.kaggle.com/radustoicescu/noaa-fisheries-steller-sea-lion-population-count/use-keras-to-classify-sea-lions-0-91-accuracy

# In[ ]:


from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python import SKCompat
from sklearn.preprocessing import label_binarize
from tensorflow.contrib import learn
from subprocess import check_output
import tensorflow as tf
import skimage.feature
import numpy as np 
import pandas as pd 
import cv2
import os
# if anyone knows me, they know my imports
# have to be ascending order

print(check_output(["ls", "../input"]).decode("utf8"))
print('# File sizes')
for f in os.listdir('../input'):
    if not os.path.isdir('../input/' + f):
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
    else:
        sizes = [os.path.getsize('../input/'+f+'/'+x)/1000000 for x in os.listdir('../input/' + f)]
        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))


# As you can see we don't exactly have all the training data.

# In[ ]:


# CREDITS GO TO:  Radu Stoicescu
# classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
classes = ['0','1','2','3','4']
file_names = os.listdir("../input/Train/")
file_names = sorted(file_names, key=lambda 
                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item)) 
# select a subset of files to run on
file_names = file_names[0:3] #INCREASE FOR YOUR OWN MACHINE
print(file_names)
# dataframe to store results in
coordinates_df = pd.DataFrame(index=file_names, columns=classes)


# In[ ]:


# CREDITS GO TO:  Radu Stoicescu
for filename in file_names:
    # read the Train and Train Dotted images
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)
    cut = np.copy(image_2)
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)
    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255
    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 
    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    # detect blobs
    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    adult_males = []
    subadult_males = []
    pups = []
    juveniles = []
    adult_females = [] 
    image_circles = image_1.copy()
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        g,b,r = image_1[int(y)][int(x)][:]
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and g < 50 and b < 50: # RED
            adult_males.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (0,0,255), 10) 
        elif r > 200 and g > 200 and b < 50: # MAGENTA
            subadult_males.append((int(x),int(y))) 
            cv2.circle(image_circles, (int(x),int(y)), 20, (250,10,250), 10)
        elif r < 100 and g < 100 and 150 < b < 200: # GREEN
            pups.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (20,180,35), 10)
        elif r < 100 and  100 < g and b < 100: # BLUE
            juveniles.append((int(x),int(y))) 
            cv2.circle(image_circles, (int(x),int(y)), 20, (180,60,30), 10)
        elif r < 150 and g < 50 and b < 100:  # BROWN
            adult_females.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (0,42,84), 10)  
        cv2.rectangle(cut, (int(x)-112,int(y)-112),(int(x)+112,int(y)+112), 0,-1)
    coordinates_df["0"][filename] = adult_males
    coordinates_df["1"][filename] = subadult_males
    coordinates_df["2"][filename] = adult_females
    coordinates_df["3"][filename] = juveniles
    coordinates_df["4"][filename] = pups


# In[ ]:


# CREDITS TO HIM
x = []
y = []
for filename in file_names:    
    image = cv2.imread("../input/Train/" + filename)
    for lion_class in classes:
        for coordinates in coordinates_df[lion_class][filename]:
            thumb = image[coordinates[1]-16:coordinates[1]+16,coordinates[0]-16:coordinates[0]+16,:]
            if np.shape(thumb) == (32, 32, 3):
                x.append(thumb)
                y.append(lion_class)
# Add negs
for i in range(0,np.shape(cut)[0],224):
    for j in range(0,np.shape(cut)[1],224):                
        thumb = cut[i:i+32,j:j+32,:]
        if np.amin(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)) != 0:
            if np.shape(thumb) == (32,32,3):
                x.append(thumb)
                y.append("5") 
classes.append("5")
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)


# ## Here is where we switch to Tensorflow ##
# 
# We need to convert our labels to numerical values so TF can validate off of those.

# In[ ]:


#y = label_binarize(y, classes=classes) # Guy in comments gave me this one
tf.logging.set_verbosity(tf.logging.INFO)


# Now that we've binarized our labels, we need to create a tensorflow model.
# ------------------------------------------------------------------------
# 
# Our layers will consist of:
# 
#  1. Convolutional (32 5x5 filters) `conv2d()`
#  2. Pooling (Max pooling 2x2, stride of 2) `max_pooling2d()`
#  3. Convolution (64 5x5 layers) `conv2d()`
#  4. Pooling (Max pooling 5x5 filters) `max_pooling2d()`
#  5. Dense 1 (512 number of neurons, dropout at 0.5) `dense()`
#  6. Dense 2 (6 neurons for each class, classes{adult male to pups} plus negative class) `dense()`
# 
# *Each of these methods accepts a tensor as input and returns a transformed tensor as output. This makes it easy to connect one layer to another: just take the output from one layer-creation method and supply it as input to another.*
# 
# ### Tis easy as: ###
# 
# ![cnn][1]
# 
# 
#   [1]: https://s18.postimg.org/xlbv52ujd/Screen_Shot_2017-04-21_at_1.17.40_AM.png

# In[ ]:


def CNN_NOAA(features, labels, mode): # create a function to pass to main run
    # Input Layer
    input_layer = tf.reshape(features, [-1, 32, 32, 3]) #batch, pixles=32x32x3
    #Note that we've indicated -1 for batch size, 
    # which specifies that this dimension should be dynamically
    # computed based on the number of input values in features

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32, # feature map 32 x 32 x 32
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu) # still 32 x 32 x 32 

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)# 16x16x32

    # Convolutional Layer #2 
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64, # 16x16x64
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2) # 8x8x64

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=6)

    loss = None # starts at none, then gets computed each time
    train_op = None # starts at none, then gets computed each time

    # Calculate Loss (for TRAIN mode)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=6)
        loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.00214, # reduced to 0.0014, maybe should go lower...
        optimizer="SGD")

    # Generate Predictions
    predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
    }
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# ## The difficulty of finding a good learning rate ##
# 
# ![lrnimg][1]
# 
# 
#   [1]: http://cs231n.github.io/assets/nn3/learningrates.jpeg

# In[ ]:


# Load training from x and y
train_data = x # Returns image arrays
train_labels = y # the label array

#for evaluation we want to grab a photo from the train and test how well we did on it
eval_data = train_data #
eval_label = train_labels #


# ## Estimator ##
# 
# **a TensorFlow class for performing high-level model training, evaluation, and inference) for our model.**

# In[ ]:


# its going to write to a model directory (output)
noaa_classifier = SKCompat(learn.Estimator(model_fn=CNN_NOAA))


# ## Add what we need to log ##

# In[ ]:


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=150)


# ## The problem with accuracy development ##
# 
# ![accimg][1]
# 
# 
#   [1]: http://cs231n.github.io/assets/nn3/accuracies.jpeg

# ## Train the model ##
# We call `fit()` on noaa_classifier

# In[ ]:


noaa_classifier.fit(
    train_data,
    train_labels,
    batch_size=128,
    steps=300,  
    monitors=[logging_hook]) # Again, on kaggle machine


# ## I planned on doing 20,000 steps ##
# If I had the time and the computer power I would to 20,000 steps. Anything over 1,000 on Kaggle just hangs..

# ## Evaluating our Model ##
# **How well did it do?**
# 
#     metric_fn.
#  
# The function that calculates and returns the value of our metric
# 
#     prediction_key
# 
# The key of the tensor that contains the predictions returned by the model function.

# In[ ]:


metrics = {
    "accuracy": # What we're tracking
        learn.MetricSpec( # calculation function
            metric_fn=tf.metrics.accuracy, prediction_key="classes"), # returns class predctions
}


# In[ ]:


# Evaluate the model and print results
eval_results = noaa_classifier.predict(
    eval_data[1], batch_size=128)
print(eval_results)


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(eval_data[1])
plt.show() # honestly IDK, but that look like a lion over a log or something


# **It calculates the probability for a sea-lion type in a cropped 32x32 picture of one that is passed through noaa_classifier.predict**
# 
# **Under "class" it shows _**, from 0, 1 , 2 :  "_"  is **___**
# 
# ----------
# Things to improve:
# 
#  1. Using a larger subset of the train images
#  2. more steps in training 
#  3. choose a model out_dir
#  4. Decode the probabilities to give a better response.

# ## What mistake I saw people getting ##
# 
#     NanLossDuringTrainingError: NaN loss during training
# 
# Solution (implemented):
# 
#  - I made the learning rate a lot, lot smaller
#  - Change the predict batch size to the same as the train (familiarity) 
# 
# 
# ----------
# **Thanks to the user on stack overflow who made a small thing very clear**
# 
# *Gradient blow up*
# 
# *Reason: large gradients throw the learning process off-track.*
# 
# ***What you should expect**: Looking at the runtime log, you should look at the loss values per-iteration. You'll notice that the loss starts to grow significantly from iteration to iteration, eventually the loss will be too large to be represented by a floating point variable and it will become nan.*
# 
# ***What can you do**: Decrease the base_lr (in the solver.prototxt) by an order of magnitude (at least). If you have several loss layers, you should inspect the log to see which layer is responsible for the gradient blow up and decrease the loss_weight (in train_val.prototxt) for that specific layer, instead of the general base_lr*
# 
# [Link is here][1]
# 
# 
#   [1]: http://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training

# Update:
# =====
# After the error fix, the model did a lot better!
# 
#     'classes': array([2]), 'probabilities': array([[ 0.10903401,  0.0960522 ,  0.38835523,  0.11640827,  0.20649746, 0.08365285
# 
# 
# as you can see, the class was array of 2 with 0.3 probability (not so confident), which is adult_female.
