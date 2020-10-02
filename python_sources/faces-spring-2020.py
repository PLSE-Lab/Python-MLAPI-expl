#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# Y'know that moment when you realize that you have probably had the optimal answer for weeks, but your unfamiliarity with the programming tools being used has hampered your progress...?
# 
# That was today.  I had not explored Keras callbacks, and my ignorance of the ModelCheckpoint callback probably cost me dozens of hours of frustration.  Lo and behold!  Once the callback was added, my results improved by an order of magnitude.
# 
# Ah well... welcome to learning Python, Pandas, Numpy, Tensorflow and Keras.
# 
# And more frustration before finally figuring out the automatic submission process which attaches the relevant notebook to the submission.  DOH!
# 
# Welcome to learning Kaggle!

# # Step 1: Exploratory Data Analysis
# Doing a visual exploratory data analysis of the training images, we see that there are a number of images that are mislabeled.  Faces 30 and 31 are both identified in the training data as having glasses, yet visual inspection shows they do not.  
# 
# Color me masochistic, but I did a visual review of the training images and found ~540 or the 4,500 images to be improperly labeled for a base error rate of 12% over the training data.  
# 
# Since this project was supposed to properly detect glasses in StyleGAN2-generated images, I knew I needed to correct the labels to avoid a garbage-in/garbage-out effect.
# 
# Yeah... that'll be in a different notebook.  This notebook is about moving towards an optimal score on the leaderboard.

# # Step 2: Process the Input 
# Although I could have used a binary classification scheme, I chose to use a multi-class classification to allow for expanded categorization in the future.  This required a one-hot-encoding of the labels which I chose to implement via Keras to_categorical() method.

# In[ ]:


DATA_DIR = "../input/applications-of-deep-learningwustl-spring-2020/"

import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical

print("Loading original data...")
df = pd.read_csv(DATA_DIR+"train.csv")
X = np.array(df.drop(['id','glasses'], axis=1))
y = np.array(to_categorical(df['glasses']))
print("Done!")


# If I did it right, *X* should have 4500 rows of 512 elements each while *y* should have 4500 rows of 2 elements which represent category 0 (no glasses) and category 1 (glasses)

# In[ ]:


print(X.shape, y.shape)


# OK.  So the basic data is ready to go... erroneous labels and all.  Do I sound salty?  Naaaaahhhhhh!  Well... a bit perhaps.  I absolutely loathe generating incorrect outcomes just to try to climb the leaderboard.  

# # Step 3: Build your Model
# The little trick of letting the label set the length of the final output was not one I had found; but, it made sense.  Let the training label data set the output size automatically!

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization

print("Creating model...")

model = Sequential([
    Dense(512, input_shape=X.shape[1:]), BatchNormalization(), LeakyReLU(0.1), Dropout(0.1),
    Dense(512), BatchNormalization(), LeakyReLU(0.1), Dropout(0.3),
    Dense(512), BatchNormalization(), LeakyReLU(0.1), Dropout(0.3),
    Dense(512), BatchNormalization(), LeakyReLU(0.1), Dropout(0.3), 
    Dense(256), BatchNormalization(), LeakyReLU(0.1), 
    Dense(y.shape[:][1], activation='softmax')    
])

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
print("Model created.")


# # Step 4: Train your Model
# ![](http://)I dunno about you, but as I wrote that I flashed on "How to Train your Dragon."  Meh.
# Here's where we add the code for the ModelCheckpoint callback.

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint

mc = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)

model.fit(X, y, 
          epochs=1000, 
          validation_split=0.2, 
          callbacks=[mc])

print("Model trained.")


# # Step 5: Load the Best, and Test the Rest
# Cheesy... but accurate!  Load the best weights saved via the checkpoint callback, and run predictions against the test data.

# In[ ]:


print("Loading best model...")
model.load_weights("best_model.h5")

print("Loading test data...")
df = pd.read_csv(DATA_DIR+"test.csv")
X = np.array(df.drop(['id'], axis=1))
y = model.predict_proba(X)
df['glasses'] = y[:,1]
df['glasses'] = df['glasses'].round(1)
df[['id','glasses']].to_csv("submission.csv", index=False)

print(df[['id','glasses']].head(20))
print("Done!")


# # Step 6: Post-Mortem Analysis
# Clearly some of the results are erroneous, but that was expected due to the erroneous labeling.  The only thing to do at this point is to play around with the model structure to try to move up the leaderboard.
