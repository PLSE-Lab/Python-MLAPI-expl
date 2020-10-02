#!/usr/bin/env python
# coding: utf-8

# # Keras callbacks and config files
# 
# ## Introduction to callbacks
# 
# Keras callbacks allow user-defined methods to be invoked at certain predetermined points in the model fitting process. This functional pattern is known as the **callback** pattern. Specifically, you can define and execute methods during any of the following points, ordered by granularity:
# 
# * `on_epoch_begin`
# * `on_epoch_end`
# * `on_batch_begin`
# * `on_batch_end`
# * `on_train_begin`
# * `on_train_end`
# 
# These callbacks can be passed to a model during `fit` time. They allow us to modify the classifier as it is run, or to log and retain historical information about the model, or to do both.
# 
# There are basically three sets of callbacks. Lambda callbacks is a quick-and-dirty callback structure useful for small things. Name callbacks allow you to implement meatier modifications on your model. Finally, a significant number of precomputed callbacks exist, which express a handful of particularly common and useful recipes, like checkpointing, Tensorboard integration, and other similar things.

# ## Demonstration
# 
# First, a simple reusable neural network model wrapper.

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
import pandas as pd

X_train = np.random.random((1000, 3))
y_train = pd.get_dummies(np.argmax(X_train[:, :3], axis=1)).values
X_test = np.random.random((100, 3))
y_test = pd.get_dummies(np.argmax(X_test[:, :3], axis=1)).values


# Reusable model fit wrapper.
def epocher(batch_size=500, epochs=10, callbacks=None):
    # Build the model.
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=3))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD())

    # Perform training.
    clf.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[callbacks])
    return clf


# The `LambdaCallback` allows invoking an anonymous function over the course of classifier training. By default the callback is provided with a counter value and a log value. The counter keeps track of which iteration of the training process we are at. If the callback is defined on batches this is the batch number, on epochs this is the epoch number, and on trainings this is the forward plus backwards pass number. The logs contains a running...log...of values that specified as being under observation. At a minimum it will contain the loss values at the callback runtime.
# 
# Here is a minimal example showing how we can use this callback to historify that information:

# In[ ]:


from keras.callbacks import LambdaCallback

history = []
weight_history = LambdaCallback(on_epoch_end=lambda batch, logs: history.append((batch, logs)))
clf = epocher(callbacks=weight_history)


# You can actually get the same information a bit more conveniently by running `clf.history.history`:

# In[ ]:


clf.history.history


# As you might notice from the akward convention, however, this seems like a slightly more private API. In practice I would rely on using the callbacks, which are an accepted approach, and treating history-dot-history as an impermanent implementation detail.
# 
# Next, here's an implementation of a classed `Callback`. The following code cell demonstrates a recipe for recording a tape of weights over time on the model.

# In[ ]:


from keras.callbacks import Callback

class WeightHistory(Callback):
    def __init__(self):
        self.tape = []
        
    def on_epoch_end(self, batch, logs={}):
        self.tape.append(self.model.get_weights()[0])

wh = WeightHistory()
clf = epocher(callbacks=wh)


# If we look at our objet now we can see the history that we retained!

# In[ ]:


wh.tape[0]


# For non-trivial applications, classed callbacks are the way to go. We can even use them to perform updates to the model parameters inline: a feature we will demonstrate in the next notebook.
# 
# The last set of callbacks are precomputed ones. These are preexisting recipes that are built into Keras, that fit a handful of useful cases that occur often enough that templates are useful. For example, there's a checkpointing callback, and a stop-on-Nan callback. To get familiar with these, [browse the Keras documentation](https://keras.io/callbacks/).

# In[ ]:


keras.callbacks.TerminateOnNaN


# ## Config files
# 
# Keras models can be serialized to JSON files using the `get_config()` function.

# In[ ]:


clf.get_config()


# To go the other way, and load a model from a config file:

# In[ ]:


my_config = clf.get_config()
Sequential.from_config(my_config)

