#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION
# 
# As you may know, Kaggle kernels have a **limit for how much time they can keep running**.   
# Sometimes, it's enough time to train a descent model, but sometimes it isn't. 
# 
# If you have a big model, using very big data, it's possible that Kaggle's kernels available time is not enough for you. (Even if you're using a GPU)
# 
# For that reason, this kernel brings you a simple way to continuously train a Keras model for the time you want.   
# This allows you to commit and run your kernel without worrying whether it will be interrupted for exceeding execution time or not. 
# 
# ## Creating a custom callback
# 
# Here, we are going to create a custom callback, called `TimerCallback`, in order to manage training time and eventually interrupt it.
# 
# A `Callback` in Keras is an object that you pass either to the `model.fit` or `model.fit_generator` methods, in order to execute additional commands between batches, epochs, etc.   
# There are several kinds of ready-to-use callbacks, such as to save the best model in each epoch, to interrupt training when some metric or loss reaches a condition, to show graphs, etc., and there is also the possibility of creating custom ones.
# 
# Ours will interrupt training shortly before our time limit.
# 
# 
# 

# In[ ]:


import time 

#let's also import the abstract base class for our callback
from keras.callbacks import Callback

#defining the callback
class TimerCallback(Callback):
    
    def __init__(self, maxExecutionTime, byBatch = False, on_interrupt=None):
        
# Arguments:
#     maxExecutionTime (number): Time in minutes. The model will keep training 
#                                until shortly before this limit
#                                (If you need safety, provide a time with a certain tolerance)

#     byBatch (boolean)     : If True, will try to interrupt training at the end of each batch
#                             If False, will try to interrupt the model at the end of each epoch    
#                            (use `byBatch = True` only if each epoch is going to take hours)          

#     on_interrupt (method)          : called when training is interrupted
#         signature: func(model,elapsedTime), where...
#               model: the model being trained
#               elapsedTime: the time passed since the beginning until interruption   

        
        self.maxExecutionTime = maxExecutionTime * 60
        self.on_interrupt = on_interrupt
        
        #the same handler is used for checking each batch or each epoch
        if byBatch == True:
            #on_batch_end is called by keras every time a batch finishes
            self.on_batch_end = self.on_end_handler
        else:
            #on_epoch_end is called by keras every time an epoch finishes
            self.on_epoch_end = self.on_end_handler
    
    
    #Keras will call this when training begins
    def on_train_begin(self, logs):
        self.startTime = time.time()
        self.longestTime = 0            #time taken by the longest epoch or batch
        self.lastTime = self.startTime  #time when the last trained epoch or batch was finished
    
    
    #this is our custom handler that will be used in place of the keras methods:
        #`on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`
    def on_end_handler(self, index, logs):
        
        currentTime      = time.time()                           
        self.elapsedTime = currentTime - self.startTime    #total time taken until now
        thisTime         = currentTime - self.lastTime     #time taken for the current epoch
                                                               #or batch to finish
        
        self.lastTime = currentTime
        
        #verifications will be made based on the longest epoch or batch
        if thisTime > self.longestTime:
            self.longestTime = thisTime
        
        
        #if the (assumed) time taken by the next epoch or batch is greater than the
            #remaining time, stop training
        remainingTime = self.maxExecutionTime - self.elapsedTime
        if remainingTime < self.longestTime:
            
            self.model.stop_training = True  #this tells Keras to not continue training
            print("\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: " + str(self.elapsedTime/60.) + " minutes )\n\n")
            
            #if we have passed the `on_interrupt` callback, call it here
            if self.on_interrupt is not None:
                self.on_interrupt(self.model, self.elapsedTime)


# ## Using callbacks
# 
# Using callbacks in Keras is very simple, you just pass a list of them to the fit method. Suppose we want to stop training before reaching 350 minutes (5:50 hours):
# 
#     timerCallback = TimerCallback(350)
#     model.fit(x_train, y_train, ..... , callbacks = [timerCallback, someOtherCallback])
#     
# You can explore the `on_interrupt` method to save the model or its weights at interruption:
# 
#     timerCallback = TimerCallback(350, 
#                     on_interrupt = lambda model, elapsed: model.save_weights('my_weights.h5'))
#     
# Or you can use a `ModelCheckpoint` or a `LambdaCallback` if you want more customization:
# 
#     from keras.callbacks import ModelCheckpoint
#     model.fit(x_train, y_train, ...., callbacks = [timerCallback, 
#                                                    ModelCheckpoint('my_weights.h5')])
# 
# See more on callbacks here:  https://keras.io/callbacks/
#     
# ## EXAMPLE
# 
# Now, let's take a toy model for digit classification and interrupt it's training in five minutes, for instance.
# 
# Since this Kernel is not about models, how to make good networks, etc., let's not take too much time explaining the details of data loading and model creation.   
# I'm sure there are lots of tutorial kernels on this :)
# 
# ### Loading and checking:

# In[ ]:


import pandas as pd
import numpy as np
from keras.utils import to_categorical

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#loading data - as for a demonstration of the callback,
#we won't worry about validation data in this kernel (but it would work the same way)
trainData = pd.read_csv('../input/train.csv')
y_train = to_categorical(np.array(trainData['label']))                   
x_train = np.array(trainData[list(trainData)[1:]]).reshape((-1,28,28,1)) 
    #labels as one-hot encoded vectors (ex: label 2 will become [0,0,1,0,0,0,0,0,0,0])
    #x shaped as images with one channel
    
    
#quick check
def quickCheck(x, y, predicted=None):
    #plotting images
    fig,ax = plt.subplots(nrows=1,ncols=10, figsize=(10,2))
    for i in range(10):
        ax[i].imshow(x[i].reshape((28,28)))
    plt.show()
    
    #printing labels
    y = np.argmax(y, axis=1) #converting from one-hot to numerical labels
    print("  " + "      ".join([str(i) for i in y[:10]]) + " <- labels")
    
    #printing predicted if passed
    if predicted is not None:
        predicted = np.argmax(predicted, axis=1)
        print("  " + "      ".join([str(i) for i in predicted[:10]]) + " <- predicted labels")

quickCheck(x_train,y_train)



    


# ### Creating a model

# In[ ]:


from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Model

#a simple convolutional model (not worried about it's capabilities)
def createModel():
    inputImage = Input((28,28,1))
    output = Conv2D(10, 3, activation='tanh')(inputImage)
    output = Conv2D(20, 3, activation='tanh')(output)
    output = MaxPooling2D((4,4))(output)
    output = Conv2D(10, 3, activation='tanh')(output)
    output = Flatten()(output)
    output = Dense(10, activation='sigmoid')(output)

    model = Model(inputImage, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

model = createModel()


# ### Training the model for 5 minutes at most 
# 
# #### For your Kaggle kernel, you could try 350 minutes, as it has a 360 minutes limit
# 
# .
# 
# Of course that usually you should enable the GPU for your kernel to run many times faster. This callback technique is absolutely not necessary for such a simple model with a simple dataset, but might come in handy when you're working on really big models and data :)
# 

# In[ ]:


model.fit(x_train, y_train, epochs = 1000000000, callbacks=[TimerCallback(5)])


# ### Saving the model's weights as outputs (you can then download these weights later or use them as inputs to other kernels)
# 
# You could also explore `model.save(file)` and `model = load_model(file)`, if you prefer. (Some models might have serialization problems, because of this I always prefer `save_weights()` )

# In[ ]:


#a function compatible with the on_interrupt handler
def saveWeights(model, elapsed):
    model.save_weights("model_weights.h5")

#fitting with the callback
callbacks = [TimerCallback(5, on_interrupt=saveWeights)]
model.fit(x_train,y_train, epochs = 100000000000, callbacks=callbacks)


#check that the weights were saved:
import os
os.listdir(".")


# ### Using the saved weights in a new model and checking predictions

# In[ ]:


#although it uses the same creator function, it's a different model from the previous one
del(model)
model2 = createModel()

#load weights - this only works if the model has the same layer types and the same parameters
model2.load_weights('model_weights.h5') #

#evaluate model2
print("\n\nEvaluating model 2:")
loss, acc = model2.evaluate(x_train, y_train)
print('model 2 loss: ' + str(loss))
print('model 2 acc:  ' + str(acc))

#predicting and checking
predicts = model2.predict(x_train[25:35])
quickCheck(x_train[25:35],y_train[25:35], predicts)

