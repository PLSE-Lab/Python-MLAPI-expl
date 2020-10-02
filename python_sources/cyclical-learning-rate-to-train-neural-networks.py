#!/usr/bin/env python
# coding: utf-8

# # Using a Cyclical Learning Rate to Train Neural Networks
# 
# I first heard about the paper ([Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)) describing the use of a **cyclical learning rate** (a learning rate that changes after every epoch, cycling between two values) to train neural networks in the fast.ai class. I decided to try it out on the **CIFAR-10** dataset. Below are my findings.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import os
print(os.listdir('../input'))


# ## Neural Network Library and Architecture
# 
# I used Keras to make a Convolutional Neural Network to classify images in the CIFAR-10 datasets. Custom callbacks in Keras allowed for easy implementation of a callback function that updates the learning rate after every epoch. The Convolutional Neural Network I used looked like this:
# 
# ```
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same',
#                  input_shape=x_train.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))
# ```
# 
# The callback that updates the learning rate at the end of each epoch looks like this:
# ```
# import keras
# import keras.backend as K
# import numpy as np
# class CyclicalLR(keras.callbacks.Callback):
#     
#     def __init__(self, lr_low, lr_high, stepsize):
#         
#         self.lr_low = lr_low
#         self.lr_high = lr_high
#         self.stepsize = stepsize
#         
#     def on_epoch_begin(self, epoch, logs={}):
#         
#         cycle = np.floor( 1 + epoch/(2*self.stepsize) )
#         x = abs(epoch/self.stepsize - 2*cycle + 1)
#         new_lr = self.lr_low + (self.lr_high - self.lr_low) * max(0, 1-x)
#         
#         # Set the updated value of the learning rate
#         K.set_value(self.model.optimizer.lr, new_lr)
# ```
# As we'll see in a while, `lr_low` and `lr_high` are the lower and higher bounds of the learning rate respectively, and `stepsize` is the number of epochs in which the learning rate will go from one bound to another.

# ## Results with Fixed Learning Rate
# 
# I trained the network described above using the adam optimizer with a fixed learning rate for 50 epochs. As shown in the graph below, I got a maximum test set accuracy of **79.53%**.

# In[ ]:


df_fixed = pd.read_csv('../input/accuracy_fixed_adam.csv')
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(len(df_fixed['train_accuracy'])), df_fixed['train_accuracy'], label='Training Accuracy')
ax.plot(range(len(df_fixed['test_accuracy'])), df_fixed['test_accuracy'], label='Test Accuracy')
plt.legend()
plt.show()
print('Test Accuracy: ', '%.2f'%(df_fixed['test_accuracy'].values[-1]*100), '%')
print('Maximum Test Accuracy: ', '%.2f'%(max(df_fixed['test_accuracy'].values)*100), '%')


# ## Results with a Cyclical Learning Rate
# 
# When using a cyclical learning rate, you have to calculate two things:
# * The bounds between which the learning rate will vary.
# * The stepsize - in how many epochs the learning rate will reach from one bound to the other.
# 
# The paper describes a 'Learning Rate test' to determine the two bounds between which the learning rate varies. I experimented with stepsizes 2, 5 and 8 (i.e. the learning rate varies from the lower value to the upper value in 2, 5 and 8 epochs respectively). The learning rate range I got from the Learning Rate test was 0.0001 - 0.002.

# In[ ]:


df_clr_2 = pd.read_csv('../input/accuracy_clr_stepsize_2.csv')
df_clr_5 = pd.read_csv('../input/accuracy_clr_stepsize_5.csv')
df_clr_8 = pd.read_csv('../input/accuracy_clr_stepsize_8.csv')

fig, ax = plt.subplots(1, 3, figsize=(20,5), sharey=True)

ax[0].plot(range(len(df_clr_2['train_accuracy'])), df_clr_2['train_accuracy'], label='Training Accuracy')
ax[0].plot(range(len(df_clr_2['test_accuracy'])), df_clr_2['test_accuracy'], label='Test Accuracy')
ax[0].set_title('Stepsize = 2')
ax[0].legend()

ax[1].plot(range(len(df_clr_5['train_accuracy'])), df_clr_5['train_accuracy'], label='Training Accuracy')
ax[1].plot(range(len(df_clr_5['test_accuracy'])), df_clr_5['test_accuracy'], label='Test Accuracy')
ax[1].set_title('Stepsize = 5')
ax[1].legend()

ax[2].plot(range(len(df_clr_8['train_accuracy'])), df_clr_8['train_accuracy'], label='Training Accuracy')
ax[2].plot(range(len(df_clr_8['test_accuracy'])), df_clr_8['test_accuracy'], label='Test Accuracy')
ax[2].set_title('Stepsize = 8')
ax[2].legend()

plt.show()

print('Stepsize = 2')
print('Test Accuracy: ', '%.2f'%(df_clr_2['test_accuracy'].values[-1]*100), '%')
print('Maximum Test Accuracy: ', '%.2f'%(max(df_clr_2['test_accuracy'].values)*100), '%')

print('\nStepsize = 5')
print('Test Accuracy: ', '%.2f'%(df_clr_5['test_accuracy'].values[-1]*100), '%')
print('Maximum Test Accuracy: ', '%.2f'%(max(df_clr_5['test_accuracy'].values)*100), '%')

print('\nStepsize = 8')
print('Test Accuracy: ', '%.2f'%(df_clr_8['test_accuracy'].values[-1]*100), '%')
print('Maximum Test Accuracy: ', '%.2f'%(max(df_clr_8['test_accuracy'].values)*100), '%')


# The paper describes exactly the results we see above - although increasing the learning rate had a short-term negative effect, in the long run it had a beneficial effect. The difficulty in minimzing the loss, the authors say, arises from saddle points. They have small gradients that slow the learning process. Increasing the learning rate allows for more rapid traversal of saddle point plateaus.

# ## Fixed vs Cyclical Learning Rate

# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(20,5), sharey=True)

ax[0].plot(range(len(df_fixed['test_accuracy'])), df_fixed['test_accuracy'], label='Fixed')
ax[0].plot(range(len(df_clr_2['test_accuracy'])), df_clr_2['test_accuracy'], label='Cyclical')
ax[0].set_title('Stepsize = 2')
ax[0].legend()

ax[1].plot(range(len(df_fixed['test_accuracy'])), df_fixed['test_accuracy'], label='Fixed')
ax[1].plot(range(len(df_clr_5['test_accuracy'])), df_clr_5['test_accuracy'], label='Cyclical')
ax[1].set_title('Stepsize = 5')
ax[1].legend()

ax[2].plot(range(len(df_fixed['test_accuracy'])), df_fixed['test_accuracy'], label='Fixed')
ax[2].plot(range(len(df_clr_8['test_accuracy'])), df_clr_8['test_accuracy'], label='Cyclical')
ax[2].set_title('Stepsize = 8')
ax[2].legend()

plt.show()


# Observe that the accuracy peaks at epochs that are a multiple of 2 \* stepsize. This is when the learning rate is at the lower bound of the range. It would therefore make sense to choose the stepsize and number of epochs such that you stop training when the number of epochs is a multiple of 2 \* stepsize.
# 
# Note also that with a cyclical learning rate, we cross the maximum test accuracy achieved using a fixed learning rate, in fewer iterations.
# 
# ## Conclusion
# Using a cyclical learning rate did give a slightly better classification accuracy than using a fixed learning rate. And the best part is, a cyclical learning rate can be used with adaptive learning rate methods. I just used cyclical learning rates with the adam optimizer. Also, calculating the cyclical learning rate has virtually no computation cost (compared to adaptive learning rate calculations).
# 
# I'm looking forward to using this method in my neural network designs!
