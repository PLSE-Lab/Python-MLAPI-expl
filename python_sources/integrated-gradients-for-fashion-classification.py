#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook applies the Intergrated Gradient tools developed in [Sundararajan, M. et al, Axiomatic Attribution for Deep Networks." arXiv preprint arXiv:1703.01365 (2017).](https://arxiv.org/abs/1703.01365) and implemented in the [github repository](https://github.com/hiranumn/IntegratedGradients). The Fashion MNIST provides a slightly more interesting dataset than MNIST to see what it actually acts up

# # Forked Info
# The basic info from the original kernel which is our baseline model
# 
# In this work, we will train a CNN classifier using Keras with the guidelines described in [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).
# 
# Our strategy will be using 20% of the train data (12000 data rows) as a validation set to optimize the classifier, while keeping test data to finally evaluate the accuracy of the model on the data it has never seen.
# 
# #### Note
# Since I was not sure if the data was already shuffled, I didn't pass `validation_split=0.2` to _fit()_ and instead explicitly shuffled and split the validation data, as `validation_split` [would](https://keras.io/getting-started/faq/#how-is-the-validation-split-computed) use last 20% of the data in that case.

# In[ ]:


fash_labels = 'T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot'.split(',')


# In[ ]:


from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('../input/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashion-mnist_test.csv')

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

#Here we split validation data to optimiza classifier during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

#Test data
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))



X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 256
num_classes = 10
epochs = 50

#input image dimensions
img_rows, img_cols = 28, 28

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[ ]:


model.summary()


# ### Training
# Let's `fit()`! Note that `fit()` will return a _History_ object which we can use to plot training vs. validation accuracy and loss.

# In[ ]:


history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ### Results
# It turns out our classifier does better then the best baseline reported [here](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/), which is an SVM classifier with mean accuracy of 0.897.
# 

# 
# Let's plot training and validation accuracy as well as loss.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# ### Classification Report
# We can summarize the performance of our classifier as follows

# In[ ]:


#get the predictions for the test data
predicted_classes = model.predict_classes(X_test)

#get the indices to be plotted
y_true = data_test.iloc[:, 0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_true, predicted_classes, target_names=fash_labels))


# It's apparent that our classifier is underperforming for class 6 in terms of both precision and recall. For class 2, classifier is slightly lacking precision whereas it is slightly lacking recall (i.e. missed) for class 4.
# 
# Perhaps we would gain more insight after visualizing the correct and incorrect predictions.

# Here is a subset of correctly predicted classes.

# In[ ]:


plt.close('all')
plt.figure()
for i, i_cor in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i_cor].reshape(28,28), cmap='gray', interpolation='none')
    plt.axis('off')
    plt.title("Pred {}, Real {}".format(fash_labels[predicted_classes[i_cor]], 
                                        fash_labels[y_true[i_cor]]))
    plt.tight_layout()


# And here is a subset of incorrectly predicted classes.

# In[ ]:


for i, in_cor in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[in_cor].reshape(28,28), cmap='gray', interpolation='none')
    plt.axis('off')
    plt.title("Pred {}, Real {}".format(fash_labels[predicted_classes[in_cor]], 
                                        fash_labels[y_true[in_cor]]))
    plt.tight_layout()


# It looks like diversity of the similar patterns present on multiple classes effect the performance of the classifier although CNN is a robust architechture. A jacket, a shirt, and a long-sleeve blouse has similar patterns: long sleeves (or not!), buttons (or not!), and so on.

# # Integrated Gradients
# ```
# ################################################################
# # Implemented by Naozumi Hiranuma (hiranumn@uw.edu)            #
# # Kears-compatible implmentation of Integrated Gradients       # 
# # proposed in "Axiomatic attribution for deep neuron networks" #
# # (https://arxiv.org/abs/1703.01365).                          #
# # Keywords: Shapley values, interpretable machine learning     #
# ################################################################
# ```
# 

# In[ ]:


from time import sleep
import sys
import keras.backend as K

from keras.models import Model, Sequential
'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements this concept.
'''
class integrated_gradients:
    # model: Keras model that you wish to explain.
    # outchannels: In case the model are multi tasking, you can specify which channels you want.
    def __init__(self, model, outchannels=[], verbose=1):
    
        # Bacnend: either tensorflow or theano)
        self.backend = K.backend()
        
        #load model supports keras.Model and keras.Sequential
        if isinstance(model, Sequential):
            self.model = model.model
        elif isinstance(model, Model):
            self.model = model
        else:
            print("Invalid input model")
            return -1
        
        #load input tensors
        self.input_tensors = []
        for i in self.model.inputs:
            self.input_tensors.append(i)
        # The learning phase flag is a bool tensor (0 = test, 1 = train)
        # to be passed as input to any Keras function that uses 
        # a different behavior at train time and test time.
        self.input_tensors.append(K.learning_phase())
        
        #If outputchanel is specified, use it.
        #Otherwise evalueate all outputs.
        self.outchannels = outchannels
        if len(self.outchannels) == 0: 
            if verbose: print("Evaluated output channel (0-based index): All")
            if K.backend() == "tensorflow":
                self.outchannels = range(self.model.output.shape[1]._value)
            elif K.backend() == "theano":
                self.outchannels = range(model1.output._keras_shape[1])
        else:
            if verbose: 
                print("Evaluated output channels (0-based index):")
                print(','.join([str(i) for i in self.outchannels]))
                
        #Build gradient functions for desired output channels.
        self.get_gradients = {}
        if verbose: print("Building gradient functions")
        
            # Evaluate over all channels.
        for c in self.outchannels:
            # Get tensor that calcuates gradient
            if K.backend() == "tensorflow":
                gradients = self.model.optimizer.get_gradients(self.model.output[:, c], self.model.input)
            if K.backend() == "theano":
                gradients = self.model.optimizer.get_gradients(self.model.output[:, c].sum(), self.model.input)
                
            # Build computational graph that calculates the tensfor given inputs
            self.get_gradients[c] = K.function(inputs=self.input_tensors, outputs=gradients)
            
            # This takes a lot of time for a big model with many tasks.
            # So lets pring the progress.
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write("Progress: "+str(int((c+1)*1.0/len(self.outchannels)*1000)*1.0/10)+"%")
                sys.stdout.flush()
        # Done
        if verbose: print("\nDone.")
            
                
    '''
    Input: sample to explain, channel to explain
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample.
    Output: list of numpy arrays to integrated over.
    '''
    def explain(self, sample, outc=0, reference=False, num_steps=50, verbose=0):
        
        # Each element for each input stream.
        samples = []
        numsteps = []
        step_sizes = []
        
        # If multiple inputs are present, feed them as list of np arrays. 
        if isinstance(sample, list):
            #If reference is present, reference and sample size need to be equal.
            if reference != False: 
                assert len(sample) == len(reference)
            for i in range(len(sample)):
                if reference == False:
                    _output = integrated_gradients.linearly_interpolate(sample[i], False, num_steps)
                else:
                    _output = integrated_gradients.linearly_interpolate(sample[i], False, num_steps)
                samples.append(_output[0])
                numsteps.append(_output[1])
                step_sizes.append(_output[2])
        
        # Or you can feed just a single numpy arrray. 
        elif isinstance(sample, np.ndarray):
            _output = integrated_gradients.linearly_interpolate(sample, reference, num_steps)
            samples.append(_output[0])
            numsteps.append(_output[1])
            step_sizes.append(_output[2])
            
        # Desired channel must be in the list of outputchannels
        assert outc in self.outchannels
        if verbose: print("Explaning the "+str(self.outchannels[outc])+"th output.")
            
        # For tensorflow backend
        _input = []
        for s in samples:
            _input.append(s)
        _input.append(0)
        
        if K.backend() == "tensorflow": 
            gradients = self.get_gradients[outc](_input)
        elif K.backend() == "theano":
            gradients = self.get_gradients[outc](_input)
            if len(self.model.inputs) == 1:
                gradients = [gradients]
        
        explanation = []
        for i in range(len(gradients)):
            _temp = np.sum(gradients[i], axis=0)
            explanation.append(np.multiply(_temp, step_sizes[i]))
            

        if isinstance(sample, list):
            return explanation
        elif isinstance(sample, np.ndarray):
            return explanation[0]
        return -1

    
    '''
    Input: numpy array of a sample
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample.
    Output: list of numpy arrays to integrated over.
    '''
    @staticmethod
    def linearly_interpolate(sample, reference=False, num_steps=50):
        # Use default reference values if reference is not specified
        if reference is False: reference = np.zeros(sample.shape);

        # Reference and sample shape needs to match exactly
        assert sample.shape == reference.shape

        # Calcuated stepwise difference from reference to the actual sample.
        ret = np.zeros(tuple([num_steps] +[i for i in sample.shape]))
        for s in range(num_steps):
            ret[s] = reference+(sample-reference)*(s*1.0/num_steps)

        return ret, num_steps, (sample-reference)*(1.0/num_steps)


# In[ ]:


ig = integrated_gradients(model)


# # Show explanations
# Here we show the explanations for 6 correctly and 6 incorrectly identified images. We show the explanations for both the predictions and the actual class (they are the same for the correct images)

# In[ ]:


fig, m_axs = plt.subplots(12, 3, figsize = (8,24))
for (ax1, ax2, ax3), img_idx in zip(m_axs, 
                       correct[0:6].tolist()+incorrect[0:6].tolist()):
    ax1.imshow(X_test[img_idx].reshape(28,28), cmap='gray', interpolation='none')
    ax1.axis('off')
    ax1.set_title("Class {}".format(fash_labels[y_true[img_idx]]))
    exp = ig.explain(X_test[img_idx].reshape(28,28, 1), 
                 reference=np.zeros((28, 28, 1)), 
                 outc=predicted_classes[img_idx])
    th = max(np.abs(np.min(exp)), np.abs(np.max(exp)))
    ax2.imshow(np.sum(exp, axis=2), cmap="seismic", vmin=-1*th, vmax=th)
    ax2.axis('off')
    ax2.set_title('Prediction Explanation\n{}'.format(fash_labels[predicted_classes[img_idx]]))
    
    exp = ig.explain(X_test[img_idx].reshape(28,28, 1), 
                 reference=np.zeros((28, 28, 1)), 
                 outc=y_true[img_idx])
    th = max(np.abs(np.min(exp)), np.abs(np.max(exp)))
    ax3.imshow(np.sum(exp, axis=2), cmap="seismic", vmin=-1*th, vmax=th)
    ax3.axis('off')
    ax3.set_title('Real Explanation\n{}'.format(fash_labels[y_true[img_idx]]))
    
fig.tight_layout()
fig.savefig('explanations.png', dpi = 300)


# In[ ]:


fig, m_axs = plt.subplots(12, len(fash_labels)+1, 
                          figsize = (12,24))
for n_axs, img_idx in zip(m_axs, 
                       correct[0:6].tolist()+incorrect[0:6].tolist()):
    ax1, *c_axs = n_axs
    ax1.imshow(X_test[img_idx].reshape(28,28), cmap='gray', interpolation='none')
    ax1.axis('off')
    ax1.set_title("{}".format(fash_labels[y_true[img_idx]]))
    for c_ix, (c_label, c_ax) in enumerate(zip(fash_labels, c_axs)):
        exp = ig.explain(X_test[img_idx].reshape(28,28, 1), 
                     reference=np.zeros((28, 28, 1)), 
                     outc=c_ix)
        # don't recalculate th since we want it to be consistently scaled
        c_ax.imshow(np.sum(exp, axis=2), cmap="seismic", vmin=-1*th, vmax=th)
        c_ax.axis('off')
        c_ax.set_title('{} {}'.format(c_label,'*' if predicted_classes[img_idx]==c_ix else ''))
    
fig.tight_layout()
fig.savefig('cat_exp.png', dpi = 300)


# In[ ]:




