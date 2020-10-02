#!/usr/bin/env python
# coding: utf-8

# # **Overview**
# 
# This notebook describes an experimental type of neural network, which was used in this competition and resulted in reasonabled scores. It should be noted that I didn't use any special "magic feature" or some deep data analysis. In fact, I totally disregard this data comes from a time series and just sort the inputs in descending order. By doing this we lose information but dramatically decreases the number of features. Even doing this it is still possible to make good predictions. All merits goes to this different neural network structure, which was able to make decent predictions all by itself.

# In[ ]:


import numpy as np # linear algebra

import matplotlib.pyplot as plt # display images

import pandas as pd # data processing

from keras.models import Model # neural network generation, training and fitting 
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
from keras.optimizers import SGD


# In[ ]:


def organize_data(X):    
    max_non_null_terms = 0
    num_rows = X.shape[0]

    # Sort each row in descending order.  
    for i in range(0,num_rows):
        temp = np.array(sorted(X[i,:],reverse=True))
        X[i,:] = temp
        # We count the number of non null values in each row and keep the largest.
        non_null_terms = np.sum(X[i,:] > 0)
        if non_null_terms > max_non_null_terms:
            max_non_null_terms = non_null_terms
    
    # After sorting we "cut" the last columns which contains only null terms.
    X = X[:,0:max_non_null_terms]
    
    return X


# In[ ]:


def normalize_data(X):   
    num_rows = X.shape[0]
    num_cols = X.shape[1]
    X_normalized = np.zeros((num_rows,num_cols), dtype = np.float32)
    for i in range(0,num_rows):
        for j in range(0,num_cols):
            if X[i,j] > 0:
                X_normalized[i,j] = np.log10(X[i,j])
    
    return X_normalized


# # Preparing the data

# In[ ]:


# load train dataset
train_data = pd.read_csv("../input/train.csv")        
train_val = train_data.values
Y = train_val[:,1].astype(np.float32)
X = train_val[:,2:].astype(np.float32)
# load test dataset
test_data = pd.read_csv("../input/test.csv")        
test_val = test_data.values
X_test = test_val[:,1:].astype(np.float32)


# The input data is very sparse, being zero almost everywhere. Also we have much more features than instances, so we will have to deal with the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). These two things indicates we will have to deal with little information in a large dataset. In order to have more useful information about the dataset we discard these features by considering the row elements in descending order, so the features are *larger element, second larger element,* and so on. 
# 
# By doing this, we reduce the number of column from 4993 to 1994, which is a big reduction of dimension.

# In[ ]:


# stack vertically the train and test datasets
X_all = np.vstack((X, X_test))
# organize data
X_all = organize_data(X_all)
num_features = X_all.shape[1]
# normalize data
X_all = normalize_data(X_all)
Y = Y.reshape(Y.shape[0],1)
Y_normalized = normalize_data(Y)


# # New features
# 
# We also introduce a few more features to have more information in the input dataset. These are showed below.

# In[ ]:


# add new features
num_new_features = 5
X_add = np.zeros((X_all.shape[0],num_features + num_new_features), dtype = np.float32)
for i in range(0,X_all.shape[0]):
    X_add[i,0] = np.mean(X_all[i,X_all[i,:]!=0])
    X_add[i,1] = np.min(X_all[i,X_all[i,:]!=0])
    X_add[i,2] = np.max(X_all[i,:])
    X_add[i,3] = np.sum(X_all[i,:] > 0)
    X_add[i,4] = np.std(X_all[i,:])
X_add[:,5:] = X_all

# update number of features
num_features = num_features + num_new_features


# In[ ]:


# split this new dataset in train and test datasets to be used in training and prediction stages
X_train_final = X_add[0:X.shape[0],:]
X_test_final = X_add[X.shape[0]:,:]


# # Neural network model description
# 
# We will use deep learning to solve this problem, which seems to be a very difficult nonlinear regression problem. Our model is a neural network with dense layers and concatenation. This model explores a new idea I had of reusing layers, interpreting them as someone who makes a job and right away receives feedback in order to redo the job better. This feedback is given by the **feeedback_hid** layer. This layer analyzes what the previous layers computed and produces a new input for these layers. This proccess of making computations over the input, analyzing the computations and producing new inputs is done several times. The idea is that the **feedback_hid** layer will simplify the input, keeping what is important for the *workers* (the previous hidden layers). After all this, the final input is passed to the **master** layer, which is a layer supposedly smarter. This layer is the last one before the output and the information goes through it only one time.

# # Model illustration
# 
# ![](https://i.imgur.com/EcgCS26.png)

# # Optimizing the parameters
# 
# We can make the loss function equals the metric RMSLE. In this case we don't need the metric parameter, i.e., the loss is the metric too.
# 
# Now it is the time to choose the number of neurons and cycles in our network (the number of cycles is the number of times the proccess described above is repeated, the computing and feedbacking proccess). We do this by making a search in grid on the number of neurons and cycles. The activation functions were choosed after a lot of trial and error, but you can try your own in order to improve the model. 
# 
# We split the train dataset in training set ($67\%$) and test set ($33\%$) in order to validate the model. The learning curves are plotted and from them we can choose the best model.  I will compute just a few models here due to the time limitation. You are welcomed to check [GitHub](https://github.com/felipebottega/Machine-Learning-Codes) version of this code to see more details.

# In[ ]:


# Check learning plots to see if the model is overfitting or not, if it is learning or not. 
# Now it is the time we choose the best parameters.
for cycles in [1,5,10]:
    for neurons in [10,50,100,150]:
        # input layer
        visible = Input(shape=(num_features,))
        # first feature extractor
        hid1 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        hid2 = Dense(neurons, kernel_initializer='normal', activation = "relu")
        hidpar2 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        hid3 = Dense(neurons, kernel_initializer='normal', activation = "relu")
        hidpar3 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        # second feature extractor
        hid1_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
        hid2_ = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        hidpar2_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
        hid3_ = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        hidpar3_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
        # interpretation layer
        feedback_hid = Dense(num_features, kernel_initializer='normal', activation = "relu")
    
        x = visible
        L = []
        LP = []
        for i in range(0,cycles):
            # first path (L = layer)
            L.append(hid1(x))
            L.append(hid2(L[0]))
            L.append(hidpar2(L[1]))
            L.append(hid3(L[2]))
            L.append(hidpar3(L[3]))
            L.append(concatenate([L[3],L[4]]))
            # second path (LP = layer in parallel)
            LP.append(hid1_(x))
            LP.append(hid2_(LP[0]))
            LP.append(hidpar2_(LP[1]))
            LP.append(hid3_(LP[2]))
            LP.append(hidpar3_(LP[3]))
            LP.append(concatenate([LP[3],LP[4]]))
            # merge both paths
            final_merge = concatenate([L[-1],LP[-1]])        
            x = feedback_hid(final_merge)
        
        # prediction output
        master = Dense(neurons, kernel_initializer='normal', activation='tanh')(x)
        output = Dense(1, kernel_initializer='normal', activation='softplus')(master)
        model = Model(inputs=visible, outputs=output)
        
        # compile the network
        sgd = SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer=sgd)

        # fit the model
        history = model.fit(X_train_final, Y_normalized,validation_split=0.33, epochs=150, batch_size=100, verbose=0)
        print('Cycles =',cycles)
        print('Neurons =', neurons)
        # show some information about the predictions
        print('\nLoss of predictions in train dataset:')
        predictions = model.predict(X_train_final)
        # Transform predictions to original format
        predictions = 10**predictions
        print( np.sqrt(1/Y.shape[0])*np.linalg.norm((np.log(predictions+1)-np.log(Y+1)),2) )

        # summarize history for loss
        plt.plot(history.history['loss'][2:])
        plt.plot(history.history['val_loss'][2:])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


# For **cycles** $\geq 10$ we can see the model starting to overfit. In the other cases the test loss curve (orange curve) has a lot of peaks, which means the model can make big errors sometimes. The only exception to this is when we have **cycles** $=5$ and **neurons** $=150$. This seems to be the best option.

# In[ ]:


# final model
cycles = 5
neurons = 150

# input layer
visible = Input(shape=(num_features,))
# first feature extractor
hid1 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
hid2 = Dense(neurons, kernel_initializer='normal', activation = "relu")
hidpar2 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
hid3 = Dense(neurons, kernel_initializer='normal', activation = "relu")
hidpar3 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
# second feature extractor
hid1_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
hid2_ = Dense(neurons, kernel_initializer='normal', activation = "tanh")
hidpar2_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
hid3_ = Dense(neurons, kernel_initializer='normal', activation = "tanh")
hidpar3_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
# interpretation layer
feedback_hid = Dense(num_features, kernel_initializer='normal', activation = "relu")
# master layer
mast = Dense(neurons, kernel_initializer='normal', activation='tanh')
# output layer
out = Dense(1, kernel_initializer='normal', activation='softplus') 
    
x = visible
L = []
LP = []
for i in range(0,cycles):
    # first path (L = layer)
    L.append(hid1(x))
    L.append(hid2(L[0]))
    L.append(hidpar2(L[1]))
    L.append(hid3(L[2]))
    L.append(hidpar3(L[3]))
    L.append(concatenate([L[3],L[4]]))
    # second path (LP = layer in parallel)
    LP.append(hid1_(x))
    LP.append(hid2_(LP[0]))
    LP.append(hidpar2_(LP[1]))
    LP.append(hid3_(LP[2]))
    LP.append(hidpar3_(LP[3]))
    LP.append(concatenate([LP[3],LP[4]]))
    # merge both paths
    final_merge = concatenate([L[-1],LP[-1]])        
    x = feedback_hid(final_merge)
        
# prediction output
master = mast(x)
output = out(master)
model = Model(inputs=visible, outputs=output)
        
# compile the network
sgd = SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)


# # Fitting and predicting
# 
# The array **predictions** (below) was obtained using normalized data. The original format should be $10$ to the power of **predictions**. 

# In[ ]:


# fit the model to make predictions over the test dataset
history = model.fit(X_train_final, Y_normalized, epochs=150, batch_size=100, verbose=2)

# since X has more columns than X_test, we fill X_test with more null columns
predictions = model.predict(X_test_final)

# We plot some histogram to visualize the distribution of the predictions and make some comparisons. We expect
#that the histogram of the predictions are similar to the histogram of the outputs in the train dataset.
predictions = 10**predictions

print('Train dataset outputs.')
plt.hist(np.log10(Y),bins=100)
plt.show()

print('Predictions of the train dataset outputs.')
plt.hist(np.log10(predictions),bins=100)
plt.show()


# In[ ]:


# Save these predictions.
submission = pd.read_csv('../input/sample_submission.csv')
submission["target"] = predictions
submission.to_csv('submission.csv', index=False)
print(submission.head())


# In[ ]:




