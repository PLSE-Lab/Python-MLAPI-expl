#!/usr/bin/env python
# coding: utf-8

# This notebook draws upon my earlier 2 kernels on [Pre-processing the training](https://www.kaggle.com/inturiyam/exploring-titanic-data-using-pandas-dataframes/) and[ test data sets](https://www.kaggle.com/inturiyam/processing-test-data/). I have imported the outputs of those here. 
# 
# To start with we will try a Neural Network approach to the prediction using Keras. Subsequently, we will try other prediction models. 

# **THE LESSONS LEARNT SECTION**
# 
# I learnt a lot about Neural Networks by doing this simple workbook. Here I have more or less recorded every step of progress I made with the implementation so that someone starting afresh can get a hang of how to progress and what possible mistakes one could make (or at least, I made !) 
# 
# **First things first - understand what the Keras Dense() layer actually does**
# The most important piece I learnt was that while Keras abstracts and simplifies a lot of things for you, its important to look at the source code. So here's what Dense() actually does. 
# 
# *class Dense(Layer):
# ...
#     def __init__(self, units,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):*
# 
# As you can see, it doesn't use kernel regularization by default. Its, therefore, important to explicity specify it. On the other hand, it does a default Kernel initialization. So I ended up getting very noise loss curves and I was left wondering for couple of days why? 
# 
# I started with a simple Dropout regularization - tried several variations from Dropout of 0.2 all the way to 0.5. Let me reproduce below some of the weird loss/accuracy graphs I got initially. Finally, I just fixed it at 0.5. I presumed Dropout regularization should work well - but it was still giving me very noisy curves. 
# 
# In parallel, I also saw the loss curve falling very sharply and the curve flattening out. I thought possibly there is some problem with initialization. Then I found the source code of Dense() and saw that there is by default glorot_uniform initialization.  I also tried Glorot_Normal initialization. I got it right after several attempts. I didn't specify the "Seed" initially. So it didn't seem to work. As you could see,  Dropout regularization alone did not help in making the loss curves less noisy and prevent overfitting.  
# 
# Finally, I tried L2 regularization. Then it all started looking better ! That said, I still don't know what is Activity Regularization - so that's on my learning list. 
# 
# Now the big question - what happened to "Data Whitening" a.k.a Principal Component Analysis? Well, I added that. Then again another big question - to do BatchNorm Before Activation or After Activation. You could take a look at this. Here, I am applying BatchNorm before activation. So that modifies the code a little bit. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/exploring-titanic-data-using-pandas-dataframes/train_processed.csv")
df_test = pd.read_csv("../input/processing-test-data/test_processed.csv")


# In[ ]:


#df_train.info()
#df_test.info()


# In[ ]:


#The df.values helps convert from a Dataframe to an Array. Without the .values, we will get a dataframe as output. 
#Temporarily removing the feature "cab_t" to keep it symmetric with test data which didn't have any passengers in T cabin
x_train = df_train[["P2","P3","norm_len_name","title_ms","title_mrs","title_others","is_male","age_norm","norm_family_size","norm_fare",
                    "cab_b","cab_c","cab_d","cab_e","cab_f","cab_g","cab_z","embQ","embS"]].values
y_train = df_train[["Survived"]].values
print(x_train.shape,y_train.shape)
x_test = df_test[["P2","P3","norm_len_name","title_ms","title_mrs","title_others","is_male","age_norm","norm_family_size","norm_fare",
                    "cab_b","cab_c","cab_d","cab_e","cab_f","cab_g","cab_z","embQ","embS"]].values

print(x_test.shape)
#"high_prob_group",,"embC","title_mr",


# In[ ]:


#All the imports in one place ! (Will clean it up a bit later.)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.initializers import he_normal, glorot_normal 
from keras import metrics
import matplotlib.pyplot as plt
from scipy import stats #To quickly get stats of test output


# In[ ]:


# Get training and test loss histories
#This code block is used with thanks to https://chrisalbon.com/deep_learning/keras/visualize_loss_history/
def plot_loss(h):
    m = h.history.keys()
    #print(m)
    training_loss = h.history['loss']
    test_loss = h.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.figure(figsize=[10,10])
    plt.plot(epoch_count, training_loss,'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'CrossVal Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();


# In[ ]:


def plot_accuracy(h):
    training_acc = h.history['acc']
    test_acc = h.history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_acc) + 1)

    # Visualize loss history
    plt.figure(figsize=[10,10])
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, test_acc, 'b-')
    plt.legend(['Training Accuracy', 'CrossVal Accuracy',])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();


# In[ ]:


#This was how I started off. And as you can see I got pretty noisy loss graphs. I was really left wondering what was happening.
#Also you can see the Training Loss and CV Loss diverging as the number of epochs increased showing that it was overfitting.  
model = Sequential()
model.add(Dense(64, activation='relu',input_dim=19))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

h=model.fit(x_train, y_train, epochs=1500, validation_split=0.3,batch_size = 50, shuffle=True, verbose=0)
plot_loss(h)
plot_accuracy(h)


# In[ ]:


#Then I thought there was something wrong with initialization and tried many things. 
#Still couldn't figure out why my validation graphs were so noisy. More precisely, I didn't know how to add L2 regularization.  
#init = he_normal(seed=None)
init = glorot_normal(seed=1)
#kernel_initializer = init, bias_initializer = 'zeros'
#I understood mush later from Keras documentation that the default kernel initializer will be Glorot Uniform
model = Sequential()
model.add(Dense(64, activation='relu',kernel_initializer = init, input_dim=19))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.000001)
model.compile(optimizer=rmsprop,
              loss='binary_crossentropy',
              metrics=['accuracy'])

h=model.fit(x_train, y_train, epochs=1500, validation_split=0.3,batch_size = 50, shuffle=True, verbose=0)
plot_loss(h)
plot_accuracy(h)


# In[ ]:


#Then I figured how to add L2 regularization and after that it was all quite smooth. 
#Then I just cranked up the training epochs and let the optimizer parameters remain the default values.
init = glorot_normal(seed=1)
model = Sequential()
model.add(Dense(64, activation='relu',kernel_initializer = init,kernel_regularizer=regularizers.l2(0.01), input_dim=19))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adamop = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=False)
#rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=adamop,
              loss='binary_crossentropy',
              metrics=['accuracy'])

h=model.fit(x_train, y_train, epochs=1500, validation_split=0.3,batch_size = 50, shuffle=True, verbose=0)
plot_loss(h)
plot_accuracy(h)


# In[ ]:


#Now we come to pretty much the last step (as is the best I know of what to do before I go back to relook at my features again.) 
#We will introduce BatchNormalization so that first of all, the input data gets cleaned (a proxy for doing PCA) and secondly, all layer weights and biases get centered 
#At this stage I notice that I have plateaued out on training accuracy - so my NN is somewhat underfitting due to all the
#dropout and L2 norm which I have added. At this stage, I am now going to experiment with changing the learning rates and trying to
#push up my learning accuracy.
init = glorot_normal(seed=1)
model = Sequential()
model.add(Dense(64, kernel_initializer = init,kernel_regularizer=regularizers.l2(0.01), input_dim=19))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adamop = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adamop = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)
#rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=adamop,
              loss='binary_crossentropy',
              metrics=['accuracy'])

h=model.fit(x_train, y_train, epochs=10000, validation_split=0.3,batch_size = 32, shuffle=True, verbose=0)
plot_loss(h)
plot_accuracy(h)


# In[ ]:


score = model.predict(x_test)
#print(score)
#print(stats.describe(score))
y_test = score>=0.5
#print(y_test[0:10,0])
print(sum(y_test))
out = np.zeros([418,2])
out[:,0]=df_test["PassengerId"]
out[:,1]=y_test[:,0]
#print(out[:,0],out[:,1])


# In[ ]:


#And then we just export it out for submission to the contest ! 
df = pd.DataFrame(out)
print(df)
df.to_csv("results.csv",header=["PassengerId","Survived"],index='False')


# **Voila ! We got ....% accuracy. Hope you found this notebook useful. **
# 
