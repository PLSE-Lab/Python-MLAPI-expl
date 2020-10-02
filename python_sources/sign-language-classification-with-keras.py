#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[3]:


#plot / img libs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# # 1 - Introduction

# The input data is stored in npy files, let's read them.

# In[7]:


# load data set
X = np.load('../input/Sign-language-digits-dataset/X.npy')
Y = np.load('../input/Sign-language-digits-dataset/Y.npy')
X.shape, Y.shape


# We can see that in X we have 2062 observations, each of them consisting in a 64x64 px image. In Y, we have the same number of rows, with 10 columns, one for each possible label. Let's take a look at the first image:

# In[8]:


plt.imshow(X[0,:,:], cmap='gray')


# This represents a 9, let's look at how the labels array represent this nine:

# In[9]:


Y[0,:]


# We can see, that the only column set to one, is the first one. Let's take a look, at the first picture of each label, by the column index.

# In[10]:


def plot_digits_colidx(X, Y):
    plt.figure(figsize=(10,10))
    plt.plot([5, 2, 11])
    for i in col_idx:
        ax = plt.subplot(5, 2, i+1)
        ax.set_title("Column_idx: " + str(i))
        plt.axis('off')
        plt.imshow(X[np.argwhere(Y[:,i]==1)[0][0],:], cmap='gray')


# In[11]:


N_classes = Y.shape[1]
col_idx = [i for i in range(N_classes)]
plot_digits_colidx(X, Y)


# Looking at this plot, that shows the first image for each column activation, we can see that the column index does not correspond to the digit that they represent. This is the relationship of column index to digit:
# 
# | Column index | Digit | 
# | --- | --- |
# | 0 | 9 |
# | 1 | 0 |
# | 2 | 7 |
# | 3 | 6 |
# | 4 | 1 |
# | 5 | 8 |
# | 6 | 4 |
# | 7 | 3 |
# | 8 | 2 |
# | 9 | 5 |

# Let't capture this in a dictionary, and let's transform the Y matrix so that its column indeces correspond to the digit they represent

# In[12]:


#dictionary that handles the column index - digit relatinship
colidx_digit = {0: 9,
                1: 0,
                2: 7,
                3: 6,
                4: 1,
                5: 8,
                6: 4,
                7: 3,
                8: 2,
                9: 5}

#digit - column index relationship dictionary
digit_colidx = {v: k for k, v in colidx_digit.items()}


# In[13]:


#create empty matrix
Y_ordered = np.zeros(Y.shape)
#fill the matrix so that the columns index also corresponds to the digit
for i in range(N_classes):
    Y_ordered[:, i] = Y[:, digit_colidx[i]]


# Let's check that the reordering worked.

# In[14]:


plot_digits_colidx(X, Y_ordered)


# Yup! now the column indeces correspond the digit they represent.

# Let's now look at the number of samples that we have for each label:

# In[11]:


Y.sum(axis=0)


# We can see that all labels have almost the same number of observations, being the least the digit 0, with 204 observations, and the label with the most observations the digit 5, with 208 observations.

# Let's now draw some samples for each digit:

# In[12]:


#N images per row
N_im_lab = 5
plt.figure(figsize=(11,11))
plt.plot([N_classes, N_im_lab, (N_im_lab * N_classes) + 1])

#for every label
for lab in range(N_classes):
    #show N_im_lab first samples
    for i in range(N_im_lab):
        ax = plt.subplot(N_classes, N_im_lab, 1 + (i + (lab*N_im_lab)))
        plt.axis('off')
        plt.imshow(X[np.argwhere(Y_ordered[:,lab]==1)[i][0],:], cmap='gray')


# We can see that there are small variations in all digits. For example, for digit 0, sometimes is possible to see the background through the 0 symbol hole, but some other times it is not possible.

# Now that we are a bit more familiar with the data, let's start with the modeling.

# # 2 - Baseline model

# The modeling framework that I will use is going to be Keras, so the baseline model we will build is a Logistic regression:

# In[13]:


import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten


# In[14]:


def keras_lr(input_shape):
    #input layer
    X_input = Input(input_shape)
    #flatten
    X = Flatten()(X_input)
    #dense layer
    X = Dense(N_classes, activation='softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='keras_lr')
    return model


# In[15]:


lr_model = keras_lr((64, 64, 1))


# In[16]:


lr_model.summary()


# Even if a logistic regression is a 'simple' model, the input dimensionality is 4096 (64x64), that results in having 40970 parameters (10 times 4096 weights, and 10 bias parameters; one for each digit). In general, when you have more parameters than observations (2062), you are in quite some trouble. Anyways, lets fit this logistic regression and see how it performs:

# In[17]:


#set the optimization
lr_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[18]:


#reshape the data, to adapt the shape to the keras expectation
X = X.reshape(X.shape[0], 64, 64, 1)

#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y_ordered, random_state=4)


# In[19]:


#fit the model
lr_fit_hist = lr_model.fit(x = X_train , y = y_train, validation_data = (X_test, y_test), epochs = 500, batch_size = 128, verbose=0)


# In[20]:


#show the train-test accuracy depending on the epoch
def plot_acc_vs_epoch(fit_history):
    plt.plot(fit_history.history['acc'])
    plt.plot(fit_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[21]:


plot_acc_vs_epoch(lr_fit_hist)


# In[22]:


#evaluate the model performance in the validation set
evs = lr_model.evaluate(x = X_test, y = y_test)
#show the accuracy metric
print(evs[1])


# We can see that this baseline, performs not so bad, having an accuracy of 74% . We can see that how the model performance evolves with the number of epochs. In the first 100 epochs, the model gets to a plateau in test performance, while the performance on the train set keeps improving, which is clear sign of over-fit. Let's set our final baseline, with this same configuration, but with just 200 epochs, as there seems to be a small improvement from the 100 epochs scenario:

# In[23]:


#fit the baseline model
base_model = keras_lr((64, 64, 1))
base_model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
basline_fit_hist = base_model.fit(x = X_train , y = y_train, validation_data = (X_test, y_test), epochs = 200, batch_size = 128, verbose=0)
#baseline model accuracy
base_model.evaluate(x = X_test, y = y_test)[1]


# Let's now take a look at the confusion matrix, to see wether if the errors are evenly distributed between classes:

# In[24]:


#compute confusion matrix
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

def plot_conf_matrix(y_true, y_pred, set_str):
    """
    This function plots a basic confusion matrix
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    df_conf = pd.DataFrame(conf_mat, index = ['Digit - ' + str(i) for i in range(N_classes)],
                           columns = ['Digit - ' + str(i) for i in range(N_classes)])

    plt.figure(figsize = (12, 12))
    sn.heatmap(df_conf, annot=True, cmap="YlGnBu")


# In[25]:


#class estimation
base_y_test_pred = base_model.predict(X_test)
plot_conf_matrix(y_test.argmax(axis=1), base_y_test_pred.argmax(axis=1), '')


# Ups, the first thing that pops up from this confusion matrix, is that there is not an uneven number of observations for each label. If I have time, I'll try to fix it using a stratified split schema.
# 
# Now, looking at the errors, we can see that most of the errors are focused in the digits 2, 4, 6 and 7.

# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_test.argmax(axis=1), base_y_test_pred.argmax(axis=1)))


# # 3 - CNN model

# In this section, we will train a CNN, to see if we can improve the performance achieved by the baseline model.

# In[27]:


from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, ZeroPadding2D, Dropout


# In[28]:


#CONV-> BatchNorm-> RELU block
def conv_bn_relu_block(X, n_channels, kernel_size=(3, 3)):
    X = Conv2D(n_channels, kernel_size)(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    return X


# We will first try a simple CNN architecture, with two convolutional layers, and then a dense layer. Each layer has relu activation. The first layer has a bigger kernel size (5x5), while the second one has a smaller kernel size (3x3). Maxpooling is used after both layers output, in order to reduce the dimensionality, and keep the most prominent activations.

# In[29]:


def keras_cnn_v1(input_shape):
    #input layer
    X_input = Input(input_shape)
    #32 filters, with 5x5 kernel size
    X = conv_bn_relu_block(X_input, 10, (5, 5))
    #Maxpooling and dropout
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #run another CONV -> BN -> RELU block
    X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 20)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #flatten
    X = Flatten()(X)
    #dense layer
    X = Dense(N_classes, activation='softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='keras_lr')
    return model


# In[30]:


cnn_v1 = keras_cnn_v1((64, 64, 1))
cnn_v1.summary()


# In[31]:


cnn_v1.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
cnn_v1_hist = cnn_v1.fit(x = X_train , y = y_train, validation_data = (X_test, y_test), epochs = 300, batch_size = 128, verbose=0)


# In[32]:


plot_acc_vs_epoch(cnn_v1_hist)
print (cnn_v1.evaluate(x = X_test, y = y_test)[1])


# This first proposed model has a accuracy of around 89%, around 14 points better than the baseline. Although we can see some overfit, as the train accuracy is around 99%. Let's now add one more layer to this CNN architecture, and see if there is any improvement:

# In[33]:


def keras_cnn_v2(input_shape):
    #input layer
    X_input = Input(input_shape)
    #32 filters, with 5x5 kernel size
    X = conv_bn_relu_block(X_input, 10, (5, 5))
    #Maxpooling and dropout
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #run another CONV -> BN -> RELU block
    X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 15)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #run another CONV -> BN -> RELU block
    X = ZeroPadding2D((1, 1))(X)
    X = conv_bn_relu_block(X, 20)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)
    #flatten
    X = Flatten()(X)
    #dense layer
    X = Dense(N_classes, activation='softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='keras_lr')
    return model


# In[34]:


cnn_v2 = keras_cnn_v2((64, 64, 1))
cnn_v2.summary()


# In[35]:


cnn_v2.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
cnn_v2_hist = cnn_v2.fit(x = X_train , y = y_train, validation_data = (X_test, y_test), epochs = 300, batch_size = 128, verbose=0)


# In[36]:


plot_acc_vs_epoch(cnn_v2_hist)
print (cnn_v2.evaluate(x = X_test, y = y_test)[1])


# The V2 of the CNN architecture achieves an accuracy of around 95%, 20 points better than the baseline model. Another good point of this model is that there is no clear sign of overfit, as both train and test performances are quite similar. Let's look at the confusion matrix of this model:

# In[37]:


cnn2_y_test_pred = cnn_v2.predict(X_test)
plot_conf_matrix(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1), '')


# In[38]:


print(classification_report(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1)))


# We see really good performance in all digits. Let's inspect some of the errors that the model is having:

# In[40]:


def visual_err_inspection(y_true, y_pred, lab_eval, N_samples=6):
    """
    This function runs a visual error inspection. It plots two rows of images,
    the first row shows true positive predictions, while the second one shows
    flase positive predictions
    """
    df_y = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    idx_y_eval_tp = df_y.loc[(df_y.y_true == lab_eval) & (df_y.y_pred == lab_eval)].index.values[:N_samples]
    idx_y_eval_fp = df_y.loc[(df_y.y_true != lab_eval) & (df_y.y_pred == lab_eval)].index.values[:N_samples]

    #capture number of false positives
    N_fp = idx_y_eval_fp.shape[0]

    N_plts = min(N_samples, N_fp)

    fig, axs = plt.subplots(2, N_plts, figsize=(15,6))
    for i in range(N_plts):
        #set plot for true positive sample
        axs[0, i].set_title("OK: " + "Digit - " + str(lab_eval))
        axs[0, i].axis('off')
        axs[0, i].imshow(X_test[idx_y_eval_tp[i], :, :, 0], cmap='gray')
        
        #set plot for false positive sample
        lab_ = df_y.iloc[idx_y_eval_fp[i]].y_true
        axs[1, i].set_title("KO: " + "Digit - " + str(lab_))
        axs[1, i].axis('off')
        axs[1, i].imshow(X_test[idx_y_eval_fp[i], :, :, 0], cmap='gray')
       

    plt.show()


# The following plot shows in the first row some samples with true positives for the digit one, while the second row shows some false positives.

# In[41]:


visual_err_inspection(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1), 1)


# We can see that the errors that the false positives are coming from samples quite similar to the digit one. Let's repeat this exercise with some more digits:

# In[42]:


visual_err_inspection(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1), 4)


# In[43]:


visual_err_inspection(y_test.argmax(axis=1), cnn2_y_test_pred.argmax(axis=1), 7)


# We can see, that in most of the false positives, there is in general some image similarity.
