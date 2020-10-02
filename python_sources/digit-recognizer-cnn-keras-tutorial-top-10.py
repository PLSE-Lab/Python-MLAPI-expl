#!/usr/bin/env python
# coding: utf-8

# # Introduction #
# This is my first computer vision kernel writtern for the [Digit Recognizer Competetion](https://www.kaggle.com/c/digit-recognizer). I am trying to understand CNN Deep Learning models. 
# 
# The layers of convolutional neural network consists of two main components :
# 1. Convolutional layers : the convolutional layer is responsible for the convolutional operation in which feature maps identifies features in the images.
# and is usually followed by two types of layers which are :
# >*   **Dropout** : Dropout is a regulization technique where you turn off part of the network's layers randomally to increase regulization and hense decrease overfitting. We use when the training set accuracy is muuch higher than the test set accuracy.
# >*   **Max Pooling** : The maximum output in a rectangular neighbourhood. It is used to make the network more flexible to slight changes and decrease the network computationl expenses by extracting the group of pixels that are highly contributing to each feature in the feature maps in the layer.
# 2. Dense layers : The dense layer is a fully connected layer that comes after the convolutional layers and they give us the output vector of the Network.
# 
# As a convention in Convolutional Neural Network we decrease the dimensions of the layers as we go deeper and increase the number of feature maps to make it detect more features and decrease the number of computational cost.
# 
# ![alt text](https://raw.githubusercontent.com/MoghazyCoder/Machine-Learning-Tutorials/master/assets/Untitled.png)
#  
# Hope it might be useful for someone else here as well. If you Like the notebook and think that it helped you, <font color="red"><b> please upvote</b></font>.
# 
# 
# ---
# ## Table of Content
# 1. Data Preprocessing
#     * Data Preparation
#     * Data Augmentation
# 2. Modeling and Evaluation
#     * Designing Neural Network Architecture
#     * Hyperparameters Tuning and Model Evaluation
# 3. Final Prediction & Submission

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep')

# import Keras and its layers
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, Conv2D, Convolution2D, MaxPool2D, MaxPooling2D, Activation, BatchNormalization, Input
from keras.optimizers import Adam ,RMSprop
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import itertools
# import time for checkpoint saving
import time

import warnings
warnings.filterwarnings('ignore')

# Load Data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)

# Save the 'Label' column
Y_train = train['label']
X_train = train.drop(labels = ["label"],axis = 1)

#Y_train.value_counts()

# To get more control over visualization we'll define our figure instead of simply using plt.
fig = plt.figure(figsize=(8, 8))  # create figure
ax = fig.add_subplot(111)  # add subplot
sns.countplot(Y_train);
ax.set_title(f'Labelled Digits Count ({Y_train.size} total)', size=20);
ax.set_xlabel('Digits');
# writes the counts on each bar
for patch in ax.patches:
        ax.annotate('{:}'.format(patch.get_height()), (patch.get_x()+0.1, patch.get_height()-150), color='w')


# # Data Preprocessing
# This section will talk about imputation, normalization, reshaping and transform/encoding of the data. Statistical summaries and visulization plots will be used to help recognizing underlying pattens to exploit in the model.
# ### Best Practice
# * If the data is balanced which means all the classes have fair contribution in the dataset regarding its numbers then we can easily use accuracy. But if the data is skewed then we won't be able to use accurace as it's results will be misleading and we may use F-beta score instead.

# In[ ]:


def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)

check_missing_data(train)
check_missing_data(test)


# In[ ]:


# perform a grayscale normalization to reduce the effect of illumination's differences
X_train = X_train / 255.0
test = test / 255.0

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def standardize(x): 
    return (x-mean_px)/std_px

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i,:, :, 0], cmap=plt.get_cmap('gray'))
    plt.title(f'Label: {Y_train[i]}');

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)

# Split the train and the validation set for the fitting
train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train, test_size = 0.1)


# ## Data Augmenting
# In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. There are many ways to augment the images like centering the images, normalization, rotation, shifting, and flipping. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.
# 
# The model augments data randomly each time the training requires a new batch of images, so the model isn't training on the same examples exactly at each epoch which is the case if we have simply fitted the model using the original images.
# 
# The augmentation generator will only use the train portion we got from the split. The model must never learn from the test images or even their augmentations, to have a meaningful performance and error analysis.

# In[ ]:


# Data Augmenting
validation_split = 0.15
img_gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        validation_split=validation_split)

batch_size = 86
# used to feed the model augmented training data
train_gen = img_gen.flow(x=train_x, y=train_y, subset='training', batch_size=batch_size)
# The valid_gen used for validation will also use the training dataset from the split, as we said before that the model must never learn from the test data.
# used to feed the model augmented validation data
valid_gen = img_gen.flow(x=train_x, y=train_y, subset='validation', batch_size=batch_size)

# used only to visualize the augmentations
visualizaion_flow = img_gen.flow(train_x, train_y, batch_size=1, shuffle=False)
# Now, let's visualize how the generator augments the original images
fig, axs = plt.subplots(2, 4, figsize=(20,10))  # let's see 4 augmentation examples
fig.suptitle('Augmentation Results', size=32)

for axs_col in range(axs.shape[1]):
    idx = np.random.randint(0, train_x.shape[0])  # get a random index
    img = train_x[idx,:, :, 0]  # the original image
    aug_img, aug_label = visualizaion_flow[idx]  # the same image after augmentation
    
    axs[0, axs_col].imshow(img, cmap='gray');
    axs[0, axs_col].set_title(f'example #{axs_col} - Label: {np.argmax(train_y[idx])}', size=20)
    
    axs[1, axs_col].imshow(aug_img[0, :, :, 0], cmap='gray');
    axs[1, axs_col].set_title(f'Augmented example #{axs_col}', size=20)


# # Modeling and Evaluation
# ## Designing Neural Network Architecture
# A typical Convolution Block would consists of
# 1. 2D Convolution - extracts features
# 2. BatchNormalization - enhances training efficiency and reduces Vanishing/Exploding Gradients effect
# 3. ReLU Activation Function
# 4. 2D Max Pool - optional, reduces overfitting and reduces the shape of the tensor
# 
# The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.
# 
# The CNN can isolate features that are useful everywhere from these transformed images (feature maps).
# 
# The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.
# 
# Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.
# 
# Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.
# 
# 'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.
# 
# The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.
# 
# In the end used the features in two fully-connected (Dense) layers which is just artificial an neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.
# 
# **Cost function**
# 
# It is a measure of the overall loss in our network after assigning values to the parameters during the forward phase so it indicates how well the parameters were chosen during the forward probagation phase. We are using categorical_crossentropy as the cost function.
# 
# **Optimizer**
# 
# It is the gradiant descent algorithm that is used. We use it to minimize the cost function to approach the minimum point. We are using adam optimizer which is one of the best gradient descent algorithms. You can refere to this paper to know how it works https://arxiv.org/abs/1412.6980v8
# 
# You can use other metrics to measure the performance other than accuracy as precision or recall or F1 score. the choice depends on the problem itself. Where high recall means low number of false negatives , High precision means low number of false positives and F1 score is a trade off between them. You can refere to this article for more about precision and recall http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
#  
# ### Best Practice
# * Getting convergence should not be a problem, unless you use an extremely large learning rate. It's easy, however, to create a net that overfits, with perfect results on the training set and very poor results on the validation data. If this happens, you could try increasing the Dropout parameters, increase augmentation, or perhaps stop training earlier. If you instead wants to increase accuracy, try adding on two more layers, or increase the number of filters.
# * In order to make the optimizer converge faster and closest to the global minimum of the loss function, its better to have a decreasing learning rate during the training to reach efficiently the global minimum of the loss function.
# *  Advanced techniques include data augmentation, nonlinear convolution layers, learnable pooling layers, ReLU activation, ensembling, bagging, decaying learning rates, dropout, batch normalization, and adam optimization.

# In[ ]:


# fix random seed for reproducibility
seed = 43
np.random.seed(seed)

learning_rate=0.00065
# Define the optimizer
#optimizer = RMSprop(lr=learning_rate, decay=learning_rate/2)
optimizer = Adam(lr=learning_rate)

def conv_block(x, filters, kernel_size, strides, layer_no, use_pool=False, padding='same'):
    x = Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides, name=f'conv{layer_no}',
               padding=padding)(x)
    
    x = BatchNormalization(name=f'bn{layer_no}')(x)
    x = Activation('relu', name=f'activation{layer_no}')(x)
    if use_pool:
        x = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], name=f'pool{layer_no}', padding='same')(x)
    return x

def build_model(X):
    h, w, c = X.shape[1:]  # get shape of input (height, width, channels)
    X = Input(shape=(h, w, c))
    conv1 = conv_block(X, filters=8, kernel_size=[3, 3], strides=[1, 1], layer_no=1)
    conv2 = conv_block(conv1, filters=16, kernel_size=[2, 2], strides=[1, 1], layer_no=2)
    conv3 = conv_block(conv2, filters=32, kernel_size=[2, 2], strides=[1, 1], layer_no=3, use_pool=True)
    
    conv4 = conv_block(conv3, filters=64, kernel_size=[3, 3], strides=[2, 2], layer_no=4)
    conv5 = conv_block(conv4, filters=128, kernel_size=[2, 2], strides=[1, 1], layer_no=5)
    conv6 = conv_block(conv5, filters=256, kernel_size=[2, 2], strides=[1, 1], layer_no=6, use_pool=True)
    
    flat1 = Flatten(name='flatten1')(conv6)
    drop1 = Dropout(0.35, name='Dopout1')(flat1)
    
    dens1 = Dense(128, name='dense1')(drop1)
    bn7 = BatchNormalization(name='bn7')(dens1)
    drop2 = Dropout(0.35, name='Dopout2')(bn7)
    relu1 = Activation('relu', name='activation7')(drop2)
    
    dens1 = Dense(256, name='dense01')(relu1)
    bn7 = BatchNormalization(name='bn07')(dens1)
    drop2 = Dropout(0.5, name='Dopout02')(bn7)
    relu1 = Activation('relu', name='activation07')(drop2)
    
    dens2 = Dense(10, name='dense2')(relu1)
    bn8 = BatchNormalization(name='bn8')(dens2)
    output_layer = Activation('softmax', name='softmax')(bn8)
    
    model = Model(inputs=X, outputs=output_layer)
    # The model needs to be compiled before training can start.
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_seq_model(X):
    h, w, c = X.shape[1:]  # get shape of input (height, width, channels)
    model = Sequential([
        Convolution2D(32,(3,3), activation='relu', input_shape = (h, w, c)),
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.25),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
        ])
    # The model needs to be compiled before training can start
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
# create the model
model = build_model(train_x)

# Set a learning rate annealer
learning_rate_annealer = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# learning_rate_annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
epochs = 10 # Turn epochs to achieve higher accuracy
hist = model.fit_generator(generator=train_gen, steps_per_epoch=train_x.shape[0]*(1-validation_split)//batch_size, epochs=epochs, 
                    validation_data=valid_gen, validation_steps=train_x.shape[0]*validation_split//batch_size, callbacks=[learning_rate_annealer])

# save model
#keras.models.save_model(model, f'model_{time.time()}.h5')

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(hist.history['loss'], color='b', label="Training loss")
ax[0].plot(hist.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(hist.history['acc'], color='b', label="Training accuracy")
ax[1].plot(hist.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# ## Hyperparameters Tuning & Model Evaluation
# ### Hyperparameters
# * One of the hyperparameters to tune is the learning rate and the decay. 
# * As you can see there are quite a few other parameters that could also be tweaked (number of layers, number of filters, Dropout parameters, augmentation settings). This is often done with trial and error, and there is no easy shortcut.
# * Confusion matrix can be very helpfull to see your model drawbacks.
# 
# ### Model Evaluation
# There are so many choices for CNN architecture. How do we choose the best one? First we must define what **best** means. The best may be the simplest, or it may be the most efficient at producing accuracy while minimizing computational complexity. In this kernel, we will run experiments to find the most accurate and efficient CNN architecture for classifying MNIST handwritten digits.
# 
# #### 1. How many convolution-subsambling pairs?
# First question, how many pairs of convolution-subsampling should we use? For example, our network could have 1, 2, or 3:
#  * 784 - **[24C5-P2]** - 256 - 10
#  * 784 - **[24C5-P2] - [48C5-P2]** - 256 - 10
#  * 784 - **[24C5-P2] - [48C5-P2] - [64C5-P2]** - 256 - 10  
#    
# It's typical to increase the number of feature maps for each subsequent pair as shown here. Let's see whether one, two, or three pairs is best. We are not doing four pairs since the image will be reduced too small before then. The input image is 28x28. After one pair, it's 14x14. After two, it's 7x7. After three it's 4x4 (or 3x3 if we don't use padding='same'). It doesn't make sense to do a fourth convolution.
# #### 2. How many feature maps?
# In the previous experiement, we decided that two pairs is sufficient. How many feature maps should we include? For example, we could do
#  * 784 - [**8**C5-P2] - [**16**C5-P2] - 256 - 10
#  * 784 - [**16**C5-P2] - [**32**C5-P2] - 256 - 10
#  * 784 - [**24**C5-P2] - [**48**C5-P2] - 256 - 10
#  * 784 - [**32**C5-P2] - [**64**C5-P2] - 256 - 10
#  * 784 - [**48**C5-P2] - [**96**C5-P2] - 256 - 10  
#  * 784 - [**64**C5-P2] - [**128**C5-P2] - 256 - 10  
# 
# #### 3. How large a dense layer?
# How many dense units should we use? For example we could use
#  * 784 - [32C5-P2] - [64C5-P2] - **0** - 10
#  * 784 - [32C5-P2] - [64C5-P2] - **32** - 10
#  * 784 - [32C5-P2] - [64C5-P2] - **64** - 10
#  * 784 - [32C5-P2] - [64C5-P2] - **128** -10
#  * 784 - [32C5-P2] - [64C5-P2] - **256** - 10
#  * 784 - [32C5-P2] - [64C5-P2] - **512** -10
#  * 784 - [32C5-P2] - [64C5-P2] - **1024** - 10
#  * 784 - [32C5-P2] - [64C5-P2] - **2048** - 10
# 
# #### 4. How much dropout?
# Dropout will prevent our network from overfitting thus helping our network generalize better. How much dropout should we add after each layer?
#  * 0%, 10%, 20%, 30%, 40%, 50%, 60%, or 70%
# 
# #### 5. Advanced features
# Instead of using one convolution layer of size 5x5, you can mimic 5x5 by using two consecutive 3x3 layers and it will be more nonlinear. Instead of using a max pooling layer, you can subsample by using a convolution layer with strides=2 and it will be learnable. Lastly, does batch normalization help? And does data augmentation help? Let's test all four of these
#  * replace '32C5' with '32C3-32C3'  
#  * replace 'P2' with '32C5S2'
#  * add batch normalization
#  * add data augmentation
#  
# **Notations:**
#  * **24C5** means a convolution layer with 24 feature maps using a 5x5 filter and stride 1
#  * **24C5S2** means a convolution layer with 24 feature maps using a 5x5 filter and stride 2 
#  * **P2** means max pooling using 2x2 filter and stride 2
#  * **256** means fully connected dense layer with 256 units 

# In[ ]:


def calculate_performance(labels, pred, dataset):
    pred_cat = np.argmax(pred, axis=1)  # categorical predictions 0-9
    labels_cat = np.argmax(labels, axis=1)  # categorical labels 0-9
    
    # a boolean vector of element-wise comparison between prediction and label
    corrects = (pred_cat == labels_cat)
    
    # get the falses data
    falses = dataset[~corrects]  # the falses images
    falses_labels = labels_cat[~corrects]  # true labels of the falsely classified images - categorical
    falses_preds = pred[~corrects]  # the false predictions of the images - 10-dim prediction
     
    examples_num = labels.shape[0]  # total numbers of examples
    accuracy = np.count_nonzero(corrects) / examples_num

    return accuracy, [falses, falses_labels, falses_preds], [labels_cat, pred_cat]

test_y_pred = model.predict(test_x)
test_accuracy, (test_falses, test_falses_labels, test_falses_preds), (true_labels, pred_labels) = calculate_performance(test_y, test_y_pred, test_x)

# calculate the F1 Score of the model on the test dataset
# micro averaging calculates it over false and correct predictions regardless of the class
test_f1 = f1_score(y_pred=pred_labels, y_true=true_labels, average='micro')

print(f'Test Dataset Accuracy: {np.round(test_accuracy*100, 3)}%')
print(f'F1 Score = {test_f1}')

plt.figure(figsize=(10, 10))

# Calculate the confusion matrix and visualize it
test_matrix = confusion_matrix(y_pred=pred_labels, y_true=true_labels)
sns.heatmap(data=test_matrix, annot=True, cmap='Blues', fmt=f'.0f')

plt.title('Confusion Matrix - Test Dataset', size=24)
plt.xlabel('Predictions', size=20);
plt.ylabel('Labels', size=20);

final_loss, final_acc = model.evaluate(test_x, test_y, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))


# In[ ]:


# Let's investigate for errors.
test_falses_preds_classes = np.argmax(test_falses_preds, axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('False Classifications Plots', size=32)

sns.countplot(test_falses_labels, ax=ax1);
ax1.set_title(f'Falses ({test_falses_labels.size} total)',size=20);
ax1.set_xlabel('Digits', size=15);
ax1.set_ylabel('Count', size=15);

for patch in ax1.patches:
    bar_height = patch.get_height()
    ax1.annotate(f'{bar_height}', (patch.get_x()+0.25, bar_height-0.2), color='w', size=15);

falses_matrix = confusion_matrix(y_pred=test_falses_preds_classes, y_true=test_falses_labels)
sns.heatmap(data=falses_matrix, annot=True, cmap='Blues', fmt=f'.0f')

def display_errors(examples, preds_probs, preds_classes, labels):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    fig, axs = plt.subplots(nrows, ncols, figsize=(30, 12))
    fig.suptitle('Misclassified Images', size=32)
    for ax, pred, pred_prob, label, example in zip(axs.ravel(), preds_classes, preds_probs, labels, examples):
        ax.imshow(example[:, :, 0] ,cmap='gray');
        ax.set_title(f'Label: {label}\nPrediction: {pred} with {np.round(np.max(pred_prob)*100, 3)}% Confidence.',
                size = 15)
        ax.axis('off')

nrows = 2
ncols = 3
plt.xlabel('Predictions', size=20);
plt.ylabel('Labels', size=20);
# Random display of errors between predicted labels and true labels
random_errors = np.random.choice(range(test_falses_labels.size), size=np.min((nrows * ncols, test_falses_labels.size)), replace=False)
# Top 6 errors 
Y_pred_errors_prob = np.max(test_falses_preds,axis = 1)
true_prob_errors = np.diagonal(np.take(test_falses_preds, test_falses_labels, axis=1))
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_dela_errors[-6:]

examples = test_falses[most_important_errors]
preds_probs = test_falses_preds[most_important_errors]
preds_classes = np.argmax(preds_probs, axis=1)
labels = test_falses_labels[most_important_errors]
display_errors(examples, preds_probs, preds_classes, labels)


# In[ ]:


# Example Experiment 2: feature maps
styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']

nets = 3
model = [0] *nets
for j in range(3):
    model[j] = Sequential()
    model[j].add(Conv2D(j*16+16,kernel_size=5,activation='relu',input_shape=(28,28,1)))
    model[j].add(MaxPool2D())
    model[j].add(Conv2D(j*32+32,kernel_size=5,activation='relu'))
    model[j].add(MaxPool2D())
    model[j].add(Flatten())
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(10, activation='softmax'))
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# CREATE VALIDATION SET
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.333)
# TRAIN NETWORKS
history = [0] * nets
names = ["16 maps","32 maps","64 maps"]
epochs = 10
for j in range(nets):
    history[j] = model[j].fit(X_train2,Y_train2, batch_size=80, epochs = epochs, 
        validation_data = (X_val2,Y_val2), verbose=0)
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        names[j],epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))

    # PLOT ACCURACIES
plt.figure(figsize=(15,5))
for i in range(nets):
    plt.plot(history[i].history['val_acc'],linestyle=styles[i])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.98,1])
plt.show()


# # Final Prediction & Submission
# Use full train dataset here to train model and predict on test set. Then submitting predictions to Kaggle.
# 
# You can increase number of epochs on your GPU enabled machine to get better results.

# In[ ]:


model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# TRAIN OUR BEST NET MORE
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x+epochs))
hist = model.fit_generator(img_gen.flow(X_train,Y_train, batch_size=64), epochs = 100, 
    steps_per_epoch = X_train.shape[0]//64, callbacks=[annealer], verbose=0)
print("Train accuracy={0:.5f}, Train loss={1:.5f}".format(max(hist.history['acc']),max(hist.history['loss'])))


# In[ ]:


# predictions = model.predict_classes(test, verbose=0)
predictions = model.predict(test)
predictions = np.argmax(predictions,axis = 1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)
print(submissions.head(5))


# In[ ]:


# from keras import models

# # Visualize the model
# model.summary()
# # Visualize the metrix of first convolutional layer
# layer1 = model.layers[0]
# layer1.name
# conv2d_1w = layer1.get_weights()[0][:,:,0,:]
# for i in range(1,17):
#       plt.subplot(4,4,i)
#       plt.imshow(conv2d_1w[:,:,i-1],interpolation="nearest",cmap="gray")
# plt.show()

# # Let's see the activation of the first layer
# test_im = X_train[2]
# plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')

# layer_outputs = [layer.output for layer in model.layers[:13]]
# activation_model = models.Model(input=model.input, output=layer_outputs)
# activations = activation_model.predict(test_im.reshape(1,28,28,1))

# first_layer_activation = activations[0]
# plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

# # Let's plot the activations of the other layers as well.
# layer_names = []
# for layer in model.layers[:-1]:
#     layer_names.append(layer.name) 
# images_per_row = 16
# for layer_name, layer_activation in zip(layer_names, activations):
#     if layer_name.startswith('conv'):
#         n_features = layer_activation.shape[-1]
#         size = layer_activation.shape[1]
#         n_cols = n_features // images_per_row
#         display_grid = np.zeros((size * n_cols, images_per_row * size))
#         for col in range(n_cols):
#             for row in range(images_per_row):
#                 channel_image = layer_activation[0,:, :, col * images_per_row + row]
#                 channel_image -= channel_image.mean()
#                 channel_image /= channel_image.std()
#                 channel_image *= 64
#                 channel_image += 128
#                 channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#                 display_grid[col * size : (col + 1) * size,
#                              row * size : (row + 1) * size] = channel_image
#         scale = 1. / size
#         plt.figure(figsize=(scale * display_grid.shape[1],
#                             scale * display_grid.shape[0]))
#         plt.title(layer_name)
#         plt.grid(False)
#         plt.imshow(display_grid, aspect='auto', cmap='viridis')
              
# # visualize full conected layer
# fc_layer = model.layers[-5]
# activation_model = models.Model(input=model.input, output=fc_layer.output)
# activations = activation_model.predict(test_im.reshape(1,28,28,1))   
# activation = activations[0].reshape(32,32)
# plt.imshow(activation, aspect='auto', cmap='viridis')

# # organize the training images by label
# #Y_train_value_df = pd.DataFrame(train["label"],columns=['label'])
# #Y_train_value_df['pos']=Y_train_value_df.index
# #Y_train_label_pos = Y_train_value_df.groupby('label')['pos'].apply(list)
# #pos = Y_train_label_pos[1][0]

# # display 3 rows of digit image [0,9], with last full connected layer at bottom
# plt.figure(figsize=(16,8))
# x, y = 10, 3
# for i in range(y):  
#     for j in range(x):
#         # digit image
#         plt.subplot(y*2, x, i*2*x+j+1)
#         #pos = Y_train_label_pos[j][i] # j is label, i in the index of the list
#         pos = i * j
#         plt.imshow(X_train[pos].reshape((28,28)),interpolation='nearest')
#         plt.axis('off')
#         plt.subplot(y*2, x, (i*2+1)*x+j+1)
#         activations = activation_model.predict(X_train[pos].reshape(1,28,28,1))   
#         activation = activations[0].reshape(32,32)
#         plt.imshow(activation, aspect='auto', cmap='viridis')
#         plt.axis('off')
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()

