#!/usr/bin/env python
# coding: utf-8

# #### Objective of this Notebook: 
# 
# Replicate the DenseNet Implentation and reproduce the (near) results obtained by the authors on CIFAR 10 image dataset using keras.
# 
# - DenseNet Paper: https://arxiv.org/pdf/1608.06993.pdf
# - CIFAR10 Data : https://www.cs.toronto.edu/~kriz/cifar.html
# 
# #### Bullets(from the paper)
# 
# - Superficially, DenseNets are quite similar to ResNets: The previous outputs from the convulation layers are concatenated instead of summed. However, the implications of this seemingly small modification lead to substantially different behaviors of the two network architectures.
# 
# - Model compactness. As a direct consequence of the input concatenation, the feature-maps learned by any of the DenseNet layers can be accessed by all subsequent layers. This encourages feature reuse throughout the network, and leads to more compact models
# 
# 
# ![image.png](attachment:image.png)

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten
from tensorflow.keras.optimizers import Adam
#
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load CIFAR10 Data
(X_train_val, y_train_val), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
print('train images:', X_train_val.shape) # no.of samples * height * width * channels
print('test images:', X_test.shape) # no.of samples * height * width * channels

# convert labels to onehot-encoding 
y_train_val = tf.keras.utils.to_categorical(y_train_val, num_classes = 10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)
print('train labels:', y_train_val.shape) # no.of samples * num_classes
print('test labels:', y_test.shape) # no.of samples * num_classes


# In[ ]:


import matplotlib.pyplot as plt
for i in range(16):
    # define subplot
    plt.subplot(4, 4, i+1)
    # plot raw pixel data
    plt.imshow(X_train_val[i])
# show the figure
plt.show()


# 10 classes in CIFAR10 data: 
# - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.1)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# ### Blocks of DenseNet

# #### convolution (with bottleneck layer) = BatchNorm layer + activtion + bottleneck conv2D + conv2D with (3,3) filter (will used to produce dense_block).
# ![image.png](attachment:image.png)

# In[ ]:


def bn_relu_convolution(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a convolution layers consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    
    """
    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(nb_channels * bottleneckWidth, (1, 1),
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

    # BN-ReLU-Conv
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(nb_channels, (3, 3), padding='same')(x)

    # Dropout
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    return x


# #### transition = BatchNorm layer + activtion + convolution for compression + pooling layer (will used after each dense block)
# ![image.png](attachment:image.png)

# In[ ]:


def bn_relu_transition(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu',)(x)
    x = layers.Convolution2D(int(nb_channels * compression), (1, 1), padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

    # Adding dropout
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


# #### dense_block = concatenated bn_relu_convolution layers.

# In[ ]:


def dense_block(x, num_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False,
                    weight_decay=1e-4):
    """
    Creates a dense block and concatenates inputs
    """

    for i in range(num_layers):
        cb = bn_relu_convolution(x, growth_rate, dropout_rate, 
                                 bottleneck) # 1 conv if bottleneck = 0 else 2 conv if bottleneck = 1
        nb_channels += growth_rate
        x = layers.concatenate([cb, x])
    return x, nb_channels


# ### DenseNet Model
# 
# ![image.png](attachment:image.png)
# 
# - (Input + Convolution) layers : Each input image is convoluted to depth of (2*growth_rate) before giveing it to dense_block.
# - transition = (convolution + pooling) after each dense block except for last one where we use (pooling + dense) layers for prediction.

# In[ ]:


def DenseNet(input_shape, dense_blocks, dense_layers, growth_rate, compression, bottleneck, 
                    weight_decay, dropout_rate, num_classes, ):
       """
       Build the model
       Returns: tf Keras Model instance
       """

       print('Creating DenseNet with Bottleneck = {}'.format(bottleneck))
       print('#############################################')
       print('No.of. dense blocks: %s' % dense_blocks)
       print('Layers per dense block: %s' % dense_layers)
       print('#############################################')

       # Input Layer
       img_input = layers.Input(shape=input_shape, name = 'img_input')
       nb_channels = growth_rate

       # Input-convolution layer
       x = layers.Conv2D(2 * growth_rate, (3, 3), padding='same', strides=(1, 1),name='input_conv', 
                         kernel_regularizer= tf.keras.regularizers.l2(weight_decay))(img_input)

       # Building dense blocks
       for block in range(dense_blocks - 1):
           # Add dense_block
           x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate,
                                    dropout_rate, bottleneck, weight_decay) 

           # Add transition
           x = bn_relu_transition(x, nb_channels, dropout_rate, compression, weight_decay) # 1 conv layer
           nb_channels = int(nb_channels * compression)

       # Add last dense block without transition but with only global average pooling
       x, nb_channels = dense_block(x, dense_layers[-1], nb_channels,
                                         growth_rate, dropout_rate, weight_decay)
       
       # prediction of class happens here
       x = layers.BatchNormalization(name = 'prediction_bn')(x)
       x = layers.Activation('relu',  name = 'prediction_relu', )(x)
       x = layers.GlobalAveragePooling2D( name = 'prediction_pool', )(x)
       prediction = layers.Dense(num_classes, name = 'prediction_dense', activation='softmax')(x)

       return tf.keras.Model(inputs=img_input, outputs=prediction, name='densenet')


# ### generalised logic to calculate no.of layers (equal) in network:
#   - This doesn't apply to DenseNet121 and Other DenseNet Architectures used for Imagenet DataSet - they have different no.of layers in dense block.
#   - `dense_layers in each dense_block  = (Total_Depth - (num_dense_blocks + 1))/num_dense_blocks`
#   - if bottlenecks layers are present then each conv_block in dense_block will have two conv2D layers then
#     `dense_layers in each dense_block (for bottleneck) = (Total_Depth - (num_dense_blocks + 1))/num_dense_blocks * 2`
#   
# ##### for 100 layer network with bottleneck & 3 dense blocks:
# 
# - num_layers in each dense_block:
#     `100 - (3+1)/3*2 = 16`
# 
# ##### for 101 layer network with bottleneck & 4 dense blocks:
# 
# - num_layers in each dense_block:
#      `101 - (4+1)/8 = 12`

# In[ ]:


dense_net = DenseNet(input_shape = (32,32,3), dense_blocks = 3, dense_layers = [16]*3,
                     growth_rate = 12, compression = 0.5, num_classes = 10, bottleneck = True, 
                     dropout_rate = None, weight_decay = 1e-5)
# dense_net.summary()


# In[ ]:


class DenseNet(object):
    
    def __init__(self,input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, num_classes=None,
                 dropout_rate=None, bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):
        
        # Parameters Check
        if num_classes == None:
            raise Exception(
                'Please define number of classes (e.g. num_classes=10). This is required to create .')

        if compression <= 0.0 or compression > 1.0:
            raise Exception('Compression have to be a value between 0.0 and 1.0.')

        if type(dense_layers) is list:
            if len(dense_layers) != dense_blocks:
                raise AssertionError('Number of dense blocks have to be same length to specified layers')
        elif dense_layers == -1:
            dense_layers = int((depth - 4) / 3)
            if bottleneck:
                dense_layers = int(dense_layers / 2)
            dense_layers = [dense_layers for _ in range(dense_blocks)]
        else:
            dense_layers = [dense_layers for _ in range(dense_blocks)]

        self.dense_blocks = dense_blocks
        self.dense_layers = dense_layers
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.bottleneck = bottleneck
        self.compression = compression
        self.num_classes = num_classes
        
        
    def build_model(self):
        """
        Build the model
        Returns: tf Keras Model instance
        """
        if self.bottleneck:
            print('Creating DenseNet with Bottlenecks')
        else:
            print('Creating DenseNet without Bottlenecks')
        print('-' * 50)
        print('No.of. dense blocks: %s' % self.dense_blocks)
        print('Layers per dense block: %s' % self.dense_layers)
        print('-'* 50)

        # Input Layer
        img_input = layers.Input(shape = self.input_shape, name = 'img_input')
        nb_channels = self.growth_rate

        # Input-convolution layer
        x = layers.Conv2D(2 * self.growth_rate, (3, 3), padding='same', strides=(1, 1),name='input_conv', 
                          kernel_regularizer= tf.keras.regularizers.l2(self.weight_decay))(img_input)

        # Building dense blocks
        for block in range(self.dense_blocks - 1):
            # Add dense_block
            x, nb_channels = self.dense_block(x, self.dense_layers[block], nb_channels, self.growth_rate,
                                      self.dropout_rate, self.bottleneck, self.weight_decay) 

            # Add transition
            x = self.bn_relu_transition(x, nb_channels, self.dropout_rate, 
                                        self.compression, self.weight_decay) # 1 conv layer
            nb_channels = int(nb_channels * self.compression)

        # Add last dense block without transition but with only global average pooling
        x, nb_channels = self.dense_block(x, self.dense_layers[-1], nb_channels,
                                          self.growth_rate, self.dropout_rate, self.weight_decay)
        
        # prediction of class happens here
        x = layers.BatchNormalization(name = 'prediction_bn')(x)
        x = layers.Activation('relu',  name = 'prediction_relu', )(x)
        x = layers.GlobalAveragePooling2D( name = 'prediction_pool', )(x)
        prediction = layers.Dense(self.num_classes, name = 'prediction_dense', activation='softmax')(x)

        return tf.keras.Model(inputs=img_input, outputs=prediction, name='DenseNet')
        
        
    def dense_block(self, x, num_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False,
                    weight_decay=1e-4):
        """
        Creates a dense block and concatenates inputs
        """

        for i in range(num_layers):
            cb = self.bn_relu_convolution(x, growth_rate, dropout_rate, 
                                     bottleneck) # 1 conv if bottleneck = 0 else 2 conv if bottleneck = 1
            nb_channels += growth_rate
            x = layers.concatenate([cb, x])
        return x, nb_channels

        
    def bn_relu_convolution(self, x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
        """
        Creates a convolution layers consisting of BN-ReLU-Conv.
        Optional: bottleneck, dropout

        """
        # Bottleneck
        if bottleneck:
            bottleneckWidth = 4
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(nb_channels * bottleneckWidth, (1, 1),
                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
            # Dropout
            if dropout_rate:
                x = layers.Dropout(dropout_rate)(x)

        # BN-ReLU-Conv
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(nb_channels, (3, 3), padding='same')(x)

        # Dropout
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        return x

    def bn_relu_transition(self, x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
        """
        Creates a transition layer between dense blocks as transition, which do convolution and pooling.
        Works as downsampling.
        """

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu',)(x)
        x = layers.Convolution2D(int(nb_channels * compression), (1, 1), padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

        # Adding dropout
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x


# In[ ]:


dense_net = DenseNet(input_shape = (32,32,3), dense_blocks = 3, dense_layers = [16]*3,
                     growth_rate = 12, compression = 0.5, num_classes = 10, bottleneck = True, 
                     dropout_rate = None, weight_decay = 1e-6).build_model()
dense_net.summary()


# In[ ]:


from tensorflow.keras.callbacks import *

# to log results
csv_logger = CSVLogger('training_results.csv')

# top 5 acc
def top5_acc(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


# model checkpoint
file_path='dense_net_cifar10.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose = 0, 
                             save_best_only=True, save_weights_only=True,
                             mode='min')

# reduce LR on plateau
lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', verbose = 0,
                               factor = 0.2, patience = 10, min_lr = 0.000001)

# determine Loss function and Optimizer
dense_net.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'],)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, 
                                   horizontal_flip=True, vertical_flip=True)
train_datagen.fit(X_train)
train_data = train_datagen.flow(X_train, y_train, batch_size = 200)


# In[ ]:


val_datagen = ImageDataGenerator(rescale=1./255)
val_datagen.fit(X_val)
val_data = val_datagen.flow(X_val, y_val, batch_size = 200)


# In[ ]:


# fits the model on batches with real-time data augmentation:
res_history = dense_net.fit_generator(train_data, epochs = 100,
                                      validation_data = (X_val/255.,y_val),
                                      callbacks = [checkpoint, lr_reduced, csv_logger])


# In[ ]:


import matplotlib.pyplot as plt
def draw_metric_plots(res):
    import pandas as pd
    df = pd.DataFrame()
    df['train_loss'] = res.history['loss']
    df['val_loss'] = res.history['val_loss']
    df['train_acc'] = res.history['acc']
    df['val_acc'] = res.history['val_acc']
    df.index = np.arange(1,len(df)+1,1)
    
    # draw Loss
    df[['train_loss', 'val_loss']].plot()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    # draw Acc
    df[['train_acc', 'val_acc']].plot()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    return df


# In[ ]:


res_df = draw_metric_plots(res_history)


# In[ ]:


# y_test_probas = dense_net.predict(X_test/255.)
scores = dense_net.evaluate(X_test/255., y_test)
print(scores)


# ### Results
# 
# - Altough the training obtained are comparable to official paper, there is slight overfitting of the model.

# In[ ]:


dense_net.fit(X_train_val/255., y_train_val, epochs = 25)


# In[ ]:


# y_test_probas = dense_net.predict(X_test/255.)
scores = dense_net.evaluate(X_test/255., y_test)
print(scores)


# In[ ]:


# from tensorflow.keras.utils import plot_model
# plot_model(dense_net, to_file='DenseNet.png')


# In[ ]:


#


# In[ ]:


#

