#!/usr/bin/env python
# coding: utf-8

# #  Low Parameter Inception Model for MNIST
# 
# This kernel provides an overview of a low parameter Inception model that is capable of 99.7% test accuracy in Kaggle's [Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer).
# 
# I have a slow computer but I still wanted to work on deep learning competitions. 
# To make things interesting (and feasible with my computer), I tried to create the lowest parameter model I could for MNIST.
# The end result is an Inception model that uses 76,264 parameters.
# As far as I can tell, other models that get state of the art results on MNIST all use 200,000 or more parameters.
# 
# - [Load and Prepare Data](#Load-and-Prepare-Data)   
# 
# - [Inception](#Inception)
#     - [Inception Modules](#Inception-Modules)
#     - [Inception Model](#Inception-Model)  
# 
# - [Training](#Training)
#     - [Dropout](#Dropout)
#     - [Label Smoothing](#Label-Smoothing)
#     - [Data Augmentation](#Data-Augmentation)
#     - [Cosine Annealing](#Cosine-Annealing)
#     - [Adamax](#Adamax)   
#     - [Train](#Train)
# 
# - [Model Evaluation](#Model-Evaluation) 
# 
# - [Predict on Test Set](#Predict-on-Test-Set)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import time

import keras
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


keras_version = keras.__version__
tf_version = K.tensorflow_backend.tf.VERSION

print("keras version:", keras_version)
print(K.backend(), "version:", tf_version)


# ## Load and Prepare Data
# 
# This part is pretty standard but there are a few things to note. The pixel values are scaled to the range $[-1, 1]$. The validation set is selected with a fixed seed so as to be reproducible. I use a validation set that is balanced across classes because it makes it easier to evaluate model behavior on MNIST&mdash;particularly when looking at misclassified examples and class bias.

# In[ ]:


# load data
rawdata = np.loadtxt('../input/train.csv', dtype=int, delimiter=',', skiprows=1)


# In[ ]:


# inspect data
print("Raw data shape:", rawdata.shape)

# split labels and pixel values
y = rawdata[:, 0]
X = rawdata[:, 1:]
print("Labels shape:", y.shape)
print("Pixels shape:", X.shape)

# convert pixel values to 2d arrays
X = np.reshape(X, (42000, 28, 28))
print("Pixels reshaped shape:", X.shape)

# display random sample of images
plt.rcParams['figure.figsize'] = (10.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

num_classes = 10
samples_per_class = 4
for cls in range(num_classes):
    idxs = np.flatnonzero(y == cls)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + cls + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X[idx])
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# In[ ]:


# one hot encode labels
y_oh = to_categorical(y, num_classes)

# scale pixel values to be between -1 and 1
X_scaled = X / 127.5 - 1
X_scaled = np.expand_dims(X_scaled, -1) # channels last

# split data into train set and balanced validation set
num_val = int(y.shape[0] * 0.1)
validation_mask = np.zeros(y.shape[0], np.bool)
np.random.seed(1)
for c in range(num_classes):
    idxs = np.random.choice(np.flatnonzero(y == c), num_val // 10, replace=False)
    validation_mask[idxs] = 1
np.random.seed(None)  
    
X_train = X_scaled[~validation_mask]
X_val = X_scaled[validation_mask]
print("Train/val pixel shapes:", X_train.shape, X_val.shape)

y_train = y_oh[~validation_mask]
y_val = y_oh[validation_mask]
print("Train/val label shapes:", y_train.shape, y_val.shape)

# confirm validation set is balanced across classes
print("Validation Set Class Distribution:", np.bincount(y[validation_mask]))


# ## Inception
# 
# The inception model was introduced in ["Going deeper with convolutions"](https://arxiv.org/abs/1409.4842) (Szegedy et al, 2014).
# The model evolved in ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567) (Szegedy et al, 2015) and mature models were outlined in ["Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"](https://arxiv.org/abs/1602.07261) (Szegedy et al, 2016). 
# Inception modules are the core of an inception model, but the way the modules and other pieces fit together is also important.
# The current basic inception module consists of:
# - Batch Normalized Convolutions
# - 1 x 1 Convolutions for Dimensionality Reduction
# - Internal Branches with Different Spatial Coverages
# - Pooling Branch
# - Some Modules use Asymmetric Convolutions
# 
# The basic model format is as follows:
# 
# `Stem` &rarr; `Inception Module Stacks` &rarr; `Global Average Pooling` &rarr; `Softmax`,
# 
# where the Stem is convolutional and pooling layers and the Inception Module Stacks include reduction modules/pooling. 
# Inception models lend themselves well to low parameter models because:
# - The 1 x 1 convolutions do efficient dimensionality reduction which means the next convolution uses fewer parameters for the incoming data.
# - Asymmetric convolutions in modules use fewer parameters for the same spatial coverage as a symmetric convolution.
# - Global average pooling results in fewer parameters than flattening.
# - There are no hidden layers.

# ### Inception Modules

# In[ ]:


def conv2D_bn_relu(x, filters, kernel_size, strides, padding='valid', kernel_initializer='glorot_uniform', name=None):
    """2D convolution with batch normalization and ReLU activation.
    """
    
    x = layers.Conv2D(filters=filters, 
                      kernel_size=kernel_size, 
                      strides=strides, 
                      padding=padding, 
                      kernel_initializer=kernel_initializer,
                      name=name,
                      use_bias=False)(x)
    x = layers.BatchNormalization(scale=False)(x)
    return layers.Activation('relu')(x)


def inception_module_A(x, filters=None, kernel_initializer='glorot_uniform'):
    """Inception module A as described in Figure 4 of "Inception-v4, Inception-ResNet 
    and the Impact of Residual Connections on Learning" (Szegedy, et al. 2016).
    
    # Arguments
        x: 4D tensor with shape: `(batch, rows, cols, channels)`.
        filters: Number of output filters for the module.
        kernel_initializer: Weight initializer for all convolutional layers in module.
    """
    
    if filters is None:
        filters = int(x.shape[-1])
    branch_filters = filters // 4
        
    b1 = conv2D_bn_relu(x, 
                        filters=(branch_filters // 3) * 2, 
                        kernel_size=1, 
                        strides=1, 
                        kernel_initializer=kernel_initializer)
    b1 = conv2D_bn_relu(b1, 
                        filters=branch_filters, 
                        kernel_size=3, 
                        strides=1, 
                        padding='same', 
                        kernel_initializer=kernel_initializer)
    
    b2 = conv2D_bn_relu(x, 
                        filters=(branch_filters // 3) * 2, 
                        kernel_size=1, 
                        strides=1, 
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters=branch_filters, 
                        kernel_size=3, 
                        strides=1, 
                        padding='same', 
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters=branch_filters, 
                        kernel_size=3, 
                        strides=1, 
                        padding='same', 
                        kernel_initializer=kernel_initializer)
        
    b3 = conv2D_bn_relu(x, 
                        filters=branch_filters, 
                        kernel_size=1, 
                        strides=1, 
                        kernel_initializer=kernel_initializer)
    
    pool = layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    pool = conv2D_bn_relu(pool, 
                          filters=branch_filters, 
                          kernel_size=1, 
                          strides=1, 
                          kernel_initializer=kernel_initializer)

    return layers.concatenate([b1, b2, b3, pool])


def inception_module_C(x, filters=None, kernel_initializer='glorot_uniform'):
    """Inception module C as described in Figure 6 of "Inception-v4, Inception-ResNet 
    and the Impact of Residual Connections on Learning" (Szegedy, et al. 2016).
    
    # Arguments
        x: 4D tensor with shape: `(batch, rows, cols, channels)`.
        filters: Number of output filters for the module.
        kernel_initializer: Weight initializer for all convolutional layers in module.
    """
        
    if filters is None:
        filters = int(x.shape[-1])
    branch_filters = filters // 6
        
    b1 = conv2D_bn_relu(x, 
                        filters=(branch_filters // 2) * 3, 
                        kernel_size=1, 
                        strides=1, 
                        kernel_initializer=kernel_initializer)
        
    b1a = conv2D_bn_relu(b1, 
                         filters=branch_filters, 
                         kernel_size=(1, 3), 
                         strides=1, 
                         padding='same', 
                         kernel_initializer=kernel_initializer)
    
    b1b = conv2D_bn_relu(b1, 
                         filters=branch_filters, 
                         kernel_size=(3, 1), 
                         strides=1, 
                         padding='same', 
                         kernel_initializer=kernel_initializer)
    
    b2 = conv2D_bn_relu(x, 
                        filters=(branch_filters // 2) * 3, 
                        kernel_size=1, 
                        strides=1, 
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters=(branch_filters // 4) * 7, 
                        kernel_size=(1, 3), 
                        strides=1, 
                        padding='same', 
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters=branch_filters * 2, 
                        kernel_size=(3, 1), 
                        strides=1, 
                        padding='same', 
                        kernel_initializer=kernel_initializer)

    b2a = conv2D_bn_relu(b2, 
                         filters=branch_filters, 
                         kernel_size=(1, 3), 
                         strides=1, 
                         padding='same', 
                         kernel_initializer=kernel_initializer)
    
    b2b = conv2D_bn_relu(b2, 
                         branch_filters, 
                         kernel_size=(3, 1), 
                         strides=1, 
                         padding='same', 
                         kernel_initializer=kernel_initializer)
        
    b3 = conv2D_bn_relu(x, 
                        filters=branch_filters, 
                        kernel_size=1, 
                        strides=1, 
                        kernel_initializer=kernel_initializer)
    
    pool = layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    pool = conv2D_bn_relu(pool, 
                          filters=branch_filters, 
                          kernel_size=1, 
                          strides=1, 
                          kernel_initializer=kernel_initializer)
    
    return layers.concatenate([b1a, b1b, b2a, b2b, b3, pool])


# ### Inception Model

# In[ ]:


K.clear_session()

stem_width = 64

inputs = layers.Input(shape=X_scaled.shape[1:])
x = conv2D_bn_relu(inputs,
                   filters=stem_width,
                   kernel_size=5,
                   strides=2,
                   padding='valid',
                   name='conv_1')

x = inception_module_A(x, filters=int(1.5*stem_width))
x = layers.SpatialDropout2D(0.2)(x)

x = inception_module_A(x, filters=int(1.5*stem_width))
x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
x = layers.SpatialDropout2D(0.2)(x)

x = inception_module_C(x, filters=int(2.25*stem_width))
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(num_classes, name='logits')(x)
x = layers.Activation('softmax', name='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()


# ## Training

# ### Dropout
# 
# The reader is probably familar with using [dropout](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) (Srivastava et al, 2014) to prevent overfitting. 
# For convolutional neural networks there is also [spatial dropout](https://arxiv.org/abs/1411.4280) (Tompson et al, 2014), which is similar to dropout but drops entire channels instead of individual coordinates. 

# ### Label Smoothing
# This is a regularization method from section *7. Model Regularization via Label Smoothing* in ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567) (Szegedy et al, 2015). I found that it resulted in lower validation loss compared to validation loss in models that did not use label smoothing, though I'm not sure if it helped validation accuracy.

# In[ ]:


epsilon = 0.001
y_train_smooth = y_train * (1 - epsilon) + epsilon / 10
print(y_train_smooth)


# ### Data Augmentation
# 
# My Kaggle kernel [MNIST Data Augmentation with Elastic Distortion](https://www.kaggle.com/babbler/mnist-data-augmentation-with-elastic-distortion) has an overview with visualizations of the data augmentation methods used here. To get the best results, I had to:
# - Use [elastic distortion](https://www.microsoft.com/en-us/research/wp-content/uploads/2003/08/icdar03.pdf) (Simard et al, 2003).
# - Shift images by pixel values to avoid interpolation blur.
# 
# #### Elastic Distortion Function
# 
# Credit to the following gists for the basic function:
# 
# - https://gist.github.com/fmder/e28813c1e8721830ff9c
# - https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
# - https://gist.github.com/erniejunior/601cdf56d2b424757de5

# In[ ]:


from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def elastic_transform(image, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       
   # Arguments
       image: Numpy array with shape (height, width, channels). 
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """
    
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


# ### Cosine Annealing
# 
# [Cosine annealing](https://arxiv.org/abs/1608.03983) (Loshchilov & Hutter, 2017) is a relatively new learning rate annealing technique that does a more thorough job of exploring the model's solution space by using warm restarts to break out of local minimums.
# As the learning rate decreases, the model gets more precise but may also get stuck in a particular state. Warm restarts raise the learning rate to get the model unstuck. I found that, as long as the model doesn't overfit on the training set, continual warm restarts can potentially discover better and better models. 

# In[ ]:


class CosineAnneal(keras.callbacks.Callback):
    """"Cosine annealing with warm restarts.
    
    As described in section 3 of "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter 2017).
    
    # Arguments
        max_lr: Maximum value of learning rate range.
        min_lr: Minimum value of learning rate range.
        T: Number of epochs between warm restarts.
        T_mul: At warm restarts, multiply `T` by this amount.
        decay_rate: At warm restarts, multiply the max_lr and min_lr by this amount.
    """
    def __init__(self, max_lr, min_lr, T, T_mul=1, decay_rate=1.0):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.T = T
        self.T_cur = 0
        self.T_mul = T_mul
        self.step = 0
        
    def on_batch_begin(self, batch, logs=None):
        if self.T <= self.T_cur:
            self.max_lr *= self.decay_rate
            self.min_lr *= self.decay_rate
            self.T *= self.T_mul
            self.T_cur = 0
            self.step = 0
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(self.T_cur * np.pi / self.T))        
        K.set_value(self.model.optimizer.lr, lr)
        # use self.step to avoid floating point arithmetic errors at warm restarts
        self.step += 1
        self.T_cur = self.step / self.params['steps']
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


# ### Adamax
# The Adamax optimizer was introduced alongside Adam in ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2014). While Adam is more popular, I found that Adamax is less prone to overfitting. Adamax also worked better than SGD with momentum when used in conjunction with cosine annealing.

# In[ ]:


model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.Adamax(lr=0.01, beta_1=0.49, beta_2=0.999),
              metrics=['accuracy'])


# ### Train
# 
# I save the best model for final evaluation and use enough epochs so cosine annealing can thoroughly explore the solution space.

# In[ ]:


batch_size = 32
epochs = 100
time_id = time.strftime("%Y-%m-%d_%H-%M-%S")
best_model_filename = 'mnist-inception-best-' + time_id + '.hdf5'

# save weights for visualization
pre_train_weights = model.get_layer('conv_1').get_weights()[0]
pre_train_weights = pre_train_weights.transpose(3, 2, 0, 1)

# setup callbacks
annealer = CosineAnneal(max_lr=0.014, min_lr=0.003, T=5, T_mul=1, decay_rate=0.99)
chkpt = keras.callbacks.ModelCheckpoint(best_model_filename, monitor='val_acc', 
                                        save_best_only=True, verbose=False)

# define data augmentations
datagen = ImageDataGenerator(
    width_shift_range=2,
    height_shift_range=2,
    preprocessing_function=lambda x: elastic_transform(x, alpha_range=[8, 10], sigma=3)
)

# train model
history = model.fit_generator(
    datagen.flow(X_train, y_train_smooth, batch_size=batch_size, shuffle=True),
    epochs=epochs,
    steps_per_epoch=(len(y_train) - 1) // batch_size + 1,
    validation_data=(X_val, y_val),
    callbacks=[annealer, chkpt]
)


# ## Model Evaluation

# In[ ]:


def plot_training_log(log, acc=True, loss=True, lr=True, figsize=(10.0, 2.5)):
    """Plot training history.
    
    # Arguments
        log: Dictionary of training history, same as ` History.history`  that  
            is returned when fitting a Keras model. Should have records for 'acc', 
            'val_acc', 'loss', 'val_loss', and optionally 'lr'.
        acc: if true, plot both 'acc' and 'val_acc' on one plot.
        loss: if true, plot both 'loss' and 'val_loss' on one plot.
        lr: if true, and if 'lr' is in log, plot both 'lr' and 'val_acc' on one plot.
        figsize: size of each plot.
    """
    
    plt.rcParams['figure.figsize'] = figsize
    max_val_acc_epoch = np.argmax(list(log['val_acc'])) + 1
    epochs = range(1, len(log['acc']) + 1)
    
    def plot(ytype, ylabel, max_val_acc_epoch):
        plt.axvline(x=max_val_acc_epoch, color='0.5', linestyle='--')
        plt.plot(epochs, log[ytype], label='Train')
        plt.plot(epochs, log['val_' + ytype], label='Validation')
        plt.minorticks_on()
        plt.grid(b=True, axis='x', which='both', color='0.8', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.xlim(0, epochs[-1] + 1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    if acc:
        plot('acc', 'Accuracy', max_val_acc_epoch)
        
    if loss:
        plot('loss', 'Loss', max_val_acc_epoch)
    
    if lr and 'lr' in log.keys():
        fig, ax1 = plt.subplots()
        plt.axvline(x=max_val_acc_epoch, color='0.5', linestyle='--')

        ln1 = ax1.plot(epochs, log['lr'], 'C4o-', label='Learning Rate')   
        ax1.set_xticks(epochs, minor=True)
        ax1.grid(b=True, axis='x', which='both', color='0.8', linestyle='-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.set_xlim(0, epochs[-1] + 1)

        ax2 = ax1.twinx()
        ln2 = ax2.plot(epochs, log['val_acc'], 'C1o-', 
                       label='Validation Accuracy [0.98 - 1.0]')
        ax2.set_yticks([])
        ax2.set_ylim(0.98, 1.0)        

        lns = ln1 + ln2
        plt.legend(lns, [l.get_label() for l in lns])
        plt.tight_layout()
        plt.show()
        
        
def evaluate_model(model, X, y, log=None, pre_train_weights=None):
    """Display accuracy, display misclassified digits, and visualize weights in 
    first convolution layer for the given model with X and y as prediction inputs.
    Optionally, accuracy and loss is plotted with the log argument and additional
    weights can be visualized with the pre_train_weights argument.
    
    # Arguments
        model: Keras model with first convolution layer named 'conv_1'.
        X: Numpy array of MNIST digits, in channels last format.
        y: Labels for X.
        log: (optional) Dictionary of training history, same as ` History.history`  that  
            is returned when fitting a Keras model. Should have records for 'acc', 
            'val_acc', 'loss', 'val_loss', and optionally 'lr'.
        pre_train_weights: (optional) Numpy array of weights in first convolution layer 
            of the model.
    """
 
    if log is not None:
        print("Max Validation Accuracy:", max(log['val_acc']))
        plot_training_log(log)

    scores = model.predict(X)
    predictions = np.argmax(scores, axis=1)
    y_digits = np.nonzero(y)[1]       
    print("Accuracy:",  np.mean(predictions == y_digits))
    
    post_train_weights = model.get_layer('conv_1').get_weights()[0]
    post_train_weights = post_train_weights.transpose(3, 2, 0, 1)
    num_weights = len(post_train_weights)
    if pre_train_weights is not None:
        plt.rcParams['figure.figsize'] = (10.0, 3.0)
        for i in range(num_weights):
            plt.subplot(4, num_weights // 4, i + 1)
            ker = pre_train_weights[i, 0]
            low, high = np.amin(ker), np.max(ker)
            plt.imshow(255 * (ker - low) / (high - low))
            plt.axis('off')
        plt.suptitle('Pre-Training Weights from First Convolutional Layer')
        plt.show()
    plt.rcParams['figure.figsize'] = (10.0, 3.0)
    for i in range(num_weights):
        plt.subplot(4, num_weights // 4, i + 1)
        ker = post_train_weights[i, 0]
        low, high = np.amin(ker), np.max(ker)
        plt.imshow(255 * (ker - low) / (high - low))
        plt.axis('off')
    plt.suptitle('Post-Training Weights from First Convolutional Layer')
    plt.show()

    misclassified_mask = (predictions != y_digits)
    samples_per_class = 7
    num_classes = 10
    plt.rcParams['figure.figsize'] = (10.0, 7.0)    
    counts = [];
    for cls in range(10):
        idxs = np.flatnonzero(y_digits[misclassified_mask] == cls)
        counts.append(len(idxs))
        if len(idxs) > samples_per_class:
            idxs = np.random.choice(idxs, samples_per_class, replace=False)       
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + cls + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X[misclassified_mask][idx,:,:,0])
            plt.axis('off')
            plt.text(14, 27, 'Pred:' + str(predictions[misclassified_mask][idx]), 
                     horizontalalignment='center', verticalalignment='top')        
            if i == 0:
                plt.title(cls)            
    plt.suptitle('Misclassified Digits | Norm of Counts = ' + str(np.linalg.norm(counts)))
    plt.show()


# #### How to Tell if the Model is Good:
# 
# While this method is capable of generating a model that can get 99.7% test accuracy, it's not guaranteed.
# Here are some of the ways I evaluate models.
# I should note that the model that got 99.7% test accuracy only got 99.62% validation accuracy. However, it performed very well in all my other evaluation measures.
# 
# - Model Behavior:
# 	1. Check that misclassification examples make sense. In most cases a human won't mistake a 3 for an 8. On the other hand, to me at least, 4 and 9 sometimes look similar. The model should have the same behavior.
# 	2. Avoid bias for or against particular classes. The norm of misclassified counts was a quick and dirty way to see this. I knew from experience with the data set that I wanted to see norms less than 8.
# - Training Metrics (loss & accuracy):
# 	1. Review the metric history from previous training runs and look for models with metrics in top X%.
# 	2. Look at metric noise over the training run&mdash;I found that huge jumps up and down from epoch to epoch would indicate the model was poor, even if validation accuracy was high. Note that I'm not refering to the jumps caused by warm restarts.
# 	3. Look for models with similar test and validation accuracy. Validation accuracy should be highr than test accuracy but not by much.
# 	4. The rate of convergence can be useful but I find it too subjective to give guidelines. Pay attention to how quickly the metrics change and eventually you'll get a feel for when a model is good.
# - Does the Model Improve with Additional Tuning Measures:
# 	1. Check for improvement with warm restarts&mdash;if there's still an upward trend in validation accuracy after 10 warm restarts, keep going.
# 	2. Check for improvement after fine tuning with SGD/Adamax and a low learning rate.
# 	3. Compare to an ensemble of models. For example, save multiple top performing models and use them to make ensembles.

# In[ ]:


model = keras.models.load_model(best_model_filename)
evaluate_model(model, X_val, y_val, log=history.history, pre_train_weights=pre_train_weights)


# ## Predict on Test Set

# In[ ]:


# load test set data
test_data = np.loadtxt('../input/test.csv', dtype=int, delimiter=',', skiprows=1)


# In[ ]:


# inspect data
print("Raw data shape:", test_data.shape)

# convert pixel values to 2d arrays and scale data
test_images = np.reshape(test_data, (28000, 28, 28))
test_images = test_images / 127.5 - 1
test_images = np.expand_dims(test_images, -1)
print("Pixels reshaped shape:", test_images.shape)

# display some images
plt.rcParams['figure.figsize'] = (10.0, 4.0)
for i in range(40):
    plt.subplot(4, 10, i + 1)
    plt.imshow(test_images[i,:,:,0])
    plt.axis('off')
plt.show()


# In[ ]:


# get predictions
test_scores = model.predict(test_images)
test_predictions = np.argmax(test_scores, axis=1)


# In[ ]:


# save submission csv
header = 'ImageId,Label'
submission = np.stack((range(1, 28001), test_predictions), axis=1)
np.savetxt('submission-' + time_id + '.csv', submission, fmt='%i', delimiter=',', header=header, comments='')

