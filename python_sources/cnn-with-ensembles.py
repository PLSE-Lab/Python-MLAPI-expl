#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.utils
import scipy.misc
import os
from PIL import Image
sns.set(style='white', context='notebook', palette='deep')
from keras.layers.normalization import BatchNormalization


# # 2. Data preparation
# ## 2.1 Load data

# In[ ]:


# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 


Y_train.value_counts()


# ## 2.2 Check for null and missing values

# In[ ]:


# Check the data
X_train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# ## 2.3 Normalization

# In[ ]:


# Normalize the data
X_train = X_train / 255.0
test = test / 255.0


# ## 2.3 Reshape

# In[ ]:


# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# ## 2.5 Label encoding

# In[ ]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)


# In[ ]:


get_ipython().system('mkdir masks')
get_ipython().system('mkdir train')
get_ipython().system('mkdir test')


# In[ ]:


for i,im in enumerate(X_train):
    scipy.misc.imsave(os.path.join(os.getcwd(),'train/image'+str(i)+'.png'),X_train[i].reshape(28,28))


# In[ ]:


for i,im in enumerate(X_train):
    scipy.misc.imsave(os.path.join(os.getcwd(),'train/image'+str(i)+'.png'),X_train[i].reshape(28,28))


# In[ ]:


X_test = test
del test


# In[ ]:


for i,im in enumerate(X_test):
    scipy.misc.imsave(os.path.join(os.getcwd(),'test/image'+str(i)+'.png'),X_test[i].reshape(28,28))


# In[ ]:


train_im_path = os.getcwd()+'/train'
test_im_path = os.getcwd() +'/test'


# In[ ]:


im2 = plt.imread(os.path.join(train_im_path,os.listdir(train_im_path)[1]))


# ## 2.6 Split training and valdiation set 

# In[ ]:


# Set the random seed
random_seed = 2


# # 3. CNN
# ## 3.1 Define the model

# In[ ]:


import math
import os
import glob

from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback


class Snapshot(Callback):

    def __init__(self, folder_path, nb_epochs, nb_cycles=5, verbose=0):
        if nb_cycles > nb_epochs:
            raise ValueError('nb_epochs has to be lower than nb_cycles.')

        super(Snapshot, self).__init__()
        self.verbose = verbose
        self.folder_path = folder_path
        self.nb_epochs = nb_epochs
        self.nb_cycles = nb_cycles
        self.period = self.nb_epochs // self.nb_cycles
        self.nb_digits = len(str(self.nb_cycles))
        self.path_format = os.path.join(self.folder_path, 'weights_cycle_{}.h5')


    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % self.period != 0: return
        # Only save at the end of a cycle, a not at the beginning

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        cycle = int(epoch / self.period)
        cycle_str = str(cycle).rjust(self.nb_digits, '0')
        self.model.save_weights(self.path_format.format(cycle_str), overwrite=True)

        # Resetting the learning rate
        K.set_value(self.model.optimizer.lr, self.base_lr)

        if self.verbose > 0:
            print('\nEpoch %05d: Reached %d-th cycle, saving model.' % (epoch, cycle))


    def on_epoch_begin(self, epoch, logs=None):
        if epoch <= 0: return

        lr = self.schedule(epoch)
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: Snapshot modifying learning '
                  'rate to %s.' % (epoch + 1, lr))


    def set_model(self, model):
        self.model = model
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get initial learning rate
        self.base_lr = float(K.get_value(self.model.optimizer.lr))


    def schedule(self, epoch):
        lr = math.pi * (epoch % self.period) / self.period
        lr = self.base_lr / 2 * (math.cos(lr) + 1)
        return lr


# In[ ]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',input_shape = (28,28,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same'))
model.add(LeakyReLU(alpha=0.1))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
model.add(LeakyReLU(alpha = 0.1))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

def get_model(tensor):
    x =  Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same')(tensor)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)


    x = Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same')(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same')(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Dropout(0.25)(x)


    x = Flatten()(x)
    x = Dense(256, activation = "relu")(x)
    x = Dropout(0.5)(x)
    x_out = Dense(10, activation = "softmax")(x)
    return Model(inputs = tensor,outputs = x_out)


# In[ ]:



model.compile(
    #optimizer = SGD(lr=0.1,momentum=0.9,nesterov=True),
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

cbs = [Snapshot('snapshots', nb_epochs=6, verbose=1, nb_cycles=2)]


# In[ ]:


batch_size = 86
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_im_path=train_im_path,
                 augmentations=None, batch_size=batch_size,img_size=28, n_channels=3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.train_im_paths = glob.glob(train_im_path+'/*')
        
        self.train_im_path = train_im_path
        self.img_size = img_size
        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.train_im_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.train_im_paths))]

        # Find list of IDs
        list_IDs_im = [self.train_im_paths[k] for k in indexes]

        # Generate data
        X = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X
        else:            
            im = []   
            for x in X:
                augmented = self.augment(image=x)
                im.append(augmented['image'])
            return np.array(im)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_im_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im),self.img_size,self.img_size, self.n_channels))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):
            
            im = np.array(Image.open(im_path))
                        
            
            if len(im.shape)==2:
                im = np.repeat(im[...,None],3,2)

#             # Resize sample
            X[i,] = cv2.resize(im,(self.img_size,self.img_size))

        return np.uint8(X)


# In[ ]:


import cv2
h,w = 28,28
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        ], p=0.3),
    ToFloat(max_value=1)
],p=1)


AUGMENTATIONS_TEST = Compose([
    ToFloat(max_value=1)
],p=1)


# In[ ]:


a = DataGenerator(batch_size=64,shuffle=False)
images = a.__getitem__(0)
max_images = 64
grid_width = 16
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

for i,im in enumerate(images):
    ax = axs[int(i / grid_width), i % grid_width]
    ax.imshow(im.squeeze(), cmap="gray")
    ax.axis('off')


# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


# In[ ]:


datagen = ImageDataGenerator(
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
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


# In[ ]:


epochs = 30
batch_size = 86


# In[ ]:


history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=cbs)


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


def load_ensemble(folder, keep_last=None):
    paths = glob.glob(os.path.join(folder, 'weights_cycle_*.h5'))
    print('Found:', ', '.join(paths))
    if keep_last is not None:
        paths = sorted(paths)[-keep_last:]
    print('Loading:', ', '.join(paths))

    x_in = Input(shape=(28, 28, 1))
    outputs = []

    for i, path in enumerate(paths):
        m = get_model(x_in)
        m.load_weights(path)
        outputs.append(m.output)
    
    shape = outputs[0].get_shape().as_list()
    
    x = Lambda(lambda x: K.mean(K.stack(x, axis=0), axis=0),
               output_shape=lambda _: shape)(outputs)
    
    
    model = Model(inputs=x_in, outputs=x)
    return model


model = load_ensemble('snapshots')
model.compile(
    optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

metrics = model.evaluate(X_val, Y_val)
print(metrics)


# In[ ]:


# Display some error results 
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    plt.subplots_adjust(top = 0.5, bottom=0.01, hspace=1.5, wspace=0.4)

    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


# In[ ]:


# predict results
results = model.predict(X_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)

