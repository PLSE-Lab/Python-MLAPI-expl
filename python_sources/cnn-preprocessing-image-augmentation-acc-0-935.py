#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks with Additional Preprocessing and Image Augmentation

# ## Content
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.
# For more detail look here: [https://www.kaggle.com/zalando-research/fashionmnist](https://www.kaggle.com/zalando-research/fashionmnist)

# This type of computations may be long, so I start with timer setting to know how much time the script will take.

# In[ ]:


import time
from time import perf_counter as timer
start = timer()


# Importing necessary modules:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow as imshow
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelBinarizer
from skimage import exposure


#  ## Data Load and Check

# In[ ]:


train = pd.read_csv('../input/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashion-mnist_test.csv')
train.head(8)


# In[ ]:


train.shape


# Our 'train' set is reworked to reduce a data size. In particular all images are in grayscale and their sizes are 28 * 28 pixels. I will show pictures  in a few steps.

# ## Data Object Preparation and Outcome Balance Check
# Let us create arrays for keras from our data. At first I take a look at labels.

# In[ ]:


labels = train['label'].values
labels[0:10]


# In[ ]:


unique_val = np.array(labels)
np.unique(unique_val)


# Is our data balanced?

# In[ ]:


plt.figure(figsize = (18,8))
sns.countplot(x =labels)


# As you can see all output numbers are about the same.
# 
# For our CNN network  I'm to create an output array with Label Binarizer from the labels.

# In[ ]:


label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
labels


# Now I drop the label column from the 'train' set and will work with the rest of data.

# In[ ]:


train.drop('label', axis = 1, inplace = True)


# At this point we can transform our data as numpy array, look at its data type, range and dimensions.

# In[ ]:


images = train.values
print(images.dtype, np.round(images.min(), 4), np.round(images.max(), 4), images.shape)


# Let us see provided images using first 10 rows. 

# In[ ]:


def plot_10(imgs):
    plt.style.use('grayscale')
    fig, axs = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
    for i in range(2): 
        for j in range(5):
            axs[i,j].imshow((225-images[5*i+j]).reshape(28,28))
    fig.suptitle('Grayscale images:\n a Pullover,   an Ankle boot,   a Shirt,   a T-shirt/top,   a Dress,\n' +
            'a Coat,   a Coat,    a Sandal,    a Coat,   a Bag')

plot_10(images)


# A coat and a shirt do not differ much.  The only difference I see is that a coat has more or less horisontal straight hem, and a shirt has a rounded one. This means that a difference is an outline.  
# 
# I wanted to see if you can preprosses images in some way to enchance recognition, and I found the `skimage` package which is supposed to do this.  For it our numbers must be in [0, 1] range, so I'm scaling them now. I need to do it for CNN anyway. I will try 3 kind of preprocessing. 

# In[ ]:


images = images/255


# In[ ]:


new_images = np.zeros((10, 784))
for i in range(10):
        new_images[i,:]= exposure.rescale_intensity(images[i, :], in_range=(0.045, 0.955))
        
plot_10(new_images)


# In[ ]:


warnings.simplefilter('once')
for i in range(10):
        new_images[i,:]= exposure.equalize_hist(images[i, :].reshape(28, 28), nbins=100).flatten() #, clip_limit=0.03, nbins=200

plot_10(new_images)


# In[ ]:


new_images = np.zeros((10, 784))
for i in range(10):
        new_images[i,:]= exposure.adjust_gamma(images[i, :])
        
plot_10(new_images)


# The the functions do not help much, although do not spoil our results as well. I will apply the first one just to show how it is done. Using numpy method to apply a function to each row is much more efficient than a loop.

# In[ ]:


def my_prep(x):
    x = exposure.adjust_gamma(x)
    return x

images = np.apply_along_axis(my_prep, 1, images)
images.shape


# For validation during a model fitting we need to divide our train set in two parts. I did not set a random state parameter because there are a lot of randomly generated values in CNN anyway, and I wanted to see how my results will change with each run.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, stratify = labels, test_size = 0.2)


# Now I need to reshape our rows as square tables because I want to use a Convolution Neural Network method.

# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# ## Convolutional Neural Network Model, or CNN
# For CNN I am using keras library here.

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator


# Setting a number of classes,  a batch size and a number of epochs.

# In[ ]:


num_classes = 10
batch_size = 500
epochs = 100


# Here goes the CNN in all its glory!

# In[ ]:


img_rows, img_cols = 28, 28

model = Sequential()
model.add(Conv2D(160, kernel_size=(6, 6),
                 padding = "same",
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(img_rows, img_cols ,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (4, 4), padding = "same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3,3), padding = "same", activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(),#keras.optimizers.Nadam()
              metrics=['accuracy'])


# This part is for image augmentation during model fitting.

# In[ ]:


train_datagen = ImageDataGenerator(shear_range = 0.1,
                                   zoom_range = [.95, 1.0],
                                   rotation_range = 10,
                                   horizontal_flip = True,
                                   fill_mode = 'constant', cval = 0,
                                   brightness_range = [.6, 1],
                                   width_shift_range = [ -2, -1, 0, +1, +2],
                                   height_shift_range = [-1, 0, +1])
test_datagen = ImageDataGenerator()


# See it running!

# In[ ]:


history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, verbose=2, batch_size=batch_size)


# You can see below how accuracy values improve with each epoch.

# In[ ]:


plt.style.use('tableau-colorblind10')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim(0.85, 0.98)
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()


# I would like to see a historgram of computed validated accuracies.

# In[ ]:


fig = plt.hist(history.history['val_acc'][60:], bins=8)
# when you want to run this code quickly with fewer epochs
#plt.hist(history.history['val_acc'], bins=8)
fig = plt.figure()
fig.savefig('plot.png')


# Let's validate with the test data. At first it must be brought to the same format so I apply the same preprocessing as I did with our data for model fitting. It means that its label column must be removed and rows must be reshaped as square arrays, and applying the same image modification (I mean my_prep function).

# In[ ]:


test_labels = test.iloc[:, 0]
test.drop('label', axis = 1, inplace = True)
test_images = test.values/255
test_images = np.apply_along_axis(my_prep, 1, test_images)
labels_as_array = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images.shape


# And for a moment of truth: checking predictions on a test set. Here are predictions and an accuracy on our provided test set. An accuracy for another run of the provided script may fluctuate due to randomness of applyed methods. The above histogram is likely to reflect possible changes.

# In[ ]:


y_pred = model.predict(test_images).round()
from sklearn.metrics import accuracy_score
accuracy_score(labels_as_array, y_pred)


# Let us look in more detail.

# In[ ]:


from sklearn.metrics import confusion_matrix
class_names = ['T-shirt/top',  'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cm = pd.DataFrame(confusion_matrix(test_labels.values, label_binrizer.inverse_transform(y_pred)))
cm.assign(Classes = class_names)


# As we see a shirt and a t-shirt are the most confused items.
# 
# ### My time count

# In[ ]:


end = timer()
elapsed_time = time.gmtime(end - start)
print("Elapsed time:")
print("{0} minutes {1} seconds.".format(elapsed_time.tm_min, elapsed_time.tm_sec))

