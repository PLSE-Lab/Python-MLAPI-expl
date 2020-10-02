#!/usr/bin/env python
# coding: utf-8

# **(1) Introduction**
# # 
# # Here is my Image Input/Digit Recognizer colab kernel.

# **(2) Package Imports**

# In[ ]:


get_ipython().system('pip install image_slicer')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
from PIL import Image, ImageFilter
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import image_slicer

get_ipython().run_line_magic('matplotlib', 'inline')


# **(2) Data Import**

# In[ ]:


train = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


data_check = train.head(10)
print(data_check)


# Check the datasets for nulls using the following code.

# In[ ]:


train.isnull().any().value_counts()


# In[ ]:


test.isnull().any().value_counts()


# Checking the numbers.

# In[ ]:


print((train['label'].value_counts()))
sns.countplot(train['label'])


# In[ ]:


# Split the dataset into features (X_) and target variable (Y_)
# Change the data types to suit.
X_train = (train.iloc[:, 1:].values).astype('float32')
Y_train = train.iloc[:, 0].values.astype('int32')

X_test = test.values.astype('float32')


# In[ ]:


# View some example 28x28pixel images of digits.
plt.figure(figsize=(20, 8))
x, y = 10, 4
for i in range(40):
    plt.subplot(y, x, i + 1)
    plt.imshow(X_train[i].reshape((28, 28)), interpolation='nearest')
plt.show()


# In[ ]:


# Function to visualise an example as a grey-scale image.
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    wdth, hgt = img.shape
    thresh = img.max()/2.5
    for x in range(wdth):
        for y in range(hgt):
            ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')


# convert the training set into integer values for the gray-scale comparison.
X_train = (train.iloc[:, 1:].values).astype('int32')
# define the size of the figure
fig = plt.figure(figsize=(12, 12))
# define subplots
ax = fig.add_subplot(111)
# execute visualize_input function with first element in training set
visualize_input(X_train[1].reshape(28, 28), ax)

# convert training set back into float values
X_train = (train.iloc[:, 1:].values).astype('float32')


# In[ ]:


# Convolutional Neural Networks (CNNs) converge quicker on [0->1] data rather than [0->255].
# Code to convert is below.
X_train = X_train/255.0
X_test = X_test/255.0

# The data now needs to be re-shaped to convert the data from 1x784 column row to a 28pixel by 28pixel 'image'.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Encode labels to one hot vectors (e.g. 2 -> [0,0,1,0,0,0,0,0,0,0])
# Dummy transforming the labels. A 2 in the label column would give:
# label = 0: 0, label = 1: 0, label = 2: 1, label = 3: 0, etc...
Y_train = to_categorical(Y_train, num_classes=10)

# Splitting the training dataset into train and validation (test) sets using the sklearn split package.
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=56)


# In[ ]:


train


# An Epoch represents one iteration over the entire dataset.
# # 
# # Because we cannot pass the entire dataset into a neural network at once.
# # 
# # We need to divide the dataset into batches to feed into the NN.
# # 
# # We have 37800 'images' as data and a batch size of 64, then an epoch should contain 37800/64 = ~590 iterations.

# In[ ]:


# Setting the CNN model.
batch_size = 64
epochs = 3
input_shape = (28, 28, 1)

# Sequential API which adds one model at a time.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# Define the optimizer and compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)


# Data augmentation to alter the images to imporve the model and to hinder any overfitting.

# In[ ]:


datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False, zca_whitening=False,
                             rotation_range=15, zoom_range=0.1, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=False, vertical_flip=False)


# In[ ]:


# View a summary of the model
model.summary()


# In[ ]:


datagen.fit(X_train)
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=epochs, validation_data=(X_val, Y_val), verbose=1,
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        callbacks=[learning_rate_reduction])


# In[ ]:


final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)
print("validation loss: {0:.6f}, validation accuracy: {1:.6f}".format(final_loss, final_acc))


# In[ ]:


# Visualise the errors using a confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Digit Confusion matrix',cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(11))


# In[ ]:


errors = (Y_pred_classes - Y_true != 0)
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n += 1


# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

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


# Function to convert an image into 28x28pixel data entry.

def imageprepare(argv):
    '''
    This function returns the pixel values.
    The input is a png file location.
    '''
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    # create white canvas of 28x28pixels.
    newImage = Image.new('L', (28, 28), (255))
    
    # Check which dimension is bigger:
    if width > height:  
        # Width is bigger. Width becomes 20 pixels.
        # Re-size height according to ratio.
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS)             .filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        # Height is bigger. Height becomes 20 pixels.
        # Re-size width according to ratio.
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS)             .filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())

    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


# In[ ]:


# Use the above function to import a 5x5 image of digits and print prediction results.

def prediction_printer(image_name, grid_size):
    splits = grid_size**2
    data = np.array(image_slicer.slice(file + image_name + '.png', splits))
    i_df = pd.DataFrame(data=data)
    i_df[0] = i_df[0].astype(str).str.strip('.png>').str.split(' - ').str[1]
    image_list = i_df[0].tolist()
    df = pd.DataFrame([])
    for img in image_list:
        df = df.append([imageprepare(file + img + ".png")])
    df = (df.iloc[:, 0:].values).astype('float32')
    df = df.reshape(df.shape[0], 28, 28, 1)
    predictions = model.predict_classes(df)
    new_df = pd.DataFrame(np.array_split(predictions.tolist(), grid_size))
    print(new_df.to_string(index=False, header=None))

file = r'/kaggle/input/images/'
prediction_printer(image_name='5x5split', grid_size=5)


# In[ ]:




