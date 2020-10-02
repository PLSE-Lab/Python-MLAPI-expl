#!/usr/bin/env python
# coding: utf-8

# # Analysing Chest X-Rays for Classification of Pneumonia
# 
# Basic analysis and prediction of the dataset, with formation and training of basic ConvNets. The first ConvNet uses basic accuracy as an evaluation metric, whilst the second improves on this by usage of a custom F1 Score metric.
# 
# During initial data exploration, the original directory of x-ray images is copied to the Kaggle working directory after re-sizing the images. In addition, 20% of the original training dataset is added to the validation set to correct the issues with the original data provided (only 8 of each class in the validation set).

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import seaborn as sns
import shutil

from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from pathlib import Path
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image

from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage import exposure

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import tensorflow as tf
import keras.backend as keras


# ---
#  
# ## 1. Dataset exploration and pre-processing

# In[ ]:


np.random.seed(1)
tf.random.set_seed(1)


# In[ ]:


# create directory paths
old_base_dir = os.path.join('/kaggle/input/chest-xray-pneumonia', 'chest_xray')
old_train_dir = os.path.join(old_base_dir, 'train')
old_val_dir = os.path.join(old_base_dir, 'val')
old_test_dir = os.path.join(old_base_dir, 'test')


# In[ ]:


def count_data(base_dir, directories):
    """ Count number of files in selected sub-dirs, where directories 
        is a list of strings for each sub-directory path. Also returns a 
        dictionary of all img paths (values) for each sub-dir (keys).
    """
    
    # list to store img counts, and dict to store image paths
    file_counts = []
    img_paths = {}
    
    for directory in directories:
        
        img_files = [x for x in os.listdir(os.path.join(base_dir, directory)) 
                 if x.endswith('.jpeg')]
        
        # find paths to all imgs
        path_names = [os.path.join(base_dir, directory, x) for x in img_files]
        
        # count img no. and append to file counts
        num_files = len(img_files)
        file_counts.append(num_files)
    
        # update dict of paths with the given imgs for the sub-dir
        key_name = directory.replace('/', '_').lower()
        img_paths[key_name] = path_names
    
    return file_counts, img_paths


# In[ ]:


split_type = ['train', 'val', 'test']
class_type = ['PNEUMONIA', 'NORMAL']
directories = [f"{x}/{y}" for x in split_type for y in class_type]

counts, img_paths = count_data(old_base_dir, directories)

for subdir, count in zip(directories, counts):
    print(f"{subdir} : {count}")
    
sns.barplot(y=directories, x=counts)
plt.show()


# The default distribution of labels, as shown above, is fundamentally flawed. Having only 8 validation samples for each class is not enough to ensure sufficient validation of our model and adjustment of hyper-parameters during training. To correct this, we'll rebalance the dataset accordingly. We'll move 20% of samples currently in the training data into the validation data.
# 
# To do this in a kaggle kernel, we'll have to move the read-only input data into our working directory, but during this move we'll rebalance the data as required.

# ###### Move original x-ray data to Kaggle working directory after resizing

# In[ ]:


# create new directory paths
base_dir = os.path.join('/kaggle/working', 'chest_xray')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# iterate through each sub-dir - count imgs and form dict of paths\nfor directory in directories:\n    \n    # create new directory structure in kaggle working dir\n    new_dir = os.path.join(base_dir, directory)\n    Path(new_dir).mkdir(parents=True, exist_ok=True)\n    \n    # gather img files in kaggle read-only dir\n    img_files = [x for x in os.listdir(os.path.join(old_base_dir, directory)) \n                 if x.endswith(\'.jpeg\')]\n    \n    # find paths to old and new paths for images in current directory\n    old_path_names = [os.path.join(old_base_dir, directory, x) for x in img_files]\n    new_path_names = [os.path.join(new_dir, x) for x in img_files]\n    \n    print(f"Moving and resizing directory: {directory}\\n")\n    \n    for i in range(len(old_path_names)):\n        \n        # load img, resize, and save to new location\n        img = Image.open(old_path_names[i])\n        img_new = img.resize((360,320), Image.ANTIALIAS)\n        img_new.save(new_path_names[i], \'JPEG\', quality=90)')


# ###### WARNING - only run this once per notebook session, since it will relocate images from the training to validation directories.

# In[ ]:


def move_img_data(source_dir, destination_dir, proportion=0.2, suffix='.jpeg'):
    """ Move a random proportion of img data from a source to destination directory """
    
    img_files = [x for x in os.listdir(source_dir) if x.endswith(suffix)]
    
    move_num = int(np.ceil(len(img_files)*proportion))
    
    # select random proportion of images to move
    random_indices = np.random.permutation(len(img_files))[:move_num]
    
    print(f"Moving a total of {move_num} images from "
          f"{source_dir} to {destination_dir}\n")
    
    # move selected images to destination loc
    for index in random_indices:
        src_path = os.path.join(source_dir, img_files[index])
        dest_path = os.path.join(destination_dir, img_files[index])
        shutil.copyfile(src_path, dest_path)


# In[ ]:


# move 20% of training samples from train to val dir for both classes - ONLY RUN ONCE
move_img_data(os.path.join(train_dir, 'NORMAL'), 
              os.path.join(validation_dir, 'NORMAL'),
              proportion=0.2)
move_img_data(os.path.join(train_dir, 'PNEUMONIA'), 
              os.path.join(validation_dir, 'PNEUMONIA'),
              proportion=0.2)


# ###### Re-run the previous code to count the number of labels distributed throughout our directories

# In[ ]:


counts, img_paths = count_data(base_dir, directories)

for subdir, count in zip(directories, counts):
    print(f"{subdir} : {count}")
    
sns.barplot(y=directories, x=counts)
plt.show()


# Much better! Now make sure the above code for moving the images from the training to validation directory is commented out, otherwise we'll loose more and more data from the training directory each time our notebook is run.
# 
# We've still got issues with an imbalance of data classes, however this can be rectified later on. We can choose to either duplicate under-represented data, or apply data augmentation to our images.
# 
# **Note:** Before moving on in this notebook, consider downloading this new re-balanced dataset, and then uploading it to the kernel as new input data. Then you can comment out all the movement and resizing code above. This will prevent the kernel outputing thousands of images each time you run it in the future. At the bottom of this kernel, we will remove this dataset from the kaggle output working directory to prevent thousands of output files.

# ###### Form a dataframe for each of our data splits

# In[ ]:


def create_dataframe(data_dir):
    """ Returns a dataframe consisting of img path and label, where
        0 is normal and 1 is pneumonia """
    data = []
    labels = []
    
    # obtain image paths for all training data
    normal_dir = os.path.join(data_dir, 'NORMAL')
    pneunomia_dir = os.path.join(data_dir, 'PNEUMONIA')
    normal_data = [x for x in os.listdir(normal_dir) if x.endswith('.jpeg')]
    pneunomia_data = [x for x in os.listdir(pneunomia_dir) if x.endswith('.jpeg')]
    
    # append img path and labels for each
    for normal in normal_data:
        data.append(os.path.join(normal_dir, normal))
        labels.append(0) 
    for pneumonia in pneunomia_data:
        data.append(os.path.join(pneunomia_dir, pneunomia_dir))
        labels.append(1)
        
    # return pandas dataframe
    return pd.DataFrame({'Image_path' : data, 'Label' : labels})


# In[ ]:


train_df = create_dataframe(train_dir)
val_df = create_dataframe(validation_dir)
test_df = create_dataframe(test_dir)


# In[ ]:


train_df['Label'].value_counts().plot.bar()
plt.show()


# To compensate for the large imbalance, we'll do the simplest option - oversample the under-represented class.

# In[ ]:


def duplicate_data(file_dir, suffix='.jpeg'):
    """ duplicate img data within destination directory """
    
    img_files = [x for x in os.listdir(file_dir) if x.endswith(suffix)]
    
    for img in img_files:
        src_path = os.path.join(file_dir, img)
        dup_img = f"{img[:-len(suffix)]}_2{suffix}"
        dest_path = os.path.join(file_dir, dup_img)
        shutil.copyfile(src_path, dest_path)


# In[ ]:


duplicate_data(os.path.join(train_dir, 'NORMAL'))


# In[ ]:


train_df = create_dataframe(train_dir)
train_df['Label'].value_counts().plot.bar()
plt.show()


# Much better - although we've only duplicated our normal data, it helps prevent our model being over-fitted to the dominant class during training.

# #### Lets visualise some normal and pneunomia examples from our data

# In[ ]:


fig = plt.figure(figsize=(12, 6))

for i, example in enumerate(img_paths['train_pneumonia'][:5]):
    
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # read image and plot
    example_img = tf.io.read_file(example)
    example_img = tf.image.decode_jpeg(example_img, channels=3)
    example_img = tf.image.resize(example_img, [360, 320])
    example_img /= 255.0
    ax.imshow(example_img)
    ax.set_title(f"Pneumonia {i}")
    
for i, example in enumerate(img_paths['train_normal'][:5]):
    
    ax = fig.add_subplot(2, 5, i+6)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # read image and plot
    example_img = tf.io.read_file(example)
    example_img = tf.image.decode_jpeg(example_img, channels=3)
    example_img = tf.image.resize(example_img, [360, 320])
    example_img /= 255.0
    ax.imshow(example_img)
    ax.set_title(f"Normal {i}")


# #### Data preprocessing and augmentation of our images using Keras ImageDataGenerator
# 
# To start, we'll take the built-in Keras ImageDataGenerator to help carry out data augmentation during training.

# In[ ]:


img_height, img_width = 150, 150
batch_size = 10

# training data augmentation - rotate, shear, zoom and flip
train_datagen = ImageDataGenerator(
    rotation_range = 30,
    rescale = 1.0 / 255.0,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip=True)

# no augmentation for test data - only rescale
test_datagen = ImageDataGenerator(rescale = 1. / 255.0)

# generate batches of augmented data from training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# generate val data from val dir
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

nb_train_samples = len(train_generator.classes)
nb_validation_samples = len(validation_generator.classes)


# In[ ]:


# get class labels dict containing index of each class for decoding predictions
class_labels = train_generator.class_indices

class_labels


# ---
# 
# ## 2. Model Training - Basic Convolutional Neural Network
# 
# **Model features:**
# - Four Conv Layers with max pooling
# - Flatten layer with 50% dropout
# - Dense layer with 512 hidden units and relu activation
# - Output sigmoid
# - RMSPROP optimiser with binary cross-entropy loss
# 
# **Results obtained:**
# - Accuracy of 91.51%
# - F1 Score: 0.93
# - Precision: 0.92 weighted average, 0.95 normal, 0.90 pneumonia
# - Recall: 0.92 weighted average, 0.81 normal, 0.98 pneumonia
# 
# **Brief overview of process:**
# 
# We need to take our dataset of jpeg images and preprocess them accordingly prior to use in our deep learning model. In summary, we need to perform the following:
# 
# - Read in each of our images as a jpeg file
# - Decode each image into floating-point tensor form, with RGB grids of pixels for each image
# - Standardise our images through rescaling of the pixel values.
# 
# These functions are performed automatically using the data generators we created previously. For our first convolutional neural network, we'll form a custom smaller sized ConvNet.

# ###### Train model - only conduct once, and import model thereafter (long training time) 

# In[ ]:


def create_CNN(input_size=(150, 150)):
    """ Basic CNN with 4 Conv layers, each followed by a max pooling """
    cnn_model = Sequential()
    
    # four Conv layers with max pooling
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    
    # flatten output and feed to dense layer, via dropout layer
    cnn_model.add(Flatten())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(512, activation='relu'))
    
    # add output layer - sigmoid since we only have 2 outputs
    cnn_model.add(Dense(1, activation='sigmoid'))
    
    cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return cnn_model


# In[ ]:


CNN_model = create_CNN()
CNN_model.summary()


# In[ ]:


# set up a check point for our model - save only the best val performance
save_path ="basic_cnn_best_weights.hdf5"

trg_checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', 
                                 verbose=1, save_best_only=True, mode='max')

trg_callbacks = [trg_checkpoint]


# In[ ]:


# batch steps before an epoch is considered complete (trg_size / batch_size):
steps_per_epoch = np.ceil(nb_train_samples/batch_size)

# validation batch steps (val_size / batch_size):
val_steps_per_epoch = np.ceil(nb_validation_samples/batch_size)


# In[ ]:


history = CNN_model.fit(train_generator, epochs=50, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_data=validation_generator, 
                        validation_steps=val_steps_per_epoch,
                        callbacks=trg_callbacks,
                        shuffle=True)


# In[ ]:


# save model as a HDF5 file with weights + architecture
CNN_model.save('basic_cnn_model_1.hdf5')

# save the history of training to a datafile for later retrieval
with open('history_basic_cnn_model_1.pickle', 
          'wb') as pickle_file:
    pickle.dump(history.history, pickle_file)
    
loaded_model = False


# ###### Load model with best weights from training conducted above (assuming the above steps have already been conducted once)

# In[ ]:


# if already trained - import history file and best training weights
CNN_model = load_model('basic_cnn_best_weights.hdf5')


# In[ ]:


# if already trained - import history file and training weights
#CNN_model = load_model('models/basic_cnn_model_1.hdf5')

# get history of trained model
#with open('models/history_basic_cnn_model_1.pickle', 'rb') as handle:
#    history = pickle.load(handle)
    
#loaded_model = True


# In[ ]:


# if loaded model set history accordingly
if loaded_model:
    trg_hist = history
else:
    trg_hist = history.history

trg_loss = trg_hist['loss']
val_loss = trg_hist['val_loss']

trg_acc = trg_hist['accuracy']
val_acc = trg_hist['val_accuracy']

epochs = range(1, len(trg_acc) + 1)

# plot losses and accuracies for training and validation 
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 2, 1)
plt.plot(epochs, trg_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='x', label='Validation Loss')
plt.title("Training / Validation Loss")
ax.set_ylabel("Loss")
ax.set_xlabel("Epochs")
plt.legend(loc='best')

ax = fig.add_subplot(1, 2, 2)
plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')
plt.title("Training / Validation Accuracy")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[ ]:


test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), 
                                                  batch_size=4, class_mode='binary')

test_loss, test_accuracy = CNN_model.evaluate_generator(test_generator)
print(f"Test accuracy: {test_accuracy}")


# When compared to the initial results obtained using the default data, no data augmentation, no regularisation (using dropout), our model is significantly better. This highlights the increases in performance that can be obtained through over-sampling of under-represented classes and performing image data-augmentation when we have a dataset with imbalanced classes or a low number of samples.

# ### Limitations with accuracy as a metric
# 
# As shown, our model weights that achieved a peak validation score of around 96% also score a test accuracy of 91.5%. This is good, however accuracy is actually a poor metric in this case. Due to the great imbalance in data, achieving a high scoring accuracy on this dataset is not difficult if we simply predict pneumonia everytime.
# 
# For prediction of pneumonia, we are much more interested in positively identifying a pneumonia case in order to help patients with the appropriate care. This means we should maximise our True Positive Rate (TPR) as much as possible, and therefore our recall is extremely important in this case:
# 
# $ Recall = TPR = \displaystyle\frac{TP}{FN + TP} $
# 
# Where TP and FN are True Positives and False Negatives respectively.
# 
# Recall isn't the only important metric though - precision is also important to reduce inadvertent classifications of patients as having pneumonia when in fact they do not:
# 
# $ Precision = \displaystyle\frac{TP}{FP + TP} $
# 
# Where FP is the number of False Positives.
# 
# Since both of these metrics are important for our model, we can combine both through calculating the F1 score. This is obtained using a combination of Precision and Recall, like so:
# 
# $ F1 = 2 \times \displaystyle\frac{Precision \times Recall}{Precision + Recall} $

# In[ ]:


# generate val data from val dir
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)


# In[ ]:


y_labels = np.expand_dims(test_generator.classes, axis=1)
nb_test_samples = len(y_labels)

y_preds = CNN_model.predict(test_generator, 
                            steps=np.ceil(nb_test_samples/batch_size))

# round predictions to 0 (Normal) or 1 (Pneumonia)
y_preds = np.rint(y_preds)


# Let's find the basic performance - accuracy, F1 score and number of samples incorrectly classified:

# In[ ]:


# number of incorrect labels
incorrect = (y_labels[:, 0] != y_preds[:, 0]).sum()

# print the basic results of the model
print(f"Accuracy: {accuracy_score(y_labels[:, 0], y_preds[:, 0])*100:.2f}%")
print(f"F1 Score: {f1_score(y_labels[:, 0], y_preds[:, 0]):.2f}")
print(f"Samples incorrectly classified: {incorrect} out of {len(y_labels)}")


# Classification report of performance, including precision, recall, and F1 scores across classes:

# In[ ]:


# print recall, precision and f1 score results
print(classification_report(y_labels[:, 0], y_preds[:, 0]))


# We can also visually represent these results using a **confusion matrix**:

# In[ ]:


def plot_confusion_matrix(true_y, pred_y, title='Confusion Matrix', figsize=(8,6)):
    """ Custom function for plotting a confusion matrix for predicted results """
    conf_matrix = confusion_matrix(true_y, pred_y)
    conf_df = pd.DataFrame(conf_matrix, columns=np.unique(true_y), index = np.unique(true_y))
    conf_df.index.name = 'Actual'
    conf_df.columns.name = 'Predicted'
    plt.figure(figsize = figsize)
    plt.title(title)
    sns.set(font_scale=1.4)
    sns.heatmap(conf_df, cmap="Blues", annot=True, 
                annot_kws={"size": 16}, fmt='g')
    plt.show()
    return


# plot a confusion matrix of our results
plot_confusion_matrix(y_labels[:, 0], y_preds[:, 0], title="Basic ConvNet Confusion Matrix")


# ---
# 
# ## 3. Making predictions on chosen invididual test set images
# 
# Above we made predictions on the entire test dataset and evaluated our performance. 
# 
# However, to actually use a model such as this, we need to be able to easily make predictions on individual chosen images. This is effectively what we want to achieve by forming a model such as this for classifying pneumonia. 
# 
# In the next few samples of code we'll make some helper functions to do precisely this.

# In[ ]:


# get class labels dict containing index of each class for decoding predictions
class_labels = train_generator.class_indices

# obtain a reverse dict to convert index into class labels
reverse_class_index = {i : class_label for class_label, i in class_labels.items()}


# In[ ]:


class_labels


# In[ ]:


def process_and_predict_img(image_path, model, img_size=(150, 150)):
    """ Utility function for making predictions for an image. """
    img_path = image_path
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = test_datagen.standardize(x)
    predictions = model.predict(x)
    return img, predictions


# In[ ]:


img, prediction = process_and_predict_img(img_paths['test_normal'][0], 
                                          model=CNN_model)
plt.imshow(img)
plt.title(f"Prediction: {reverse_class_index[np.argmax(prediction)]}\n")
plt.show()


# ---
# 
# ## 4. Further experimentation - Training our previous CNN with a custom F1 Metric
# 
# Keras, by default, does not support the optimisation of F1 score directly during training like it does with accruracy. We'll compensate for this by producing a set of custom F1 Score and F1 loss functions to perform this, and evaluate the difference in performance between the previous model and this newly optimised model.
# 
# One thing to be careful with in this case is that Keras works in batches for calculating metrics, which in the case of precision, recall and F1 score can lead to inconsistent and misleading end-products. Never the less, we'll proceed and create a custom F1 score and loss function, and evaluate how well (or not) our model improves compared to previously.

# In[ ]:


def f1_score(y_true, y_pred):
    """ Find and return the F1 Score """
    y_pred = keras.round(y_pred)
    
    # calculate true pos, true neg, false pos, false neg
    true_pos = keras.sum(keras.cast(y_true*y_pred, 'float'), axis=0)
    true_neg = keras.sum(keras.cast((1 - y_true)*(1 - y_pred), 'float'), axis=0)
    false_pos = keras.sum(keras.cast((1- y_true)*y_pred, 'float'), axis=0)
    false_neg = keras.sum(keras.cast(y_true*(1 - y_pred), 'float'), axis=0)

    # calculate precision / recall, adding epsilon to prevent zero div error(s)
    precision = true_pos / (true_pos + false_pos + keras.epsilon())
    recall = true_pos / (true_pos + false_neg + keras.epsilon())

    # calculate f1 score and return
    f1_score = (2.0 * precision * recall) / (precision + recall + keras.epsilon())
    f1_score = tf.where(tf.math.is_nan(f1_score), tf.zeros_like(f1_score), f1_score)
    return keras.mean(f1_score)


def f1_loss(y_true, y_pred):
    """ Calculate mean F1 and return minimising function to approximate a loss equivalent. """
    return 1 - f1_score(y_true, y_pred)


# All we need to do now is repeat the same training performed previously for the basic CNN, but this time include F1 score in the metrics, and use our custom F1 loss function above as the selected loss.
# 
# Due to these changes that have taken place, let's redefine our CNN creation function:

# In[ ]:


def basic_CNN_2(input_size=(150, 150)):
    """ Basic CNN with 4 Conv and max pooling layers, with custom F1 loss """
    cnn_model = Sequential()
    
    # four Conv layers with max pooling
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(2, 2))
    
    # flatten output and feed to dense layer, via dropout layer
    cnn_model.add(Flatten())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(512, activation='relu'))
    
    # add output layer - sigmoid since we only have 2 outputs
    cnn_model.add(Dense(1, activation='sigmoid'))
    
    cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', f1_score])
    
    return cnn_model


# In[ ]:


CNN_model_2 = basic_CNN_2()
CNN_model_2.summary()


# In[ ]:


# set up a check point for our model - save only the best val performance
save_path ="basic_cnn_2_best_weights.hdf5"

trg_checkpoint = ModelCheckpoint(save_path, monitor='val_f1_score', 
                                 verbose=1, save_best_only=True, mode='max')

trg_callbacks = [trg_checkpoint]

# batch steps before an epoch is considered complete (trg_size / batch_size):
steps_per_epoch = np.ceil(nb_train_samples/batch_size)
val_steps_per_epoch = np.ceil(nb_validation_samples/batch_size)


# In[ ]:


history = CNN_model_2.fit(train_generator, epochs=50, 
                          steps_per_epoch=steps_per_epoch, 
                          validation_data=validation_generator, 
                          validation_steps=val_steps_per_epoch,
                          callbacks=trg_callbacks,
                          shuffle=True)


# In[ ]:


# save model as a HDF5 file with weights + architecture
CNN_model_2.save('basic_cnn_model_2.hdf5')

# save the history of training to a datafile for later retrieval
with open('history_basic_cnn_model_2.pickle', 
          'wb') as pickle_file:
    pickle.dump(history.history, pickle_file)
    
loaded_model = False


# ###### Load best weights model (if the above steps have already been conducted once)
# 
# 

# In[ ]:


# if already trained - import history file and best training weights
CNN_model_2 = load_model('basic_cnn_2_best_weights.hdf5', custom_objects={'f1_score' : f1_score})


# In[ ]:


# get history of trained model
#with open('history_basic_cnn_model_2.pickle', 'rb') as handle:
#    history = pickle.load(handle)
    
#loaded_model = True


# ###### Plot model training results
# 
# Note - these results could be improved by importing the best weights found during training (as saved above) instead.

# In[ ]:


# if loaded model set history accordingly
if loaded_model:
    trg_hist = history
else:
    trg_hist = history.history

trg_loss = trg_hist['loss']
val_loss = trg_hist['val_loss']

trg_acc = trg_hist['accuracy']
val_acc = trg_hist['val_accuracy']

epochs = range(1, len(trg_acc) + 1)

trg_loss = trg_hist['loss']
val_loss = trg_hist['val_loss']

trg_acc = trg_hist['accuracy']
val_acc = trg_hist['val_accuracy']

trg_f1 = trg_hist['f1_score']
val_f1 = trg_hist['val_f1_score']

epochs = range(1, len(trg_acc) + 1)

# plot losses and accuracies for training and validation 
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(epochs, trg_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='x', label='Validation Loss')
plt.title("Training / Validation Loss")
ax.set_ylabel("Loss")
ax.set_xlabel("Epochs")
plt.legend(loc='best')

ax = fig.add_subplot(1, 2, 2)
plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')
plt.title("Training / Validation Accuracy")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epochs")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# plot F1 scores
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(epochs, trg_acc, marker='o', label='Training F1')
plt.plot(epochs, val_acc, marker='^', label='Validation F1')
plt.title("Training / Validation F1 Score")
ax.set_ylabel("F1 Score")
ax.set_xlabel("Epochs")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ###### Evaluate results
# 
# Note - could improve on these results through use of the best weights saved during training (above). We can simply import this 'best' model and use that instead.

# In[ ]:


# generate val data from val dir
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

y_labels = np.expand_dims(test_generator.classes, axis=1)
nb_test_samples = len(y_labels)


# In[ ]:


y_preds = CNN_model_2.predict(test_generator, 
                              steps=np.ceil(nb_test_samples/batch_size))

# round predictions to 0 (Normal) or 1 (Pneumonia)
y_preds = np.rint(y_preds)


# In[ ]:


# number of incorrect labels
incorrect = (y_labels[:, 0] != y_preds[:, 0]).sum()

# print the basic results of the model
print(f"Accuracy: {accuracy_score(y_labels[:, 0], y_preds[:, 0])*100:.2f}%")
print(f"F1 Score: {f1_score(y_labels[:, 0], y_preds[:, 0]):.2f}")
print(f"Samples incorrectly classified: {incorrect} out of {len(y_labels)}")


# In[ ]:


# print recall, precision and f1 score results
print(classification_report(y_labels[:, 0], y_preds[:, 0]))


# In[ ]:


def plot_confusion_matrix(true_y, pred_y, title='Confusion Matrix', figsize=(8,6)):
    """ Custom function for plotting a confusion matrix for predicted results """
    conf_matrix = confusion_matrix(true_y, pred_y)
    conf_df = pd.DataFrame(conf_matrix, columns=np.unique(true_y), index = np.unique(true_y))
    conf_df.index.name = 'Actual'
    conf_df.columns.name = 'Predicted'
    plt.figure(figsize = figsize)
    plt.title(title)
    sns.set(font_scale=1.4)
    sns.heatmap(conf_df, cmap="Blues", annot=True, 
                annot_kws={"size": 16}, fmt='g')
    plt.show()
    return


# plot a confusion matrix of our results
plot_confusion_matrix(y_labels[:, 0], y_preds[:, 0], title="ConvNet (F1 loss) Confusion Matrix")


# ## Finally - remove dataset from kaggle output dir (prevents thousands of images being output - uncomment if you want to keep the dataset formed)

# In[ ]:


try:
    shutil.rmtree(base_dir)
except OSError as e:
    print("Error: %s : %s" % (base_dir, e.strerror))

