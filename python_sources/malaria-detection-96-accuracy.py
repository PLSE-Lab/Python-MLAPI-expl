#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# since I can't do much about warnings from libraries, just ignore them!!
import warnings
warnings.filterwarnings('ignore')

import os, random
import numpy as np 
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
print('Using Tensorflow version: ', tf.__version__)
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2
print('Using OpenCV version: ', cv2.__version__)
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn')
sns.set_style('darkgrid')

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

float_formatter = lambda x: '%.4f' % x
np.set_printoptions(formatter={'float_kind':float_formatter})
np.set_printoptions(threshold=np.inf, suppress=True, precision=4, linewidth=110)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# all images are available in /kaggle/input/cell-images-for-detecting-malaria/cell_images/ folder
# this has 2 sub-folders holding "Parasitized" and "Uninfected"
images_root = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images"


# In[ ]:


# some globals used throught this workbook (hyper-parameters)
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS = 128, 128, 3
NUM_EPOCHS, BATCH_SIZE = 100, 32
ADAM_LR = 0.001


# ## Loading the images

# In[ ]:


def load_image(image_path):
    img = cv2.resize(cv2.imread(image_path),(IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img.clip(0, 255).astype('uint8')
    # convert from BGR colorspace to RGB, so other libraries can process easily
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

infected_images, infected_labels = [], []

print('Processing Infected images...', end='', flush=True)
infected_images_glob = glob.glob("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/*.png")
for i in infected_images_glob:
    img = load_image(i)
    infected_images.append(img)
    infected_labels.append(1)
    
infected_images = np.array(infected_images)
infected_labels = np.array(infected_labels)
print('\rInfected images: found %d images & %d labels' % (len(infected_images), len(infected_labels)))

uninfected_images, uninfected_labels = [], []

print('Processing Un-infected images...', end='', flush=True)
uninfected_images_glob = glob.glob("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/*.png")
for i in uninfected_images_glob:
    img = load_image(i)
    uninfected_images.append(img)
    uninfected_labels.append(0)
    
uninfected_images = np.array(uninfected_images)
uninfected_labels = np.array(uninfected_labels)
print('\rUn-infected images: found %d images & %d labels' % (len(uninfected_images), len(uninfected_labels)))

all_images = np.vstack([infected_images, uninfected_images])
all_labels = np.hstack([infected_labels, uninfected_labels])
print('Combining into all: we have %d images & %d labels' % (len(all_images), len(all_labels)))

# delete unwanted arrays to conserve memory, especially since we'll be using pre-trained (huge) models
del infected_images, infected_labels
del uninfected_images, uninfected_labels


# In[ ]:


# shuffle the data
indexes = np.random.permutation(np.arange(all_images.shape[0]))
#for _ in range(5): np.random.shuffle(indexes)

all_images = all_images[indexes]
all_labels = all_labels[indexes]
all_labels[:25]


# In[ ]:


# split into train/test sets (80:20)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =     train_test_split(all_images, all_labels, test_size=0.20, random_state=seed)
test_images, test_labels = X_test.copy(), y_test.copy()
# NOTE: we set aside a copy of the test set for visualization purposes
# we WILL NOT apply any further pre-processing steps to test_images/test_labels
X_train.shape, y_train.shape, X_test.shape, y_test.shape, test_images.shape, test_labels.shape


# In[ ]:


def display_sample(sample_images, sample_labels, sample_predictions=None, num_rows=5, num_cols=10,
                   plot_title=None, fig_size=None):
    """ display a random selection of images & corresponding labels, optionally with predictions
        The display is laid out in a grid of num_rows x num_col cells
        If sample_predictions are provided, then each cell's title displays the prediction 
        (if it matches actual) or actual/prediction if there is a mismatch
    """
    from PIL import Image
    import seaborn as sns
    assert sample_images.shape[0] == num_rows * num_cols

    # a dict to help encode/decode the labels
    LABELS = {
        0: 'Uninfected',
        1: 'Infected',
    }
    
    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=((14, 9) if fig_size is None else fig_size),
            gridspec_kw={"wspace": 0.02, "hspace": 0.25}, squeeze=True)
        #fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.93)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                pil_image = Image.fromarray(sample_images[image_index])
                ax[r, c].imshow(pil_image, cmap="Greys")

                if sample_predictions is None:
                    # show the actual labels in the cell title
                    title = ax[r, c].set_title("%s" % LABELS[sample_labels[image_index]])
                else:
                    # else check if prediction matches actual value
                    true_label = sample_labels[image_index]
                    pred_label = sample_predictions[image_index]
                    prediction_matches_true = (sample_labels[image_index] == sample_predictions[image_index])
                    if prediction_matches_true:
                        # if actual == prediction, cell title is prediction shown in green font
                        title = LABELS[true_label]
                        title_color = 'g'
                    else:
                        # if actual != prediction, cell title is actua/prediction in red font
                        title = '%s/%s' % (LABELS[true_label], LABELS[pred_label])
                        title_color = 'r'
                    # display cell title
                    title = ax[r, c].set_title(title)
                    plt.setp(title, color=title_color)
        # set plot title, if one specified
        if plot_title is not None:
            f.suptitle(plot_title)

        plt.show()
        plt.close()


# In[ ]:


# display a random sample of 64 images from test_images/test_labels set
sample_size = 64
rand_indexes = np.random.choice(np.arange(len(test_images)), sample_size)
sample_images = test_images[rand_indexes]
sample_labels = test_labels[rand_indexes]
#sample_images.shape, sample_labels.shape, len(sample_images), len(sample_labels)
display_sample(sample_images, sample_labels, plot_title="Random sample of %d images" % sample_size, 
               num_rows=8, num_cols=8, fig_size=(13,13))


# In[ ]:


# normalize images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# ### Helper Function(s)

# In[ ]:


def show_plots(history, plot_title=None, fig_size=None):
    
    import seaborn as sns
    
    """ Useful function to view plot of loss values & accuracies across the various epochs
        Works with the history object returned by the train_model(...) call """
    assert type(history) is dict

    # NOTE: the history object should always have loss & acc (for training data), but MAY have
    # val_loss & val_acc for validation data
    loss_vals = history['loss']
    val_loss_vals = history['val_loss'] if 'val_loss' in history.keys() else None
    
    # accuracy is an optional metric chosen by user
    # NOTE: in Tensorflow 2.0, the keys are 'accuracy' and 'val_accuracy'!! Why Google why??
    acc_vals = history['acc'] if 'acc' in history.keys() else None
    if acc_vals is None:
        # try 'accuracy' key, could be using Tensorflow 2.0 backend!
        acc_vals = history['accuracy'] if 'acc' in history.keys() else None
        
    val_acc_vals = history['val_acc'] if 'val_acc' in history.keys() else None
    if val_acc_vals is None:
        # try 'val_accuracy' key, could be using Tensorflow 2.0 backend!
        val_acc_vals = history['val_accuracy'] if 'val_accuracy' in history.keys() else None       
        
    epochs = range(1, len(history['loss']) + 1)
    
    col_count = 1 if ((acc_vals is None) and (val_acc_vals is None)) else 2
    
    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(nrows=1, ncols=col_count, figsize=((16, 5.5) if fig_size is None else fig_size))
    
        # plot losses on ax[0]
        #ax[0].plot(epochs, loss_vals, color='navy', marker='o', linestyle=' ', label='Training Loss')
        ax[0].plot(epochs, loss_vals, label='Training Loss')
        if val_loss_vals is not None:
            #ax[0].plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Validation Loss')
            ax[0].plot(epochs, val_loss_vals, label='Validation Loss')
            ax[0].set_title('Training & Validation Loss')
            ax[0].legend(loc='best')
        else:
            ax[0].set_title('Training Loss')
    
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].grid(True)
    
        # plot accuracies, if exist
        if col_count == 2:
            #acc_vals = history['acc']
            #val_acc_vals = history['val_acc'] if 'val_acc' in history.keys() else None

            #ax[1].plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Training Accuracy')
            ax[1].plot(epochs, acc_vals, label='Training Accuracy')
            if val_acc_vals is not None:
                #ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validation Accuracy')
                ax[1].plot(epochs, val_acc_vals, label='Validation Accuracy')
                ax[1].set_title('Training & Validation Accuracy')
                ax[1].legend(loc='best')
            else:
                ax[1].set_title('Training Accuracy')

            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].grid(True)
    
        if plot_title is not None:
            plt.suptitle(plot_title)
    
        plt.show()
        plt.close()

    # delete locals from heap before exiting (to save some memory!)
    del loss_vals, epochs, acc_vals
    if val_loss_vals is not None:
        del val_loss_vals
    if val_acc_vals is not None:
        del val_acc_vals


# In[ ]:


def save_keras_model(model, base_file_name, save_dir=os.path.join('.', 'model_states')):
    """ save keras model graph + weights to one HDF5 file """
    # check if save_dir exists, else create it
    if not os.path.exists(save_dir):
        try:
            os.mkdir(save_dir)
        except OSError as err:
            print("Unable to create folder {} to save Keras model. Can't continue!".format(save_dir))
            raise err
    
    # save the model
    if not base_file_name.lower().endswith('.h5'):
        base_file_name = base_file_name + '.h5'
        
    model_save_path = os.path.join(save_dir, base_file_name)
    model.save(model_save_path)
    print('Saved model to file %s' % model_save_path)


# In[ ]:


def load_keras_model(base_file_name, save_dir=os.path.join('.', 'model_states'), 
                     use_tf_keras_impl=True):            
    """load keras model graph + weights from HDF5 file"""
    if not base_file_name.lower().endswith('.h5'):
        base_file_name = base_file_name + '.h5'
        
    model_save_path = os.path.join(save_dir, base_file_name)
    if not os.path.exists(model_save_path):
        raise IOError('Cannot find model state file at %s!' % model_save_path)
        
    # load the state/weights etc.
    if use_tf_keras_impl:
        from tensorflow.keras.models import load_model 
    else:
        from keras.models import load_model

    # load the state/weights etc. from .h5 file        
    model = load_model(model_save_path)
    print('Loaded Keras model from %s' % model_save_path)
    return model


# In[ ]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    
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
        plt.text(j, i, cm[i, j], horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Training the model
# We will use the **pre-trained VGG16** model. As a first step, we use the Conv2d layers of this model and freeze it, so that weights are not updated.

# In[ ]:


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K

K.clear_session()

# NOTE: will download the weights for imagenet
vgg16_base = VGG16(
    weights='imagenet',    # use weights for ImageNet
    include_top=False,     # don't use upper Dense layers
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
print(vgg16_base.summary())


# In[ ]:


# build our model above the VGG16 pre-trained model
def build_model_xfer(use_l2_loss=True):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    
    l2_loss_lambda = 0.00002  # just a wee-bit :)
    l2_loss = l2(l2_loss_lambda) if use_l2_loss else None
    if l2_loss is not None: print('Using l2_loss_lambda = %f' % l2_loss_lambda)
        
    model = tf.keras.models.Sequential([
        # our vgg16_base model added as a layer
        vgg16_base,
        # here is our custom prediction layer (same as before)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2_loss),
        tf.keras.layers.Dropout(0.20),        
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_loss),
        tf.keras.layers.Dropout(0.10),         
        tf.keras.layers.Dense(1, activation='sigmoid')    
    ])
    
    # mark mobilenet layer as non-trainable, so training updates
    # weights and biases of just our newly added layers
    vgg16_base.trainable = False
    
    model.compile(optimizer=Adam(lr=0.001), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# In[ ]:


model = build_model_xfer()
print(model.summary())


# Notice that the entire `14,714,688` parameters of the `vgg16_base` layer are marked as _Non-trainable_ parameters!

# In[ ]:


# train the model
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_step(epoch):
    power = (epoch % 20) # [0-19] = 0, [20-39] = 1, [40-59] = 2...
    new_lr = ADAM_LR_START * 1.0 / (1 + ADAM_LR_DECAY * power)
    return new_lr

lr_callback = LearningRateScheduler(lr_step)

hist = model.fit(X_train, y_train, epochs=50, batch_size=BATCH_SIZE, validation_split=0.20)
                 #callbacks=[lr_callback])


# In[ ]:


show_plots(hist.history, plot_title='Malaria Detection - VGG16 Base Model')


# In[ ]:


# evaluate performance on train & test data
loss, acc = model.evaluate(X_train, y_train, batch_size=64, verbose=1)
print('Training data  -> loss: %.3f, acc: %.3f' % (loss, acc))
loss, acc = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
print('Testing data   -> loss: %.3f, acc: %.3f' % (loss, acc))


# In[ ]:


save_keras_model(model, "kr_malaria_vgg16")
del model


# In[ ]:


model = load_keras_model("kr_malaria_vgg16")
print(model.summary())


# In[ ]:


# run predictions
y_pred = (model.predict(X_test, batch_size=BATCH_SIZE) >= 0.5).ravel().astype('int32')
print('Actuals    : ', y_test[:30])
print('Predictions: ', y_pred[:30])
print('We got %d of %d wrong!' % ((y_pred != y_test).sum(), len(y_test)))


# In[ ]:


# let's see the confusion matrix & classification report
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=["Uninfected","Parasitized"])


# **Observations:**
# 
# >Configuration | Training Acc | Test Acc | Incorrect Predictions | Precision | Recall | F1-Score
# >:---|:---:|:---:|:---:|:---:|:---:|:---:|
# >**VGG16 Base Model**|96-97%|94-95%|315|0.95|0.93|0.94
# 
# * From the loss and accuracy curves, we see that the cross-validation loss & accuracy follows the training loss & accuracy curves
# * From the accuracy metrics, we can conclude that the model is overfitting the data slightly (2% difference between training & test accuracies).
# * Though the precision & recall metrics are good, we'd like to reduce False Positives (i.e. predicting as Parasitized when actually not).

# In[ ]:


# display a random sample of 64 images from test_images/test_labels set with predictions
sample_size = 64
rand_indexes = np.random.choice(np.arange(len(test_images)), sample_size)
rand_images = test_images[rand_indexes]
rand_labels = test_labels[rand_indexes]
rand_predictions = y_pred[rand_indexes]
#sample_images.shape, sample_labels.shape, len(sample_images), len(sample_labels)
display_sample(rand_images, rand_labels, sample_predictions=rand_predictions,
               plot_title='Malaria Detection - VGG16 Base Model - Sample Predictions for %d images' % sample_size, 
               num_rows=8, num_cols=8, fig_size=(16,16))


# * The cells with red titles ('red cells') are the once displaying incorrect predictions
# * For such 'red cells' the title is displayed as 'correct_value'/'prediction'
# * For all other cells, with green titles - these are cells displaying correct predictions

# In[ ]:


# cleanup to save space
del vgg16_base
del model


# ## Fune-tuning the VGG16 Model
# In ths section, we will try and further improve performance of the model by finetuning some Conv2D layers of the pre-trained VGG16 model.

# In[ ]:


try:
    del vgg16_base_ft
except NameError:
    pass # model is not defined!    

try:
    del model
except NameError:
    pass # model is not defined!


# In[ ]:


# here is the pre-trained model again
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K

K.clear_session()

# NOTE: will download the weights for imagenet
vgg16_base_ft = VGG16(
    weights='imagenet',    # use weights for ImageNet
    include_top=False,     # don't use upper Dense layers
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
print(vgg16_base_ft.summary())


# ![](http://)Let's **unfreeze** layers `block_conv3` and `block5_pool` and keep all others frozen. Consequently, these layers will also have their weights updated as the model trains across epochs.

# In[ ]:


trainable = False

# this will iterate over the layers in sequence from bottom to top
# i.e. from input_1 to block5_pool
for layer in vgg16_base_ft.layers:
    if layer.name == 'block5_conv3':
        trainable = True
    layer.trainable = trainable
    
# let's see what that just did
for layer in vgg16_base_ft.layers:
    print('%s -> trainable? %s' % (layer.name, "Yes" if layer.trainable else "No"))


# **Cool!** So just  the last 2 layers of `vgg16_base` are now trainable. The `block5_pool` (MaxPooling2D) layer carries no weights, but the 2 Conv2D layers do!

# In[ ]:


print(vgg16_base_ft.summary())


# Notice that some of the weights (parameters) are now marked as trainable - out of `14,714,688` parameters, `4,719,616` are now trainable (these are the # parameters in the `block5_conv2` + `block5_conv3` layer)
# 
# Now let's build our model again, using `vgg16_base` layer with 2 of it's `Conv2D` layers _unlocked_. The code below is exactly the same as before, with one exception - we have commented out the `vgg16_base_ft.trainable = False` line.

# In[ ]:


# now let's build our model
def build_model_xfer_ft(use_l2_loss=True):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    
    l2_loss_lambda = 0.00010  # just a "pinch" of L2 regularization :)
    l2_loss = l2(l2_loss_lambda) if use_l2_loss else None
    if l2_loss is not None: print('Using l2_loss_lambda = %f' % l2_loss_lambda)
        
    model = tf.keras.models.Sequential([
        # our vgg16_base model added as a layer
        vgg16_base_ft,
        # here is our custom prediction layer (same as before)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2_loss),
        tf.keras.layers.Dropout(0.30),        
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_loss),
        tf.keras.layers.Dropout(0.20),     
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2_loss),
        tf.keras.layers.Dropout(0.10),           
        tf.keras.layers.Dense(1, activation='sigmoid')    
    ])
    
    # NOTE: we do not freeze vgg_base entirely as done previously!!
    
    model.compile(optimizer=Adam(lr=0.001), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# In[ ]:


model = build_model_xfer_ft()
print(model.summary())


# In[ ]:


# train model with pre-trained VGG16 layer
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

def lr_step(epoch):
    epoch_step = epoch // 20  # [0-19] == 0, [20-39] == 1, [40-59] == 2...
    divisor = 10 ** (epoch_step)
    new_lr = ADAM_LR / divisor
    return new_lr

lr_scheduler = LearningRateScheduler(lr_step)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, mode='auto', patience=5, verbose=1)

# NOTE: since there are many more parameters to update, I am training for 75 epochs
hist = model.fit(X_train, y_train, epochs=50, batch_size=BATCH_SIZE, validation_split=0.20)
                 #callbacks=[lr_scheduler])


# In[ ]:


show_plots(hist.history, plot_title='Malaria Detection - VGG16 Fine-tuned Model')


# In[ ]:


# evaluate performance on train & test data
loss, acc = model.evaluate(X_train, y_train, batch_size=64, verbose=1)
print('Training data  -> loss: %.3f, acc: %.3f' % (loss, acc))
loss, acc = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
print('Testing data   -> loss: %.3f, acc: %.3f' % (loss, acc))


# In[ ]:


save_keras_model(model, "kr_malaria_vgg16_ft-1")
del model


# In[ ]:


model = load_keras_model("kr_malaria_vgg16_ft-1")
print(model.summary())


# In[ ]:


# run predictions
y_pred = (model.predict(X_test, batch_size=BATCH_SIZE) >= 0.5).ravel().astype('int32')
print('Actuals    : ', y_test[:30])
print('Predictions: ', y_pred[:30])
print('We got %d of %d wrong!' % ((y_pred != y_test).sum(), len(y_test)))


# In[ ]:


# let's see the confusion matrix & classification report
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=["Uninfected","Parasitized"])


# **Observations:**
# 
# >Configuration | Training Acc | Test Acc | Incorrect Predictions | Precision | Recall | F1-Score
# >:---|:---:|:---:|:---:|:---:|:---:|:---:|
# >**VGG16 Base Model**|96-97%|94-95%|322|0.95|0.93|0.94
# >**VGG16 Fine Tuned**|98-99%|95-96%|276|0.96|0.94|0.95
# 
# * Though the metrics look good, this is still an overfitting model. We should look at increasing the regularization.
# * Notice that the incidence of False Positives (predicting +ve, when actual uninfected) has come down to 106 from 132 and false negatives (predicting not infected, when infected) have also come down to 170 from 183. This is a better model than before.

# In[ ]:


# display a random sample of 64 images from test_images/test_labels set with predictions
sample_size = 64
rand_indexes = np.random.choice(np.arange(len(test_images)), sample_size)
rand_images = test_images[rand_indexes]
rand_labels = test_labels[rand_indexes]
rand_predictions = y_pred[rand_indexes]
#sample_images.shape, sample_labels.shape, len(sample_images), len(sample_labels)
display_sample(rand_images, rand_labels, sample_predictions=rand_predictions,
               plot_title='Malaria Detection - VGG16 Fine-tuned Model - Sample Predictions for %d images' % sample_size, 
               num_rows=8, num_cols=8, fig_size=(16,16))


# In[ ]:


# cleanup
del vgg16_base_ft
del model


# In[ ]:




