#!/usr/bin/env python
# coding: utf-8

# **Deep Learning in Skin Lesion Analysis Towards Melanoma Detection** - Skin Lesion Classification

# In[ ]:


PLEASE ASSIST IN DEBUGGING THIS CODE. THANK YOU


# **COMMON PARAMETERS**

# In[ ]:


from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, Reshape
from keras import backend as K
from keras.optimizers import Nadam, Adam, SGD
from keras.metrics import categorical_accuracy, binary_accuracy
#from keras_contrib.losses import jaccard

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import preprocess_input, decode_predictions

import tensorflow as tf
import platform
import tensorflow.python.client
import numpy as np
import pandas as pd
import os
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys, os 
sys.path.append(os.getcwd())
import cv2
import os.path

data_folder = '../input'
model_folder = '../input/models'
history_folder = '../input/history'
pred_result_folder_val = '../input/val_predict_results'
out_dist_pred_result_folder = '../input/out_dist_predict_results'
workers = os.cpu_count()
plt.rcParams['svg.fonttype'] = 'none'


# **Importing Training Data**

# In[ ]:


import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def load_isic_training_data(image_folder, ground_truth_file):
    df_ground_truth = pd.read_csv(ground_truth_file)
    # Category names
    known_category_names = list(df_ground_truth.columns.values[1:9])
    unknown_category_name = df_ground_truth.columns.values[9]
    
    # Add path and category columns
    df_ground_truth['path'] = df_ground_truth.apply(lambda row : os.path.join(image_folder, row['image']+'.jpg'), axis=1)
    df_ground_truth['category'] = np.argmax(np.array(df_ground_truth.iloc[:,1:10]), axis=1)
    return df_ground_truth, known_category_names, unknown_category_name

def load_isic_training_and_out_dist_data(isic_image_folder, ground_truth_file, out_dist_image_folder):
    """ISIC training data and Out-of-distribution data are combined"""
    df_ground_truth = pd.read_csv(ground_truth_file)
    # Category names
    known_category_names = list(df_ground_truth.columns.values[1:9])
    unknown_category_name = df_ground_truth.columns.values[9]
    
    # Add path and category columns
    df_ground_truth['path'] = df_ground_truth.apply(lambda row : os.path.join(isic_image_folder, row['image']+'.jpg'), axis=1)
    
    df_out_dist = get_dataframe_from_img_folder(out_dist_image_folder, has_path_col=True)
    for name in known_category_names:
        df_out_dist[name] = 0.0
    df_out_dist[unknown_category_name] = 1.0
    # Change the order of columns
    df_out_dist = df_out_dist[df_ground_truth.columns.values]

    df_combined = pd.concat([df_ground_truth, df_out_dist])
    df_combined['category'] = np.argmax(np.array(df_combined.iloc[:,1:10]), axis=1)

    category_names = known_category_names + [unknown_category_name]
    return df_combined, category_names

def train_validation_split(df):
    df_train, df_val = train_test_split(df, stratify=df['category'], test_size=0.2, random_state=1)
    return df_train, df_val

def compute_class_weight_dict(df):
    """Compute class weights for weighting the loss function on imbalanced data."""
    class_weights = class_weight.compute_class_weight('balanced', np.unique(df['category']), df['category'])
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict, class_weights

def get_dataframe_from_img_folder(img_folder, has_path_col=True):
    if has_path_col:
        return pd.DataFrame([[Path(x).stem, x] for x in sorted(Path(img_folder).glob('**/*.jpg'))], columns=['image', 'path'], dtype=np.str)
    else:
        return pd.DataFrame([Path(x).stem for x in sorted(Path(img_folder).glob('**/*.jpg'))], columns=['image'], dtype=np.str)


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_complexity_graph(csv_file, title=None, figsize=(14, 10), feature_extract_epochs=None,
                          loss_min=0, loss_max=2, epoch_min=None, epoch_max=90, accuracy_min=0, accuracy_max=1):
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=14)

    ax1.plot(df['loss'], label='Training Loss')
    ax1.plot(df['val_loss'], label='Validation Loss')
    ax1.set(title='Training and Validation Loss', xlabel='', ylabel='Loss')
    ax1.set_xlim([epoch_min, epoch_max])
    ax1.set_ylim([loss_min, loss_max])
    ax1.legend()

    ax2.plot(df['balanced_accuracy'], label='Training Accuracy')
    ax2.plot(df['val_balanced_accuracy'], label='Validation Accuracy')
    ax2.set(title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Balanced Accuracy')
    ax2.set_xlim([epoch_min, epoch_max])
    ax2.set_ylim([accuracy_min, accuracy_max])
    ax2.legend()

    if feature_extract_epochs is not None:
        ax1.axvline(feature_extract_epochs-1, color='green', label='Start Fine Tuning')
        ax2.axvline(feature_extract_epochs-1, color='green', label='Start Fine Tuning')
        ax1.legend()
        ax2.legend()
    
    # tight_layout() only considers ticklabels, axis labels, and titles. Thus, other artists may be clipped and also may overlap.
    # [left, bottom, right, top]
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig

def plot_grouped_2bars(scalars, scalarlabels, xticklabels, title=None, xlabel=None, ylabel=None):
    x = np.arange(len(xticklabels))  # the label locations
    width = 0.35  # the width of the bars

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    fig.patch.set_facecolor('white')
    rects1 = ax.bar(x - width/2, scalars[0], width, label=scalarlabels[0])
    rects2 = ax.bar(x + width/2, scalars[1], width, label=scalarlabels[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()

def autolabel(ax, rects):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    # References
        https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, figsize=(8, 6)):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    # References
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set(title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    im, cbar = heatmap(cm, classes, classes, ax=ax, cmap=plt.cm.Blues, cbarlabel='', grid=False)
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    return fig


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", grid=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    # References
        https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    if grid:
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_prob_bars(img_title, img_path, labels, probs, topk=5, title=None, figsize=(10, 4)):
    fig, (ax1, ax2) = plt.subplots(figsize=figsize, ncols=2)
    fig.patch.set_facecolor('white')

    if title is not None:
        fig.suptitle(title)

    ax1.set_title(img_title)
    ax1.imshow(plt.imread(img_path))

    # Plot probabilities bar chart
    ax2.set_title("Top {0} probabilities".format(topk))
    ax2.barh(np.arange(topk), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(topk))
    ax2.set_yticklabels(labels, size='medium')
    ax2.yaxis.tick_right()
    ax2.set_xlim(0, 1.0)
    ax2.invert_yaxis()
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


# In[ ]:


from collections import Counter
#from data import load_isic_training_data
#from visuals import autolabel

training_image_folder = os.path.join(data_folder, 'ISIC_2019_Training_Input')
ground_truth_file = os.path.join(data_folder, '../input/isic-2019-training-groundtruth/ISIC_2019_Training_GroundTruth.csv')

df_ground_truth, known_category_names, unknown_category_name = load_isic_training_data(training_image_folder, ground_truth_file)
known_category_num = len(known_category_names)
print("Number of known categories: {}".format(known_category_num))
print(known_category_names, '\n')
unknown_category_num = 1
print("Number of unknown categories: {}".format(unknown_category_num))
print(unknown_category_name, '\n')
all_category_names = known_category_names + [unknown_category_name]
all_category_num = known_category_num + unknown_category_num

# mapping from category to index
print('Category to Index:')
category_to_index = dict((c, i) for i, c in enumerate(all_category_names))
print(category_to_index, '\n')

count_per_category = Counter(df_ground_truth['category'])
total_sample_count = sum(count_per_category.values())
print("Original training data has {} samples.".format(total_sample_count))
for i, c in enumerate(all_category_names):
    print("'%s':\t%d\t(%.2f%%)" % (c, count_per_category[i], count_per_category[i]*100/total_sample_count))
    
# Create a bar chart
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set(xlabel='Category', ylabel='Number of Images')
# plt.bar(count_per_category.keys(), count_per_category.values())
rects = plt.bar(all_category_names, [count_per_category[i] for i in range(all_category_num)])
autolabel(ax, rects)
fig.tight_layout()

df_ground_truth.head()


# **Shuffle and Split Training Data into Training and Validation Sets**

# In[ ]:


#from data import train_validation_split
#from visuals import plot_grouped_2bars

df_train, df_val = train_validation_split(df_ground_truth)

# Training Set
sample_count_train = df_train.shape[0]
print("Training set has {} samples.".format(sample_count_train))
count_per_category_train = Counter(df_train['category'])
for i, c in enumerate(all_category_names):
    print("'%s':\t%d\t(%.2f%%)" % (c, count_per_category_train[i], count_per_category_train[i]*100/sample_count_train))

# Validation Set
sample_count_val = df_val.shape[0]
print("\nValidation set has {} samples.".format(sample_count_val))
count_per_category_val = Counter(df_val['category'])
for i, c in enumerate(all_category_names):
    print("'%s':\t%d\t(%.2f%%)" % (c, count_per_category_val[i], count_per_category_val[i]*100/sample_count_val))

plot_grouped_2bars(
    scalars=[[count_per_category_train[i] for i in range(all_category_num)],
             [count_per_category_val[i] for i in range(all_category_num)]],
    scalarlabels=['Training', 'Validation'],
    xticklabels=all_category_names,
    xlabel='Category',
    ylabel='Number of Images',
    title='Distribution of Training and Validation Sets'
)


# **Class Weights based on the Training Set**

# In[ ]:


#from data import compute_class_weight_dict

class_weight_dict, class_weights = compute_class_weight_dict(df_train)
print('Class Weights Dictionary (without UNK):')
print(class_weight_dict)

# Create a bar chart
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
ax.set_title('Class Weights')
ax.set(xlabel='Category', ylabel='Weight')
plt.bar(known_category_names, [class_weight_dict[i] for i in range(known_category_num)]);


# Per - Channel Mean and Standard Deviation over the Training Set

# In[ ]:


from __future__ import division
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import pandas as pd
import os
import cv2

def path_to_tensor(img_path, size=(224, 224)):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=size)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, size=(224, 224)):
    list_of_tensors = [path_to_tensor(img_path, size) for img_path in img_paths]
    return np.vstack(list_of_tensors)

def calculate_mean_std(img_paths):
    """
    Calculate the image per channel mean and standard deviation.

    # References
        https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
    """
    
    # Number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
    channel_num = 3
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(channel_num)
    channel_sum_squared = np.zeros(channel_num)

    for path in img_paths:
        im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im/255.
        pixel_num += (im.size/channel_num)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std

def preprocess_input(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images. Each image is normalized by subtracting the mean and dividing by the standard deviation channel-wise.
    This function only implements the 'torch' mode which scale pixels between 0 and 1 and then will normalize each channel with respect to the training dataset of approach 1 (not include validation set).

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    # References
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(K.floatx(), copy=False)

    # Mean and STD from ImageNet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Mean and STD calculated over the Training Set
    # Mean:[0.6236094091893962, 0.5198354883713194, 0.5038435406338101]
    # STD:[0.2421814437693499, 0.22354427793687906, 0.2314805420919389]
    x /= 255.
    mean = [0.6236, 0.5198, 0.5038]
    std = [0.2422, 0.2235, 0.2315]

    if data_format is None:
        data_format = K.image_data_format()

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

def preprocess_input_2(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images. Each image is normalized by subtracting the mean and dividing by the standard deviation channel-wise.
    This function only implements the 'torch' mode which scale pixels between 0 and 1 and then will normalize each channel with respect to the training dataset of approach 2 (not include validation set).

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    # References
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(K.floatx(), copy=False)

    # Mean and STD calculated over the training set of approach 2
    # Mean:[0.6296238064420809, 0.5202302775509949, 0.5032952297664738]
    # STD:[0.24130893564897463, 0.22150225707876617, 0.2297057828857888]
    x /= 255.
    mean = [0.6296, 0.5202, 0.5033]
    std = [0.2413, 0.2215, 0.2297]

    if data_format is None:
        data_format = K.image_data_format()

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

def ensemble_predictions(result_folder, category_names, save_file=True,
                         model_names=['inception_resnet' 'nasnet'],
                         postfixes=['best_balanced_acc', 'best_loss', 'latest']):
    """ Ensemble predictions of different models. """
    for postfix in postfixes:
        # Load models' predictions
        df_dict = {model_name : pd.read_csv(os.path.join(result_folder, "{}_{}.csv".format(model_name, postfix))) for model_name in model_names}

        # Check row number
        for i in range(1, len(model_names)):
            if len(df_dict[model_names[0]]) != len(df_dict[model_names[i]]):
                raise ValueError("Row numbers are inconsistent between {} and {}".format(model_names[0], model_names[i]))

        # Check whether values of image column are consistent
        for i in range(1, len(model_names)):
            inconsistent_idx = np.where(df_dict[model_names[0]].image != df_dict[model_names[i]].image)[0]
            if len(inconsistent_idx) > 0:
                raise ValueError("{} values of image column are inconsistent between {} and {}"
                                .format(len(inconsistent_idx), model_names[0], model_names[i]))

        # Copy the first model's predictions
        df_ensemble = df_dict[model_names[0]].drop(columns=['pred_category'])

        # Add up predictions
        for category_name in category_names:
            for i in range(1, len(model_names)):
                df_ensemble[category_name] = df_ensemble[category_name] + df_dict[model_names[i]][category_name]

        # Take average of predictions
        for category_name in category_names:
            df_ensemble[category_name] = df_ensemble[category_name] / len(model_names)

        # Ensemble Predictions
        df_ensemble['pred_category'] = np.argmax(np.array(df_ensemble.iloc[:,1:(1+len(category_names))]), axis=1)

        # Save Ensemble Predictions
        if save_file:
            ensemble_file = os.path.join(result_folder, "Ensemble_{}.csv".format(postfix))
            df_ensemble.to_csv(path_or_buf=ensemble_file, index=False)
            print('Save "{}"'.format(ensemble_file))
    return df_ensemble

def logistic(x, x0=0, L=1, k=1):
    """ Calculate the value of a logistic function.

    # Arguments
        x0: The x-value of the sigmoid's midpoint.
        L: The curve's maximum value.
        k: The logistic growth rate or steepness of the curve.
    # References https://en.wikipedia.org/wiki/Logistic_function
    """

    return L / (1 + np.exp(-k*(x-x0)))


# In[ ]:


#from utils import calculate_mean_std

### Uncomment below codes to calculate per-channel mean and standard deviation over the training set
#rgb_mean, rgb_std = calculate_mean_std(df_train['path'])
#print("Mean:{}\nSTD:{}".format(rgb_mean, rgb_std))

# Output was:
# Mean:[0.6236094091893962, 0.5198354883713194, 0.5038435406338101]
# STD:[0.2421814437693499, 0.22354427793687906, 0.2314805420919389]

#or
# Mean:[0.6296238064420809, 0.5202302775509949, 0.5032952297664738]
# STD:[0.24130893564897463, 0.22150225707876617, 0.2297057828857888]


# **Samples of each Known Category**

# In[ ]:


from IPython.display import Image

category_groups = df_train.groupby('category')

# Number of samples for each category
num_per_category = 5

fig, axes = plt.subplots(nrows=known_category_num, ncols=num_per_category, figsize=(12, 24))
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
fig.patch.set_facecolor('white')

for idx, val in enumerate(known_category_names):
    i = 0
    for index, row in category_groups.get_group(idx).head(num_per_category).iterrows():
        ax = axes[idx, i]
        ax.imshow(plt.imread(row['path']))
        ax.set_xlabel(row['image'])
        if ax.is_first_col():
            ax.set_ylabel(val, fontsize=20)
            ax.yaxis.label.set_color('blue')
        i += 1
    
fig.tight_layout()
fig.savefig('Samples of training data.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# **Transfer Learning**

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
import keras 
from keras.utils import np_utils
from glob import glob


# In[ ]:


def load_dataset(path):
    data = load_files(path, shuffle=True)
    img_files = np.array(data['filenames'])
    img_targets = np_utils.to_categorical(np.array(data['target']), 3)
    return img_files, img_targets


# In[ ]:


train_files, train_labels = load_dataset(df_train)
valid_files, valid_labels = load_dataset(df_valid)


# In[ ]:


from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(path):
    
    img = image.load_img(path, target_size = (224,224))
    x= image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(paths):
    list_of_tensors = [path_to_tensor(path) for path in tqdm(paths)]
    return np.vstack(list_of_tensors)


# In[ ]:


train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(df_val).astype('float32')/255
test_tensors = paths_to_tensor(df_test).astype('float32')/255


# **Pre-trained Models**

# 1. Inception_Resnet

# In[ ]:


from keras.applications.inception_resnet_v2 import InceptionResNetV2

model_inception_resnet = InceptionResNetV2(weights = 'imagenet', include_top = False)


# In[ ]:


train_features_inception_resnet = model_inception_resnet.predict(train_tensors, verbose=1)
valid_features_inception_resnet = model_inception_resnet.predict(valid_tensors, verbose=1)
test_features_inception_resnet = model_inception_resnet.predict(test_tensors, verbose=1)


# In[ ]:


model_inception_resnet = Sequential()

model_inception_resnet.add(GlobalAveragePooling2D(input_shape = train_features_inception_resnet.shape[1:]))
model_inception_resnet.add(Dropout(0.2))
model_inception_resnet.add(Dense(1024, activation = 'relu'))
model_inception_resnet.add(Dropout(0.2))
model_inception_resnet.add(Dense(512, activation = 'relu'))
model_inception_resnet.add(Dropout(0.2))
model_inception_resnet.add(Dense(128, activation = 'relu'))
model_inception_resnet.add(Dropout(0.2))
model_inception_resnet.add(Dense(3, activation ='softmax'))

model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=input_shape))
model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Flatten())
  
model.add(Dropout(0.5))
model.add(BatchNormalization())
    
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
    
model.add(Dense(num_class , activation='softmax'))


model_inception_resnet.summary()


# In[ ]:


opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
model_inception_resnet.compile(optimizer= opt, metrics = ['accuracy'], loss='categorical_crossentropy')


# In[ ]:


from keras.callbacks import ModelCheckpoint

checkpoint_inception = ModelCheckpoint(
    save_best_only = True, 
    verbose = 1, 
    filepath = 'models/weights.best.from_inception_resnet_v2.hdf5')

model_inception_resnet.fit(train_features_inception_resnet, 
          train_labels, 
          epochs=35, 
          batch_size= 64, 
          validation_data=(valid_features_inception, valid_labels), callbacks=[checkpoint_inception], verbose=1
         )


# In[ ]:


model_inception_resnet.load_weights('models/weights.best.from_inception_resnet_v2.hdf5')


# In[ ]:


submission_inception_resnet = pd.DataFrame({'Id':test_files, 'task_1':test_predictions_task1,'task_2':test_predictions_task2})
pd.DataFrame.to_csv(submission_inception_resnet, 'submission.csv', index=False)


# **end of Inception_Resnet**

# 2. NasNetLarge

# In[ ]:


from keras.applications.nasnet import NASNetLarge
model_nasnet = NASNetLarge(weights = 'imagenet', include_top = False)


# In[ ]:


train_features_nasnet = model_nasnet.predict(train_tensors, verbose=1)
valid_features_nasnet = model_nasnet.predict(valid_tensors, verbose=1)
test_features_nasnet = model_nasnet.predict(test_tensors, verbose=1)


# In[ ]:



model_nasnet = Sequential()

model_nasnet.add(GlobalAveragePooling2D(input_shape = train_features_nasnet.shape[1:]))
model_nasnet.add(Dropout(0.2))
model_nasnet.add(Dense(1024, activation = 'relu'))
model_nasnet.add(Dropout(0.2))
model_nasnet.add(Dense(512, activation = 'relu'))
model_nasnet.add(Dropout(0.2))
model_nasnet.add(Dense(128, activation = 'relu'))
model_nasnet.add(Dropout(0.2))
model_nasnet.add(Dense(3, activation ='softmax'))

    model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Flatten())
    
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(num_class , activation='softmax'))

model_nasnet.summary()


# In[ ]:


opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
model.compile(optimizer= opt, metrics = ['accuracy'], loss='categorical_crossentropy')


# In[ ]:


from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(save_best_only = True, verbose =1, 
                             filepath = 'models/weights.best.from_nasnet.hdf5')

model.fit(train_features_nasnet, 
          train_labels, 
          epochs=25, 
          batch_size= 64, 
          validation_data=(valid_features_nasnet, valid_labels), callbacks=[checkpoint], verbose=1
         )


# In[ ]:


model.load_weights('models/weights.best.from_nasnet.hdf5')


# In[ ]:


test_predictions = np.argmax(model.predict(test_features_nasnet), axis=1)
accuracy = 100 * np.sum(np.array(test_predictions) == np.argmax(test_labels, axis=1))/len(test_predictions)
print ('Accuracy of NasNet model on test set = %.4f%%' % accuracy)


# In[ ]:


print(np.argmax(test_labels[25]))
print(test_predictions[25])


# In[ ]:


import cv2
img = cv2.imread(test_files[25])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


# In[ ]:


test_predictions = model.predict(test_features_nasnet)
test_predictions_task1 = test_predictions[:,0]
test_predictions_task2 = test_predictions[:,2]


# In[ ]:


submission_nasnet = pd.DataFrame({'Id':test_files, 'task_1':test_predictions_task1,'task_2':test_predictions_task2})
pd.DataFrame.to_csv(submission_nasnet, 'submission.csv', index=False)


# **Train Models by Transfer Learning**

# In[ ]:


get_ipython().system('python3 main.py /home --approach 2 --modelfolder models --training --epoch 100 --batchsize 32 --maxqueuesize 10 --model weights.best.from_inception_resnet_v2 weights.best.from_nasnet ')


# **Training**

# In[ ]:


print("Starting...\n")

start_time = time.time()
print(date_time(1))

# batch_size = 32
# train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(batch_size=batch_size)

print("\n\nCompliling Model ...\n")
learning_rate = 0.0001
optimizer = Adam(learning_rate)
# optimizer = Adam()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# steps_per_epoch = 180
# validation_steps = 40

verbose = 1
epochs = 10

print("Trainning Model ...\n")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_steps, 
    class_weight=class_weights)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print("\nElapsed Time: " + elapsed_time)
# print("Elapsed Time/Epoch: " + elapsed_time/epochs)
print("Completed Model Trainning", date_time(1))


# **Model Performance**

# Model Performance Visualization over the Epochs

# In[ ]:


def plot_performance(history=None, figure_directory=None):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    ylim_pad = [0.01, 0.1]


    plt.figure(figsize=(20, 5))

    # Plot training & validation Accuracy values

    y1 = history.history['acc']
    y2 = history.history['val_acc']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()


    # Plot training & validation loss values

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()


# In[ ]:


plot_performance(history=history)


# In[ ]:


ypreds = model.predict_generator(generator=test_generator, steps = len(test_generator),  verbose=1)


# In[ ]:


ypred = ypreds[:,1]


# In[ ]:


sample_df = pd.read_csv(input_directory+"sample_submission.csv")
sample_list = list(sample_df.id)

pred_dict = dict((key, value) for (key, value) in zip(test_generator.filenames, ypred))

pred_list_new = [pred_dict[f+'.tif'] for f in sample_list]

test_df = pd.DataFrame({'id':sample_list,'label':pred_list_new})

test_df.to_csv('submission.csv', header=True, index=False)


# In[ ]:


test_df.head()


# **Complexity Graph of Transfer Learning Models**

# In[ ]:


from visuals import *

model_names = ['weights.best.from_inception_resnet_v2', 'weights.best.from_nasnet']
feature_extract_epochs = 3

for model_name in model_names:
    file_path = os.path.join(history_folder, "{}.training.csv".format(model_name))
    if os.path.exists(file_path):
        fig = plot_complexity_graph(csv_file=file_path,
                              title="Complexity Graph of {}".format(model_name),
                              figsize=(14, 10),
                              feature_extract_epochs=feature_extract_epochs)
        fig.savefig(os.path.join(history_folder, "{}.training.svg".format(model_name)), format='svg',
                    bbox_inches='tight', pad_inches=0)


# **Predict Validation Set**

# **Predict Validation Set by Different Models**

# In[ ]:


# !python3 main.py /home --approach 2 --modelfolder models_2 --predval --predvalresultfolder predict_results_2 --model model_inception_resnet, model_nasnet
get_ipython().system('python main.py C:\\ISIC_2019 --approach 2 --modelfolder models_2 --predval --predvalresultfolder predict_results_2 --model weights.best.from_inception_resnet_v2 weights.best.from_nasnet ')


# **Ensemble Models' Predictions on Validation Set**

# In[ ]:


from utils import ensemble_predictions

ensemble_predictions(pred_result_folder_val, category_names)


# **Load Prediction Results on Validation Set**

# In[ ]:


import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from visuals import plot_confusion_matrix
from keras.utils import np_utils
from keras_numpy_backend import categorical_crossentropy

model_names = ['weights.best.from_inception_resnet_v2', 'weights.best.from_nasnet']
postfix = 'best_balanced_acc'
print('Model selection criteria: ', postfix)

for model_name in model_names:
    # Load predicted results
    file_path = os.path.join(pred_result_folder_val, "{}_{}.csv".format(model_name, postfix))
    if not os.path.exists(file_path):
        continue

    print("========== {} ==========".format(model_name))
    df = pd.read_csv(file_path)
    y_true = df['category']
    y_pred = df['pred_category']

    # Compute Balanced Accuracy
    print('balanced_accuracy_score: ', balanced_accuracy_score(y_true, y_pred))
    print('macro recall_score: ', recall_score(y_true, y_pred, average='macro'))

    # Compute categorical_crossentropy
    y_true_onehot = np_utils.to_categorical(df['category'], num_classes=category_num)
    y_pred_onehot = np.array(df.iloc[:,1:1+category_num])
    print('categorical_crossentropy: ',
          np.average(categorical_crossentropy(y_true_onehot, y_pred_onehot)))

    # Compute weighted categorical_crossentropy
    print('weighted categorical_crossentropy: ',
          np.average(categorical_crossentropy(y_true_onehot, y_pred_onehot, class_weights=class_weights)))

    # Confusion Matrix
    fig = plot_confusion_matrix(y_true, y_pred, category_names, normalize=True,
                                title="Confusion Matrix of {}".format(model_name),
                                figsize=(8, 6))
    fig.savefig(os.path.join(pred_result_folder_val, "{}_{}.svg".format(model_name, postfix)), format='svg',
                bbox_inches='tight', pad_inches=0)
    print('')


# In[ ]:


from visuals import plot_grouped_2bars

sample_count_val = y_true.shape[0]
print("Validation set has {} samples.\n".format(sample_count_val))

print('========== Ground Truth ==========')
count_true = Counter(y_true)
for i, c in enumerate(category_names):
    print("'%s':\t%d\t(%.2f%%)" % (c, count_true[i], count_true[i]*100/sample_count_val))

for model_name in model_names:
    # Load predicted results
    file_path = os.path.join(pred_result_folder_val, "{}_{}.csv".format(model_name, postfix))
    if not os.path.exists(file_path):
        continue

    print("\n========== {} Prediction ==========".format(model_name))
    df = pd.read_csv(file_path)
    y_pred = df['pred_category']
    
    count_pred = Counter(y_pred)
    for i, c in enumerate(category_names):
        print("'%s':\t%d\t(%.2f%%)" % (c, count_pred[i], count_pred[i]*100/sample_count_val))

    # Plot Prediction Distribution
    plot_grouped_2bars(
        scalars=[[count_true[i] for i in range(category_num)],
                 [count_pred[i] for i in range(category_num)]],
        scalarlabels=['Ground Truth', 'Prediction'],
        xticklabels=category_names,
        xlabel='Category',
        ylabel='Number of Images',
        title="Prediction Distribution of {}".format(model_name)
    )


# **Test DATA**

# **Predict Test Data by Different Models**

# In[ ]:


get_ipython().system('python3 main.py /home --approach 2 --modelfolder models_2 --predtest --predtestresultfolder test_predict_results_2 --model weights.best.from_inception_resnet_v2 weights.best.from_nasnet')

