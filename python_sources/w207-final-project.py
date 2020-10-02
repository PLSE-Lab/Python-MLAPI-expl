#!/usr/bin/env python
# coding: utf-8

# # W207 Applied Machine Learning
# 
# ## Summer 2018 Final Project
# 
# **Team Members**: Rachel Ho, Dan Rasband, Subha Vadakkumkoor, Matt Vay
# 
# **Topic**: ASL Alphabet
# 
# #### Kaggle Datasets
# 
# * [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) - this data set is the basis for all models.
# * [ASL Alphabet Test](https://www.kaggle.com/danrasband/asl-alphabet-test) - this data set is used to test against more realistic images.
# 
# #### Running this Notebook Locally
# 
# 1. Download the datasets from the links above and update the directory paths accordingly in the relevant cell blocks.
# 
# ```
# TRAIN_DIR = "directory path where training dataset is saved"
# ```
# 
# Note that the current structure is set to how things would look if this notebook were to be run on Kaggle.
# 
# 2. Install libraries below.
# 
# ```
# pip install --upgrade opencv-python tqdm scikit-image pandas \
#     numpy matplotlib keras tensorflow scikit-learn seaborn
# ```
# 
# #### Credits
# 
# 1. Code for loading images is credited to Paul Mooney: https://www.kaggle.com/paultimothymooney/interpret-sign-language-with-deep-learning
# 
# 2. Image processing credited to Adrian Rosebrock:  https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
# 
# 3. Convolutional Neural Network: https://www.kaggle.com/grassknoted/asl-alphabet
# 
# #### Libraries to Install
# 
# To run this notebook, you may have to run the following commands to install some prerequisites:
# 
# ```bash
# pip install --upgrade opencv-python tqdm scikit-image pandas \
#     numpy matplotlib keras tensorflow scikit-learn seaborn
# ```

# In[ ]:


# General libraries
import cv2
import random
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from os import getenv
import itertools

# Deep Learning using Keras
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import print_summary
get_ipython().run_line_magic('matplotlib', 'inline')

# Sci-kit learn libraries
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Image Preprocessing
from skimage.transform import resize
from skimage.color import rgb2gray

# Set random seeds so results are the same each time
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)


# ### 1. Introduction
# 
# Our dataset contains 29 classes: one for each letter from A to Z, space, delete, and nothing.  Each example in the dataset is a three-dimensional array of 200 X 200 X 3 pixels.  For the purpose of this project, in the step where the images are loaded from their corresponding sub-folder, we re-size the images to 50 x 50 x 3 (and 64 x 64 x 3 in the section on convolutional neural networks) to make the run-times of training more managable and reduce the number of pixels from 120,000 to 7,500.
# 
# We set the mini training dataset to 2900 examples as there are 29 labels so there are 100 examples for each label.  When splitting up the dataset, we collapse the three dimensional feature arrays (50X50X3) into one dimensional feature arrays (7,500) so they can be used for training.  
# 
# First, we read and load the data into arrays. As the data size is huge, instead of reading all the images, we read in only a sample. As we have 3000 images of each letter named in the format of A1, A2, ..., A3000, B1, B2, ...., B3000, we create a random sample of indices and then read in only the sampled indiced. We use the `train_test_split` functionality to split the indices into two disjoint train and test indices.

# In[ ]:


# Define parameters and functions

# Change this to point to your data directory (if the default relative path 
# below does not work, please provide complete path: 
# eg:'C:/Users/subsh/Documents/207/final_project/asl-alphabet/asl_alphabet_train/asl_alphabet_train/)
TRAIN_DIR = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'
FILE_PATTERN = TRAIN_DIR + '{letter}/{letter}{index}.jpg'

# Define train size - this is the number of training samples for each letter
FULL_TRAIN_SIZE = 3000
MINI_TRAIN_SIZE = 100

# Define the classes (letters of the ASL alphabet) that are being analyzed.
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Define a hash table (dict) to store labels
LABEL_DICT = dict(zip(LETTERS, np.arange(1, 30)))

# Image sizing
IMAGE_SIZE = 50


# In[ ]:


# Some visualization helpers

def plot_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
    '''
    Plot a confusion matrix heatmap using matplotlib. This code was obtained from
    the scikit-learn documentation:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return


def plot_confusion_matrix_with_default_options(y_pred, y_true, classes):
    '''Plot a confusion matrix heatmap with a default size and default options.'''
    cm = confusion_matrix(y_true, y_pred)
    with sns.axes_style('ticks'):
        plt.figure(figsize=(16, 16))
        plot_confusion_matrix(cm, classes)
        plt.show()
    return


# In[ ]:


def get_data(indices):
    '''Read data into two numpy arrays.'''
    xdata = []
    ydata = []

    for letter in LETTERS:
        for index in indices:
            # Create label
            ydata.append(LABEL_DICT[letter])

            # Read original image
            image_path = FILE_PATTERN.format(letter=letter, index=index)
            img_file = cv2.imread(image_path)

            if img_file is None:
                continue
            img_file = resize(img_file, (IMAGE_SIZE, IMAGE_SIZE, 3))

            # Append to data array
            xdata.append(np.asarray(img_file))

    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    return xdata, ydata


# Read in train and dev data
INDEX_LIST = np.arange(1, 3001)
TRAIN_INDICES, DEV_INDICES = train_test_split(
    INDEX_LIST,
    train_size=MINI_TRAIN_SIZE,
    test_size=MINI_TRAIN_SIZE,
    random_state=42)
X_MINI_TRAIN_ORIG, Y_MINI_TRAIN = get_data(TRAIN_INDICES)
X_DEV_ORIG, Y_DEV = get_data(DEV_INDICES)

print('Initial shape of training data:', X_MINI_TRAIN_ORIG.shape)
print('Initial shape of dev data:', X_DEV_ORIG.shape)

# Reshape the images from 3d to 2d arrays. The 3d images will be used for color
# based pre-processing and plotting while the 2d arrays will be used for other
# analyses.

NSAMPLES, NX, NY, NZ = X_MINI_TRAIN_ORIG.shape
X_MINI_TRAIN = X_MINI_TRAIN_ORIG[:NSAMPLES, :, :, :].reshape(NSAMPLES, 7500)
Y_MINI_TRAIN = Y_MINI_TRAIN[:NSAMPLES]

NSAMPLES_DEV, NX, NY, NZ = X_DEV_ORIG.shape
X_DEV = X_DEV_ORIG[:NSAMPLES_DEV, :, :, :].reshape(NSAMPLES_DEV, 7500)
Y_DEV = Y_DEV[:NSAMPLES_DEV]

print('Current shape of training data:', X_MINI_TRAIN.shape)
print('Current shape of dev data:', X_DEV.shape)


# ### 2. Exploration Data Analysis
# 
# We will explore the data to ensure that all labels are read in and also to visually examine the variations in the images.

# In[ ]:


def show_class_counts():
    '''Look at counts of each label in the mini training data to
    ensure the data is spread across the various labels.'''
    df = pd.DataFrame(Y_MINI_TRAIN, columns=['labels'])
    df['count'] = 1
    pd.pivot_table(df, columns='labels', values='count',
                   aggfunc=np.count_nonzero)

    plt.subplots(figsize=(10, 5))
    sns.countplot(df['labels'])
    return


show_class_counts()


# We can see the the variations in gestures in sample plots below of 10 random images of each letter. The length of the hands, extent of background images, color intensity and brightness, all vary from image to image.

# In[ ]:


def plot_examples(num_examples):
    '''Define counter variable to iterate through unique Y label values.'''
    plt.figure(figsize=(16, len(LETTERS) * 2))

    plot_index = 0
    for counter in np.unique(Y_MINI_TRAIN):
        # Subset data for each letter
        x2 = X_MINI_TRAIN_ORIG[np.where(Y_MINI_TRAIN == counter)[0]]
        y2 = Y_MINI_TRAIN[np.where(Y_MINI_TRAIN == counter)[0]]

        for _, (image, label) in enumerate(zip(x2[0:num_examples], y2[0:num_examples])):
            plot_index += 1
            plt.subplot(len(LETTERS), num_examples, plot_index)
            plt.imshow(image, cmap=plt.cm.gray)
            plt.title(LETTERS[label - 1])
            plt.axis('off')

    return


plot_examples(10)


# In[ ]:


def plot_symbol(symbol, examples):
    '''Function to plot a symbol a certain number of times (examples variable).'''
    # Pulls the index for the symbol. This is also the label in our y data.
    label = LETTERS.index(symbol)

    # Filter data so it only has the symbol.
    data = X_MINI_TRAIN_ORIG[np.where(Y_MINI_TRAIN == label)]

    # Random sampling of filtered data with the number of examples specified.
    img_arr = data[random.sample(range(0, data.shape[0]-1), examples)]

    # Create subplots and titles.
    fig, ax = plt.subplots(nrows=examples, ncols=4,
                           sharex=True, sharey=True, figsize=(10, 20))
    title = "Subplots of " + LETTERS[label]
    fig.suptitle(title, fontsize=30, y=0.93)

    # For each example, grab the Red, Green, Blue, and Complete representation of the data
    # and show it in the subplot.
    for i in range(examples):
        data_all = img_arr[i, :, :, :]
        data_red = img_arr[i, :, :, 0]
        data_green = img_arr[i, :, :, 1]
        data_blue = img_arr[i, :, :, 2]

        ax[i][0].imshow(data_all)
        ax[i][0].set_title("Original")
        ax[i][0].axis('off')

        ax[i][1].imshow(data_red)
        ax[i][1].set_title("Red")
        ax[i][1].axis('off')

        ax[i][2].imshow(data_green)
        ax[i][2].set_title("Green")
        ax[i][2].axis('off')

        ax[i][3].imshow(data_blue)
        ax[i][3].set_title("Blue")
        ax[i][3].axis('off')
    return


# In[ ]:


plot_symbol('B', 5)


# ### 3. Images Pre-processing
# 
# #### 3.1 Extracting R, G and B
# 
# Since each image has the color dimension that we saw above, one of our pre-processing steps is to extract each of these to create separate blue, red and green training examples. 

# In[ ]:


# Break the data into red, green and blue mini training sets to use in our later algorithms.
# For this, first we need to break down the third dimension and reshape to a 2d array

# Reshape train data
NSAMPLES, NX, NY, _ = X_MINI_TRAIN_ORIG.shape
X_RED_MINI_TRAIN = X_MINI_TRAIN_ORIG[:NSAMPLES, :, :, 0].reshape(
    NSAMPLES, 2500)
X_GREEN_MINI_TRAIN = X_MINI_TRAIN_ORIG[:NSAMPLES, :, :, 1].reshape(
    NSAMPLES, 2500)
X_BLUE_MINI_TRAIN = X_MINI_TRAIN_ORIG[:NSAMPLES, :, :, 2].reshape(
    NSAMPLES, 2500)

print("X_RED_MINI_TRAIN Shape:", X_RED_MINI_TRAIN.shape)
print("X_GREEN_MINI_TRAIN Shape:", X_GREEN_MINI_TRAIN.shape)
print("X_BLUE_MINI_TRAIN Shape:", X_BLUE_MINI_TRAIN.shape)

# Reshape dev data
NSAMPLES_DEV, _, _, _ = X_DEV_ORIG.shape
X_RED_DEV = X_DEV_ORIG[:NSAMPLES_DEV, :, :, 0].reshape(NSAMPLES_DEV, 2500)
X_GREEN_DEV = X_DEV_ORIG[:NSAMPLES_DEV, :, :, 1].reshape(NSAMPLES_DEV, 2500)
X_BLUE_DEV = X_DEV_ORIG[:NSAMPLES_DEV, :, :, 2].reshape(NSAMPLES_DEV, 2500)


# #### 3.2 Gaussian blur
# 
# We create a Gaussian blur as a pre-processing step to see if it can help improve training accuracy. The Gaussian blur takes the average of a pixel's 8 nearest neighbors and returns the average as its new pixel value. This technique was used on a few different algorithms such as k nearest neighbors and support vector machines with varying degrees of success which will be discussed further on in our report.

# In[ ]:


# This function will take a data set as an input and perform a gaussian blur to
# the data. It will then return that data set with a blur, meaning a pixel will
# take on the average of its 8 closest neighbors (plus itself).
def blur(data):

    # Initialize a blank data set that is the same size as the 'data' set
    # parameter passed to our function We then iterate through the rows in our
    # data set, transforming them into a 50X50 matrix and performing the
    # gaussian blur on them
    blurred_data_set = np.zeros((data.shape))
    for i in range(data.shape[0]):
        blur_matrix = np.zeros((50, 50))
        test_matrix = np.reshape(data[i], (50, 50))

        # for each 50X50 matrix we have made, we iterate through each pixel
        # (excluding the edges) and take the average of that respective pixel
        # and its 8 closest neighbors and make that the pixel's new value. After
        # the loop we then return that matrix back to its original shape and add
        # it to a blurred_data_set that will end up being the transformed
        # version of the original 'data' parameter. This blurred_data_set will
        # then be returned.
        for j in range(1, 49):
            for k in range(1, 49):
                blur_matrix[j][k] = (test_matrix[j-1][k-1] + test_matrix[j-1][k]
                                     + test_matrix[j][k] + test_matrix[j-1][k+1] + test_matrix[j][k-1]
                                     + test_matrix[j][k+1] + test_matrix[j+1][k-1] + test_matrix[j+1][k]
                                     + test_matrix[j+1][k+1]) / 9
        blur_matrix = np.reshape(blur_matrix, (2500))
        blurred_data_set[i] = blur_matrix
    return blurred_data_set


# Blur our mini train data and dev data for use in our models
BLUR_RED_MINI_TRAIN = blur(X_RED_MINI_TRAIN)
BLUR_RED_DEV = blur(X_RED_DEV)

BLUR_GREEN_MINI_TRAIN = blur(X_GREEN_MINI_TRAIN)
BLUR_GREEN_DEV = blur(X_GREEN_DEV)

BLUR_BLUE_MINI_TRAIN = blur(X_BLUE_MINI_TRAIN)
BLUR_BLUE_DEV = blur(X_BLUE_DEV)


# In[ ]:


# Take the blurred red, green and blue data pixels and stack them back together
# into an array of the original size and shape of the mini training data. This
# leads to a blurring of the original image.
X_BLUR = np.stack(
    [BLUR_RED_MINI_TRAIN, BLUR_GREEN_MINI_TRAIN, BLUR_BLUE_MINI_TRAIN], axis=2)
X_BLUR_DEV = np.stack([BLUR_RED_DEV, BLUR_GREEN_DEV, BLUR_BLUE_DEV], axis=2)

X_BLUR_MINI_TRAIN = X_BLUR.reshape(NSAMPLES, 7500)
X_BLUR_DEV = X_BLUR_DEV.reshape(NSAMPLES_DEV, 7500)


# #### 3.3 Edge Detection
# 
# In this segment we try various image processing functionalities available in Python to blur in the data ways other than Gaussian, binarize and to extract the contour/edges of images. Edge detection can be tricky because there can be edges in the background as well and thus how tightly/loosely to identify the edges can affect the quality of the output image and the prediction. In addition to a 'manual' edging using parameters provided, we also use a function to detect the image edges automatically. This function has been adapted from https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

# In[ ]:


def auto_canny(image, sigma=0.33):
    '''Define automatic canny detection

    See: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    '''
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


# In[ ]:


def get_data2(indices):
    nrows = 1
    ncols = 8

    xdata = []
    ydata = []

    for letter in LETTERS:
        for index in indices:
            # Create label
            ydata.append(LABEL_DICT[letter])

            # Read original image
            image_path = FILE_PATTERN.format(letter=letter, index=index)
            image = cv2.imread(image_path)

            # Process image

            # Apply Guassian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)

            # Convert to HSV
            imgray = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

            # Apply median blur - need to pick one of Guassian or median
            imgray2 = cv2.medianBlur(imgray, 5)

            # Resize
#             resized = resize(imgray2, (IMAGE_SIZE, IMAGE_SIZE, 3))
            resized = imgray2

            # Convert to gray - THIS IS NOT WORKING WITH CANNY
            resized2 = rgb2gray(resized)

            # Get the edges - tight
            # Try changing this for better edge detection
            edge_t = cv2.Canny(resized, 100, 250)

            # Get the edges - auto
            edge_a = auto_canny(resized)

            # Convert to gray - THIS IS NOT WORKING WITH CANNY
            resized2 = rgb2gray(edge_t)
            # Binarize to be done
#             https://sourcedexter.com/manipulating-image-pixels-with-python-scikit-image-color-schemes/

            # Pick which image to use for training
            xdata.append(np.asarray(resized2))

        # THIS DOES NOT SEEM TO BE WORKING  - FIX SIZE
        plt.figure(figsize=(16, 5))

        plots = {
            'Original {}'.format(letter): image,
            'Gaussian Blur': blurred,
            'HSV': imgray,
            'Median blur': imgray2,
            'Resized': resized,
            'Tight edged': edge_t,
            'Auto edged': edge_a,
            'Edged grayscale': edge_a,
        }

        plot_index = 0
        for name, image in plots.items():
            plot_index += 1
            plt.subplot(nrows, ncols, plot_index)
            plt.imshow(image)
            plt.title("Original " + letter)
            plt.axis('off')

        plt.show()

    return xdata, ydata


X_IMAGE_MINI_TRAIN_ORIG, Y_TRAIN = get_data2(TRAIN_INDICES)
X_IMAGE_DEV_ORIG, Y_DEV = get_data2(DEV_INDICES)

X_IMAGE_MINI_TRAIN_ORIG = np.asarray(X_IMAGE_MINI_TRAIN_ORIG)
X_IMAGE_DEV_ORIG = np.asarray(X_IMAGE_DEV_ORIG)

# Reshape train data
NSAMPLES, NX, NY = X_IMAGE_MINI_TRAIN_ORIG.shape
X_IMAGE_MINI_TRAIN = X_IMAGE_MINI_TRAIN_ORIG[:NSAMPLES, :, :].reshape(NSAMPLES, NX * NY)

# Reshape dev data
NSAMPLES_DEV, NX, NY = X_IMAGE_DEV_ORIG.shape
X_IMAGE_DEV = X_IMAGE_DEV_ORIG[:NSAMPLES_DEV, :, :].reshape(NSAMPLES_DEV, NX * NY)


# ### 4. Feature Engineering - Dimensionality Reduction with PCA
# 
# As the dimensions (7,500) of our training data are still quite large after re-sizing the images, we try to reduce the dimensions using PCA in order to see if we can make the algorithms run more efficiently and if we can improve the accuracy for some of the classifiers. Our goal is to retain 95% of the variance. Originally we wanted to retain 99% of the variance but discovered that this requires the number of componenets to go up to thousands, essentially defeating the purpose of PCA. As can be seen from the results below, if we set the number of principal components to 110, we can reach 95% explained variance. The plot below shows the number of principal components vs. cumulative explained variance as we increase the number of components. We reduce the dimensions of the training dataset and development dataset by transforming them with the fitted PCA and store them separately to be used for training.
# Interestingly, we find that the accuracy worsens for the tree classifiers including random forests and bagged tress but improves for other classifers including KNN and SVM.

# In[ ]:


# Set the number of components to be used below.
N_COMPONENTS = 110

def reduce_dimensionality(model, data):
    '''Reduce dimensionality with PCA.'''
    n_components = 110
    model.fit(data)

    # # look at cumulative explained variance in increments of 10 components
    i = [1] + list(range(10, n_components + 1, 10))
    df = pd.DataFrame(
        index=i, columns=["num_components", "cum_pct_variance"])
    for k in i:
        df.loc[k] = [k, sum(model.explained_variance_ratio_[:k])]

    return df


PCA_MODEL = PCA(n_components=N_COMPONENTS, random_state=0)
DF_PCA = reduce_dimensionality(PCA_MODEL, X_MINI_TRAIN)


# In[ ]:


def plot_pca(df):
    '''Plot number of principal components vs cumulative explained variance.'''
    plt.plot(df['num_components'], df['cum_pct_variance'], marker='.')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Percentage')
    plt.show()
    return


plot_pca(DF_PCA)


# In[ ]:


# Transform mini training data and development data using PCA fitted with mini
# training data.
X_MINI_TRAIN_PCA = PCA_MODEL.transform(X_MINI_TRAIN)
X_DEV_PCA = PCA_MODEL.transform(X_DEV)


def pca_transform(model, data, data_pca):
    model.fit(data)
    return model.transform(data_pca)


X_BLUR_MINI_TRAIN_PCA = pca_transform(PCA_MODEL, X_BLUR_MINI_TRAIN, X_BLUR_MINI_TRAIN)
X_BLUR_DEV_PCA = pca_transform(PCA_MODEL, X_BLUR_MINI_TRAIN, X_BLUR_DEV)

X_RED_MINI_TRAIN_PCA = pca_transform(PCA_MODEL, X_RED_MINI_TRAIN, X_RED_MINI_TRAIN)
X_RED_DEV_PCA = pca_transform(PCA_MODEL, X_RED_MINI_TRAIN, X_RED_DEV)
BLUR_RED_MINI_TRAIN_PCA = pca_transform(PCA_MODEL, BLUR_RED_MINI_TRAIN, BLUR_RED_MINI_TRAIN)
BLUR_RED_DEV_PCA = pca_transform(PCA_MODEL, BLUR_RED_MINI_TRAIN, BLUR_RED_DEV)

X_BLUE_MINI_TRAIN_PCA = pca_transform(PCA_MODEL, X_BLUE_MINI_TRAIN, X_BLUE_MINI_TRAIN)
X_BLUE_DEV_PCA = pca_transform(PCA_MODEL, X_BLUE_MINI_TRAIN, X_BLUE_DEV)
BLUR_BLUE_MINI_TRAIN_PCA = pca_transform(PCA_MODEL, BLUR_BLUE_MINI_TRAIN, BLUR_BLUE_MINI_TRAIN)
BLUR_BLUE_DEV_PCA = pca_transform(PCA_MODEL, BLUR_BLUE_MINI_TRAIN, BLUR_BLUE_DEV)

X_GREEN_MINI_TRAIN_PCA = pca_transform(PCA_MODEL, X_GREEN_MINI_TRAIN, X_GREEN_MINI_TRAIN)
X_GREEN_DEV_PCA = pca_transform(PCA_MODEL, X_GREEN_MINI_TRAIN, X_GREEN_DEV)
BLUR_GREEN_MINI_TRAIN_PCA = pca_transform(PCA_MODEL, BLUR_GREEN_MINI_TRAIN, BLUR_GREEN_MINI_TRAIN)
BLUR_GREEN_DEV_PCA = pca_transform(PCA_MODEL, BLUR_GREEN_MINI_TRAIN, BLUR_GREEN_DEV)


# ### 5. Model Development
# 
# #### 5.1 K Nearest Neighbors
# 
# We use KNN as a baseline model to compare against more sophisticated models including bagged trees, SVM, and neural networks. Below we fit the KNN classifiers with both the dataset with original dimensions and the dataset with reduced dimensions. We use GridSearchCV to find the optimal value of k for both datasets. As seen from the results, in both cases, the optimal value of k is 1. This is expected as the feature space is very sparse even for the dataset with reduced dimensions. The training exmmples are spread out far from each other. In both cases, the accuray rates are similar at around 61-62%.

# In[ ]:


def fit_and_tune(traindata, devdata, clf, params_list, clf_name, message, print_cr, print_cm, **kwargs):
    '''For our baseline KNN classifier, we look for optimal k using gridsearch,
    both with and wihtout PCA. We willram use these best parameters to fit train
    data as well as evaluate accuracy on dev data.'''
    gs = GridSearchCV(estimator=clf, param_grid=params_list, refit=True)
    gs.fit(traindata, Y_MINI_TRAIN)
    print('Tuning parameters for {} for classifier {}'.format(message, clf_name))
    print('Best paramaters:', gs.best_params_)
    print('Best grid score: {:.3f}'.format(gs.best_score_))

    # Fit and score with the best parameter found
#     pred = gs.fit_transform(traindata)
    clf.set_params(**gs.best_params_)
    clf.fit(traindata, Y_MINI_TRAIN)

    dev_pred = gs.predict(devdata)
    print("Dev accuracy: ", 100*np.sum(dev_pred == Y_DEV)/devdata.shape[0])

    if print_cr == 1:
        print("Classification Report\n")
        print(classification_report(Y_DEV, dev_pred))

    if print_cm == 1:
        plot_confusion_matrix_with_default_options(
            y_pred=dev_pred,
            y_true=Y_DEV,
            classes=LETTERS)

    return


# Define gridsearch parameters
def fit_and_tune_knn():
    params_list = {'n_neighbors': [1, 5, 10, 15]}
    clf = KNeighborsClassifier()
    clf_name = 'KNN'

    # Call function once on data without PCA and one without:
    fit_and_tune(X_MINI_TRAIN, X_DEV, clf, params_list,
                 clf_name, 'without PCA', 0, 0)
    fit_and_tune(X_MINI_TRAIN_PCA, X_DEV_PCA, clf,
                 params_list, clf_name, 'with PCA', 0, 0)
    return


fit_and_tune_knn()


# After determinig the optimal value of k is 1 with the mini training data and PCA lead to a slightly higher accuracy, we ran the k nearest neighbors algorithm against 8 different variations of our mini train data with our optimal k value. The data sets we used included the original and the original with PCA applied. Along with those, we took our Red, Green, and Blue data sets and applied PCA and/or a gaussian blur. This lead to some interesting results such as the data set with the highest score was taking only the blue pixels from our image, applying a PCA to it and then a gaussian blur lead to a 3.6% accuracy increase over our original mini training data. This could mean a few things such as maybe the blue pixels did a better job at hiding images in the background and brining out the hand and shadows in the foreground. Overall, each time we applied a blur we saw an increase in the accuracy over the original data. It is worth noting that while the "Blur_Blue_PCA" dataset had the highest score, our "Blur_PCA" (original mini training data with a blur applied) came in second place with only an accuracy decrease of 0.2% which is likely insignificant. 

# In[ ]:


# Place data into lists to make it iterable for use in loops when running algorithms.
TRAINING_SETS = [X_MINI_TRAIN, X_MINI_TRAIN_PCA, X_BLUR_MINI_TRAIN_PCA, X_RED_MINI_TRAIN_PCA, 
                 BLUR_RED_MINI_TRAIN_PCA, X_BLUE_MINI_TRAIN_PCA, BLUR_BLUE_MINI_TRAIN_PCA,
                 X_GREEN_MINI_TRAIN_PCA, BLUR_GREEN_MINI_TRAIN_PCA, X_IMAGE_MINI_TRAIN]
DEVELOPMENT_SETS = [X_DEV, X_DEV_PCA, X_BLUR_DEV_PCA, X_RED_DEV_PCA, BLUR_RED_DEV_PCA, X_BLUE_DEV_PCA,
                    BLUR_BLUE_DEV_PCA, X_GREEN_DEV_PCA, BLUR_GREEN_DEV_PCA, X_IMAGE_DEV]
SET_DESCRIPTIONS = ['No_PCA', 'PCA', 'Blur_PCA', 'Red_PCA', 'Blur_Red_PCA', 'Blue_PCA', 'Blur_Blue_PCA',
                    'Green_PCA', 'Blur_Green_PCA', 'Image_Contour']


# In[ ]:


KNN_MODEL = KNeighborsClassifier(n_neighbors=1)
print("K Nearest Neighbors Classifier:")


def run_knn(model, training, development, description):
    scores = []
    for train, dev in zip(training, development):
        model.fit(train, Y_MINI_TRAIN)
        scores.append("{:.1%}".format(model.score(dev, Y_DEV)))

    df_knn = pd.DataFrame(columns=['Data Description', 'Accuracy'])
    i = 0
    for desc, score in zip(description, scores):
        df_knn.loc[i] = [desc, score]
        i += 1
    return df_knn


run_knn(KNN_MODEL, TRAINING_SETS, DEVELOPMENT_SETS, SET_DESCRIPTIONS)


# In[ ]:


# Classification report and error analysis on blurred data

KNN_MODEL.fit(X_BLUR_MINI_TRAIN_PCA, Y_MINI_TRAIN)
DEV_PREDICTED_LABELS = KNN_MODEL.predict(X_BLUR_DEV_PCA)
print(classification_report(Y_DEV, DEV_PREDICTED_LABELS))


# For our classification task it seems we should focus on precision and f1-score as a few false positives would not cause a great deal of harm unlike if we were predicting bank fraud or sick patient detection. We can see that a few of our letters stand out as being easier to predict than others. These include C, F, G and M to name a few that have an f1 score falling between 0.80 and 0.85. The outright winner would be the 'nothing' sign with an f1 score of 0.93. This is a fairly easy one to classify though as it means no hand is found in the image. As far as letters where our model struggled, A, U, V and W all had low f1 scores which is interesting because the images for V and W seem fairly easy to distinguish compared to others. As we move towards signing full sentences instead of just predicting images, we can likely use our image classification along with some kind of 'bag of letters' model to determine which letters are likely to come after each other. 

# #### 5.2 Bagged Trees
# 
# This section covers the bagged trees classifier which is the bagging classifier with a decision tree as the underlying estimator. There are two main hyperparameters to tune for a bagged trees classifier: 1) the size of the underlying decision tree, and 2) the number of estimators used for bagging. 
# 
# In the first code block, we train the bagged trees classifer with both the dataset with original dimensions and the dataset with reduced dimensions without tuning any parameters aside from setting the random state. As can be seen, the bagged trees classifer does much worse with dimensionality reduction. Therefore, we proceed to tune the bagged trees classifier without dimensionality reduction.
# 
# In the second code block, we tune the bagged trees classifier by varying: 1) the size of the underlying decision tree, and 2) the number of estimators used for bagging. The parameters we try for tree size include (100, 500, and 800) and number of estimators include (10, 20, 30, 50). From the output result, we conclude that limiting the size of the underlying tree i.e. pruning does not improve accuracy. On the other hand, as the number of estimators increases the accuracy rate increases accordingly. We proceed to focus on tuning the number of estimators without limiting the size of the underlying decision tree.
# 
# In the final code block, we tune the bagged trees classifier by trying a few more parameters for the number of estimators, including (50, 70, 80, 100). As evident from the results, accuracy increases steadily as the number of estimators increases until it starts to decrease again after the number of estimators reaches 80. The best accuracy of 87% is achieved with 80 estimators.

# In[ ]:


def run_bagged_trees():
    # base bagged trees classifier with dimensionality reduction
    bg = BaggingClassifier(random_state=0)
    bg.fit(X_MINI_TRAIN_PCA, Y_MINI_TRAIN)
    print("Bagged Trees Classifier:\n")
    print("Accuracy with PCA dimensionality reduction:", "{:.1%}".format(bg.score(X_DEV_PCA, Y_DEV)))

    # base bagged trees classifier without dimensionality reduction
    bg = BaggingClassifier(random_state=0)
    bg.fit(X_MINI_TRAIN, Y_MINI_TRAIN)
    print("Accuracy without PCA dimensionality reduction:", "{:.1%}".format(bg.score(X_DEV, Y_DEV)))
    return
    

run_bagged_trees()


# In[ ]:


# Bagged trees tuning by varying size of underlying decision tree (leaf_nodes)
# and number of estimators for bagging.
def make_decision_tree():
    leaf_nodes = [100, 500, 800]
    estimators = [10, 20, 30, 50]
    num_leaf_nodes = len(leaf_nodes)
    num_estimators = len(estimators)

    df1 = pd.DataFrame(index=range(num_leaf_nodes*num_estimators),
                       columns=["leaf_nodes", "estimators", "accuracy"])
    for i in range(num_leaf_nodes):
        dt = DecisionTreeClassifier(max_leaf_nodes=leaf_nodes[i])
        for j in range(num_estimators):
            bg = BaggingClassifier(
                base_estimator=dt, n_estimators=estimators[j], random_state=0)
            bg.fit(X_MINI_TRAIN, Y_MINI_TRAIN)
            accuracy = bg.score(X_DEV, Y_DEV)
            df1.loc[(i*num_estimators)+j] = [leaf_nodes[i],
                                             estimators[j], accuracy]

    return df1


BAGGING_DF = make_decision_tree()


# In[ ]:


def plot_bagging(df):
    '''Plot number of estimators vs accuracy with different number of leaf nodes
    for the underlying decision tree note that the lines for 500 leaf nodes and
    800 leaf nodes completely overlap each other.'''
    plt.plot(df.loc[0:3]["estimators"], df.loc[0:3]["accuracy"], color='steelblue',
             marker='.', linestyle='dotted', label="Bagging w/100 Leaf Nodes")
    plt.plot(df.loc[4:7]["estimators"], df.loc[4:7]["accuracy"], color='green',
             marker='.', linestyle='dotted', label="Bagging w/500 Leaf Nodes")
    plt.plot(df.loc[8:11]["estimators"], df.loc[8:11]["accuracy"], color='red',
             marker='.', linestyle='dotted', label="Bagging w/800 Leaf Nodes")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    return


plot_bagging(BAGGING_DF)


# In[ ]:


def run_bagged_trees2():
    '''As seen above varying size of leaf_nodes does not seem to impact the accruracy much
    in contrast number of estimators for bagging seems to have a direct impact on accuracy
    here we try a few more parameters for number of estimators.'''
    estimators = [50, 70, 80, 100]
    num_estimators = len(estimators)

    df = pd.DataFrame(index=range(num_estimators),
                       columns=["estimators", "accuracy"])
    for j in range(num_estimators):
        bg = BaggingClassifier(n_estimators=estimators[j], random_state=0)
        bg.fit(X_MINI_TRAIN, Y_MINI_TRAIN)
        accuracy = bg.score(X_DEV, Y_DEV)
        df.loc[j] = [estimators[j], accuracy]

    return df


BAGGING_DF2 = run_bagged_trees2()


# In[ ]:


def plot_bagging2(df):
    # plot number of estimators vs accuracy without no limites on size of underlying decision tree
    plt.plot(df["estimators"], df["accuracy"], color='steelblue', marker='.', linestyle='dotted', markersize=8)
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.show()
    return
    
    
plot_bagging2(BAGGING_DF2)


# In this section we perform error analysis for the bagged tree classifier with number of estimators set to 50.   Since accuracy hovers around 85% for estimators equal to or above 50, we picked this over a larger number of estimators to run things more efficiently.   From the confusion matrix, we can see that for this classifier the most often mis-labeled pairs of hand-signs include "B" with "A" and "D" with "C".  Consistent with the results of the confusion matrix, we see from the classification report that the hand-sign "B" has the lowest F1 score among all the symbols.

# In[ ]:


def bagging_cm():
    # output confusion matrix for bagged tree classifier with n_estimators = 50
    bg = BaggingClassifier(n_estimators=50, random_state=0)
    bg.fit(X_MINI_TRAIN, Y_MINI_TRAIN)
    predicted_labels = bg.predict(X_DEV)
    cm = confusion_matrix(Y_DEV, predicted_labels)
    plot_confusion_matrix_with_default_options(
        y_pred=predicted_labels,
        y_true=Y_DEV,
        classes=LETTERS)

    # Print total number of incorrect predictions for each digit.
    print("Number of incorrect predictions for each hand-sign:",
          cm.sum(axis=1) - cm.diagonal(), "\n")

    # Print maximum number of incorrect predictions for each digit.
    print("Maximum number of incorrect predictions for each hand-sign:",
          cm[np.where(cm != cm.diagonal())].reshape(29, 28).max(axis=1), "\n")
    
    print(classification_report(Y_DEV, predicted_labels))
    return


bagging_cm()


# #### 5.3 Support Vector Machines
# 
# Next we will take a look at the support vector machine algorithm to predict our data. We will run the RGB and blurred data through it along with transformations through PCA to determine which data set has the highest accuracy. From there we will tune the parameters C and gamma. Support vector machines work by placing hyperplanes through the data in N-dimensional space when a simple linear line can not separate the data.

# In[ ]:


SVC_MODEL = SVC(random_state=0)
print("Support Vector Machine:")


def run_svc(training, development, description):
    scores = []
    for train, dev in zip(training, development):
        SVC_MODEL.fit(train, Y_MINI_TRAIN)
        scores.append("{:.1%}".format(SVC_MODEL.score(dev, Y_DEV)))

    df_svc = pd.DataFrame(columns=['Data Description', 'Accuracy'])
    i = 0
    for desc, score in zip(description, scores):
        df_svc.loc[i] = [desc, score]
        i += 1
    return df_svc


run_svc(TRAINING_SETS, DEVELOPMENT_SETS, SET_DESCRIPTIONS)


# In[ ]:


def fit_and_tune_svc():
    c = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    gamma = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    params_list = {'C': c, 'gamma': gamma}
    clf = SVC(kernel='rbf')
    clf_name = "SVM"

    return fit_and_tune(
        X_MINI_TRAIN_PCA,
        X_DEV_PCA,
        clf,
        params_list,
        clf_name,
        "with PCA",
        1,
        1)


fit_and_tune_svc()


# After performing a grid search we have found the optimal parameters for C and gamma. C, also known as regularization, will choose a smaller-margin hyperplane to make sure all the data is classified correctly for large values of C. For small values of C, it will look for a larger-margin hyperplane even if that means misclassifying points. Our model found a C value somewhere in the middle to be optimal for our case. Next we looked at the gamma parameter. A low gamma means points far from the likely separation line are considered in the calculation while a high gamma means points close to the separation line are considered in the calculation. Our optimal gamma value was fairly low which could mean our data was sparse and far from the decision boundary. After performing the gridsearch on the these two parameters, we were able to increase our accuracy from 60.9% up to 67.0%.
# 
# Based off the confusion matrix shown above, two of the most confused signs are C & D, the model incorrectly predicts them a total of 7 times. A & E also have a high count of incorrect predictions at 9 due to the signs being very similar. They both make fists with the only difference being E tucking in the thumb and A not tucking. Based off the signs, I would expect R & U to be confused frequently as the only differnce between the two signs is R crosses the index and middle finger while U does not. However, based off the confusion matrix the signs are only confused for each other a total of 3 times. I think much of the confusion boiled down to darker photos and not being able to distinguish the lines of the fingers. Maybe increasing the brightness would be a further option for us to consider in future iterations of this model.

# ### 6. Convolutional Neural Network
# 
# The following section shows another methodology we tried out. We were able to use Kaggle's GPU-enabled kernels to train a model in less than 15 minutes.

# In[ ]:


CUSTOM_TEST_DIR = '../input/asl-alphabet-test/asl-alphabet-test'

TARGET_SIZE = (64, 64)
TARGET_DIMS = (64, 64, 3)  # add channel for RGB
N_LETTERS = len(LETTERS)
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 64

# Model saving for easier local iterations
MODEL_DIR = '.'
MODEL_PATH = MODEL_DIR + '/cnn-model.h5'
MODEL_WEIGHTS_PATH = MODEL_DIR + '/cnn-model.weights.h5'
MODEL_SAVE_TO_DISK = getenv('KAGGLE_WORKING_DIR') != '/kaggle/working'

print('Save model to disk? {}'.format('Yes' if MODEL_SAVE_TO_DISK else 'No'))


# ### 6.1 Data Processing Set-Up
# 
# In the next snippet, I make a generator for use by Keras. The `make_generator` function is versatile enough to be used for setting up a generator for training, validation, prediction, and testing.

# In[ ]:


def preprocess_image(image):
    '''Function that will be implied on each input. The function
    will run after the image is resized and augmented.
    The function should take one argument: one image (Numpy tensor
    with rank 3), and should output a Numpy tensor with the same
    shape. This preprocessor tries to simplify the image by finding
    its edges.'''
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobely

def make_generator(options):
    '''Creates two generators for dividing and preprocessing data.'''
    validation_split = options.get('validation_split', 0.0)
    preprocessor = options.get('preprocessor', None)
    data_dir = options.get('data_dir', TRAIN_DIR)

    augmentor_options = {
        'samplewise_center': True,
        'samplewise_std_normalization': True,
    }
    if validation_split is not None:
        augmentor_options['validation_split'] = validation_split

    if preprocessor is not None:
        augmentor_options['preprocessing_function'] = preprocessor

    flow_options = {
        'target_size': TARGET_SIZE,
        'batch_size': BATCH_SIZE,
        'shuffle': options.get('shuffle', None),
        'subset': options.get('subset', None),
    }

    data_augmentor = ImageDataGenerator(**augmentor_options)
    return data_augmentor.flow_from_directory(data_dir, **flow_options)


# ### 6.1 Model Specification
# 
# The model used here is taken from a [Kaggle kernel called Running Kaggle Kernels with a GPU](https://www.kaggle.com/grassknoted/asl-alphabet), and is an example of a convolutional neural network. It is made up of 12 layers, as is diagrammed below. I've provided a helper function below to load the model from disk so that this part of the notebook can be run without rebuilding the model each time.

# In[ ]:


def load_model_from_disk():
    '''A convenience method for re-running certain parts of the
    analysis locally without refitting all the data.'''
    model_file = Path(MODEL_PATH)
    model_weights_file = Path(MODEL_WEIGHTS_PATH)

    if model_file.is_file() and model_weights_file.is_file():
        print('Retrieving model from disk...')
        model = load_model(model_file.__str__())

        print('Loading CNN model weights from disk...')
        model.load_weights(model_weights_file)
        return model

    return None


CNN_MODEL = load_model_from_disk()
REPROCESS_MODEL = (CNN_MODEL is None)

print('Need to reprocess? {}'.format(REPROCESS_MODEL))


# In this part, we build out the convolutional neural network. Again, the model building process is skipped if the model has already been saved to disk.

# In[ ]:


def build_model(save=False):
    print('Building model afresh...')

    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=1,
                     activation='relu', input_shape=TARGET_DIMS))
    model.add(Conv2D(64, kernel_size=5, strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(N_LETTERS, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    if save:
        model.save(MODEL_PATH)

    return model


if REPROCESS_MODEL:
    CNN_MODEL = build_model(save=MODEL_SAVE_TO_DISK)

print_summary(CNN_MODEL)


# In this next section, the model is fitted against the data, splitting the data into training and validation sets.

# In[ ]:


def make_generator_for(subset):
    '''Create a generator for the training or validation set.'''
    generator_options = dict(
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        subset=subset,
        preprocessor=preprocess_image,
    )
    return make_generator(generator_options)


def fit_model(model, train_generator, val_generator, save=False):
    '''Fit the model with the training and validation generators.'''
    history = model.fit_generator(
        train_generator, epochs=5, validation_data=val_generator)

    if save:
        model.save_weights(MODEL_WEIGHTS_PATH)

    return history


CNN_TRAIN_GENERATOR = make_generator_for('training')
CNN_VAL_GENERATOR = make_generator_for('validation')

def reprocess_model():
    start_time = time.time()
    history = fit_model(CNN_MODEL, CNN_TRAIN_GENERATOR,
                        CNN_VAL_GENERATOR, save=MODEL_SAVE_TO_DISK)
    print('Fitting the model took ~{:.0f} second(s).'.format(
        time.time() - start_time))
    return history


def show_cnn_model_weights():
    columns = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4']
    return pd.DataFrame(data=[x.shape for x in CNN_MODEL.weights], columns=columns)

HISTORY = None
if REPROCESS_MODEL:
    HISTORY = reprocess_model()
show_cnn_model_weights()


# Here we show the "history" of the fitting process, which shows the loss and the accuracy of the fit.

# In[ ]:


# HISTORY is only available if the model was run this time around,
# which won't be the case when the model is read in from a file, so
# we just skip it in that case.
if HISTORY:
    print('Final Accuracy: {:.2f}%'.format(HISTORY.history['acc'][4] * 100))
    print('Validation set accuracy: {:.2f}%'.format(
        HISTORY.history['val_acc'][4] * 100))


# ### 6.2 Validation Against Real-World Data
# 
# The data provided in the ASL Alphabet data set is very much contrived. It's obvious that the images are made with one person's hand, in basically one environment. Because if this, it seemed like a good idea to validate that the models were not overfitting to images in this controlled environment. Below you can see a video compilation of all the "A" images.

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<div align="middle">\n    <video width="80%" controls>\n        <source src="https://s3-us-west-2.amazonaws.com/danrasband-w207/A.mp4" type="video/mp4">\n    </video>\n</div>')


# Below, we evaluate the CNN model against a set of images made with various backgrounds and with a different person's hand to see how well the model works on a related, but different, set of images.

# In[ ]:


def evaluate_model(generator):
    start_time = time.time()
    evaluations = CNN_MODEL.evaluate_generator(generator)
    for i in range(len(CNN_MODEL.metrics_names)):
        print("{}: {:.2f}%".format(
            CNN_MODEL.metrics_names[i], evaluations[i] * 100))
    print('Took {:.0f} seconds to evaluate this set.'.format(
        time.time() - start_time))

    start_time = time.time()
    predictions = CNN_MODEL.predict_generator(generator)
    print('Took {:.0f} seconds to get predictions on this set.'.format(
        time.time() - start_time))

    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes
    return dict(y_pred=y_pred, y_true=y_true)


def evaluate_validation_dataset():
    gen_options = dict(
        validation_split=0.1,
        data_dir=TRAIN_DIR,
        shuffle=False,
        subset='validation',
        preprocessor=preprocess_image,
    )
    val_gen = make_generator(gen_options)
    return evaluate_model(val_gen)


def evaluate_test_dataset():
    gen_options = dict(
        validation_split=0.0,
        data_dir=CUSTOM_TEST_DIR,
        shuffle=False,
        preprocessor=preprocess_image,
    )
    test_gen = make_generator(gen_options)
    return evaluate_model(test_gen)


# In[ ]:


CNN_VALIDATION_SET_EVAL = evaluate_validation_dataset()


# The model performs quite well on the original validation set, as seen below:

# In[ ]:


print(classification_report(**CNN_VALIDATION_SET_EVAL, target_names=LETTERS))


# In[ ]:


with sns.axes_style('ticks'):
    plot_confusion_matrix_with_default_options(
        **CNN_VALIDATION_SET_EVAL, classes=LETTERS)


# In[ ]:


CNN_TEST_SET_EVAL = evaluate_test_dataset()


# Running predictions on another set of data, however, shows that the feature engineering is insufficient to have a reasonable success rate, though the predictions are better than pure chance. In fact, it's apparent that there are a few letters that do extremely well, while others struggle. This may be an artifact of where those particular images were taken, though further analysis would have to be made to determine the exact cause.

# In[ ]:


print(classification_report(**CNN_TEST_SET_EVAL, target_names=LETTERS))


# In[ ]:


with sns.axes_style('ticks'):
    plot_confusion_matrix_with_default_options(
        **CNN_TEST_SET_EVAL, classes=LETTERS)


# ### 7. Other Classifiers
# 
# This section includes other classifiers we explored but did not tune in depth for this project.
# 
# #### Random forest

# In[ ]:


def fit_and_tune_random_forest():
    clf = RandomForestClassifier()
    clf_name = "Random_Forest"
    params_list = {
        'n_estimators': [10, 20, 50, 100, 200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    return fit_and_tune(
        X_MINI_TRAIN_PCA,
        X_DEV_PCA,
        clf,
        params_list,
        clf_name,
        "with PCA",
        0,
        0)


fit_and_tune_random_forest()


# #### Naive Bayes
# 
# Here we briefly look at performance of Naive Bayes on the contoured data. We used a Bernoulli NB as the data is contoured and has only 2 levels of pixels. NaiveBayes seems to work better on the contoured processed images better. Yet the accuracy at <45% is low 

# In[ ]:


def run_naive_bayes(train, dev):
    clf = BernoulliNB()
    params_list = {'alpha': [0.0001, 0.001, 0.01, 1, 10, 100]}
    clf_name = "Bernoulli NB"
    fit_and_tune(train, dev, clf, params_list, clf_name, "Contoured data", 0, 0)
    return


run_naive_bayes(X_IMAGE_MINI_TRAIN, X_IMAGE_DEV)


# ### 8. Conclusion
# 
# Using 29 different classes, we looked to take an image of a hand and predict the letter or symbol it was trying to sign. When plotting out all the images, we realized the images varied greatly ranging from what was in the background, the brightness to the angles of the hand. To combat some of this noise, we used filtering techniques such as gaussian blurs, cropping, edge detection and even breaking the images down by their red, green and blue pixels. After this initial processing, we used PCA to reduce the dimensionality of our images down to 110 components which explained 95% of the variance.
# 
# The models we used first for our predictions were K Nearest Neighbors, Bagged Trees, and Support Vector Machines. Out of these 3 models, Bagged Trees performed the best with an accuracy of 87% with the number of estimators set to 80.   K Nearest Neighbors performed the second with an accuracy of around 70%. We found that applying PCA to reduce the dimensionality and also performing a gaussian blur increased the accuracy over just using our original unfiltered data. The least accurate model was SVM at around 60%. The SVM's accuracy was also greatly helped by transformation through PCA. For future iterations, we would like to explore random forests, naive bayes and multilayer perceptrons more in depth.
# 
# We next tried a convolutional neural network. It had good results with some minor preprocessing (converting images using the Sobel method). When this model was applied to similar images from a less contrived environment, it still performed better than chance, but had much lower accuracy than when run on the validation set. In fact, it was only about half as good.
# 
# Lastly, we spent time looking at classifying the images with Naive Bayes and Multi-layer Perceptron classifiers, but eventually pulled the Multi-layer Perceptron classifier because of how long it was taking to run.
