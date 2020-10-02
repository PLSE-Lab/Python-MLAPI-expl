#!/usr/bin/env python
# coding: utf-8

# # Introduction to GroupKFold

# In this kernel, I will be performing **scikit-learn's** `GroupKFold`. Ideally, since the data is severely imbalanced, we should implement `Stratified GroupKFold`, but its not implemented in the existing libraries; if one is interested, you can refer to [the link](https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation) here for this implementation.
# 
# The reason I think `GroupKFold` might be better than normal `KFold` is as follows: Quoting from **scikit-learn's** [website](https://scikit-learn.org/stable/modules/cross_validation.html), it says: 
# 
# **The Independent and identically distributed (i.i.d.) assumption is broken if the underlying generative process yield groups of dependent samples.**
# 
# **Such a grouping of data is domain specific. An example would be when there is medical data collected from multiple patients, with multiple samples taken from each patient. And such data is likely to be dependent on the individual group. In our example, the patient id for each sample will be its group identifier.**
# 
# **In this case we would like to know if a model trained on a particular set of groups generalizes well to the unseen groups. To measure this, we need to ensure that all the samples in the validation fold come from groups that are not represented at all in the paired training fold.**
# 
# 
# **GroupKFold is a variation of k-fold which ensures that the same group is not represented in both testing/validation and training sets. For example if the data is obtained from different subjects with several samples per-subject and if the model is flexible enough to learn from highly person specific features it could fail to generalize to new subjects. `GroupKFold` makes it possible to detect this kind of overfitting situations.**

# Do note this is my first time implementing this, and there might be things that I overlooked; Please do point out if you find any errors in the pipeline or logic flow.

# # Installing Libraries

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os 
import re
import math
from matplotlib import pyplot as plt
from math import ceil
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn import model_selection

import tensorflow as tf
import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

from kaggle_datasets import KaggleDatasets

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# **References:**
# 1. [Parul's Awesome EDA kernel](https://www.kaggle.com/parulpandey/melanoma-classification-eda-starter)
# 2. [Tarun's PlantPathology2020 awesome kernel](https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models/notebook)
# 3. [Wei Hao's awesome kernel](https://www.kaggle.com/khoongweihao/siim-isic-multiple-model-training-inference/input)
# 4. [Abhishek Thakur's kernel](https://www.kaggle.com/abhishek/melanoma-detection-with-pytorch)
# 5. [Chris's Kernel](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96/data)

# # Defining the paths

# In[ ]:


# Defining data path
train_images_dir = '../input/siim-isic-melanoma-classification/train/'
test_images_dir = '../input/siim-isic-melanoma-classification/test/'
train_csv = '../input/siim-isic-melanoma-classification/train.csv'
test_csv  = '../input/siim-isic-melanoma-classification/test.csv'
sample_submission = '../input/siim-isic-melanoma-classification/sample_submission.csv'

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_df.head()

print("The unique number of patiend_ids are {}".format(train_df['patient_id'].nunique()))


# From the above dataframe, it is not immediately obvious that one unique patient may have multiple **image_names** attached to them because the dataframe is not sorted according to the `patiend_id`. We can use `train_df['patient_id'].nunique()` to check that there are 2056 unique `patient_ids` out of a whooping 33126 rows. This means that many patients have multiple images. And as I mentioned earlier, when we do `KFold` splitting, let's say there are 20 images for patient 1, we may have patient 1's data/images in both the training and validation set. For example, in the splitting process, there are 15 images of patient 1 in the training set, and there are 5 images in the validation set; then this may not be ideal since the model has already seen 15 of the images for patient 1 and can easily remember features that are **unique** to patient 1, and therefore predict well in the validation set for the same patient 1. Therefore, this cross validation method may give over optimistic results and fail to generalize well to more unseen images.

# # Splitting the dataset according to GroupKFold

# In[ ]:


group_by_patient_id = train_df.groupby(['patient_id', 'image_name']) 
group_by_patient_id.first()


# As you can see here, it is indeed the case that one patient can have multiple images. We will do the splitting below.

# In[ ]:


groups_by_patient_id_list = train_df['patient_id'].copy().tolist()
# the below code should work better in fact
# groups_by_patient_id_list = np.array(train_df['patient_id'].values)

y_labels = train_df["target"].values
# x_train = train_df[["image_name","patient_id","sex","age_approx","anatom_site_general_challenge"]]
# y_train = train_df[["target"]]


# Here I created 5 folds and appended the 10 dataframes into a list. 

# In[ ]:


n_splits = 5
gkf = GroupKFold(n_splits = 5)

result = []   
for train_idx, val_idx in gkf.split(train_df, y_labels, groups = groups_by_patient_id_list):
    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]
    result.append((train_fold, val_fold))
    
train_fold_1, val_fold_1 = result[0][0],result[0][1]
train_fold_2, val_fold_2 = result[1][0],result[1][1]
train_fold_3, val_fold_3 = result[2][0],result[2][1]
train_fold_4, val_fold_4 = result[3][0],result[3][1]
train_fold_5, val_fold_5 = result[4][0],result[4][1]



# just to check if it works as intended
sample = train_fold_1.groupby("patient_id")
sample.get_group("IP_0147446")
sample.get_group("IP_0147446").count()
# sample2 = val_fold_1.groupby("patient_id")
# sample2.get_group("IP_0063782")


# # Modelling

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

# Configuration
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
image_size = 256
EPOCHS = 3


# `format_path_train` and `format_path_test` merely takes in an image name, and returns the path to the image.

# In[ ]:


def format_path_train(img_name):
    return GCS_PATH + '/jpeg/train/' + img_name + '.jpg'

def format_path_test(img_name):
    return GCS_PATH + '/jpeg/test/' + img_name + '.jpg'


# As you can see here, we used `format_path_train` and `.apply` to get the image path.

# In[ ]:


train_paths_fold_1 = train_fold_1.image_name.apply(format_path_train).values
val_paths_fold_1 = val_fold_1.image_name.apply(format_path_train).values

train_labels_fold_1 = train_fold_1.target.values
val_labels_fold_1 = val_fold_1.target.values

test_paths = test_df.image_name.apply(format_path_test).values


# In[ ]:


def decode_image(filename, label=None, image_size=(image_size, image_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, size = image_size)
    
    if label is None:
        return image
    else:
        return image, label

# def data_augment(image, label=None):
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
    
#     if label is None:
#         return image
#     else:
#         return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
#     image = tf.image.random_saturation(image, lower = 1, upper = 3)
#     image = tf.image.adjust_brightness(image, delta = 0.3)
    image = tf.image.random_contrast(image, lower = 1, upper = 2)
    if label is None:
        return image
    else:
        return image, label


# ## Explanation of `decode_image`

# To understand what the function `decode_image` does: we will use a sample filename to test it out.

# In[ ]:


train_paths_fold_1


# In[ ]:


sample_filename = 'gs://kds-dd1bd3efd29ee7a66da730e2ae6f3007dcccabecdd1e2263d5f9e88b/jpeg/train/ISIC_2637011.jpg' 
sample_label = 0
image_size = 256

# 1. tf.io_read_file takes in a Tensor of type string and outputs a ensor of type string. 
#    Reads and outputs the entire contents of the input filename. 
bits = tf.io.read_file(sample_filename)

# 2. Decode a JPEG-encoded image to a uint8 tensor. You can also use tf.io.decode_jpeg but according to 
#    tensorflow's website, it might be cleaner to use tf.image.decode_jpeg
image = tf.image.decode_jpeg(bits, channels=3)

image.shape  # outputs TensorShape([4000, 6000, 3])

# 3. image = tf.cast(image, tf.float32) / 255.0 is easy to understand, it takes in 
#    an image, and cast the image into the data type you want. Here we also normalized by dividing by 255.

image = tf.cast(image, tf.float32) / 255.0


# 4. image = tf.image.resize(image, image_size) is also easy to understand. We merely resize this image to the image_size we wish for.
#    take note in our function defined above, the argument image_size is a tuple already. So we must pass in a tuple of our desired image_size.
image = tf.image.resize(image, size = (image_size, image_size))

image.shape


# ## Explanation of `tf.data.Dataset`

# According to the tensorflow website: The `tf.data.Dataset` API supports writing descriptive and efficient input pipelines. Dataset usage follows a common pattern:
# 
# - Create a source dataset from your input data.
# - Apply dataset transformations to preprocess the data.
# - Iterate over the dataset and process the elements.
# 
# Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory.

# In[ ]:


# example from tensorflow's website
sample_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for element in sample_dataset:
    print(element)


# Printing out the first element of `tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1))` gives us **(<tf.Tensor: shape=(), dtype=string, numpy=b'gs://kds-c89313da1d85616eec461ab327fed61e1335defb486fb7729cf897b1/jpeg/train/ISIC_2637011.jpg'>, <tf.Tensor: shape=(), dtype=int64, numpy=0>)**

# As you can see, we printed out the first `data` in the `dataset`. The returned `data` is actually a tuple of length 2. Why length 2? Because the first element of the `data` contains the image's information, but currently its still stored as a String format. The second element of the `data` returns the label which in this case the label is 0 (non-malignant).

# In[ ]:


dataset = tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1))
for data in dataset:
    print(len(data))
    print(data[0])
    print(data[1])   
    break


# Our next step is to decode the image using our functions defined earlier. As you can see, we used the `map` function and used our `decode_image` to make our image data into the "Tensor Numpy Array Format".

# In[ ]:


dataset = tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1)).map(decode_image, num_parallel_calls=AUTO)
for data in dataset:
    print(len(data))
    print(data[0])
    print(data[1])
    break


# In[ ]:


sample_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
sample_dataset = sample_dataset.repeat(3)
list(sample_dataset)


# We will explan `repeat()` and `batch()` together. When specified a batch_size,then `.batch(32)` will dictate 32 training examples and will undergo training. Using `.repeat()` we can specify the number of times we want the dataset to be iterated. If no parameter is passed it will loop forever, usually is good to just loop forever and directly control the number of epochs with a standard loop. [References](https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428)

# In[ ]:


# dataset = tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1)).map(decode_image, num_parallel_calls=AUTO).repeat()
# # for data in dataset:
# #     print(len(data))
# #     print(data[0])
# #     print(data[1])
# #     break


# In[ ]:


# dataset = tf.data.Dataset.from_tensor_slices((train_paths_fold_1, train_labels_fold_1)).map(decode_image, num_parallel_calls=AUTO).repeat().batch(32)
# # here it returns 32 images and its labels, because we specified our batch size to be 32! 
# for data in dataset:
#     print(data)
#     break


# ## Explanation of Data Augment

# Data Augmentation, as always, is helpful in training Neural Networks. In this particular competition, I believe tweaking the shades of the skin by using contrast, saturation and brightness etc may be helpful to generalize. Of course, we will include the good old horizontal and vertical flip as well. Below is a brief visualization of the augmentations that one can use in this pipeline.

# In[ ]:


image_folder_path = '../input/siim-isic-melanoma-classification/jpeg/train/'
chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0074542.jpg"))[:,:,::-1]
plt.imshow(chosen_image)


# In[ ]:


horizontal_flipped = tf.image.flip_left_right(chosen_image)
vertically_flipped = tf.image.flip_up_down(chosen_image)
adjusted_saturation = tf.image.adjust_saturation(chosen_image, saturation_factor = 2)
adjusted_brightness = tf.image.adjust_brightness(chosen_image, delta = 0.3)
adjusted_contrast = tf.image.adjust_contrast(chosen_image, contrast_factor = 2)


# In[ ]:


def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=ceil(len(img_matrix_list) / ncols), ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()


# In[ ]:


img_matrix_list = [chosen_image,horizontal_flipped,vertically_flipped,adjusted_saturation,adjusted_brightness,adjusted_contrast]
title_list = ["Original", "HorizontalFlipped", "VerticallyFlipped", "Saturated","Brightness","Contrast"]
plot_multiple_img(img_matrix_list, title_list, ncols = 3)


# ## Defining the dataset

# In[ ]:


# shuffle() (if used) should be called before batch() - we want to shuffle records not batches.
train_dataset_fold_1 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_1, train_labels_fold_1))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

# Generally we don't shuffle a test/val set at all - 
# only the training set (We evaluate using the entire test set anyway, right? So why shuffle?).
# https://stackoverflow.com/questions/56944856/tensorflow-dataset-questions-about-shuffle-batch-and-repeat
# https://stackoverflow.com/questions/49915925/output-differences-when-changing-order-of-batch-shuffle-and-repeat
valid_dataset_fold_1 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_1, val_labels_fold_1))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE))


# ## Defining our Learning Rate Scheduler function.

# In[ ]:


def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *                 lr_exp_decay**(epoch - lr_rampup_epochs                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
rng = [i for i in range(25 if EPOCHS<25 else EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# ## Model CheckPoint

# In[ ]:


STEPS_PER_EPOCH = train_labels_fold_1.shape[0] // BATCH_SIZE
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('GroupKFold.h5', monitor='val_loss', verbose=2, save_best_only=True)


# ## Define Model

# In[ ]:


def get_model():
    with strategy.scope():
        model = tf.keras.Sequential([
            efn.EfficientNetB3(
                input_shape=(256,256, 3),
                weights="imagenet",
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss = 'binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()])
    
    return model


# # Fold 1

# The code below is commented out as it was already defined earlier. But for the sake of completeness of each fold, I will include it below to make all folds look the same.

# In[ ]:


# train_paths_fold_1 = train_fold_1.image_name.apply(format_path_train).values
# val_paths_fold_1 = val_fold_1.image_name.apply(format_path_train).values

# train_labels_fold_1 = train_fold_1.target.values
# val_labels_fold_1 = val_fold_1.target.values

# train_dataset_fold_1 = (
#     tf.data.Dataset
#     .from_tensor_slices((train_paths_fold_1, train_labels_fold_1))
#     .map(decode_image, num_parallel_calls=AUTO)
#     .map(data_augment, num_parallel_calls=AUTO)
#     .repeat()
#     .shuffle(512)
#     .batch(BATCH_SIZE)
#     .prefetch(AUTO))

# valid_dataset_fold_1 = (
#     tf.data.Dataset
#     .from_tensor_slices((val_paths_fold_1, val_labels_fold_1))
#     .map(decode_image, num_parallel_calls=AUTO)
#     .batch(BATCH_SIZE)
#     .cache()
#     .prefetch(AUTO))


# In[ ]:


model_fold_1 = get_model()
history_1 = model_fold_1.fit(train_dataset_fold_1,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_1)


# In[ ]:


probs_fold_1 = model_fold_1.predict(test_dataset,verbose = 1)


# # Fold 2

# In[ ]:


train_paths_fold_2 = train_fold_2.image_name.apply(format_path_train).values
val_paths_fold_2 = val_fold_2.image_name.apply(format_path_train).values

train_labels_fold_2 = train_fold_2.target.values
val_labels_fold_2 = val_fold_2.target.values

train_dataset_fold_2 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_2, train_labels_fold_2))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valid_dataset_fold_2 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_2, val_labels_fold_2))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))


# In[ ]:


model_fold_2 = get_model()
history_2 = model_fold_2.fit(train_dataset_fold_2,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_2)


# In[ ]:


probs_fold_2 = model_fold_2.predict(test_dataset,verbose = 1) 


# # Fold 3

# In[ ]:


train_paths_fold_3 = train_fold_3.image_name.apply(format_path_train).values
val_paths_fold_3 = val_fold_3.image_name.apply(format_path_train).values

train_labels_fold_3 = train_fold_3.target.values
val_labels_fold_3 = val_fold_3.target.values

train_dataset_fold_3 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_3, train_labels_fold_3))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valid_dataset_fold_3 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_3, val_labels_fold_3))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))


# In[ ]:


model_fold_3 = get_model()
history_3 = model_fold_3.fit(train_dataset_fold_3,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_3)


# In[ ]:


probs_fold_3 = model_fold_3.predict(test_dataset,verbose = 1) 


# # Fold 4

# In[ ]:


train_paths_fold_4 = train_fold_4.image_name.apply(format_path_train).values
val_paths_fold_4 = val_fold_4.image_name.apply(format_path_train).values

train_labels_fold_4 = train_fold_4.target.values
val_labels_fold_4 = val_fold_4.target.values

train_dataset_fold_4 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_4, train_labels_fold_4))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valid_dataset_fold_4 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_4, val_labels_fold_4))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))


# In[ ]:


model_fold_4 = get_model()
history_4 = model_fold_4.fit(train_dataset_fold_4,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_4)


# In[ ]:


probs_fold_4 = model_fold_4.predict(test_dataset,verbose = 1)


# # Fold 5

# In[ ]:


train_paths_fold_5 = train_fold_5.image_name.apply(format_path_train).values
val_paths_fold_5 = val_fold_5.image_name.apply(format_path_train).values

train_labels_fold_5 = train_fold_5.target.values
val_labels_fold_5 = val_fold_5.target.values

train_dataset_fold_5 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths_fold_5, train_labels_fold_5))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valid_dataset_fold_5 = (
    tf.data.Dataset
    .from_tensor_slices((val_paths_fold_5, val_labels_fold_5))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))


# In[ ]:


model_fold_5 = get_model()
history_5 = model_fold_5.fit(train_dataset_fold_5,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset_fold_5)


# In[ ]:


probs_fold_5 = model_fold_5.predict(test_dataset,verbose = 1)


# In[ ]:


# sample_submission = '../input/siim-isic-melanoma-classification/sample_submission.csv'
# submission = pd.read_csv(sample_submission)
# submission['target'] = probs_fold_5   
# submission.head(20)
# submission.to_csv('submission_contrast_group_k_fold_5.csv', index=False)


# ## TODO
# Plot AUC-ROC curve for validation and see confusion matrix, can see if we are having a lot of false negatives. or is the model just blindly predicting 0.

# # Submission for GroupKFold

# In[ ]:


sample_submission = '../input/siim-isic-melanoma-classification/sample_submission.csv'
submission = pd.read_csv(sample_submission)
# fold1 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_1.csv")
# fold2 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_2.csv")
# fold3 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_3.csv")
# fold4 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_4.csv")
# fold5 = pd.read_csv("../input/ensemble5folds/submission_group_k_fold_5.csv")
# ensembled = (fold1['target'] + fold2['target']  + fold3['target'] + fold4['target'] + fold5['target'])/5
# submission['target'] = ensembled

ensembled = (probs_fold_1 + probs_fold_2 + probs_fold_3 + probs_fold_4 + probs_fold_5)/5
submission['target'] = ensembled
submission.head(20)

#submitting to csv
submission.to_csv('ensembled.csv', index=False)


# I got an LB score of 0.908 after ensembling.

# # Stratified GroupKFold

# **References:**
# 
# [1. https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation](https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation)
# 
# [2. https://www.kaggle.com/graf10a/siim-stratified-groupkfold-5-folds](https://www.kaggle.com/graf10a/siim-stratified-groupkfold-5-folds)

# In[ ]:


import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

def stratified_group_k_fold(X, y, groups, k, seed=None):
    """ https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in tqdm(sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])), total=len(groups_and_y_counts)):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


# In[ ]:


group_by_patient_id_array = np.array(train_df['patient_id'].values)
y_labels = train_df["target"].values


# In[ ]:


skf = stratified_group_k_fold(X=train_df, y=y_labels, groups=group_by_patient_id_array, k=5, seed=42)


# In[ ]:


def get_distribution(y_vals):
        y_distr = Counter(y_vals)
        y_vals_sum = sum(y_distr.values())
        return [f'{y_distr[i] / y_vals_sum:.5%}' for i in range(np.max(y_vals) + 1)]


# In[ ]:


distrs = [get_distribution(y_labels)]
index = ['training set']

for fold_ind, (dev_ind, val_ind) in enumerate(skf, 1):
    dev_y, val_y = y_labels[dev_ind], y_labels[val_ind]
    dev_groups, val_groups = group_by_patient_id_array[dev_ind], group_by_patient_id_array[val_ind]
    # making sure that train and validation group do not overlap:
    assert len(set(dev_groups) & set(val_groups)) == 0
    
    distrs.append(get_distribution(dev_y))
    index.append(f'training set - fold {fold_ind}')
    distrs.append(get_distribution(val_y))
    index.append(f'validation set - fold {fold_ind}')

display('Distribution per class:')
pd.DataFrame(distrs, index=index, columns=[f'Label {l}' for l in range(np.max(y_labels) + 1)])


# Each fold has almost the same percentages of 0's and 1's and this code given above allows the fact that unique `patient_id` values do not overlap between different folds.

# In[ ]:


df = train_df.copy()
df['fold'] = -1
df.head()


# In[ ]:


# somehow you need to redefine this skf line here for the .loc to work
skf = stratified_group_k_fold(X=train_df, y=y_labels, groups=group_by_patient_id_array, k=5, seed=42)
for fold_number, (train_idx, val_idx) in enumerate(skf):
    df.loc[val_idx, "fold"] = fold_number
    
df.to_csv("sgkfold.csv", index=False)


# # Training and Modelling

# Here I conveniently took the idea from [the grandmaster Abhishek Thakur](https://www.kaggle.com/abhishek/melanoma-detection-with-pytorch/data) way to modularize our training process.

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

def format_path_train(img_name):
    return GCS_PATH + '/jpeg/train/' + img_name + '.jpg'

def format_path_test(img_name):
    return GCS_PATH + '/jpeg/test/' + img_name + '.jpg'


# In[ ]:


image_size = 256

def decode_image(filename, label=None, image_size=(image_size, image_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, size = image_size) 

    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if label is None:
        return image
    else:
        return image, label


# In[ ]:


def train(fold_number):
    training_data_path = '../input/siim-isic-melanoma-classification/train/'
    df = pd.read_csv("/kaggle/working/sgkfold.csv")
    df_train = df[df.fold != fold_number].reset_index(drop=True)
    df_valid = df[df.fold == fold_number].reset_index(drop=True)
    df_train_path = df_train.image_name.apply(format_path_train).values
    df_val_path   = df_valid.image_name.apply(format_path_train).values
    df_train_labels = df_train.target.values
    df_val_labels   = df_valid.target.values
    
    AUTO = tf.data.experimental.AUTOTUNE
    # For tf.dataset
    BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    EPOCHS = 3    
    
    train_dataset = (tf.data.Dataset
    .from_tensor_slices((df_train_path, df_train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))
    
    valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((df_val_path, df_val_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO))
    
    def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
                   lr_min=0.00001, lr_rampup_epochs=5, 
                   lr_sustain_epochs=0, lr_exp_decay=.8):
        lr_max = lr_max * strategy.num_replicas_in_sync

        def lrfn(epoch):
            if epoch < lr_rampup_epochs:
                lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
            elif epoch < lr_rampup_epochs + lr_sustain_epochs:
                lr = lr_max
            else:
                lr = (lr_max - lr_min) *                     lr_exp_decay**(epoch - lr_rampup_epochs                                    - lr_sustain_epochs) + lr_min
            return lr
        return lrfn    
    
    lrfn = build_lrfn()
    STEPS_PER_EPOCH = df_train_labels.shape[0] // BATCH_SIZE
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('StratifiedGroupKFold.h5', monitor='val_loss', verbose=2, save_best_only=True)
    
    
    with strategy.scope():
        model = tf.keras.Sequential([
            efn.EfficientNetB3(
                input_shape=(256,256, 3),
                weights="imagenet",
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss = 'binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()])
    
    history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint,lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)
    return model


# # Test Paths

# In[ ]:


test_paths = test_df.image_name.apply(format_path_test).values
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 3  

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE))


# # Training using Stratified GroupKFold

# In[ ]:


# fold_1 = train(0)


# In[ ]:


# probs_fold_1 = fold_1.predict(test_dataset, verbose = 1)


# In[ ]:


# sample_submission = '../input/siim-isic-melanoma-classification/sample_submission.csv'
# submission = pd.read_csv(sample_submission)
# submission['target'] = probs_fold_1  
# submission.head(20)
# submission.to_csv('submission_stratified_group_k_fold_1.csv', index=False)


# **Work in progress...**

# In[ ]:


# fold_2 = train(1)


# In[ ]:


# probs_fold_2 = fold_2.predict(test_dataset, verbose = 1)


# In[ ]:


# fold_3 = train(2)


# In[ ]:


# probs_fold_3 = fold_3.predict(test_dataset, verbose = 1)


# In[ ]:


# fold_4 = train(3)


# In[ ]:


# probs_fold_4 = fold_4.predict(test_dataset, verbose = 1)


# In[ ]:


# fold_5 = train(4)


# In[ ]:


# probs_fold_5 = fold_5.predict(test_dataset, verbose = 1)

