#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is a simple CNN kernel to do binary classification if an image contains a dog or a cat. 1 for dog, 0 for cat. The competition itself finished long time ago, but I found this a good dataset to try to learn some basics of CNN classification.
# 
# I use the pre-trained ResNet50 model as a basis to try a little simple analysis of the given images, and try to different image augmentation methods from Keras. I was not familiar with any of this, so I tried most of them to see how they work and what is the modification to the original image they create.

# In[ ]:


import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import math
import PIL
from PIL import ImageOps
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K 
from sklearn.preprocessing import LabelEncoder

from tqdm.auto import tqdm
tqdm.pandas()


# Including the cat and dog images, and the pretrained ResNet model weights.

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


ls ../input/dogs-vs-cats/test1


# In[ ]:


#!ls ../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5


# In[ ]:


train_dir = "../input/dogs-vs-cats/train/train"
file_list = os.listdir(train_dir)
DOG = "dog"
CAT = "cat"
TRAIN_TOTAL = len(file_list)
labels = []
df_train = pd.DataFrame()


# Collect the image labels (cat/dog), width/height, and aspect ratio to take a look at the shapes. How far are they from a square shape that the ResNet expects as input?

# In[ ]:


get_ipython().run_cell_magic('time', '', 'idx = 0\nimg_sizes = []\nwidths = np.zeros(TRAIN_TOTAL, dtype=int)\nheights = np.zeros(TRAIN_TOTAL, dtype=int)\naspect_ratios = np.zeros(TRAIN_TOTAL) #defaults to type float\nfor filename in file_list:\n    if "cat" in filename.lower():\n        labels.append(CAT)\n    else:\n        labels.append(DOG)\n    img = PIL.Image.open(f"{train_dir}/{filename}")\n    img_size = img.size\n    img_sizes.append(img_size)\n    widths[idx] = img_size[0]\n    heights[idx] = img_size[1]\n    aspect_ratios[idx] = img_size[0]/img_size[1]\n    img.close()\n    idx += 1')


# In[ ]:


df_train["filename"] = file_list
df_train["cat_or_dog"] = labels
label_encoder = LabelEncoder()
df_train["cd_label"] = label_encoder.fit_transform(df_train["cat_or_dog"])
df_train["size"] = img_sizes
df_train["width"] = widths
df_train["height"] = heights
df_train["aspect_ratio"] = aspect_ratios
df_train.head()


# # Simple data exploration

# In[ ]:


df_train["aspect_ratio"].max()


# In[ ]:


df_train["aspect_ratio"].min()


# In[ ]:


max_idx = df_train["aspect_ratio"].values.argmax()
max_idx


# In[ ]:


df_train.iloc[max_idx]


# In[ ]:


filename = df_train.iloc[max_idx]["filename"]
img = PIL.Image.open(f"{train_dir}/{filename}")


# In[ ]:


### The Broadest Image in the Set


# In[ ]:


plt.imshow(img)


# In[ ]:


img.close()


# The above shows that the data can contain some other elements besides just cats and dogs. My guess is that this dataset was scraped somehow, and the Yahoo logo got left in there by mistake. Are there any other similar images in the set?

# In[ ]:


df_sorted = df_train.sort_values(by="aspect_ratio")


# In[ ]:


def plot_first_9(df_to_plot):
    plt.figure(figsize=[30,30])
    for x in range(9):
        filename = df_to_plot.iloc[x].filename
        img = PIL.Image.open(f"{train_dir}/{filename}")
        print(filename)
        plt.subplot(3, 3, x+1)
        plt.imshow(img)
        title_str = filename+" "+str(df_to_plot.iloc[x].aspect_ratio)
        plt.title(title_str)


# ## Tallest Images
# 
# This should now show the "tallest" images in the dataset:

# In[ ]:


plot_first_9(df_sorted)


# The first one is of multiple dogs in one, stacked vertically. The rest are just tight pics of cats and dogs.

# ## Widest Images
# 
# A look at the broaders/widest similar to the "tallest" ones was above:

# In[ ]:


df_sorted = df_train.sort_values(by="aspect_ratio", ascending=False)


# In[ ]:


plot_first_9(df_sorted)


# So there is another image that looks like it might have come from scraping some websites. The one with the pink flower/rose in it. Not sure what the third one is, looks a bit like a flea market sale outside. Similar to the stacked dog photo, there are also some with multiple cats next to each other as well.
# 
# Funny thing is, someone seems to have labeled each of these as on of cat or dog. It might be worth a closer look at the overall dataset, but I just drop those 3 strange ones for now:

# In[ ]:


df_sorted.drop(df_sorted.index[:3], inplace=True)


# And look again:

# In[ ]:


plot_first_9(df_sorted)


# It looks quite fine now.

# In[ ]:


df_train = df_sorted


# In[ ]:


df_train.dtypes


# Some basic attributes for training:
# 

# In[ ]:


#This batch size seemed to work without memory issues
batch_size = 32
#299 is the input size for some of the pre-trained networks. I think ResNet50 is actually 224x224 but I left this as 299 anyway.
img_size = 299 #TODO: 224
#I will try a few variations of training my model on top of ResNet, 5 seems to be enough to get results but leave some time to try the variants.
epochs = 7


# # Trying out the Keras Generators
# 
# First a generic function to create generators with different augmentation configurations:

# In[ ]:


from keras.applications.resnet50 import preprocess_input

def create_generators(validation_perc, shuffle=False, horizontal_flip=False, 
                      zoom_range=0, w_shift=0, h_shift=0, rotation_range=0, shear_range=0,
                     fill_zeros=False, preprocess_func=None):
    #the "nearest" mode copies image pixels on borders when shifting/rotation/etc to cover empty space
    fill_mode = "nearest"
    if fill_zeros:
        #with constant mode, we fill created empty space with zeros
        fill_mode = "constant"
        
    #rescale changes pixels from 1-255 integers to 0-1 floats suitable for neural nets
    rescale = 1./255
    if preprocess_func is not None:
        #https://stackoverflow.com/questions/48677128/what-is-the-right-way-to-preprocess-images-in-keras-while-fine-tuning-pre-traine
        #no need to rescale if using Keras in-built ResNet50 preprocess_func: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L157
        rescale = None

    train_datagen=ImageDataGenerator(
        rescale = rescale, 
        validation_split = validation_perc, #0.25, #subset for validation. seems to be subset='validation' in flow_from_dataframe
        horizontal_flip = horizontal_flip,
        zoom_range = zoom_range,
        width_shift_range = w_shift,
        height_shift_range=h_shift,
        rotation_range=rotation_range,
        shear_range=shear_range,
        fill_mode=fill_mode,
        cval=0,#this is the color value to fill with when "constant" mode used. 0=black
        preprocessing_function=preprocess_func
    )

    #Keras has this two-part process of defining generators. 
    #First the generic properties above, then the actual generators with filenames and all.
    train_generator=train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=train_dir,
        x_col="filename", #the name of column containing image filename in dataframe
        y_col="cat_or_dog", #the y-col in dataframe
        batch_size=batch_size, 
        shuffle=shuffle,
        class_mode="binary", #categorical if multiple. then y_col can be list or tuple also 
        #classes=lbls, #list of ouput classes. if not provided, inferred from data
        target_size=(img_size,img_size),
        subset='training') #the subset of data from the ImageDataGenerator definition above. The validation_split seems to produce these 2 values.

    valid_generator=train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=train_dir,
        x_col="filename",
        y_col="cat_or_dog",
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode="binary",
        #classes=lbls,
        target_size=(img_size,img_size), #gave strange error about tuple cannot be interpreted as integer
        subset='validation') #the subset of data from the ImageDataGenerator definition above. The validation_split seems to produce these 2 values.

    return train_generator, valid_generator, train_datagen


# In[ ]:





# ## For reference, the plain images, no transformation

# In[ ]:


train_generator, valid_generator, train_datagen = create_generators(0, False, False, 0, 0, 0)


# In[ ]:


train_generator.class_indices


# In[ ]:


class_map = {v: k for k, v in train_generator.class_indices.items()}


# In[ ]:


import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


# In[ ]:


def plot_batch_9():
    train_generator.reset()
    # configure batch size and retrieve one batch of images
    plt.clf() #clears matplotlib data and axes
    #for batch in train_generator:
    plt.figure(figsize=[30,30])
    batch = next(train_generator)
    for x in range(0,9):
    #    print(train_generator.filenames[x])
        plt.subplot(3, 3, x+1)
        plt.imshow(batch[0][x], interpolation='nearest')
        item_label = batch[1][x]
        item_label = class_map[int(item_label)]
        plt.title(item_label)

    plt.show()


# In[ ]:


plot_batch_9()


# So the above images are the "plain" versions directly from the training data, no transformations. They may look a bit "stretched" because Keras resizes them all to the given target shape (299,299 here) on loading. But no other augmentation transformations were done.
# 
# To illustrate this stretching, lets see a few plain images without stretching:

# In[ ]:


def show_img(idx):
    filename = df_train.iloc[idx]["filename"]
    img = PIL.Image.open(f"{train_dir}/{filename}")
    plt.imshow(img)
    img.close()


# In[ ]:


show_img(2)


# In[ ]:


show_img(1)


# ## Horizontal Flipping
# 
# So how does Keras apply the augmentation transformations? Lets try with only horizontal flipping enabled, and nothing else:

# In[ ]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0, 
                                                                    h_shift = 0)


# In[ ]:


plot_batch_9()


# Comparing to the above plotted plain images, we can see that it has now randomly picked to flip the image left or right. Others are unchanged. So it applies the transformation sometimes, but not always.

# ## Horizontal Shift
# 
# So what does the horizontal shift look like? Here we set width-shift to max 20%, combined with the horizontal flip transformation.

# In[ ]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0.2, 
                                                                    h_shift = 0)


# In[ ]:


plot_batch_9()


# It seems to pick one or both of the enabled transformations. Since it is random and I have not fixed a seed, it varies which one is applied above (between runs, so cannot say which one was applied now). Getting the original "non-augmented" seems rather rare, and likely much more rare if doing this with more transformations. 
# 
# The horizontal shifting is filled now with the image pixes on that side, on a random set of 0-20% of the width. This pixel duplication is the "nearest" method.
# 
# ## Multiple in one
# 
# Let's try with all the transformations I put in the function:

# In[ ]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0.2, 
                                                                    w_shift = 0.2, 
                                                                    h_shift = 0.2)


# In[ ]:


plot_batch_9()


# So now we have combinations of zooming, horizontal flipping, horizontal and vertical shifting. And the choice of which are applied varies over epochs.

# ## Fill with Zeros
# 
# To try and see some of the effects better, change to use "constant" mode and fill empty space with black:

# In[ ]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0.2, 
                                                                    w_shift = 0.2, 
                                                                    h_shift = 0.2,
                                                                   fill_zeros = True)


# In[ ]:


plot_batch_9()


# It seems clearer to see what happened. Not sure what is the impact on training.

# ## Rotation
# 
# Well, later I also added rotation as an option, so lets see:

# In[ ]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = False, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0, 
                                                                    h_shift = 0,
                                                                    fill_zeros = True,
                                                                   rotation_range=20)


# In[ ]:


plot_batch_9()


# ## Shearing
# 
# And shearing. This one was a bit unclear to me what it does:
# 

# In[ ]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = False, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0, 
                                                                    h_shift = 0,
                                                                    fill_zeros = True,
                                                                   shear_range=20)


# In[ ]:


plot_batch_9()


# So it vertically "tilts" the image. In this case I set 20 for 0-20% "tilt". What about a bigger value, what is the effect:

# In[ ]:


train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0, 
                                                                    shuffle = False, 
                                                                    horizontal_flip = False, 
                                                                    zoom_range = 0, 
                                                                    w_shift = 0, 
                                                                    h_shift = 0,
                                                                    fill_zeros = True,
                                                                   shear_range=90)


# In[ ]:


plot_batch_9()


# It seems like smaller shear values might make sense sometimes but the bigger ones can produce a bit of a mess. I tried various values to see if I could make the shearing work also horizontally. No luck. I guess it only support vertical "shearing". There is probably some good reason, I just dont know it.

# ## ResNet50 Preprocessing function
# 
# What dos the preprocessing function from Keras ResNet50 model itself do?

# In[ ]:


from keras.applications import resnet50

train_generator, valid_generator, train_datagen = create_generators(validation_perc = 0.2, 
                                                                    shuffle = True, 
                                                                    horizontal_flip = True, 
                                                                    zoom_range = 0.2, 
                                                                    w_shift = 0.2, 
                                                                    h_shift = 0.2,
                                                                    fill_zeros = True,
                                                                    preprocess_func = resnet50.preprocess_input,
                                                                   shear_range=10)


# In[ ]:


plot_batch_9()


# It all looks rather psychedelic. Probably because the [docs](https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L157)
# say the function changes values to scale of -1 to 1, and they get clipped on plotting..

# # Create the Model for Training
# 
# Recapt the data and create and train some models:

# In[ ]:


df_train.head()


# In[ ]:





# In[ ]:


#the total number of images we have:
train_size = len(train_generator.filenames)
#train_steps is how many steps per epoch Keras runs the genrator. One step is batch_size*images
train_steps = train_size/batch_size
#use 2* number of images to get more augmentations in. some do, some dont. up to you
train_steps = int(2*train_steps)
#same for the validation set
valid_size = len(valid_generator.filenames)
valid_steps = valid_size/batch_size
valid_steps = int(2*valid_steps) 


# Now a method to create the mode(s) for training. Using a previous pre-trained model as a basis, this is called "transfer learning". Transferring the learned features and weights from before to now. 
# 
# Different number of the previously defined and trained layers are often set as "trainable" of "fixed" during training for a new dataset. This is the "trainable_layer_count" given here. I will try three variants of this to see how it performs. Hence the parameter.
# 
# This pre-trained model works as a form of readily provided "feature engineering" for us. On top of that, a custom classification layer, or a few, is provided. So we add those too here.

# In[ ]:


from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)
from keras.applications.resnet50 import ResNet50

def create_model(trainable_layer_count):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = ResNet50(include_top=False,
                          #the weights value can apparently also be a file path..
                   weights=None, #loading weights from dataset, avoiding need for internet conn
                   input_tensor=input_tensor)
    base_model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if trainable_layer_count == "all":
        #the full pre-trained model is fine-tuned in this case
        for layer in base_model.layers:
            layer.trainable = True
    else:
        #if not all should be trainable, first set them all as non-trainable (fixed)
        for layer in base_model.layers:
            layer.trainable = False
        #and finally set the last N layers as trainable
        #idea is to re-use higher level features and fine-tune the finer details
        for layer in base_model.layers[-trainable_layer_count:]:
            layer.trainable = True
    print("base model has {} layers".format(len(base_model.layers)))
    #here on it is the fully custom classification on top of pre-trained layers above
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)
    x = Dropout(0.5)(x)
    #doing binary prediction, so just 1 neuron is enough
    final_output = Dense(1, activation='sigmoid', name='final_output')(x)
    model = Model(input_tensor, final_output)
    
    return model


# ## Model Callbacks
# 
# Set up some general callbacks for all instances of training. These are:
# 
# - checkpoints: save weights of model for best scores while training
# - reduce learning rate: if training gains hit a plateau, try lowering learning rate
# - early stopping: if no gain for several epochs in row, stop
# - logger: log some training details to a file
# 
# Not all of these are really needed for this simple example, but I left them here since I find them useful to have around when generally training longer iterations.

# In[ ]:


# create callbacks list
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger)
                             
from sklearn.model_selection import train_test_split


checkpoint = ModelCheckpoint('../working/Resnet50_best.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                   verbose=1, mode='auto', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=7)

csv_logger = CSVLogger(filename='../working/training_log.csv',
                       separator=',',
                       append=True)

callbacks_list = [checkpoint, csv_logger, early]
# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]


# ## ResNet50 with all layers trainable

# In[ ]:


model = create_model("all")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs = epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        callbacks=callbacks_list,
    verbose = 1
)
#this would load the best scoring weights from above for prediction
model.load_weights("../working/Resnet50_best.h5")


# And this shows the training score evolution over the epochs:

# In[ ]:


fit_history.history


# This scores around 90-91% validation accuracy at best when I ran it. It seems to have quite some diversity, even if final convergence is close. Still, 90% is not too bad really. Lets see the other two configurations though.

# In[ ]:


pd.DataFrame(fit_history.history).head(20)


# In[ ]:


def plot_loss_and_accuracy(fit_history):
    plt.clf()
    plt.plot(fit_history.history['acc'])
    plt.plot(fit_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.clf()
    # summarize history for loss
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[ ]:


plot_loss_and_accuracy(fit_history)


# ## ResNet50 all Layers Fixed
# 
# This version keeps the pre-trained weights as they are and just trains the custom classification layer on top.

# In[ ]:


model = create_model(0)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


train_generator.reset()
valid_generator.reset()
fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs = epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        callbacks=callbacks_list,
    verbose = 1
)
model.load_weights("../working/Resnet50_best.h5")


# At best this scored around 92-93% validation accuracy during my test run. Slightly better than the fully trainable. Maybe both of these could use many more epochs, but not today.

# In[ ]:


pd.DataFrame(fit_history.history).head(20)


# In[ ]:


plot_loss_and_accuracy(fit_history)


# In[ ]:





# ## ResNet50 with last 5 layers trainable, others fixed
# 
# So keep all the rest of the pre-trained layers fixed, but keep finetuning the last 5 in the network along with custom classification layers.

# In[ ]:


model = create_model(5)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


train_generator.reset()
valid_generator.reset()
fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs = epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        callbacks=callbacks_list,
    verbose = 1
)
model.load_weights("../working/Resnet50_best.h5")


# As visible, at best this scores close to 98% validation accuracy at best. Nice. Clearly better than the other two above.

# In[ ]:


pd.DataFrame(fit_history.history).head(20)


# In[ ]:


plot_loss_and_accuracy(fit_history)


# # A look at the models biggest mistakes
# 
# Above, I looked at the images in general. But can we learn anything more about the ones it gives biggest misclassifications for? Lets see what those are.
# 
# First, re-run predictions on the whole validation set, collect probability of cat/dog:

# In[ ]:


valid_generator.reset()
df_valid = pd.DataFrame()


# In[ ]:


np.set_printoptions(suppress=True)
diffs = []
predictions = []
cat_or_dog = []
cd_labels = []
for filename in tqdm(valid_generator.filenames):
    img = PIL.Image.open(f'{train_dir}/{filename}')
    resized = img.resize((img_size, img_size))
    np_img = np.array(resized)
    if "cat" in filename.lower():
        reference = 0 #cat
        cat_or_dog.append(CAT)
    else:
        reference = 1 #dog
        cat_or_dog.append(DOG)
    cd_labels.append(reference)
    score_predict = model.predict(preprocess_input(np_img[np.newaxis]))
#    print(reference)
#    print(score_predict[0][0])
    diffs.append(abs(reference-score_predict[0][0]))
    predictions.append(score_predict)


# In[ ]:


max(diffs)


# "diffs" now has the deviation of assigned label vs predicted, so sorting by that should give the most "confident mistakes" the model makes.

# In[ ]:


df_valid["filename"] = valid_generator.filenames
df_valid["cat_or_dog"] = cat_or_dog
df_valid["cd_label"] = cd_labels
df_valid["diff"] = diffs
df_valid["prediction"] = predictions


# In[ ]:


df_valid.sort_values(by="diff", ascending=False).head()


# This method plots the top N misclassifications:

# In[ ]:


def show_diff_imgs(n):
    sorted_diffs = df_valid.sort_values(by="diff", ascending=False)
    x = 0
    rows = int(math.ceil(n/3))
    height = rows*10
    plt.figure(figsize=[30,height])
    for index, row in sorted_diffs.iterrows():
        filename = row["filename"]
        cat_or_dog = row["cat_or_dog"]
        cd_label = row["cd_label"]
        diff = row["diff"]
        prediction = row["prediction"]
        #print(prediction)
        pred_str = "{:.2f}".format(prediction[0][0])
        img = PIL.Image.open(f"{train_dir}/{filename}")
        print(filename+" "+cat_or_dog+" "+str(diff))
        plt.subplot(3, rows, x+1)
        plt.imshow(img)
        title_str = f"{cat_or_dog}: {cd_label} vs {pred_str}"
        plt.title(title_str)        
        img.close()
        x += 1
        if x > n:
            break


# In[ ]:


show_diff_imgs(10)


# Some of these seem like there is both a cat and a dog in the picture, and thus either label might be correct. Others show small and furry dogs that seem to get mistaken for cats. Overall, humans have no problem identifying these but the model did not. Still, the 5-layer version scores around 98% which is really quite good already.

# # Predictions for Submission
# 
# This competition ended long time ago, so no submitting anything. But it is always good to finish with a full set so I can come back and copy parts for the next real competition where I go take part and finish in the bottom 10% :).
# 

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


test_dir = "../input/dogs-vs-cats/test1/test1"
test_filenames = os.listdir(test_dir)
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# In[ ]:


np.set_printoptions(suppress=True)
predictions = []
for filename in tqdm(test_filenames):
    img = PIL.Image.open(f'{test_dir}/{filename}')
    resized = img.resize((img_size, img_size))
    np_img = np.array(resized)
    np_img = resnet50.preprocess_input(np_img)
    score_predict = model.predict(np_img[np.newaxis])
    predictions.append(score_predict)


# In[ ]:


#1=dog,0=cat
threshold = 0.5
test_df['probability'] = predictions
test_df['category'] = np.where(test_df['probability'] > threshold, 1,0)


# Just as a final sanity check, lets compare some image(s) from the test set and the predicted label. Did we get it right? So no silly mistake of giving does a 0 and cats 1 when it should have been the other way around..

# In[ ]:


test_df.head()


# In[ ]:


filename = test_df.iloc[1]["filename"]
img = PIL.Image.open(f'{test_dir}/{filename}')
plt.imshow(img)


# It looks like a cat, and the prediction set in test_df seems to match what a cat should have been given. Trying a few values in place of the .iloc(X) above shows similarly correct, so leaving it at that..
# 
# And a submission file. Of course, I never tried to submit this, so take it with that:

# In[ ]:


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)


# # Summary
# 
# My goal here was to learn how to use CNNs to do image classification. I guess writing "raw" CNN models would be another good step, but this was still an interesting look into building an image classifier with Keras. Have to have a decent understanding to use transfer learning as well. Thats what I tell myself.
# 
# The augmentation experiments were helpful to see how they really work, and I guess I would use similar approaches to understand the transformations against the target domain when selecting the (hyper)parameters for augmentation.
# 
# The best score I got here was from the partly trainable ResNet model. I guess doing longer training sessions on the fully trainable model, and trying different depths of trainable vs fixed in the mixed model would be useful to try. My takeaway is to use transfer learning, or try it at least if I have large-ish datasets and problems, but limited computing power and resources.
# 
# 
