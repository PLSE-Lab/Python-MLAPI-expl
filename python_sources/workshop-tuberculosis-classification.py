#!/usr/bin/env python
# coding: utf-8

# # Contents
# - Preparing Data
# - Creating a Simple Model
# - Adding Augmentations
# - Creating a Simple Neural Network
# - Some black magic techniques...

# We'll start by importing some dependencies:

# In[ ]:


import os
import gc
import re
import cv2
import sys
import glob
import keras

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style  as style

from tqdm  import tqdm
from keras import backend as K
from sklearn.metrics     import accuracy_score, roc_auc_score
from keras.layers        import Dense, Dropout, Flatten, BatchNormalization, GlobalMaxPooling2D
from keras.models        import Sequential, Model, load_model
from keras.callbacks     import ModelCheckpoint,ReduceLROnPlateau, CSVLogger
from keras.activations   import elu
from keras.engine        import Layer, InputSpec
from keras.applications  import MobileNetV2
from keras.optimizers    import Adam
from keras.preprocessing import image
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenetv2 import preprocess_input, decode_predictions


# In[ ]:


# Disable SetCopyWarnings
pd.options.mode.chained_assignment = None


# # Data Preparation

# The first thing we'll do is index all the images we have from both datasets, we will use `glob` to accomplish this.

# In[ ]:


# Compose filenames
filelist_montgommery = glob.glob('../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png/*.png')
filelist_shenzen     = glob.glob('../input/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/*.png')
filelist             = filelist_montgommery + filelist_shenzen


# From the documentation, we know that the label is contained within the filename. To extract such label, we'll use the `re` module to implement regex to get the label. 

# In[ ]:


def extract_label(file_list):
    '''
    Label Extraction Function
    Reads a filename and extracts label from it
    '''
    labels = []
    for file in tqdm(file_list):
        current_label = re.findall('[0-9]{4}_(.+?).png', file)
        labels.append(current_label[0])
    return(labels)


# In[ ]:


labels = extract_label(filelist)


# Let's now transform our lists into a dataframe.

# In[ ]:


# Create dataframe
full_data = pd.DataFrame(filelist, columns=['filepath'])
full_data['target'] = labels
full_labels         = pd.DataFrame(full_data.pop('target'), columns=['target'])

# Preview dataframe
full_data.head()


# ![](https://miro.medium.com/max/1552/1*Nv2NNALuokZEcV6hYEHdGA.png)

# Now we have all the data into a single dataframe, but we need to split a certain amount for **model selection** (validation / probe) and for evaluating the model generalisation - the **testing set**.

# In[ ]:


# Split data into training and testing sets
train_df,test_df,train_y,test_y = train_test_split(full_data,
                                                   full_labels,
                                                   stratify     = full_labels,
                                                   test_size    = 0.3,
                                                   random_state = 451)


# In[ ]:


# Reassign labels so that we can split them again
train_df['target'] = train_y['target']
test_df['target']  = test_y['target']


# In[ ]:


# Split once more, so that we may produce a validation set
labels = train_df.pop('target')
train_df, probe_df, train_y, probe_y = train_test_split(train_df,
                                                        labels,
                                                        stratify     = labels,
                                                        test_size    = 0.2,
                                                        random_state = 451)

# Reassemble labels
train_df['target'] = train_y
probe_df['target'] = probe_y


# # Exploratory Analysis

# Now that we have our sets, let's have a look at them and check some images as well.

# In[ ]:


# Inspect Training Dataframe
train_df.head()


# In[ ]:


def plot_multiple_images(image_dataframe, rows = 4, columns = 4, figsize = (16, 20), resize=(1024,1024), preprocessing=None):
    '''
    Plots Multiple Images
    Reads, resizes, applies preprocessing if desired and plots multiple images from a given dataframe
    '''
    image_dataframe = image_dataframe.reset_index(drop=True)
    fig = plt.figure(figsize=figsize)
    ax  = []

    for i in range(rows * columns):
        img = plt.imread(image_dataframe.loc[i,'filepath'])
        img = cv2.resize(img, resize)
        
        if preprocessing:
            img = preprocessing(img)
        
        ax.append(fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title("Xray "+str(i+1))
        plt.imshow(img, alpha=1, cmap='gray')
    
    plt.show()


# In[ ]:


plot_multiple_images(train_df)


# Instead of going for a fancy model, we'll start simple.
# The most simple approach might be to just resize and flatten the pixels and send them to a simple model, such as a logistic regression.

# ![](http://pwmpcnhis.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/programming_assignments/week1_week2/2.png)

# In[ ]:


def load_image(image_path, image_dims = (128,128), grayscale=True, flatten=True, interpolation = cv2.INTER_AREA):
    '''
    Loads an image, resizes and removes redudant channels if so desired
    '''
    image         = cv2.imread(image_path)
    resized_image = cv2.resize(image, image_dims, interpolation = interpolation)
    
    if grayscale:
        resized_image = resized_image[:,:,0]
    
    if flatten:
        resized_image = resized_image.flatten()
    
    return(resized_image)


# In[ ]:


def create_flattened_dataframe(df, interpolation = cv2.INTER_AREA):
    df     = df.reset_index(drop=True)
    result = pd.DataFrame()
    
    for i in tqdm(range(df.shape[0])):
        im_path = df.loc[i,'filepath']
        current = load_image(im_path, interpolation = interpolation).tolist()
        current = pd.DataFrame(current).T
        result  = result.append(current)
    
    return(result)


# In[ ]:


# Flatten Training dataframe
train_df_flat = create_flattened_dataframe(train_df)
probe_df_flat = create_flattened_dataframe(probe_df)
test_df_flat  = create_flattened_dataframe(test_df)


# In[ ]:


train_df_flat.head()


# With that, our flattened dataframe is ready to go and we can now create our model.

# # Train a Simple Logistic Regression Model

# We'll create a logistic regression model, using `random state` for reproducibility and setting the `max iterations` to `1000` so that the model has more than enough iterations to converge.

# In[ ]:


# Create Logistic Regression
logit_model = LogisticRegression(random_state=451, solver='lbfgs', max_iter=1000)
logit_model.fit(train_df_flat, train_df['target'])


# Our model is now ready, let us now create a function to evaluate the `Accuracy` and `AUC` of our model.

# In[ ]:


def evaluate_predictions(preds, eval_df = test_df):
    '''
    Evaluate Predictions Function
    Returns accuracy and auc of the model
    '''
    auroc = roc_auc_score(eval_df['target'].astype('uint8'), preds)
    accur = accuracy_score(eval_df['target'].astype('uint8'), preds >= 0.5)
    print('Accuracy: ' + str(auroc))
    print('AUC: ' + str(accur))


# In[ ]:


# Evaluate Model Results - Validation Set
logit_preds_val  = logit_model.predict_proba(probe_df_flat)
evaluate_predictions(logit_preds_val[:,1], eval_df = probe_df)


# In[ ]:


# Evaluate Model Results - Testing Set
logit_preds  = logit_model.predict_proba(test_df_flat)
evaluate_predictions(logit_preds[:,1], eval_df = test_df)


# Yeet!
# 
# Our Logistic regression is doing quite well.
# Now let's create simple CNN.

# # Simple CNN - MobileNetV2

# Notice that, for our last exercise, we loaded all images in memory and this is not ideal.
# In the case of massive models, we usually create generators that will load a few instances of images each time, hence the idea of `batch_size`.
# 
# Our model, MobileNetV2 was created with the input size of 224x224, so we'll keep that in our model.

# In[ ]:


batch_size = 32
input_size = (224,224)


# In order to help us load few images at a time, we'll use keras `ImageDataGenerator` - a class that allows us to load some images at a time using it's `flow` method and add data augmentation as well.

# ![](https://rock-it.pl/content/images/2017/05/doggs.jpg)

# Data augmentation plays a major role in increasing the assertiveness of our models, especially when the dataset is rather small (as it is the case here).
# Let's create a generator to do that.

# In[ ]:


# Create training data generator
train_generator = ImageDataGenerator(rescale = 1./255,
                                     horizontal_flip = True,
                                     zoom_range      = 0.1,
                                     shear_range     = 0,
                                     rotation_range  = 5,
                                     width_shift_range  = 0.05,
                                     height_shift_range = 0.05,
                                     fill_mode = 'constant',
                                     cval      = 0,
                                     preprocessing_function = preprocess_input)

# Create testing data generator
test_generator  = ImageDataGenerator(rescale = 1./255,
                                     preprocessing_function = preprocess_input)


# Augmentation is one thing, but we also need to configure how the `ImageDataGenerator` will walk - or rather flow - through our dataset.

# In[ ]:


train = train_generator.flow_from_dataframe(dataframe = train_df,
                                    class_mode  = 'binary',
                                    x_col       = 'filepath',
                                    y_col       = 'target',
                                    shuffle     = True,
                                    batch_size  = batch_size,
                                    target_size = input_size,
                                    seed=451)

probe = train_generator.flow_from_dataframe(dataframe = probe_df,
                                    class_mode  = 'binary',
                                    x_col       = 'filepath',
                                    y_col       = 'target',
                                    shuffle     = True,
                                    batch_size  = batch_size,
                                    target_size = input_size,
                                    seed=451)


# With that done, we can now build our neural network.

# In[ ]:


# Create model instance
model = Sequential()

# Add Pretrained Model
model.add(MobileNetV2(weights = 'imagenet', input_shape = (224,224,3), include_top=False))

# Add FC and Output layers
model.add(GlobalMaxPooling2D())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(optimizer=Adam(lr=1e-1), loss='binary_crossentropy', metrics=['acc'])
model.summary()


# We can add support elements to our training, also known in keras as `callbacks`. They can help us manage our learning rate, create model checkpoints, use tensorboard and many more things.
# 
# For now we'll just use the `CSVLogger`, which save training logs a csv file.

# In[ ]:


# Create callback
csv_logger = CSVLogger('training_log_mobilenet_try_1.csv')


# In[ ]:


# Train model
model.fit_generator(
        train,
        callbacks = [csv_logger],
        epochs    = 10,
        steps_per_epoch  = train.samples // batch_size,
        validation_data  = probe,
        validation_steps = probe.samples // batch_size)


# This is rather strange, our model seems like garbage...let's see the performance on the testing set.

# In[ ]:


# Create test generator
test = test_generator.flow_from_dataframe(dataframe = test_df,
                                    class_mode  = 'binary',
                                    x_col       = 'filepath',
                                    y_col       = 'target',
                                    shuffle     = False,
                                    batch_size  = batch_size,
                                    target_size = input_size)

# Generate Predictions
preds = model.predict_generator(test, steps= test.samples / batch_size)

# Evaluate Model Results
evaluate_predictions(preds)


# Garbage confirmed!
# 
# Let's have a look at the training history

# In[ ]:


def plot_training_hist(keras_model):
    '''
    Plot training History
    Creates two plots of model training logs
    '''
    hist = keras_model.history.history
    style.use('fivethirtyeight')
    
    # Loss Plot
    fig = plt.figure(figsize=(12,4))
    plt.title('Loss Plot')
    plt.plot(hist['loss'], '#07e9ed')
    plt.plot(hist['val_loss'], '#0791ed')
    plt.ylim([0,1.2 * max(max(hist['loss'], hist['val_loss']))])
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()
    
    # Accuracy Plot
    fig = plt.figure(figsize=(12,4))
    plt.title('Accuracy Plot')
    plt.plot(hist['acc'], '#07e9ed')
    plt.plot(hist['val_acc'], '#0791ed')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.ylim([0,1])
    
    # Show Plot, reset style
    plt.show()
    style.use('default')


# In[ ]:


plot_training_hist(model)


# Our network is way worse than our logistic regression, how can that be?
# 
# In many ways this is a lesson in deep learning vs machine learning.
# 
# The idea that deep learning is superior is in many ways inadequate.
# And the role of the human behind those models is often more important than the methods thenselves.
# 
# 
# Deep learning often overshadows machine learning for it's capacity to accomplish many tasks.
# While it certainly has some merit, other aproaches can also be just as good, sometimes better and frequently more `consistent`.
# 
# Giba, ranked 1st in kaggle for a couple years mentioned that on a comment once: while deep learning is great it might take you the whole day to make it good
# While you can get as of a result with a single hour using ML methods.
# 
# Either way, we can improve our CNN model, but don't walk the path of becoming closed minded about CNNs as the single possible solution.
# I teach this workshop so that you expand your way of thinking - not constrain it to a single method.

# # Simple CNN - Trying Again

# Now, let's make some small changes to our model.
# We'll just lower the learning rate. Just that.

# In[ ]:


# Create model instance
model = Sequential()

# Add Pretrained Model
model.add(MobileNetV2(weights = 'imagenet', input_shape = (224,224,3), include_top=False))

# Add FC and Output layers
model.add(GlobalMaxPooling2D())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(optimizer=Adam(lr=2e-4), loss='binary_crossentropy', metrics=['acc'])
model.summary()


# In[ ]:


# Train model
model.fit_generator(
        train,
        callbacks = [csv_logger],
        epochs    = 15,
        steps_per_epoch  = train.samples // batch_size,
        validation_data  = probe,
        validation_steps = probe.samples // batch_size)


# In[ ]:


# Create test generator
test = test_generator.flow_from_dataframe(dataframe = test_df,
                                    class_mode  = 'binary',
                                    x_col       = 'filepath',
                                    y_col       = 'target',
                                    shuffle     = False,
                                    batch_size  = batch_size,
                                    target_size = input_size)

# Generate Predictions
preds = model.predict_generator(test, steps= test.samples / batch_size)

# Evaluate Model Results
evaluate_predictions(preds)


# In[ ]:


plot_training_hist(model)


# Now that's far better, and we managed to improve our previous baseline score from the logistic regression model.
# 
# Let's same the model weights to disk so that we can use them later if we want to.

# In[ ]:


# Save Model Weights as a File
model.save_weights('mobilenetv2_12sep.h5')


# That concludes the core part of our workshop.
# We created a simple model and used it as baseline to improve creating another simple CNN model.
# 
# From here on, we will dive into some neat tricks you can use.

# # Viceroy Model

# For now we are not allowed to use NN as our final deliverable, so let's dial down a bit.
# 
# However...we can use the convolutional layers to better interpret the information and then train a logistic regression on what the previous neural network 'saw'...
# 
# Let's try that...

# In[ ]:


# Create function to extract bottlenet features
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=1).output)

# Create training dataset using bottleneck features
intermediate_output = intermediate_layer_model.predict_generator(test_generator.flow_from_dataframe(dataframe = train_df,
                                    class_mode  = 'binary',
                                    x_col       = 'filepath',
                                    y_col       = 'target',
                                    shuffle     = False,
                                    batch_size  = batch_size,
                                    target_size = input_size,
                                    seed        = 451), steps = train_df.shape[0] / batch_size)

intermediate_output_df = pd.DataFrame(intermediate_output)
intermediate_output_df.head()

# Create bottleneck features dataset for testing set
intermediate_output_df_test = pd.DataFrame(intermediate_layer_model.predict_generator(test_generator.flow_from_dataframe(dataframe = test_df,
                                    class_mode  = 'binary',
                                    x_col       = 'filepath',
                                    y_col       = 'target',
                                    shuffle     = False,
                                    batch_size  = batch_size,
                                    target_size = input_size,
                                    seed        = 451), steps = test_df.shape[0] / batch_size))


# In[ ]:


# Our new model
viceroy_logreg = LogisticRegression(multi_class='auto', solver='lbfgs', random_state=451, max_iter=1000)
viceroy_logreg.fit(intermediate_output_df, train_y)

# Evaluate Model Results
viceroy_logit_preds  = viceroy_logreg.predict_proba(intermediate_output_df_test)
evaluate_predictions(viceroy_logit_preds[:,1])


# # Performing Test Time Augmentation

# We can use augmentation to improve the training of neural nets.
# 
# We can also use it to 

# In[ ]:


def TTA_wraper(dl_model, df = test_df, steps = 10, bs = batch_size, sz = input_size, seed = 451):
    # Taken from: https://towardsdatascience.com/test-time-augmentation-tta-and-how-to-perform-it-with-keras-4ac19b67fb4d
    tta_steps   = steps
    predictions = []

    for i in tqdm(range(tta_steps)):
        tta_pred = dl_model.predict_generator(train_generator.flow_from_dataframe(dataframe = df,
                                        class_mode  = 'binary',
                                        x_col       = 'filepath',
                                        y_col       = 'target',
                                        shuffle     = False,
                                        batch_size  = bs,
                                        target_size = sz,
                                        seed        = seed * i), steps = df.shape[0] / batch_size)
        predictions.append(tta_pred)

    tta_preds = np.mean(predictions, axis=0)
    return(tta_preds)


# In[ ]:


tta_preds = TTA_wraper(model)
evaluate_predictions(tta_preds)


# # Experiment 1: Interpolation

# One of the experiments we had was checking how the interpolation can change the result of the model performance.
# 
# Let's use a consistent model to check which interpolation produces better scores (and by how much).

# In[ ]:


all_interpolations = ['cv2.INTER_AREA', 'cv2.INTER_LINEAR', 'cv2.INTER_NEAREST', 'cv2.INTER_CUBIC', 'cv2.INTER_LANCZOS4']

def interpolation_experiment(interpolation_list):
    for current_interpolation in interpolation_list:
        # Create flattenned dataframes
        experimental_train_df_flat = create_flattened_dataframe(train_df, interpolation = eval(current_interpolation))
        experimental_probe_df_flat = create_flattened_dataframe(probe_df, interpolation = eval(current_interpolation))
        experimental_test_df_flat  = create_flattened_dataframe(test_df,  interpolation = eval(current_interpolation))
        
        print(current_interpolation)
        
        # Create Logistic Regression Model
        experimental_logit_model = LogisticRegression(random_state=451, solver='lbfgs', max_iter=1000)
        experimental_logit_model.fit(experimental_train_df_flat, train_df['target'])
        
        # Evaluate Results
        print('--------------------------')
        print('Validation Set Results')
        print('--------------------------')
        # Evaluate Model Results - Testing Set
        experimental_logit_preds  = logit_model.predict_proba(experimental_probe_df_flat)
        evaluate_predictions(experimental_logit_preds[:,1], eval_df = probe_df)
        
        print('--------------------------')
        print('Testing Set Results')
        print('--------------------------')
        # Evaluate Model Results - Testing Set
        experimental_logit_preds  = logit_model.predict_proba(experimental_test_df_flat)
        evaluate_predictions(experimental_logit_preds[:,1], eval_df = test_df)


# In[ ]:


interpolation_experiment(all_interpolations)


# # Experiment 2: Preprocessing

# Soon

# # Experiment 3: Lung Segmentation

# Soon

# # Experiment 4: Lung Detection

# Soon

# # Visualizing ConvNet Outputs

# Soon

# # Masked Classification
# 
# Unfinished;
# 
# Montgommery only

# In[ ]:


# Compose masks for each lung
left_mask  = glob.glob('../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/ManualMask/leftMask/*.png')
right_mask = glob.glob('../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/ManualMask/rightMask/*.png')

# Sort all image files so that the filenames match
left_mask.sort()
right_mask.sort()
filelist_montgommery.sort()


# In[ ]:


# Create dataframe with all Montgommery data
mont_df = pd.DataFrame(filelist_montgommery, columns=['filepath'])
mont_df['target'] = extract_label(filelist_montgommery)
mont_df.head()


# In[ ]:


def fuse_masks(list1,list2, dir_name='fused_images'):
    '''
    Fuse Masks Functions
    Fuses left and right lung mask into a single mask
    '''
    try:
        os.mkdir(dir_name)
    except:
        print('Directory {} already exists'.format(dir_name))
        pass
    
    for i in tqdm(range(len(list1))):
        # Read and fuse arrays together
        array1 = cv2.imread(list1[i])
        array2 = cv2.imread(list2[i])
        fused  = np.add(array1,array2)
        
        # Export to disk
        filename = 'M' + re.findall(r'Mask\/M(.+?)\.png', list1[i])[0] + '.png'
        cv2.imwrite(os.path.join(dir_name, filename), fused)


# In[ ]:


# Fuse masks together
fuse_masks(left_mask,right_mask)


# In[ ]:


# Add fused masks to dataframe
fused_masks_list    = glob.glob('fused_images/*.png')
mont_df['maskpath'] = fused_masks_list


# In[ ]:


# Split Mont. Data into training and testing set
mont_train, mont_test = train_test_split(mont_df,
                                         test_size = 0.3,
                                         stratify  = mont_df['target'],
                                         random_state = 451)

# Pop labels
mont_train_y = mont_train.pop('target')
mont_test_y  = mont_test.pop('target')


# In[ ]:


def iou_loss_core(y_true, y_pred, smooth=1):
    '''
    IoU, aka jacquard index
    Usable as a metric for keras
    '''
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union        = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou          = (intersection + smooth) / ( union + smooth)
    return(iou)


# In[ ]:


def dice_coef(y_true, y_pred):
    '''
    Dice Coefficient
    Usable as metric for keras
    '''
    y_true_f     = K.flatten(y_true)
    y_pred_f     = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    result       = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return(result)


# In[ ]:


# Clean up
get_ipython().system('rm -rf fused_images')

