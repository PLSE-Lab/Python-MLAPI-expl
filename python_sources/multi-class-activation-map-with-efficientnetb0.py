#!/usr/bin/env python
# coding: utf-8

# # Visualizing the areas of an image used for making predictions: Class Activation Maps on Bengali
# 
# The goal of this notebook is to investigate the use of Class Activation Maps for an EfficientNetB0.
# 
# This notebook is heavily inspired form the work of @cdeotte in his [notebook for the Understanding Clouds competitions](https://www.kaggle.com/cdeotte/unsupervised-masks-cv-0-60) and the work he refers to that can be found [here](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/). (This details why the Global Average Pooling is used, and shows a nice implementation for ResNet50). I also wanted to turn towards this after seeing what @pnussbaum did for his [mind reading notebook](https://www.kaggle.com/pnussbaum/grapheme-mind-reader-panv12-nogpu) where we explores what the different conolutional layers look like. His work is a bit different, but still very interesting nonetheless! I'm sure combining filter + convolutional visualizations with class activation maps can provide some really interesting feedback on how models make there predictions, and how to improve them!.

# ### Multi-Class Activation Maps
# 
# There doesn't seem to be much word on multi-class activation maps. There has been a lot of dicussions about wether 1 model with 3 outputs, or 3 models with 1 output each is better for this competition, but everytime it comes down to compute time, and local cross-validation scores. People then _assume_ that this or that is better, but it seems like we don't know much. Hopefully methods like this will help give more evidence towards one method or the other.

# In[ ]:


import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Modelling
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform

# Metrics & Loss
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import f1_score

import os


# In[ ]:


# Adding the EfficientNet library

sys.path.append(os.path.abspath("../input/efficientnet-keras-source-code/repository/qubvel-efficientnet-c993591"))
import efficientnet.tfkeras as efn


# In[ ]:


SEED_VALUE = 42


# In[ ]:


model_nb_output_dictionary = {
    'grapheme_root': 168,
    'vowel_diacritic': 11,
    'consonant_diacritic': 7
}


# In[ ]:


# Original image size
HEIGHT = 137
WIDTH = 236

# Size of the images that will be resized
RESIZED_HEIGHT = 75
RESIZED_WIDTH = 75


# We tweak a little bit how the model is created, by adding a GlobalAveragePooling layer (once again, more on why [here](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)), as well as also creating a CAM_model and keeping the weights of the last convolutional layer. More on that later.

# In[ ]:


def EfficientNetB0_with_CAM(input_shape=(HEIGHT, WIDTH, 1),
                            classes_dict=model_nb_output_dictionary, 
                            print_last_conv_info = False):
    """
    EfficientNetB0 implementation
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes_dict -- dict of integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape)
    x = Conv2D(3, (3,3), padding='same')(X_input)

    # Base model for transfer learning
    base_model = efn.EfficientNetB0(weights=None, include_top=False, input_tensor=x)

    x = base_model.output
    
    # The Global Average Pooling is important to add here.
    x = GlobalAveragePooling2D()(x)    
    
    # 3 outputs for our three symbols
    out_root = Dense(classes_dict['grapheme_root'], activation='softmax', name='fc_root', kernel_initializer=glorot_uniform(seed=0))(x)
    out_vowel = Dense(classes_dict['vowel_diacritic'], activation='softmax', name='fc_vowel', kernel_initializer=glorot_uniform(seed=0))(x)
    out_consonant = Dense(classes_dict['consonant_diacritic'], activation='softmax', name='fc_consonant', kernel_initializer=glorot_uniform(seed=0))(x)

    # Create model
    model = Model(inputs=X_input,
                  outputs=[out_root,
                           out_vowel,
                           out_consonant],
                  name='Base_model')
    
    # For CAM
    last_conv = base_model.layers[-3] # In the EfficientNetB0, the last Conv2D layer is the 3rd to last
    if print_last_conv_info:
        print(f'last conv layer: {last_conv}')
        print(f'\nConfig of the last conv layer:\n{last_conv.get_config()}')
    
    last_dense_root = model.layers[-3]
    last_dense_vowel = model.layers[-2]
    last_dense_consonant = model.layers[-1]
    last_dense_root_weights = last_dense_root.get_weights()[0]
    last_dense_vowel_weights = last_dense_vowel.get_weights()[0]
    last_dense_consonant_weights = last_dense_consonant.get_weights()[0]
    
    model_cam = Model(inputs = X_input,
                     outputs = (last_conv.output,
                                last_dense_root.output,
                                last_dense_vowel.output,
                                last_dense_consonant.output),
                     name = 'CAM_model')
    
    dense_layer_weights_list = [last_dense_root_weights, last_dense_vowel_weights, last_dense_consonant_weights]

    return model, model_cam, dense_layer_weights_list


# In[ ]:


model, model_cam, dense_layer_weights_list = EfficientNetB0_with_CAM(
                                                    input_shape = (RESIZED_HEIGHT, RESIZED_WIDTH, 1),
                                                    classes_dict=model_nb_output_dictionary,
                                                    print_last_conv_info=False
)


# The EfficientNetB0 + Global Average Pooling was trained in another notebook in order to only focus on the Class Activaton Maps in this notebook.

# In[ ]:


# Loading the model trained with GlobalAveragePooling after the EfficientNetB0.

model.load_weights('/kaggle/input/efficientnetb0-cam-weights/EfficientNetB0_for_CAM_common_10_epochs_Earlystop (2).h5')


# In[ ]:


def crop_and_resize_images(df, resized_df, resize_size = RESIZED_HEIGHT):
    """
    Function used to resize & center the images.
    """
    cropped_imgs = {}
    for img_id in tqdm(range(df.shape[0])):
        img = resized_df[img_id]
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        idx = 0 
        ls_xmin = []
        ls_ymin = []
        ls_xmax = []
        ls_ymax = []
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            ls_xmin.append(x)
            ls_ymin.append(y)
            ls_xmax.append(x + w)
            ls_ymax.append(y + h)
        xmin = min(ls_xmin)
        ymin = min(ls_ymin)
        xmax = max(ls_xmax)
        ymax = max(ls_ymax)

        roi = img[ymin:ymax,xmin:xmax]
        resized_roi = cv2.resize(roi, (resize_size, resize_size))
        cropped_imgs[df.image_id[img_id]] = resized_roi.reshape(-1)
        
    resized = pd.DataFrame(cropped_imgs).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


# In[ ]:


df = pd.read_parquet("/kaggle/input/bengaliai-cv19/train_image_data_0.parquet")


# In[ ]:


NUMBER_OF_IMAGES_FOR_CAM = 50


# In[ ]:


# Resizing the images to 75x75 and centering them

df_subset = df.head(NUMBER_OF_IMAGES_FOR_CAM)
resized = df_subset.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
cropped_df = crop_and_resize_images(df_subset, resized, RESIZED_HEIGHT)
indexes = df_subset.index.values
df_subset.set_index('image_id', inplace=True)
resized = cropped_df.iloc[:, 1:].values.reshape(-1, RESIZED_HEIGHT, RESIZED_WIDTH)
resized = resized.reshape(-1, RESIZED_HEIGHT, RESIZED_WIDTH, 1)


# We're going to only compute these for a small number of images, in this case 50 (75x75x1) images.

# In[ ]:


print(f'Shape of the resized images : {resized.shape}')


# In[ ]:


# Getting the features and the predictions for each of the 3 symbols:

features, preds_root, preds_vowel, preds_consonant = model_cam.predict(resized)


# In[ ]:


print(f'Features shape: {features.shape}')
print(f'Root predictions shape: {preds_root.shape}')
print(f'Vowel predictions shape: {preds_vowel.shape}')
print(f'Consonant predictions shape: {preds_consonant.shape}')


# If we break down what we have:
# 
# features: [ number of images (50) ; height of filters (3) ;  width of filters (3) ; number of filters (1280) ] 
# 
# These is the output from the last convolutional 2 layer of the EfficientNetB0 model. However, simply taken alone, those filters don't represent much. We need to multiple them by the different weights of the Global Average Pooling layer, which will then tell us which ones are activated for each image, for each of the 3 symbols.

# # Creating the Class Activation Maps
# 
# A Class Activation Map, as its name implies, gives the activation map for a given class. We are interested in seeing the one for each of our 3 symbols. For each of those, we could respectfully create 168, 11 and 7 class activation map, for each of the possible values. However we'll only show the one for the predicted class from the model.

# In[ ]:


# We need to choose 1 image to plot:

IMG_TO_PLOT = 42


# In[ ]:


# Getting the predictions for each symbol:
raw_preds_list = [preds_root, preds_vowel, preds_consonant]

root_preds = np.argmax(preds_root[IMG_TO_PLOT])
vowel_preds = np.argmax(preds_vowel[IMG_TO_PLOT])
consonant_preds = np.argmax(preds_consonant[IMG_TO_PLOT])

preds_for_img_list = [root_preds, vowel_preds, consonant_preds]


# In[ ]:


# We then can then see the features for this specific image:
print(f'Shape of the feature map of a given image: {features[IMG_TO_PLOT,:,:,:].shape}')


# However, these are a lot of different 3x3 filters. If we want to see their activations, we need to scale them up to the size of the image

# In[ ]:


# Upscaling those features to the size of the image:
import scipy

scale_factor_height = RESIZED_HEIGHT/features[IMG_TO_PLOT,:,:,:].shape[0]
scale_factor_width = RESIZED_HEIGHT/features[IMG_TO_PLOT,:,:,:].shape[1]

print(f'Scale height factor: {scale_factor_height}')
print(f'Scale width factor: {scale_factor_width}')

upscaled_features = scipy.ndimage.zoom(features[IMG_TO_PLOT,:,:,:],
                                       (scale_factor_height, scale_factor_width, 1), order=1)
print(f'\nScaled feature map size for a given image: {upscaled_features.shape}')


# Finally, in order to visualize which features are activated and which aren't, we need to multiply them by the weights of the final dense layer of the symbol & class we wish to visualise. This is a simple dot product.

# In[ ]:


symbol_to_plot = 0

cam_output = np.dot(upscaled_features, 
                    dense_layer_weights_list[symbol_to_plot][:,preds_for_img_list[symbol_to_plot]])


# For visual interpretation, we can plot the original feature map, and the upscaled one, to see how the upscaling works.

# In[ ]:


original_cam_output = np.dot(features[IMG_TO_PLOT,:,:,:],
                                          dense_layer_weights_list[symbol_to_plot][:,preds_for_img_list[symbol_to_plot]])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))

ax0.imshow(original_cam_output)
ax0.set_title('Original CAM output')

ax1.imshow(cam_output)
ax1.set_title("Rescaled CAM output")

plt.plot()


# All that is left to do is to perform this for each of the 3 symbols, and plot it above the original image. This will then allow us to have a sense of which areas are mostly used for making the predictions in the image.

# In[ ]:


def showing_cam(img, 
                img_arrays, 
                features=features, 
                #predicted_img_list=preds_for_img_list, 
                raw_preds_list=raw_preds_list, 
                dense_layer_weights_list=dense_layer_weights_list):
    
    features_for_img = features[img,:,:,:]
    
    root_preds = np.argmax(raw_preds_list[0][img])
    vowel_preds = np.argmax(raw_preds_list[1][img])
    consonant_preds = np.argmax(raw_preds_list[2][img])
    predicted_img_list = [root_preds, vowel_preds, consonant_preds]
    
    preds_root_round = np.round(raw_preds_list[0][img][root_preds], 3)
    preds_vowel_round = np.round(raw_preds_list[1][img][vowel_preds], 3)
    preds_consonant_round = np.round(raw_preds_list[2][img][consonant_preds], 3)
    
    # Upscaling those features to the size of the image:
    scale_factor_height = RESIZED_HEIGHT/features[IMG_TO_PLOT,:,:,:].shape[0]
    scale_factor_width = RESIZED_HEIGHT/features[IMG_TO_PLOT,:,:,:].shape[1]
    
    upscaled_features = scipy.ndimage.zoom(features[img,:,:,:], 
                                           (scale_factor_height, scale_factor_width, 1), 
                                           order=1)
    
    prediction_for_img = []
    cam_weights = []
    cam_output = []
    
    for symbol in range(3):
        prediction_for_img.append(predicted_img_list[symbol])
        cam_weights.append(dense_layer_weights_list[symbol][:,prediction_for_img[symbol]])
        cam_output.append(np.dot(upscaled_features, cam_weights[symbol]))
    
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(15, 10))
    
    squeezed_img = np.squeeze(img_arrays[img], -1)
    
    #fig.suptitle(img_prediction_class)
    ax0.imshow(squeezed_img, cmap='Greys')
    ax0.set_title("Original image")
    
    ax1.imshow(squeezed_img, cmap='Greys', alpha=0.5)
    ax1.imshow(cam_output[0], cmap='jet', alpha=0.5)
    ax1.set_title(f"CAM - Root ({root_preds} with proba = {preds_root_round})")
    #ax1.set_title(f"CAM - Root")
    
    ax2.imshow(squeezed_img, cmap='Greys', alpha=0.5)
    ax2.imshow(cam_output[1], cmap='jet', alpha=0.5)
    ax2.set_title(f"CAM - Vowel ({vowel_preds} with proba = {preds_vowel_round})")

    
    ax3.imshow(squeezed_img, cmap='Greys', alpha=0.5)
    ax3.imshow(cam_output[2], cmap='jet', alpha=0.5)
    ax3.set_title(f"CAM - Consonant ({consonant_preds} with proba = {preds_consonant_round})")

    
    plt.show()


# In[ ]:


for img in range(10,15):
    showing_cam(img, img_arrays=resized)


# These provide some interesting information about which section of the image is used for each image & symbol.
# 
# Once again, it is important to keep in mind these are made from the 3x3 filters from the last Conv2D layer of the model, and so can only provide some limited information about the overage region of the image used for making the prediction, and not the small details. Having a last convolutional layer with bigger features might be a nice addition to try to find more fine-tuned details in the areas used to make the predictions. Choosing a model that have a bigger filter sizes in the last convolutional layer might also be something to take into account when deploying models that require a high degree of explainability.

# ### Ideas & improvements to be made
# 
# Obvioulsy, as is, there isn't that much we can take form these. However, we could look at all the images with the same root in them for example and visualize if the model picks up the same things in the image all the time. Visualizing the different areas in correctly and incorreclty classified image might also provide helpful information!
# 
# As mentioned above, I would also like to try this on models having bigger filter sizes. Maybe it could be interesting to try to add Convolutional layers after the final ones in the EfficientNet, in a similar fashion as a UNet would, to visualize which areas are being used.
# 
# I'm also totally open to any feedback & suggestions!
