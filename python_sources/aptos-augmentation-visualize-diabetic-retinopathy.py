#!/usr/bin/env python
# coding: utf-8

# # 0. Spotting Blindness -- Real or Spurious?
# 
# Above is the old title I used, in this new version I also show how to analyze robustness of our model using the great **albumentation**
# 
# ![Albumentation meets Grad-CAM](https://i.ibb.co/H4MJWVz/gradcam-album.png)
# 
# Therefore, I decide to change the title to more appropriately reflect the techniques I used here. Please see Section 5 below for the updated material.
# 
# 
# # 1. Introduction : This eye is in danger, I estimate severity level 4
# 
# Do you want to understand, when the model saying the above statement, how does it know? Does the model look at the same bloody or cotton wool spots like us? Below are what CNN actually see. Does this make sense? 
# 
# Look at the picture below, in the first case, it seems that the model works great! It is able to identify important spots in the eye. In the second case, however, even though the model estimate the severity to be level 3, it almost entirely misses the big wool spots in the middle. It might infer that our model still doesn't grasp an important concept of 'hard exudates' well enough. (ref. https://www.eyeops.com/)

# ![grad-cam](https://i.ibb.co/6FM6VCC/gradcam-resized.png)
# 
# ![ref https://www.eyeops.com/](https://sa1s3optim.patientpop.com/assets/images/provider/photos/1947516.jpeg)
# 
# The technique I applied to visualize two cases above is called  "Grad-CAM" ([Gradient-weighted Class Activation Mapping](http://gradcam.cloudcv.org/) ; please see the link and reference therein for original materials). In this kernel, I will illustrate how to use it to get more insights from your model. 
# 
# I prefer to write in Keras and so I choose to visualize the [public Keras model of @xhlulu](https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter) who achieve the highest LB score at the moment I write this kernel. (Good job @xhlulu!). For pytorch lovers, please refer to [this kernel of @daisukelab](https://www.kaggle.com/daisukelab/verifying-cnn-models-with-cam-and-etc-fast-ai) where he applied similar techniques in the recent Freesound 2019 competition.
# 
# The Keras version of Grad-CAM is adapted from [this article](http://www.hackevolve.com/where-cnn-is-looking-grad-cam/), which in turns adapted from F.Chollet's book.

# ## 1.1 Brief Introduction on Grad-CAM
# 
# ![](http://gradcam.cloudcv.org/static/images/network.png)

# The idea of Grad-CAM visualization may be intuitively and non-formally summarized like this :
# 
# **Objective** Emphasize pixel regions (spatial information) which make the model make a decision on the final predicted class (here, diabetic retinophaty severity level). We visualize these regions using **heatmap** (as shown in the above figure).
# 
# **Method Intuition** 
# * We bellieve that most important spatial information come from the 3D-tensor of the *last convolutional layer* (just before `GlobalPooling layer`), which is the nearest spatial information flowing to the last FC layer.
# 
# * For each channel of this 3D-tensor, each activated pixel region represent important features (e.g. blood vessel / scab / cotton wool) of the input image. Note that some features are important to determine class 0 (perfectly fine blood vessel), some features are important to determine class 4 (big cotton wools). Normally, we expect each channel to capture different set of features
# 
# * To emphasize features which finally affected the final prediction, we calculate the **gradient of the final predicted class with respect to each feature.** If that feature is important to this class, it should have high gradient (i.e. increase the value of this feature, the prediction confidence increases)
# 
# * Therefore, we multiply the activated values of this 3D-tensor and gradients together, to obtain the visualized heatmap for each channel. Note that we have multi-channels, and each channel usually have multi-features.
# 
# * Finally, we combine heatmaps of all channels using simple average, and remove negative value (the `ReLu` step in the above picture) to obtain the final heatmap
# 
# For formal introduction please refer to the Author's paper. For now let us proceed to the programming part.

# In[ ]:


import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Prepare tools of the original kernel
# 
# In order to visualize the model output using Grad-CAM, we do not need to retrain model. We can just use the already-trained weights from @xhlulu original kernel directly. Therefore, in fact, we will also not need to preprocess the train/test offline here, we will just preprocess them on-the-fly. Just note that the original kernel use 224x224 image, and use the preprocess_image function below.
# 
# Nevertheless, to ensure that we load model weights correctly, I will preprocess the test data and ensure that we obtain exact same predictions as original kernel.

# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
print(train_df.shape)
print(test_df.shape)
test_df.head()


# In[ ]:


def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
#     im = im.resize((desired_size, )*2)
    
    return im


# In[ ]:


N = test_df.shape[0]
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test_df['id_code'])):
    x_test[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'
    )


# ### Displaying some original test images
# 
# Just to have an idea, let us visualize first 10 eyes in the test set. Just quickly notice that only 2/10 eyes here look fine. Below we shall also define Ben's preprocessing function as it will be easier for us human to see abnormal spots, and it is cleaner when we combine the eye picture with heatmap later.

# In[ ]:


# model.summary()
def load_image_ben_orig(path,resize=True,crop=False,norm255=True,keras=False):
    image = cv2.imread(path)
    
#     if crop:
#         image = crop_image(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#     if resize:
#         image = cv2.resize(image,(SIZE,SIZE))
        
    image=cv2.addWeighted( image,4, cv2.GaussianBlur( image , (0,0) ,  10) ,-4 ,128)
#     image=cv2.addWeighted( image,4, cv2.medianBlur( image , 10) ,-4 ,128)
    
    # NOTE plt.imshow can accept both int (0-255) or float (0-1), but deep net requires (0-1)
    if norm255:
        return image/255
    elif keras:
        #see https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py for mode
        #see https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py for inception,xception mode
        #the use of tf based preprocessing (- and / by 127 respectively) will results in [-1,1] so it will not visualize correctly (directly)
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)[0]
    else:
        return image.astype(np.int16)
    
    return image

def transform_image_ben(img,resize=True,crop=False,norm255=True,keras=False):  
    image=cv2.addWeighted( img,4, cv2.GaussianBlur( img , (0,0) ,  10) ,-4 ,128)
    
    # NOTE plt.imshow can accept both int (0-255) or float (0-1), but deep net requires (0-1)
    if norm255:
        return image/255
    elif keras:
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)[0]
    else:
        return image.astype(np.int16)
    
    return image


# In[ ]:


def display_samples(df, columns=5, rows=2, Ben=True):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
#         image_id = df.loc[i,'diagnosis']
        path = f'../input/aptos2019-blindness-detection/test_images/{image_path}.png'
        if Ben:
            img = load_image_ben_orig(path)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
#         plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(test_df, Ben=False)
display_samples(test_df, Ben=True)


# # 3. Construct model. Some hacks to get gradients.
# 
# Conceptually, we can just load pretrained model and calculate the wanted gradient and that's enough to have heatmap! However, technically speaking, the original kernel use Keras' `Sequential method` to construct a fine-tuned DenseNet model instead of Keras' `Functional method`. Unfortunately, by using `Sequential method` we cannot access the last convolutional layer directly. Therefore, we cannot calulate activations and gradients, so we need some hack.
# 
# The hack I use below is to use `Sequential method` to construct a model using shared layers and then apply pretrained weights. After that, I construct another model using `Functional method` but using the same shared layers. Since all layers are shared, the two models are exactly the same, having the same weights.

# In[ ]:


from keras import layers
from keras.models import Model
import keras.backend as K


# First, let us define the `DenseNet` backbone.

# In[ ]:


K.clear_session()
densenet = DenseNet121(
    weights=None,
    include_top=False,
    input_shape=(None,None,3)
)


# Next, we define 3 shared head layers, exactly the same types as used in original kernel. Then, construct it using `Sequential()` module also same process as the original kernel. You can see from `model.summary()` below that by using `Sequential()` module, the layer details of the backbone is hidden and we cannot use it directly. Therefore, we cannot obtain gradients here.

# In[ ]:


GAP_layer = layers.GlobalAveragePooling2D()
drop_layer = layers.Dropout(0.5)
dense_layer = layers.Dense(5, activation='sigmoid', name='final_output')


# In[ ]:


def build_model_sequential():
    model = Sequential()
    model.add(densenet)
    model.add(GAP_layer)
    model.add(drop_layer)
    model.add(dense_layer)
    return model


# In[ ]:


modelA = build_model_sequential()
modelA.load_weights('../input/aptos-data/dense_xhlulu_731.h5')

modelA.summary()


# Below, we construct another model using exactly the same (shared) layers. When pretrained weights are loaded into the first model, the second model also get the same weights (since all layers are shared)

# In[ ]:


def build_model_functional():
    base_model = densenet
    
    x = GAP_layer(base_model.layers[-1].output)
    x = drop_layer(x)
    final_output = dense_layer(x)
    model = Model(base_model.layers[0].input, final_output)
    
    return model


# Now by using functional module, we can access to all layers in the backbone which is evident when executing `model.summary()`. Since the output is too long, it is hidden, but you can press the output button to see all layers.

# In[ ]:


model = build_model_functional() # with pretrained weights, and layers we want
model.summary()


# Therefore, we can access the last convolutional layer here. Note that we may use either `conv5_block16_concat` or `relu` which is the rectified and batch-normalized version of `conv5_block16_concat`.

# ## 3.1 Just to make sure that we have the correct weights
# 
# Note that this section purpose is to make sure that we already loaded the correct weights. Since it is proven by the LB score in the previous version, in this version I just comment out everything
# 

# In[ ]:


# y_test = model.predict(x_test) > 0.5
# y_test = y_test.astype(int).sum(axis=1) - 1

# test_df['diagnosis'] = y_test
# test_df.to_csv('submission.csv',index=False)
# y_test.shape, x_test.shape


# In[ ]:


# import seaborn as sns
# import cv2

# SIZE=224
# def create_pred_hist(pred_level_y,title='NoTitle'):
#     results = pd.DataFrame({'diagnosis':pred_level_y})

#     f, ax = plt.subplots(figsize=(7, 4))
#     ax = sns.countplot(x="diagnosis", data=results, palette="GnBu_d")
#     sns.despine()
#     plt.title(title)
#     plt.show()


# In[ ]:


# create_pred_hist(y_test,title='predicted level distribution in test set')


# # 4. Real or Spurious Features?
# 
# Now we finally come to the main section of this kernel. It's time to investigate the model performance. First let us define a heatmap calculation function. As said in Introduction, codes are adapted from [this article](http://www.hackevolve.com/where-cnn-is-looking-grad-cam/), which in turns adapted from F.Chollet's book.
# 
# This function will recieve 4 arguments as inputs. (1) the image to make a prediction/visualization, remember to insert the correct preprocessed version here (2) the model (3) a layer to get gradients and (4) an auxiliary image just to combine with heatmap and visualize the final result; I use Ben's preprocessed image here since it eliminate lightning conditions in the pictures, and so easy for us to visualize the final result.

# In[ ]:


def gen_heatmap_img(img, model0, layer_name='last_conv_layer',viz_img=None,orig_img=None):
    preds_raw = model0.predict(img[np.newaxis])
    preds = preds_raw > 0.5 # use the same threshold as @xhlulu original kernel
    class_idx = (preds.astype(int).sum(axis=1) - 1)[0]
#     print(class_idx, class_idx.shape)
    class_output_tensor = model0.output[:, class_idx]
    
    viz_layer = model0.get_layer(layer_name)
    grads = K.gradients(
                        class_output_tensor ,
                        viz_layer.output
                        )[0] # gradients of viz_layer wrt output_tensor of predicted class
    
    pooled_grads=K.mean(grads,axis=(0,1,2))
    iterate=K.function([model0.input],[pooled_grads, viz_layer.output[0]])
    
    pooled_grad_value, viz_layer_out_value = iterate([img[np.newaxis]])
    
    for i in range(pooled_grad_value.shape[0]):
        viz_layer_out_value[:,:,i] *= pooled_grad_value[i]
    
    heatmap = np.mean(viz_layer_out_value, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)

    viz_img=cv2.resize(viz_img,(img.shape[1],img.shape[0]))
    heatmap=cv2.resize(heatmap,(viz_img.shape[1],viz_img.shape[0]))
    
    heatmap_color = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_SPRING)/255
    heated_img = heatmap_color*0.5 + viz_img*0.5
    
    print('raw output from model : ')
    print_pred(preds_raw)
    
    if orig_img is None:
        show_Nimages([img,viz_img,heatmap_color,heated_img])
    else:
        show_Nimages([orig_img,img,viz_img,heatmap_color,heated_img])
    
    plt.show()
    return heated_img


# Here are simple tools to easily show images and prediction

# In[ ]:


def show_image(image,figsize=None,title=None):
    
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
#     else: # crash!!
#         fig = plt.figure()
        
    if image.ndim == 2:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)
        
    if title is not None:
        plt.title(title)

def show_Nimages(imgs,scale=1):

    N=len(imgs)
    fig = plt.figure(figsize=(25/scale, 16/scale))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])
        show_image(img)
        
def print_pred(array_of_classes):
    xx = array_of_classes
    s1,s2 = xx.shape
    for i in range(s1):
        for j in range(s2):
            print('%.3f ' % xx[i,j],end='')
        print('')


# First let us test the first 10 test examples. For each test example, I show original input, Ben's preprocessed input, heatmap and combined heatmap respectively.

# In[ ]:


NUM_SAMP=10
SEED=77
layer_name = 'relu' #'conv5_block16_concat'
for i, (idx, row) in enumerate(test_df[:NUM_SAMP].iterrows()):
    path=f"../input/aptos2019-blindness-detection/test_images/{row['id_code']}.png"
    ben_img = load_image_ben_orig(path)
    input_img = np.empty((1,224, 224, 3), dtype=np.uint8)
    input_img[0,:,:,:] = preprocess_image(path)
        
    print('test pic no.%d' % (i+1))
    _ = gen_heatmap_img(input_img[0],
                        model, layer_name=layer_name,viz_img=ben_img)


# There are many interesting observations here. To name a few, 
# 
# * The 1st, 4th, 5th, 6th predictions look great
# * The 2nd prediction misses the whole big spots in the middle
# * The 3rd and 7th predictions also miss important spots
# * In the 9th image, bloody spots are all over the places and the model is able to capture 4 regions, not all.
# * The 8th and 10th images look normal to me, but in the 10th, it looks like the model capture a **spuriou (false)** feature and identify severity level 1.

# # 5. [updated] Robustness Test with Albumentation
# 
# In this new section, I show how to apply five (maybe sensible?) transforms of albumentation and test with our model in order to see that the model still give the same prediction as non-augmented or not. The sixth and ultimate augmentation is to combine all five transform altogether! You can see examples of resulted transforms one-by-one for each row below. Note that the first image is the original one.

# In[ ]:


from albumentations import *
import time

IMG_SIZE = (224,224)

'''Use case from https://www.kaggle.com/alexanderliao/image-augmentation-demo-with-albumentation/'''
def albaugment(aug0, img):
    return aug0(image=img)['image']
idx=8
image1=x_test[idx]

'''1. Rotate or Flip'''
aug1 = OneOf([
    Rotate(p=0.99, limit=160, border_mode=0,value=0), # value=black
    Flip(p=0.5)
    ],p=1)

'''2. Adjust Brightness or Contrast'''
aug2 = RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45,p=1)
h_min=np.round(IMG_SIZE[1]*0.72).astype(int)
h_max= np.round(IMG_SIZE[1]*0.9).astype(int)
print(h_min,h_max)

'''3. Random Crop and then Resize'''
#w2h_ratio = aspect ratio of cropping
aug3 = RandomSizedCrop((h_min, h_max),IMG_SIZE[1],IMG_SIZE[0], w2h_ratio=IMG_SIZE[0]/IMG_SIZE[1],p=1)

'''4. CutOut Augmentation'''
max_hole_size = int(IMG_SIZE[1]/10)
aug4 = Cutout(p=1,max_h_size=max_hole_size,max_w_size=max_hole_size,num_holes=8 )#default num_holes=8

'''5. SunFlare Augmentation'''
aug5 = RandomSunFlare(src_radius=max_hole_size,
                      num_flare_circles_lower=10,
                      num_flare_circles_upper=20,
                      p=1)#default flare_roi=(0,0,1,0.5),

'''6. Ultimate Augmentation -- combine everything'''
final_aug = Compose([
    aug1,aug2,aug3,aug4,aug5
],p=1)


img1 = albaugment(aug1,image1)
img2 = albaugment(aug1,image1)
print('Rotate or Flip')
show_Nimages([image1,img1,img2],scale=2)
# time.sleep(1)

img1 = albaugment(aug2,image1)
img2 = albaugment(aug2,image1)
img3 = albaugment(aug2,image1)
print('Brightness or Contrast')
show_Nimages([img3,img1,img2],scale=2)
# time.sleep(1)

img1 = albaugment(aug3,image1)
img2 = albaugment(aug3,image1)
img3 = albaugment(aug3,image1)
print('Rotate and Resize')
show_Nimages([img3,img1,img2],scale=2)
print(img1.shape,img2.shape)
# time.sleep(1)

img1 = albaugment(aug4,image1)
img2 = albaugment(aug4,image1)
img3 = albaugment(aug4,image1)
print('CutOut')
show_Nimages([img3,img1,img2],scale=2)
# time.sleep(1)

img1 = albaugment(aug5,image1)
img2 = albaugment(aug5,image1)
img3 = albaugment(aug5,image1)
print('Sun Flare')
show_Nimages([img3,img1,img2],scale=2)
# time.sleep(1)

img1 = albaugment(final_aug,image1)
img2 = albaugment(final_aug,image1)
img3 = albaugment(final_aug,image1)
print('All above combined')
show_Nimages([img3,img1,img2],scale=2)
print(img1.shape,img2.shape)


# Now let us see how our model reacts with each type of augmentation! Note that for this particular test image the model predict non-augmentation as level 3 (score `[0.998 1.000 0.999 0.953 0.068]`)
# 
# Since the augmentation is random, you will see different results from when I wrote this kernel. In my experiment, the model is pretty robust as it predict almost the same level everytime, except the ultimate (everything) augmentation where it sometimes confusingly predicts as level 4. Detected features are quite consistent as well.
# 
# The key benefit in my opinion is to use all these intuitions to adjust your augmentation scheme. Make the system more robust, and know the features it should know.
# 
# In fact, we can have fun test a lot more examples but I will leave the rest to you at the moment.

# In[ ]:


aug_list = [aug5, aug2, aug3, aug4, aug1, final_aug]
aug_name = ['SunFlare', 'brightness or contrast', 'crop and resized', 'CutOut', 'rotate or flip', 'Everything Combined']

idx=8
layer_name = 'relu' #'conv5_block16_concat'
for i in range(len(aug_list)):
    path=f"../input/aptos2019-blindness-detection/test_images/{test_df.iloc[idx]['id_code']}.png"
    input_img = np.empty((1,224, 224, 3), dtype=np.uint8)
    input_img[0,:,:,:] = preprocess_image(path)
    aug_img = albaugment(aug_list[i],input_img[0,:,:,:])
    ben_img = transform_image_ben(aug_img)
    
    print('test pic no.%d -- augmentation: %s' % (i+1, aug_name[i]))
    _ = gen_heatmap_img(aug_img,
                        model, layer_name=layer_name,viz_img=ben_img,orig_img=input_img[0])


# There are many more possible creative uses of this heatmap visualization which I will update in the future.
# 
# * Visualize augmented images and see if our model is robust enough (predict the same), or our augmentation makes sense or not (it preserve important information?) **[DONE.]**
# 
# * In the case of overfitting traning data, visualize training data to see some spurious features. Then design effective augmentations to eliminate that spurious features
# 
# * Visualize for each level from 0 to 4, to get an idea what features our model use to determine each severity level
# 
# That's it for now!! Hope this kernel be helpful somehow!
# 
# -- 

# In[ ]:




