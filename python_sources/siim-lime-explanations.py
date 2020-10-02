#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **Running out of TPU quota? Want to improve your model?**
# 
# Don't worry. There is always something to work with in your model (until it is not ideal). This notebook shows LIME method applied to SIIM melanoma competition data. Although LIME (Local Interpretable Model-agnostic Explanations) is a great tool to explain what machine learning classifiers (or models) are doing, there is a link between citrus consumption and melanoma (link in refferences)- incredible!

# ![lime](https://www.cancertherapyadvisor.com/wp-content/uploads/sites/12/2018/12/citrusmelanomariskskincanc_794020.jpg)

# ## Refferences:
# * https://towardsdatascience.com/understanding-model-predictions-with-lime-a582fdff3a3b - LIME described
# * https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5 - also LIME described
# * https://github.com/marcotcr/lime - package lime on github
# * https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html segmentation algorithms
# * https://www.cancertherapyadvisor.com/home/cancer-topics/skin-cancer/link-between-citrus-consumption-and-melanoma/ - additional article
# 
# ## Versions:
# * V1-V4: Testing and choosing segmentation algorithm
# * V5: Added predictions from csv 
# * V6: Added case comparison vs SHAP explanations
# * V7: Model with microscope transform applied to training data (wow!)
# * V9: Re-run with DEFAULT_NUM_SAMPLES = 100 (checking stability of results)
# * V10: Re-run with DEFAULT_NUM_SAMPLES = 500 (checking stability of results)
# * V11: Added transform view
# * V12: Bug fixed
# * V13: Investigating @cdeotte model fold 0 from notebook: https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords

# ## Loading packages, initial config, model, dataset and predictions

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import math, re, os
import tensorflow as tf, tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
import efficientnet.tfkeras as efn
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from sklearn import metrics
from skimage.segmentation import mark_boundaries
from sklearn.metrics import confusion_matrix
import time

AUTO = tf.data.experimental.AUTOTUNE
GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-384x384')
GCS_PATH2 = KaggleDatasets().get_gcs_path('isic2019-384x384')
IMAGE_SIZE = [384, 384]
BATCH_SIZE = 128
#VALIDATION_FILENAMES = [tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')[fi] for fi in [ 0,  3, 12, 15, 21, 25]]
DEFAULT_NUM_SAMPLES = 100


# ### Model
# Model used in this notebook is loaded from great @cdeotte notebook https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords

# In[ ]:


get_ipython().system('pip install -q efficientnet >> /dev/null')


# In[ ]:


EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6]
import efficientnet.tfkeras as efn

def build_model(dim=128, ef=0):
    inp = tf.keras.layers.Input(shape=(dim,dim,3))
    base = EFNS[ef](input_shape=(dim,dim,3),weights='imagenet',include_top=False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])
    return model

model = build_model(dim=IMAGE_SIZE[0],ef=6)
model.load_weights('/kaggle/input/triple-stratified-kfold-with-tfrecords/fold-0.h5')


# ### Dataset pipelines

# In[ ]:


ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, DIM=256):    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    XDIM = DIM%2 #fix for size 331
    
    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM, DIM,3])

def read_labeled_tfrecord(example):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),
        'target'                       : tf.io.FixedLenFeature([], tf.int64)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['target']


def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['image_name'] if return_image_name else 0

 
def prepare_image(img, augment=True, dim=256):    
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    if augment:
        img = transform(img,DIM=dim)
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
                      
    img = tf.reshape(img, [dim,dim, 3])
            
    return img

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

def get_dataset(files, augment = False, shuffle = False, repeat = False, 
                labeled=True, return_image_names=True, batch_size=16, dim=256):
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()
    
    if repeat:
        ds = ds.repeat()
    
    if shuffle: 
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        
    if labeled: 
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 
                    num_parallel_calls=AUTO)      
    
    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, dim=dim), 
                                               imgname_or_label), 
                num_parallel_calls=AUTO)
    
    ds = ds.batch(batch_size * REPLICAS)
    ds = ds.prefetch(AUTO)
    return ds

SEED = 42
FOLDS = 5
REPLICAS = 1
TTA = 11
from sklearn.model_selection import KFold
skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)

for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):
    if fold == 0:
        files_valid = tf.io.gfile.glob([GCS_PATH + '/train%.2i*.tfrec'%x for x in idxV])


NUM_VALIDATION_IMAGES = count_data_items(files_valid)
print('Dataset: {} validation images'.format(NUM_VALIDATION_IMAGES))


# ### Predictions
# This notebook runs without TPU, so doing predictions could take a long time. Therefore we will previously writed predictions to csv on validation dataset (after training model).

# In[ ]:


#%%time
#N_SAMPLE = 2000
#ds = get_dataset().unbatch().batch(N_SAMPLE)
#predictions = model.predict(next(iter(ds)))

#ds = get_dataset().unbatch().batch(N_SAMPLE).map(lambda image,image_name, target: image_name)
#image_names = next(iter(ds)).numpy().astype('U')
#ds = get_dataset().unbatch().batch(N_SAMPLE).map(lambda image,image_name, target: target)
#targets = next(iter(ds)).numpy().astype('int32')


# In[ ]:


#df = pd.DataFrame({"image_name": image_names, "prob": np.concatenate(predictions),  "prediction": np.concatenate(np.round(predictions)), "target": targets})
df = pd.read_csv('/kaggle/input/triple-stratified-kfold-with-tfrecords/oof.csv')
df = df[df['fold'].isin([0])]
df.head()


# ## LIME
# We will use latest version of package directly from github (although older version is available on Kaggle and via pip install):
# 

# In[ ]:


get_ipython().system('pip install -q git+https://github.com/marcotcr/lime.git')


# In[ ]:


from lime import lime_image


# ## Segmentation algorithms
# Before applying LIME, note that one of the major parts of this method is doing segmentation (the division of the image into smaller parts called **superpixels**). Let's see how most common algorithms segments our SIIM example image:

# In[ ]:


img = next(iter(get_dataset(files_valid, augment=False, repeat=False, dim=IMAGE_SIZE[0],
        labeled=True, return_image_names=True).unbatch().batch(1).map(lambda image, image_name: image))).numpy().squeeze()
plt.imshow(img)


# In[ ]:


from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

segments_fz = felzenszwalb(img, scale=200, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=50, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=2, max_dist=50, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=50, compactness=0.001)

print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()


# Looks like Felzenszwalb method doesn't handle well hairs, so we will use SLIC segmenation with custom parameters. I prepared wrapper for standard lime functions, so explanation using LIME of single SIIM image looks like this:

# In[ ]:


def segment_fn(image):
    return slic(image, n_segments=50, compactness=10, sigma=1)

DIM = IMAGE_SIZE[0]
circle = 1-cv2.circle((np.ones([DIM, DIM, 3])).astype(np.uint8),(DIM//2, DIM//2),np.random.randint(DIM//2 - 3, DIM//2 + 5),
                    (0, 0, 0),-1)
def get_explanations(image_names, num_samples=  DEFAULT_NUM_SAMPLES, random_state = 0, progress_bar = False):

    n_img = len(image_names)
    id_img = 1
    
    for image_name in image_names:
        explainer = lime_image.LimeImageExplainer(random_state = random_state)
        
        img = cv2.imread('/kaggle/input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/'+image_name+'.jpg', cv2.IMREAD_UNCHANGED)
        width = IMAGE_SIZE[0]
        height = IMAGE_SIZE[1]
        dim = (width, height)
        img = cv2.resize(img, dim)
        img = img/255.0
        
        if df[df['image_name'].isin([image_name])].shape[0]==0:# if case wasnt in validation data
            prob = np.concatenate(model.predict(tf.expand_dims(img,axis = 0)))
            prd = np.round(prob)
            df2  = pd.read_csv("/kaggle/input/melanoma-merged-external-data-512x512-jpeg/marking.csv")
            target =df2[df2['image_id'].isin([image_name])][['target']].values[0][0]
        else:
            prd = df[df['image_name'].isin([image_name])][['pred']].values[0][0]
            target = df[df['image_name'].isin([image_name])][['target']].values[0][0]
        
        plot_num_cols = 4
        sp = plt.subplot(n_img, plot_num_cols,id_img)
        id_img+=1
        title = 'original (target = '+str(target)+')'
        sp.set_title(title)
        sp.set_ylabel(image_name+'.jpg')
        sp.imshow(img)

        #sp = plt.subplot(n_img, plot_num_cols,id_img)
        #id_img+=1
        #title = 'transformed (microscope)'
        #img*=circle
        #sp.set_title(title)
        #sp.imshow(img)        

        sp = plt.subplot(n_img, plot_num_cols,id_img)
        id_img+=1
        title = 'segmentation (prediction = '+str(prd)+')'
        
        sp.set_title(title)
        sp.imshow(mark_boundaries(img, segment_fn(img)))
        
        explanation = explainer.explain_instance(img, model.predict, num_samples=num_samples, segmentation_fn = segment_fn,progress_bar=progress_bar)
        temp, mask = explanation.get_image_and_mask(0, positive_only=False,  num_features=5, hide_rest=False, min_weight=0.0)
        sp = plt.subplot(n_img, plot_num_cols,id_img)
        sp.set_title('positive and negative regions')
        id_img+=1
        sp.imshow(mark_boundaries(temp, mask))
        
        temp, mask = explanation.get_image_and_mask(0, positive_only=True if round(prd) == 1 else False, negative_only = True if round(prd) == 0 else False,  num_features=1, hide_rest=False, min_weight=0.0)
        sp = plt.subplot(n_img, plot_num_cols,id_img)
        sp.set_title('top '+ ('positive' if round(prd) == 1 else 'negative') + ' region')
        id_img+=1
        sp.imshow(mark_boundaries(temp, mask))


# In[ ]:


img_list = ['ISIC_2637011']
plt.rcParams['figure.figsize'] = [18, 5*len(img_list)]
get_explanations(img_list, num_samples=  DEFAULT_NUM_SAMPLES, random_state = 0, progress_bar = True)


# Starting from left, we have:
# * original image (described with actual target), 
# * segmented image (described with actual prediction), 
# * top 5 **positive** and **negative** regions. We have only one class to explain (target) so: positive regions (marked as green) are these superpixels, which, when activated, are increasing the probability in model.predict() output. Respectively  negative regions (marked as red) are these superpixels, which, when activated, are decreasing the probability in model.predict() output.
# * and top decisive region (positive if prediction = 1, negative if prediction = 0)
# 
# Let's go further and find out, how our model is working on another examples from dataset! Note that, **num_samples** is one of crucial parameter for this method. More samples, better explanations. In fact, maximum number of samples could be 2^num_superpixels. We will use num_samples = 100.

# # Results diagnosis
# Model used in this notebook, scores about 0.896 auc on validation dataset:

# In[ ]:


y = df['target']
prob = df['pred']
fpr, tpr, thresholds = metrics.roc_curve(y, prob, pos_label=1)
metrics.auc(fpr, tpr)


# There are 6397 true negatives, 23 true positives, 23 false positives and 93 (!!) false negatives:

# In[ ]:


tn, fp, fn, tp = confusion_matrix(df['target'], round(df['pred'])).ravel()
(tn, fp, fn, tp)


# ## True negatives

# In[ ]:


tn = df[df['target'].isin([0]) & round(df['pred']).isin([0])]
tn.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "suspects = tn.head()['image_name']\nplt.rcParams['figure.figsize'] = [18, 5*len(suspects)]\nget_explanations(suspects)")


# ## True positives 

# In[ ]:


tp = df[df['target'].isin([1]) & round(df['pred']).isin([1])]
tp.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "suspects = tp.head(20)['image_name']\nplt.rcParams['figure.figsize'] = [18, 5*len(suspects)]\nget_explanations(suspects)")


# ## False positives

# In[ ]:


fp = df[df['target'].isin([0]) & round(df['pred']).isin([1])]
fp


# In[ ]:


get_ipython().run_cell_magic('time', '', "suspects = fp.head(20)['image_name']\nplt.rcParams['figure.figsize'] = [18, 5*len(suspects)]\nget_explanations(suspects)")


# ## False negatives

# In[ ]:


fn = df[df['target'].isin([1]) & round(df['pred']).isin([0])]
fn


# In[ ]:


suspects = fn.head(20)['image_name']
plt.rcParams['figure.figsize'] = [18, 5*len(suspects)]
get_explanations(suspects)


# At the first glance, very similar story to FP. So many cases, where skin colors are decideable!

# ## Case compared to SHAP
# As an exercise, I want to compare LIME results with SHAP with some case described here:
# https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/163425

# In[ ]:


img_list = ['ISIC_0232101']
plt.rcParams['figure.figsize'] = [18, 5*len(img_list)]
get_explanations(img_list, num_samples=  DEFAULT_NUM_SAMPLES, random_state = 0, progress_bar = True)


# # Conclusions
# First of all, this example shows that there are many ideas to try with this competiton:
# * microscope transformation 
# * unmicroscope augmentation (fill dark corners with average color)
# * batch normalization,
# * hair augmentation
# 
# And many others that maybe I didn't see!
# 
# Second thing, it's worth to save your model_weights or model after training. Thank you for reading this notebook!
