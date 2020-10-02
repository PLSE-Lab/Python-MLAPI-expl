#!/usr/bin/env python
# coding: utf-8

# # Overview
# Here we use the montgomery dataset for Tuberculosis (not very healthy lungs) since it includes lung segmentations as a basis for learning how to segment lungs in the pneumonia dataset. We then generate masks for all of the images which can be used in future steps for detecting pneumonia better
# 
# 1. Organize the Training Data for Segmentation
# 1. Build Augmentation Pipeline and Generators
# 1. Build the U-Net Model
# 1. Train the Model
# 1. Adapt model for full images
# 1. Apply to RSNA Data
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import tensorflow as tf # for tensorflow based registration
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.util.montage import montage2d
import os
from cv2 import imread, createCLAHE # read and equalize images
import cv2
from glob import glob
import matplotlib.pyplot as plt
from IPython.display import Image
from keras.utils.vis_utils import model_to_dot
def show_model(in_model):
    f = model_to_dot(in_model, show_shapes=True, rankdir='UD')
    return Image(f.create_png())


# In[ ]:


cxr_paths = glob(os.path.join('..', 'input', 'pulmonary-chest-xray-abnormalities',
                              'Montgomery', 'MontgomerySet', '*', '*.png'))
cxr_images = [(c_path, 
               [os.path.join('/'.join(c_path.split('/')[:-2]),'ManualMask','leftMask', os.path.basename(c_path)),
               os.path.join('/'.join(c_path.split('/')[:-2]),'ManualMask','rightMask', os.path.basename(c_path))]
              ) for c_path in cxr_paths]
print('CXR Images', len(cxr_paths), cxr_paths[0])
print(cxr_images[0])


# In[ ]:


from skimage.io import imread as imread_raw
from skimage.transform import resize
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore', category=UserWarning, module='skimage') # skimage is really annoying
OUT_DIM = (512, 512)
def imread(in_path, apply_clahe = False):
    img_data = imread_raw(in_path)
    n_img = (255*resize(img_data, OUT_DIM, mode = 'constant')).clip(0,255).astype(np.uint8)
    if apply_clahe:
        clahe_tool = createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        n_img = clahe_tool.apply(n_img)
    return np.expand_dims(n_img, -1)


# In[ ]:


img_vol, seg_vol = [], []
for img_path, s_paths in tqdm(cxr_images):
    img_vol += [imread(img_path)]    
    seg_vol += [np.max(np.stack([imread(s_path, apply_clahe = False) for s_path in s_paths],0),0)]
img_vol = np.stack(img_vol,0)
seg_vol = np.stack(seg_vol,0)
print('Images', img_vol.shape, 'Segmentations', seg_vol.shape)


# In[ ]:


np.random.seed(2018)
t_img, m_img = img_vol[0], seg_vol[0]

fig, (ax_img, ax_mask) = plt.subplots(1,2, figsize = (12, 6))
ax_img.imshow(np.clip(255*t_img, 0, 255).astype(np.uint8) if t_img.shape[2]==3 else t_img[:,:,0],
              interpolation = 'none', cmap = 'bone')
ax_mask.imshow(m_img[:,:,0], cmap = 'bone')


# # Make a Simple Model
# Here we make a simple U-Net to create the lung segmentations

# In[ ]:


from keras.layers import Conv2D, Activation, Input, UpSampling2D, concatenate, BatchNormalization
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal
from keras import layers
from keras.models import Model
def c2(x_in, nf, strides=1, dr = 1):
    x_out = Conv2D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', 
                   activation='linear',
                   strides=strides, dilation_rate=(dr, dr))(x_in)
    x_out = LeakyReLU(0.1)(x_out)
    return x_out
def unet_enc(vol_size, enc_nf, pre_filter = 8):
    src = Input(shape=vol_size + (1,), name = 'EncoderInput')
    # down-sample path.
    x_in = BatchNormalization(name = 'NormalizeInput')(src)
    x_in = c2(x_in, pre_filter, 1)
    x0 = c2(x_in, enc_nf[0], 2)  
    x1 = c2(x0, enc_nf[1], 2)  
    x2 = c2(x1, enc_nf[2], 2)  
    x3 = c2(x2, enc_nf[3], 2) 
    x4 = c2(x3, enc_nf[4], 2) 
    return Model(inputs = [src], 
                outputs = [x_in, x0, x1, x2, x3, x4],
                name = 'UnetEncoder')


# In[ ]:


i_low = layers.Input((20, 22, 12))
i_high = layers.Input((40, 44, 4))


# In[ ]:


def up_cat_conv(x_low_level, x_high_level, filter_count):
    x = layers.UpSampling2D()(x_low_level)
    x = layers.concatenate([x, x_high_level])
    return c2(x, filter_count)
out_model = Model(inputs = [i_low, i_high], outputs=[up_cat_conv(i_low, i_high, 8)])
show_model(out_model)


# In[ ]:


def gau(x_low_level, x_high_level, filter_count):
    """
    The global attention upsample to replace the up_cat_conv element
    """
    low_feat = c2(x_low_level, filter_count)
    high_gap = layers.GlobalAveragePooling2D()(x_high_level)
    high_feat = layers.Dense(filter_count, activation='linear', use_bias=False)(high_gap)
    high_feat = layers.BatchNormalization()(high_feat)
    high_feat = layers.LeakyReLU(0.1)(high_feat)
    high_feat = layers.Reshape((1, 1, -1))(high_feat)
    high_feat_gate = layers.UpSampling2D((x_low_level._keras_shape[1], 
                                     x_low_level._keras_shape[2]))(high_feat)
    gated_low = layers.multiply([low_feat, high_feat_gate])
    gated_low = c2(gated_low, filter_count)
    gated_high = layers.UpSampling2D((2, 2))(gated_low)
    high_clamped = c2(x_high_level, filter_count)
    return layers.add([gated_high, high_clamped])
out_model = Model(inputs = [i_low, i_high], outputs=[gau(i_low, i_high, 8)])
show_model(out_model)


# In[ ]:


up_cat_conv = gau


# In[ ]:


from keras.models import Model
def unet(vol_size, enc_nf, dec_nf, full_size=True, edge_crop=48):
    """
    unet network for voxelmorph 
    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]
            TODO: make this flexible.
        full_size
    """

    # inputs
    raw_src = Input(shape=vol_size + (1,), name = 'ImageInput')
    src = layers.GaussianNoise(0.2)(raw_src)
    enc_model = unet_enc(vol_size, enc_nf)
    # run the same encoder on the source and the target and concatenate the output at each level
    x_in, x0, x1, x2, x3, x4 = [s_enc for s_enc in enc_model(src)]
    x = c2(x4, dec_nf[0])
    x = up_cat_conv(x, x3, dec_nf[1])
    x = up_cat_conv(x, x2, dec_nf[2])
    x = up_cat_conv(x, x1, dec_nf[3])
    x = up_cat_conv(x, x0, dec_nf[4])
    x = up_cat_conv(x, x_in, dec_nf[5])

    # transform the results into a flow.
    y_seg = Conv2D(1, kernel_size=3, padding='same', name='lungs', activation='sigmoid')(x)
    y_seg = layers.Cropping2D((edge_crop, edge_crop))(y_seg)
    y_seg = layers.ZeroPadding2D((edge_crop, edge_crop))(y_seg)
    # prepare model
    model = Model(inputs=[raw_src], outputs=[y_seg])
    return model


# In[ ]:


# use the predefined depths
nf_enc=[16,32,32,64,128]
nf_dec=[32,32,32,32,32,16,16,2]
net = unet(OUT_DIM, nf_enc, nf_dec)
# ensure the model roughly works
a = net.predict([np.zeros((1,)+OUT_DIM+(1,))])
print(a.shape)
net.summary()


# In[ ]:


show_model(net)


# In[ ]:


from keras.optimizers import Adam
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

reg_param = 1.0
lr = 3e-4
dice_bce_param = 0.3
use_dice = True

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return dice_bce_param*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

net.compile(optimizer=Adam(lr=lr), 
              loss=[dice_p_bce], 
           metrics = [true_positive_rate, 'binary_accuracy'])


# # Create Training Data Generator
# Here we make a tool to generate training data from the X-ray scans

# In[ ]:


from sklearn.model_selection import train_test_split
train_vol, test_vol, train_seg, test_seg = train_test_split((img_vol-127.0)/127.0, 
                                                            (seg_vol>127).astype(np.float32), 
                                                            test_size = 0.2, 
                                                            random_state = 2018)
print('Train', train_vol.shape, 'Test', test_vol.shape, test_vol.mean(), test_vol.max())
print('Seg', train_seg.shape, train_seg.max(), np.unique(train_seg.ravel()))
fig, (ax1, ax1hist, ax2, ax2hist) = plt.subplots(1, 4, figsize = (20, 4))
ax1.imshow(test_vol[0, :, :, 0])
ax1hist.hist(test_vol.ravel())
ax2.imshow(test_seg[0, :, :, 0]>0.5)
ax2hist.hist(train_seg.ravel());


# ## Adding Augmentation
# Here we use augmentation to get more data into the model

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 5, 
                  width_shift_range = 0.05, 
                  height_shift_range = 0.05, 
                  shear_range = 0.01,
                  zoom_range = [0.8, 1.2],  
               # anatomically it doesnt make sense, but many images are flipped
                  horizontal_flip = True,  
                  vertical_flip = False,
                  fill_mode = 'nearest',
               data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)

def gen_augmented_pairs(in_vol, in_seg, batch_size = 16):
    while True:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_vol = image_gen.flow(in_vol, batch_size = batch_size, seed = seed)
        g_seg = image_gen.flow(in_seg, batch_size = batch_size, seed = seed)
        for i_vol, i_seg in zip(g_vol, g_seg):
            yield i_vol, i_seg


# In[ ]:


train_gen = gen_augmented_pairs(train_vol, train_seg, batch_size = 16)
test_gen = gen_augmented_pairs(test_vol, test_seg, batch_size = 16)
train_X, train_Y = next(train_gen)
test_X, test_Y = next(test_gen)
print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)


# ### Training Data

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage2d(train_X[:, :, :, 0]), cmap = 'bone')
ax1.set_title('CXR Image')
ax2.imshow(montage2d(train_Y[:, :, :, 0]), cmap = 'bone')
ax2.set_title('Seg Image')


# ### Validation Data

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage2d(test_X[:, :, :, 0]), cmap = 'bone')
ax1.set_title('CXR Image')
ax2.imshow(montage2d(test_Y[:, :, :, 0]), cmap = 'bone')
ax2.set_title('Seg Image')


# ## Show Untrained Results
# Here we show random untrained results

# In[ ]:


from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
try:
    from skimage.util.montage import montage2d
except:
    from skimage.util import montage2d
def add_boundary(in_img, in_seg, cmap = 'bone', norm = True, add_labels = True):
    if norm:
        n_img = (1.0*in_img-in_img.min())/(1.1*(in_img.max()-in_img.min()))
    else:
        n_img = in_img
    rgb_img = plt.cm.get_cmap(cmap)(n_img)[:, :, :3]
    if add_labels:
        return label2rgb(image = rgb_img, label = in_seg.astype(int), bg_label = 0)
    else:
        return mark_boundaries(image = rgb_img, label_img = in_seg.astype(int), color = (0, 1, 0), mode = 'thick')
def show_full_st(in_img, in_seg, gt_seg):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))
    out_mtg = add_boundary(montage2d(in_img[:, :, :, 0]), 
                           montage2d(gt_seg[:, :, :, 0]>0.5))
    ax1.imshow(out_mtg)
    ax1.set_title('Ground Truth')
    out_mtg = add_boundary(montage2d(in_img[:, :, :, 0]), 
                           montage2d(in_seg[:, :, :, 0]>0.5))
    ax2.imshow(out_mtg)
    ax2.set_title('Prediction')
    out_mtg = montage2d(in_seg[:, :, :, 0]-gt_seg[:, :, :, 0])
    ax3.imshow(out_mtg, cmap='RdBu', vmin=-1, vmax=1)
    ax3.set_title('Difference')
def show_examples(n=1, with_roi = True):
    roi_func = lambda x: x[:, 
                               OUT_DIM[0]//2-32:OUT_DIM[0]//2+32,
                               OUT_DIM[1]//2-64:OUT_DIM[1]//2,
                               :
                              ]
    for (test_X, test_Y), _ in zip(test_gen, range(n)):
        seg_Y = net.predict(test_X)
        show_full_st(test_X, seg_Y, test_Y)
        show_full_st(roi_func(test_X), roi_func(seg_Y), roi_func(test_Y))

show_examples(1)


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('cxr_reg')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=5, 
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=25) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


from IPython.display import clear_output
loss_history = net.fit_generator(train_gen, 
                  steps_per_epoch=len(train_vol)//train_X.shape[0],
                  epochs = 75,
                  validation_data = (test_vol, test_seg),
                  callbacks=callbacks_list
                 )
clear_output()


# In[ ]:


net.load_weights(weight_path)
net.save('full_model.h5')


# In[ ]:


for k, v in zip(net.metrics_names, 
                net.evaluate(test_vol, test_seg)):
    print('{}: {:.1%}'.format(k, v))


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
ax1.plot(loss_history.history['loss'], '-', label = 'Loss')
ax1.plot(loss_history.history['val_loss'], '-', label = 'Validation Loss')
ax1.legend()

ax2.plot(100*np.array(loss_history.history['binary_accuracy']), '-', 
         label = 'Accuracy')
ax2.plot(100*np.array(loss_history.history['val_binary_accuracy']), '-',
         label = 'Validation Accuracy')
ax2.legend()


# ### Show results on the training data

# In[ ]:


show_examples(2)


# # Apply to RSNA Data
# Here we load the RSNA data and apply the model to all of the images

# In[ ]:


import pydicom
from glob import glob
base_rsna_dir = os.path.join('..', 'input', 'rsna-pneumonia-detection-challenge')
test_mean, test_std = test_X.mean(), test_X.std()
def read_dicom_as_float(in_path):
    out_mat = pydicom.read_file(in_path).pixel_array
    norm_mat = (out_mat-1.0*np.mean(out_mat))/np.std(out_mat)
    # make the RSNA distribution look like the training distribution
    norm_mat = norm_mat*test_std+test_mean
    return np.expand_dims(norm_mat, -1).astype(np.float32)
all_rsna_df = pd.DataFrame({'path': glob(os.path.join(base_rsna_dir, 
                                                      'stage_*_images', '*.dcm'))})
all_rsna_df.sample(3)


# In[ ]:


from keras import layers
in_shape = read_dicom_as_float(all_rsna_df.iloc[0,0]).shape
in_img = layers.Input(in_shape, name='DICOMInput')
scale_factor = (2,2)
ds_dicom = layers.AvgPool2D(scale_factor)(in_img)
unet_out = net(ds_dicom)
us_out = layers.UpSampling2D(scale_factor)(unet_out)
unet_big = Model(inputs=[in_img], outputs=[us_out])
unet_big.summary()


# In[ ]:


fig, m_axs = plt.subplots(2, 3, figsize = (10, 8))
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
                            all_rsna_df.sample(6).iterrows()):
    c_img = read_dicom_as_float(c_row['path'])
    c_seg = unet_big.predict(np.expand_dims(c_img, 0))[0]
    c_ax.imshow(add_boundary(c_img[:, :, 0], c_seg[:, :, 0]>0.5))
    c_ax.axis('off')


# ## Make Predictions and Export Zip
# Here we make all of the predictions and create a zip file with all of the masks

# In[ ]:


import zipfile as zf
from io import BytesIO
from PIL import Image
batch_size = 12
with zf.ZipFile('masks.zip', 'w') as f:
    for i, c_rows in tqdm(all_rsna_df.groupby(lambda x: x//batch_size)):
        cur_x = np.stack(c_rows['path'].map(read_dicom_as_float), 0)
        cur_pred = unet_big.predict(cur_x)>0.5
        for out_img, (_, c_row) in zip(cur_pred[:, :, :, 0], c_rows.iterrows()):
            arc_name = os.path.relpath(c_row['path'], base_rsna_dir)
            arc_name, _ = os.path.splitext(arc_name)
            out_pil_obj = Image.fromarray((255*out_img).astype(np.uint8))
            out_obj = BytesIO()
            out_pil_obj.save(out_obj, format='png')
            out_obj.seek(0)
            f.writestr('{}.png'.format(arc_name), out_obj.read(), zf.ZIP_STORED)


# In[ ]:


get_ipython().system('ls -lh *.zip')

