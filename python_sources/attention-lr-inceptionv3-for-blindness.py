#!/usr/bin/env python
# coding: utf-8

# # Overview
# The goal is to make a nice retinopathy model by using a pretrained inception v3 as a base and retraining some modified final layers with attention. The notebook was originally written for the original Data Science Bowl for diabetic retinopathy and adapted to the new competition (and better Kaggle features!)
# 
# This can be massively improved with 
# * high-resolution images
# * better data sampling
# * ensuring there is no leaking between training and validation sets, ```sample(replace = True)``` is real dangerous
# * better target variable (age) normalization
# * pretrained models
# * attention/related techniques to focus on areas

# In[ ]:


# copy the weights and configurations for the pre-trained models (so the kernel can be offline)
get_ipython().system('mkdir ~/.keras')
get_ipython().system('mkdir ~/.keras/models')
get_ipython().system('cp ../input/keras-pretrained-models/*notop* ~/.keras/models/')
get_ipython().system('cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/')


# ### Parameters
# Here we keep track of all the parameters so we can easily change them and track performance

# In[ ]:


BATCH_SIZE = 64
IMG_SIZE = (256, 256) # slightly smaller than vgg16 normally expects
INTERMEDIATE_SIZE = (384, 384)
BRIGHTNESS_DELTA = 0.1
SATURATION_DELTA = 0.2
HUE_DELTA = 0.05
CONTRAST_DELTA = 0.1


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
import tensorflow as tf
import random

# not needed in Kaggle, but required in Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    
SEED = 2019
seed_everything(SEED)


# In[ ]:


base_image_dir = os.path.join('..', 'input', 'aptos2019-blindness-detection')
retina_df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
retina_df['path'] = retina_df['id_code'].map(lambda x: os.path.join(base_image_dir, 'train_images',
                                                         '{}.png'.format(x)))
retina_df['exists'] = retina_df['path'].map(os.path.exists)
print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')
retina_df.dropna(inplace = True)
retina_df = retina_df[retina_df['exists']]
retina_df.sample(3)


# # Examine the distribution of eye and severity

# In[ ]:


retina_df[['diagnosis']].hist(figsize = (10, 5))


# # Split Data into Training and Validation

# In[ ]:


from sklearn.model_selection import train_test_split
rr_df = retina_df[['id_code', 'diagnosis']].drop_duplicates()
train_ids, valid_ids = train_test_split(rr_df['id_code'], 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = rr_df['diagnosis'])
raw_train_df = retina_df[retina_df['id_code'].isin(train_ids)]
valid_df = retina_df[retina_df['id_code'].isin(valid_ids)]
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])


# # Balance the distribution in the training set

# In[ ]:


train_df = raw_train_df.groupby(['diagnosis']).apply(lambda x: x.sample(500, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
train_df[['diagnosis']].hist(figsize = (10, 5))


# In[ ]:


import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
import numpy as np

def tf_image_loader(out_size, 
                      horizontal_flip = True, 
                      vertical_flip = False, 
                     random_brightness = True,
                     random_contrast = True,
                    random_saturation = True,
                    random_hue = True,
                      color_mode = 'rgb',
                       preproc_func = preprocess_input,
                       on_batch = False):
    def _func(X):
        with tf.name_scope('image_augmentation'):
            with tf.name_scope('input'):
                X = tf.image.decode_png(tf.read_file(X), channels = 3 if color_mode == 'rgb' else 0)
                X = tf.image.resize_images(X, out_size)
            with tf.name_scope('augmentation'):
                if horizontal_flip:
                    X = tf.image.random_flip_left_right(X)
                if vertical_flip:
                    X = tf.image.random_flip_up_down(X)
                if random_brightness:
                    X = tf.image.random_brightness(X, max_delta = BRIGHTNESS_DELTA)
                if random_saturation:
                    X = tf.image.random_saturation(X, lower = 1-SATURATION_DELTA, upper = 1+SATURATION_DELTA)
                if random_hue:
                    X = tf.image.random_hue(X, max_delta = HUE_DELTA)
                if random_contrast:
                    X = tf.image.random_contrast(X, lower = 1-CONTRAST_DELTA, upper = 1+CONTRAST_DELTA)
                return preproc_func(X)
    if on_batch: 
        # we are meant to use it on a batch
        def _batch_func(X, y):
            return tf.map_fn(_func, X), y
        return _batch_func
    else:
        # we apply it to everything
        def _all_func(X, y):
            return _func(X), y         
        return _all_func
    
def tf_augmentor(out_size,
                intermediate_size = INTERMEDIATE_SIZE,
                 intermediate_trans = 'crop',
                 batch_size = 16,
                   horizontal_flip = True, 
                  vertical_flip = False, 
                 random_brightness = True,
                 random_contrast = True,
                 random_saturation = True,
                    random_hue = True,
                  color_mode = 'rgb',
                   preproc_func = preprocess_input,
                   min_crop_percent = 0.001,
                   max_crop_percent = 0.005,
                   crop_probability = 0.5,
                   rotation_range = 10):
    
    load_ops = tf_image_loader(out_size = intermediate_size, 
                               horizontal_flip=horizontal_flip, 
                               vertical_flip=vertical_flip, 
                               random_brightness = random_brightness,
                               random_contrast = random_contrast,
                               random_saturation = random_saturation,
                               random_hue = random_hue,
                               color_mode = color_mode,
                               preproc_func = preproc_func,
                               on_batch=False)
    def batch_ops(X, y):
        batch_size = tf.shape(X)[0]
        with tf.name_scope('transformation'):
            # code borrowed from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
            # The list of affine transformations that our image will go under.
            # Every element is Nx8 tensor, where N is a batch size.
            transforms = []
            identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
            if rotation_range > 0:
                angle_rad = rotation_range / 180 * np.pi
                angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
                transforms += [tf.contrib.image.angles_to_projective_transforms(angles, intermediate_size[0], intermediate_size[1])]

            if crop_probability > 0:
                crop_pct = tf.random_uniform([batch_size], min_crop_percent, max_crop_percent)
                left = tf.random_uniform([batch_size], 0, intermediate_size[0] * (1.0 - crop_pct))
                top = tf.random_uniform([batch_size], 0, intermediate_size[1] * (1.0 - crop_pct))
                crop_transform = tf.stack([
                      crop_pct,
                      tf.zeros([batch_size]), top,
                      tf.zeros([batch_size]), crop_pct, left,
                      tf.zeros([batch_size]),
                      tf.zeros([batch_size])
                  ], 1)
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), crop_probability)
                transforms += [tf.where(coin, crop_transform, tf.tile(tf.expand_dims(identity, 0), [batch_size, 1]))]
            if len(transforms)>0:
                X = tf.contrib.image.transform(X,
                      tf.contrib.image.compose_transforms(*transforms),
                      interpolation='BILINEAR') # or 'NEAREST'
            if intermediate_trans=='scale':
                X = tf.image.resize_images(X, out_size)
            elif intermediate_trans=='crop':
                X = tf.image.resize_image_with_crop_or_pad(X, out_size[0], out_size[1])
            else:
                raise ValueError('Invalid Operation {}'.format(intermediate_trans))
            return X, y
    def _create_pipeline(in_ds):
        batch_ds = in_ds.map(load_ops, num_parallel_calls=4).batch(batch_size)
        return batch_ds.map(batch_ops)
    return _create_pipeline


# In[ ]:


def flow_from_dataframe(idg, 
                        in_df, 
                        path_col,
                        y_col, 
                        shuffle = True, 
                        color_mode = 'rgb'):
    files_ds = tf.data.Dataset.from_tensor_slices((in_df[path_col].values, 
                                                   np.stack(in_df[y_col].values,0)))
    in_len = in_df[path_col].values.shape[0]
    while True:
        if shuffle:
            files_ds = files_ds.shuffle(in_len) # shuffle the whole dataset
        
        next_batch = idg(files_ds).repeat().make_one_shot_iterator().get_next()
        for i in range(max(in_len//32,1)):
            # NOTE: if we loop here it is 'thread-safe-ish' if we loop on the outside it is completely unsafe
            yield K.get_session().run(next_batch)


# In[ ]:



core_idg = tf_augmentor(out_size = IMG_SIZE, 
                        color_mode = 'rgb', 
                        vertical_flip = True,
                        horizontal_flip = True,
                        crop_probability=0.0, # crop doesn't work yet
                        batch_size = BATCH_SIZE) 
valid_idg = tf_augmentor(out_size = IMG_SIZE, color_mode = 'rgb', 
                         crop_probability=0.0, 
                         horizontal_flip = False, 
                         vertical_flip = False, 
                         random_brightness = False,
                         random_contrast = False,
                         random_saturation = False,
                         random_hue = False,
                         rotation_range = 0,
                        batch_size = BATCH_SIZE)

train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'diagnosis')

valid_gen = flow_from_dataframe(valid_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'diagnosis') # we can use much larger batches for evaluation


# # Validation Set
# We do not perform augmentation at all on these images

# In[ ]:


t_x, t_y = next(valid_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(np.clip(c_x*127+127, 0, 255).astype(np.uint8))
    c_ax.set_title('Severity {}'.format(c_y))
    c_ax.axis('off')


# # Training Set
# These are augmented and a real mess

# In[ ]:


t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(np.clip(c_x*127+127, 0, 255).astype(np.uint8))
    c_ax.set_title('Severity {}'.format(c_y))
    c_ax.axis('off')


# # Attention Model
# The basic idea is that a Global Average Pooling is too simplistic since some of the regions are more relevant than others. So we build an attention mechanism to turn pixels in the GAP on an off before the pooling and then rescale (Lambda layer) the results based on the number of pixels. The model could be seen as a sort of 'global weighted average' pooling. There is probably something published about it and it is very similar to the kind of attention models used in NLP.
# It is largely based on the insight that the winning solution annotated and trained a UNET model to segmenting the hand and transforming it. This seems very tedious if we could just learn attention.

# In[ ]:


from keras.applications.vgg16 import VGG16 as PTModel
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
from keras.applications.inception_v3 import InceptionV3 as PTModel
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
from keras.layers import BatchNormalization
in_lay = Input(t_x.shape[1:])
base_pretrained_model = PTModel(input_shape =  t_x.shape[1:], include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = False
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)

bn_features = BatchNormalization()(pt_features)

# here we do an attention mechanism to turn pixels in the GAP on an off

attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(1, 
                    kernel_size = (1,1), 
                    padding = 'valid', 
                    activation = 'sigmoid')(attn_layer)
# fan it out to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
               activation = 'linear', use_bias = False, weights = [up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.25)(gap)
dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
out_layer = Dense(train_df['diagnosis'].max()+1, activation = 'softmax')(dr_steps)
retina_model = Model(inputs = [in_lay], outputs = [out_layer])
retina_model.summary()


# In[ ]:


from keras.metrics import sparse_top_k_categorical_accuracy
def top_2_accuracy(in_gt, in_pred):
    return sparse_top_k_categorical_accuracy(in_gt, in_pred, k=2)

retina_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
                           metrics = ['categorical_accuracy', top_2_accuracy])


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('retina')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=6) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


get_ipython().system('rm -rf ~/.keras # clean up before starting training')
import gc
gc.enable()
gc.collect()


# In[ ]:


retina_model.fit_generator(train_gen, 
                           steps_per_epoch = train_df.shape[0]//BATCH_SIZE,
                           validation_data = valid_gen, 
                           validation_steps = valid_df.shape[0]//BATCH_SIZE,
                              epochs = 5, 
                              callbacks = callbacks_list,
                             workers = 0, # tf-generators are not thread-safe
                             use_multiprocessing=False, 
                             max_queue_size = 0
                            )


# In[ ]:


# load the best version of the model
retina_model.load_weights(weight_path)
retina_model.save('full_retina_model.h5')


# In[ ]:


##### create one fixed dataset for evaluating
from tqdm import tqdm_notebook
# fresh valid gen
valid_gen = flow_from_dataframe(valid_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'diagnosis') 
vbatch_count = (valid_df.shape[0]//BATCH_SIZE-1)
out_size = vbatch_count*BATCH_SIZE
test_X = np.zeros((out_size,)+t_x.shape[1:], dtype = np.float32)
test_Y = np.zeros((out_size,)+t_y.shape[1:], dtype = np.float32)
for i, (c_x, c_y) in zip(tqdm_notebook(range(vbatch_count)), 
                         valid_gen):
    j = i*BATCH_SIZE
    test_X[j:(j+c_x.shape[0])] = c_x
    test_Y[j:(j+c_x.shape[0])] = c_y


# # Show Attention
# Did our attention model learn anything useful?

# In[ ]:


# get the attention layer since it is the only one with a single output dim
for attn_layer in retina_model.layers:
    c_shape = attn_layer.get_output_shape_at(0)
    if len(c_shape)==4:
        if c_shape[-1]==1:
            print(attn_layer)
            break


# In[ ]:


import keras.backend as K
rand_idx = np.random.choice(range(len(test_X)), size = 6)
attn_func = K.function(inputs = [retina_model.get_input_at(0), K.learning_phase()],
           outputs = [attn_layer.get_output_at(0)]
          )
fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):
    cur_img = test_X[c_idx:(c_idx+1)]
    attn_img = attn_func([cur_img, 0])[0]
    img_ax.imshow(np.clip(cur_img[0,:,:,:]*127+127, 0, 255).astype(np.uint8))
    attn_ax.imshow(attn_img[0, :, :, 0]/attn_img[0, :, :, 0].max(), cmap = 'viridis', 
                   vmin = 0, vmax = 1, 
                   interpolation = 'lanczos')
    real_cat = test_Y[c_idx]
    img_ax.set_title('Eye Image\nCat:%2d' % (real_cat))
    pred_cat = retina_model.predict(cur_img)
    attn_ax.set_title('Attention Map\nPred:%2.2f%%' % (100*pred_cat[0,int(real_cat)]))
fig.savefig('attention_map.png', dpi = 300)


# # Evaluate the results
# Here we evaluate the results by loading the best version of the model and seeing how the predictions look on the results. We then visualize spec

# In[ ]:


from sklearn.metrics import accuracy_score, classification_report
pred_Y = retina_model.predict(test_X, batch_size = 32, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
test_Y_cat = test_Y
print('Accuracy on Test Data: %2.2f%%' % (100*accuracy_score(test_Y_cat, pred_Y_cat)))
print(classification_report(test_Y_cat, pred_Y_cat))


# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(test_Y_cat, pred_Y_cat), 
            annot=True, fmt="d", cbar = False, cmap = plt.cm.Blues, vmax = test_X.shape[0]//16)


# # ROC Curve for healthy vs sick
# Here we make an ROC curve for healthy (```severity == 0```) and sick (```severity>0```) to see how well the model works at just identifying the disease

# In[ ]:



from sklearn.metrics import roc_curve, roc_auc_score
sick_vec = test_Y_cat>0
sick_score = np.sum(pred_Y[:,1:],1)
fpr, tpr, _ = roc_curve(sick_vec, sick_score)
fig, ax1 = plt.subplots(1,1, figsize = (6, 6), dpi = 150)
ax1.plot(fpr, tpr, 'b.-', label = 'Model Prediction (AUC: %2.2f)' % roc_auc_score(sick_vec, sick_score))
ax1.plot(fpr, fpr, 'g-', label = 'Random Guessing')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');


# In[ ]:


fig, m_axs = plt.subplots(2, 4, figsize = (32, 20))
for (idx, c_ax) in enumerate(m_axs.flatten()):
    c_ax.imshow(np.clip(test_X[idx]*127+127,0 , 255).astype(np.uint8), cmap = 'bone')
    c_ax.set_title('Actual Severity: {}\n{}'.format(test_Y_cat[idx], 
                                                           '\n'.join(['Predicted %02d (%04.1f%%): %s' % (k, 100*v, '*'*int(10*v)) for k, v in sorted(enumerate(pred_Y[idx]), key = lambda x: -1*x[1])])), loc='left')
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png', dpi = 300)


# # Prediction on Test Set

# In[ ]:


test_df = pd.read_csv(os.path.join(base_image_dir, 'test.csv'))
test_df['path'] = test_df['id_code'].map(lambda x: os.path.join(base_image_dir, 'test_images',
                                                         '{}.png'.format(x)))
test_df['exists'] = test_df['path'].map(os.path.exists)
print(test_df['exists'].sum(), 'images found of', test_df.shape[0], 'total')
test_df.dropna(inplace = True)
test_df = test_df[test_df['exists']]
test_df.sample(3)


# In[ ]:


##### create one fixed dataset for evaluating
from tqdm import tqdm_notebook
# fresh valid gen
test_gen = flow_from_dataframe(valid_idg, test_df, 
                             path_col = 'path',
                            y_col = 'path' # we want the path so we can match it back up to the input
                              ) 
vbatch_count = (test_df.shape[0]//BATCH_SIZE+1)
out_size = vbatch_count*BATCH_SIZE
pred_Y = []
path_Y = []
for i, (c_x, c_y) in zip(tqdm_notebook(range(vbatch_count)), 
                         test_gen):
    j = i*BATCH_SIZE
    pred_Y[j:(j+c_x.shape[0])] = np.argmax(retina_model.predict(c_x), -1)
    path_Y[j:(j+c_x.shape[0])] = c_y


# Make it into a dictionary so we can read out each row (missing ones we assume are zero like the `submissions_sample`

# In[ ]:


prediction_dict = dict(zip(path_Y, pred_Y))
test_df['diagnosis'] = test_df['path'].map(lambda x: prediction_dict.get(x.encode(), 0))
test_df[['id_code', 'diagnosis']].to_csv('submission.csv', index=False)


# In[ ]:


test_df['diagnosis'].hist(figsize=(10, 5))
test_df['diagnosis'].value_counts()


# In[ ]:




