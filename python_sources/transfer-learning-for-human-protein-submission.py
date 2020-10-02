#!/usr/bin/env python
# coding: utf-8

# # Overview
# Here we use a pretrained model and transfer learning to try and identify the different types of proteins present in the image.
# 
# ## Beyond
# The model currently just uses the green channel of the image (arbitrary) and includes all labels although some are exceedingly rare, better image usage and stratification would definitely help

# ## Model Parameters
# We might want to adjust these later (or do some hyperparameter optimizations). It is slightly easier to keep track of parallel notebooks with different parameters if they are all at the beginning in a clear (machine readable format, see Kaggling with Kaggle (https://www.kaggle.com/kmader/kaggling-with-kaggle).

# In[ ]:


GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# number of validation images to use
VALID_IMG_COUNT = 1000
# maximum number of training images
MAX_TRAIN_IMAGES = 15000 
BASE_MODEL='RESNET52' # ['VGG16', 'RESNET52', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121']
IMG_SIZE = (299, 299) # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 32 # [1, 8, 16, 24]
DROPOUT = 0.5
DENSE_COUNT = 128
LEARN_RATE = 1e-4
EPOCHS = 10
RGB_FLIP = 1 # should rgb be flipped when rendering images


# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util import montage
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
base_dir = '../input'
train_image_dir = os.path.join(base_dir, 'train')
test_image_dir = os.path.join(base_dir, 'test')
import gc; gc.enable() # memory is tight


# In[ ]:


image_df = pd.read_csv(os.path.join('../input/',
                                 'train.csv'))
print(image_df.shape[0], 'masks found')
print(image_df['Id'].value_counts().shape[0])
# just use green for now
image_df['green_path'] = image_df['Id'].map(lambda x: os.path.join(train_image_dir, '{}_green.png'.format(x)))
image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])
image_df.head()


# In[ ]:


from itertools import chain
from collections import Counter
all_labels = list(chain.from_iterable(image_df['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)
fig, ax1 = plt.subplots(1,1, figsize = (10, 5))
ax1.bar(n_keys, [c_val[k] for k in n_keys])
for k,v in c_val.items():
    print(k, 'count:', v)


# In[ ]:


# create a categorical vector
image_df['target_vec'] = image_df['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])
image_df.sample(3)


# # Split into training and validation groups
# We stratify by the number of boats appearing so we have nice balances in each set

# In[ ]:


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(image_df, 
                 test_size = 0.3, 
                  # hack to make stratification work                  
                 stratify = image_df['Target'].map(lambda x: x[:3] if '27' not in x else '0'))
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')


# ### Examine Number of Ship Images
# Here we examine how often ships appear and replace the ones without any ships with 0

# In[ ]:


train_df = train_df.sample(min(MAX_TRAIN_IMAGES, train_df.shape[0])) # limit size of training set (otherwise it takes too long)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
train_sum_vec = np.sum(np.stack(train_df['target_vec'].values, 0), 0)
valid_sum_vec = np.sum(np.stack(valid_df['target_vec'].values, 0), 0)
ax1.bar(n_keys, [train_sum_vec[k] for k in n_keys])
ax1.set_title('Training Distribution')
ax2.bar(n_keys, [valid_sum_vec[k] for k in n_keys])
ax2.set_title('Validation Distribution')


# # Augment Data

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
if BASE_MODEL=='VGG16':
    from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
elif BASE_MODEL=='RESNET52':
    from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
elif BASE_MODEL=='InceptionV3':
    from keras.applications.inception_v3 import InceptionV3 as PTModel, preprocess_input
elif BASE_MODEL=='Xception':
    from keras.applications.xception import Xception as PTModel, preprocess_input
elif BASE_MODEL=='DenseNet169': 
    from keras.applications.densenet import DenseNet169 as PTModel, preprocess_input
elif BASE_MODEL=='DenseNet121':
    from keras.applications.densenet import DenseNet121 as PTModel, preprocess_input
else:
    raise ValueError('Unknown model: {}'.format(BASE_MODEL))


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 45, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  brightness_range = [0.5, 1.5],
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last',
              preprocessing_function = preprocess_input)
valid_args = dict(fill_mode = 'reflect',
                   data_format = 'channels_last',
                  preprocessing_function = preprocess_input)

core_idg = ImageDataGenerator(**dg_args)
valid_idg = ImageDataGenerator(**valid_args)


# In[ ]:


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# In[ ]:


train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'green_path',
                            y_col = 'target_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE)

# used a fixed dataset for evaluating the algorithm
valid_x, valid_y = next(flow_from_dataframe(valid_idg, 
                               valid_df, 
                             path_col = 'green_path',
                            y_col = 'target_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = VALID_IMG_COUNT)) # one big batch
print(valid_x.shape, valid_y.shape)


# In[ ]:


t_x, t_y = next(train_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
fig, (ax1) = plt.subplots(1, 1, figsize = (10, 10))
ax1.imshow(montage_rgb((t_x-t_x.min())/(t_x.max()-t_x.min()))[:, :, ::RGB_FLIP], cmap='gray')
ax1.set_title('images')


# # Build a Model
# We build the pre-trained top model and then use a global-max-pooling (we are trying to detect any ship in the image and thus max is better suited than averaging (which would tend to favor larger ships to smaller ones). 

# In[ ]:


base_pretrained_model = PTModel(input_shape =  t_x.shape[1:], 
                              include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = True


# ## Setup the Subsequent Layers
# Here we setup the rest of the model which we will actually be training

# In[ ]:


from keras import models, layers
from keras.optimizers import Adam
img_in = layers.Input(t_x.shape[1:], name='Image_RGB_In')
img_noise = layers.GaussianNoise(GAUSSIAN_NOISE)(img_in)
pt_features = base_pretrained_model(img_noise)
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
bn_features = layers.BatchNormalization()(pt_features)
feature_dropout = layers.SpatialDropout2D(DROPOUT)(bn_features)
gmp_dr = layers.GlobalMaxPooling2D()(feature_dropout)
dr_steps = layers.Dropout(DROPOUT)(layers.Dense(DENSE_COUNT, activation = 'relu')(gmp_dr))
out_layer = layers.Dense(t_y.shape[1], activation = 'sigmoid')(dr_steps)

protein_model = models.Model(inputs = [img_in], outputs = [out_layer], name = 'full_model')

protein_model.compile(optimizer = Adam(lr=LEARN_RATE), 
                   loss = 'binary_crossentropy',
                   metrics = ['binary_accuracy'])

protein_model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('boat_detector')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


train_gen.batch_size = BATCH_SIZE
protein_model.fit_generator(train_gen, 
                            steps_per_epoch = train_gen.samples//BATCH_SIZE,
                      validation_data = (valid_x, valid_y), 
                      epochs = EPOCHS, 
                      callbacks = callbacks_list,
                      workers = 3)


# In[ ]:


protein_model.load_weights(weight_path)
protein_model.save('full_protein_model.h5')


# In[ ]:


for k, v in zip(protein_model.metrics_names, 
        protein_model.evaluate(valid_x, valid_y)):
    if k!='loss':
        print('{:40s}:\t{:2.1f}%'.format(k, 100*v))


# # Run the test data
# We use the sample_submission file as the basis for loading and running the images.

# In[ ]:


test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')
submission_df = pd.read_csv('../input/sample_submission.csv')
submission_df['green_path'] = submission_df['Id'].map(lambda x: 
                                                      os.path.join(test_image_dir, '{}_green.png'.format(x)))
submission_df.sample(3)


# # Setup Test Data Generator
# We use the same generator as before to read and preprocess images

# In[ ]:


test_gen = flow_from_dataframe(valid_idg, 
                               submission_df, 
                             path_col = 'green_path',
                            y_col = 'Predicted', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE, 
                              shuffle = False)


# In[ ]:


fig, m_axs = plt.subplots(3, 1, figsize = (20, 30))
for (ax1), (t_x, c_img_names) in zip(m_axs, test_gen):
    t_y = protein_model.predict(t_x)
    t_stack = ((t_x-t_x.min())/(t_x.max()-t_x.min()))[:, :, :, ::RGB_FLIP]
    ax1.imshow(montage_rgb(t_stack))
    ax1.set_title('images')
fig.savefig('test_predictions.png')


# # Prepare Submission
# Process all images (batchwise) and keep the score at the end

# In[ ]:


BATCH_SIZE = BATCH_SIZE*2 # we can use larger batches for inference
test_gen = flow_from_dataframe(valid_idg, 
                               submission_df, 
                             path_col = 'green_path',
                            y_col = 'Id', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE, 
                              shuffle = False)


# In[ ]:


from tqdm import tqdm_notebook
all_scores = dict()
for _, (t_x, t_names) in zip(tqdm_notebook(range(test_gen.n//BATCH_SIZE+1)),
                            test_gen):
    t_y = protein_model.predict(t_x)
    for c_id, c_score in zip(t_names, t_y):
        all_scores[c_id] = ' '.join([str(i) for i,s in enumerate(c_score) if s>0.5])


# # Show the Scores
# Here we see the scores and we have to decide about a cut-off for counting an image as ship or not. We can be lazy and pick 0.5 but some more rigorous cross-validation would definitely improve this process.

# In[ ]:


submission_df['Predicted'] = submission_df['Id'].map(lambda x: all_scores.get(x, '0'))
submission_df['Predicted'].value_counts()[:20]


# In[ ]:


out_df = submission_df[['Id', 'Predicted']]
out_df.to_csv('submission.csv', index=False)
out_df.head(20)

