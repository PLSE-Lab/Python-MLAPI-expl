#!/usr/bin/env python
# coding: utf-8

# # Overview
# Here we use a pretrained model and transfer learning to try and identify the different types of proteins present in the image.
# 
# ## Beyond
# The model currently just uses all 3 channels and includes all labels although some are exceedingly rare, better image usage and stratification would definitely help

# ## Model Parameters
# We might want to adjust these later (or do some hyperparameter optimizations). It is slightly easier to keep track of parallel notebooks with different parameters if they are all at the beginning in a clear (machine readable format, see Kaggling with Kaggle (https://www.kaggle.com/kmader/kaggling-with-kaggle).

# In[ ]:


GAUSSIAN_NOISE = 0.05
# number of validation images to use
VALID_IMG_COUNT = 1500
# number of images per category to keep 
TRAIN_IMAGES_PER_CATEGORY = 500 
BASE_MODEL='InceptionV3' # ['VGG16', 'RESNET52', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121']
IMG_SIZE = (299, 299) # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 64 # [1, 8, 16, 24]
DROPOUT = 0.5
DENSE_COUNT = 256
LEARN_RATE = 3e-4
EPOCHS = 30
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
name_label_dict = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles"   ,
5:  "Nuclear bodies"   ,
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus"   ,
8:  "Peroxisomes"   ,
9:  "Endosomes"   ,
10:  "Lysosomes"   ,
11:  "Intermediate filaments",   
12:  "Actin filaments"   ,
13:  "Focal adhesion sites",   
14:  "Microtubules"   ,
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle"   ,
18:  "Microtubule organizing center" ,  
19:  "Centrosome"   ,
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions"  , 
23:  "Mitochondria"   ,
24:  "Aggresome"   ,
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}


# In[ ]:


image_df = pd.read_csv(os.path.join('../input/',
                                 'train.csv'))
print(image_df.shape[0], 'masks found')
print(image_df['Id'].value_counts().shape[0])
# just use green for now
image_df['path'] = image_df['Id'].map(lambda x: os.path.join(train_image_dir, '{}.rgb'.format(x)))
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
ax1.set_xticks(range(max_idx))
ax1.set_xticklabels([name_label_dict[k] for k in range(max_idx)], rotation=90)
for k,v in c_val.items():
    print(name_label_dict[k], 'count:', v)


# In[ ]:


# create a categorical vector
image_df['target_vec'] = image_df['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])
image_df.sample(3)


# # Split into training and validation groups
# We stratify by the number of boats appearing so we have nice balances in each set

# In[ ]:


from sklearn.model_selection import train_test_split
raw_train_df, valid_df = train_test_split(image_df, 
                 test_size = 0.3, 
                  # hack to make stratification work                  
                 stratify = image_df['Target'].map(lambda x: x[:3] if '27' not in x else '0'))
print(raw_train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')


# ### Balancing Training Classes
# We balance out the training classes a bit more by sampling (or oversampling) a fixed number with each category so we have a better balance when training. The approach is not perfect but serves as a good starting point for this problem.
# 

# In[ ]:


# keep labels with more then 50 objects
out_df_list = []
for k,v in c_val.items():
    if v>100:
        keep_rows = raw_train_df['target_list'].map(lambda x: k in x)
        out_df_list += [raw_train_df[keep_rows].sample(TRAIN_IMAGES_PER_CATEGORY, 
                                                       replace=True)]
train_df = pd.concat(out_df_list, ignore_index=True)
print(train_df.shape[0])
train_df.sample(3)


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


# ## Very dark magic
# Keras reads single images and so we have to hack it to get it to read 3 images as one color image. Please do not try this at home. This will definitely break!

# In[ ]:


try:
    # keras 2.2
    import keras_preprocessing.image as KPImage
except:
    # keras 2.1
    import keras.preprocessing.image as KPImage
    
from PIL import Image
class rgb_pil():
    @staticmethod
    def open(in_path):
        if '.rgb' in in_path:
            r_img = np.array(Image.open(in_path.replace('.rgb', '_red.png'))).astype(np.float32)
            g_img = np.array(Image.open(in_path.replace('.rgb', '_green.png'))).astype(np.float32)
            y_img = np.array(Image.open(in_path.replace('.rgb', '_yellow.png'))).astype(np.float32)
            b_img = Image.open(in_path.replace('.rgb', '_blue.png'))
            
            rgb_arr = np.stack([r_img/2+y_img/2, g_img/2+y_img/2, b_img], -1).clip(0, 255).astype(np.uint8)
            return Image.fromarray(rgb_arr)
        else:
            return Image.open(in_path)
    fromarray = Image.fromarray
KPImage.pil_image = rgb_pil


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 45, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  brightness_range = [0.75, 1.25],
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
                             path_col = 'path',
                            y_col = 'target_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE)

# used a fixed dataset for evaluating the algorithm
valid_x, valid_y = next(flow_from_dataframe(valid_idg, 
                               valid_df, 
                             path_col = 'path',
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
ax1.imshow(montage_rgb((t_x-t_x.min())/(t_x.max()-t_x.min()))[:, :, ::RGB_FLIP])
ax1.set_title('images')


# In[ ]:


fig, (m_axs) = plt.subplots(4, 4, figsize = (20, 20))
for i, c_ax in enumerate(m_axs.flatten()):
    c_ax.imshow(((t_x[i]-t_x.min())/(t_x.max()-t_x.min()))[:, ::RGB_FLIP])
    c_title = '\n'.join([name_label_dict[j] for j, v in enumerate(t_y[i]) if v>0.5])
    c_ax.set_title(c_title)
    c_ax.axis('off')


# # Build a Model
# We build the pre-trained top model and then use a global-max-pooling (we are trying to detect any ship in the image and thus max is better suited than averaging (which would tend to favor larger ships to smaller ones). 

# In[ ]:


base_pretrained_model = PTModel(input_shape =  t_x.shape[1:], 
                              include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = False


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
gmp_dr = layers.GlobalAvgPool2D()(bn_features)
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


from IPython.display import clear_output
train_gen.batch_size = BATCH_SIZE
fit_results = protein_model.fit_generator(train_gen, 
                            steps_per_epoch = train_gen.samples//BATCH_SIZE,
                      validation_data = (valid_x, valid_y), 
                      epochs = EPOCHS, 
                      callbacks = callbacks_list,
                      workers = 3)
clear_output()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.plot(fit_results.history['loss'], label='Training')
ax1.plot(fit_results.history['val_loss'], label='Validation')
ax1.legend()
ax1.set_title('Loss')
ax2.plot(fit_results.history['binary_accuracy'], label='Training')
ax2.plot(fit_results.history['val_binary_accuracy'], label='Validation')
ax2.legend()
ax2.set_title('Binary Accuracy')
ax2.set_ylim(0, 1)


# In[ ]:


protein_model.load_weights(weight_path)
protein_model.save('full_protein_model.h5')


# In[ ]:


for k, v in zip(protein_model.metrics_names, 
        protein_model.evaluate(valid_x, valid_y)):
    if k!='loss':
        print('{:40s}:\t{:2.1f}%'.format(k, 100*v))


# In[ ]:


t_x, t_y = next(train_gen)
t_yp = protein_model.predict(t_x)
fig, (m_axs) = plt.subplots(4, 4, figsize = (20, 20))
for i, c_ax in enumerate(m_axs.flatten()):
    c_ax.imshow(((t_x[i]-t_x.min())/(t_x.max()-t_x.min()))[:, ::RGB_FLIP])
    c_title = '\n'.join(['{}: Pred: {:2.1f}%'.format(name_label_dict[j], 100*t_yp[i, j]) 
                         for j, v in enumerate(t_y[i]) if v>0.5])
    c_ax.set_title(c_title)
    c_ax.axis('off')


# # Run the test data
# We use the sample_submission file as the basis for loading and running the images.

# In[ ]:


test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')
submission_df = pd.read_csv('../input/sample_submission.csv')
submission_df['path'] = submission_df['Id'].map(lambda x: 
                                                      os.path.join(test_image_dir, '{}.rgb'.format(x)))
submission_df.sample(3)


# # Setup Test Data Generator
# We use the same generator as before to read and preprocess images

# In[ ]:


test_gen = flow_from_dataframe(valid_idg, 
                               submission_df, 
                             path_col = 'path',
                            y_col = 'Predicted', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE, 
                              shuffle = False)


# In[ ]:


t_x, _ = next(test_gen)
fig, (m_axs) = plt.subplots(4, 4, figsize = (20, 20))
for i, c_ax in enumerate(m_axs.flatten()):
    c_ax.imshow(((t_x[i]-t_x.min())/(t_x.max()-t_x.min()))[:, ::RGB_FLIP])
    c_title = '\n'.join(['{}: Pred: {:2.1f}%'.format(name_label_dict[j], 100*v)
                         for j, v in enumerate(t_y[i]) if v>0.25])
    c_ax.set_title(c_title)
    c_ax.axis('off')
fig.savefig('labeled_predictions.png')


# # Prepare Submission
# Process all images (batchwise) and keep the score at the end

# In[ ]:


BATCH_SIZE = BATCH_SIZE*2 # we can use larger batches for inference
test_gen = flow_from_dataframe(valid_idg, 
                               submission_df, 
                             path_col = 'path',
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
        all_scores[c_id] = ' '.join([str(i) for i,s in enumerate(c_score) if s>0.25])


# # Show the Scores
# Here we see the scores and we have to decide about a cut-off for counting an image as ship or not. We can be lazy and pick 0.5 but some more rigorous cross-validation would definitely improve this process.

# In[ ]:


submission_df['Predicted'] = submission_df['Id'].map(lambda x: all_scores.get(x, '0'))
submission_df['Predicted'].value_counts()[:20]


# In[ ]:


out_df = submission_df[['Id', 'Predicted']]
out_df.to_csv('submission.csv', index=False)
out_df.head(20)

