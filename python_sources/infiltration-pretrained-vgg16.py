#!/usr/bin/env python
# coding: utf-8

# # Overview
# This is just a simple first attempt at a model using VGG16 as a basis and attempting to do classification directly on the chest x-ray using low-resolution images (192x192)
# 
# This can be massively improved with 
# * high-resolution images
# * better data sampling
# * ensuring there is no leaking between training and validation sets, ```sample(replace = True)``` is real dangerous
# * pretrained models
# * attention/related techniques to focus on areas

# ### Copy
# copy the weights and configurations for the pre-trained models

# In[35]:


get_ipython().system('mkdir ~/.keras')
get_ipython().system('mkdir ~/.keras/models')
get_ipython().system('cp ../input/keras-pretrained-models/*notop* ~/.keras/models/')
get_ipython().system('cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/')
get_ipython().system('cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob 
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from skimage.util.montage import montage2d
from skimage.io import imread
base_dir = os.path.join('..', 'input', 'pulmonary-chest-xray-abnormalities')
all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')
all_xray_df.sample(5)


# ## Preprocessing
# Turn the HDF5 into a data-frame and a folder full of TIFF files

# In[6]:


all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input', 'data',  'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df['infiltration'] = all_xray_df['Finding Labels'].map(lambda x: 'Infiltration' in x)
all_xray_df.sample(3)


# # Examine the distributions
# Show how the data is distributed and why we need to balance it

# In[7]:


all_xray_df[['Patient Age', 'Patient Gender', 'infiltration']].hist(figsize = (10, 5))


# # Split Data into Training and Validation

# In[26]:


more_balanced_df = all_xray_df.sample(12000)


# In[56]:


from sklearn.model_selection import train_test_split
raw_train_df, test_valid_df = train_test_split(more_balanced_df, 
                                   test_size = 0.30, 
                                   random_state = 2018,
                                   stratify = more_balanced_df[['infiltration', 'Patient Gender']])
valid_df, test_df = train_test_split(test_valid_df, 
                                   test_size = 0.40, 
                                   random_state = 2018,
                                   stratify = test_valid_df[['infiltration', 'Patient Gender']])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0], 'test', test_df.shape[0])
raw_train_df.sample(1)


# # Balance the distribution in the training set

# In[29]:


train_df = raw_train_df.groupby(['infiltration']).apply(lambda x: x.sample(3000, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
train_df[['infiltration', 'Patient Age']].hist(figsize = (10, 5))


# In[30]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
ppi = lambda x: Image.fromarray(preprocess_input(np.array(x).astype(np.float32)))
IMG_SIZE = (128, 128) # slightly smaller than vgg16 normally expects
core_idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.10)


# In[31]:


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


# In[32]:


train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'infiltration', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 8)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'infiltration', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'infiltration', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 400)) # one big batch
# used a fixed dataset for final evaluation
final_test_X, final_test_Y = next(flow_from_dataframe(core_idg, 
                               test_df, 
                             path_col = 'path',
                            y_col = 'infiltration', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 400)) # one big batch


# In[33]:


t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = 0, vmax = 255)
    c_ax.set_title('%s' % ('Inflitration' if c_y>0.5 else 'Healthy'))
    c_ax.axis('off')


# # Pretrained Features
# Here we generate the pretrained features for a large batch of images to accelerate the training process

# In[36]:


from keras.applications.vgg16 import VGG16
from tqdm import tqdm
base_pretrained_model = VGG16(input_shape =  t_x.shape[1:], 
                              include_top = False, weights = 'imagenet')
attn_train_X, attn_train_y = [], []

for _, (c_x, c_y) in zip(tqdm(range(100)), train_gen):
    attn_train_X += [base_pretrained_model.predict(c_x, verbose = False, batch_size = 4)]
    attn_train_y += [c_y]
    
attn_train_X = np.concatenate(attn_train_X, 0)
attn_train_y = np.concatenate(attn_train_y, 0)
print(attn_train_X.shape, attn_train_y.shape)


# In[37]:


attn_test_X = base_pretrained_model.predict(test_X, 
                                            verbose = True, 
                                            batch_size = 4)
attn_test_y = test_Y


# In[ ]:


get_ipython().system('rm -rf ~/.keras # clean up the model / make space for other things')


# # Attention Model
# The basic idea is that a Global Average Pooling is too simplistic since some of the regions are more relevant than others. So we build an attention mechanism to turn pixels in the GAP on an off before the pooling and then rescale (Lambda layer) the results based on the number of pixels. The model could be seen as a sort of 'global weighted average' pooling. There is probably something published about it and it is very similar to the kind of attention models used in NLP.
# It is largely based on the insight that the winning solution annotated and trained a UNET model to segmenting the hand and transforming it. This seems very tedious if we could just learn attention.

# In[39]:


from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, AvgPool2D
from keras.models import Model
pt_features = Input(base_pretrained_model.get_output_shape_at(0)[1:], name = 'feature_input')
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
from keras.layers import BatchNormalization
bn_features = BatchNormalization()(pt_features)
# here we do an attention mechanism to turn pixels in the GAP on an off
attn_layer = Conv2D(128, kernel_size = (1,1), padding = 'same', activation = 'elu')(bn_features)
attn_layer = Conv2D(32, kernel_size = (1,1), padding = 'same', activation = 'elu')(attn_layer)
attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'elu')(attn_layer)
attn_layer = AvgPool2D((2,2), strides = (1,1), padding = 'same')(attn_layer) # smooth results
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
gap_dr = Dropout(0.5)(gap)
dr_steps = Dropout(0.5)(Dense(128, activation = 'elu')(gap_dr))
out_layer = Dense(1, activation = 'sigmoid')(dr_steps)

attn_model = Model(inputs = [pt_features], outputs = [out_layer], name = 'attention_model')

attn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy'])

attn_model.summary()


# In[40]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('tb_attn')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[41]:


attn_model.fit(attn_train_X, attn_train_y, 
                batch_size = 32,
              validation_data = (attn_test_X, attn_test_y), 
               shuffle = True,
              epochs = 40, 
              callbacks = callbacks_list)


# In[42]:


# load the best version of the model
attn_model.load_weights(weight_path)


# # Build the whole model
# We build the whole model and fine-tune the results (much lower learning rate)

# In[43]:


from keras.models import Sequential
from keras.optimizers import Adam
tb_model = Sequential(name = 'combined_model')
base_pretrained_model.trainable = False
tb_model.add(base_pretrained_model)
tb_model.add(attn_model)
tb_model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy'])
tb_model.summary()


# In[ ]:


tb_model.fit_generator(train_gen, 
                      steps_per_epoch = 5,
                      validation_data = (test_X, test_Y), 
                      epochs = 1, 
                      callbacks = callbacks_list)


# # Show Attention
# Did our attention model learn anything useful?

# In[44]:


# get the attention layer since it is the only one with a single output dim
for attn_layer in attn_model.layers:
    c_shape = attn_layer.get_output_shape_at(0)
    if len(c_shape)==4:
        if c_shape[-1]==1:
            print(attn_layer)
            break


# In[45]:


import keras.backend as K
rand_idx = np.random.choice(range(len(test_X)), size = 6)
attn_func = K.function(inputs = [attn_model.get_input_at(0), K.learning_phase()],
           outputs = [attn_layer.get_output_at(0)]
          )
fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):
    cur_img = test_X[c_idx:(c_idx+1)]
    cur_features = base_pretrained_model.predict(cur_img)
    attn_img = attn_func([cur_features, 0])[0]
    img_ax.imshow(cur_img[0,:,:,0], cmap = 'bone')
    attn_ax.imshow(attn_img[0, :, :, 0], cmap = 'viridis', 
                   vmin = 0, vmax = 1, 
                   interpolation = 'lanczos')
    real_label = test_Y[c_idx]
    img_ax.set_title('TB\nClass:%s' % (real_label))
    pred_confidence = tb_model.predict(cur_img)[0]
    attn_ax.set_title('Attention Map\nPred:%2.1f%%' % (100*pred_confidence[0]))
fig.savefig('attention_map.png', dpi = 300)


# > # Evaluate the validation results
# Here we evaluate the results by loading the best version of the model and seeing how the predictions look on the results. We use the validation data which was not directly trained on but still *tainted* since it was used as a stopping criteria

# In[46]:


pred_Y = attn_model.predict(attn_test_X, 
                          batch_size = 16, 
                          verbose = True)


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix
plt.matshow(confusion_matrix(test_Y, pred_Y>0.5))
print(classification_report(test_Y, pred_Y>0.5, target_names = ['Healthy', 'Infiltration']))


# In[48]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(test_Y, pred_Y)
fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)
ax1.plot(fpr, tpr, 'b.-', label = 'VGG-Model (AUC:%2.2f)' % roc_auc_score(test_Y, pred_Y))
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend(loc = 4)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');
fig.savefig('roc.pdf')


# # Test Results
# The results on the hold-out data which hasn't been touched yet

# In[49]:


final_pred_Y = tb_model.predict(final_test_X, 
                                verbose = True, 
                                batch_size = 4)


# In[53]:


plt.matshow(confusion_matrix(final_test_Y, final_pred_Y>0.5))
print(classification_report(final_test_Y, final_pred_Y>0.5, target_names = ['Healthy', 'Infiltration']))


# In[54]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(final_test_Y, final_pred_Y)
fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)
ax1.plot(fpr, tpr, 'b.-', label = 'VGG-Model (AUC:%2.2f)' % roc_auc_score(test_Y, pred_Y))
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend(loc = 4)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');
fig.savefig('roc.pdf')


# In[51]:


tb_model.save('full_pred_model.h5')


# # Export the Attention Model
# We can run the attention model in addition to the classification to produce attention maps for the image. Here we just package the relevant inputs and outputs together

# In[52]:


img_in = Input(t_x.shape[1:])
feat_lay = base_pretrained_model(img_in)
just_attn = Model(inputs = attn_model.get_input_at(0), 
      outputs = [attn_layer.get_output_at(0)], name = 'pure_attention')
attn_img = just_attn(feat_lay)
pure_attn_model = Model(inputs = [img_in], outputs = [attn_img], name = 'just_attention_model')
pure_attn_model.save('pure_attn_model.h5')
pure_attn_model.summary()


# In[ ]:




