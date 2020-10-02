#!/usr/bin/env python
# coding: utf-8

# # Overview
# Since Silicon Valley and Medium managed to make a Hot Dog / Not Hot Dog classifier, we should be able to do something similar with the free GPU Kaggle gives us. We use a pretrained VGG16 as a starting point and then add a few layers on top

# ### Copy
# copy the weights and configurations for the pre-trained models

# In[1]:


get_ipython().system('mkdir ~/.keras')
get_ipython().system('mkdir ~/.keras/models')
get_ipython().system('cp ../input/keras-pretrained-models/*notop* ~/.keras/models/')
get_ipython().system('cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/')


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
base_dir = os.path.join('..', 'input', 'food41')
base_image_dir = os.path.join('..', 'input', 'food41')


# ## Preprocessing
# Find and read image files

# In[3]:


from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
image_paths = glob(os.path.join(base_image_dir, 'images', '*', '*'))
print('Total Images Files', len(image_paths))
all_paths_df = pd.DataFrame(dict(path = image_paths))
all_paths_df['food_name'] = all_paths_df['path'].map(lambda x: x.split('/')[-2].replace('_', ' ').strip())
all_paths_df['source'] = all_paths_df['food_name'].map(lambda x: 'Hot Dog' if x.lower() == 'hot dog' else 'Not Hot Dog')
source_enc = LabelEncoder()
all_paths_df['source_id'] = source_enc.fit_transform(all_paths_df['source'])
all_paths_df['source_vec'] = all_paths_df['source_id'].map(lambda x: to_categorical(x, len(source_enc.classes_)))
all_paths_df['file_id'] = all_paths_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
all_paths_df['file_ext'] = all_paths_df['path'].map(lambda x: os.path.splitext(x)[1][1:])
# balance a bit
all_paths_df = all_paths_df.groupby(['source_id']).apply(lambda x: x.sample(min(5000, x.shape[0]))
                                                      ).reset_index(drop = True)
all_paths_df.sample(5)


# # Examine the distributions
# Show how the data is distributed and why we need to balance it

# In[4]:


all_paths_df['source'].hist(figsize = (20, 7), xrot = 90)


# # Split Data into Training and Validation

# In[5]:


from sklearn.model_selection import train_test_split
raw_train_df, valid_df = train_test_split(all_paths_df, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = all_paths_df[['source_id']])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
raw_train_df.sample(1)


# # Balance the distribution in the training set

# In[46]:


train_df = raw_train_df.groupby(['source_id']).apply(lambda x: x.sample(3000, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
train_df['food_name'].hist(bins = 101, figsize = (20, 5), xrot = 90)


# In[7]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
from keras.applications.inception_v3 import InceptionV3 as PTModel, preprocess_input
#from keras.applications.xception import Xception as PTModel, preprocess_input
from PIL import Image
ppi = lambda x: Image.fromarray(preprocess_input(np.array(x).astype(np.float32)))
IMG_SIZE = (299, 299) # slightly smaller than vgg16 normally expects
core_idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.1, 
                              width_shift_range = 0.1, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'reflect',
                              zoom_range=0.15, 
                             preprocessing_function = preprocess_input)


# In[8]:


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


# In[47]:


train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'source_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 16)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'source_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 16) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'source_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 1024)) # one big batch


# In[10]:


t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (tc_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_x = np.clip((tc_x-tc_x.min())/(tc_x.max()-tc_x.min())*255, 0 , 255).astype(np.uint8)[:,:,::]
    c_ax.imshow(c_x[:,:])
    c_ax.set_title('%s' % source_enc.classes_[np.argmax(c_y)])
    c_ax.axis('off')


# # Pretrained Features
# Here we generate the pretrained features for a large batch of images to accelerate the training process

# In[11]:


# clean up resources
import gc
gc.enable()
print(gc.collect())
plt.close('all')


# # Attention Model
# The basic idea is that a Global Average Pooling is too simplistic since some of the regions are more relevant than others. So we build an attention mechanism to turn pixels in the GAP on an off before the pooling and then rescale (Lambda layer) the results based on the number of pixels. The model could be seen as a sort of 'global weighted average' pooling. There is probably something published about it and it is very similar to the kind of attention models used in NLP.
# It is largely based on the insight that the winning solution annotated and trained a UNET model to segmenting the hand and transforming it. This seems very tedious if we could just learn attention.

# In[12]:


from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, AvgPool2D, ActivityRegularization
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
base_pretrained_model = PTModel(input_shape =  t_x.shape[1:], 
                              include_top = False, weights = 'imagenet')
real_input = Input(shape=t_x.shape[1:])
pt_features = base_pretrained_model(real_input)
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
# we dont need the attention layer now
use_attention = False
if use_attention:
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
else:
    gap = GlobalAveragePooling2D()(bn_features)
gap_dr = Dropout(0.5)(gap)
dr_steps = Dropout(0.5)(Dense(128, activation = 'elu')(gap_dr))
out_layer = Dense(len(source_enc.classes_), 
                  activation = 'softmax')(dr_steps)

attn_model = Model(inputs = [real_input], 
                   outputs = [out_layer], name = 'attention_model')

attn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])

attn_model.summary()


# In[13]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('tb_attn')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', 
                                   factor=0.8, 
                                   patience=5,
                                   verbose=1, 
                                   mode='auto', 
                                   epsilon=0.0001, 
                                   cooldown=5, min_lr=1e-5)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


train_gen.batch_size = 32
attn_model.fit_generator(train_gen, 
              validation_data = (test_X, test_Y), 
               shuffle = True,
              epochs = 10, 
              callbacks = callbacks_list,
                workers = 2)


# In[ ]:


# load the best version of the model
attn_model.load_weights(weight_path)


# In[ ]:


get_ipython().system('rm -rf ~/.keras # clean up the model / make space for other things')


# # Evaluate the results
# Here we evaluate the results by loading the best version of the model and seeing how the predictions look on the results. We then visualize spec

# In[15]:


pred_Y = attn_model.predict(test_X, 
                          batch_size = 16, 
                          verbose = True)


# In[16]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize, dpi = 300)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

print_confusion_matrix(confusion_matrix(np.argmax(test_Y,-1), 
                             np.argmax(pred_Y,-1), labels = range(test_Y.shape[1])), 
                       class_names = source_enc.classes_, figsize = (10, 1)).savefig('confusion_matrix.png')

print(classification_report(np.argmax(test_Y,-1), 
                            np.argmax(pred_Y,-1), 
                            target_names = source_enc.classes_))


# In[35]:


from sklearn.metrics import roc_curve, roc_auc_score
tpr, fpr, _ = roc_curve(np.argmax(test_Y,-1), pred_Y[:, 1])
fig, ax1 = plt.subplots(1,1)
ax1.plot(tpr, fpr, 'r.', label = 'ROC (%2.2f)' % (roc_auc_score(np.argmax(test_Y,-1), pred_Y[:, 1])))
ax1.plot(tpr, tpr, 'b-')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')


# In[ ]:


attn_model.save('full_pred_model.h5')


# # Examine Misclassifications
# 

# In[39]:


class_df = pd.DataFrame(dict(label = np.argmax(test_Y,-1), 
                             prediction= pred_Y[:, 1]))
class_df['mismatch'] = np.abs(class_df['label']-class_df['prediction'])


# In[43]:


fig, m_axs = plt.subplots(2, 5, figsize = (20, 8))
for (c_grp, c_df), c_axs in zip(class_df.groupby('label'), m_axs):
    for (c_idx, c_row), c_ax in zip(c_df.sort_values('mismatch', ascending = False).iterrows(), c_axs):
        tc_x = test_X[c_idx]
        c_x = np.clip((tc_x-tc_x.min())/(tc_x.max()-tc_x.min())*255, 0 , 255).astype(np.uint8)[:,:,::]
        c_ax.imshow(c_x[:,:])
        c_ax.set_title('Actual: %s\n(HotDog Pred: %2.2f%%)' % (source_enc.classes_[int(c_row['label'])], 100*(1-c_row['prediction'])))
        c_ax.axis('off')
fig.savefig('mismatch.png')


# In[ ]:




