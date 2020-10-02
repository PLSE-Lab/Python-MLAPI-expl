#!/usr/bin/env python
# coding: utf-8

# # Overview
# In this notebook we try and automatically detect bleeds in Head CT scans. We have readings from 3 independent physicians for the entire scan and we associate the readings with each slice

# In[ ]:


# params
NR_EPOCHS = 30
MODEL_ARCH = 'MOBILE'
BATCH_SIZE = 128
DENSE_SIZE = 64
IMG_X = 96
IMG_Y = 96


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from pydicom import read_file as read_dicom
import SimpleITK as sitk
base_dir = os.path.join('..', 'input', 'qureai-headct')
reads_dir = os.path.join('..', 'input', 'headctreads')


# In[ ]:


all_dicom_paths = glob(os.path.join(base_dir, '*', '*', '*', '*', '*'))
print(len(all_dicom_paths), 'dicom files')
dicom_df = pd.DataFrame(dict(path = all_dicom_paths))
dicom_df['SliceNumber'] = dicom_df['path'].map(lambda x: int(os.path.splitext(x.split('/')[-1])[0][2:]))
dicom_df['SeriesName'] = dicom_df['path'].map(lambda x: x.split('/')[-2])
dicom_df['StudyID'] = dicom_df['path'].map(lambda x: x.split('/')[-3])
dicom_df['PatientID'] = dicom_df['path'].map(lambda x: x.split('/')[-4].split(' ')[0])
dicom_df['PatSeries'] = dicom_df.apply(lambda x: '{PatientID}-{SeriesName}'.format(**x), 1)
dicom_df.sample(3)


# In[ ]:


small_scans = dicom_df.groupby('PatSeries').count().reset_index().query('SliceNumber<240')
dicom_df = dicom_df[dicom_df['PatSeries'].isin(small_scans['PatSeries'])]
print('Removed big scans', dicom_df.shape[0], 'remaining images')


# In[ ]:


dicom_df.describe(include = 'all')


# # Read Physician Reads
# Here we load the physician reads and preprocess them so we can associate them with each scan. We average them here to make it easier.

# In[ ]:


read_overview_df = pd.read_csv(os.path.join(reads_dir, 'reads.csv'))
read_overview_df['PatientID'] = read_overview_df['name'].map(lambda x: x.replace('-', '')) 
read_overview_df.sample(2).T


# In[ ]:


from collections import OrderedDict
new_reads = []
for _, c_row in read_overview_df.iterrows():
    base_dict = OrderedDict(PatientID = c_row['PatientID'], Category = c_row['Category'])
    for reader in ['R1', 'R2', 'R3']:
        c_dict = base_dict.copy()
        c_dict['Reader'] = reader
        for k,v in c_row.items():
            if (reader+':') in k:
                c_dict[k.split(':')[-1]] = v
        new_reads += [c_dict]
new_reads_df = pd.DataFrame(new_reads)
new_reads_df.to_csv('formatted_reads.csv')
new_reads_df.sample(5)


# In[ ]:


avg_reads_df = new_reads_df.groupby(['PatientID', 'Category']).agg('mean').reset_index()
read_dicom_df = pd.merge(avg_reads_df, dicom_df, on = 'PatientID')
read_dicom_df['Bleed'] = read_dicom_df.apply(lambda x: np.clip(x['BleedLocation-Left']+x['BleedLocation-Right']+x['ChronicBleed'], 0, 1), 1)
print(read_dicom_df.shape[0], 'total weakly-labeled slices')
read_dicom_df.sample(3)


# # Using SimpleITK
# Using SimpleITK instead of pydicom lets us load the images correctly now

# In[ ]:


fig, m_axs = plt.subplots(3, 3, figsize = (20, 20))
for c_ax, (_, c_row) in zip(m_axs.flatten(), read_dicom_df.sample(9).iterrows()):
    try:
        c_img = sitk.ReadImage(c_row['path'])
        c_slice = sitk.GetArrayFromImage(c_img)[0]
        c_ax.imshow(c_slice, cmap = 'bone')
        c_ax.set_title('Bleed: {Bleed:2.2f}, Fracture: {Fracture:2.2f}\n{SeriesName}'.format(**c_row))
    except Exception as e:
        c_ax.set_title('{}'.format(str(e)[:40]))
        print(e)
    #c_ax.axis('off')


# # Classify series name from image
# We can make a simple model here to identify which series type an image came from

# In[ ]:


from sklearn.model_selection import train_test_split
valid_df = read_dicom_df[['PatientID', 'Bleed']].drop_duplicates()
print('Patients', valid_df.shape[0])
train_ids, test_ids = train_test_split(valid_df[['PatientID']], 
                                       test_size = 0.25, 
                                       stratify = valid_df['Bleed'].map(lambda x: x>0))

train_unbalanced_df = read_dicom_df[read_dicom_df['PatientID'].isin(train_ids['PatientID'])]
test_df = read_dicom_df[read_dicom_df['PatientID'].isin(test_ids['PatientID'])]
print(train_unbalanced_df.shape[0], 'training images', test_df.shape[0], 'testing images')
train_unbalanced_df['Bleed'].hist(figsize = (10, 5))


# In[ ]:


train_df = train_unbalanced_df.groupby(['Bleed', 'SeriesName']).apply(lambda x: x.sample(200, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', train_unbalanced_df.shape[0])
train_df['Bleed'].hist(figsize = (20, 5))


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (IMG_X, IMG_Y) # many of the ojbects are small so 512x512 lets us see them
img_gen_args = dict(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.05, 
                              width_shift_range = 0.02, 
                              rotation_range = 3, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range = 0.05)
img_gen = ImageDataGenerator(**img_gen_args)


# In[ ]:


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, seed = None, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways: seed: {}'.format(seed))
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                              seed = seed,
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values,0)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# # Replace PIL with SimpleITK
# Since we want to open images that are DICOMs we use SimpleITK to open them

# In[ ]:


import keras.preprocessing.image as KPImage
from PIL import Image
def apply_window(data, center, width):
    low = center - width/2.
    high = center + width/2
    data = np.clip(data, low, high)
    data += -1 * low
    data /= width
    return data
def read_dicom_image(in_path):
    c_img = sitk.ReadImage(in_path)
    c_slice = sitk.GetArrayFromImage(c_img)[0]
    return c_slice
    
class medical_pil():
    @staticmethod
    def open(in_path):
        if '.dcm' in in_path:
            # we only want to keep the positive labels not the background
            c_slice = read_dicom_image(in_path)
            wind_slice = apply_window(c_slice, 40, 80)
            int_slice =  (255*wind_slice).clip(0, 255).astype(np.uint8) # 8bit images are more friendly
            return Image.fromarray(int_slice)
        else:
            return Image.open(in_path)
    fromarray = Image.fromarray
KPImage.pil_image = medical_pil


# In[ ]:


batch_size = BATCH_SIZE
train_gen = lambda: flow_from_dataframe(img_gen, train_df, 
                             path_col = 'path',
                            y_col = 'Bleed', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = batch_size)
test_gen = lambda: flow_from_dataframe(img_gen, test_df, 
                             path_col = 'path',
                            y_col = 'Bleed', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = batch_size)


# In[ ]:


t_x, t_y = next(train_gen())
print(t_x.shape, '->', t_y.shape)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    c_ax.set_title('Bleed: {:2.2f}'.format(c_y))
    c_ax.axis('off')


# In[ ]:


if MODEL_ARCH=='NAS':
    from keras.applications.nasnet import NASNetMobile as BaseModel
elif MODEL_ARCH=='MOBILE':
    from keras.applications.mobilenet import MobileNet as BaseModel
else:
    raise ValueError('Model {} not supported'.format(MODEL_ARCH))
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
ct_model = Sequential()
ct_model.add(BatchNormalization(input_shape = t_x.shape[1:]))
ct_model.add(BaseModel(input_shape = t_x.shape[1:], include_top = False, weights = None))
ct_model.add(GlobalAveragePooling2D())
ct_model.add(Dropout(0.5))
ct_model.add(Dense(DENSE_SIZE))
ct_model.add(Dropout(0.25))
ct_model.add(Dense(1, activation = 'sigmoid'))

ct_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['mae', 'binary_accuracy'])
ct_model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('cthead')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=6) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


ct_model.fit_generator(train_gen(), 
                       steps_per_epoch = 8000//BATCH_SIZE,
                        validation_data = test_gen(), 
                       validation_steps = 4000//BATCH_SIZE,
                              epochs = NR_EPOCHS, 
                              callbacks = callbacks_list,
                             workers = 4,
                             use_multiprocessing=False, 
                             max_queue_size = 10
                            )


# In[ ]:


ct_model.load_weights(weight_path)
ct_model.save('full_bleed_model.h5')


# In[ ]:


get_ipython().run_cell_magic('time', '', "out_vals = ct_model.evaluate_generator(test_gen(), steps = 4000//BATCH_SIZE, workers=4)\nprint('Mean Absolute Error: %2.1f%%\\nAccuracy %2.1f%%' % (out_vals[1]*100, out_vals[2]*100))")


# In[ ]:


eval_df = pd.DataFrame([dict(zip(ct_model.metrics_names, out_vals))])
eval_df.to_csv('test_score.csv', index = False)
eval_df


# In[ ]:


test_gen.batch_size = 128
t_x, t_y = next(test_gen())
pred_y = ct_model.predict(t_x)
print(t_x.shape, '->', t_y.shape)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, p_y, c_ax) in zip(t_x, t_y, pred_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    c_ax.set_title('Bleed: {:2.2f}%\nPrediction: {:2.2f}%'.format(100*c_y, 100*p_y[0]))
    c_ax.axis('off')


# # Regression Prediction
# In a perfect world this would be a line, but as we see it is quite far from that

# In[ ]:


import seaborn as sns
fig,ax1 = plt.subplots(1,1)
sns.swarmplot(x = (100*t_y).astype(int), y = pred_y[:,0], ax = ax1)
ax1.set_xlabel('Bleed')
ax1.set_ylabel('Prediction')


# In[ ]:


sns.lmplot(x = 'x', y = 'y', data = pd.DataFrame(dict(x = (100*t_y).astype(int), y = pred_y[:,0])))


# # Run a whole scan
# Here we take a random scan and run every slice

# In[ ]:


bleed_patient_id = test_df.query('Bleed==1.0').query('SliceNumber>100 and SliceNumber<300').sample(1, random_state = 2018)[['PatientID', 'SeriesName']]
scan_df = pd.merge(test_df,bleed_patient_id).sort_values('SliceNumber')
print('Slices', scan_df.shape[0])
scan_df.head(5)


# #### fix slice order (names arent always right)

# In[ ]:


series_reader = sitk.ImageSeriesReader()
scan_df['path'] = series_reader.GetGDCMSeriesFileNames(os.path.dirname(scan_df['path'].values[0]))


# In[ ]:


scan_gen = flow_from_dataframe(img_gen, scan_df, 
                             path_col = 'path',
                            y_col = 'Bleed', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = scan_df.shape[0], 
                              shuffle = False)


# In[ ]:


t_x, t_y = next(scan_gen)
pred_y = ct_model.predict(t_x, batch_size = batch_size)
print(t_x.shape, '->', t_y.shape)


# In[ ]:


fig, m_axs = plt.subplots(8, 8, figsize = (16, 16))
for (c_x, c_y, p_y, c_ax) in zip(t_x, t_y, pred_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    c_ax.set_title('{:2.1f}%'.format(100*p_y[0]))
    c_ax.axis('off')


# In[ ]:


fig, ax1 = plt.subplots(1,1, figsize = (10, 10))
ax1.plot(scan_df['SliceNumber'], pred_y[:,0], 'r.-')
ax1.set_xlabel('Slice Number')
ax1.set_ylabel('Bleed Prediction')


# # Show the Most Suspicious Slices

# In[ ]:


new_idx = np.argsort(-pred_y[:,0])
fig, m_axs = plt.subplots(5, 5, figsize = (16, 16))
for (c_x, c_y, p_y, c_ax) in zip(t_x[new_idx], t_y[new_idx], pred_y[new_idx], m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    c_ax.set_title('{:2.1f}%'.format(100*p_y[0]))
    c_ax.axis('off')
fig.savefig('suspicious_slices.png')


# In[ ]:




