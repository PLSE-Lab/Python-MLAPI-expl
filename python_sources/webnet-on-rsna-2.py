#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# Welcome to the 2018 RSNA Challenge co-hosted by Kaggle. In this competition, the primary endpoint will be the detection of bounding boxes corresponding to the diagnosis of pneumonia (e.g. lung infection) on chest radiographs, a special 2D high resolution grayscale medical image. Note that pnuemonia is just one of many possible disease processes that can occur on a chest radiograph, and that any given single image may contain 0, 1 or many boxes corresponding to possible pneumonia locations.
# 
# My name is Peter Chang, MD. I am both a radiologist physician and a data scientist / software engineer with machine learning experience. Today, in this Jupyter notebook, we will explore the 2018 RSNA Challenge dataset including underlying data structures, imaging file formats and label types.

# In[ ]:


import glob, pylab, pandas as pd
import pydicom, numpy as np


# # Challenge Data
# 
# The challenge data is organized in several files and folders. If you are following along in the Kaggle kernel, this data will be preloaded in the `../input` directory:

# In[ ]:


get_ipython().system('ls ../input')


# The several key items in this folder:
# * `stage_1_train_labels.csv`: CSV file containing training set patientIds and  labels (including bounding boxes)
# * `stage_1_detailed_class_info.csv`: CSV file containing detailed labels (explored further below)
# * `stage_1_train_images/`:  directory containing training set raw image (DICOM) files
# 
# Let's go ahead and take a look at the first labels CSV file first:

# In[ ]:


'''
with open('../input/GCP Credits Request Link - RSNA.txt')as f:
    content=f.readlines()
    print(content)
'''


# In[ ]:


df = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')
print(df.iloc[0])


# As you can see, each row in the CSV file contains a `patientId` (one unique value per patient), a target (either 0 or 1 for absence or presence of pneumonia, respectively) and the corresponding abnormality bounding box defined by the upper-left hand corner (x, y) coordinate and its corresponding width and height. In this particular case, the patient does *not* have pneumonia and so the corresponding bounding box information is set to `NaN`. See an example case with pnuemonia here:

# In[ ]:


print(df.info())
print(df.iloc[4])


# One important thing to keep in mind is that a given `patientId` may have **multiple** boxes if more than one area of pneumonia is detected (see below for example images).

# # Overview of DICOM files and medical images
# 
# Medical images are stored in a special format known as DICOM files (`*.dcm`). They contain a combination of header metadata as well as underlying raw image arrays for pixel data. In Python, one popular library to access and manipulate DICOM files is the `pydicom` module. To use the `pydicom` library, first find the DICOM file for a given `patientId` by simply looking for the matching file in the `stage_1_train_images/` folder, and the use the `pydicom.read_file()` method to load the data:

# In[ ]:


'''
patientId = df['patientId'][0]
dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
dcm_data = pydicom.read_file(dcm_file)

print(dcm_data)
'''


# Most of the standard headers containing patient identifable information have been anonymized (removed) so we are left with a relatively sparse set of metadata. The primary field we will be accessing is the underlying pixel data as follows:

# In[ ]:


'''
im = dcm_data.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)
'''


# ## Considerations
# 
# As we can see here, the pixel array data is stored as a Numpy array, a powerful numeric Python library for handling and manipulating matrix data (among other things). In addition, it is apparent here that the original radiographs have been preprocessed for us as follows:
# 
# * The relatively high dynamic range, high bit-depth original images have been rescaled to 8-bit encoding (256 grayscales). For the radiologists out there, this means that the images have been windowed and leveled already. In clinical practice, manipulating the image bit-depth is typically done manually by a radiologist to highlight certain disease processes. To visually assess the quality of the automated bit-depth downscaling and for considerations on potentially improving this baseline, consider consultation with a radiologist physician.
# 
# * The relativley large original image matrices (typically acquired at >2000 x 2000) have been resized to the data-science friendly shape of 1024 x 1024. For the purposes of this challenge, the diagnosis of most pneumonia cases can typically be made at this resolution. To visually assess the feasibility of diagnosis at this resolution, and to determine the optimal resolution for pneumonia detection (oftentimes can be done at a resolution *even smaller* than 1024 x 1024), consider consultation with a radiogist physician.
# 
# ## Visualizing An Example
# 
# To take a look at this first DICOM image, let's use the `pylab.imshow()` method:

# In[ ]:


#pylab.imshow(im, cmap=pylab.cm.gist_gray)
#pylab.axis('off')


# # Exploring the Data and Labels
# 
# As alluded to above, any given patient may potentially have many boxes if there are several different suspicious areas of pneumonia. To collapse the current CSV file dataframe into a dictionary with unique entries, consider the following method:

# In[ ]:


def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


# Let's use the method here:

# In[ ]:


#parsed = parse_data(df)


# As we saw above, patient `00436515-870c-4b36-a041-de91049b9ab4` has pnuemonia so lets check our new `parsed` dict here to see the patients corresponding bounding boxes:

# In[ ]:


#print(parsed[ 'c542f0f4-1903-4fee-ba0f-186203d35226'])


# # Visualizing Boxes
# 
# In order to overlay color boxes on the original grayscale DICOM files, consider using the following  methods (below, the main method `draw()` requires the method `overlay_box()`):

# In[ ]:


'''
def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
    '''


# As we saw above, patient `00436515-870c-4b36-a041-de91049b9ab4` has pnuemonia so let's take a look at the overlaid bounding boxes:

# In[ ]:


#draw(parsed['00436515-870c-4b36-a041-de91049b9ab4'])


# ## Exploring Detailed Labels
# 
# In this challenge, the primary endpoint will be the detection of bounding boxes consisting of a binary classification---e.g. the presence or absence of pneumonia. However, in addition to the binary classification, each bounding box *without* pneumonia is further categorized into *normal* or *no lung opacity / not normal*. This extra third class indicates that while pneumonia was determined not to be present, there was nonetheless some type of abnormality on the image---and oftentimes this finding may mimic the appearance of true pneumonia. Keep in mind that this extra class is provided as supplemental information to help improve algorithm accuracy if needed; generation of this separate class **will not** be a formal metric used to evaluate performance in this competition.
# 
# As above, we saw that the first patient in the CSV file did not have pneumonia. Let's look at the detailed label information for this patient:

# In[ ]:


#df_detailed = pd.read_csv('../input/stage_1_detailed_class_info.csv')
#print(df_detailed.iloc[0])


# As we see here, the patient does not have pneumonia however *does* have another imaging abnormality present. Let's take a closer look:

# In[ ]:


#patientId = df_detailed['patientId'][0]
#draw(parsed[patientId])


# While the image displayed inline within the notebook is small, as a radiologist it is evident that the patient has several well circumscribed nodular densities in the left lung (right side of image). In addition there is a large chest tube in the right lung (left side of the image) which has been placed to drain fluid accumulation (e.g. pleural effusion) at the right lung base that also demonstrates overlying patchy densities (e.g. possibly atelectasis or partial lung collapse).
# 
# As you can see, there are a number of abnormalities on the image, and the determination that none of these findings correlate to pneumonia is somewhat subjective even among expert physicians. Therefore, as is almost always the case in medical imaging datasets, the provided ground-truth labels are far from 100% objective. Keep this in mind as you develop your algorithm, and consider consultation with a radiologist physician to help determine an optimal strategy for mitigating these discrepencies.

# ## Label Summary
# 
# Finally, let us take a closer look at the distribution of labels in the dataset. To do so we will first parse the detailed label information:

# In[ ]:


'''
summary = {}
for n, row in df_detailed.iterrows():
    if row['class'] not in summary:
        summary[row['class']] = 0
    summary[row['class']] += 1
    
print(summary)
'''


# As we can see, there is a relatively even split between the three classes, with nearly 2/3rd of the data comprising of no pneumonia (either completely *normal* or *no lung opacity / not normal*). Compared to most medical imaging datasets, where the prevalence of disease is quite low, this dataset has been significantly enriched with pathology.

# # Next Steps
# 
# Now that you understand the data structures, imaging file formats and label types, it's time to make an algorithm! Keep in mind that the primary endpoint is the detection of bounding boxes, thus you will likely be considering various **object localization** algorithms. An alternative strategy is to consider the related family of **segmentation** algorithms with the acknowledgement that bounding boxes will only be a coarse approximation to true pixel-by-pixel image segmentation masks.
# 
# Finally, as alluded to several times in this notebook, a radiologist physican may often times provide useful ancillary information, strategy for algorithm development and/or additional label reconciliation. In addition to physicians you may have access to locally, the RSNA will reach out to radiologists and facilitate engagement remotely through the Kaggle online forums. As a medical professional, I know that many of my colleagues are very interested in getting started so please feel free to reach out and start a conversation! 
# 
# Good luck!

# In[ ]:


import glob, pylab, pandas as pd
import pydicom, numpy as np

import os
import csv
import random
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
import skimage.exposure

from matplotlib import pyplot as plt






# In[ ]:


df_detailed = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_detailed_class_info.csv')
print(df_detailed.iloc[6])
print(df_detailed.iloc[80])


# In[ ]:


# empty dictionary
nodule_locations = {}
# load table
with open(os.path.join('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv'), mode='r') as infile:
    reader = csv.reader(infile)
    # skip header
    next(reader, None)

    for rows in reader:
        filename = rows[0]
        location = rows[1:5]
        nodule = rows[5]
        # if row contains a nodule add label to dictionary
        # which contains a list of nodule locations per filename
        if nodule == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            # save nodule location in dictionary
            if filename in nodule_locations:
                nodule_locations[filename].append(location)
            else:
                nodule_locations[filename] = [location]


# In[ ]:


folder = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images'
filenames = os.listdir(folder)
random.shuffle(filenames)
# split into train and validation filenames
n_valid_samples = 2000
train_filenames = filenames[n_valid_samples:]
valid_filenames = filenames[:n_valid_samples]
print('n train samples', len(train_filenames))
print('n valid samples', len(valid_filenames))
n_train_samples = len(filenames) - n_valid_samples


# In[ ]:


class generator(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, nodule_locations=None, batch_size=32, image_size=128, shuffle=True, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.nodule_locations = nodule_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        #msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains nodules
        if filename in nodule_locations:
            # loop through nodules
            pneumonia=1
        else:
            pneumonia=0
                
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        #msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        #msk = np.expand_dims(msk, -1)
        return img,pneumonia # ,msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, pneumonia = zip(*items)
            
            # create numpy batch
            imgs = np.array(imgs)
            #imgs= [skimage.transform.resize(imgs, (128,128,1))]   
            pneumonia = np.array(pneumonia)
            #pneumonia=pneumonia.reshape(16,1,1,1)
            return imgs,pneumonia #, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)


# In[ ]:


# create train and validation generators
folder = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images'
train_gen = generator(folder, train_filenames, nodule_locations, batch_size=16, image_size=128, shuffle=True, predict=False)
valid_gen = generator(folder, valid_filenames, nodule_locations, batch_size=16, image_size=128, shuffle=False, predict=False)
#x_val_new, y_val=valid_gen
#validation_generator = test_datagen.flow(valid_gen,batch_size=16)


# Define Model

# In[ ]:





# In[ ]:


def identity_block(inputs,kernel_size,filters):
    filters1, filters2, filters3 = filters
    
    x = keras.layers.Conv2D(filters1, (1, 1)) (inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters2, kernel_size,
               padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x =keras.layers.Conv2D(filters3, (1, 1))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.add([x, inputs])
    x = keras.layers.ReLU()(x)
    return x


# In[ ]:


def webnet(input_size):
    inputs= keras.Input(shape=(input_size, input_size, 1))
    conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(inputs)
    #conv1 = Dropout(0.5)(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    #conv1 = Dropout(0.5)(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    
    conv2=identity_block(pool1,3,[64,64,64])
    #conv2 = Dropout(0.5)(conv2)
    conv2 = identity_block(conv2,3,[64,64,64])
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = identity_block(pool2,3,[32,32,64])
    #conv3 = Dropout(0.5)(conv3)
    conv3 = identity_block(conv3,3,[32,32,64])
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = identity_block(pool3,3,[32,32,64])
    #conv4 = Dropout(0.5)(conv4)
    conv4 = identity_block(conv4,3,[32,32,64])
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = identity_block(pool4,3,[32,32,64])
    #conv5 = Dropout(0.5)(conv5)
    conv5 = identity_block(conv5,3,[32,32,64])
    conv5 = identity_block(conv5,3,[32,32,64])

    up6 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up6)
    conv6 = identity_block(conv6,3,[32,32,64])
    #conv6 = Dropout(0.5)(conv6)
    conv6 = identity_block(conv6,3,[32,32,64])

    up7 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up7)
    conv7 = identity_block(conv7,3,[32,32,64])
    #conv7 = Dropout(0.5)(conv7)
    conv7 = identity_block(conv7,3,[32,32,64])
    pool7 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv7)

    concat8 = keras.layers.Concatenate(axis=-1)([pool7, conv6])
    conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(concat8)
    conv8 = identity_block(conv8,3,[32,32,64])
    #conv8 = Dropout(0.5)(conv8)
    conv8 = identity_block(conv8,3,[32,32,64])
    pool8 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv8)

    concat9 =keras.layers.Concatenate()([pool8, conv5])
    conv9 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(concat9)
    conv9 = identity_block(conv9,3,[32,32,64])
    #conv9 = Dropout(0.5)(conv9)
    conv9 = identity_block(conv9,3,[32,32,64])
    #conv9 = Dropout(0.5)(conv9)
       
    up10 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv9), conv8])
    conv10 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up10)
    conv10 = identity_block(conv10,3,[32,32,64])
    #conv10 = Dropout(0.5)(conv6)
    conv10 = identity_block(conv10,3,[32,32,64])
    
    up11 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv10), conv7])
    conv11 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up11)
    conv11 =identity_block(conv11,3,[32,32,64])
    #conv11 = Dropout(0.5)(conv11)
    conv11 = identity_block(conv11,3,[32,32,64])
    pool11 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv11)
    
    concat12 = keras.layers.Concatenate()([pool11, conv10])
    conv12 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(concat12)
    conv12 = identity_block(conv12,3,[32,32,64])
    #conv12 = Dropout(0.5)(conv12)
    conv12 = identity_block(conv12,3,[32,32,64])
    pool12 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv12)
    
    concat13 = keras.layers.Concatenate(axis=-1)([pool12, conv9])
    conv13 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(concat13)
    conv13 = identity_block(conv13,3,[32,32,64])
    #conv13 = Dropout(0.5)(conv13)
    conv13 = identity_block(conv13,3,[32,32,64])
     
    up14 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv13), conv12])
    conv14 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up14)
    conv14 = identity_block(conv14,3,[32,32,64])
    conv14 = identity_block(conv14,3,[32,32,64])
    
    up15 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv14), conv11])
    conv15 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up15)
    conv15 = identity_block(conv15,3,[32,32,64])
    #conv15 = Dropout(0.5)(conv15)
    conv15 = identity_block(conv15,3,[32,32,64])
    
    up16 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv15), conv2])
    conv16 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up16)
    conv16 = identity_block(conv16,3,[32,32,64])
    #conv16 = Dropout(0.5)(conv16)
    conv16 = identity_block(conv16,3,[32,32,64])
    
    up17 = keras.layers.Concatenate(axis=-1)([keras.layers.UpSampling2D(size=(2, 2))(conv16), conv1])
    conv17 = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(up17)
    conv17 = identity_block(conv17,3,[32,32,64])
    #conv17 = Dropout(0.5)(conv17)
    #conv17 = AveragePooling2D((7, 7))(conv17)
    conv17 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv17)
    flat1=keras.layers.Flatten()(conv17)
    dense1= keras.layers.Dense(1, activation='sigmoid')(flat1)
    model = keras.Model(inputs=inputs, outputs=dense1)

    return model


# In[ ]:


model = webnet(input_size=128)
model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


def cosine_annealing(x):
    lr = 0.001
    epochs = 25
    return lr*(np.cos(np.pi*x/epochs)+1.)/2
learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)


# In[ ]:





# In[ ]:


model_info=model.fit_generator(
 train_gen,
    steps_per_epoch=200,
    epochs=100,
    validation_data=valid_gen,
    validation_steps=50,
     callbacks=[tf.keras.callbacks.CSVLogger(os.path.join('training_log.csv'), append=True),
                                         #ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, verbose=1, patience=50),
                                         tf.keras.callbacks.ModelCheckpoint(os.path.join(
                                               #'weights.ep-{epoch:02d}-val_mean_IOU-{val_mean_IOU_gpu:.2f}_val_loss_{val_loss:.2f}.hdf5',
                                               'last_checkpoint.hdf5'),
                                               monitor='val_loss', mode='min', save_best_only=True, 
                                               save_weights_only=False, verbose=0)])


# In[ ]:


def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# In[ ]:


#plot_model_history(model_info)


# In[ ]:


model_path = os.path.join('../input/webnet1-rsna/last_checkpoint.hdf5')


# In[ ]:


model.load_weights(model_path)


# In[ ]:



folder = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images'
filenames = os.listdir(folder)

# split into train and validation filenames
n_valid_samples = 2000

valid_filenames2 = filenames[:n_valid_samples]


# In[ ]:


valid_gen_ = generator(folder, valid_filenames2, nodule_locations, batch_size=16, image_size=128, shuffle=False, predict=False)


# In[ ]:


pred = model.predict_generator(valid_gen_)


# In[ ]:


print (pred.shape)


# In[ ]:


print ((pred[2]))


# In[ ]:


#import math


correct=0
for i in range(2000):
  
    if (pred[i]   >= 0.1 and df['Target'][i] == 1):
        correct +=1
    if (pred[i]  < 0.1 and df['Target'][i] == 0):
        correct +=1
        
    
        

accuracy= correct/2000

print(accuracy)   


# In[ ]:


#print(accuracy)


# In[ ]:


#print (df['Target'][:2000])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




        


# In[ ]:




