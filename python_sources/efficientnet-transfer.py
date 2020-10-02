#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ulrich - B0, B2, B4
# Piyush - B1, B3, B5
# Kai - B6, B7
# Ming - metadata


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tensorflow as tf
from tqdm import tqdm
import csv
import PIL

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError as e:
    tpu = None
    print(e)

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

REPLICAS = strategy.num_replicas_in_sync
print("REPLICAS: ", REPLICAS)


# In[ ]:


pip install -U efficientnet


# In[ ]:


train_folder = '../input/melanoma-foldered-collection/kaggle/working/224x224-dataset-melanoma'
test_folder = '../input/melanoma-test/kaggle/working/224x224-test'


# In[ ]:


nb_mel = len(os.listdir(os.path.join(train_folder,'melanoma')))
nb_other = len(os.listdir(os.path.join(train_folder,'other')))
print(f'{nb_mel} melanoma training samples')
print(f'{nb_other} other training samples')


# In[ ]:


import random

seed_nr = 10
random.seed(seed_nr)

def add_unique_idx(seq,i):
    idx = random.randint(0,nb_other-1)
    if not idx in seq:
        seq[i] = idx
    else:
        add_unique_idx(seq,i)

random_sequence = np.zeros(nb_mel,dtype='int32')
for i in range(nb_mel):
    add_unique_idx(random_sequence,i)
values, counts = np.unique(random_sequence,return_counts=True)
assert(len(values)==nb_mel) # nb_mel unique indices


# In[ ]:


custom_train_folder = '/kaggle/working/training_subset'

melanoma_folder = '/kaggle/working/training_subset/melanoma'
if not os.path.exists(melanoma_folder):
    os.makedirs(melanoma_folder)

other_subset_folder = '/kaggle/working/training_subset/other'
if not os.path.exists(other_subset_folder):
    os.makedirs(other_subset_folder)


# In[ ]:


from shutil import copyfile

# Transfer files
# Melanoma
melanoma_src = os.path.join(train_folder,'melanoma')
for file in tqdm(os.listdir(melanoma_src)):
    dest = os.path.join(melanoma_folder,file)
    src = os.path.join(melanoma_src,file)
    copyfile(src,dest)
    
# Other
other_src = os.path.join(train_folder,'other')
for idx, file in tqdm(enumerate(os.listdir(other_src))):
    if idx in random_sequence:
        dest = os.path.join(other_subset_folder,file)
        src = os.path.join(other_src,file)
        copyfile(src,dest)


# In[ ]:


assert(len(os.listdir(os.path.join(melanoma_folder)))==nb_mel)
assert(len(os.listdir(os.path.join(other_subset_folder)))==nb_mel)


# In[ ]:


import efficientnet.tfkeras as efn 

WIDTH = 224
HEIGHT = 224
CHANNELS = 3
input_shape = (WIDTH,HEIGHT,CHANNELS)
# TODO
# Try EfficientNetB0 up until EfficientNetB7 with varying pooling parameter as well.
# Running all combinations is not necessary, we just want to explore some configurations.
with strategy.scope():
    eff_model = efn.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None) # TODO: try pooling = 'avg' and pooling = 'max'


# In[ ]:


print(eff_model.summary())
# There is no average or max global pooling at the end, probably not ideal but this is only a demo.


# In[ ]:


# TODO: find first layer from last block
# in case of EfficientNetB0, it is block7a_expand_conv (first layer from last block 7a)
# All layers starting from this layer are finetuned and are thus trainable, all previous layers' knowledge will be transferred and are thus untrainable
with strategy.scope():
    first_trainable_layer = 'block7a_expand_conv'
    reached = False
    for idx, layer in enumerate(eff_model.layers):
        if layer.name == first_trainable_layer:
            reached = True
        if not reached:
            layer.trainable = False
        #print(layer.name)
        #print(layer.trainable)


# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam, Adamax, RMSprop, SGD

with strategy.scope():
    # 1 output melanoma probability, 1 output other probability (sum = 1)
    final_layer = Dense(2,activation='softmax',name='final_layer')(Flatten()(eff_model.output))
    eff_finetuning = Model(eff_model.input,final_layer)
    print(eff_finetuning.summary())
    acc = Accuracy()
    pre = Precision()
    rec = Recall()
    auc = AUC()

    lr = 0.0001
    opt = Adam(learning_rate=lr)

    eff_finetuning.compile(optimizer=opt,loss='categorical_crossentropy',metrics=[acc,pre,rec,auc])


# In[ ]:


# TODO: inspect amount of trainable parameters in your model
# 1.254.834 trainable parameters in EfficientNetB0


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# Augmentation arguments will be added in the future. We first start exploring non-augmented models (runs faster).
train_generator = ImageDataGenerator(rescale=1./255,validation_split=0.0)
#test_generator = ImageDataGenerator(rescale=1./255)


# In[ ]:


batch_size = 128*REPLICAS
train_flow = train_generator.flow_from_directory(custom_train_folder,target_size=(WIDTH,HEIGHT),batch_size=batch_size,class_mode='categorical')
#test_flow = test_generator.flow_from_directory(test_folder,target_size=(WIDTH,HEIGHT),batch_size=batch_size,class_mode='categorical')


# In[ ]:


# TODO: change epochs
# 50 epochs should be enough for first impression
epochs = 50
# TODO
# Check final loss and metrics after training
eff_finetuning.fit(train_flow,epochs=epochs,verbose=1)


# In[ ]:


# TODO
# Save model (change name)
model_name = 'effnet_b0_nopool.hdf5'
eff_finetuning.save_weights(model_name)


# In[ ]:


test_csv = '../input/siim-isic-melanoma-classification/test.csv'
with open(test_csv,'r') as test_file:
    reader = csv.reader(test_file)
    header = next(reader)
    with open('effnet_b0_nopool_subm.csv', 'w') as subm_file:
        writer = csv.writer(subm_file,delimiter=',')
        writer.writerow(['image_name','target'])
        for row in tqdm(reader):
            img_name = os.path.join(test_folder,row[0]+'.jpg')
            img_input = np.array([np.asarray(PIL.Image.open(img_name))])
            prob = eff_finetuning.predict(img_input)[0][0]
            writer.writerow([row[0],prob])
        subm_file.close()
    test_file.close()


# In[ ]:


# TODO
# Report the following in the 'evaluation' channel on Discord :)

# 2. Chosen model (b0 - b7)
# 3. No global pooling / max global pooling / avg global pooling?
# 4. Name of your first trainable layer (just to make sure the highest level features are finetuned)
# 5. Amount of trainable parameters
# 6. Epochs (50 is fine)
# 7. Final loss, accuracy, precision, recall, auc
# 8. HDF5 file of your model weights
# 9. Submission file
# 10. Submission score

# Now take a coffee


# ## **Code below is not important!**

# In[ ]:


with open('submission.csv', 'w') as subm_file:
        writer = csv.writer(subm_file,delimiter=',')
        writer.writerow(['image_name','target'])
        predictions = eff_finetuning.predict(test_flow)
        for pred in tqdm(predictions):
            prob = pred[0] # or pred[1]
            name = ???
            writer.writerow([name,prob])
        subm_file.close()


# In[ ]:


# Output labels are mainly [0,1], so 'other' class has index 1 and 'melanoma' class index 0
test = next(train_flow)
print(test)

