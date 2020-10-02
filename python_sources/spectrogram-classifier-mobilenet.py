#!/usr/bin/env python
# coding: utf-8

# In[ ]:


batch_size = 64
epochs = 150


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import layers
import librosa
import numpy as np
import glob
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py
audio_dir = '../input/freesound-audio-tagging/'
feature_dir = '../input/spectrogram-data-preparation/'


# In[ ]:


train_labels = pd.read_csv(os.path.join(audio_dir, "train.csv"))
print(len(train_labels), 'training')
train_labels.groupby(['label']).size().plot.bar()
train_labels.sample(3)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
full_train_ds_pointer = h5py.File(os.path.join(feature_dir, 'spectrograms.h5'))
for k in full_train_ds_pointer.keys():
    print(k, full_train_ds_pointer[k].shape)
input_length = full_train_ds_pointer['spectrograms'].shape[1]
N_MEL_COUNT = full_train_ds_pointer['spectrograms'].shape[2]
lab_enc = LabelEncoder()
all_labels = lab_enc.fit_transform(full_train_ds_pointer['label'].value)
nclass = len(lab_enc.classes_)


# # Show a few example spectrograms
# We can see what sort of patterns the model should be looking for

# In[ ]:


fig, m_axs = plt.subplots(15, 8, figsize = (10, 30))
for c_class, c_axs in zip(np.random.permutation(range(nclass)), m_axs):
    idxs = np.where(all_labels==c_class)[0]
    c_axs[0].set_title(lab_enc.classes_[c_class].decode())
    c_axs[0].set_xlabel('Time')
    c_axs[1].set_ylabel('Frequency')
    for c_idx, c_ax in zip(np.random.permutation(idxs), c_axs):
        c_ax.axis('off')
        c_ax.imshow(full_train_ds_pointer['spectrograms'][c_idx][:, :, 0].swapaxes(0, 1))


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(
    full_train_ds_pointer['spectrograms'].value,
    all_labels,
    test_size = 0.2, 
    random_state = 2018)
print('Training', train_X.shape, train_y.shape)
print('Validation', valid_X.shape, valid_y.shape)


# ## Setup the Model

# In[ ]:


from keras import layers, metrics, applications
# this is what the results are scored on, so we should keep this
def top_3_accuracy(x, y): 
    return metrics.sparse_top_k_categorical_accuracy(x,y,3)
def create_model():
    model = applications.mobilenet.MobileNet(input_shape=train_X.shape[1:],
                                    classes=nclass,
                                    weights=None)
    opt = optimizers.Adam(lr=4e-4)
    
    model.compile(optimizer=opt, 
                  loss=losses.sparse_categorical_crossentropy, 
                  metrics=[metrics.sparse_categorical_accuracy,
                           top_3_accuracy])
    model.summary()
    return model
model = create_model()


# In[ ]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
weight_path="{}_weights.best.hdf5".format('spectro_sound_model')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', 
                                   factor=0.8, patience=5, 
                                   verbose=1, mode='auto', 
                                   epsilon=0.0001, cooldown=5, 
                                   min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


from IPython.display import clear_output
fit_results = model.fit(train_X, train_y,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(valid_X, valid_y), 
          callbacks=callbacks_list)
clear_output()


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
ax1.plot(fit_results.history['loss'], label='Training')
ax1.plot(fit_results.history['val_loss'], label='Validation')
ax1.legend()
ax1.set_title('Loss')
ax2.plot(fit_results.history['sparse_categorical_accuracy'], label='Training')
ax2.plot(fit_results.history['val_sparse_categorical_accuracy'], label='Validation')
ax2.legend()
ax2.set_title('Top 1 Accuracy')
ax2.set_ylim(0, 1)
ax3.plot(fit_results.history['top_3_accuracy'], label='Training')
ax3.plot(fit_results.history['val_top_3_accuracy'], label='Validation')
ax3.legend()
ax3.set_title('Top 3 Accuracy')
ax3.set_ylim(0, 1);


# In[ ]:


model.load_weights(weight_path)
for k, v in zip(model.metrics_names, 
        model.evaluate(valid_X, valid_y)):
    if k!='loss':
        print('{:40s}:\t{:2.1f}%'.format(k, 100*v))


# In[ ]:


model.save("spectro_baseline_cnn.h5")


# # Run Predictions on Test Data

# In[ ]:


full_test_ds_pointer = h5py.File(os.path.join(feature_dir, 'test_spectrograms.h5'))
for k in full_test_ds_pointer.keys():
    print(k, full_test_ds_pointer[k].shape)
test_files = [x.decode() for x in full_test_ds_pointer['fname'].value]
test_preds = model.predict(full_test_ds_pointer['spectrograms'].value, 
                           batch_size=batch_size, verbose=True)


# In[ ]:


top_3 = lab_enc.classes_[np.argsort(-test_preds, axis=1)[:, :3]] #https://www.kaggle.com/inversion/freesound-starter-kernel
pred_labels = [' '.join([cat.decode() for cat in row]) for row in top_3]
pred_labels[0:2]


# In[ ]:


df = pd.DataFrame(test_files, columns=["fname"])
df['label'] = pred_labels
df['fname'] = df.fname.apply(lambda x: x.split("/")[-1])
df.sample(3)


# In[ ]:


df.to_csv("baseline.csv", index=False)


# In[ ]:




