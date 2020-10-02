#!/usr/bin/env python
# coding: utf-8

# **Using this dataset for classifying single palying card by discarding dataset with multiple cards**

# In[ ]:


import pandas as pd


# In[ ]:


#custom function that returns only datasets with single playing card in it
def get_unique_classes(csv_file):
    df = pd.read_csv(csv_file)
    
    df_unique_series = df.groupby('filename')['filename'].count() <= 1
    
    unique_files = []
    for filename, isUnique in zip(df_unique_series.index, df_unique_series):
        if(isUnique):
            unique_files.append(filename)
            
    unique_set = []
    for idx, row in df.iterrows():
        if(row['filename'] in unique_files):
            unique_set.append((row['filename'], row['class']))
            
    return unique_set


# In[ ]:


unique_train_set = get_unique_classes('/kaggle/input/playing-card/card_dataset/train_labels.csv')
unique_test_set = get_unique_classes('/kaggle/input/playing-card/card_dataset/test_labels.csv')

print('unique_train_set', len(unique_train_set))
print('unique_test_set', len(unique_test_set))


# **Create new dataset folder structure with selected datasets**

# In[ ]:


from shutil import copyfile
import os
os.mkdir('/kaggle/cleaned_data')
os.mkdir('/kaggle/cleaned_data/train')
os.mkdir('/kaggle/cleaned_data/test')


# In[ ]:


src_dir = '/kaggle/input/playing-card/card_dataset/train/'
dst_dir = '/kaggle/cleaned_data/train'

for f, c in unique_train_set:
    src = os.path.join(src_dir, f)
    dst = os.path.join(dst_dir, c, f)
    if(os.path.exists(os.path.join(dst_dir, c)) != True):
        os.mkdir(os.path.join(dst_dir, c))
    copyfile(src, dst)
    
src_dir = '/kaggle/input/playing-card/card_dataset/test'
dst_dir = '/kaggle/cleaned_data/test'

for f, c in unique_test_set:
    src = os.path.join(src_dir, f)
    dst = os.path.join(dst_dir, c, f)
    if(os.path.exists(os.path.join(dst_dir, c)) != True):
        os.mkdir(os.path.join(dst_dir, c))
    copyfile(src, dst)


# **Looking inside created data**
# 
# We can see that each tarining and test data contains six classes of cards
# jack, queen, nine, king, ten, ace
# 
# total of 180 taining data
# 
# total of 30 taining data
# 
# Each of class in both training and test dataset are more or less equally spread

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/cleaned_data/'):
    if(len(filenames) != 0):
        print(os.path.basename(dirname), ' ==> ', len(filenames))
    else:
        print(dirname)


# Create baseline model that beats wildguess (accuracy > 16%)

# In[ ]:


from keras_preprocessing import image


# In[ ]:


imgGen = image.ImageDataGenerator(rescale=1/255.)
train_datagen = imgGen.flow_from_directory('/kaggle/cleaned_data/train',
                                           target_size=(150,150),
                                           batch_size=10,
                                           )
val_datagen = imgGen.flow_from_directory('/kaggle/cleaned_data/test',
                                           target_size=(150,150),
                                           batch_size=10,
                                           )


# In[ ]:


from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(6, activation='sigmoid'))


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=0.0006),loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_datagen, epochs=50,
                    steps_per_epoch = 18,
                    validation_data = val_datagen,
                    validation_steps = 3)


# In[ ]:


def get_smoothed(samples, factor = 0.9):
    smoothed = []
    for sample in samples:
        if smoothed:
            prev = smoothed[-1]
            smoothed.append((prev * factor) + (sample * (1-factor)))
        else:
            smoothed.append(sample)
    return smoothed


# In[ ]:


from matplotlib import pyplot as plt


# We can see from training accuracy that due to very low training samples model starts to overfit after first
# 10 epochs
# 
# Also it even roll-over due to close to zero loss (NaN) due to overfitting causing gradient descent update to fail

# In[ ]:


plt.style.use('ggplot')
plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['acc']), 'b', label='Training Accuracy')
plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['val_acc']), 'bo', label='Validation Accuracy')
plt.legend()


# We can improve this by adding regularization and data agumentation during trainin

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.4))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(6, activation='sigmoid'))


# Agumenting data with random transformation while generating training data

# In[ ]:


imgTrainGen = image.ImageDataGenerator(rescale=1/255., rotation_range=40,zoom_range = 0.2, 
                                  horizontal_flip=True, vertical_flip = True, 
                                  shear_range=0.2, height_shift_range=0.2, 
                                  width_shift_range=0.2)

train_datagen = imgTrainGen.flow_from_directory('/kaggle/cleaned_data/train',
                                           target_size=(150,150),
                                           batch_size=10,
                                           )


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=0.0006),loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_datagen, epochs=50,
                    steps_per_epoch = 18,
                    validation_data = val_datagen,
                    validation_steps = 3)


# Now training accuray reached ~37% and validation accuracy is around 25% which is slightly better than random guessing

# In[ ]:


plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['acc']), 'b', label='Training Accuracy')
plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['val_acc']), 'bo', label='Validation Accuracy')
plt.legend()


# Let's try few more epochs

# In[ ]:


history = model.fit_generator(train_datagen, epochs=50,
                    steps_per_epoch = 18,
                    validation_data = val_datagen,
                    validation_steps = 3)


# In[ ]:


plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['acc']), 'b', label='Training Accuracy')
plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['val_acc']), 'bo', label='Validation Accuracy')
plt.legend()

