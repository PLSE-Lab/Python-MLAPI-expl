#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import imageio
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import binary_accuracy
import shutil
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.figure_factory as ff

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_tb=pd.read_csv('/kaggle/input/tuberculosis-chest-xrays-shenzhen/shenzhen_metadata.csv')
data_tb


# In[ ]:


dat = ff.create_table(data_tb.head())
py.iplot(dat)


# In[ ]:



color = plt.cm.Blues(np.linspace(0, 1, 2))
data_tb['sex'].value_counts().plot.pie(colors = color, figsize = (10, 10), startangle = 75)

plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.show()


# In[ ]:


# replacing categorical values in the age column

data_tb['age'] = data_tb['age'].replace('5-14 years', 0)
data_tb['age'] = data_tb['age'].replace('15-24 years', 1)
data_tb['age'] = data_tb['age'].replace('25-34 years', 2)
data_tb['age'] = data_tb['age'].replace('35-54 years', 3)
data_tb['age'] = data_tb['age'].replace('55-74 years', 4)
data_tb['age'] = data_tb['age'].replace('75+ years', 5)

#data['age'].value_counts()

# suicides in different age groups

x1 = data_tb[data_tb['age'] == 0].sum()
x2 = data_tb[data_tb['age'] == 1].sum()
x3 = data_tb[data_tb['age'] == 2].sum()
x4 = data_tb[data_tb['age'] == 3].sum()
x5 = data_tb[data_tb['age'] == 4].sum()
x6 = data_tb[data_tb['age'] == 5].sum()

x = pd.DataFrame([x1, x2, x3, x4, x5, x6])
x.index = ['5-14', '15-24', '25-34', '35-54', '55-74', '75+']
x.plot(kind = 'bar', color = 'grey')

plt.xlabel('Age Group')
plt.ylabel('count')
plt.show()


# In[ ]:


data_tb['findings'].value_counts()


# In[ ]:


def extract_target(x):
    target = int(x[-5])
    if target == 0:
        return 'Normal'
    if target == 1:
        return 'Tuberculosis'


# In[ ]:


data_tb['target'] = data_tb['study_id'].apply(extract_target)


# In[ ]:


def draw_category_images(col_name,figure_cols, df, IMAGE_PATH):

    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories),ncols=figure_cols, 
                         figsize=(4*figure_cols,4*len(categories)))
    for i, cat in enumerate(categories):
        sample = df[df[col_name]==cat].sample(figure_cols) 
        for j in range(0,figure_cols):
            file=IMAGE_PATH + sample.iloc[j]['study_id']
            im=imageio.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=14)  
    plt.tight_layout()
    plt.show()


# In[ ]:


IMAGE_PATH = '/kaggle/input/tuberculosis-chest-xrays-shenzhen/images/images/' 
draw_category_images('target',4, data_tb, IMAGE_PATH)


# In[ ]:


def read_image_sizes(file_name):
    image = cv2.imread(IMAGE_PATH + file_name)
    max_pixel_val = image.max()
    min_pixel_val = image.min()
    
   
    if len(image.shape) > 2:
        output = [image.shape[0], image.shape[1], image.shape[2], max_pixel_val, min_pixel_val]
    else:
        output = [image.shape[0], image.shape[1], 1, max_pixel_val, min_pixel_val]
    return output


# In[ ]:


m = np.stack(data_tb['study_id'].apply(read_image_sizes))
df = pd.DataFrame(m,columns=['w','h','c','max_pixel_val','min_pixel_val'])
data_tb = pd.concat([data_tb,df],axis=1, sort=False)

data_tb.head()


# In[ ]:


data_tb['c'].value_counts()


# In[ ]:


data_tb['labels'] = data_tb['target'].map({'Normal':0, 'Tuberculosis':1})


# In[ ]:


data_tb


# In[ ]:


y = data_tb['labels']

df_train, df_val = train_test_split(data_tb, test_size=0.15, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)


# In[ ]:


df_train['target'].value_counts()


# In[ ]:


df_val['target'].value_counts()


# In[ ]:


base_dir = 'base_dir'
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)
Normal = os.path.join(train_dir, 'Normal')
os.mkdir(Normal)
Tuberculosis = os.path.join(train_dir, 'Tuberculosis')
os.mkdir(Tuberculosis)
Normal = os.path.join(val_dir, 'Normal')
os.mkdir(Normal)
Tuberculosis = os.path.join(val_dir, 'Tuberculosis')
os.mkdir(Tuberculosis)


# In[ ]:


data_tb.set_index('study_id', inplace=True)


# In[ ]:


data_tb.head()


# In[ ]:


NUM_AUG_IMAGES_WANTED = 1000 

# We will resize the images
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40


# In[ ]:


folder_1 = os.listdir('/kaggle/input/tuberculosis-chest-xrays-shenzhen/images/images/')


# Get a list of train and val images
train_list = list(df_train['study_id'])
val_list = list(df_val['study_id'])



# Transfer the train images

for image in train_list:
    
    fname = image
    label = data_tb.loc[image,'target']
    
    if fname in folder_1:
        src = os.path.join('/kaggle/input/tuberculosis-chest-xrays-shenzhen/images/images/', fname)
        dst = os.path.join(train_dir, label, fname)
        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        cv2.imwrite(dst, image)
        

   


for image in val_list:
    
    fname = image
    label = data_tb.loc[image,'target']
    
    if fname in folder_1:
        src = os.path.join('/kaggle/input/tuberculosis-chest-xrays-shenzhen/images/images/', fname)
        dst = os.path.join(val_dir, label, fname)
        image = cv2.imread(src)
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        cv2.imwrite(dst, image)


# In[ ]:


print(len(os.listdir('base_dir/train_dir/Normal')))
print(len(os.listdir('base_dir/train_dir/Tuberculosis')))


# In[ ]:


print(len(os.listdir('base_dir/val_dir/Normal')))
print(len(os.listdir('base_dir/val_dir/Tuberculosis')))


# In[ ]:


class_list = ['Normal','Tuberculosis']

for item in class_list:
    
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    img_class = item

    img_list = os.listdir('base_dir/train_dir/' + img_class)

    for fname in img_list:
            src = os.path.join('base_dir/train_dir/' + img_class, fname)
            dst = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst)


    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                           save_to_dir=save_path,
                                           save_format='png',
                                                    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                                    batch_size=batch_size)
    
    
    
    
    num_files = len(os.listdir(img_dir))
    
    num_batches = int(np.ceil((NUM_AUG_IMAGES_WANTED-num_files)/batch_size))

    for i in range(0,num_batches):

        imgs, labels = next(aug_datagen)
        
    shutil.rmtree('aug_dir')


# In[ ]:


print(len(os.listdir('base_dir/train_dir/Normal')))
print(len(os.listdir('base_dir/train_dir/Tuberculosis')))


# In[ ]:


print(len(os.listdir('base_dir/val_dir/Normal')))
print(len(os.listdir('base_dir/val_dir/Tuberculosis')))


# In[ ]:


def plots(ims, figsize=(20,10), rows=5, interp=False, titles=None): # 12,6
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
plots(imgs, titles=None) 


# In[ ]:


train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


# In[ ]:


datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                        batch_size=val_batch_size,
                                        class_mode='categorical',
                                        shuffle=False)


# In[ ]:



kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', 
                 input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()


# In[ ]:


model.compile(Adam(lr=0.0001), loss='binary_crossentropy', 
              metrics=['accuracy'])


# In[ ]:


aug_dir_1 = 'model5'
os.mkdir(aug_dir_1)
filepath = "/kaggle/working/model5/model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            epochs=100, verbose=1,
                           callbacks=callbacks_list)


# In[ ]:


model.metrics_names


# In[ ]:


val_loss, val_acc = model.evaluate_generator(test_gen, 
                        steps=val_steps)

print('val_loss:', val_loss)
print('val_acc:', val_acc)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


test_labels = test_gen.classes


# In[ ]:


test_labels


# In[ ]:


predictions = model.predict_generator(test_gen, steps=val_steps, verbose=1)


# In[ ]:


predictions.shape


# In[ ]:


test_labels.shape


# In[ ]:


cm = confusion_matrix(test_labels, predictions.argmax(axis=1))


# In[ ]:


test_gen.class_indices


# In[ ]:


import seaborn as sns
sns.heatmap(cm, annot=True)

