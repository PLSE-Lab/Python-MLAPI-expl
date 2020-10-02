#!/usr/bin/env python
# coding: utf-8

# Skin cancer is the most common human malignancy, is primarily diagnosed visually, beginning with an initial clinical screening and followed potentially by dermoscopic analysis, a biopsy and histopathological examination. Automated classification of skin lesions using images is a challenging task owing to the fine-grained variability in the appearance of skin lesions.
# 
# This the **HAM10000 ("Human Against Machine with 10000 training images")** dataset.It consists of 10015 dermatoscopicimages which are released as a training set for academic machine learning purposes and are publiclyavailable through the ISIC archive. This benchmark dataset can be used for machine learning and for comparisons with human experts. 
# 
# It has 7 different classes of skin cancer which are listed below :<br>
# **1. Melanocytic nevi <br>**
# **2. Melanoma <br>**
# **3. Benign keratosis-like lesions <br>**
# **4. Basal cell carcinoma <br>**
# **5. Actinic keratoses <br>**
# **6. Vascular lesions <br>**
# **7. Dermatofibroma<br>**

# <img src = "https://miiskin.com/wp-content/uploads/2019/08/types-of-skin-cancer-700x514.jpg">

# #### Imports Done!

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
import plotly.express as px


import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# # 1. Exploratory Data Analysis

# In[ ]:


## Kernal Ref - https://www.kaggle.com/kmader/dermatology-mnist-loading-and-processing
base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000/')

# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[ ]:


skin_data = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))


# In[ ]:


skin_data.head()


# In[ ]:


skin_data.info()


# In[ ]:


skin_data['path'] = skin_data['image_id'].map(imageid_path_dict.get)
skin_data['cell_type'] = skin_data['dx'].map(lesion_type_dict.get) 
skin_data['cell_type'] = pd.Categorical(skin_data['cell_type'])
skin_data['cell_type_idx'] = pd.Categorical(skin_data['cell_type']).codes


# In[ ]:


skin_data.isna().sum()
skin_data['age'] = skin_data['age'].fillna(skin_data['age'].median())


# In[ ]:


skin_data.head()


# In[ ]:


tmp = skin_data['cell_type'].value_counts()
tmp = tmp.reset_index()
tmp.columns = ['Classes','count']
fig = px.bar(tmp, x="Classes", y="count", color='Classes')
fig.show()


# In[ ]:


tmp = skin_data['sex'].value_counts()
tmp = tmp.reset_index()
tmp.columns = ['sex','count']
fig = px.pie(tmp, values='count', names='sex')
fig.show()


# In[ ]:


sns.pairplot(skin_data)


# In[ ]:


from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 17.0,8.27
sns.countplot(x="cell_type", hue="sex", data=skin_data,facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("inferno", 3))


# In[ ]:


from skimage.io import imread
skin_data['image'] = skin_data['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))


# In[ ]:


skin_data.head()


# # 2. Data Preprocessing

# In[ ]:


n_samples=5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs,skin_data.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')


# In[ ]:


features=skin_data.drop(columns=['cell_type_idx'],axis=1)
label=skin_data['cell_type_idx']


# In[ ]:


Xtrain, X_test, y_train, y_test = train_test_split(features, label, test_size=0.20,random_state=42)


# In[ ]:


x_train = np.asarray(Xtrain['image'].tolist())
x_test = np.asarray(X_test['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std


# In[ ]:


# Let's confirm the number of classes :p
no_of_classes = len(np.unique(y_train))
no_of_classes


# In[ ]:


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,no_of_classes)
y_test = np_utils.to_categorical(y_test,no_of_classes)
y_train[0] 


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_valid = x_valid.reshape(x_valid.shape[0], *(75, 100, 3))


# In[ ]:


x_train = x_train.astype('float32')/255
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train[0]


# In[ ]:


fig = plt.figure(figsize =(30,5))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_train[i]))


# # 3. Model Build

# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(75,100,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(7,activation = 'softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:


batch_size = 32

checkpointer = ModelCheckpoint(filepath = 'model.hdf5', verbose = 1, save_best_only = True)

history = model.fit(x_train,y_train,
        batch_size = 32,
        epochs=30,
        validation_data=(x_valid, y_valid),
        callbacks = [checkpointer],
        verbose=2, shuffle=True)


# In[ ]:


model.load_weights('model.hdf5')
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])


# # 4.Predictions

# In[ ]:


plt.figure(1)  
   
 # summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
 # summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7)) 


# <b> Streamlit App for Skin cancer Analyzer on my Github </b> <br>
# 
# ### [Github](https://github.com/shashwatwork/Skin-cancer-Analyzer) <br> <br>
# ## Demo Application Available [here](https://skin-cancer-analysis.herokuapp.com/)
# 
# ### Don't hesitate to Upvote if You like this kernal !!!!

# ![Capture.PNG](attachment:Capture.PNG)
