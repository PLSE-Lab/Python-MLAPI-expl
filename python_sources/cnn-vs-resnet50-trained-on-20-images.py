#!/usr/bin/env python
# coding: utf-8

# # Comparison of CNN vs. ResNet for small data sets
# ### Introduction
# The number of images needed to train deep neural network is dependent on many factors, one being the difficulty of the problem at hand. For example, if you are trying to classify black versus white images you would only need a few images, but if you want to discriminate between nuanced images with a lot of background noise, you may need thousands of images, per class. Therefore it can be helpful to look at previous work on a similar problem that is solved well by a deep neural network, and reduce the training set to see how well it does. The problem we examine here has been solved with >90% classification accuracy when the model was trained on 8,000 images. Can we achieve decent accuracy with fewer images?
# 
# **The goal of this exploration is to get a sense of how many images are needed for difficult problems while utilizing the power of transfer learning.**
# 
# ### Approach:
# In this kernel I will narrow the problem to detect 2 different classes of skin cancer using only 20 images per class for training. I will compare two methods to see how they fare on the small dataset: a Convolution Neural Network (CNN) vs. [ResNet50](https://www.kaggle.com/keras/resnet50), a pre-trained deep neural network which can be altered for different use cases. From work seen in other [kernels](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/kernels), we know that training with 8,000 images allows the CNN to achieve [77% accuracy](https://www.kaggle.com/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy/notebook), and ResNet50 achieves [98% accuracy](https://www.kaggle.com/ingbiodanielh/skin-cancer-classification-with-resnet-50-fastai/notebook). <br>
# 
# ### Main findings:
# We can see that when trained on only 20 images per class, the relatively shallow CNN cannot perform better than random guessing (50% probability of accuracy when guessing between 2 classes).
# 
# However, the deep pre-trained network, ResNet50, achieves 70% accuracy using only 20 images per class. This is a 40% improvement from random guessing, which is quite impressive given the very small training set. This demonstrates the power of transfer learning, as well as the possibility for similarly difficult problems to be solved sufficiently with relatively small datasets.
# 
# ### Context:
# Skin cancer is the most common human malignancy, is primarily diagnosed visually, beginning with an initial clinical screening and followed potentially by dermoscopic analysis, a biopsy and histopathological examination. Automated classification of skin lesions using images is a challenging task owing to the fine-grained variability in the appearance of skin lesions.
# 
# This the **HAM10000 ("Human Against Machine with 10000 training images")** dataset.It consists of 10015 dermatoscopicimages which are released as a training set for academic machine learning purposes and are publiclyavailable through the ISIC archive. This benchmark dataset can be used for machine learning and for comparisons with human experts. 
# 
# **Data sample:**
# ![](https://i.imgur.com/aEVsiFv.png)
# 
# HAM10000 contains 7 different classes of skin cancer, and those used in this experiment are in bold:<br>
# 1. Melanocytic nevi <br>
# 2. Melanoma <br>
# 3. Benign keratosis-like lesions <br>
# 4. **Basal cell carcinoma <br>**
# 5. **Actinic keratoses <br>**
# 6. Vascular lesions <br>
# 7. Dermatofibroma<br>
# 
# ### Contents:
# 
# In this kernel you will find the following steps: <br>
# **Step 1 : Importing Essential Libraries**<br>
# **Step 2: Making Dictionary of images and labels** <br>
# **Step 3: Reading and Processing Data** <br>
# **Step 4: Data Cleaning** <br>
# **Step 5: Exploratory data analysis (EDA)** <br>
# **Step 6: Loading & Resizing of images **<br>
# **Step 7: Label Encoding** <br>
# **Step 8: Data Augmentation** <br>
# **Step 9: Model Buidling** <br>

# # Step 1 : Importing Essential Libraries

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

import keras
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


# In[ ]:


#1. Function to plot model's validation loss and validation accuracy
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


# # Step 2 : Making Dictionary of images and labels
# In this step I have made the image path dictionary by joining the folder path from base directory base_skin_dir and merge the images in jpg format from both the folders HAM10000_images_part1.zip and HAM10000_images_part2.zip

# In[ ]:


base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')

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


# # Step 3 : Reading & Processing data
# 
# In this step we have read the csv by joining the path of image folder which is the base folder where all the images are placed named base_skin_dir.
# After that we made some new columns which is easily understood for later reference such as we have made column path which contains the image_id, cell_type which contains the short name of lesion type and at last we have made the categorical column cell_type_idx in which we have categorize the lesion type in to codes from 0 to 6

# In[ ]:


skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

# Creating New Columns for better readability

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes


# In[ ]:


# Now lets see the sample of tile_df to look on newly made columns
skin_df.head()


# # Step 4 : Data Cleaning
# In this step we check for Missing values and datatype of each field 

# In[ ]:


skin_df.isnull().sum()


# As it is evident from the above that only age has null values which is 57 so we will fill the null values by their mean.

# In[ ]:


skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)


# Now, lets check the presence of null values  again

# In[ ]:


skin_df.isnull().sum()


# In[ ]:


print(skin_df.dtypes)


# # Step 5 : EDA
# In this we will explore different features of the dataset , their distrubtions and actual counts

# Plot to see distribution of 7 different classes of cell type

# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)


# Its seems from the above plot that in this dataset cell type Melanecytic nevi has very large number of instances in comparison to other cell types

# Plotting of Technical Validation field (ground truth) which is dx_type to see the distribution of its 4 categories which are listed below :<br>
# **1. Histopathology(Histo):**  Histopathologic diagnoses of excised lesions have been
# performed by specialized dermatopathologists. <br>
# **2. Confocal:** Reflectance confocal microscopy is an in-vivo imaging technique with a resolution at near-cellular level , and some facial benign with a grey-world assumption of all training-set images in Lab-color space before
# and after  manual histogram changes.<br>
# **3. Follow-up:** If nevi monitored by digital dermatoscopy did not show any changes during 3 follow-up visits or 1.5 years biologists  accepted this as evidence of biologic benignity. Only nevi, but no other benign diagnoses were labeled with this type of ground-truth because dermatologists usually do not monitor dermatofibromas, seborrheic keratoses, or vascular lesions. <br>
# **4. Consensus:** For typical benign cases without histopathology or followup biologists  provide an expert-consensus rating of authors PT and HK. They applied the consensus label only if both authors independently gave the same unequivocal benign diagnosis. Lesions with this type of groundtruth were usually photographed for educational reasons and did not need
# further follow-up or biopsy for confirmation.
# 

# In[ ]:


skin_df['dx_type'].value_counts().plot(kind='bar')


# Plotting the distribution of localization field 

# In[ ]:


skin_df['localization'].value_counts().plot(kind='bar')


# It seems back , lower extremity,trunk and upper extremity are heavily compromised regions of skin cancer 

# Now, check the distribution of Age

# In[ ]:


skin_df['age'].hist(bins=40)


# It seems that there are larger instances of patients having age from 30 to 60

# Lets see the distribution of males and females

# In[ ]:



skin_df['sex'].value_counts().plot(kind='bar')


# Now lets visualize agewise distribution of skin cancer types

# In[ ]:


sns.scatterplot('age','cell_type_idx',data=skin_df)


# It seems that skin cancer types 0,1, 3 and 5 which are Melanocytic nevi,dermatofibroma,Basal cell carcinoma and Vascular lesions are not much prevalant below the age of 20 years 

# Sexwise distribution of skin cancer type

# In[ ]:


sns.factorplot('sex','cell_type_idx',data=skin_df)


# # Step 6: Loading and resizing of images
# In this step images will be loaded into the column named image from the image path from the image folder. We also resize the images as the original dimension of images are 450 x 600 x3 which TensorFlow can't handle, so that's why we resize it into 100 x 75. As this step resize all the 10015 images dimensions into 100x 75 so be patient it will take some time.

# In[ ]:


skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))


# In[ ]:


skin_df.head()


# **As we can see image column has been added in its color format code** 

# In[ ]:


def remove_dups(df, subset=None, index=False):
    """
    Drop all for EXTRA occurences. 
    Arguments:
        ``subset`` - column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by default use all of the columns
        ``index`` - True or False; default False.
            Whether you want to duplication to be judged solely on the index.
    """
    if index: dedup_tf = df.index.duplicated(subset=subset, keep='first') #returns an array with T/F for EXTRA occurences of dates
    else: dedup_tf = df.duplicated(subset=subset, keep='first') #returns an array with T/F for EXTRA occurences of dates
    dedup_indx = np.where(dedup_tf == False) #record non-duplicate indices
    return df.iloc[dedup_indx] #slice df by indicies


# In[ ]:


skin_df_dd = remove_dups(skin_df, subset='lesion_id').copy()


# In[ ]:


#check for duplicate image_ids
skin_df_dd['lesion_id'].value_counts().head()


# ## Reduce to 20 images per class

# In[ ]:


#order by diagnosis
skin_df_dd.sort_values(by=['dx'], inplace=True)


# In[ ]:


#look at the existing abbreviations and choose two
set(skin_df_dd['dx'])


# In[ ]:


#grab 20 images from 2 classes for the training set
akiec = skin_df_dd.iloc[np.where(skin_df_dd['dx']=='akiec')[0][0:20], :] 
bcc = skin_df_dd.iloc[np.where(skin_df_dd['dx']=='bcc')[0][0:20], :] 


# In[ ]:


#concatonate them
train = pd.concat([akiec, bcc])
train.head()


# In[ ]:


#grab 200 images from 2 classes for the test set
akiec_test = skin_df_dd.iloc[np.where(skin_df_dd['dx']=='akiec')[0][20:220], :] 
bcc_test = skin_df_dd.iloc[np.where(skin_df_dd['dx']=='bcc')[0][20:220], :] 


# In[ ]:


#concatonate them
test = pd.concat([akiec_test, bcc_test])
test.head()


# In[ ]:


print(train.shape)
print(test.shape)


# Look at some image samples

# In[ ]:


n_samples = 5
fig, m_axs = plt.subplots(2, n_samples, figsize = (4*n_samples, 3*2))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         train.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)


# In[ ]:


# Checking the image size distribution
train['image'].map(lambda x: x.shape).value_counts()


# In[ ]:


#split x and y columns
x_train = train.drop(columns=['cell_type_idx'],axis=1)
y_train = train['cell_type_idx']

x_test = test.drop(columns=['cell_type_idx'],axis=1)
y_test = test['cell_type_idx']


# In[ ]:


x_train_arr = np.asarray(x_train['image'].tolist())
x_test_arr = np.asarray(x_test['image'].tolist())


# In[ ]:


x_train_arr.shape


# # Step 7 : Label Encoding
# Labels are 7 different classes of skin cancer types from 0 to 6. We need to encode these labels to one hot vectors 

# In[ ]:


# Perform one-hot encoding on the labels
y_train = to_categorical(y_train, num_classes = 2)
y_test = to_categorical(y_test, num_classes = 2)


# In[ ]:


y_train[:5,:]


# In[ ]:


y_train[-6:-1,:]


# # Step 8: Data Augmentation
# It is the optional step. In order to avoid overfitting problem, we need to expand artificially our HAM 10000 dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations 
# 
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.
# 
# By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.
# 
# For the data augmentation, I chose to:
# Randomly rotate some training images by 10 degrees Randomly Zoom by 10% some training images Randomly shift images horizontally by 10% of the width Randomly shift images vertically by 10% of the height 
# Once our model is ready, we fit the training dataset.

# In[ ]:


# With data augmentation to prevent overfitting 

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train_arr)


# #### NOTE: 
# The augmentation object is now instantiated and fit to the data. We will augment the data in real-time during model training using `datagen.flow()`.
# 

# # Step 9: Model Building 

# ## CNN on 20 images per class

# In[ ]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input_shape = (75, 100, 3)
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[ ]:


# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# In[ ]:


# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["acc"])


# In[ ]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


# Fit the model
epochs = 50 
batch_size = 10
history = model.fit_generator(datagen.flow(x_train_arr, y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (x_test_arr, y_test,),
                              verbose = 0, 
                              steps_per_epoch=x_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction])


# In[ ]:


plot_model_history(history)


# ![](http://)Here we can see that the relatively shallow CNN cannot perform better than chance (50% probability when guessing between 2 classes).

# ## Fine-tune ResNet50 on 20 images per class

# In[ ]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# In[ ]:


my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False


# In[ ]:


#compile model
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


#fit model
epochs = 50 
batch_size = 10

history = my_new_model.fit_generator(datagen.flow(x_train_arr, y_train, batch_size=batch_size),
                           verbose = 0,
                           epochs = epochs,
                           steps_per_epoch=x_train.shape[0] // batch_size,
                           validation_data = (x_test_arr, y_test)
                          )


# In[ ]:


plot_model_history(history)


# Here we can see that the deep pre-trained network, ResNet50, achieves 70% accuracy using only 20 images. This demonstrates the power of transfer learning, as well as the possiibility for similarly difficult problems to be solved well with relatively small datasets.

# # Conclusion
# 
# We can see that when trained on only 20 images per class, the relatively shallow CNN cannot perform better than random guessing (50% probability of accuracy when guessing between 2 classes). 
# 
# However, the deep pre-trained network, ResNet50, achieves 70% accuracy using only 20 images per class. This is a 40% improvement from random guessing, which is quite impressive given the very small training set. This demonstrates the power of transfer learning, as well as the possibility for similarly difficult problems to be solved sufficiently with relatively small datasets.
