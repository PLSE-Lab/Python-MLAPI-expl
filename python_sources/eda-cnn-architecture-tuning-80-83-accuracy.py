#!/usr/bin/env python
# coding: utf-8

# ## Introduction:
# 
# Cancer is a leading cause of death worldwide, accounting for over 10 million deaths in 2019. One in every three cancers diagnosed is a skin cancer.  Currently, between 2 and 3 million non-melanoma skin cancers and 132,000 melanoma skin cancers occur globally each year. In this kernel we build a model to predict various types of skin cancer. The dataset contains lesion images for the following types of skin cancer:
# 1. Melanocytic nevi
# 2. Melanoma
# 3. Benign keratosis-like lesions
# 4. Basal cell carcinoma
# 5. Actinic keratoses
# 6. Vascular lesions
# 7. Dermatofibroma
# 
# ### Dataset & background information:
# Training of neural networks for automated diagnosis of pigmented skin lesions is hampered by the small size and lack of diversity of available dataset of dermatoscopic images. We tackle this problem by releasing the HAM10000 ("Human Against Machine with 10000 training images") dataset. We collected dermatoscopic images from different populations, acquired and stored by different modalities. The final dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).
# 
# More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (followup), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). 

# ## Importing Libraries

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
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

sns.set(color_codes = True)

#Global Variables
styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# ## Loading the data and pre-processing
# 
# We start with creating a dictionary to map ImageId and ImagePath which will be used into add images to the master dataframe `HAM10000_metadata.csv`.

# In[ ]:


# Create a dictionary with image_id, image_path as key value pairs. 
base_skin_dir = os.path.join('..', 'input')

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir,'*','*','*.jpg'))}

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


skin_df = pd.read_csv(os.path.join(base_skin_dir,'skin-cancer-mnist-ham10000','HAM10000_metadata.csv'))

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

skin_df.head()


# In[ ]:


skin_df.isnull().sum()


# In[ ]:


skin_df.age.describe()


# In[ ]:


skin_df["age"].fillna(skin_df.age.median(), inplace = True)


# ## Univariate EDA

# In[ ]:


sns.set(font_scale = 1.25)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 2, figsize=(20, 25))
sns.despine(left=True)

# Age distribution
sns.kdeplot(skin_df["age"], legend = False, shade = True, ax=axes[0, 0])
axes[0,0].set_xlabel("Age", fontsize=17)
axes[0,0].set_title("Age: Distribution Plot", fontsize=20)

# Gender distribution
sns.countplot(x = "sex", data = skin_df, ax=axes[0, 1])
axes[0,1].set_xlabel("Gender", fontsize=17)
axes[0,1].set_title("Gender: Distribution Plot", fontsize=20)
axes[0,1].set_ylabel("Count", fontsize=17)

# Diagnosis Test type distribution
sns.countplot(x = "dx_type", data = skin_df, ax=axes[1, 0])
axes[1,0].set_xlabel("Diagnosis Test Type", fontsize=17)
axes[1,0].set_ylabel("Count", fontsize=17)
axes[1,0].set_title("Diagnosis Test Type: Distribution Plot", fontsize=20)

# Lesion type distribution
sns.countplot(x = "cell_type", data = skin_df, ax=axes[1, 1])
axes[1,1].set_xlabel("Lesion Type", fontsize=17)
axes[1,1].set_ylabel("Count", fontsize=17)
axes[1,1].set_title("Lesion Type: Distribution Plot", fontsize=20)

## Lesion type distribution
sns.countplot(x = "localization", data = skin_df, ax=axes[2, 0])
axes[2,0].set_xlabel("Localization Area", fontsize=17)
axes[2,0].set_ylabel("Count", fontsize=17)
axes[2,0].set_title("Localization Area: Distribution Plot", fontsize=20)

c = 0
for ax in f.axes:
    c+=1
    if c<=3:
        continue
    plt.sca(ax)
    plt.xticks(rotation=90)
    
plt.subplots_adjust(top=0.95)
f.suptitle('Univariate Distributions', fontsize=25)
f.delaxes(axes[2,1]) 


# ### Summary:
# 
# 1. Lesion type Melanocytic nevi has maximum count.
# 2. Majority of the images are from the following regions: back, lower extremity and trunk.

# ## Bivariate EDA

# In[ ]:


sns.set(font_scale = 1.25)

# Set up the matplotlib figure
f, axes = plt.subplots(1, 2, figsize=(20, 15))
sns.despine(left=True)

# Gender distribution
sns.boxplot(x = "sex", y = "age", data = skin_df, ax=axes[0])
axes[0].set_xlabel("Gender", fontsize=17)
axes[0].set_ylabel("Age", fontsize=17)
axes[0].set_title("Gender V. Age Boxplot", fontsize=20)

# Lesion distribution
sns.boxplot(x = "cell_type", y = "age", data = skin_df, ax=axes[1])
axes[1].set_xlabel("Lesion Type", fontsize=17)
axes[1].set_ylabel("Age", fontsize=17)
axes[1].set_title("Lesion Type V. Age Boxplot", fontsize=20)

c = 0
for ax in f.axes:
    c+=1
    if c<=1:
        continue
    plt.sca(ax)
    plt.xticks(rotation=90)


# ### Summary:
# 
# 1. We observe that IQR for male ages in the dataset is higher than females.
# 2. In general, paitents with Melanocytic nevi lesion type are younger.

# ## Loading Images into the dataframe

# In[ ]:


get_ipython().run_cell_magic('time', '', "skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))")


# In[ ]:


skin_df.head()


# ## Lesion Images by type

# In[ ]:


n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
# fig.savefig('category_samples.png', dpi=300)


# ## Preparing model input data

# In[ ]:


# Checking the image size distribution
skin_df['image'].map(lambda x: x.shape).value_counts()


# We will keep the train, val and test split ratio to be 75%, 15% and 10% respectively

# In[ ]:


features = skin_df["image"]
target = skin_df["cell_type_idx"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.1, random_state = 999)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.165, random_state = 999)
print("Shape of entire dataset: {}".format(str(features.shape)))
print("Shape of train data: {}".format(str(X_train.shape)))
print("Shape of test data: {}".format(str(X_test.shape)))
print("Shape of val data: {}".format(str(X_val.shape)))


# In[ ]:


l = ["train","val","test"]

for x in l:
    globals()["X_{}".format(x)] = np.asarray(globals()["X_{}".format(x)].tolist())
    globals()["X_{}_mean".format(x)] = np.mean(globals()["X_{}".format(x)])
    globals()["X_{}_std".format(x)] = np.std(globals()["X_{}".format(x)])
    globals()["X_{}".format(x)] = (globals()["X_{}".format(x)]-globals()["X_{}_mean".format(x)])/globals()["X_{}_std".format(x)]


# In[ ]:


# Perform one-hot encoding on the labels
y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)
y_val = to_categorical(y_val, num_classes = 7)


# In[ ]:


# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
X_train = X_train.reshape(X_train.shape[0], *(75, 100, 3))
X_test = X_test.reshape(X_test.shape[0], *(75, 100, 3))
X_val = X_val.reshape(X_val.shape[0], *(75, 100, 3))


# In[ ]:


input_shape = (75, 100, 3)
num_classes = 7


# ## Notation:
# 
# The following notations will be used for the rest of this book:
# 1. $_fC_kS_n$: Denotes a convolution layer with $f$ feature maps, kernel size $(k,k)$ and $n$ strides
# 2. $M(_fC_kS_n)$: Denotes $M$ convolution layers with $_fC_kS_n$ configuration
# 2. $P_k$: Denotes a max pooling layer with a kernel size $(k,k)$ and $2$ strides

# ## Data Augmentation
# 
# We will apply data augmentation to improve model generalisation and reduce overfitting. 

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

datagen.fit(X_train)


# ## Step-1 : Number of convolution-subsampling pairs
# 
# We will train 3 models with different number of conv-subsampling pairs. Our aim is to obtain a high prediction accuracy with a relatively low training time. <br />
# 
# Model architectures:
# 1. **Model 1.1:**  $784$ -- $2(_{32}C_3S_1) P_2$ -- $256$ -- $7$ <br/>
# 2. **Model 1.2:**  $784$ -- $2(_{32}C_3S_1) P_2$ -- $2(_{64}C_3S_1) P_2$ -- $256$ -- $7$ <br/>
# 3. **Model 1.3:**  $784$ -- $2(_{32}C_3S_1) P_2$ -- $2(_{64}C_3S_1) P_2$ -- $2(_{128}C_3S_1) P_2$ -- $256$ -- $7$ <br/>

# In[ ]:


nets = 3
input_shape = (75, 100, 3)
num_classes = 7
model = [0]*nets

for i in range(nets):
    model[i] = Sequential()
    model[i].add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
    model[i].add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
    model[i].add(MaxPool2D(pool_size = (2,2)))
    if i>0:
        model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
        model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
        model[i].add(MaxPool2D(pool_size = (2,2)))
    if i>1:
        model[i].add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
        model[i].add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
        model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Flatten())
    model[i].add(Dense(256, activation = 'relu'))
    model[i].add(Dense(num_classes, activation = 'softmax'))
    model[i].compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])        


# In[ ]:


history = [0]*nets
names = ['Model 1.1','Model 1.2','Model 1.3']
epochs = 20 
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                        epochs = epochs, validation_data = (X_val,y_val),
                                        verbose = 0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
          epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))


# In[ ]:


# Plot Model Performance
plt.figure(figsize=(15,5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'],linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.72,0.79])
plt.show()


# ### Summary:
# 
# As **Model 1.3** has the highest validation set accuracy with a similar runtime as the other two models, we will use it as a base for further tuning.

# ## Step-2: Number of feature maps in each layer
# 
# In this step we tune the number of feature maps in the model. <br />
# 
# Model architectures:
# 1. **Model 2.1:** $784$ -- $2(_{32}C_3S_1) P_2$ -- $2(_{64}C_3S_1) P_2$ -- $2(_{128}C_3S_1) P_2$ -- $256$ -- $7$ 
# 2. **Model 2.2:** $784$ -- $2(_{64}C_3S_1) P_2$ -- $2(_{128}C_3S_1) P_2$ -- $2(_{256}C_3S_1) P_2$ -- $256$ -- $7$ 
# 3. **Model 2.3:** $784$ -- $2(_{128}C_3S_1) P_2$ -- $2(_{256}C_3S_1) P_2$ -- $2(_{512}C_3S_1) P_2$ -- $256$ -- $7$ 

# In[ ]:


nets = 3
model = [0]*nets
for i in zip(range(nets),[32,64,128]):
    model[i[0]] = Sequential()
    model[i[0]].add(Conv2D(filters = i[1], kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(Conv2D(filters = i[1], kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(MaxPool2D(pool_size = (2,2)))
    model[i[0]].add(Conv2D(filters = i[1]*2, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(Conv2D(filters = i[1]*2, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(MaxPool2D(pool_size = (2,2)))
    model[i[0]].add(Conv2D(filters = i[1]*4, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(Conv2D(filters = i[1]*4, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(MaxPool2D(pool_size = (2,2)))
    model[i[0]].add(Flatten())
    model[i[0]].add(Dense(256, activation = 'relu'))
    model[i[0]].add(Dense(num_classes,activation = 'softmax'))
    model[i[0]].compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


history = [0]*nets
names = ['Model 2.1','Model 2.2','Model 2.3']
epochs = 20 
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                        epochs = epochs, validation_data = (X_val,y_val),
                                        verbose = 0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
          epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))


# In[ ]:


# Plot Model Performance
plt.figure(figsize=(15,5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'],linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.73,.78])
# axes.set_xlim([0,20])
plt.show()


# ### Summary:
# 
# **Model 2.2** performs much better than the rest of the models. We will use it for further tuning.

# ## Step-3: Number of nodes in the dense layer
# 
# In Step-2, **Model 2.2** was trained with $256$ nodes in the dense layer. We train the same model with $512$ and $1024$ nodes and observe whether or not there is a significant improvement in the performance.<br />
# 
# Model architectures:
# 1. **Model 3.1:** $784$ -- $2(_{64}C_3S_1) P_2$ -- $2(_{128}C_3S_1) P_2$ -- $2(_{256}C_3S_1) P_2$ -- $512$ -- $7$ 
# 2. **Model 3.2:** $784$ -- $2(_{64}C_3S_1) P_2$ -- $2(_{128}C_3S_1) P_2$ -- $2(_{256}C_3S_1) P_2$ -- $1024$ -- $7$ 

# In[ ]:


nets = 2
model = [0]*nets

for i in range(2):
    model[i] = Sequential()
    model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Flatten())
    if i == 0:
        model[i].add(Dense(512, activation = 'relu'))
    elif i == 1:
        model[i].add(Dense(1024, activation = 'relu'))
    model[i].add(Dense(num_classes,activation = 'softmax'))
    model[i].compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


history = [0]*nets
names = ['512N','1024N']
epochs = 20 
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                        epochs = epochs, validation_data = (X_val,y_val),
                                        verbose = 0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
          epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))


# In[ ]:


# Plot Model Performance
nets = 2
names = ['512N','1024N']

plt.figure(figsize=(15,5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'],linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.72,.78])
plt.show()


# ### Summary:
# 
# **Model 2.2** built in **Step-2** had 256 nodes in the dense layer with peak validation set accuracy of $77.55\%$. We will consider this as the baseline to compare the models with $512$ and $1024$ nodes respectively.
# 
# #### 1. **Model 3.1:** 
#     Max Validation Set Accuracy: 77.82% 
#        
# #### 2. **Model 3.2:**
#     Max Validation Set Accuracy: 77.75% 
#        
# **Model 3.1** with $512$ nodes in the dense layer gives the highest accuracy with similar train-time/epoch. Hence we use this architecture.

# ## Step-4: Adding batch normalization
# 
# In this step we add batch normalization after each convolution layer and dense layer. Batch normalization reduces overfitting and speeds up convergence. We also increase the number of epochs for final tuning and introduce early stopping.

# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', patience=6)
mod_chckpt = ModelCheckpoint(filepath='model_v1.h5', monitor='val_loss', save_best_only=True)


# In[ ]:


history = [0]
epochs = 25
batch_size = 50

history[0] = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                epochs = epochs, validation_data = (X_val,y_val),
                                verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size,
                                callbacks=[learning_rate_reduction, early_stop, mod_chckpt])


# ## Test Set Accuracy:

# In[ ]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(X_val, y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))


# A test set accuracy of 80.83% is obtained.
