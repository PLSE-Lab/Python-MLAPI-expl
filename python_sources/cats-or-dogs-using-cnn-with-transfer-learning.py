#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Cats or Dogs - using CNN with Transfer Learning</font></center></h1>
# 
# 
# <center><img src="https://www.theladders.com/wp-content/uploads/dog-cat-190709-1000x563.jpg" width="900"></img></center>
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Load packages and set parameters</a>  
# - <a href='#3'>Read the data</a>  
# - <a href='#4'>Data exploration</a>
#     - <a href='#41'>Class distribution</a>
#     - <a href='#42'>Images samples</a>
# - <a href='#5'>Model</a>  
#     - <a href='#51'>Prepare the model</a>  
#     - <a href='#52'>Train the model</a>  
#     - <a href='#53'>Validation accuracy and loss</a>  
#     - <a href='#54'>Validation accuracy per class</a>  
# - <a href='#6'>Prepare submission</a>     
# - <a href='#7'>Conclusions</a>
# - <a href='#8'>References</a>
# 
# 

# # <a id="1">Introduction</a>  
# 
# 
# ## Dataset
# 
# The **train** folder contains **25,000** images of **dogs** and **cats**. Each image in this folder has the label as part of the filename. The **test** folder contains **12,500** images, named according to a numeric id.  
# For each image in the test set, you should predict a probability that the image is a dog (**1** = **dog**, **0** = **cat**).
# 
# 
# ## Method
# 
# For the solution of this problem we will use a pre-trained model, ResNet-50, replacing only the last layer.

# # <a id="2">Load packages</a>

# In[ ]:


import os, cv2, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from random import shuffle 
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Parameters
# 
# Here we set few parameters used in the model. The image size is **224**.    
# The images are stored in two folders, **train** and **test**.  
# There are two image classes: **Dog** and **Cat**.  
# We will use a subset of the training data set (**20,000** images).  From the training set, **50%** will be used for training, **50%** for validation.  
# A pre-trained model from **ResNet-50** will be used.  
# A number of **10** epochs will be used for training.  
# 
# 

# In[ ]:


TEST_SIZE = 0.5
RANDOM_STATE = 2018
BATCH_SIZE = 64
NO_EPOCHS = 20
NUM_CLASSES = 2
SAMPLE_SIZE = 20000
PATH = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'
TRAIN_FOLDER = './train/'
TEST_FOLDER =  './test/'
IMG_SIZE = 224
RESNET_WEIGHTS_PATH = '/kaggle/input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# # <a id="3">Read the data</a>
# 
# We set the train image list.   
# Setting the **SAMPLE_SIZE** value we can reduce/enlarge the size of the training set.    
# Currently **SAMPLE_SIZE** is set to **20,000**.
# 

# In[ ]:


train_image_path = os.path.join(PATH, "train.zip")
test_image_path = os.path.join(PATH, "test.zip")


# In[ ]:


import zipfile
with zipfile.ZipFile(train_image_path,"r") as z:
    z.extractall(".")


# In[ ]:


with zipfile.ZipFile(test_image_path,"r") as z:
    z.extractall(".")


# In[ ]:


train_image_list = os.listdir("./train/")[0:SAMPLE_SIZE]
test_image_list = os.listdir("./test/")


# We set a function for parsing the image names to extract the first 3 letters from the image names, which gives the label of the image. It will be either a cat or a dog. We are using one hot encoder, storing [1,0] for **cat** and [0,1] for **dog**.

# In[ ]:


def label_pet_image_one_hot_encoder(img):
    pet = img.split('.')[-3]
    if pet == 'cat': return [1,0]
    elif pet == 'dog': return [0,1]


# We are defining as well a function to process the data (both train and test set). 

# In[ ]:


def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img),np.array(label)])
    shuffle(data_df)
    return data_df


# # <a id="4">Data exploration</a>
# 
# 
# ## <a id="41">Class distribution</a>
# 
# Let's inspect the train data to check the **cat**/**dog** distribution.   We show first the split in the reduced train data.

# In[ ]:


def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Cats and Dogs')
    
plot_image_list_count(train_image_list)


# Let's show also the class distribution in the full train data set.

# In[ ]:


plot_image_list_count(os.listdir(TRAIN_FOLDER))


# ## <a id="42">Images samples</a>
# 
# Let's represet some of the images. We start with a selection from the train set. We will show the first 25 images from the train set.
# 
# First,  we process the train data, reading the images and creating a table with images and labels. If the data is trom train set, the label is the one calculated with one hot encoding; if the data is from test set, the label will be the image number.

# In[ ]:


train = process_data(train_image_list, TRAIN_FOLDER)


# Then, we plot the image selection.

# In[ ]:


def show_images(data, isTest=False):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    for i,data in enumerate(data[:25]):
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)
        if label  == 1: 
            str_label='Dog'
        elif label == 0: 
            str_label='Cat'
        if(isTest):
            str_label="None"
        ax[i//5, i%5].imshow(img_data)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("Label: {}".format(str_label))
    plt.show()

show_images(train)


# Let's also show a selection of the train set. We prepare the test set.

# In[ ]:


test = process_data(test_image_list, TEST_FOLDER, False)


# Then, we show a selection of the test set.

# In[ ]:


show_images(test,True)


# # <a id="5">Model</a>
# 
# ## <a id="51">Prepare the model</a>
# 
# Let's start by preparing the model.
# 
# ### Prepare the train data

# In[ ]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array([i[1] for i in train])


# ### Prepare the model
# 
# We initialize the **ResNet-50** model, adding an additional last layer of type **Dense**, with **softmax** activation function.   
# 
# We also set the first layer of the model to be not trainable, becaise **ResNet-50** model was already trained.

# In[ ]:


model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights=RESNET_WEIGHTS_PATH))
model.add(Dense(NUM_CLASSES, activation='softmax'))
# ResNet-50 model is already trained, should not be trained
model.layers[0].trainable = True


# ### Compile the model
# 
# We compile the model, using a **sigmoid** optimized, the loss function as **categorical crossentropy** and the metric **accuracy**.

# In[ ]:


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Model summary
# 
# We plot the model description. We can see that the **ResNet-50** model represent the 1st layer of our model, of type **Model**.

# In[ ]:


model.summary()


# Let's also show the model graphical representation using **plot_model**.

# In[ ]:


plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))


# ### Split the train data in train and validation
# 
# We split the train data in two parts. One will be reserved for train set, the second for validation set. Only the train subset of the data will be used for training the model; the validation set will be used for validation, during training.

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ## <a id="52">Train the model</a>
# 
# We are now ready to train our model.

# In[ ]:


train_model = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val))


# ## <a id="53">Validation accuracy and loss</a>
# 
# Let's show the train and validation accuracy on the same plot. As well, we will represent the train and validation loss on the same graph.

# In[ ]:


def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()
plot_accuracy_and_loss(train_model)


# Let's also show the numeric validation accuracy and loss.

# In[ ]:


score = model.evaluate(X_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])


# ## <a id="54">Validation accuracy per class</a>
# 
# Let's show the validation accuracy per each class.
# 
# We start by predicting the labels for the validation set.

# In[ ]:


#get the predictions for the test data
predicted_classes = model.predict_classes(X_val)
#get the indices to be plotted
y_true = np.argmax(y_val,axis=1)


# We create two indices, **correct** and **incorrect**, for the images in the validation set with class predicted correctly and incorrectly, respectively.

# In[ ]:


correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]


# 
# We saw what is the number of correctly vs. incorrectly predicted values in the validation set.    
# 
# We show here the classification report for the validation set, with the accuracy per class and overall.

# In[ ]:


target_names = ["Class {}:".format(i) for i in range(NUM_CLASSES)]
print(classification_report(y_true, predicted_classes, target_names=target_names))


# # <a id="6">Prepare the submission</a>
# 
# ### Show test images with predicted class
# 
# Let's show few of the test images with the predicted class. For this, we will have to predict the class.
# 

# In[ ]:


f, ax = plt.subplots(5,5, figsize=(15,15))
for i,data in enumerate(test[:25]):
    img_num = data[1]
    img_data = data[0]
    orig = img_data
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: 
        str_predicted='Dog'
    else: 
        str_predicted='Cat'
    ax[i//5, i%5].imshow(orig)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(str_predicted))    
plt.show()


# ### Test data prediction

# In[ ]:


pred_list = []
img_list = []
for img in tqdm(test):
    img_data = img[0]
    img_idx = img[1]
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    predicted = model.predict([data])[0]
    img_list.append(img_idx)
    pred_list.append(predicted[1])


# ### Submission file
# 
# Let's prepare now the submission file.

# In[ ]:


submission = pd.DataFrame({'id':img_list , 'label':pred_list})
submission.head()
submission.to_csv("submission.csv", index=False)


# # <a id="7">Conclusions</a>
# 
# Using a pretrained model for Keras, ResNet-50, with a Dense model with softmax activation added on top and training with a reduced set of  we were able to obtain quite good model in terms of validation accuracy.   
# The model was used to predict the classes of the images from the independent test set and results were submitted to test the accuracy of the prediction with fresh data.  
# 

# # <a id="8">References</a>
# 
# [1] Dogs vs. Cats Redux: Kernels Edition, https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition  
# [2] ResNet pretrained models for Keras, https://www.kaggle.com/keras/resnet50  
# 
# 
# 
# 
