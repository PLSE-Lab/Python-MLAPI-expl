#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install imutils')


# In[ ]:


# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os


# In[ ]:


dataset_path = './dataset'


# # Building the dataset

# In[ ]:



get_ipython().run_cell_magic('bash', '', 'rm -rf dataset\nmkdir -p dataset/covid\nmkdir -p dataset/normal\n')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '\nmkdir -p dataset/try')


# In[ ]:


samples = 25
covid_dataset_path = '../input/covid-chest-xray'


# *** Copy only the covid images from the covid-19 dataset and copy to working/dataset/covid directory ***

# In[ ]:


# construct the path to the metadata CSV file and load it
csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
df = pd.read_csv(csvPath)

# loop over the rows of the COVID-19 data frame
for (i, row) in df.iterrows():
    # if (1) the current case is not COVID-19 or (2) this is not
    # a 'PA' view, then ignore the row
    if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue

    # build the path to the input image file
    imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])

    # if the input image file does not exist (there are some errors in
    # the COVID-19 metadeta file), ignore the row
    if not os.path.exists(imagePath):
        continue

    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = row["filename"].split(os.path.sep)[-1]
    
    outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# In[ ]:


df.head(3)


# In[ ]:


dataframe = df[['patientid','view','finding']]


# In[ ]:


dataframe.head(10)


# In[ ]:


pneumonia_dataset_path ='../input/chest-xray-pneumonia/chest_xray'


# In[ ]:


basePath = os.path.sep.join([pneumonia_dataset_path, "train", "NORMAL"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:samples]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/normal", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# In[ ]:


covid = '../working/dataset/covid/'
normal='../working/dataset/normal/'

print("Number of images in covid = {} and normal ={}".format(len(covid),len(normal)))


# # Ploting the images

# In[ ]:


def ceildiv(a, b):
    return -(-a // b)

def plots_from_files(imspaths, figsize=(10,5), rows=1, titles=None, maintitle=None):
    """Plot the images in a grid"""
    f = plt.figure(figsize=figsize)
    if maintitle is not None: plt.suptitle(maintitle, fontsize=10)
    for i in range(len(imspaths)):
        sp = f.add_subplot(rows, ceildiv(len(imspaths), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img = plt.imread(imspaths[i])
        plt.imshow(img)


# In[ ]:


normal_images = list(paths.list_images(f"{dataset_path}/normal"))
covid_images = list(paths.list_images(f"{dataset_path}/covid"))


# In[ ]:


plots_from_files(normal_images, rows=5, maintitle="Normal X-ray images")


# In[ ]:


plots_from_files(covid_images, rows=5, maintitle="Covid-19 X-ray images")


# # Data Preprocessing

# In[ ]:


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)


# In[ ]:




# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# In[ ]:


print("Total images = {} and labels = {}".format(len(data),len(labels)))


# ***Data Partioning and image augmentation initalizer***

# In[ ]:


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
print("Size of train image : {} and train label : {}".format(len(trainX),len(trainY)))
print("Size of test image : {} and test label : {}".format(len(testX),len(testY)))
trainAug = ImageDataGenerator(rotation_range=15, 
                            fill_mode="nearest",
    
                            zoom_range=0.2,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True)


# # Here i count the number of covid and normal images in trainX and testX

# In[ ]:


y_train = trainY.tolist()
print('Total images in train :',len(y_train))
covid=0
noncovid=0

for idx,i in enumerate(y_train):
    if(i == [1.0, 0.0]):
        covid+=1
    else:
        noncovid+=1
print("In distribution of train covid = {} and noncovid = {}".format(covid,noncovid))


# In[ ]:


y_test = testY.tolist()
print('Total images in test :',len(y_test))
covid=0
noncovid=0

for idx,i in enumerate(y_test):
    if(i == [1.0, 0.0]):
        covid+=1
    else:
        noncovid+=1
print("In distribution of test covid = {} and noncovid = {}".format(covid,noncovid))


# ***Importing the vgg16***

# In[ ]:


baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False


# # Training the model

# In[ ]:


# initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 8


# In[ ]:


from tensorflow import keras
METRICS = ["accuracy",
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]


# In[ ]:


# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=METRICS)

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=15)


# In[ ]:


plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train','Test'],loc='best')
plt.show()

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Test'],loc='best')
plt.show()


# # Evaluation and classification report

# In[ ]:


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))


# In[ ]:



import seaborn as sns 
plt.style.use('fivethirtyeight')
matrix = confusion_matrix(predIdxs,testY.argmax(axis=1))
plt.figure(figsize=(8,8))
ax = plt.subplot()
sns.heatmap(matrix,annot=True,ax=ax)

ax.set_xlabel('Predicted Labels',size=20)
ax.set_ylabel('True Labels',size=20)
ax.set_title('Confusion Matrix(0=Covid and 1=NonCovid)',size=20)


# In[ ]:


labels=['covid','normal']
plt.figure(figsize =(40,40))
for i in range(31):
    plt.subplot(7,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(labels[(np.argmax(testY[i], axis=0))])
    plt.imshow(trainX.reshape(-1,224,224,3)[i])
    
    if(predIdxs[i]==(np.argmax(testY[i], axis=0))):
        plt.xlabel(labels[predIdxs[i]],color='blue')
    else:
        plt.xlabel(labels[predIdxs[i]],color='red')
plt.show()


# # AUC PRECISION RECALL curve

# In[ ]:


colors=['b','r']
def plot_metrics(history):
    plt.figure(figsize =(15,10))
    
    metrics =  [ 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        
        name = metric.replace("_"," ").capitalize()
        plt.subplot(3,3,n+1)
        plt.plot(H.epoch,  H.history[metric], color=colors[0], label='Train')
        plt.plot(H.epoch, H.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()


# In[ ]:


plot_metrics(H)


# In[ ]:




