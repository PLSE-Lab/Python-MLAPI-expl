#!/usr/bin/env python
# coding: utf-8

# # **CNN Hummingbird Species and Gender Image Classification**

# The purpose of this project is to create an image classification for hummingbird species and genders that visit feeders in Colorado. (It is applicable to anywhere that hummingbirds migrate or breed given additional datasets for those species.) The ultimate goal is to have a classification system that can be deployed at any feeder. This is important to the continued monitoring of hummingbird species and migration patterns. Humming bird migration is otherwise reliant on individual bird watchers to see and report their observations. If avid bird lovers setup a similar system, then conservation organizations would have better information on migratory and breeding patterns. This knowledge can be used to determine if specific changes in the environment or ecology has positively or negatively impacted bird life.

#  This task for my 8 week project will use the photos collected from my feeders' cameras with a binary bird-finder classification model to create a CNN multi-species and gender classification model. The primary birds at the feeders this year are broad-tailed (*Selasphorous platycerus*) and a few rufous (*Selasphorous rufus*) hummingbirds. So these are the species that are currently in the dataset being analyzed.  

# In[ ]:


from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
#Load an image and determine image shape for analysis.
IMAGE = load_img("/kaggle/input/hummingbirds-at-my-feeders/video_test/06090116image.jpg")
plt.imshow(IMAGE)
plt.axis("off")
plt.show()


# The model I created here with this code was able to find and classify birds with a 97.5% accuracy. The video feed test was a little more difficult due to bird size.  All of the photos (including those in test) are sized to the majority of the window. The video feed test is 'as-is' from the motion detection of the camera. At this time, I am leaving my CNN model as is, because it is likely that my video stream capture / motion detection was focused on the bird feeder motion and not the bird, so it is likely that I would have 2 indicators in the video stream if I were to put a box around the object identified. I would rather capture the motion of the bird.  Two examples that are likely from the same frame-set in the video are presented below. 

# My model classified as No_bird:

# In[ ]:


#Load an image.
IMAGE = load_img("/kaggle/input/hummingbirds-at-my-feeders/video_test/Broadtailed_male/010.jpg")
plt.imshow(IMAGE)
plt.axis("off")
plt.show()


# Although it is able to classify this as a hummingbird.

# In[ ]:


#Load an image.
IMAGE = load_img("/kaggle/input/hummingbirds-at-my-feeders/video_test/Broadtailed_male/012.jpg")
plt.imshow(IMAGE)
plt.axis("off")
plt.show()


# Other errors in my classification vs the CNN model are presented here. I classified this as No_bird and the CNN model classified it as a broadtailed_female. I thought that I had deleted all of the wingtips from the video-set.

# In[ ]:


#Load an image.
IMAGE = load_img("/kaggle/input/hummingbirds-at-my-feeders/video_test/No_bird/203.jpg")
plt.imshow(IMAGE)
plt.axis("off")
plt.show()


# I initially install my needed file management libraries:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))
import random
        
       
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# Setup the directory structure to the training, test, and valid folders: 

# In[ ]:


#This will setup my directories for all of the data files in the 100-bird-species dataset. 
basedir = '/kaggle/input/hummingbirds-at-my-feeders/hummingbirds'
print('Base directory contains ', os.listdir(basedir))
traindir = os.path.join(basedir, 'train')
validdir = os.path.join(basedir, 'valid')
testdir = os.path.join(basedir, 'test')


# Define some functions for evaluating the data: 
# * cntSamples
# * * Counts the samples in each category/directory.
# * indexID
# * * Indexes the categories so I can reverse classify the image sets.
# * label_images
# * * Matches the directory name to the image. Used for presenting/plotting images in the dataset.
# * getKeysbyValues
# * * Matches all the videos to the index from indexID.
#     
# 

# In[ ]:


# Count images for each species
def cntSamples(directory):
    specs = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs.sort()
        for name in dirs:
            if name not in specs:
                specs.append(name)

    # file counts for each species/gender category 
    nums = []
    for b in specs:
        path = os.path.join(directory,b)
        num_files = len(os.listdir(path))
        nums.append(num_files)
         
    # Create Dictionary
    adict = {specs[i]:nums[i] for i in range(len(specs))}
    return adict

#Provide an index number for each species for future reference
def indexID(Dict):
    i=0
    for key, value in Dict.items():
        Dict[key] = i
        i+=1
    return(Dict)

#create seperate labels for images 
def label_images(DIR, dataset):
    label = []
    image = []
    j=0
    for i in range (0,30):
        j = random.randint(0, len(dataset.filenames))
        label.append(dataset.filenames[j].split('/')[0])
        image.append(DIR + '/' + dataset.filenames[j])
    return [label,image]

#create seperate labels for images 
def label_images2(DIR, dataset):
    label = []
    image = []
    qty = len(dataset.filenames)
    for i in range(0,qty):
        label.append(dataset.filenames[i].split('/')[0])
        image.append(DIR + '/' + dataset.filenames[i])
    return [label,image]


#Return names for the predicted species listing.
def getKeysbyValues(LabelDict, testvaluelist):
    testLabellist = []
    listItems = LabelDict.items()
    for testvalue in testvaluelist:
        for value in listItems:
            if value[1] == testvalue:
            #if np.allclose(value[1],testvalue):
                testLabellist.append( {value[0]} )     
    return testLabellist

#Code obtained from: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
#Don't forget that data generator needs Shuffle=False
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm=np.round(cm,2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

        


# Let's look at the hummingbird training dataset.

# In[ ]:


testDict =  cntSamples(testdir)
trainDict = cntSamples(traindir)
validDict = cntSamples(validdir)
index = indexID(cntSamples(traindir))
Dict = {'Index':index,'Train Images':trainDict, 'Test Images':testDict, 'Valid Images':validDict}
speciescnt = pd.DataFrame.from_dict(Dict)
display(HTML(speciescnt.to_html()))


# Load the image processing and displaying packages:

# In[ ]:


from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
import matplotlib.pyplot as plt


# Create data generators for training set and test/valid sets.

# In[ ]:


###################
general_datagen = ImageDataGenerator(rescale=1./255, )
train_datagen = ImageDataGenerator(rescale=1./255,
        #width_shift_range=0.2, ###added
        #height_shift_range=0.2, ###added
        #rotation_range=5, ###added
        brightness_range = [0.5,1.5],
        shear_range=0.2,
        zoom_range=0.2, ###from .2 to .4
        horizontal_flip=True#,
        #fill_mode='nearest'
                                  ) ###added


# Load the files using the data generators.

# In[ ]:


batch_size = 32
train_data = train_datagen.flow_from_directory(
    traindir, classes = index, batch_size=batch_size,
    target_size=(224,224))
test_data = general_datagen.flow_from_directory(
    testdir, classes = index, batch_size=batch_size,
    target_size=(224,224), shuffle = False)
valid_data = general_datagen.flow_from_directory(
    validdir, classes = index, batch_size=batch_size,
    target_size=(224,224), shuffle = False)


# Plot some random samples from our training dataset.

# In[ ]:


#plot the random images.
y,x = label_images(traindir, train_data)

for i in range(0,6):
    X = load_img(x[i])
    plt.subplot(2,3,+1 + i)
    plt.axis(False)
    plt.title(y[i], fontsize=8)
    plt.imshow(X)
plt.show()


# Load the libraries need for Modeling here:

# In[ ]:


import tensorflow as tf
from tensorflow.keras import backend, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Setup some parameters for the model(s).

# In[ ]:


#Modeling parameters
traingroups = len(train_data)
testgroups = len(test_data)
validgroups = len(valid_data)
imagedata=img_to_array(load_img(x[1])) 
SHAPE=imagedata.shape
#This will establish the prediction groups for the model.
classes = os.listdir(traindir)
class_count = len(classes)


# Create the model - in this case using transfer learning with MobileNet - "small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used. MobileNets can be run efficiently on mobile devices" [1] or small computers for offline video feed/detection. This is a good choice for this project. 
# 
# Reference:
# [1] https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

# I will define the model using the classes 'class_count' from the datasets.Compile and fit the model.

# In[ ]:


#Let's try the mobilenet with ReduceLROnPlateau with augmentation
backend.clear_session()

#Bring in the imagenet dataset training weights for the Mobilenet CNN model.
#Remove the classification top.
base_mobilenet = MobileNet(weights = 'imagenet', include_top = False, 
                           input_shape = SHAPE)
base_mobilenet.trainable = False # Freeze the mobilenet weights.

model = Sequential()
model.add(base_mobilenet)

model.add(Flatten()) 
model.add(Activation('relu'))
model.add(Dense(class_count)) 
model.add(Activation('softmax'))

model.summary()

#Compile
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.01, 
                                                  momentum=.9, nesterov=True), #lr=.001/mo=.9:91%,lr=.01/momentum=.8:94%,lr=.01mo=.9:97.5%, lr=.01/mo=.8 92%, lr=.01/mo=.85:95% accuracy
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
#fit modelz
history = model.fit_generator( 
    train_data, 
    steps_per_epoch = traingroups, 
    epochs = 50,
    validation_data = valid_data,
    validation_steps = validgroups,
    verbose = 1,
    callbacks=[EarlyStopping(monitor = 'val_accuracy', patience = 10, 
                             restore_best_weights = True),
               ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, #0.2 to 0.5 dropped to fast 0.7
                                 patience = 3, verbose = 1)]) 
                # left verbose 1 so I could see the learning rate decay


# In[ ]:


#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate against test data.
scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# Here is the confusion matrix for this run on test data.

# In[ ]:



true_label, true_image = label_images2(testdir, test_data)
pred = model.predict_generator(test_data)
pred_classes=np.argmax(pred,axis=1)
label = [index[k] for k in true_label]

conf_mat=confusion_matrix(label,pred_classes)
plt.figure()
plot_confusion_matrix(conf_mat, classes = index,normalize=True)


# In[ ]:



testDict =  cntSamples(testdir)

videofile = '/kaggle/input/hummingbirds-at-my-feeders/video_test'
videodata = general_datagen.flow_from_directory(
    directory = videofile,
    classes = index,
    target_size=(224,224),
    #batch_size=batch_size,
    class_mode = None,
    shuffle = False)
videodata.reset()
pred=model.predict_generator(videodata,verbose=1,steps=267/batch_size)
predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)
predictions = getKeysbyValues(index, predicted_class_indices)
filenames=videodata.filenames
results=pd.DataFrame({"Filename":filenames,
                 "Predictions":predictions})
display(HTML(results.to_html()))


# In[ ]:


true_label, true_image = label_images2(videofile, videodata)
pred = model.predict_generator(videodata)
pred_classes=np.argmax(pred,axis=1)
from sklearn.preprocessing import LabelEncoder
code = LabelEncoder()
label = code.fit_transform(true_label)
#confusion matrix
conf_mat=confusion_matrix(label,pred_classes)
newindex = indexID(cntSamples(videofile))
plt.figure()
plot_confusion_matrix(conf_mat, classes = newindex)


# * Starting point
# * * 25 correct male class, 14 incorrect no bird out of 64 - 95% on test
# * Increased zoom range from .2 to .4 
# * * 20 correct male class, 15 incorrect no bird out of 64. - 94% on test
# * Added augmentation (shift hieght/width 20%) arguments:
# * * 9 correctly classified, 13 incorrect no bird - 94% on test.
# * Added brightness (.5-1.5)
# * * 34 correctly male class, 13 incorrect no bird, 93% on test 
# 
# 
# 
# 
# 
# 
# * Added augmentation (shift hieght/width 20% and zoom range increased from .2 to .4) arguments: 
# * * 28 correct class, 14 incorrect no bird out of 64 - 95% on test
# * Added 40 degree rotation augmentation.
# * * 13 correct 14 incorrect no bird out of 64 91% on test.
# * changed rotation from 40 to 5
# * * 15 correct class 14 incorrect nobird out of 64 and 94% on test
# * Removed rotation and added brightness level (.2,1)
# * * 25 correct male class, 14 incorrect no bird out of 64 - 93% on test
# * Changed from (.2,1) brightness level (.5,1.5)
# 

# In[ ]:


model.save('bird_classifier.h5')


# In[ ]:




#plot the random images.



# In[ ]:


All_images='../input/hummingbirds-at-my-feeders/All_images'
alldata = general_datagen.flow_from_directory(
    All_images, batch_size=batch_size,
    target_size=(224,224), shuffle = False)

true_label, true_image = label_images2(All_images, alldata)
pred = model.predict_generator(alldata)
pred_classes=np.argmax(pred,axis=1)
from sklearn.preprocessing import LabelEncoder
code = LabelEncoder()
label = code.fit_transform(true_label)
#confusion matrix
conf_mat=confusion_matrix(label,pred_classes)
newindex = indexID(cntSamples(All_images))
plt.figure()
plot_confusion_matrix(conf_mat, classes = newindex, normalize =True)


# In[ ]:




