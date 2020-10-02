#!/usr/bin/env python
# coding: utf-8

# # Residual Networks(50 Layers)
# 
# This model is made using residual networks with identity blocks and convolutional blocks

# Importing the necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import os,cv2
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Sequential,Input,Model

from keras.initializers import *


# ### Visualisation of Dataset
# 
# To get a feel of the data, exploratory data analysis is done.
# 
# The total number of train and test images from each subdirectory are stored and used in the bar and pie charts

# In[ ]:


datasets = ['../input/intel-image-classification/seg_train/seg_train','../input/intel-image-classification/seg_test/seg_test']

output = []

class_names = ['buildings','forest','glacier','mountain','sea','street']

class_name_labels = {class_name:i for i,class_name in enumerate(class_names)}

nb_classes = len(class_names)
class_name_labels


# In[ ]:


from tqdm import tqdm

def load_data():
    for dataset in datasets:
        print("Loading {}".format(dataset))

        images,labels = [],[]

        for folder in os.listdir(dataset):
            label = class_name_labels[folder]
            
            for file in tqdm(os.listdir(os.path.join(dataset,folder))):
            
                img_path = os.path.join(os.path.join(dataset,folder),file)
    #             print(img_path)
                img = cv2.imread(img_path,cv2.IMREAD_COLOR)
                img = cv2.resize(img,(150,150))

                images.append(img)
                labels.append(label)
                pass
            pass
        
        images = np.array(images,dtype=np.float32)
        labels = np.array(labels,dtype=np.float32)

        output.append((images,labels))
        pass

    return output
    pass


# In[ ]:


(train_images,train_labels),(test_images,test_labels) = load_data()


# In[ ]:


n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

_, train_count = np.unique(train_labels, return_counts=True)
_, test_count = np.unique(test_labels, return_counts=True)

df = pd.DataFrame(data = (train_count,test_count))
df = df.T
df['Index']=['buildings','forest','glacier','mountain','sea','street']
df.columns = ['Train','Test','Name']
df


# Bar chart of the train and test data.

# In[ ]:


plt.figure()
df.set_index('Name').plot.bar(rot=0)
# plt.xticks(df['Name'])


# #### Pie chart of train data
# 
# It can be observed from the pie chart that there is very little class imbalance, which is good for the model. Even then, data augmentation is advised as it helps in making the model more robust
# 
# The maximum number of images are present in the `Mountain` folder with `Glacier` second and `Street` a close third

# In[ ]:


plt.pie(train_count,
       explode=(0,0,0,0,0,0),
       labels = class_names,
       autopct = '%1.1f%%')
plt.axis('equal')
plt.title('Proportion of each observed quantity in train dataset')
plt.show()


# #### Pie chart of test data
# 
# Even here there is no class imbalance with more `Glacier` images compared to `Mountain`, the opposite of the train data.
# 
# `Buildings` in both datasets occupies the lowest position.

# In[ ]:


plt.pie(test_count,
       explode=(0,0,0,0,0,0),
       labels = class_names,
       autopct = '%1.1f%%')
plt.axis('equal')
plt.title('Proportion of each observed quantity in test dataset')
plt.show()


# In[ ]:


def show_final_history(history):
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].set_title('Loss')
    ax[1].set_title("Accuracy")
    ax[0].plot(history.history['loss'],label='Train Loss')
    ax[0].plot(history.history['val_loss'],label='Test loss')
    ax[1].plot(history.history['accuracy'],label='Train Accuracy')
    ax[1].plot(history.history['val_accuracy'],label='Test Accuracy')
    
#     ax.set_ylim(20)
    
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='lower right')


# ### Loading Train and Test Images
# 
# keras.ImageDataGenerator is used for for making the train and test datasets. It allows for augmentation, dividing into batch size,shuffling without going through numpy and making large methods.

# In[ ]:


train_dir = '/kaggle/input/intel-image-classification/seg_train/seg_train'
test_dir = '/kaggle/input/intel-image-classification/seg_test/seg_test'
batch_size = 128

IGD = ImageDataGenerator(rescale=1./255)
train_generator = IGD.flow_from_directory(train_dir,
                                         target_size=(150,150),
                                         color_mode='rgb',
                                         batch_size=batch_size,
                                         class_mode='categorical',
                                         shuffle=True,
                                         seed = 42)
test_generator = IGD.flow_from_directory(test_dir,
                                        target_size=(150,150),
                                        color_mode='rgb',
                                        batch_size=batch_size,
                                        class_mode='categorical',
                                        shuffle=True,
                                        seed = 42)


# In[ ]:


nb_classes = len(train_generator.class_indices)
nb_classes


# ### Identity Block
# 
# This block consists of three Convolutional blocks with the residual block same as the input joining just before the final Activation layer.
# 
# Relu is used for activation with padding `same` so as to maintain the same shape as the input image.

# In[ ]:


def identity_block(X,f,filters,stage,block):
    
    conv_name_base = 'res_'+str(stage)+block+'_branch'
    bn_name_base = 'bn_'+str(stage)+block+'_branch'
    
    F1,F2,F3 = filters
    
    X_shortcut = X
    
    # First Component of Main Path
    X = Conv2D(filters=F1,kernel_size=(3,3),strides=(1,1),
               padding='same',name=conv_name_base+'2a',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    # Second Component of Main Path
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),
              padding='same',name=conv_name_base+'2b',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    # Third Component of Main Path
    X = Conv2D(filters=F3,kernel_size=(3,3),strides=(1,1),
              padding='same',name=conv_name_base+'2c',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2c')(X)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
    pass


# ### Convolutional Block
# 
# This block consists of 4 Convolutional layers, with three in the main path and the fourth in the residual(secondary) path.
# 
# The two paths meet before the final activation layer.
# 
# Relu is used for activation with padding `same` so as to maintain the input shape.

# Padding is maintaned `same` as it allows for greater variations in the model as defined by a user.

# In[ ]:


def convolutional_block(X,f,filters,stage,block,s=2):
    
    conv_base_name = 'res_' + str(stage) + block + '_branch'
    bn_base_name = 'bn_' + str(stage) + block + '_branch'
    
    F1,F2,F3 = filters
    
    X_shortcut = X
    
    ### MAIN PATH ###
    # First component of main path
    X = Conv2D(filters=F1,kernel_size=(3,3),strides=(s,s),
              padding='same',name=conv_base_name+'2a',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2a')(X)
    X = Activation('relu')(X)
    
    # Second Component of main path
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),
              padding='same',name=conv_base_name+'2b',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2b')(X)
    X = Activation('relu')(X)
    
    # Third Component of main path
    X = Conv2D(filters=F3,kernel_size=(3,3),strides=(1,1),
              padding='same',name=conv_base_name+'2c',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2c')(X)
    
    # Shortcut path
    X_shortcut = Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),
                       padding='same',name=conv_base_name+'1',
                       kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(name=bn_base_name+'1')(X_shortcut)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
    pass


# ### ResNet model
# 
# This a 50 layer(convolutional layers) model.
# 
# It consists of 5 stages:
#     
#     Stage 1: Taking the input with kernel size = (7,7) and number of filters = 64.
#     Stage 2: Consists of 1 convolutional block and 2 identity blocks
#     Stage 3: Consists of 1 convolutional block and 3 identity blocks
#     Stage 4: Consists of 1 convolutional block and 5 identity blocks
#     Stage 5: Consists of 1 convolutional block and 2 identity blocks
#     Stage 6: Consists of 1 convolutional block and 3 identity blocks
#     
# Each stage has a Dropout layer with rate set at 0.25(This is introduced in version 6, not in the previous versions)
# 
# Version 6: 66 layers(counting only convolutional layers), previous versions 50.

# In[ ]:


def ResNet(input_shape,classes):
    
    X_input = Input(input_shape)
    
    # Zero Padding
    X = ZeroPadding2D((3,3))(X_input)
    
    # Stage 1
    X = Conv2D(64,(7,7),strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)
    X = Dropout(0.25)(X)
    
    # Stage 2
    X = convolutional_block(X,f=3,filters=[64,64,128],stage=2,block='A',s=1)
    X = identity_block(X,3,[64,64,128],stage=2,block='B')
    X = identity_block(X,3,[64,64,128],stage=2,block='C')
    X = Dropout(0.25)(X)
    
    # Stage 3
    X = convolutional_block(X,f=3,filters=[128,128,256],stage=3,block='A',s=2)
    X = identity_block(X,f=3,filters=[128,128,256],stage=3,block='B')
    X = identity_block(X,f=3,filters=[128,128,256],stage=3,block='C')
    X = identity_block(X,f=3,filters=[128,128,256],stage=3,block='D')
    X = Dropout(0.25)(X)
    
    # Stage 4
    X = convolutional_block(X,f=3,filters=[256,256,512],stage=4,block='A',s=2)
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='B')
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='C')
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='D')
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='E')
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='F')
    X = Dropout(0.25)(X)
    
    # Stage 5
    X = convolutional_block(X,f=3,filters=[512,512,1024],stage=5,block='A',s=1)
    X = identity_block(X,f=3,filters=[512,512,1024],stage=5,block='B')
    X = identity_block(X,f=3,filters=[512,512,1024],stage=5,block='C')
    X = Dropout(0.25)(X)
    
    # Stage 6
    X = convolutional_block(X,f=3,filters=[1024,1024,2048],stage=6,block='A',s=2)
    X = identity_block(X,f=3,filters=[1024,1024,2048],stage=6,block='B')
    X = identity_block(X,f=3,filters=[1024,1024,2048],stage=6,block='C')
    X = identity_block(X,f=3,filters=[1024,1024,2048],stage=6,block='D')
    X = Dropout(0.25)(X)
    
    # Average Pool Layer
    X = AveragePooling2D((2,2),name="avg_pool")(X)
    
    # Output layer
    X = Flatten()(X)
    X = Dense(classes,activation='softmax',name='fc'+str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input,outputs=X,name='ResNet')
    
    return model
    pass


# ### Defining the Model.
# 
# * Input shape = (150,150,3)
# 
# * Classes = 6
# 
# * The method ResNet is called which initializes the model defined by the user.

# In[ ]:


model = ResNet(input_shape=(150,150,3),classes=nb_classes)


# Displaying a summary of the model and storing a flowchart of the model in `model.png`.

# In[ ]:


plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

model.summary()


# Using Stochastic Gradient Descent instead of Adam as there is no erratic changes in the test accuracy and loss values. 
# 
# Learning rate is set at 0.003 and momentum to prevent the optimizer from settling in a local minimum is set at 0.85.
# 
# In version 4 learning rate was set at 0.0001 and momemtum at 0.9, which gave a train accuracy of 99.82% and test accuracy of 76.80%.
# 
# Version 6: learning rate = 0.01 and momentum = 0.7.

# In[ ]:


opt = SGD(lr=0.01,momentum=0.7)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


# Saving the weights of the model to `model_weights.h5`.

# In[ ]:


checkpoint = ModelCheckpoint("model_weights.h5",monitor='val_accuracy',verbose=1,
                              save_best_only=True,mode='max')
callbacks_list=[checkpoint]


# Running the model for 100 epochs.

# In[ ]:


epochs = 100

history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = train_generator.n//batch_size,
                              epochs = epochs,
                              validation_data = test_generator,
                              validation_steps = test_generator.n//batch_size,
                              callbacks = callbacks_list,
                              verbose = 1)


# ### Evaluating the Model
# 
# The model is evaluated using the evaluate_generator method of keras on the test dataset. The method outputs two values, the loss and accuracy.
# 
# The loss and accuracy will be different compared to the one obtained through the epochs. It will be close to the best accuracy obtained through the model.

# In[ ]:


test_evaluate = model.evaluate_generator(test_generator)
print("Loss: {} Accuracy: {}".format(test_evaluate[0], test_evaluate[1]*100))


# The accuracy of the model over 100 epochs increased to 84.767% as compared to 76.8% in version 4.
# 
# Version 6: Accuracy increased to 81.766%

# ### Visualising the loss and accuracy of the model (train and test)
# 
# Two plots are plotted, one for the loss and the other for the accuracy.
# 
# It allows for a user to determine whether the model overfitted or underfitted.
# 
# Advised to check the loss as compared to checking accuracy.

# In[ ]:


show_final_history(history)


# In simple terms, the model overfits. Maybe by removing the last stage or reducing the size of the filters the model may not overfit.
# 
# Of course transfer learning with `ResNet50` module, which is available in Keras, will give an accuracy approaching 90% or higher.

# ### Plotting a Confusion Matrix
# 
# A confusion matrix is plotted to obtain how the model fared with respect to the test dataset. This gives a more accurate representation as compared to accuracy, since one gets to know which label the model can easily predict and one in which there is difficulty.
# 
# A threshold is set at half of the maximum correlation. Above that values are shown in white while below that are shown in black.
# 
# Bluer the shade of the block stronger the correlation between the predicted value and actual value. This type of matrix can also be shown for train data to show how the model fared there.
# 
# The model can be changed using this confusion matrix as a base calculation.

# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools

test_pred = model.predict_generator(generator=test_generator)
y_pred = [np.argmax(probas) for probas in test_pred]
y_test = test_generator.classes
class_names = test_generator.class_indices.keys()

def plot_confusion_matrix(cm,classes,title='Confusion Matrix',cmap=plt.cm.Blues):
    
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")
        pass
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    pass

cnf_mat = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)


plt.figure()
plot_confusion_matrix(cnf_mat,classes=class_names)
plt.show()


# 
# 
# The model is good but can be improved by changing the hyperparameters of the model such as learning rate.

# The model is confusing `sea` with `forest` for an unknown reason.
# 
# But it is evident that the model overfits and takes the noise as features too. This is evudent when using the test dataset by comparing actual class label and predicted class label.

# ### Testing Model on Validation Dataset
# 
# To test how good the model runs in the real world, it is tested on a validation dataset which os not used by the model while training. The images are loaded using the ImageDataGenerator module and the model predicts their classes.

# In[ ]:


pred_dir = '../input/intel-image-classification/seg_pred/'

pred_generator = IGD.flow_from_directory(pred_dir,
                                         target_size=(150,150),
                                         color_mode='rgb',
                                         batch_size=batch_size,
                                         class_mode=None,
                                         shuffle=False)


# Finding to which class the images belong to.

# In[ ]:


pred_generator.reset()
pred_result = model.predict_generator(pred_generator,verbose=1,steps=len(pred_generator))


# Converting the probabilities into class labels, namely, buildings, forest, sea, glacier.

# In[ ]:


predicted_class = np.argmax(pred_result,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred_class = [labels[k] for k in predicted_class]

filenames = pred_generator.filenames
# actual_class = test_generator_1.classes
results = pd.DataFrame({"Filename":filenames,
                        "Predictions":pred_class})
results.head()


# In[ ]:


results_class = results.groupby('Predictions').count()

plt.figure()
results_visualisation = results_class.plot(kind='bar')
results_visualisation.legend_.remove()

plt.title("Number of Images per Class")
plt.xlabel("Class label")
plt.ylabel("Total number of predictions");


# In[ ]:


_,pred_count = np.unique(predicted_class,return_counts=True)

plt.pie(pred_count,
       explode=(0,0,0,0,0,0),
       labels = class_names,
       autopct = '%1.1f%%')
plt.axis('equal')
plt.title('Proportion of labels in validation dataset')
plt.show()


# As it can be seen from the above image `Glacier` is easily predicted with `Street` and `Sea` being a close second

# ### Visualising predictions of validation dataset
# 
# 10 random images from the validation dataset along with the predicted classes are displayed to show how well the model works on unseen data.
# 
# The results dataframe provides the filename and predicted class.

# In[ ]:


from random import randint

base_path = '../input/intel-image-classification/seg_pred/'
l = len(os.listdir(base_path+'seg_pred/'))
# print(l)
for i in range(10):
    
    rnd_number = randint(0,l-1)
#     print(rnd_number)
    filename,pred_class = results.loc[rnd_number]
    img_path = os.path.join(base_path,filename)
#     print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(pred_class)
    plt.show()
    pass


# ### Comparing Predicted labels with Test Data
# 
# The test dataset is also used to evaluate how the model predicts and what it is predicting. 
# 
# The test dataset already has labels which helps the user in determining the actual label of an image.
# 
# The code is similar to the one used in the validation dataset.

# In[ ]:


test_generator_1 = IGD.flow_from_directory(test_dir,
                                          target_size=(150,150),
                                          color_mode='rgb',
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=False)


# In[ ]:


test_generator_1.reset()
test_result = model.predict_generator(test_generator_1,verbose=1,steps=len(test_generator))


# Converting the probabilities into class labels, namely, buildings, glaciers, mountains, etc. 

# In[ ]:


test_class = np.argmax(test_result,axis=1)

test_class = [labels[k] for k in test_class]

filenames = test_generator_1.filenames
actual_class = [labels[h] for h in test_generator_1.classes]
test_results = pd.DataFrame({"Filename":filenames,
                        "Predictions":test_class,
                        "Actual Values":actual_class})
test_results.head()


# In[ ]:


test_class_pred = test_results["Filename"].groupby(test_results["Predictions"]).count()

plt.figure()
plt.figure()
test_results_visualisation = test_class_pred.plot(kind='bar')
# test_results_visualisation.legend_.remove()

plt.title("Number of Images per Class")
plt.xlabel("Class label")
plt.ylabel("Total number of predictions");


# In[ ]:


_,train_count = np.unique(test_class,return_counts=True)

plt.pie(train_count,
       explode=(0,0,0,0,0,0),
       labels = class_names,
       autopct = '%1.1f%%')
plt.axis('equal')
plt.title('Proportion of labels in validation dataset')
plt.show()


# In[ ]:


l = len(filenames)

for i in range(10):
    
    rnd_number = randint(0,l-1)
#     print(rnd_number)
    filename,pred_class,actual_class = test_results.loc[rnd_number]
    img_path = os.path.join(test_dir,filename)
#     print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Predicted class: {} {} Actual Class: {}".format(pred_class,'\n',actual_class))
    plt.show()
#     break
    pass


# ### References
# 
# [ResNet Keras by Priya Dwivedi](https://github.com/priya-dwivedi/Deep-Learning/tree/master/resnet_keras).
# She has explained the model in quite good detail along with the necessary visualisations along for an easier learning.
# 
# Please feel free to comment and fork this notebook. 
