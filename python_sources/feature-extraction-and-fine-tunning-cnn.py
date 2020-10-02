#!/usr/bin/env python
# coding: utf-8

# **Note: I am referencing course material and some examples which has been taught as part of the module from our Prof. Ted Scully.**

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,GradientBoostingClassifier
from sklearn import metrics

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2  
import h5py
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

#TL pecific modules
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


def loadDataH5():
    with h5py.File('/kaggle/input/data1h5/data1.h5','r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
        valX = np.array(hf.get('valX'))
        valY = np.array(hf.get('valY'))
        print (trainX.shape,trainY.shape)
        print (valX.shape,valY.shape)
    return trainX, trainY, valX, valY


# In[ ]:


trainX, trainY, testX, testY = loadDataH5()


# **Feature Extraction **
# *We consider the VGG 16 Model as our baseline model for feature extraction and set the include_top value to `False`*

# In[ ]:


base_model=VGG16(include_top=False, weights='imagenet',input_shape=(128,128,3))


# In[ ]:


print (base_model.summary())


# In[ ]:


#Feature extraction using VGG16 as baseline model
featuresTrain= base_model.predict(trainX)

#reshape to flatten feature for Train data
featuresTrain= featuresTrain.reshape(featuresTrain.shape[0], -1)

featuresVal= base_model.predict(testX)
#reshape to flatten feature for Test data
featuresVal= featuresVal.reshape(featuresVal.shape[0], -1)


# **feed the extracted feature data into a RandomForest Classifier**

# In[ ]:



model = RandomForestClassifier(400,verbose=1)
model.fit(featuresTrain, trainY)

# evaluate the model

results = model.predict(featuresVal)
print (metrics.accuracy_score(results, testY))


# In[ ]:


#Bagging classifier
model23 = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, 
                            bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                            n_jobs=None, random_state=None, verbose=1)
model23.fit(featuresTrain, trainY)

# evaluate the model

results23 = model23.predict(featuresVal)
print (metrics.accuracy_score(results23, testY))


# In[ ]:


# model32 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
#                                      min_samples_split=2, min_samples_leaf=1, 
#                                      min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
#                                      min_impurity_split=None, init=None, random_state=None, max_features=None, 
#                                      verbose=0, max_leaf_nodes=None, warm_start=False,
#                                      validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
# model32.fit(featuresTrain, trainY)

# # evaluate the model

# results32 = model32.predict(featuresVal)
# print (metrics.accuracy_score(results32, testY))


# **feed the extracted feature data into a Logistic Regression**

# In[ ]:


model1 = LogisticRegression(random_state=0, solver='lbfgs', dual= False, multi_class='multinomial',verbose=1, max_iter=1000).fit(featuresTrain, trainY)

model1.predict(featuresTrain)

#evaluate the model

results_1 = model1.predict(featuresVal)
print(metrics.accuracy_score(results_1,testY))


# **Feature Extraction **
# *We consider the InceptionV3 Model as our another baseline model for feature extraction and set the include_top value to `False`**

# In[ ]:


base_model_1 = InceptionV3(include_top=False, weights='imagenet',input_shape=(128,128,3))


# In[ ]:


#Feature extraction using InceptionV3 as baseline model
featuresTrain_IncepV3= base_model.predict(trainX)

#reshape to flatten feature for Train data
featuresTrain_IncepV3= featuresTrain_IncepV3.reshape(featuresTrain_IncepV3.shape[0], -1)

featuresVal_IncepV3= base_model.predict(testX)
#reshape to flatten feature for Test data
featuresVal_IncepV3= featuresVal_IncepV3.reshape(featuresVal_IncepV3.shape[0], -1)


# **Feeding the extracted feature to the random forest classifier**

# In[ ]:


#Random forest classifier
model2 = RandomForestClassifier(5000,verbose = 1)
model2.fit(featuresTrain_IncepV3, trainY)

# evaluate the model

results_2 = model2.predict(featuresVal_IncepV3)
print (metrics.accuracy_score(results_2, testY))


# In[ ]:


#Bagging classifier
model44 = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, 
                            bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                            n_jobs=None, random_state=None, verbose=1)
model44.fit(featuresTrain_IncepV3, trainY)

# evaluate the model

results44 = model44.predict(featuresVal_IncepV3)
print (metrics.accuracy_score(results44, testY))


# In[ ]:


#Logistic regression
model3 = LogisticRegression(random_state=0, solver='lbfgs',dual= False,max_iter=1000, multi_class='multinomial',verbose =1).fit(featuresTrain_IncepV3, trainY)

model3.predict(featuresTrain_IncepV3)

#evaluate the model

results_3 = model3.predict(featuresVal_IncepV3)
print(metrics.accuracy_score(results_3,testY))


# **Observation**
# * VGG16 model
#             **We noticed that Random forest classifier and Logistic regression are the machine learning algorithm that give us very good results with Random forest classifier giving about 84% and Logistic regression of about 87%. We also tried other classifier's like Bagging but did not notice any improvement in the accuracy. We also observed that the time taken to compute the accuracy was least in Logistic regression.
# * InceptionV3
#             **Here we can observe that we have increased the depth of the nodes for Random forest classifier from 400 to 5000. we can notice that the Random forest classifier gave us 86% and logistic regression gave us 87%. We can alsoo notice that for this baseline model, the Bagging classifier has scored better compared to VGG16 model. 
#             
# **From the above observation and the results obtained we can say that, the best machine learning model that can be used as the secondary model with the baseline model is `logistic regression`. This accuracy obtained is higher and the time for computation is also less compared to other algorithms.**

# **We also create a new model using a portion of original baseline model, here in this case we use VGG19 as baseline model**
# 
# **We also freez the weights of the layers for our initial model.**
# 
# **In our new model we take only the portion of layer from the initial model. Here in this case `block5_conv2`.**
# 
# **Here we are trying to see if we obtain any increase in the accuracy if we unfreeze any layer in our baseline model and then  feed the extracted feature to the machine learning model.

# In[ ]:


initialModel = tf.keras.applications.VGG19(weights = 'imagenet',include_top = False, input_shape =(128,128,3))

newModel = tf.keras.Model(inputs = initialModel.input, outputs = initialModel.get_layer('block5_conv2').output)


# In[ ]:


print(newModel.summary())


# In[ ]:


featuresTrain_1= newModel.predict(trainX)

#reshape to flatten feature for Train data
featuresTrain_1= featuresTrain_1.reshape(featuresTrain_1.shape[0], -1)

featuresVal_1= newModel.predict(testX)
#reshape to flatten feature for Test data
featuresVal_1= featuresVal_1.reshape(featuresVal_1.shape[0], -1)


# In[ ]:



model = RandomForestClassifier(200)
model.fit(featuresTrain, trainY)

# evaluate the model

results = model.predict(featuresVal)
print (metrics.accuracy_score(results, testY))


# In[ ]:


model1 = LogisticRegression(random_state=0, solver='lbfgs',dual= False,max_iter=1000, multi_class='multinomial').fit(featuresTrain, trainY)

model1.predict(featuresTrain)

#evaluate the model

results_1 = model1.predict(featuresVal)
print(metrics.accuracy_score(results_1,testY))


# **We notice and conclude that , Logistic regression is the best machine learning model that can be used as the secondary model after the features have been extracted.**

# In[ ]:





# **Part B: 2** 
# 
# **We now explore Fine Tuning for CNNs as a method of transfer learning. We are going to see which is the best configuration that can be used in order to `increase` the `validation accuracy` for the flower dataset**.
# 
# **At first lets check how much accuracy we are getting for this model without unfreezing any layer in the baseline model.**
# 
# **Second we are going to see by `unfreezing` which layer of the exisiting baseline model, our validation score increases. We are going to try with various layers and see the result. By unfreezing particular portion of the model, we allow weights to be changed as the model learns.**

# **Let us consider the VGG16 baseline model**
# 
# **As we can notice below here we are taking the baseline model and adding few additional layer of our own**

# In[ ]:


NUM_EPOCHS = 50


# In[ ]:


keep_prob = 0.5
# Load the ImageNet VGG model. Notice we exclude the densely #connected layer at the top
vggModel= tf.keras.applications.VGG16( weights='imagenet', include_top=False, input_shape=(128, 128, 3))

vggModel.trainable= False

model = tf.keras.models.Sequential()
#We now add the vggModel directly to our new model
model.add(vggModel)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(rate = 1 - keep_prob))
model.add(tf.keras.layers.Dense(17, activation='softmax'))

print (model.summary())


# **Since this is a multi class classification problem, we are using the loss function as sparse**

# In[ ]:


model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])

H = model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# **We obtained the accuracy of around `87.94`, which is very good.**
# 
# **Now lets try to unfreeze some of the layers and see how the accuracy is obtained. Let us also lower the learning rate by `0.001`**

# In[ ]:


#We set the trainable parameter to True
vggModel.trainable = True

#A flag variable used to change the status. 
trainableFlag = False

for layer in vggModel.layers:
    #As commented previously we are unfreezing the below mentioned layer from the baseline model 
    #for updating the weights
    if layer.name == 'block5_conv2':
        trainableFlag = True
    layer.trainable = trainableFlag
    
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
H1 = model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H1.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H1.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H1.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H1.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# **We can notice that we obtained the validation accuracy of `89.12` by just unfreeezing one layer of the baseline model.**
# 
# **Finally let's try and see, if by unfreezing any other layer, can we increase the validation accuracy**
# 
# **Note I have commented few if statements just to check how many layer's can be unfreezed to increase accuracy**
# 
# **You can uncomment to unfreeze all the layer's mentioned under the if statment or additionlly add your own layer to unfreeze**

# In[ ]:


vggModel.trainable = True
trainableFlag = False

for layer in vggModel.layers:
    if layer.name == 'block4_conv1':
        trainableFlag = True
    if layer.name == 'block4_conv2':
        trainableFlag = True
    if layer.name == 'block4_conv3':
        trainableFlag = True
    if layer.name == 'block4_pool':
        trainableFlag = True
#    if layer.name == 'block5_conv1':
#        trainableFlag = True
#     if layer.name == 'block5_conv2':
#         trainableFlag = True
#     if layer.name == 'block5_conv3':
#         trainableFlag = True
#     if layer.name == 'block5_pool':
#         trainableFlag = True
    layer.trainable = trainableFlag
    
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.0001),metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                               # epsilon=None, decay=0.0, amsgrad=Fals,metrics=['accuracy'])
print (model.summary())
H2 = model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))


# In[ ]:


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H2.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H2.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H2.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H2.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# **We can notice that by unfreezing 4 layer's and reducing the `learning rate = 0.001` we are able to increase the accuracy of the model to `90.29`. This is a significant improvement.**
# 
# **We can notice that by unfreezing 4 layer's and reducing the `learning rate = 0.0001` we are able to increase the accuracy of the model to `92.65`. This is a significant improvement.**
# 
# **We then tried to unfreez another layer with same learning rate and noticed that the accuracy of the model now was slightly less at arounf `90%`. Hence we can  Some of the other techniques which I have tried is by trying different optimizer, but noticed that the best accuracy was found using Stochastic gradient descent.**
# 
# **We have noticed that when we unfreeze more number of layers, when we reduce the learning rate, the accuracy obtained is very high.**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




