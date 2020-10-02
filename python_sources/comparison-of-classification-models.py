#!/usr/bin/env python
# coding: utf-8

# # A Comparison of Classification models for Traffic Signs
# 
# 

# ## 1. Loading necessary libraries and data

# In[ ]:


# Loading necessary libraries

# Loading numpy and pandas
import numpy as np 
import pandas as pd 
from numpy.random import seed

# Loading libraries to read images
from PIL import Image
from skimage.io import imread
import cv2

# Loading necessary libraries from sklearns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# Loading keras for CNN
from keras.utils import to_categorical
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Loading libraries for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Loading other required libraries
import random
import time
import datetime


# In[ ]:


# Setting seed

seed(111)


# In[ ]:


# loading train data

train = pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Train.csv')
train.head()


# In[ ]:


# loading test data

test = pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Test.csv')
test.head()


# In[ ]:


# Loading train images and their respective classes

train_x=[]
train_x_vis=[]
p='/kaggle/input/gtsrb-german-traffic-sign/'
for i in train['Path']:
    try:
        img = Image.fromarray(cv2.imread(p+i), 'RGB')
        train_x.append(np.array(img.resize((32, 32))))
        train_x_vis.append(np.array(img.resize((1,1)))) #For data visualization
    except AttributeError:
        print("Error in loading image")
train_x=np.array(train_x)
train_x_vis=np.array(train_x_vis)
train_y = np.array(train['ClassId'].values)
train_x.shape


# In[ ]:


# Loading train images and their respective classes

test_x=[]
p='/kaggle/input/gtsrb-german-traffic-sign/'
for i in test['Path']:
    try:
        img = Image.fromarray(cv2.imread(p+i), 'RGB')
        test_x.append(np.array(img.resize((32, 32))))
    except AttributeError:
        print("Error in loading image")
test_x=np.array(test_x)
test_y = np.array(test['ClassId'].values)
test_x.shape


# ## 2. Exploring the data

# In[ ]:


# Loading random images to check if images have been stored in correct format
plot1 = plt
plot1.figure(figsize=(5,5))
plot1.subplot(221), plot1.imshow(train_x[100])
plot1.subplot(222), plot1.imshow(train_x[500])
plot1.subplot(223), plot1.imshow(train_x[1400])
plot1.subplot(224), plot1.imshow(train_x[6000])


# In[ ]:


# Number of images per class in train data

unique_class, counts_class = np.unique(train_y, return_counts=True)
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,1.5])
ax.bar(unique_class,counts_class)
ax.set_xlabel('Classes', fontsize='large')
ax.set_ylabel('Count', fontsize='large')
ax.set_title('Total number of train images for each class', fontsize='large', pad=20)
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+.03,str(round((i.get_height()), 1)), fontsize=10,color='black')
plt.show()


# In[ ]:


# Visualization of images on 3-D plot with classes

# Creating array for all 3 axes
x=[]
y=[]
z=[]

for i in range(0,train_x_vis.shape[0]):
    temp = train_x_vis[i][0][0]
    x.append(temp[0])
    y.append(temp[1])
    z.append(temp[2])

# Plotting 3-D graph

get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c = train_y,s=1, alpha=0.8,cmap="gist_ncar",marker=',')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Image data on 3-D plot with different color representing different classes', fontsize='large', pad=20)

plt.show()


# ## 3. Processing Data

# In[ ]:


# Creating copy of train and test set for cnn

train_x_cnn = np.copy(train_x)
test_x_cnn = np.copy(test_x)


# In[ ]:


# Dimensions of train and test data

print(train_x.shape,train_y.shape,test_x.shape,test_y.shape) 


# In[ ]:


# Resizing images to fit SVM and RF

train_x.resize(39209,3072)
test_x.resize(12630,3072)


# In[ ]:


# Normaliazing data for SVM and RF

train_x = preprocessing.scale(train_x)
test_x = preprocessing.scale(test_x)


# In[ ]:


# Normalizing data for CNN

train_x_min = train_x_cnn.min(axis=(0, 1), keepdims=True)
train_x_max = train_x_cnn.max(axis=(0, 1), keepdims=True)
train_x_cnn=(train_x_cnn - train_x_min)/(train_x_max - train_x_min)

test_x_min = test_x_cnn.min(axis=(0, 1), keepdims=True)
test_x_max = test_x_cnn.max(axis=(0, 1), keepdims=True)
test_x_cnn=(test_x_cnn - test_x_min)/(test_x_max - test_x_min)


# In[ ]:


# Spliting data into train and validation set for CNN

x_train, x_val, y_train, y_val = train_test_split(train_x_cnn, train_y, test_size=0.1, random_state=121)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


# In[ ]:


#Converting the class labels into categorical variables for CNN

y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)


# ## 4. Implementation of Algorithms

# In[ ]:


# Creating blank dataframe to store model scores

df_scores  = pd.DataFrame(columns = ['Model', 'Score', 'Value'])
df_model = pd.DataFrame(columns = ['Model','Accuracy (%)','Time (mins)'])


# ### 4.1 Support Vector Machine (SVM)

# In[ ]:


# Defining SVM model

svm_clf = svm.NuSVC(nu=0.05,kernel='rbf',gamma=0.00001,random_state=121)


# In[ ]:


# Fitting SVM

tic = time.perf_counter()

svm_clf.fit(train_x, train_y)

toc = time.perf_counter()
m_svm, s_svm = divmod((toc - tic), 60)
time_svm=float(str(str(int(m_svm))+"."+str(int(m_svm))))


# In[ ]:


# Predicting values for test data

y_pred_svm = svm_clf.predict(test_x)


# In[ ]:


# Calculating recall, precision, f1 score and accuracy of SVM

recall_svm = metrics.recall_score(test_y, y_pred_svm,average='macro')
df_scores.loc[len(df_scores)] = ["SVM","Recall",recall_svm]

precision_svm = metrics.precision_score(test_y, y_pred_svm,average='macro')
df_scores.loc[len(df_scores)] = ["SVM","Precision",precision_svm]

f1_svm = metrics.f1_score(test_y, y_pred_svm,average='macro')
df_scores.loc[len(df_scores)] = ["SVM","F1",f1_svm]

acc_svm=metrics.accuracy_score(test_y,y_pred_svm)
df_scores.loc[len(df_scores)] = ["SVM","Accuracy",acc_svm]

df_model.loc[len(df_model)] = ["SVM",acc_svm*100,time_svm]
acc_svm


# In[ ]:


# Classification report for SVM

print("Classification report for SVM classifier %s:\n%s\n"
      % (svm_clf, metrics.classification_report(test_y, y_pred_svm)))


# ### 4.2 Random Forest

# In[ ]:


# Creating list of number of trees

tree_list = [50,100,200,300,500]


# In[ ]:


y_pred_list=[]
time_rf_list=[]
rf_accuracy=[]
for n in tree_list:
    
    # Defining RF model with 'n' trees
    rf_clf = RandomForestClassifier(n_estimators=n, random_state=121,criterion='entropy')
    tic = time.perf_counter()
    
    # Fitting RF
    rf_clf.fit(train_x, train_y)
    toc = time.perf_counter()
    
    # Predicting values for test data
    y_pred_list.append(rf_clf.predict(test_x))
    
    # Calculating time taken
    m_rf, s_rf = divmod((toc - tic), 60)
    time_rf_list.append(float(str(str(int(m_rf))+"."+str(int(m_rf)))))
    
    # Calculating accuracy of RF
    rf_accuracy.append(metrics.accuracy_score(test_y,rf_clf.predict(test_x)))


# In[ ]:


# Plotting time and accuracy for all RF models
# Epochs vs Accuracy
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(10,7))
ax = plt.axes()
ax.plot(time_rf_list,rf_accuracy,'bo')
ax.plot(time_rf_list,rf_accuracy)

ax.set_title('Time taken by RF for n trees with Accuracy',pad=20)
ax.set_xlabel('Time')  
ax.set_ylabel('Accuracy')

for x,y,i in zip(time_rf_list,rf_accuracy,tree_list):

    label = "n= {},\n Time = {} mins,\n Accuracy = {} ".format(i,round(x,4),round(y,4))
    ax.text(x-2,y,label, fontsize=10)
plt.show()  


# In[ ]:


# Selecting best RF model

rf_clf = RandomForestClassifier(n_estimators=300, random_state=121,criterion='entropy')
y_pred_rf = y_pred_list[3]
time_rf = time_rf_list[3]

acc_rf = rf_accuracy[3]
df_scores.loc[len(df_scores)] = ["RF","Accuracy",acc_rf]

df_model.loc[len(df_model)] = ["RF",acc_rf*100,time_rf]
acc_rf


# In[ ]:


# Calculating recall, precision and f1 score for RF

recall_rf = metrics.recall_score(test_y, y_pred_rf,average='macro')
df_scores.loc[len(df_scores)] = ["RF","Recall",recall_rf]

precision_rf = metrics.precision_score(test_y, y_pred_rf,average='macro')
df_scores.loc[len(df_scores)] = ["RF","Precision",precision_rf]

f1_rf = metrics.f1_score(test_y, y_pred_rf,average='macro')
df_scores.loc[len(df_scores)] = ["RF","F1",f1_rf]


# In[ ]:


# Classification report for RF

print("Classification report for RF classifier %s:\n%s\n"
      % (rf_clf, metrics.classification_report(test_y, y_pred_rf)))


# ### 4.3 Convolution Neural Network (CNN)

# In[ ]:


# Defining CNN model

cnn_clf = Sequential()
cnn_clf.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
cnn_clf.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
cnn_clf.add(MaxPool2D(pool_size=(2, 2)))
cnn_clf.add(Dropout(rate=0.2))
cnn_clf.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
cnn_clf.add(MaxPool2D(pool_size=(2, 2)))
cnn_clf.add(Dropout(rate=0.2))
cnn_clf.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu'))
cnn_clf.add(MaxPool2D(pool_size=(2, 2)))
cnn_clf.add(Dropout(rate=0.2))
cnn_clf.add(Flatten())
cnn_clf.add(Dense(256, activation='relu'))
cnn_clf.add(Dropout(rate=0.2))
cnn_clf.add(Dense(128, activation='relu'))
cnn_clf.add(Dropout(rate=0.2))
cnn_clf.add(Dense(43, activation='softmax'))


# In[ ]:


# Compilation of the model

cnn_clf.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[ ]:


# Fitting CNN

epochs = 10
tic = time.perf_counter()

cnn_fit = cnn_clf.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_val, y_val))

toc = time.perf_counter()

m_cnn, s_cnn = divmod((toc - tic), 60)
time_cnn=float(str(str(int(m_cnn))+"."+str(int(m_cnn))))


# In[ ]:


# Epochs vs Accuracy
fig = plt.figure(figsize=(7,4))
ax = plt.axes()
ep=list(range(1, 11))
ax.plot(ep,cnn_fit.history['accuracy'],label="Train Accuracy");
ax.plot(ep,cnn_fit.history['val_accuracy'],label="Validation Accuracy")
ax.set_title('Train, Validation Accuracy')
ax.set_xlabel('Epochs')  
ax.set_ylabel('Accuracy')
ax.legend()
plt.show()  


# In[ ]:


# Epochs vs Loss
fig = plt.figure(figsize=(7,4))
ax = plt.axes()
ep=list(range(1, 11))
ax.plot(ep,cnn_fit.history['loss'],label="Train Loss");
ax.plot(ep,cnn_fit.history['val_loss'],label="Validation Loss")
ax.set_title('Train, Validation Loss')
ax.set_xlabel('Epochs')  
ax.set_ylabel('Loss')
ax.legend()
plt.show() 


# In[ ]:


# Predicting values for test data

y_pred_cnn = cnn_clf.predict_classes(test_x_cnn)


# In[ ]:


# Calculating and storing recall, precision, f1 score and accuracy of CNN

recall_cnn = metrics.recall_score(test_y, y_pred_cnn,average='macro')
df_scores.loc[len(df_scores)] = ["CNN","Recall",recall_cnn]

precision_cnn = metrics.precision_score(test_y, y_pred_cnn,average='macro')
df_scores.loc[len(df_scores)] = ["CNN","Precision",precision_cnn]

f1_cnn = metrics.f1_score(test_y, y_pred_cnn,average='macro')
df_scores.loc[len(df_scores)] = ["CNN","F1",f1_cnn]

acc_cnn=metrics.accuracy_score(test_y,y_pred_cnn)
df_scores.loc[len(df_scores)] = ["CNN","Accuracy",acc_cnn]

df_model.loc[len(df_model)] = ["CNN",acc_cnn*100,time_cnn]
acc_cnn


# In[ ]:


# Classification report for CNN

print("Classification report for CNN classifier %s:\n%s\n"
      % (cnn_clf, metrics.classification_report(test_y, y_pred_cnn)))


# In[ ]:


df_model


# In[ ]:



fig = plt.figure(figsize=(11,8))
ax = plt.axes()
ax = sns.barplot(x="Model", y="Value", hue="Score", data=df_scores, palette="ch:.25",edgecolor="1")
ax.set_title('Comparision of SVM, RF, CNN for Traffic Signs')

for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+.02,str(round((i.get_height()), 4)), fontsize=10,color='dimgrey')
plt.show() 


# In[ ]:


# Saving CNN model

cnn_clf.save('cnn_classifier.h5')

