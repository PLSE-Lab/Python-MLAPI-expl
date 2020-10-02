#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# **COVID**

# In[ ]:


img_size = (224,224)
dir_name = 'input/covid19'
img_list = glob.glob('../' + dir_name + '/*')

list_covid = []
for img in img_list:
    temp_img = load_img(img,grayscale=True,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    list_covid.append(temp_img_array)
list_covid = np.array(list_covid)
list_covid2 = list_covid.reshape(-1,50176)
df_covid=pd.DataFrame(list_covid2)
df_covid['label'] = np.full(df_covid.shape[0],2)


# In[ ]:


df_covid.shape


# NORMAL

# In[ ]:


img_size = (224,224)
dir_name2 = 'input/chest-xray-images-pneumonia-and-covid19/Chest_xray_image_dataset_covid_19_and_others/NORMAL'
img_list2 = glob.glob('../' + dir_name2 + '/*')

list_normal = []
for img in img_list2[:150]:
    temp_img = load_img(img,grayscale=True,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    list_normal.append(temp_img_array)
list_normal = np.array(list_normal)
list_normal2 = list_normal.reshape(-1,50176)
df_normal=pd.DataFrame(list_normal2)
df_normal['label'] = np.full(df_normal.shape[0],0)


# In[ ]:


df_normal.shape


# other PNEUMONIA

# In[ ]:


img_size = (224,224)
dir_name3 = 'input/chest-xray-images-pneumonia-and-covid19/Chest_xray_image_dataset_covid_19_and_others/PNEUMONIA'
img_list3 = glob.glob('../' + dir_name3 + '/*')

list_others = []
for img in img_list3[:150]:
    temp_img = load_img(img,grayscale=True,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    list_others.append(temp_img_array)
list_others = np.array(list_others)
list_others2 = list_others.reshape(-1,50176)
df_others=pd.DataFrame(list_others2)
df_others['label'] = np.full(df_others.shape[0],1)


# In[ ]:


df_others.shape


# show x-ray

# In[ ]:


f = plt.figure(figsize=(15,7))
f.suptitle('COVID19',fontsize=20)
f.subplots_adjust(top=2.35)
for i in range(3):
    sp = f.add_subplot(1,3,i+1)
    img = cv2.imread(img_list[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    plt.imshow(img_gray)
    plt.gray()
plt.show()


# In[ ]:


f = plt.figure(figsize=(15,7))
f.suptitle('Other pneumonia',fontsize=20)
f.subplots_adjust(top=2.25)
for i in range(3):
    sp = f.add_subplot(1,3,i+1)
    img = cv2.imread(img_list3[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    plt.imshow(img_gray)
plt.gray()
plt.show()


# In[ ]:


f = plt.figure(figsize=(15,7))
f.suptitle('Normal',fontsize=20)
f.subplots_adjust(top=2.4)
for i in range(3):
    sp = f.add_subplot(1,3,i+1)
    img = cv2.imread(img_list2[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    plt.imshow(img_gray)
plt.gray()
plt.show()


# making Database

# In[ ]:


Df = pd.concat([df_covid, df_normal , df_others], ignore_index=True)


# train-validation split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(Df.iloc[:,0:-1], Df.iloc[:,-1], test_size=0.20, random_state=0)

X_train = x_train.values.reshape(-1,224,224,1)
X_test = x_test.values.reshape(-1,224,224,1)
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)


# CNN

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


np.random.seed(0)
model = Sequential()

model.add(BatchNormalization(input_shape=(224,224,1)))
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model_chkpt = ModelCheckpoint('best_mod.h5', save_best_only=True, monitor='accuracy')
early_stopping = EarlyStopping(monitor='loss', restore_best_weights=False, patience=10)


# In[ ]:


history = model.fit(X_train, Y_train, 
          validation_split=0.20,
          epochs=10, batch_size=16, shuffle=True, 
          callbacks=[model_chkpt ,early_stopping]
         )


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(12, 3))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


plt.figure()

ax = plt.subplot()

ax.set_title('Confusion Matrix')
pred = model.predict_classes(X_test)
Y_TEST = np.argmax(Y_test, axis =1)
cm = metrics.confusion_matrix(Y_TEST,pred)
classes=['normal', 'other pneumonia', 'covid19']
sns.heatmap(cm, annot=True,xticklabels=classes, yticklabels=classes,cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show


# classification report

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_TEST, pred))
print('normal = 0 , other pneumonia = 1, covid19 = 2')


# multiclass ROC

# In[ ]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

PRED = to_categorical(pred)
y = Df['label'].values
# Binarize the output
y = label_binarize(y, classes=[0,1,2])
n_classes = y.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
       fpr[i], tpr[i], _ = roc_curve(Y_test[:,i], PRED[:,i])
       roc_auc[i] = auc(fpr[i], tpr[i])


# In[ ]:


colors = ['blue', 'red', 'green']
cls = {0:'normal', 1:'other pneumonia', 2:'covid'}
for i, color ,c in zip(range(n_classes), colors, cls.values()):
    plt.plot(fpr[i], tpr[i], color=color, lw=0.5,
             label='ROC curve of '+c+ '(AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--',linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for multi-class data')
plt.legend(loc="lower right")
plt.show()


# using eigenfaces

# In[ ]:


from sklearn.decomposition import PCA
from time import time

n_components = 40
n_samples=411
h=224
w =224

print("Extracting the top %d eigenfaces from %d cases"
      % (n_components, x_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(x_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(x_train)
X_test_pca = pca.transform(x_test)
print("done in %0.3fs" % (time() - t0))


# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1, 10, 100,1e3, 5e3, 1e4],
              'gamma': [0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# In[ ]:


t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

target_names = ['normal','other pneumonia', 'covid']
n_classes=3
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# In[ ]:


fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(224, 224), cmap='bone')


# In[ ]:


from skimage.io import imshow
loadeigen = eigenfaces[0]
imshow(loadeigen) 


# In[ ]:


f = plt.figure(figsize=(15,4))
f.suptitle('eigenfaces',fontsize=20)
f.subplots_adjust(top=1)
for i in range(5):
    sp = f.add_subplot(1,5,i+1)
    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    loadeigen = eigenfaces[i]
    imshow(loadeigen) 
    sp.title.set_text('PCA'+str(i+1))
plt.show()


# In[ ]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# PCA visualize

# In[ ]:


plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)


# In[ ]:


pca_2D = PCA(n_components=2).fit_transform(x_train)


# In[ ]:


pca_2D


# In[ ]:


plt.scatter(pca_2D[:,0],pca_2D[:,1], marker=".", c=y_train, cmap='jet')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# In[ ]:


mycolors=["r","b","g"]
labelTups = ['normal','other','COVID']
label=y_train
for i,mycolor in enumerate(mycolors):
        plt.scatter(pca_2D[label == i, 0],
                    pca_2D[label == i, 1], color=mycolor)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(labelTups, loc='upper right')
plt.title('PCA(n_component=2)')
plt.show()


# class activation maps

# In[ ]:


import pandas as pd
import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model

K.set_learning_phase(1) #set learning phase

def Grad_Cam(input_model, x, layer_name):
    X = np.expand_dims(x, axis=0)
    X = X.astype('float32')
    preprocessed_input = X / 255.0

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    conv_output = model.get_layer(layer_name).output   
    grads = K.gradients(class_output, conv_output)[0]  
    gradient_function = K.function([model.input], [conv_output, grads])  

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR) 
    cam = np.maximum(cam, 0) 
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  
    jetcam = (np.float32(jetcam) + x / 2)   

    return jetcam


# In[ ]:


x = img_to_array(load_img(img_list[0],grayscale=True, target_size=(224,224)))
array_to_img(x)


# In[ ]:


image = Grad_Cam(model, x, 'conv2d') 
array_to_img(image)


# pixel importance

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier(n_estimators=1000)
estimator.fit(x_train, y_train)


# In[ ]:


importances = estimator.feature_importances_
importances = importances.reshape(224,224)


# In[ ]:


plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances")
plt.tick_params(length=0)
plt.xticks(color="None")
plt.yticks(color="None")
plt.show()


# In[ ]:




