#!/usr/bin/env python
# coding: utf-8

# In this notbook I will use Nearest Neighber Classification using Triplet loss training (FaceNet: A Unified Embedding for Face Recognition and Clustering), in five steps:
# 1. Loading first sub-set of data-base (Whales having at least 29 images per whale) and train cross entropy CNN
# 2. Optimize the first sub-set using triplet loss to extract 128 features
# 3. Load secound sub-set (Whales with 18-28  images per whale) , split it to train and test.
# 4. Predict the train set features
# 5. Classification of test set using Nearest neighbor features to the train set.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import keras.backend as K
from keras.layers import Input,Conv2D,MaxPool2D,Dense,Dropout,Flatten,BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.preprocessing import image
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Labels = pd.read_csv('../input/train.csv')
Labels.head()


# In[ ]:


Pics = os.listdir('../input/train/')


# In[ ]:


SIZE = 128
def ImportImage( filename):
    img = Image.open('../input/train/'+filename).convert("LA").resize( (SIZE,SIZE))
    return np.array(img)[:,:,0]


# In[ ]:


# load pictares and label them , "MinPicsPerUser" is the number of pictures requers for class to load.
# At the begginig for easy training and convargse I will load only those with high number of pics

def LoadImage_And_MatchLabels(Pics,Labels,Unique_Labels,MinPicsPerUser,MaxPicPerUser = 1000,SIZE= 128):
    ManyImageIndex = np.array(Unique_Labels['Count']
                              [Unique_Labels.index[ Unique_Labels['Count'] > MinPicsPerUser ]].tolist())
    ManyImageIndex_Sum = np.sum(ManyImageIndex[ManyImageIndex<MaxPicPerUser]) 

    Train_img_Array = np.zeros((ManyImageIndex_Sum,SIZE,SIZE))
    PicInd = 0 
    ImageLabel =  [] 
    for Pic in  Pics : 
        #print(Pic,PicInd)
        ID = Labels['Id'][Labels.index[Labels['Image']==Pic ].tolist()].tolist()[0]
        NumImages = Unique_Labels['Count'][Unique_Labels.index[Unique_Labels['Id']== ID].tolist()].tolist()
        if (NumImages[0] > MinPicsPerUser and NumImages[0] < MaxPicPerUser) : 
            Train_img_Array[PicInd,:,:] = ImportImage(Pic)
            PicInd += 1
            ImageLabel.append(ID)
    return Train_img_Array,ImageLabel


# In[ ]:


# load only clasess with >29 images per class 

MinPic = 20 
Unique_Labels = Labels.drop_duplicates(subset='Id').reset_index()
Unique_Labels['Count'] = 0
SumImages = np.zeros(Unique_Labels.shape[0])
for i in range(Unique_Labels.shape[0]):
    SumImages[i] = np.sum(Labels['Id']== Unique_Labels['Id'][i])
    Unique_Labels['Count'][i] = SumImages[i]
SumImages_Sort = np.sort(SumImages)
Train_Phase_1_Array,ImageLabel = LoadImage_And_MatchLabels(Pics,Labels,Unique_Labels,MinPic,MaxPicPerUser= 19,SIZE=SIZE)             


# In[ ]:


# Helper function to Categorical classes
def List_To_Categorical(Label_List):
    LabelsArray = np.zeros(len(Label_List))
    for j,label in enumerate(set(Label_List)):
        inds = [i for i,e in enumerate(Label_List) if (e == label)]
        for i in inds: 
            LabelsArray[i] = j
    LabelsCategorical = to_categorical(LabelsArray)
    print(LabelsCategorical.shape)
    return LabelsCategorical,LabelsArray


# In[ ]:


CtegoricalLabel,LabelsArray = List_To_Categorical(ImageLabel)


# In[ ]:



inputs1 = Input((SIZE,SIZE,1))
C1 = Conv2D(32,kernel_size=(3,3),activation='relu',padding='SAME')(inputs1)
C1 = BatchNormalization()(C1)
C1 = MaxPool2D(pool_size=(2,2))(C1)
C2 = Conv2D(32,kernel_size=(3,3),activation='relu',padding='SAME')(C1)
C2 = BatchNormalization()(C2)
C2 = MaxPool2D(pool_size=(2,2))(C2)
C3 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='SAME')(C2)
C3 = BatchNormalization()(C3)
C3 = MaxPool2D(pool_size=(2,2))(C3)
C4 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='SAME')(C3)
C4 = MaxPool2D(pool_size=(2,2))(C4)
C5 = Flatten()(C4)
Danse1 = Dense(128,activation='relu')(C5)
Danse1 = Dropout(0.5)(Danse1)
Danse2 = Dense(128)(Danse1)
#Danse2d = Dropout(0.5)(Danse2)
Dense3 = Dense(CtegoricalLabel.shape[1],activation='softmax')(Danse2)


# In[ ]:


model = Model(inputs1,Dense3)
model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(x=Train_Phase_1_Array.reshape([-1,SIZE,SIZE,1]),y=CtegoricalLabel,batch_size=32,epochs=50,verbose=1,
          validation_split=0.15)


# Define a Triplet network, with the wightes from previse CNN model:

# In[ ]:



Triplet_model = Sequential()
for layer in model.layers[:-2]:
    Triplet_model.add(layer)
    Triplet_model
Triplet_model.summary()


# In[ ]:


# My Triplet loss, and sorting the data according to the loss definition
def TripletLoss_3(yTrue,y_pred):
    y_pred = K.l2_normalize(y_pred,axis=0)
    yTrue= K.l2_normalize(yTrue,axis=0)
    PosDiff = K.sqrt(K.mean(K.square(y_pred-yTrue[:,:128])))
    NegDiff = K.sqrt(K.mean(K.square(y_pred-yTrue[:,128:256])))
    Dist_Pos_Neg =   - (NegDiff) + (PosDiff)
    #loss = K.maximum(0.0,Dist_Pos_Neg)
    loss = K.log(1 + K.exp(Dist_Pos_Neg))
    return loss


def SortFeatures(ImArray,Features,LabelsArray):
    FeaturesOut = np.zeros((Features.shape[0],Features.shape[1]*2))
    for j in range(ImArray.shape[0]) :
        Ind_Same_Whale = np.array([i for i,e in enumerate(LabelsArray) if (e == LabelsArray[j]) & (j != i)])
        Ind_different_Whale = np.array([i for i,e in enumerate(LabelsArray) if (e != LabelsArray[j]) & (j != i)])
        PosInd = np.random.choice(Ind_Same_Whale)
        NegInd = np.random.choice(Ind_different_Whale)

        FeaturesOut[j,:128] = Features[PosInd,:]
        FeaturesOut[j,128:256] = Features[NegInd,:]
    return FeaturesOut


# In[ ]:


# Optmize the Nearest neighber classification using the Triplet loss: 
CtegoricalLabel = List_To_Categorical(ImageLabel)
Triplet_model.compile(optimizer='adam',loss=TripletLoss_3)
Pred = Triplet_model.predict(Train_Phase_1_Array.reshape([-1,SIZE,SIZE,1]))


# In[ ]:


Pred = Triplet_model.predict(Train_Phase_1_Array.reshape([-1,SIZE,SIZE,1]))
SortedPred = SortFeatures(Train_Phase_1_Array,Pred,ImageLabel)
Triplet_model.fit(x=Train_Phase_1_Array.reshape([-1,SIZE,SIZE,1]),y=SortedPred,batch_size=32,epochs=3,verbose=1)


# In[ ]:


# load other set of Classes for test , those with Number of Pics >16 <28
MinPicsPerUser = 10
MaxPic = 19
Train_Phase_2_Array,ImageLabel_2 = LoadImage_And_MatchLabels(Pics,Labels,Unique_Labels,MinPicsPerUser,MaxPicPerUser=MaxPic,SIZE=SIZE)             


# In[ ]:



# Extract features
Pred_Features = Triplet_model.predict(Train_Phase_2_Array.reshape([-1,SIZE,SIZE,1]))
LabelsCategorical_2,LabelsArray_2 = List_To_Categorical(ImageLabel_2)

#  train and test split
X_train, X_test, y_train, y_test = train_test_split(Pred_Features, LabelsArray_2, test_size=0.25, random_state=42)

# L2 normlize 
X_train = X_train/np.linalg.norm(X_train, ord=2,axis=1).reshape(X_train.shape[0],1)
X_test = X_test/np.linalg.norm(X_test, ord=2,axis=1).reshape(X_test.shape[0],1)

# Nearest neighbor distance calculation
Dist = np.zeros((len(y_test),len(np.unique(y_train))))
for i in range(X_test.shape[0]):
    DiffMat = np.sum(np.square(np.subtract(X_train,X_test[i,:])),axis=1)
    for j in range(len(np.unique(y_train))):
        Dist[i,j] = np.sum(DiffMat[np.where(j==y_train)])
        


# In[ ]:


# top 5 nearest neighbor classification 
SumTop5 = 0 
for k  in range(len(Dist)):
    if np.sum(np.argsort(Dist[k,:])[:5]== y_test[k]) == 1 : 
        SumTop5 += 1
        
PercentCorrect_top5 = SumTop5/len(Dist) 
print(PercentCorrect_top5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




