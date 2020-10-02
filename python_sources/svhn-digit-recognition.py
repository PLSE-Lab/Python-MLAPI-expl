#!/usr/bin/env python
# coding: utf-8

# <h1 class='font-effect-3d' style='font-family:Akronim; color:#ff55ee'> Modules, Helpful Functions, & Styling</h1>

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Akronim|Roboto&effect=3d');\nspan {color:black; text-shadow:3px 3px 3px #aaa;}\ndiv.output_prompt {color:darkblue;} \ndiv.input_prompt {color:#ff55ee;} \ndiv.output_area pre,div.output_subarea,div.output_stderr pre  \n      {background-color:ghostwhite; font-size:15px; color:darkblue;}\n</style>")


# In[ ]:


import warnings; warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,pylab as pl
import glob
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.optimizers import SGD,RMSprop,Adam,Nadam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense,Dropout,LSTM
from keras.layers import Activation,Flatten,Input,BatchNormalization
from keras.layers import Conv1D,MaxPooling1D,Conv2D,MaxPooling2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D


# In[ ]:


path='../input/svhn-preproccessed-fragments/'
fw='weights.housenumbers.hdf5'
glob.glob(path+'*')


# <h1 class='font-effect-3d' style='font-family:Akronim; color:#ff55ee'> Data Loading & Preprocessing</h1>

# In[ ]:


train_images=pd.read_csv(path+'train_images.csv')
train_labels=pd.read_csv(path+'train_labels.csv')
test_images=pd.read_csv(path+'test_images.csv')
test_labels=pd.read_csv(path+'test_labels.csv')
extra_images=pd.read_csv(path+'extra_images.csv')
extra_labels=pd.read_csv(path+'extra_labels.csv')
train_images=train_images.iloc[:,1:].as_matrix().astype('float32').reshape(-1,32,32,1)
train_labels=train_labels.iloc[:,1:].as_matrix().astype('int16')
test_images=test_images.iloc[:,1:].as_matrix().astype('float32').reshape(-1,32,32,1)
test_labels=test_labels.iloc[:,1:].as_matrix().astype('int16')
extra_images=extra_images.iloc[:,1:].as_matrix().astype('float32').reshape(-1,32,32,1)
extra_labels=extra_labels.iloc[:,1:].as_matrix().astype('int16')
ctrain_labels=to_categorical(train_labels,num_classes=11).astype('int16')
ctest_labels=to_categorical(test_labels,num_classes=11).astype('int16')
cextra_labels=to_categorical(extra_labels,num_classes=11).astype('int16')
n=np.random.randint(1,2000,1)[0]
print('Label: ',train_labels[n])
print(ctrain_labels[n])
pl.imshow(train_images[n].reshape(32,32),
          cmap=pl.cm.bone);


# In[ ]:


X=np.concatenate((train_images,
                        test_images),axis=0)
X=np.concatenate((X,extra_images),axis=0)
y=np.concatenate((ctrain_labels,
                  ctest_labels),axis=0)
y=np.concatenate((y,cextra_labels),axis=0)
def tts(X,y): 
    x_train,x_test,y_train,y_test=    train_test_split(X,y,test_size=.2,random_state=1)
    n=int(len(x_test)/2)
    x_valid,y_valid=x_test[:n],y_test[:n]
    x_test,y_test=x_test[n:],y_test[n:]
    return x_train,x_valid,x_test,y_train,y_valid,y_test
x_train,x_valid,x_test,y_train,y_valid,y_test=tts(X,y)
y_train_list=[y_train[:,i] for i in range(5)]
y_test_list=[y_test[:,i] for i in range(5)]
y_valid_list=[y_valid[:,i] for i in range(5)]
for el in [x_train,x_valid,x_test,
           y_train,y_valid,y_test]:
    print(el.shape)


# <h1 class='font-effect-3d' style='font-family:Akronim; color:#ff55ee'>Build the Model</h1>

# In[ ]:


def cnn_model():    
    model_input=Input(shape=(32,32,1))
    x=BatchNormalization()(model_input)        
    x=Conv2D(32,(3,3),activation='relu',
             padding='same')(model_input)
    x=MaxPooling2D(pool_size=(2,2))(x)     
    x=Conv2D(32,(3,3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)    
    x=Dropout(.25)(x)    
    x=Conv2D(64,(3,3),activation='relu')(x)       
    x=Conv2D(64,(3,3),activation='relu')(x)    
    x=Dropout(.25)(x)    
    x=Conv2D(196,(3,3),activation='relu')(x)    
    x=Dropout(.25)(x)              
    x=Flatten()(x)    
    x=Dense(512,activation='relu')(x)    
    x=Dropout(.5)(x)    
    y=[Dense(11,activation='softmax')(x)
       for i in range(5)]    
    model=Model(input=model_input,output=y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
cnn_model=cnn_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,
                             save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=2,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=16,verbose=2)
history=cnn_model.fit(x_train,y_train_list,
                      validation_data=(x_valid,y_valid_list), 
                      epochs=200,batch_size=128,verbose=2, 
                      callbacks=[checkpointer,lr_reduction,estopping])


# In[ ]:


cnn_model.load_weights(fw)
cnn_scores=cnn_model.evaluate(x_test,y_test_list,verbose=0)
print("CNN. Scores: \n" ,(cnn_scores))
print("First digit. Accuracy: %.2f%%"%(cnn_scores[6]*100))
print("Second digit. Accuracy: %.2f%%"%(cnn_scores[7]*100))
print("Third digit. Accuracy: %.2f%%"%(cnn_scores[8]*100))
print("Fourth digit. Accuracy: %.2f%%"%(cnn_scores[9]*100))
print("Fifth digit. Accuracy: %.2f%%"%(cnn_scores[10]*100))
avg_accuracy=sum([cnn_scores[i] for i in range(6,11)])/5
print("CNN Model. Average Accuracy: %.2f%%"%(avg_accuracy*100))


# In[ ]:


pl.figure(figsize=(11,5)); k=10
keys=list(history.history.keys())[17:]
pl.plot(history.history[keys[0]][k:],label='First digit')
pl.plot(history.history[keys[1]][k:],label='Second digit')
pl.plot(history.history[keys[2]][k:],label='Third digit')
pl.plot(history.history[keys[3]][k:],label='Fourth digit')
pl.plot(history.history[keys[4]][k:],label='Fifth digit')
pl.legend(); pl.title('Accuracy');

