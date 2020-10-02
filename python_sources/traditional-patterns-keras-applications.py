#!/usr/bin/env python
# coding: utf-8

# <h1 class="font-effect-3d-float" style="color:firebrick; font-family:'Monoton'; ">Code Modules, Helpful Functions, Styling, and Links</h1>
# 
# [GitHub Repository](https://github.com/OlgaBelitskaya/deep_learning_projects/tree/master/DL_PP5) & [Colaboratory Version](https://colab.research.google.com/drive/1Tt3qZePsf2P6kNNao-hQ58DlG71Abj5a)

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style> \n@import url('https://fonts.googleapis.com/css?family=Monoton|Roboto&effect=3d-float|');\nbody {background-color:#f7e8e8;} \na,h4 {color:crimson; font-family:Roboto;}\nspan {color:black; text-shadow:4px 4px 4px #aaa;}\ndiv.output_prompt {color:crimson;} \ndiv.input_prompt {color:firebrick;} \ndiv.output_area pre,div.output_subarea {font-size:15px; color:crimson}\ndiv.output_stderr pre {background-color:#f7e8e8;}\n</style>")


# In[ ]:


import warnings; warnings.filterwarnings('ignore')
import scipy,h5py,pandas as pd,numpy as np,pylab as pl
import seaborn as sn,keras as ks,tensorflow as tf
from scipy import misc
from skimage.transform import resize
from skimage import color,measure
from IPython.core.magic import register_line_magic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
np.set_printoptions(precision=6)
pl.style.use('seaborn-whitegrid')
fpath='../input/traditional-decor-patterns/'
fw='weights.decor.hdf5'
n=np.random.choice(484,size=6,replace=False)
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential,load_model,Model
from keras.layers import Input,Activation,Dense,LSTM
from keras.layers import Flatten,Dropout,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D,GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.applications.resnet50 import ResNet50,preprocess_input as rn50pi
from keras.applications.inception_v3 import InceptionV3,preprocess_input as iv3pi
from keras.applications.xception import Xception,preprocess_input as xpi
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input as iv2pi
from keras import __version__
print('keras version:', __version__)
print('tensorflow version:',tf.__version__)


# In[ ]:


def ohe(x): 
    return OneHotEncoder(n_values='auto')           .fit(x.reshape(-1,1))           .transform(x.reshape(-1,1))           .toarray().astype('int64')
def tts(X,y): 
    x_train,x_test,y_train,y_test=    train_test_split(X,y,test_size=.2,random_state=1)
    n=int(len(x_test)/2)
    x_valid,y_valid=x_test[:n],y_test[:n]
    x_test,y_test=x_test[n:],y_test[n:]
    return x_train,x_valid,x_test,y_train,y_valid,y_test
def history_plot(fit_history):
    pl.figure(figsize=(12,10)); pl.subplot(211)
    keys=list(fit_history.history.keys())[0:4]
    pl.plot(fit_history.history[keys[0]],
            color='crimson',label='train')
    pl.plot(fit_history.history[keys[2]],
            color='firebrick',label='valid')
    pl.xlabel("Epochs"); pl.ylabel("Loss")
    pl.legend(); pl.grid()
    pl.title('Loss Function')     
    pl.subplot(212)
    pl.plot(fit_history.history[keys[1]],
            color='crimson',label='train')
    pl.plot(fit_history.history[keys[3]],
            color='firebrick',label='valid')
    pl.xlabel("Epochs"); pl.ylabel("Accuracy")    
    pl.legend(); pl.grid()
    pl.title('Accuracy'); pl.show()


# <h1 class="font-effect-3d-float" style="color:firebrick; font-family:'Monoton'; ">Data Preprocessing</h1>

# In[ ]:


data=pd.read_csv(fpath+'decor.csv')
f=h5py.File(fpath+'DecorColorImages.h5','r')
keys=list(f.keys())
[countries,decors,images,types]=[np.array(f[keys[i]]) for i in range(4)]
sh=[el.shape for el in [countries,decors,images,types]]
data.loc[n]


# In[ ]:


images=images/255
gray_images=np.dot(images[...,:3],[.299,.587,.114])
fig=pl.figure(figsize=(12,5))
for i,idx in enumerate(n):
    ax=fig.add_subplot(2,3,i+1,xticks=[],yticks=[])
    ax.imshow(gray_images[idx])
    ax.set_title(data['country'][idx]+'; '+                 data['decor'][idx]+'; '+data['type'][idx])


# In[ ]:


ccountries,cdecors,ctypes=ohe(countries),ohe(decors),ohe(types)
ctargets=np.concatenate((ccountries,cdecors),axis=1)
ctargets=np.concatenate((ctargets,ctypes),axis=1)
pd.DataFrame([images.shape,gray_images.shape,
              ccountries.shape,cdecors.shape,
              ctypes.shape,ctargets.shape])


# In[ ]:


# spliting the data 
# Color Images / Countries 
x_train1,x_valid1,x_test1,y_train1,y_valid1,y_test1=tts(images,ccountries)
# Grayscaled Images / Countries 
x_train2,x_valid2,x_test2,y_train2,y_valid2,y_test2=tts(gray_images,ccountries)
# Color Images / Decors 
x_train3,x_valid3,x_test3,y_train3,y_valid3,y_test3=tts(images,cdecors)
# Grayscaled Images / Decors 
x_train4,x_valid4,x_test4,y_train4,y_valid4,y_test4=tts(gray_images,cdecors)
# Color Images / Multi-Label Targets
x_train5,x_valid5,x_test5,y_train5,y_valid5,y_test5=tts(images,ctargets)
# Grayscaled Images / Multi-Label Targets 
x_train6,x_valid6,x_test6,y_train6,y_valid6,y_test6=tts(gray_images,ctargets)
sh=[el.shape for el in [x_train1,y_train1,x_valid1,y_valid1,x_test1,y_test1,
 x_train3,y_train3,x_valid3,y_valid3,x_test3,y_test3,
 x_train5,y_train5,x_valid5,y_valid5,x_test5,y_test5,
 x_train2,y_train2,x_valid2,y_valid2,x_test2,y_test2,
 x_train4,y_train4,x_valid4,y_valid4,x_test4,y_test4,
 x_train6,y_train6,x_valid6,y_valid6,x_test6,y_test6]]
pd.DataFrame(sh)


# In[ ]:


y_train5_list=[y_train5[:,:4],y_train5[:,4:11],y_train5[:,11:]]
y_test5_list=[y_test5[:,:4],y_test5[:,4:11],y_test5[:,11:]]
y_valid5_list=[y_valid5[:,:4],y_valid5[:,4:11],y_valid5[:,11:]]
y_train6_list=[y_train6[:,:4],y_train6[:,4:11],y_train6[:,11:]]
y_test6_list=[y_test6[:,:4],y_test6[:,4:11],y_test6[:,11:]]
y_valid6_list=[y_valid6[:,:4],y_valid6[:,4:11],y_valid6[:,11:]]


# <h1 class="font-effect-3d-float" style="color:firebrick; font-family:'Monoton'; ">Classification Models</h1>
# ## ResNet50

# In[ ]:


# creating bottleneck features
resize_x_train3=np.array([scipy.misc.imresize(x_train3[i],(224,224,3)) 
                          for i in range(0,len(x_train3))]).astype('float32')
resize_x_valid3=np.array([scipy.misc.imresize(x_valid3[i],(224,224,3)) 
                          for i in range(0,len(x_valid3))]).astype('float32')
resize_x_test3=np.array([scipy.misc.imresize(x_test3[i],(224,224,3)) 
                          for i in range(0,len(x_test3))]).astype('float32')
x_train_bn3=rn50pi(resize_x_train3)
x_valid_bn3=rn50pi(resize_x_valid3)
x_test_bn3=rn50pi(resize_x_test3)
fn = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet50base_model=ResNet50(weights=fn,include_top=False)
x_train_bn3=resnet50base_model.predict(x_train_bn3)
x_valid_bn3=resnet50base_model.predict(x_valid_bn3)
x_test_bn3=resnet50base_model.predict(x_test_bn3)


# In[ ]:


sh=x_train_bn3.shape[1:]
def resnet50_model():
    model=Sequential()
    model.add(GlobalAveragePooling2D(input_shape=sh))    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))        
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))
    model.add(Dense(7, activation='softmax'))   
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
resnet50_model=resnet50_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=2,factor=.5)
estopping=EarlyStopping(monitor='val_loss',patience=16,verbose=2)
history=resnet50_model.fit(x_train_bn3,y_train3,
                   validation_data=(x_valid_bn3,y_valid3),
                   epochs=100,batch_size=128,verbose=2,
                   callbacks=[checkpointer,lr_reduction,estopping]);


# In[ ]:


history_plot(history)
resnet50_model.load_weights(fw)
resnet50_scores=resnet50_model.evaluate(x_test_bn3,y_test3)
print("Accuracy: %.2f%%"%(resnet50_scores[1]*100))
resnet50_scores


# ## Inception V3

# In[ ]:


resize_x_train1=np.array([scipy.misc.imresize(x_train1[i],(224,224,3)) 
                          for i in range(0,len(x_train1))]).astype('float32')
resize_x_valid1=np.array([scipy.misc.imresize(x_valid1[i],(224,224,3)) 
                          for i in range(0,len(x_valid1))]).astype('float32')
resize_x_test1=np.array([scipy.misc.imresize(x_test1[i],(224,224,3)) 
                         for i in range(0,len(x_test1))]).astype('float32')
x_train_bn1=iv3pi(resize_x_train1)
x_valid_bn1=iv3pi(resize_x_valid1)
x_test_bn1=iv3pi(resize_x_test1)
fn='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
inceptionv3base_model=InceptionV3(weights=fn,include_top=False)
x_train_bn1=inceptionv3base_model.predict(x_train_bn1)
x_valid_bn1=inceptionv3base_model.predict(x_valid_bn1)
x_test_bn1=inceptionv3base_model.predict(x_test_bn1)


# In[ ]:


sh=x_train_bn1.shape[1:]
def inception_v3_model():
    model=Sequential()    
    model.add(GlobalAveragePooling2D(input_shape=sh))    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.25))       
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.25))
    model.add(Dense(4,activation='softmax'))    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
inception_v3_model=inception_v3_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,
                               verbose=2,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=16,verbose=2)
history=inception_v3_model.fit(x_train_bn1,y_train1,
                               validation_data=(x_valid_bn1,y_valid1),
                               epochs=100,batch_size=128,verbose=2,
                               callbacks=[checkpointer,lr_reduction,estopping]);


# In[ ]:


history_plot(history)
inception_v3_model.load_weights(fw)
inception_v3_scores=inception_v3_model.evaluate(x_test_bn1,y_test1)
print("Accuracy: %.2f%%" % (inception_v3_scores[1]*100))
inception_v3_scores


# ## Xception

# In[ ]:


resize_x_train3=np.array([scipy.misc.imresize(x_train3[i],(71,71,3)) 
                         for i in range(0,len(x_train3))]).astype('float32')
resize_x_valid3=np.array([scipy.misc.imresize(x_valid3[i],(71,71,3)) 
                           for i in range(0,len(x_valid3))]).astype('float32')
resize_x_test3=np.array([scipy.misc.imresize(x_test3[i],(71,71,3)) 
                         for i in range(0,len(x_test3))]).astype('float32')
x_train_bn3=xpi(resize_x_train3)
x_valid_bn3=xpi(resize_x_valid3)
x_test_bn3=xpi(resize_x_test3)
fn='../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
xceptionbase_model=Xception(weights=fn,include_top=False)
x_train_bn3=xceptionbase_model.predict(x_train_bn3)
x_valid_bn3=xceptionbase_model.predict(x_valid_bn3)
x_test_bn3=xceptionbase_model.predict(x_test_bn3)


# In[ ]:


sh=x_train_bn3.shape[1:]
def xception_model():
    model=Sequential()    
    model.add(GlobalAveragePooling2D(input_shape=sh))   
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))        
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))
    model.add(Dense(7, activation='softmax'))    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
xception_model=xception_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,
                               verbose=2,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=16,verbose=2)
history=xception_model.fit(x_train_bn3,y_train3,
                           validation_data=(x_valid_bn3,y_valid3),
                           epochs=100,batch_size=128,verbose=2,
                           callbacks=[checkpointer,lr_reduction,estopping]);


# In[ ]:


history_plot(history)
xception_model.load_weights(fw)
xception_scores=xception_model.evaluate(x_test_bn3,y_test3)
print("Accuracy: %.2f%%"%(xception_scores[1]*100))
xception_scores


# ## InceptionResNetV2

# In[ ]:


resize_x_train1=np.array([scipy.misc.imresize(x_train1[i],(139,139,3)) 
                          for i in range(0,len(x_train1))]).astype('float32')
resize_x_valid1=np.array([scipy.misc.imresize(x_valid1[i],(139,139,3)) 
                           for i in range(0,len(x_valid1))]).astype('float32')
resize_x_test1=np.array([scipy.misc.imresize(x_test1[i],(139,139,3)) 
                         for i in range(0,len(x_test1))]).astype('float32')
x_train_bn1=iv2pi(resize_x_train1)
x_valid_bn1=iv2pi(resize_x_valid1)
x_test_bn1=iv2pi(resize_x_test1)
fn='../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
inceptionresnetv2base_model=InceptionResNetV2(weights=fn,include_top=False)
x_train_bn1=inceptionresnetv2base_model.predict(x_train_bn1)
x_valid_bn1=inceptionresnetv2base_model.predict(x_valid_bn1)
x_test_bn1=inceptionresnetv2base_model.predict(x_test_bn1)


# In[ ]:


sh=x_train_bn1.shape[1:]
def inceptionresnetv2_model():
    model=Sequential()   
    model.add(GlobalAveragePooling2D(input_shape=sh))    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))        
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))
    model.add(Dense(4,activation='softmax'))     
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
inceptionresnetv2_model=inceptionresnetv2_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,
                               verbose=2,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=16,verbose=2)
history=inceptionresnetv2_model.fit(x_train_bn1,y_train1,validation_data=(x_valid_bn1,y_valid1),
     epochs=100,batch_size=128,verbose=2,
     callbacks=[checkpointer,lr_reduction,estopping]);


# In[ ]:


history_plot(history)
inceptionresnetv2_model.load_weights(fw)
inceptionresnetv2_scores=inceptionresnetv2_model.evaluate(x_test_bn1,y_test1)
print("Accuracy: %.2f%%"%(inceptionresnetv2_scores[1]*100))
inceptionresnetv2_scores

