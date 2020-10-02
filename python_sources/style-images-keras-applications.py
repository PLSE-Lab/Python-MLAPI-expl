#!/usr/bin/env python
# coding: utf-8

# <h1 class="font-effect-fire-animation" style="color:crimson; font-family:Akronim;">Code Modules, Styling, Helpful Functions and Links</h1>
# #### [Github Version](https://github.com/OlgaBelitskaya/deep_learning_projects/blob/master/DL_PP4) & [Colaboratory Version](https://colab.research.google.com/drive/1r5yRD-3tQwN6lSql_VRoVuwQ8DaY5zUt)

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style> \n@import url('https://fonts.googleapis.com/css?family=Akronim|Roboto&effect=fire-animation');\nspan {color:black; text-shadow:4px 4px 4px #aaa;}\ndiv.output_prompt {color:crimson;} \ndiv.input_prompt {color:firebrick;} \ndiv.output_area pre,div.output_subarea {font-size:15px; color:crimson}\ndiv.output_stderr pre {background-color:#f7e8e8;}\n</style>")


# In[ ]:


import warnings; warnings.filterwarnings('ignore')
import h5py,os,pandas as pd,numpy as np,pylab as pl
import seaborn as sn,keras as ks,tensorflow as tf
from skimage.transform import resize
import scipy; from scipy import misc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
np.set_printoptions(precision=6)
pl.style.use('seaborn-whitegrid')
fw='weights.style.hdf5'
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential,load_model,Model
from keras.layers import Input,Activation,Dense,LSTM
from keras.layers import Flatten,Dropout,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50pi
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inceptionv3pi
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xceptionpi
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2pi
from keras import __version__
print('keras version:', __version__)
print('tensorflow version:',tf.__version__)
print(os.listdir("../input"))


# In[ ]:


def ohe(x): 
    return OneHotEncoder(n_values='auto')           .fit(x.reshape(-1,1)).transform(x.reshape(-1,1))           .toarray().astype('int64')
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


# <h1 class="font-effect-fire-animation" style="color:crimson; font-family:Akronim;">Data Preprocessing</h1>

# In[ ]:


data=pd.read_csv("../input/style-color-images/style/style.csv")
data.tail()


# In[ ]:


f=h5py.File('../input/style-color-images/StyleColorImages.h5','r')
keys=list(f.keys())
brands=np.array(f[keys[0]])
images=np.array(f[keys[1]])
products=np.array(f[keys[2]])
[keys,[brands.shape,images.shape,products.shape]]


# In[ ]:


# normalization of image arrays
images=images.astype('float32')/255
print('Product: ',data['product_name'][100])
print('Brand: ',data['brand_name'][100])
pl.figure(figsize=(3,3)); pl.imshow(images[100])
pl.show()


# In[ ]:


# one-hot encoding
cbrands,cproducts=ohe(brands),ohe(products)
ctargets=np.concatenate((cbrands,cproducts),axis=1)
pd.DataFrame([images.shape,cbrands.shape,
              cproducts.shape,ctargets.shape])


# In[ ]:


# splitting the data
# Color Images / Brands 
x_train1,x_valid1,x_test1,y_train1,y_valid1,y_test1=tts(images,cbrands)
# Color Images / Products 
x_train3,x_valid3,x_test3,y_train3,y_valid3,y_test3=tts(images,cproducts)


# <h1 class="font-effect-fire-animation" style="color:crimson; font-family:Akronim;">Classification Models</h1>
# ## ResNet50

# In[ ]:


# creating bottleneck features
resize_x_train=np.array([scipy.misc.imresize(x_train1[i],(197,197,3)) 
                           for i in range(0,len(x_train1))]).astype('float32')
resize_x_valid=np.array([scipy.misc.imresize(x_valid1[i],(197,197,3)) 
                           for i in range(0,len(x_valid1))]).astype('float32')
resize_x_test=np.array([scipy.misc.imresize(x_test1[i],(197,197,3)) 
                         for i in range(0,len(x_test1))]).astype('float32')
x_train_bn=resnet50pi(resize_x_train)
x_valid_bn =resnet50pi(resize_x_valid)
x_test_bn=resnet50pi(resize_x_test)
fn='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet50_base_model=ResNet50(weights=fn,include_top=False)
x_train_bn=resnet50_base_model.predict(x_train_bn)
x_valid_bn=resnet50_base_model.predict(x_valid_bn)
x_test_bn=resnet50_base_model.predict(x_test_bn)


# In[ ]:


sh=x_train_bn.shape[1:]
def resnet50_model():
    model=Sequential()   
    model.add(GlobalAveragePooling2D(input_shape=sh))    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))        
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))
    model.add(Dense(7,activation='softmax'))    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
resnet50_model=resnet50_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=2,factor=.5)
estopping=EarlyStopping(monitor='val_loss',patience=25,verbose=2)
history=resnet50_model.fit(x_train_bn,y_train1,
                   validation_data=(x_valid_bn,y_valid1),
                   epochs=100,batch_size=128,verbose=2,
                   callbacks=[checkpointer,lr_reduction,estopping]);


# In[ ]:


history_plot(history)
resnet50_model.load_weights(fw)
resnet50_scores=resnet50_model.evaluate(x_test_bn,y_test1)
print("Accuracy: %.2f%%"%(resnet50_scores[1]*100))
resnet50_scores


# ## Inception V3

# In[ ]:


resize_x_train=np.array([scipy.misc.imresize(x_train3[i],(139,139,3)) 
                         for i in range(0,len(x_train3))]).astype('float32')
resize_x_valid=np.array([scipy.misc.imresize(x_valid3[i],(139,139,3)) 
                         for i in range(0,len(x_valid3))]).astype('float32')
resize_x_test=np.array([scipy.misc.imresize(x_test3[i],(139,139,3)) 
                        for i in range(0,len(x_test3))]).astype('float32')
x_train_bn=inceptionv3pi(resize_x_train)
x_valid_bn=inceptionv3pi(resize_x_valid)
x_test_bn=inceptionv3pi(resize_x_test)
fn='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_v3_base_model=InceptionV3(weights=fn,include_top=False)
x_train_bn=inception_v3_base_model.predict(x_train_bn)
x_valid_bn=inception_v3_base_model.predict(x_valid_bn)
x_test_bn=inception_v3_base_model.predict(x_test_bn)


# In[ ]:


sh=x_train_bn.shape[1:]
def inception_v3_model():
    model=Sequential()  
    model.add(GlobalAveragePooling2D(input_shape=sh))    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))       
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))
    model.add(Dense(10,activation='softmax'))     
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',metrics=['accuracy'])
    return model
inception_v3_model=inception_v3_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=2,factor=.5)
estopping=EarlyStopping(monitor='val_loss',patience=25,verbose=2)
history=inception_v3_model.fit(x_train_bn,y_train3,
                       validation_data=(x_valid_bn,y_valid3),
                       epochs=100,batch_size=128,verbose=2,
                       callbacks=[checkpointer,lr_reduction,estopping]);


# In[ ]:


history_plot(history)
inception_v3_model.load_weights(fw)
inception_v3_scores=inception_v3_model.evaluate(x_test_bn,y_test3)
print("Accuracy: %.2f%%"%(inception_v3_scores[1]*100))
inception_v3_scores


# ## Xception

# In[ ]:


resize_x_train=np.array([scipy.misc.imresize(x_train1[i],(71,71,3)) 
                         for i in range(0,len(x_train1))]).astype('float32')
resize_x_valid=np.array([scipy.misc.imresize(x_valid1[i],(71,71,3)) 
                         for i in range(0,len(x_valid1))]).astype('float32')
resize_x_test=np.array([scipy.misc.imresize(x_test1[i],(71,71,3)) 
                        for i in range(0,len(x_test1))]).astype('float32')
x_train_bn=xceptionpi(resize_x_train)
x_valid_bn=xceptionpi(resize_x_valid)
x_test_bn=xceptionpi(resize_x_test)
fn='../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
xception_base_model=Xception(weights=fn,include_top=False)
x_train_bn=xception_base_model.predict(x_train_bn)
x_valid_bn=xception_base_model.predict(x_valid_bn)
x_test_bn=xception_base_model.predict(x_test_bn)


# In[ ]:


sh=x_train_bn.shape[1:]
def xception_model():
    model=Sequential()   
    model.add(GlobalAveragePooling2D(input_shape=sh))    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))        
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))
    model.add(Dense(7,activation='softmax'))   
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
xception_model=xception_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=2,factor=.5)
estopping=EarlyStopping(monitor='val_loss',patience=25,verbose=2)
history=xception_model.fit(x_train_bn,y_train1,
                   validation_data=(x_valid_bn,y_valid1),
                   epochs=100,batch_size=128,verbose=2,
                   callbacks=[checkpointer,lr_reduction,estopping]);


# In[ ]:


history_plot(history)
xception_model.load_weights(fw)
xception_scores=xception_model.evaluate(x_test_bn,y_test1)
print("Accuracy: %.2f%%"%(xception_scores[1]*100))
xception_scores


# ## InceptionResNetV2

# In[ ]:


resize_x_train=np.array([scipy.misc.imresize(x_train3[i],(139,139,3)) 
                         for i in range(0,len(x_train3))]).astype('float32')
resize_x_valid=np.array([scipy.misc.imresize(x_valid3[i],(139,139,3)) 
                         for i in range(0,len(x_valid3))]).astype('float32')
resize_x_test=np.array([scipy.misc.imresize(x_test3[i],(139,139,3)) 
                        for i in range(0,len(x_test3))]).astype('float32')
x_train_bn=inceptionresnetv2pi(resize_x_train)
x_valid_bn=inceptionresnetv2pi(resize_x_valid)
x_test_bn=inceptionresnetv2pi(resize_x_test)
fn='../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
inceptionresnetv2_base_model=InceptionResNetV2(weights=fn,include_top=False)
x_train_bn=inceptionresnetv2_base_model.predict(x_train_bn)
x_valid_bn=inceptionresnetv2_base_model.predict(x_valid_bn)
x_test_bn=inceptionresnetv2_base_model.predict(x_test_bn)


# In[ ]:


sh=x_train_bn.shape[1:]
def inceptionresnetv2_model():
    model=Sequential()   
    model.add(GlobalAveragePooling2D(input_shape=sh))    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))
    model.add(Dense(10,activation='softmax'))    
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',metrics=['accuracy'])
    return model
inceptionresnetv2_model=inceptionresnetv2_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=2,factor=.5)
estopping=EarlyStopping(monitor='val_loss',patience=25,verbose=2)
history=inceptionresnetv2_model.fit(x_train_bn,y_train3,
                            validation_data=(x_valid_bn,y_valid3),
                            epochs=100,batch_size=128,verbose=2,
                            callbacks=[checkpointer,lr_reduction,estopping]);


# In[ ]:


history_plot(history)
inceptionresnetv2_model.load_weights(fw)
inceptionresnetv2_scores=inceptionresnetv2_model.evaluate(x_test_bn,y_test3)
print("Accuracy: %.2f%%"%(inceptionresnetv2_scores[1]*100))
inceptionresnetv2_scores

