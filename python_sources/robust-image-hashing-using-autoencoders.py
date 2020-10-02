# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from keras import Model,Input,layers,optimizers
import numpy as np
from keras.engine import Layer
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 04:27:03 2018

@author: arnab
"""




from time import time
from keras import Model,Input,layers,optimizers
from keras.callbacks import TensorBoard,EarlyStopping
import os
import logging
from random import choice,random,randint
from keras import backend as K
stopping_criteria=EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
batch_size=400
training_batch_size=80

visualizer_128=TensorBoard(log_dir='logs/{}'.format(time()),update_freq=100)
visualizer_512=TensorBoard(log_dir='./logs/model_512/',histogram_freq=1, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False,update_freq=100)
visualizer_128_reg=TensorBoard(log_dir='./logs/model_128_reg/',histogram_freq=1, batch_size=batch_size, write_graph=True, write_grads=False, write_images=True,update_freq=100)

class Round(Layer):
  def __init__(self,input_dim,**kwargs):
    self.input_dim=input_dim
    super().__init__(**kwargs)
  def build(self,input_shape):
    assert(len(input_shape)>0)
    
    self.built=True
  def call(self,inputs):
    
    output=K.flatten(K.round(inputs))
    print(output.shape)
    return output
    
  def compute_output_shape(self,input_shape):
    return (input_shape[0],1024)
  
class Add(Layer):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
  def build(self,input_shape):
    self.built=True
  def call(self,inputs):
    output=K.sum(inputs,axis=-2,keepdims=True)
    print(output.shape)
    return output
  def compute_output_shape(self,input_shape):
    return (input_shape[0],1024)
    
      






def binary_regularizer(activity_vector):
    return 0.001*K.sum(K.abs(activity_vector))



def make_model_128():
    
    input_img=Input(shape=(128,128,3))
    with K.name_scope('Encoder'):
            
        with K.name_scope('Convolution_layer_1'):
            
            x=layers.Conv2D(128,(3,3),activation='relu',padding='same',use_bias=True)(input_img)
            x=layers.BatchNormalization()(x);
            x=layers.MaxPool2D((2,2),padding='same')(x)
            #x=layers.Dropout(0.3)(x)
        #64
        with K.name_scope('Convolution_layer_2'):
            
            x=layers.Conv2D(64,(3,3),activation='relu',padding='same',use_bias=True)(x)
            x=layers.BatchNormalization()(x);
            x=layers.MaxPool2D((2,2),padding='same')(x)
            #x=layers.Dropout(0.3)(x)
        #32
        with K.name_scope('Convolution_layer_3'):
            
            x=layers.Conv2D(32,(3,3),activation='relu',padding='same',use_bias=True)(x)
            x=layers.BatchNormalization()(x);
            x=layers.MaxPool2D((2,2),padding='same')(x)
            #x=layers.Dropout(0.3)(x)
        #16
        with K.name_scope('Convolution_layer_4'):
            
            x=layers.Conv2D(16,(3,3),activation='relu',padding='same',use_bias=True)(x)
            x=layers.BatchNormalization()(x);
            x=layers.MaxPool2D((2,2),padding='same',name='encoded_output')(x)
            #x=layers.Dropout(0.3)(x)
    with K.name_scope('Decoder'):    
        with K.name_scope('DeConvolution_layer_1'):
            
            x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x)
            x1=layers.BatchNormalization()(x1);
            x1=layers.Conv2D(16,(3,3),activation='sigmoid',padding='same',use_bias=True)(x)
            
        
        #x1=layers.Dropout(0.3)(x1)
        #16
        
        with K.name_scope('DeConvolution_layer_2'):
            
            x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
            x1=layers.BatchNormalization()(x1);
            x1=layers.Conv2D(32,(3,3),activation='sigmoid',padding='same',use_bias=True)(x1)
            #x1=layers.Dropout(0.3)(x1)
        #3
        with K.name_scope('DeConvolution_layer_3'):
            
            x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
            x1=layers.BatchNormalization()(x1);
            x1=layers.Conv2D(64,(3,3),activation='sigmoid',padding='same',use_bias=True)(x1)
            #x1=layers.Dropout(0.3)(x1)
        #32
        with K.name_scope('DeConvolution_layer_4'):
            x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
            x1=layers.BatchNormalization()(x1);
            x1=layers.Conv2D(128,(3,3),activation='sigmoid',padding='same',use_bias=True)(x1)
           #64
        with K.name_scope('DeConvolution_layer_5_with_output'):
            x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
            x1=layers.BatchNormalization()(x1);
            decoder_output=layers.Conv2D(3,(3,3),activation='sigmoid',padding='same',name='final_output',use_bias=True)(x1)
           #128
    final_model=Model(input_img,decoder_output)
    final_model.compile(loss='mean_squared_error',optimizer='adam',metrics=['acc'])
    return final_model
def make_model_512():

    input_img=Input(shape=(512,512,3))
    x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(input_img)
    x=layers.BatchNormalization()(x);
    x=layers.MaxPool2D((2,2),padding='same')(x)
    # x=layers.Dropout(0.3)(x)
    #256
    x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    x=layers.BatchNormalization()(x);
    x=layers.MaxPool2D((2,2),padding='same')(x)
    # x=layers.Dropout(0.3)(x)
    #128
    x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    x=layers.BatchNormalization()(x);
    x=layers.MaxPool2D((2,2),padding='same')(x)
    # x=layers.Dropout(0.3)(x)
    #64
    x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    x=layers.BatchNormalization()(x);
    x=layers.MaxPool2D((2,2),padding='same')(x)
    # x=layers.Dropout(0.3)(x)
    #32
    x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    x=layers.BatchNormalization()(x);
    x=layers.MaxPool2D((2,2),padding='same')(x)
    # x=layers.Dropout(0.3)(x)
    #16

    # x=layers.Dense(16*16*16,activation='relu')(x)
    # x=layers.Dropout(0.3)(x)
    # x=layers.Dense(16*16*16,activation='relu')(x)
    # x=layers.Dense(8*8*16,activation='relu')(x)
    # encoder_output=layers.Reshape((8,8,16))(x)
    x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    x=layers.BatchNormalization()(x);
    x=layers.MaxPool2D((2,2),padding='same')(x)
    # x=layers.Dropout(0.3)(x)
    #8
    # x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    # encoder_output=layers.MaxPool2D((2,2),padding='same')(x)


    # # now the decoder part



    # #decoder_input=Input(shape=(4,4,16))
    # x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(encoder_output)
    # x1=layers.UpSampling2D((2,2))(x1)


    x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x)
    x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
    # x1=layers.Dropout(0.3)(x1)
    #16


    x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
    x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x1)
    # x1=layers.Dropout(0.3)(x1)
    #32

    x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
    x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x1)
    # x1=layers.Dropout(0.3)(x1)
    #64

    x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
    x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x1)
    # x1=layers.Dropout(0.3)(x1)
    #128
    x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
    x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x1)
    # x1=layers.Dropout(0.3)(x1)
    #256

    x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
    x1=layers.Conv2D(1,(3,3),activation='relu',padding='same')(x1)
    # x1=layers.Dropout(0.3)(x1)
    #512
    x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
    decoder_output=layers.Conv2D(3,(3,3),activation='relu',padding='same')(x1)

    final_model=Model(input_img,decoder_output)
     
    final_model.compile(loss='mean_squared_error',optimizer=optimizers.Adagrad(lr=1e-4),metrics=['acc'])
    return final_model
 




from sklearn.model_selection import train_test_split
import glob
from PIL import Image
import os
import random
import numpy as np
import re
import gc
from functools import partial
image_x=128
image_y=128

x_dir=['../input/aerials/aerials/operations_aerials','../input/operations_animals/operations_animals','../input/operations_scenery/operations_scenery']
y_dir=['../input/aerials/aerials','../input/animals_bmp/animals_bmp','../input/scenery_bmp/scenery_bmp']
 

def find_file_name(image):
    image_name=re.sub(r"(.npy)|(.tiff)|(.jpg)|(.bmp)|(.tif)","",image)     
    return image_name

class Data:
    def __init__(self,x_dir=x_dir,y_dir=y_dir,image_x=image_x,image_y=image_y):
        self.x_dir=x_dir
        self.y_dir=y_dir
        self.image_x=image_x
        self.image_y=image_y
        self.x_train=[]
        self.y_train=[]
        self.transform=lambda image_name,image_x,image_y:np.asarray(Image.open(image_name).resize((image_x,image_y)))/255
        
            
        for i in range(len(x_dir)):
            print('Currently doing {} folder '.format(y_dir[i][y_dir[i].rfind('/')+1:]))
            x_train_dir=x_dir[i]
            y_train_dir=y_dir[i]
            for image_name in os.listdir(y_train_dir):
                image=find_file_name(image_name)
                logging.debug(image,image_name)
                operations_image=glob.glob(x_train_dir+'/*/'+image+'/*')
                operations_image=[image_name for image_name in operations_image if 'RST' not in image_name]
                print('Found {} number of images in {}'.format(len(operations_image),image_name))
                _y=[y_train_dir+'/'+image_name]
                self.x_train+=operations_image        
                self.y_train+=_y*len(operations_image)
        assert(len(self.x_train)==len(self.y_train))
        logging.debug(self.x_train)
        logging.debug(self.y_train)
        self.x_train=np.array(self.x_train)
        self.y_train=np.array(self.y_train)
        
    def __call__(self,batch_size,image_x=image_x,image_y=image_y):
        '''
        generates the data that will be used for training the model
        
        '''
        random_choices=[random.randint(0,len(self.x_train)-1) for i in range(batch_size)]
        x=self.x_train[random_choices]
        y=self.y_train[random_choices]
        x=np.array(list(map(partial(self.transform,image_x=image_x,image_y=image_y),x)))
        y=np.array(list(map(partial(self.transform,image_x=image_x,image_y=image_y),y)))
        
        assert(len(x)==len(y))
        # print(x.shape)
        # print(y.shape)
        return x,y
    
    def __repr__(self):
        return 'x_train contains {} images\ny_train contains {} images\n'.format(len(self.x_train),len(self.y_train))

class OperationData(Data):
    def __init__(self,operation=None):
        super().__init__()
        
        self.indexes=[i for i,data in enumerate(self.x_train) if operation in data]
        logging.debug('Found {}  number of images corresponding to {}'.format(len(self.indexes),operation))
        print('Loading data for  {} operation'.format(operation))
        self.x_train=self.x_train[self.indexes]
        self.y_train=self.y_train[self.indexes]

class CompressionData(OperationData):
    def __init__(self,operation='compression'):
        super().__init__(operation=operation)
        
class ContrastData(OperationData):
    def __init__(self,operation='contrast'):
        super().__init__(operation=operation)
        
class GammaData(OperationData):
    def __init__(self,operation='gamma'):
        super().__init__(operation=operation)
        

class GaussianData(OperationData):
    def __init__(self,operation='gaussian'):
        super().__init__(operation=operation)
        
class BrightnessData(OperationData):
    def __init__(self,operation='brightness'):
        super().__init__(operation=operation)
        
class RotationData(OperationData):
    def __init__(self,operation='RST'):
        super().__init__(operation=operation)
    
class WatermarkData(OperationData):
    def __init__(self,operation='watermark'):
        super().__init__(operation=operation)
        
class SaltAndPepperData(OperationData):
    def __init__(self,operation='salt and pepper'):
        super().__init__(operation=operation)
        
class ScalingData(OperationData):
    def __init__(self,operation='scaling'):
        super().__init__(operation=operation)
        
class SpeckleData(OperationData):
    def __init__(self,operation='speckle'):
        super().__init__(operation=operation)
        

        
def train(data:Data,final_model=None,visualizer=None,image_x=image_x,image_y=image_y,batch_size:int=batch_size,iterations=10):
    count=0
    train_acc,train_loss,val_acc,val_loss=[],[],[],[]
    while count<iterations:
        x,y=data(batch_size,image_x,image_y)
        if final_model==None and Visualizer==None:
            raise ValueError('Final model and Visualizer was not given as a parameter to the method by the user')
        if visualizer is not None:
            history=final_model.fit(x,y,epochs=10,validation_split=0.2,verbose=2,batch_size=training_batch_size,callbacks=[visualizer])
            
        else:
            history=final_model.fit(x,y,epochs=10,validation_split=0.2,batch_size=training_batch_size,verbose=2)
        train_acc.append(history.history['acc'][0])
        train_loss.append(history.history['loss'][0])
        val_acc.append(history.history['val_acc'][0])
        val_loss.append(history.history['val_loss'][0])
        del x,y
        gc.collect()
        logging.debug('Collecting garbage')
        count+=1
    
        logging.info('Iteration {}/{}'.format(count,iterations))
    
    assert(len(train_acc)==iterations)
    return train_acc,train_loss,val_acc,val_loss


if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.disable(logging.DEBUG)
    final_model_128=make_model_128()
    # final_model_512=make_model_512()
    # final_model_512.summary()
     
    
    train_data=Data()
    #rotation_data=RotationData()
    
     
    print('Training the model for an input of 128 X 128')
    
    train_acc,train_loss,val_acc,val_loss=train(train_data,final_model_128,iterations=1)
    
    #train(rotation_data,final_model_512,iterations=100)
    
    
    
    
    #final_model_128.save('8_bilinear_v6_128.h5')
 