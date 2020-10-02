#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import math
import sys
from scipy import misc
import tensorflow as tf
from skimage import img_as_float
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,BatchNormalization,Input,UpSampling2D,Concatenate,Activation,Add,Flatten,Concatenate,Lambda,Reshape,AveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import Adam
import keras.metrics
import matplotlib.pyplot as plt
from keras import backend as K
#from keras import Callback
from sklearn.model_selection import train_test_split


# In[ ]:


config = tf.ConfigProto()
jit_level = tf.OptimizerOptions.ON_1
config.graph_options.optimizer_options.global_jit_level = jit_level
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


# In[ ]:


name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }


# In[ ]:


PATH = '../input/'
TRAIN = '../input/dadada/tdd/train/'
TEST = '../input/human-protein-atlas-image-classification/test/'
LABEL = '../input/ce-unet-1/nntrain.csv'
SAMPLE = '../input/human-protein-atlas-image-classification/sample_submission.csv'


# In[ ]:


label = pd.read_csv(LABEL)


# In[ ]:


train_names = list(set('_'.join((f.split('_'))[:-1]) for f in os.listdir(TRAIN)))
test_names = list(set('_'.join((f.split('_'))[:-1]) for f in os.listdir(TEST)))


# In[ ]:


train_data, valid_data = train_test_split(train_names, test_size=0.1, random_state=42)


# In[ ]:


def open_rgby(path,pid): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    tmp = np.zeros(shape=(512,512,4))
    for i,color in enumerate(colors) :
        tmp[:,:,i] = img_as_float(misc.imread(os.path.join(path, str(pid)+'_'+color+'.png')))
    return tmp


# In[ ]:


def generator(train,label,batch_size) :
    batch_features = np.zeros((batch_size,512,512,4))
    batch_labels = np.zeros((batch_size,28))
    while True :
        for i in range(batch_size) :
            idx= np.random.choice(len(train),1)[0]
            batch_features[i,:,:,:] = open_rgby(TRAIN,train[idx])
            temp = set([int(i) for i in (label.loc[label['Id']==train[idx]]['Target'].values[0]).split()])
            batch_labels[i,:] =  np.array([1 if i in temp else 0 for i in range(28)])
        
        out = {'main':batch_labels,'encoder':batch_features}
        yield batch_features,out


# In[ ]:


def progressbar(name,i,n):    
    sys.stdout.write('\r')
    sys.stdout.write(name+": [%-20s] %d%% %d/%d" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n) , i,n))
    sys.stdout.flush()


# In[ ]:


def padding(tensor,h,w,r):
    return tf.pad(tensor, tf.constant([[0,0],[h,h], [w,w],[r,r]]), "CONSTANT")


# inp = Input(shape=(512,512,4))
# x = Conv2D(64,(3,3),padding='same')(inp)#256
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# 
# enc1 = MaxPooling2D((2,2),padding='same')(x)
# x = Conv2D(32,(3,3),padding='same')(enc1)#128
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# 
# enc2 = MaxPooling2D((2,2),padding='same')(x)
# x = Conv2D(16,(3,3),padding='same')(enc2)#64
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# 
# enc3 = MaxPooling2D((2,2),padding='same')(x)
# x = Conv2D(16,(3,3),padding='same')(enc3)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# 
# x = UpSampling2D((2,2))(x)
# x = Conv2D(32,(3,3),padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# 
# x = UpSampling2D((2,2))(x)
# x = Conv2D(64,(3,3),padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# 
# x = UpSampling2D((2,2))(x)
# decoder = Conv2D(4,(3,3),activation='sigmoid',padding='same',name='encoder')(x)
# 
# padded0 = Lambda(lambda t: padding(t,0,0,30))(inp)
# padded1 = Lambda(lambda t: padding(t,128,128,0))(enc1) #256,256,64
# padded2 = Lambda(lambda t: padding(t,192,192,16))(enc2) #128,128,32
# padded3 = Lambda(lambda t: padding(t,224,224,24))(enc3) #64,64,16
# 
# add = Concatenate()([padded0,padded1,padded2,padded3])
# avg = AveragePooling2D((64,64))(add)
# flat = (Flatten())(avg)
# dense1 = (Dense(128, activation='relu'))(flat)
# dense2 = (Dense(64, activation='relu'))(dense1)
# out = (Dense(28, activation='sigmoid',name='main'))(dense2)
# 
# model = Model(inputs=inp,outputs=[out,decoder])

# model.compile(optimizer=Adam(),
#               loss={'main':'binary_crossentropy','encoder':'mse'},
#               loss_weights={'main':1,'encoder':0.4},
#               metrics=['acc'])

# model = load_model('../input/c_Unet_pc/c_Unet_save.h5',
#                    custom_objects={'padding': padding})

# In[ ]:


model = load_model('../input/ce-unet-2/ce_Unet_save28.h5')


# In[ ]:


epoch = 4
history = model.fit_generator(generator(train_data,label,4), 
                                  samples_per_epoch=len(train_data)//4, epochs=epoch,
                                  validation_data=generator(valid_data,label,4),
                                  validation_steps=len(valid_data)//4,
                                  callbacks=[ModelCheckpoint('mymodel_valid.h5',monitor='val_loss',save_best_only=True),
                                  ReduceLROnPlateau(monitor='val_loss',patience=3)])
model.save('ce_Unet_save32.h5')


# In[ ]:


model.save('ce_Unet_save32.h5')


# In[ ]:


plt.plot(history.history['main_loss'])
plt.plot(history.history['val_main_loss'])
plt.savefig("ce_Unet_loss_history.png")


# batch_features = np.zeros((1,512,512,4))
# o = []
# m=[]
# train_len = len(train_names)
# pred = np.zeros((train_len,28))
# for k in range(train_len):
#     batch_features[:,:,:,:] = open_rgby(TRAIN,train_names[k])
#     pred[k],l = model.predict(batch_features)
#     o.append(train_names[k])
#     m.append(label.loc[label['Id']==o[k],'Target'].values[0])
#     progressbar('thres find',k,train_len)

# pred = (pred).T

# pdf = pd.DataFrame(data = {'Id':o,'Target':m,
#                           0:pred[0],1:pred[1],2:pred[2],3:pred[3],
#                           4:pred[4],5:pred[5],6:pred[6],7:pred[7],
#                           8:pred[8],11:pred[11],9:pred[9],10:pred[10],
#                           12:pred[12],13:pred[13],14:pred[14],15:pred[15],
#                           16:pred[16],17:pred[17],18:pred[18],19:pred[19],
#                           20:pred[20],21:pred[21],22:pred[22],23:pred[24],
#                           24:pred[24],25:pred[25],26:pred[26],27:pred[27]
#                           })
# pdf = pdf.sort_values(by = ['Id'])

# cal=np.zeros((28,2))
# for index, row in pdf.iterrows():
#     t  = row['Target'].split()
#     for e in t:
#         cal[int(e)][1]+=row[int(e)]
#         cal[int(e)][0]+=1

# cal

# thres = [e[1]/e[0]*0.98 for e in cal]
# print(thres)

# thres = [0.7332285743126011, 0.5805582254419689, 0.6830990579013652, 0.33133608226415084, 0.5892030431602163, 
#          0.33554810233834226, 0.30124813610105283, 0.48676254640934774, 0.05069314378334587, 0.07186778056655506,
#          0.07550342306674565, 0.27888409512220735, 0.13545893820158397, 0.16009892984130975, 0.7817713042352804, 
#          0.0065349138485544245, 0.04680300266810713, 0.22093092964865113, 0.2273163021918054, 0.30589077523285696,
#          0.07060120899301446, 0.4474626562126171, 0.14693263627129444, 0.002407802767594962, 0.22767899205469075, 
#          0.5782695626512265, 0.09819474722240953, 0.10218797941007664]

# out = []
# batch_features = np.zeros((1,512,512,4))
# test_len = len(test_names)
# pred = np.zeros((test_len,28))
# for k in range(test_len):
#     batch_features[:,:,:,:] = open_rgby(TEST,test_names[k])
#     pred[k],l = model.predict(batch_features)
#     
#     outli=''
#     for j in range(28):
#         if(pred[k][j]>thres[j]): 
#             outli+=(str(j)+" ")
#     out.append(outli.strip())
#     progressbar('test predict',k,test_len)

# df = pd.DataFrame(data = {'Id':test_names,'Predicted':out})
# df = df.sort_values(by = ['Id'])
# df.head()

# df.to_csv('c_unet_32'+".csv", header=True, index=False)

# pred = (pred).T

# pdf = pd.DataFrame(data = {'Id':test_names,
#                           0:pred[0],1:pred[1],2:pred[2],3:pred[3],
#                           4:pred[4],5:pred[5],6:pred[6],7:pred[7],
#                           8:pred[8],11:pred[11],9:pred[9],10:pred[10],
#                           12:pred[12],13:pred[13],14:pred[14],15:pred[15],
#                           16:pred[16],17:pred[17],18:pred[18],19:pred[19],
#                           20:pred[20],21:pred[21],22:pred[22],23:pred[24],
#                           24:pred[24],25:pred[25],26:pred[26],27:pred[27]
#                           })
# pdf = pdf.sort_values(by = ['Id'])
# pdf.head()

# pdf.to_csv('c_unet_32_raw'+".csv", header=True, index=False)

# pdf.mean()
