#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Model,Sequential
from keras.preprocessing.image import img_to_array,load_img
from keras.layers import Dense,GlobalAveragePooling2D,Input,Embedding,InputLayer,Activation,Flatten,Conv2D,MaxPooling2D
import os
print(os.listdir("../input"))
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm,tqdm_notebook
import tensorflow as tf
# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/flickr30k_images/flickr30k_images/results.csv",delimiter='|')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


imgbasedir="../input/flickr30k_images/flickr30k_images/flickr30k_images/"


# In[ ]:


imgdir=glob.glob(imgbasedir+"*.jpg")


# In[ ]:


imgdir[:5]


# In[ ]:


img=cv2.imread(imgbasedir+df.image_name.iloc[0])
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
print(df[' comment'].iloc[0])
print(img.shape)


# In[ ]:


img=cv2.imread(imgbasedir+df.image_name.iloc[15])
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
print(df[' comment'].iloc[15])
print(img.shape)


# In[ ]:


imagesize=(299,299,3)


# In[ ]:


def fun1(df1):
    m=list(df1[" comment"].values)
    return m


# In[ ]:


df1=df.groupby(by='image_name').apply(fun1)


# In[ ]:


index1=df1.index


# In[ ]:


values=df1.values


# In[ ]:


index1[1]


# In[ ]:


dict1=dict([(index1[i],values[i]) for i in range(len(values))])


# In[ ]:





# In[ ]:


model=InceptionResNetV2(include_top=False,weights='imagenet',input_shape=(299,299,3))


# In[ ]:





# In[ ]:


for layer in model.layers:
    layer.trainable=False


# In[ ]:


bottommodel=model.output
topmodel=GlobalAveragePooling2D()(bottommodel)


# In[ ]:


model1=Model(model.input,topmodel)


# In[ ]:





# In[ ]:





# In[ ]:


index2=index1[:6000]   # taking 5000 images


# In[ ]:


imgbasedir


# In[ ]:


len(index1)


# In[ ]:


xtrain=[]
for i in range(len(index2)):
    img=cv2.imread(imgbasedir+index2[i])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(299,299)).astype('float16')
    xtrain.append(img)


# In[ ]:


xtrain=np.array(xtrain).astype('float16')


# In[ ]:


xtrain.dtype


# In[ ]:


xtrain=xtrain.astype(np.float16)/255


# In[ ]:


pred=model1.predict(xtrain)


# In[ ]:


xtrain=np.zeros([0,0])


# In[ ]:


pred=pred.astype('float16')


# In[ ]:


pred[0]


# In[ ]:


pred.shape


# In[ ]:


Imgbottleneck=120
wordembedsize=32
rnnsize=256
ns=1536


# In[ ]:


import random


# In[ ]:


tokendata=[random.sample(dict1[index2[i]],1)[0] for i in range(6000)]


# In[ ]:


tokendata[:5]


# In[ ]:


img=cv2.imread(imgbasedir+index2[0])
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)


# In[ ]:


import re


# In[ ]:


# Text Preprocessing
def fun(text):
    text=text.lower()
    text=re.sub(r"[^\w\d]"," ",text)
    text=re.sub(r"\s{2,}"," ",text)
    text=text.strip()
    return text


# In[ ]:


tokendata1=[fun(i) for i in tokendata]


# In[ ]:


tokendata1[:5]


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


tokenizer=Tokenizer()


# In[ ]:


tokenizer.fit_on_texts(tokendata1)


# In[ ]:


tokendata2=tokenizer.texts_to_sequences(tokendata1)


# In[ ]:


word2index=tokenizer.word_index


# In[ ]:


word2index['<pad>']=0


# In[ ]:


word2index['<start>']=len(word2index)
word2index['<end>']=len(word2index)


# In[ ]:


index2word=dict([(i,j) for j,i in word2index.items()])


# In[ ]:


len(index2word)


# In[ ]:


index2word[0]


# In[ ]:


tokendata3=[[word2index['<start>']]+tokendata2[i]+[word2index['<end>']] for i in range(len(tokendata2))]


# In[ ]:


len(tokendata3)


# In[ ]:


text=pad_sequences(tokendata2,padding='post')


# In[ ]:


imgembedsize=pred.shape[1]


# In[ ]:


logitsbottleneck=200


# In[ ]:


ns=400


# In[ ]:


Imgbottleneck=128


# In[ ]:


imgemb=tf.placeholder(shape=[None,1536],dtype=tf.float32)
sentences=tf.placeholder(shape=[None,None],dtype=tf.int32)
imgembed_bottleneck=Dense(Imgbottleneck,input_shape=(None,imgembedsize),activation='elu')
imgbottle_h=Dense(ns,input_shape=(None,Imgbottleneck),activation='elu')
embedding=Embedding(len(word2index),wordembedsize)
lstm=tf.nn.rnn_cell.LSTMCell(ns)
token_logits_bottleneck=Dense(logitsbottleneck,input_shape=(None,ns),activation='elu')
tokenlogits=Dense(len(word2index),input_shape=(None,logitsbottleneck))


# In[ ]:


c0=h0=imgbottle_h(imgembed_bottleneck(imgemb))


# In[ ]:


c0.get_shape()


# In[ ]:


word_embeds=embedding(sentences[:,:-1])


# In[ ]:


word_embeds.get_shape()


# In[ ]:


hiddenstates,_=tf.nn.dynamic_rnn(lstm,word_embeds,initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0,h0))


# In[ ]:


hiddenstates.get_shape()


# In[ ]:


flat_token_logits=tf.reshape(hiddenstates,(-1,ns))
flat_token_logits=tokenlogits(token_logits_bottleneck(flat_token_logits))
flat_ground_truth=tf.reshape(sentences[:,1:],[-1,])
flat_loss_mask=tf.not_equal(word2index['<pad>'],flat_ground_truth)
entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_ground_truth,logits=flat_token_logits)


# In[ ]:


loss=tf.reduce_mean(tf.boolean_mask(entropy,flat_loss_mask))


# In[ ]:


optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
trainstep=optimizer.minimize(loss)


# In[ ]:


s=tf.InteractiveSession()


# In[ ]:


s.run(tf.global_variables_initializer())


# In[ ]:


def batch_matrix(batchcaption,padidx,maxlen=None):
    maxlen=max(map(len,batchcaption))
    matrix=np.zeros((len(batchcaption),maxlen))+padidx
    for i in range(len(batchcaption)):
        matrix[i,:len(batchcaption[i])]=batchcaption[i]
    return matrix


# In[ ]:


def generate_batch(imgemb1,indxcaption,batchsize,maxlen=None):
    m=np.random.choice(len(imgemb1),size=batchsize,replace=False)
    batch_img_embed=imgemb1[m]
    batch_captions=[tokendata3[i] for i in m]
    
    batch_padcaption=batch_matrix(batch_captions,0)
    
    return {imgemb:batch_img_embed,sentences:batch_padcaption}   


# In[ ]:


batchsize=64
n_epochs=8
n_batches_per_epoch=1000


# In[ ]:


lstm_c=tf.Variable(tf.zeros([1,400]))
lstm_h=tf.Variable(tf.zeros([1,400]))


# In[ ]:


s = tf.InteractiveSession()
tf.set_random_seed(42)


# In[ ]:


s.run(tf.global_variables_initializer())


# In[ ]:


trainlosslist=[]
for epoch in range(n_epochs):
    trainloss=0
    count=0
    for i in range(n_batches_per_epoch):
        trainloss1,_=s.run([loss,trainstep],feed_dict=generate_batch(pred,tokendata3,batchsize))
        count+=1
        trainlosslist.append(trainloss1)
        if i%50==0:
            
            print(trainloss1)


# In[ ]:





# In[ ]:


imgs=tf.placeholder('float32',[1,299,299,3])


# In[ ]:


testpred=model1(imgs)


# In[ ]:


in_h=in_c=imgbottle_h(imgembed_bottleneck(testpred))


# In[ ]:


in_h.get_shape()


# In[ ]:


in_c.get_shape()


# In[ ]:


lstm_c.assign(in_c)
lstm_h.assign(in_h)


# In[ ]:


currentword=tf.placeholder("int32",shape=[1])


# In[ ]:


wordembed=embedding(currentword)


# In[ ]:


wordembed.get_shape()


# In[ ]:


lstm.weights


# In[ ]:


lstm(wordembed,state=tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]


# In[ ]:




