#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))"""

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from tensorflow.keras import applications
import tensorflow as tf 

model=applications.InceptionV3(weights="imagenet")

model.layers.pop()
model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
model.summary()


# In[ ]:


from tensorflow.keras.preprocessing import image
from tqdm import tqdm 
file=open('/kaggle/input/flickr8k/captions.txt','r')
file.close()
features=dict()
from tqdm import tqdm 
directory='/kaggle/input/flickr8k/Images'
list_images=os.listdir(directory)

for i in tqdm(range(8091)):
    path=os.path.join(directory,list_images[i])
    load_image=image.load_img(path, target_size=(299, 299))
    feature = image.img_to_array(load_image)
    feature = np.expand_dims(feature, axis=0)
    name=list_images[i].split('.')[0]
    feature=model.predict(feature,verbose=0)
    features[name]=feature
    
    


# In[ ]:


import pickle
pickle.dump(features, open( "features.p", "wb"))


# In[ ]:


import pickle
features=pickle.load( open( "features.p", "rb" ) )


# In[ ]:


file=open('/kaggle/input/flickr8k/captions.txt','r')

lines=file.readlines()
describtion=dict()
keys=[]
i=0
for line in lines[1:]:
    index=line.index(',') 
    if line[:index].split('.')[0] not  in keys:
        describtion[line[:index].split('.')[0]]=list()
    describtion[line[:index].split('.')[0]].append(line[index+1:])    
    keys=list(describtion.keys())
file.close()    


# In[ ]:


from tqdm import tqdm
len(keys)


# In[ ]:


import re
import string
def clean_desc(description,keys,length,start):
    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
    for i in tqdm(range(start,length)):
        comments=description[keys[i]]
        cleaned_comment=[]
        for comment in comments:
            comment=[word.lower() for word in comment.split()]
            comment=[re_punc.sub('',w) for w in comment]
        
            words=[word for word in comment if len(word)>1]
            comment=' '.join(words)
    
            comments='startseq '+comment+' endseq'
            cleaned_comment.append(comment)
        description[keys[i]]=cleaned_comment
    return description


# In[ ]:


train_desc=clean_desc(describtion,keys,6000,0)
test_desc=clean_desc(describtion,keys,7000,6000)


# In[ ]:


train_desc[keys[2]]


# In[ ]:


from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = '/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'
file=open('glove.6B.100d.txt.word2vec','w')
file.close()
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
filename = 'glove.6B.100d.txt.word2vec'
glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)


# In[ ]:


lis=list(train_desc.values())
text=[]
for i in lis:
    for j in i:
        text.append(j)

total_words=[]
lengths=[]
for i in text:
    total_words.extend(i.split())
    lengths.append(len(i.split()))
from collections import Counter
print(len(total_words))


# In[ ]:


len(text)


# In[ ]:


import seaborn as sns
sns.distplot(lengths)


# In[ ]:


max_len=max(lengths)

print(max_len)


# In[ ]:


word_dict=Counter(total_words)

words=list(word_dict.keys())

values=list(word_dict.values())

def greater_than_10(x):
    if x>10:
        return x
    return -1
valid_index=list(map(greater_than_10,values))
valid_words=[words[i] for i in range(len(words)) if valid_index[i]!=-1]    


# In[ ]:


len(words)


# In[ ]:


len(valid_words)


# In[ ]:


word_rank=dict()
j=1
for i in range(len(word_dict)):
    if max(values)<=10:
        break
    index=values.index(max(values))
    word_rank[words[index]]=j
    j+=1
    values[index]=-1


# In[ ]:


vocab_size=len(word_rank)+1
print(vocab_size)


# In[ ]:


train_seq=dict()

for i in range(6000):
    train_seq[keys[i]]=list()
    for k in train_desc[keys[i]]:
        seq=[]
        for j in k.split():
            try:
                seq.append(word_rank[j])
            except:
                continue
        train_seq[keys[i]].append(seq) 
    
test_seq=dict()

for i in range(6000,7000):
    test_seq[keys[i]]=list()
    for k in test_desc[keys[i]]:
        seq=[]
        for j in k.split():
            try:
                seq.append(word_rank[j])
            except:
                continue
            
        test_seq[keys[i]].append(seq)


# In[ ]:


embedding_matrix = np.zeros((vocab_size, 100))
for word, i in word_rank.items():
    try:
        embedding_matrix[i] = glove_model[word]
    except:
        continue


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def create_sequences(descriptions,vocab_size,max_len,images,keys,start,end):
    x_images,x_seq,y=list(),list(),list()
    for j in tqdm(range(start,end)):
        seqs=descriptions[keys[j]]
        image=features[keys[j]]
        for seq in seqs:
        
            for i in range(1,len(seq)):
                in_seq,out_seq=seq[:i],seq[i]
                in_seq=pad_sequences([in_seq],maxlen=max_len)
                out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
            
                x_images.append(image)
                x_seq.append(in_seq)
                y.append(out_seq)
    return  np.array(x_images),np.array(x_seq),np.array(y)

x_train_features,x_train_seq,y_train=create_sequences(train_seq,vocab_size,max_len,features,keys,0,6000)
#x_test_features,x_test_seq,y_test=create_sequences(test_seq,vocab_size,max_len,features,keys,6000,7000)


# In[ ]:


x_train_features.shape


# In[ ]:


x_test_features.shape


# In[ ]:


x_train_features=x_train_features.reshape(x_train_features.shape[0],2048)
#x_test_features=x_test_features.reshape(x_test_features.shape[0],2048)

x_train_seq=x_train_seq.reshape(x_train_features.shape[0],32)
#x_test_seq=x_test_seq.reshape(x_test_features.shape[0],32)


# In[ ]:


def caption_model(vocab_size,max_len):
    input1=tf.keras.layers.Input(shape=(2048,))
    x1=tf.keras.layers.Dropout(0.5)(input1)
    x2=tf.keras.layers.Dense(256,activation='relu')(x1)
    
    input2=tf.keras.layers.Input(shape=(max_len,))
    e1=tf.keras.layers.Embedding(vocab_size,100,weights=[embedding_matrix],mask_zero=True)(input2)
    e2=tf.keras.layers.Dropout(0.5)(e1)
    e3=tf.keras.layers.LSTM(256)(e2)
    
    d1=tf.keras.layers.add([x2,e3])
    d2=tf.keras.layers.Dense(256,activation='relu')(d1)
    output=tf.keras.layers.Dense(vocab_size,activation='softmax')(d2)
    
    model=tf.keras.Model(inputs=[input1,input2],outputs=output)
    
    model.summary()
    return model
caption_model=caption_model(vocab_size,max_len)    


# In[ ]:


from tensorflow.keras.utils import plot_model

plot_model(caption_model)


# In[ ]:


checkpoint = tf.keras.callbacks.ModelCheckpoint( 'model.h5' , monitor= 'val_loss' , verbose=1,save_best_only=True, mode= 'min' )
caption_model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam')


# In[ ]:


callbacks("=[checkpoint],")
validation_data=([x_test_features, x_test_seq], y_test)


# In[ ]:


caption_model.fit([x_train_features, x_train_seq], y_train, epochs=10, verbose=2)


# In[ ]:


def convert_image(path):
    load_image=tf.keras.preprocessing.image.load_img(path,target_size=(299, 299))
    
    image=tf.keras.preprocessing.image.img_to_array(load_image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image = np.expand_dims(image, axis=0) 
    feature=model.predict(image,verbose=0)
    return feature  


# In[ ]:


def generate_text(tokenizer,max_len,caption_model,path,id_to_word):
    input_text='start '
    for i in range(max_len):
        seq=[]
        for i in input_text.split():
            seq.append(word_rank[i])
        print(seq)    
        padded=pad_sequences([seq],maxlen=max_len)
        print(padded.shape)
        #image_feature=convert_image(path)
        image_feature=features[keys[0]]
        y_hat=caption_model.predict([image_feature,padded],verbose=0)
        y_hat=np.argmax(y_hat)
        
        word=id_to_word[y_hat]
        
        input_text+=' '+word
        if word=='end':
            break

    return input_text   
        
generate_text=generate_text(tokenizer,max_len,caption_model,path,id_to_word)  


# In[ ]:


path=os.path.join(directory,list_images[0])


# In[ ]:


generate_text


# In[ ]:


word_rank['start']


# In[ ]:


id_to_word=dict()
for word,ind in word_rank.items():
    id_to_word[ind]=word


# In[ ]:


id_to_word[2030]


# In[ ]:




