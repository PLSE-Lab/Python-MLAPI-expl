#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# In[ ]:


import os
for dirname, _, file_names in os.walk('/kaggle/input'):
    for file_name in file_names:
        print(os.path.join(dirname, file_name))


# In[ ]:


def loading_doc(file_name):
    file = open(file_name, 'r')
    txt = file.read()
    file.close()
    return txt

file_name = "/kaggle/input/flicker8k-dataset/Flickr8k_text/Flickr8k.token.txt"
doc = loading_doc(file_name)
print(doc[:300])


# In[ ]:


def loading_description(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping

descriptions = loading_description(doc)
print('Loaded: %d ' % len(descriptions))


# In[ ]:


list(descriptions.keys())[:5]


# In[ ]:


descriptions['1000268201_693b08cb0e']


# In[ ]:


descriptions['1001773457_577c3a7d70']


# In[ ]:


def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] =  ' '.join(desc)

clean_descriptions(descriptions)


# In[ ]:


descriptions['1000268201_693b08cb0e']


# In[ ]:


descriptions['1001773457_577c3a7d70']


# In[ ]:


def to_vocabulary(descriptions):

    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Original Vocabulary Size: %d' % len(vocabulary))


# In[ ]:


def save_descriptions(descriptions, file_name):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(file_name, 'w')
    file.write(data)
    file.close()

save_descriptions(descriptions, 'descriptions.txt')


# In[ ]:


def load_set(file_name):
    doc = loading_doc(file_name)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

file_name = '/kaggle/input/flicker8k-dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(file_name)
print('Dataset: %d' % len(train))


# In[ ]:


images = '/kaggle/input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'
img = glob.glob(images + '*.jpg')


# In[ ]:


train_images_file = '/kaggle/input/flicker8k-dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
train_img = []

for i in img: 
    if i[len(images):] in train_images:
        train_img.append(i)


# In[ ]:


test_images_file = '/kaggle/input/flicker8k-dataset/Flickr8k_text/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
test_img = []

for i in img:
    if i[len(images):] in test_images:
        test_img.append(i)


# In[ ]:


def load_clean_descriptions(file_name, dataset):    
    doc = loading_doc(file_name)
    descriptions = dict()
    for line in doc.split('\n'):        
        tokens = line.split()        
        image_id, image_desc = tokens[0], tokens[1:]        
        if image_id in dataset:            
            if image_id not in descriptions:
                descriptions[image_id] = list()            
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'            
            descriptions[image_id].append(desc)
    return descriptions

train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))


# In[ ]:


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# In[ ]:


model = InceptionV3(weights='imagenet')


# In[ ]:


model_new = Model(model.input, model.layers[-2].output)


# In[ ]:


def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) 
    return fea_vec


# In[ ]:


start = time()
encoding_train = {}
i=0;
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)
    print(i)
    i=i+1
print("Time taken in seconds =", time()-start)


# In[ ]:


import pickle


# In[ ]:


with open("encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)


# In[ ]:


start = time()
encoding_test = {}
i=0
for img in test_img:
    encoding_test[img[len(images):]] = encode(img)
    i=i+1
    print(i)
print("Time taken in seconds =", time()-start)


# In[ ]:


with open("encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)


# In[ ]:


train_features = load(open("encoded_train_images.pkl", "rb"))
print('Photos: train=%d' % len(train_features))


# In[ ]:


all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)


# In[ ]:


word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))


# In[ ]:


ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1


# In[ ]:


vocab_size = len(ixtoword) + 1
vocab_size


# In[ ]:


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)


# In[ ]:


def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            photo = photos[key+'.jpg']
            for desc in desc_list:
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0


# In[ ]:


glove_dir = '/kaggle/input/glove6b200d'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:


embedding_matrix.shape


# In[ ]:





# In[ ]:


inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)


# In[ ]:


model.summary()


# In[ ]:


model.layers[2]


# In[ ]:


model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


epochs = 10
number_pics_per_bath = 3
steps = len(train_descriptions)//number_pics_per_bath


# In[ ]:


for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model_' + str(i) + '.h5')


# In[ ]:


model.save_weights('model_9.h5')


# In[ ]:


model.load_weights('model_9.h5')


# In[ ]:


images = '/kaggle/input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'


# In[ ]:


with open("encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)


# In[ ]:


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# In[ ]:


for i in range(20):
    rn =  np.random.randint(0, 1000)
    pic = list(encoding_test.keys())[rn]
    image = encoding_test[pic].reshape((1,2048))
    x=plt.imread(images+pic)
    plt.imshow(x)
    plt.show()
    print("Greedy:",greedySearch(image))


# In[ ]:




