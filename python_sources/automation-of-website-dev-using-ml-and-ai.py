# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense, Flatten
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# Input data files are available in the "../input/" directory.
images = []
path='../input/images/'
for files in glob.glob(path+'*.jpg'):
    images.append(img_to_array(load_img(files, target_size=(299, 299))))
images = np.array(images, dtype=float)
images = preprocess_input(images)
# Run the images through inception-resnet and extract the features without the classification layer
IR2 = InceptionResNetV2(weights='imagenet', include_top=False)
features = IR2.predict(images)

# We will cap each input sequence to 100 tokens
max_caption_len = 100
# Initialize the function that will create our vocabulary 
tokenizer = Tokenizer(filters='', split=" ", lower=False)

# Read a document and return a string
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# Load all the HTML files
X = []
path='../input/__html/'
for filename in glob.glob(path+'*.html'):
    X.append(load_doc(filename))

# Create the vocabulary from the html files
tokenizer.fit_on_texts(X)

# Add +1 to leave space for empty words
vocab_size = len(tokenizer.word_index) + 1
# Translate each word in text file to the matching vocabulary index
sequences = tokenizer.texts_to_sequences(X)
# The longest HTML file
#Changes made here
max_length=0
for s in sequences:
    if len(s)>max_length:
        max_length= len(s)
#max_length = max(len(s) for s in sequences)

# Intialize our final input to the model
X, y, image_data = list(), list(), list()
for img_no, seq in enumerate(sequences):
    for i in range(1, len(seq)):
        # Add the entire sequence to the input and only keep the next word for the output
        in_seq, out_seq = seq[:i], seq[i]
        # If the sentence is shorter than max_length, fill it up with empty words
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        # Map the output to one-hot encoding
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # Add and image corresponding to the HTML file
        image_data.append(features[img_no])
        # Cut the input sentence to 100 tokens, and add it to the input data
        X.append(in_seq[-100:])
        y.append(out_seq)

X, y, image_data = np.array(X), np.array(y), np.array(image_data)

# Create the encoder
image_features = Input(shape=(8, 8, 1536, ))
image_flat = Flatten()(image_features)
image_flat = Dense(128, activation='relu')(image_flat)
ir2_out = RepeatVector(max_caption_len)(image_flat)

language_input = Input(shape=(max_caption_len,))
language_model = Embedding(vocab_size, 200, input_length=max_caption_len)(language_input)
language_model = LSTM(256, return_sequences=True)(language_model)
language_model = LSTM(256, return_sequences=True)(language_model)
language_model = TimeDistributed(Dense(128, activation='relu'))(language_model)

# Create the decoder
decoder = concatenate([ir2_out, language_model])
decoder = LSTM(512, return_sequences=False)(decoder)
decoder_output = Dense(vocab_size, activation='softmax')(decoder)

# Compile the model
model = Model(inputs=[image_features, language_input], outputs=decoder_output)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit([image_data, X], y, batch_size=64, shuffle=False, epochs=5)