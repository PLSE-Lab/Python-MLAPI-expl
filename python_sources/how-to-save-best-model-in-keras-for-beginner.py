#Modified by Dexter Cheong

#Date: 26-4-2018

#Thursday

#This is a kernel on how to save the best model in keras



import numpy as np

import re

import itertools

from collections import Counter

import pandas as pd



from keras.optimizers import Adam

from keras.models import Model, load_model



from keras.layers.core import Reshape, Flatten

from keras.callbacks import ModelCheckpoint

from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout, concatenate



import os



file_path = "../output"

directory = os.path.dirname(file_path)



try:

    os.stat(directory)

except:

    os.mkdir(directory) 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



dtype = {

    'id': str,

    'teacher_id': str,

    'teacher_prefix': str,

    'school_state': str,

    'project_submitted_datetime': str,

    'project_grade_category': str,

    'project_subject_categories': str,

    'project_subject_subcategories': str,

    'project_title': str,

    'project_essay_1': str,

    'project_essay_2': str,

    'project_essay_3': str,

    'project_essay_4': str,

    'project_resource_summary': str,

    'teacher_number_of_previously_posted_projects': int,

    'project_is_approved': np.uint8,

}



def clean_str(string):

    """

    Tokenization/string cleaning for datasets.

    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

    string = re.sub(r"\'s", " \'s", string)

    string = re.sub(r"\'ve", " \'ve", string)

    string = re.sub(r"n\'t", " n\'t", string)

    string = re.sub(r"\'re", " \'re", string)

    string = re.sub(r"\'d", " \'d", string)

    string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r",", " , ", string)

    string = re.sub(r"!", " ! ", string)

    string = re.sub(r"\(", " \( ", string)

    string = re.sub(r"\)", " \) ", string)

    string = re.sub(r"\?", " \? ", string)

    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r" \- ", " ", string)

    string = re.sub(r"\"", "", string)

    return string.strip().lower()





def load_data_and_labels():

    """

    Loads polarity data from files, splits the data into words and generates labels.

    Returns split sentences and labels.

    """

    # Load data from files

    positive_examples = list(open("../input/positive-and-negative-sentences/positive.txt", "r", encoding='"utf-8', errors='replace').readlines())

    positive_examples = [s.strip() for s in positive_examples]

    negative_examples = list(open("../input/positive-and-negative-sentences/negative.txt", "r", encoding='"utf-8', errors='replace').readlines())

    negative_examples = [s.strip() for s in negative_examples]

    # Split by words

    x_text = positive_examples + negative_examples

    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels

    positive_labels = [[0, 1] for _ in positive_examples]

    negative_labels = [[1, 0] for _ in negative_examples]

    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]





def pad_sentences(sentences, padding_word="<PAD/>"):

    """

    Pads all sentences to the same length. The length is defined by the longest sentence.

    Returns padded sentences.

    """

    sequence_length = max(len(x) for x in sentences)

    padded_sentences = []

    for i in range(len(sentences)):

        sentence = sentences[i]

        num_padding = sequence_length - len(sentence)

        new_sentence = sentence + padding_word

        for j in range(num_padding-1):

            new_sentence = new_sentence + padding_word

        padded_sentences.append(new_sentence)

    return padded_sentences





def build_vocab(sentences):

    """

    Builds a vocabulary mapping from word to index based on the sentences.

    Returns vocabulary mapping and inverse vocabulary mapping.

    """

    # Build vocabulary

    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word

    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    vocabulary_inv = list(sorted(vocabulary_inv))

    # Mapping from word to index

    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]





def build_input_data(sentences, labels, vocabulary):

    """

    Maps sentences and labels to vectors based on a vocabulary.

    """

    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])

    y = np.array(labels)

    return [x, y]





def load_data():

    """

    Loads and preprocessed data for the dataset.

    Returns input vectors, labels, vocabulary, and inverse vocabulary.

    """

    # Load and preprocess data

    sentences, labels = load_data_and_labels()

    sentences_padded = pad_sentences(sentences)

    vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]







def load_embedding_vectors_glove(vocabulary, vector_size):

    # load embedding_vectors from the glove

    # initial matrix with random uniform

    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))

    GLOVE_DIR = ""

    f = open(os.path.join(GLOVE_DIR, '../input/glove6b300dtxt/glove.6B.300d.txt'))

    for line in f:

        values = line.split()

        word = values[0]

        vector = np.asarray(values[1:], dtype="float32")

        idx = vocabulary.get(word)

        if idx != 0:

            embedding_vectors[idx] = vector

    f.close()

    return embedding_vectors



print ('Loading data')

x, y, vocabulary, vocabulary_inv = load_data()



split = 10060



X_train = x[:split]

y_train = y[:split]



X_test = x[split:]

y_test = y[split:]



sequence_length = 100

vocabulary_size = len(vocabulary_inv)



print("Vocabulary Size : ", vocabulary_size)

print("Sequence Length : ", sequence_length)

print("X Size : ", x.shape)

print("Y Length : ", y.shape)



embedding_dim = 300

filter_sizes = [2,3,4]

num_filters = 300

drop = 0.55



nb_epoch = 2

batch_size = 100



inputs = Input(shape=(100,), dtype='int32')

embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)

reshape = Reshape((sequence_length,embedding_dim,1))(embedding)



conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)



maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)

maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)

maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)



merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2])



flatten = Flatten()(merged_tensor)



dropout = Dropout(drop)(flatten)

output = Dense(output_dim=2, activation='softmax')(flatten)



# this creates a model that includes

model = Model(input=inputs, output=output)



checkpoint = ModelCheckpoint('output/{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training



model.save("best.h5s")
