__author__ = 'Ahmed Hani Ibrahim'

"""
    This method of vectorization based on the analysis made on the characters and words level. Check it from here
    
    https://www.kaggle.com/ahmedhanibrahim/word-and-character-level-analysis
"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import numpy as np
import copy as cp

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


class DataManager(object):
    def __init__(self, file_name):
        self.__file_name = file_name
        self.__data = None

        self.__read()

    def __read(self):
        self.__data = pd.read_csv(str(self.__file_name), encoding='latin-1')
        self.__data = self.__data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
        self.__sentences = self.__data['v2']
        self.__labels = self.__data['v1']

    def count(self):
        print(self.__data.v1.value_counts())

        sb.countplot(x='v1', data=self.__data)
        plt.show()

    def most_frequent_character_in_spam(self):
        spams = self.__data.loc[self.__data['v1'] == 'spam']
        spams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(spams['v2'])))
        spams = list(map(lambda v: v.split(), spams))
        spams = list(chain.from_iterable(spams))
        spams_characters_frequencies = list(map(lambda v: Counter(v), spams))
        all_counter = spams_characters_frequencies[0]

        for i in range(1, len(spams_characters_frequencies)):
            all_counter += spams_characters_frequencies[i]

        print("Characters frequency in spams\n")
        print(all_counter)

        plt.bar(range(len(all_counter)), all_counter.values(), align='center')
        plt.xticks(range(len(all_counter)), all_counter.keys())

        plt.show()

    def most_frequent_character_in_legit(self):
        spams = self.__data.loc[self.__data['v1'] == 'ham']
        spams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(spams['v2'])))
        spams = list(map(lambda v: v.split(), spams))
        spams = list(chain.from_iterable(spams))
        spams_characters_frequencies = list(map(lambda v: Counter(v), spams))
        all_counter = spams_characters_frequencies[0]

        for i in range(1, len(spams_characters_frequencies)):
            all_counter += spams_characters_frequencies[i]

        print("Characters frequency in hams\n")
        print(all_counter)

        plt.bar(range(len(all_counter)), all_counter.values(), align='center')
        plt.xticks(range(len(all_counter)), all_counter.keys())

        plt.show()

    def most_frequent_characters(self):
        spams = self.__data.loc[self.__data['v1'] == 'spam']
        hams = self.__data.loc[self.__data['v1'] == 'ham']

        spams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(spams['v2'])))
        hams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(hams['v2'])))

        spams = list(map(lambda v: v.split(), spams))
        hams = list(map(lambda v: v.split(), hams))

        spams = list(chain.from_iterable(spams))
        hams = list(chain.from_iterable(hams))

        spams_characters_frequencies = list(map(lambda v: Counter(v), spams))
        hams_characters_frequencies = list(map(lambda v: Counter(v), hams))

        spams_all_counter = spams_characters_frequencies[0]
        hams_all_counter = hams_characters_frequencies[0]

        for i in range(1, len(spams_characters_frequencies)):
            spams_all_counter += spams_characters_frequencies[i]

        print(spams_all_counter)

        for i in range(1, len(hams_characters_frequencies)):
            hams_all_counter += hams_characters_frequencies[i]

        max_length = max(len(spams_all_counter), len(hams_all_counter))

        plt.bar(range(len(spams_all_counter)), spams_all_counter.values(), align='center', color='red')
        plt.bar(range(len(hams_all_counter)), hams_all_counter.values(), align='center', color='blue')
        plt.xticks(range(max_length), spams_all_counter.keys() if max_length == len(spams_all_counter) else hams_all_counter.keys())

        plt.show()

    def average_text_length(self):
        spams = self.__data.loc[self.__data['v1'] == 'spam']
        hams = self.__data.loc[self.__data['v1'] == 'ham']

        spams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(spams['v2'])))
        hams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(hams['v2'])))

        spams_average_length = sum(len(text) for text in spams) / len(spams)
        hams_average_length = sum(len(text) for text in hams) / len(hams)

        print("Spams average text length: " + str(spams_average_length))
        print("Hams average text length: " + str(hams_average_length))

    def get_text(self):
        return list(self.__data['v2'])

    def get_labels(self):
        return list(self.__data['v1'])


"""
Vectorization based on this paper
http://www.aclweb.org/anthology/P15-2081
"""

class Vectorizer(object):
    def __init__(self, sentences):
        self.__sentences = sentences
        self.__char2onehot = {}
        self.__idx2char = {}

        self.__get_chars_vector()

    def __get_chars_vector(self):
        sentences_tokens = list(map(lambda v: v.split(), self.__sentences))
        sentences_tokens = list(chain.from_iterable(sentences_tokens))
        characters_frequencies = list(map(lambda v: Counter(v), sentences_tokens))
        all_counter = characters_frequencies[0]

        for i in range(1, len(characters_frequencies)):
            all_counter += characters_frequencies[i]

        one_hot = np.asarray([0] * len(all_counter.keys()))

        for i, char in enumerate(all_counter.keys()):
            temp_one_hot = cp.copy(one_hot)
            temp_one_hot[i] = 1

            self.__char2onehot[char] = temp_one_hot
            self.__idx2char[i] = char

    def get_char2onehot(self):
        return self.__char2onehot

    def get_char_vector(self, char):
        return self.__char2onehot[char]

    def text_to_vec(self, text, alpha=0.3):
        text_tokens = text.strip().split()
        z = np.asarray([0] * len(self.__char2onehot))

        for word in text_tokens:
            for char in word:
                z = np.add(np.multiply(z, alpha), self.__char2onehot[char] if char in self.__char2onehot else [0]*len(self.__char2onehot))

        return z

dm = DataManager('../input/spam.csv')
dm.most_frequent_character_in_spam()
dm.most_frequent_character_in_legit()
dm.most_frequent_characters()
dm.average_text_length()

sentences, labels = dm.get_text(), dm.get_labels()
labels = list(map(lambda v: 0 if v == 'ham' else 1, labels))
vectorizer = Vectorizer(sentences)

sentences_features = []

for sentence in sentences:
    sentence_vector = vectorizer.text_to_vec(sentence, alpha=0.3)
    sentences_features.append(sentence_vector)

train_x, train_y = sentences_features[0:5000], labels[0:5000]
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

test_x, test_y = sentences_features[5000:], labels[5000:]
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)

np.savetxt('train_x.txt', train_x)
np.savetxt('train_y.txt', train_y)
np.savetxt('test_x.txt', test_x)
np.savetxt('test_y.txt', test_y)

train_x, train_y = np.loadtxt('train_x.txt'), np.loadtxt('train_y.txt')
features_length = train_x.shape[1]

test_x, test_y = np.loadtxt('test_x.txt'), np.loadtxt('test_y.txt')

model = Sequential()
model.add(Dense(100, input_shape=(features_length,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=100, epochs=50)
preds = model.predict_classes(test_x)
score = model.evaluate(test_x, test_y)

print(score)