#!/usr/bin/env python
# coding: utf-8

# # <a class="anchor" id=0.>Contents</a>
# * [1. Introduction](#1.)
# * [2. About Dataset](#2.)
# * [3. Loading Dataset](#3.)
# * [4. Processing Reviews](#4.)
# * * [4.1. Text Processing](#4.1.)
# * * * [4.1.1. Make String](#4.1.1.)
# * * * [4.1.2. Replace By](#4.1.2.)
# * * * [4.1.3. Lower String](#4.1.3.)
# * * * [4.1.4. Reduce Text Length](#4.1.4.)
# * * [4.2. Word Vector Processing](#4.2.)
# * * * [4.2.1 Vectorize Text](#4.2.1.)
# * * * [4.2.2 Filter Punctuation](#4.2.2.)
# * * * [4.2.3 Filter Nonalpha](#4.2.3.)
# * * * [4.2.4 Filter Stop Words](#4.2.4.)
# * * * [4.2.5 Filter Short Words](#4.2.5.)
# * * [4.3. Text Processor](#4.3.)
# * * [4.4. Create Vocabulary](#4.4.)
# * * * [4.4.1. Reduce Vocabulary Length](#4.4.1.)
# * * * [4.4.2. Filter Not In Vocabulary](#4.4.2.)
# * * * [4.4.3. Join With Space](#4.4.3.)
# * [5. Building Dataset](#5.)
# * * [5.1. Spliting Dataset into Train and Test Parts](#5.1.)
# * * [5.2. Process Train and Test Dataset](#5.2.)
# * * [5.3. Train Keras Tokenizer on Train Dataset](#5.3.)
# * * [5.4. Transform Train Dataset with Trained Tokenizer](#5.4.)
# * * [5.5. Transform Test Dataset with Trained Tokenizer](#5.5.)
# * [6. Using Keras Model](#6.)
# * * [6.1. Create Keras Model: Embedding + Conv1D](#6.1.)
# * * [6.2. Plot The Model](#6.2.)
# * * [6.3. Compile The Model](#6.3.)
# * * [6.4. Train The Model](#6.4.)
# * * [6.5. Evaluate The Model](#6.5.)
# * * [6.6. Save The Model](#6.6.)
# * [7. Make Prediction with Loaded Model](#7.) -- will be added soon
# * * [7.1. Load The Model](#7.1.)
# * * [7.2. Make Prediction](#7.2.)

# # <a class="anchor" id="1.">1. Introduction</a>
# 
# Natural Language Processing covers studies aimed at the automatic evaluation of texts and speeches. Natural language processing has been around for more than 50 years and has achieved success that can turn into commercial value through deep learning models.
# 
# NLP aims at engineering-oriented and experimental use of statistical methods and makes inferences from speech and texts which means language data by using statistical methods. NLP can take the language data as input and also produce the output of the language data as well as the estimation output.
# 
# Deep learning is a subfield of machine learning and the name given to the multi-layered architectures of artificial neural studies that attempt to model the workings of the human brain. The advantages of deep learning models for NLP studies can be simply divided into two; better performance and requiring less linguistics knowledge.
# 
# Text Classification, which defines the analysis of emotions from text data such as tweets, comments, articles, reviews, and the separation of emails from spam and non-spam, is an important field of study of natural language processing. Deep learning methods have come to the forefront in this field with significant achievements in the shortening data sets used for text classification.
# 
# In brief, the work done in this Kernel is as follows;
# * Movie comments in text files are loaded and data set is created
# * The data set is divided into training and test sets.
# * Processors are defined for processing data sets
# * Using the processor, first the training dataset and then the test data set were processed
# * Embedding + Conv1D deep learning model with Keras
# * Compiling, training, evaluation and saving of the model.
# * The saved model is re-loaded and used for classification on new reviews.
# 
# In order to understand, reuse and extend the code, text processor are defined and used with Object Oriented Programming approach.

# In[ ]:


from nltk.corpus import stopwords
import string, re
from collections import Counter
import wordcloud

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os

review_dataset_path="/kaggle/input/movie-review/movie_reviews/movie_reviews"
print(os.listdir(review_dataset_path))


# [Go to Content Menu](#0.)
# 
# # <a class="anchor" id="2.">2. About Dataset</a>

# [Go to Content Menu](#0.)
# 
# # <a class="anchor" id="3.">3. Loading The Dataset</a>

# In[ ]:


#Positive and negative reviews folder paths 
pos_review_folder_path=review_dataset_path+"/"+"pos"
neg_review_folder_path=review_dataset_path+"/"+"neg"


# In[ ]:


#Positive and negative file names
pos_review_file_names=os.listdir(pos_review_folder_path)
neg_review_file_names=os.listdir(neg_review_folder_path)


# In[ ]:


def load_text_from_textfile(path):
    file=open(path,"r")
    review=file.read()
    file.close()
    
    return review

def load_review_from_textfile(path):
    return load_text_from_textfile(path)


# [Go to Contents](#0.)

# In[ ]:


def get_data_target(folder_path, file_names, review_type):
    data=list()
    target =list()
    for file_name in file_names:
        full_path = folder_path + "/" + file_name
        review =load_review_from_textfile(path=full_path)
        data.append(review)
        target.append(review_type)
    return data, target


# In[ ]:


pos_data, pos_target=get_data_target(folder_path=pos_review_folder_path,
               file_names=pos_review_file_names,
               review_type="positive")
print("Positive data ve target builded...")
print("positive data length:",len(pos_data))
print("positive target length:",len(pos_target))


# In[ ]:


neg_data, neg_target = get_data_target(folder_path = neg_review_folder_path,
                                      file_names= neg_review_file_names,
                                      review_type="negative")
print("Negative data ve target builded..")
print("negative data length :",len(neg_data))
print("negative target length :",len(neg_target))


# In[ ]:


data = pos_data + neg_data
target_ = pos_target + neg_target
print("Positive and Negative sets concatenated")
print("data length :",len(data))
print("target length :",len(target_))


# In[ ]:


le = LabelEncoder()
le.fit(target_)
target = le.transform(target_)
print("Target labels transformed to number...")


# In[ ]:


print(le.inverse_transform([0,0,0,1,1,1]))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, stratify=target, random_state=24)
print("Dataset splited into train and test parts...")
print("train data length  :",len(X_train))
print("train target length:",len(y_train))
print()
print("test data length  :",len(X_test))
print("test target length:",len(y_test))


# In[ ]:


import seaborn as sns
fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(8,4),sharey=True)
axarr[0].set_title("Number of samples in train")
sns.countplot(x=y_train, ax=axarr[0])
axarr[1].set_title("Number of samples in test")
sns.countplot(x=y_test, ax=axarr[1])
plt.show()


# [Go to Content Menu](#0.)
# 
# # <a class="anchor" id="4.">4. Processing Reviews</a>
# ...

# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="4.1.">4.1. Text Processing</a>

# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.1.1.">4.1.1. Make String</a>

# In[ ]:


class MakeString:
    def process(self, text):
        return str(text)


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.1.2.">4.1.2. Replace By</a>

# In[ ]:


class ReplaceBy:
    def __init__(self, replace_by):
        #replace_by is a tuple contains pairs of replace and by characters.
        self.replace_by = replace_by
    def process(self, text):
        for replace, by in replace_by:
            text = text.replace(replace, by)
        return text


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.1.3.">4.1.3. Lower Text</a>

# In[ ]:


class LowerText:
    def process(self, text):
        return text.lower()


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.1.4.">4.1.4. Reduce Text Length</a>

# In[ ]:


class ReduceTextLength:
    def __init__(self, limited_text_length):
        self.limited_text_length = limited_text_length
    def process(self, text):
        return text[:self.limited_text_length]


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="4.2.">4.2. Word Vector Processing</a>

# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.2.1.">4.2.1. Vectorize Text</a>

# In[ ]:


class VectorizeText:
    def __init__(self):
        pass
    def process(self, text):
        return text.split()


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.2.2.">4.2.2. Filter Punctuation</a>

# In[ ]:


class FilterPunctuation:
    def __init__(self):
        print("Punctuation Filter created...")
    def process(self, words_vector):
        reg_exp_filter_rule=re.compile("[%s]"%re.escape(string.punctuation))
        words_vector=[reg_exp_filter_rule.sub("", word) for word in words_vector]
        return words_vector


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.2.3.">4.2.3. Filter Nonalpha</a>

# In[ ]:


class FilterNonalpha:
    def __init__(self):
        print("Nonalpha Filter created...")
    def process(self, words_vector):
        words_vector=[word for word in words_vector if word.isalpha()]
        return words_vector


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.2.4.">4.2.4. Filter Stop Words</a>

# In[ ]:


class FilterStopWord:
    def __init__(self, language):
        self.language=language
        print("Stopwords Filter created...")
    def process(self, words_vector):
        stop_words=set(stopwords.words(self.language))
        words_vector=[word for word in words_vector if not word in stop_words]
        return words_vector


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.2.5.">4.2.5. Filter Short Words</a>

# In[ ]:


class FilterShortWord:
    def __init__(self, min_length):
        self.min_length=min_length
        print("Short Words Filter created...")
    def process(self, words_vector):
        words_vector=[word for word in words_vector if len(word)>=self.min_length]
        return words_vector        


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="4.3.">4.3. Text Processor</a>

# In[ ]:


class TextProcessor:
    def __init__(self, processor_list):
        self.processor_list = processor_list
    def process(self, text):
        for processor in self.processor_list:
            text = processor.process(text)
        return text


# In[ ]:


text_len = np.vectorize(len)

text_lengths = text_len(X_train)


# In[ ]:


mean_review_length =int(text_lengths.mean())
print("Mean length of reviews   :",mean_review_length)    
print("Minimum length of reviews:",text_lengths.min())
print("Maximum length of reviews:",text_lengths.max())


# In[ ]:


sns.distplot(a=text_lengths)


# In[ ]:


makeString = MakeString()

replace_by = [("."," "), ("?"," "), (","," "), ("!"," "),(":"," "),(";"," ")]
replaceBy =ReplaceBy(replace_by=replace_by)

lowerText = LowerText()

FACTOR=8
reduceTextLength = ReduceTextLength(limited_text_length=mean_review_length*FACTOR)

vectorizeText = VectorizeText()
filterPunctuation = FilterPunctuation()
filterNonalpha = FilterNonalpha()
filterStopWord = FilterStopWord(language = "english")

min_length = 2
filterShortWord = FilterShortWord(min_length=min_length)
processor_list_1 = [makeString,
                      replaceBy,
                      lowerText,
                      reduceTextLength,
                      vectorizeText,
                      filterPunctuation,
                      filterNonalpha,
                      filterStopWord,
                      filterShortWord]


# In[ ]:


textProcessor1 = TextProcessor(processor_list=processor_list_1)


# In[ ]:


random_number=np.random.randint(0,len(X_train))
print("Original Review:\n",X_train[random_number][:500])
print("="*100)
print("Processed Review:\n",textProcessor1.process(text=X_train[random_number][:500]))


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="4.4.">4.4. Create Vocabulary</a>

# In[ ]:


class VocabularyHelper:
    def __init__(self, textProcessor):
        self.textProcessor=textProcessor
        self.vocabulary=Counter()
    def update(self, text):
        words_vector=self.textProcessor.process(text=text)
        self.vocabulary.update(words_vector)
    def get_vocabulary(self):
        return self.vocabulary


# In[ ]:


vocabularyHelper=VocabularyHelper(textProcessor=textProcessor1)
print("VocabularyHelper created...")


# In[ ]:


for text in X_train:
    vocabularyHelper.update(text)
vocabulary = vocabularyHelper.get_vocabulary()
print("Vocabulary filled...")


# In[ ]:


print("Length of vocabulary:",len(vocabulary))
n=10
print("{} most frequented words in vocabulary:{}".format(n, vocabulary.most_common(n)))


# In[ ]:


print("{} least frequented words in vocabulary:{}".format(n, vocabulary.most_common()[:-n-1:-1]))


# In[ ]:


vocabulary_list = " ".join([key for key, _ in vocabulary.most_common()])
plt.figure(figsize=(15, 35))
wordcloud_image = wordcloud.WordCloud(width = 1000, height = 1000, 
                background_color ='white', 
                #stopwords = stopwords, 
                min_font_size = 10).generate(vocabulary_list)


plt.xticks([])
plt.yticks([])
plt.imshow(wordcloud_image)


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.4.1.">4.4.1 Reduce Vocabulary Length</a>

# In[ ]:


min_occurence=2
vocabulary = Counter({key:value for key, value in vocabulary.items() if value>min_occurence})


# In[ ]:


print("{} least frequented words in vocabulary:{}".format(n, vocabulary.most_common()[:-n-1:-1]))


# In[ ]:


print("Length of vocabulary after removing words occurenced less than {} times:{}".format(min_occurence, len(vocabulary)))


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.4.2.">4.4.2 Filter Not In Vocabulary</a>

# In[ ]:


class FilterNotInVocabulary:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
    def process(self, words_vector):
        words_vector = [word for word in words_vector if word in self.vocabulary]
        return words_vector


# [Go to Content Menu](#0.)
# 
# ### <a class="anchor" id="4.4.3.">4.4.3. Join With Space</a>

# In[ ]:


class JoinWithSpace:
    def __init__(self):
        pass
    def process(self, words_vector):
        return " ".join(words_vector)


# In[ ]:


filterNotInVocabulary = FilterNotInVocabulary(vocabulary = vocabulary)
joinWithSpace = JoinWithSpace()
processor_list_2 = [makeString,
                    replaceBy,
                    lowerText,
                    reduceTextLength,
                    vectorizeText,
                    filterPunctuation,
                    filterNonalpha,
                    filterStopWord,
                    filterShortWord,
                    filterNotInVocabulary,
                    joinWithSpace
                   ]
textProcessor2=TextProcessor(processor_list = processor_list_2)


# In[ ]:


review = X_train[np.random.randint(0,len(X_train))]
print("Original Text:\n",review[:500])
processed_review = textProcessor2.process(review[:500])
print("="*100)
print("Processed Text:\n",processed_review)


# [Go to Content Menu](#0.)
# 
# # <a class="anchor" id="5.">5. Building Dataset</a>

# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="5.1.">5.1. Spliting Dataset into Train and Test Parts</a>

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target, random_state=24)
print("Dataset splited into train and test parts...")
print("train data length  :",len(X_train))
print("train target length:",len(y_train))
print()
print("test data length  :",len(X_test))
print("test target length:",len(y_test))


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="5.2.">5.2. Process Train and Test Datasets</a>

# In[ ]:


def  process_text(texts, textProcessor):
    processed_texts=list()
    for text in texts:
        processed_text = textProcessor.process(text)
        processed_texts.append(processed_text)
    return processed_texts


# In[ ]:


X_train_processed = process_text(texts=X_train, textProcessor=textProcessor2)
print("X_train processed...")


# In[ ]:


X_test_processed = process_text(texts=X_test, textProcessor=textProcessor2)
print("X_test processed...")


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="5.3.">5.3. Train Keras Tokenizer on Train Dataset </a>

# In[ ]:


from keras.preprocessing.text import Tokenizer
def create_and_train_tokenizer(texts):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
def encode_reviews(tokenizer, max_length, docs):
    encoded=tokenizer.texts_to_sequences(docs)
    
    padded=pad_sequences(encoded, maxlen=max_length, padding="post")
    
    return padded


# In[ ]:


tokenizer=create_and_train_tokenizer(texts = X_train)
vocab_size=len(tokenizer.word_index) + 1
print("Vocabulary size:", vocab_size)


# In[ ]:


max_length=max([len(row.split()) for row in X_train])
print("Maximum length:",max_length)


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="5.4.">5.4. Transform Trained Dataset with Trained Tokenizer </a>

# In[ ]:


X_train_encoded=encode_reviews(tokenizer, max_length, X_train_processed)
print("X_train encoded...")


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="5.5.">5.5. Transform Test Dataset with Trained Tokenizer </a>

# In[ ]:


X_test_encoded = encode_reviews(tokenizer, max_length, X_test_processed)
print("X_test encoded...")


# [Go to Content Menu](#0.)
# 
# # <a class="anchor" id="6.">6. Using Keras Model </a>

# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="6.1.">6.1. Create Keras Model: Embedding + Conv1D </a>

# In[ ]:


from keras import layers, models
def create_embedding_model(vocab_size, max_length):
    model=models.Sequential()
    model.add(layers.Embedding(vocab_size, 100, input_length=max_length))
    model.add(layers.Conv1D(32, 8, activation="relu"))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(1,  activation="sigmoid"))   
    return model


# In[ ]:


embedding_model = create_embedding_model(vocab_size=vocab_size, max_length=max_length)
embedding_model.summary()


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="6.2.">6.2. Plot The Model </a>

# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="6.3.">6.3. Compile The  Model </a>

# In[ ]:


embedding_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="6.4.">6.4. Train The Model </a>

# In[ ]:


from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor="val_accuracy", patience=1)
modelHistory = embedding_model.fit(X_train_encoded, 
                                   y_train, 
                                   validation_data=(X_test_encoded, y_test),
                                   epochs=10, 
                                   callbacks=[earlyStopping])
print("Model trained...")


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="6.5.">6.5. Evaluate The  Model </a>

# In[ ]:


_, acc = embedding_model.evaluate(X_train_encoded, y_train, verbose=0)
print("Train accuracy:{:.2f}".format(acc*100))


# In[ ]:


_,acc= embedding_model.evaluate(X_test_encoded, y_test, verbose=0)
print("Test accuracy:{:.2f}".format(acc*100))


# [Go to Content Menu](#0.)
# 
# ## <a class="anchor" id="6.6.">6.6. Save The  Model </a>

# In[ ]:


embedding_model.save("embedding_model.h5")
print(os.listdir("/kaggle/working"))


# In[ ]:


loaded_model=models.load_model("/kaggle/working/embedding_model.h5")
print("Saved model loaded...")


# In[ ]:




