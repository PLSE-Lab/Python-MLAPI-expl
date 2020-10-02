#!/usr/bin/env python
# coding: utf-8

# # Text Classification using different Models
# 
# In this notebook I'll compare the accuracies of different classification models. I will try to keep this notebook clear & concise as I feel it will be a huge one :-), so without wasting anytime lets get straight to it !

# # Libraries

# In[ ]:


#Install Contractions library
get_ipython().system('pip install contractions -q')


# In[ ]:


#Generic 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re,string,unicodedata
from string import punctuation
import contractions #import contractions_dict
import chardet
import matplotlib.pyplot as plt
import seaborn as sns

# SK Learn Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, model_selection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Keras Libraries
from keras.models import Model,Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

#NLTK Libraries
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

#Warnings
import warnings
warnings.filterwarnings("ignore")

#Garbage Collection
import gc

#downloading wordnet/punkt dictionary
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

#WordCloud Generator
from wordcloud import WordCloud,STOPWORDS

#Tabulation Library
from tabulate import tabulate


# # Load Data

# In[ ]:


url = '../input/binary-reviews-from-imdb/pos_neg_reviews.csv'

# to find encoding of the file. 
with open(url, 'rb') as f:
    result = chardet.detect(f.read()) 

# loading the data with the detected encoding from above
raw_data = pd.read_csv(url, header='infer', encoding=result['encoding'])


# In[ ]:


#backup of raw data
raw_data_bkp = raw_data.copy()


# # Explore & Prep Data

# In[ ]:


# creating a new dataframe with specific columns
data = raw_data[['text','polarity']]


# In[ ]:


print("Dataset Shape: ", data.shape)


# In[ ]:


#Checking for null/missing value
data.isnull().sum()


# In[ ]:


#Checking the records per polarity
data.groupby('polarity').size()


# In[ ]:


#Encode the Polarity Label to convert it into numerical values
lab_enc = LabelEncoder()

#Applying to the dataset
data['polarity'] = lab_enc.fit_transform(data['polarity'])


# In[ ]:


# Data Preparation Function 

def data_prep(text):
    
    #Lowering the case
    text = text.lower()

    # Stripping leading spaces (if any)
    text = text.strip()
        
    # Remove Punctuations
    for punctuations in punctuation:
        text = text.replace(punctuations, '')
    
    # Remove macrons & accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Expand contractions
    text = contractions.fix(text)

    # Remove special characters & retain alphabets
    pattern = r'[^a-zA-z0-9\s\w\t\.]'
    text = re.sub(pattern, '', text)
    
    # Stopword removal
    stopword_list = set(stopwords.words('english'))
    tokenizer = ToktokTokenizer()
    
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    text = ' '.join(filtered_tokens)    
    
    #Normalize the text
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    
    return text
    
#Applying the Data Prep function to 'text' column
data['text'] = data['text'].apply(data_prep)


# The text data is now cleaned, prep'd and normalized. This data is now ready for classification, but before that this data needs to be backed up.

# In[ ]:


#Backup of cleaned & normalized text data
norm_data_bkup = data.copy()


# # Data Split

# In[ ]:


#Splitting the normalized data into train [90%] & test[10%] data
x_train,x_test,y_train,y_test = train_test_split(data['text'], data.polarity, test_size=0.1, random_state=0)


# In[ ]:


#Inspect the split dataset
info = [ ["Training" , x_train.shape[0]], ["Testing", x_test.shape[0]]  ]
print(tabulate(info, headers=['Dataset','Shape']))


# In[ ]:


#List of store model accuracy
mod_acc = []


# # Multinomial Naive Bayes Classifier [SKLearn]

# In[ ]:


# Constructing Pipeline to Extract Features, Transform Count Matrix & then build/train Model

pipe = Pipeline([('vect', CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))),
                 ('tfidf', TfidfTransformer()),
                 ('model', MultinomialNB()) ])

mnb_model = pipe.fit(x_train, y_train)


# In[ ]:


# Making Prediction on Test Data & Calculating Accuracy
mnb_pred = mnb_model.predict(x_test)
print("Multinomial Naive Bayes Model Accuracy: ",'{:.2%}'.format(accuracy_score(y_test,mnb_pred)))


# In[ ]:


mnb_acc = accuracy_score(y_test,mnb_pred)
mod_acc.extend([["Multinomial Naive Bayes",'{:.2%}'.format(mnb_acc)]]) 


# # Support Vector Machines [SKLearn]

# In[ ]:


# Constructing Pipeline to Extract Features, Transform Count Matrix & then build/train Model

pipe = Pipeline([('vect', CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))),
                 ('tfidf', TfidfTransformer()),
                 ('model', LinearSVC()) ])

svm_model = pipe.fit(x_train, y_train)


# In[ ]:


# Making Prediction on Test Data & Calculating Accuracy
svm_pred = svm_model.predict(x_test)
print("Support Vector Machines Model Accuracy: ",'{:.2%}'.format(accuracy_score(y_test,svm_pred)))


# In[ ]:


svm_acc = accuracy_score(y_test,svm_pred)
mod_acc.extend([["Support Vector Machines",'{:.2%}'.format(svm_acc)]]) 


# # LSTM [Keras]

# In[ ]:


# Creating a RNN Function

max_words = 500
max_len = 100

tokn = Tokenizer(num_words=max_words) 
tokn.fit_on_texts(x_train)
sequences = tokn.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


# In[ ]:


#Instantiating the RNN Model
rnn_mod = RNN()

#Compile Model
rnn_mod.compile(loss='binary_crossentropy', optimizer= 'Nadam' , metrics=['accuracy'] )


# In[ ]:


#Model Summary
rnn_mod.summary()


# In[ ]:


#Train the model
rnn_mod.fit(sequences_matrix, y_train, batch_size=256, epochs=5,
            validation_split=0.1)
           


# In[ ]:


#Calculating Accuracy
test_sequences = tokn.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = rnn_mod.evaluate(test_sequences_matrix,y_test)
print('LSTM (Keras) Accuracy: ','{:.2%}'.format(accr[1]))


# In[ ]:


mod_acc.extend([["LSTM (Keras)",'{:.2%}'.format(accr[1])]]) 


# In[ ]:


#Tabulating the results:
print (tabulate(mod_acc, headers=["Models", "Accuracy"]))

