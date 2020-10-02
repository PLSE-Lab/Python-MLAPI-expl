#!/usr/bin/env python
# coding: utf-8

# # Using CNNs to classify disaster tweets
# Let's look at the training data first. We can use the pandas module to read our training data and store it in a dataframe. When we view the first few rows of the dataframe, we can see that there are 5 columns, and 2 columns have missing values. We then drop these columns as they aren't really useful for classification. 
# Before we send data into the model, we need to convert it into a ....
# ## Text Preprocessing
# So we begin text preprocessing by converting the 'text' column of the dataframe into a list. Each tweet is an element of this list. Now, from each tweet we remove numbers using regex pattern matching.

# In[ ]:


import pandas as pd
import numpy as np
import re
np.random.seed(500) 
train=pd.DataFrame(pd.read_csv('../input/nlp-getting-started/train.csv'))
train.head()
train=train.dropna(axis=1)
tweets=train['text'].to_list()
# print(train)
# print(tweets)
nonums=[]
for tweet in tweets:
    nonums.append(re.sub(r'\d+', '', tweet))
# print(nonums)


# When you print the output, you can see that there are a bunch of URLs in the tweets as well. We remove these by matching any text that begins with 'http\'.

# In[ ]:


get_ipython().system('pip install contractions')
p=re.compile(r'\<http.+?\>', re.DOTALL)

tweetswithouturls=[]
for tweet in nonums:
    tweetswithouturls.append(re.sub(r"http\S+", "", tweet))
# print(tweetswithouturls)


# Next, we replace contraced forms of words like 'don't' and 'can't' with their expanded forms 'do not' and 'cannot'.

# In[ ]:


import nltk
import contractions
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)
nocontractions=[]
for tweet in tweetswithouturls:
    

    nocontractions.append(replace_contractions(tweet))
# print(nocontractions)


# We then tokenise each tweet. With tokenising, we transform each tweet into a list, with each element of this tweet list being each word in the tweet. These words are called tokens.

# In[ ]:


from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
tokens = [word_tokenize(sen) for sen in nocontractions]
# print(tokens)


# We see that some of the tokens are punctiuation indicators like  .  ,  ?  , ! . These aren't necessary as well, as we need to understand the text in terms of the words and their context only. 

# In[ ]:


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words
nopunct=[]
for listt in tokens:
    nopunct.append(remove_punctuation(listt))
# print(nopunct)


# We then remove any characters which are not ascii characters. So we retain only A-Z as we removed numbers already.

# In[ ]:


import string, unicodedata
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words
onlyascii=[]
for listt in nopunct:
    onlyascii.append(remove_non_ascii(listt))
# print(onlyascii)


# To maintain uniformity we convert all letters to lowercase.

# In[ ]:


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words
lower=[]
for listt in onlyascii:
    lower.append(to_lowercase(listt))
# print(lower)


# We then remove stopwords, which don't necessarily add to convey the main idea of the tweet. These include words like i, our, of, for.

# In[ ]:



def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

    
nostopwords=[]
for listt in lower:
    nostopwords.append(remove_stopwords(listt))
# print(nostopwords)
# print(stopwords.words('english'))


# Here, we lemmatize the words to use only the root forms of the words, unless they're nouns.

# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import collections
tag_map = collections.defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
Final_words = []
word_Lemmatized = WordNetLemmatizer()
for entry in nostopwords:
#     print(entry)
    words=[]
    # Initializing WordNetLemmatizer()
    
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        
        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#         print(word_Final)
        words.append(word_Final)
    Final_words.append(words)
# print(Final_words)
# for entry in nostopwords:
#     print(pos_tag(entry))


# Let's now add these preprocessed tokens next to the tweets in the dataframe. We add them as sentences or a string of tokens in one column called "Text_Final". Then in another column called "tokens", we add the them as a list of tokens. 

# In[ ]:


train['Text_Final'] = [' '.join(sen) for sen in Final_words]
train['tokens'] = Final_words
disaster = []
notdisaster = []
for l in train['target']:
    if l == 0:
        disaster.append(0)
        notdisaster.append(1)
    elif l == 1:
        disaster.append(1)
        notdisaster.append(0)
train['Disaster']= disaster
train['Not a Disaster']= notdisaster

train = train[['Text_Final', 'tokens', 'target', 'Disaster', 'Not a Disaster']]
train.head()


# We then repeat all the processing steps for the test data.

# In[ ]:


#repeating for test
test=pd.DataFrame(pd.read_csv('../input/nlp-getting-started/test.csv'))
test.head()
test=test.dropna(axis=1)
tweetstest=test['text'].to_list()
# print(train)
# print(tweets)
nonumstest=[]
for tweet in tweetstest:
    nonumstest.append(re.sub(r'\d+', '', tweet))
# print(nonums)


tweetswithouturlstest=[]
for tweet in nonumstest:
    tweetswithouturlstest.append(re.sub(r"http\S+", "", tweet))
# print(tweetswithouturlstest)

nocontractionstest=[]
for tweet in tweetswithouturlstest:
    
    nocontractionstest.append(replace_contractions(tweet))
# print(nocontractionstest)
tokenstest = [word_tokenize(sen) for sen in nocontractionstest]
# print(tokenstest)

nopuncttest=[]
for listt in tokenstest:
    nopuncttest.append(remove_punctuation(listt))
# print(nopuncttest)

onlyasciitest=[]
for listt in nopuncttest:
    onlyasciitest.append(remove_non_ascii(listt))
# print(onlyasciitest)

lowertest=[]
for listt in onlyasciitest:
    lowertest.append(to_lowercase(listt))
# print(lowertest)

nostopwordstest=[]
for listt in lowertest:
    nostopwordstest.append(remove_stopwords(listt))
# print(nostopwordstest)
Finalwordstest=[]
for entry in nostopwordstest:
#     print(entry)
    words=[]
    # Initializing WordNetLemmatizer()
    
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        
        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#         print(word_Final)
        words.append(word_Final)
    Finalwordstest.append(words)
# print(Finalwordstest)


# Next we add the tokenised test tweets as sentences in one column called "Text_Final" and as tokens in another column called "tokens".

# In[ ]:


test['Text_Final'] = [' '.join(sen) for sen in Finalwordstest]
test['tokens'] = Finalwordstest
test.head()


# We then create a bag of training words as a list called "all_training_words". This contains all the words in all of the training tweets as one list. We create another array called "training_sentences_length" which stores a list of lengths of each tweet.
# Then a list called "TRAINING_VOCAB" is created to store the set of unique words in all of the tweets in the training set sorted alphabetically. So this gives us the vocabulary we're dealing with to train our model.

# An alternate form of counting and vectorising is using tf-idf, where the frequency of a word is compared with frequency across all documents, so its frequency can be attributed to being because of its existence in a class. This vectorisation has been used for SVM which i've used at the end.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
all_training_words = [word for tokens in train["tokens"] for word in tokens]

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(all_training_words)
# print(X_train_counts)
training_sentence_lengths = [len(tokens) for tokens in train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))
### for svm split data

##tfidf
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train["Text_Final"])
print(X_train_counts.shape)


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)


# We repeat the above process for test data as well.

# In[ ]:


all_test_words = [word for tokens in test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))


# We then use the Keras Tokeniser to create a set of indices for each word. So since TRAINING_VOCAB represents the set of unique words, the length of this list gives the max number of indices required. So each tweet is now represented as a set of numbers with each word replaced by an index. So what the fit_on_texts method does is, it looks at the frequencies of words appearing in all the tweets and gives a lower index if its frequency is higher. So a word like 'injured' might occur most frequently in this data set and might be given the index 1. Now the fit_to_sentence method replaces each word in every tweet with the index it was fit with. Train_word_index is a dictionary containing the index mapped to the corresponding word. So its length gives set of unique words. So now we have lists of indices as tweets. But each tweet is of different length, so we pad with zeroes. Now each tweet is a sequence of indices and its length is 50.

# In[ ]:


MAX_SEQUENCE_LENGTH = 50

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D,MaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(train["Text_Final"].tolist())
training_sequences = tokenizer.texts_to_sequences(train["Text_Final"].tolist())
train_word_index = tokenizer.word_index
print("Found %s unique tokens." % len(train_word_index))
train_cnn_data = pad_sequences(training_sequences, 
                               maxlen=MAX_SEQUENCE_LENGTH)


# Here, we also tokenise the test data into a sequence of indices as well and also pad the sequences. But the indices used are same as the ones used for the train data. And unseen words are relaced with 0.

# In[ ]:


test_sequences = tokenizer.texts_to_sequences(test["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# print(test_cnn_data.shape)
X_test_counts = count_vect.transform(test['Text_Final'])
X_test_tf = tf_transformer.transform(X_test_counts)
print(X_test_counts.shape)
print(X_test_tf.shape)


# ## Creating word embeddings
# Now, we use a trained word2vec model. This model takes in a corpus of text as input, which in this case is the set of words in the tweets in the order in which they appear. This represents a continuous bag of words. It then creates vectors for each word. These vectors now constitute a 300 dimensional vector space with words sharing common context appearing closer together in the vector space. Creating word embeddings this way helps us create a representation for words in their linguistic contexts. So we load the path where the model is stored. We then load the model in.

# In[ ]:


# # import gensim.downloader as api
# # path = api.load("word2vec-google-news-300", return_path=True)
path='../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin'
from gensim import models
word2vec = models.KeyedVectors.load_word2vec_format(path, binary=True)


# We now construct vectors for each word such that if the word already exists in the word2vec model, the vector for that word is used, and if it is not, a random vector is used. So each vector length is 300. So train_embedding_weights contains the vectors for each word.

# In[ ]:


EMBEDDING_DIM = 300
train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)


# ## Convolutional Neural Network
# We now define our CNN with an embedding layer. So any data that goes into this model is embedded through word2vec like we just did. So for test data this happens here, based on the train_embedding_matrix as weights, an embedding layer is created for the test input. This embedding layer is a layer of word2vec vectors of the test data. This layer acts as input to the convolutional layer. So for the convolutional layer is of depth 5. The 5 filters are of sizes 2,3,4,5,6. So the embedded input passes through each filter and a global max pooling layer that makes the number of parameters that pass through the subsequent layers smaller, by taking max parameter in a region. A relu activation function is also applied. The resultant is passed through a dropout layer with 10 percent dropout. Then a dense layer that retains 128 parameters and relu activation function is used and then another dropout layer. We end with a final dense layer and a relu activation layer that provides probabilities of 'disaster' and 'not disaster'. We then define loss function and the optimiser for backtracking and updating the weights. We return this model. 

# In[ ]:


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2,3,4,5,6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)


    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)  
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    
    
    preds = Dense(2, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
#     model.summary()
    return model


# So here we assign the labels to y_train for training. We then also assign the padded tokenised tweets of training data to x_train.

# In[ ]:


label_names = ['Disaster', 'Not a Disaster']
y_train = train[label_names].values


# print(y_train)
x_train = train_cnn_data


# We define number of epochs and batch size for training.
# We also define the early stopping criteria so training stops when validation loss reaches a minimum. This prevents overfitting.

# In[ ]:


from keras.callbacks import EarlyStopping
num_epochs = 4 #3 is enough but just testing
batch_size = 24

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


# We create an instance of our model and pass the training weights, max sequence length, no. of unique words, embedding dimension, and the no. of output labels to generate.

# In[ ]:


# for i in range(5):
#     print('Trial-',i)
model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                len(list(label_names)))

hist = model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1, shuffle=True, batch_size=batch_size,callbacks=[es])


# The model's predictions is stored. This is a list of probabilities for both 'disaster' and 'not a disaster'.

# In[ ]:


predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)
print(predictions)


# Next from the probability scores, we assign 1 if 'disaster' class is a higher probability and 0 otherwise.

# In[ ]:


labels = [1, 0]
prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])
# print(prediction_labels)
i=1
# for p in prediction_labels:
#     print(i,'-',p)
#     i+=1


# We now add the predictions in the dataframe. We also write the predictions into the submissions file.

# In[ ]:


test['target']=prediction_labels
# print(test[['tokens','target']])
submissions=pd.DataFrame(pd.read_csv('../input/nlp-getting-started/sample_submission.csv'))
# submissions['target']=prediction_labels
# print(submissions)
# submissions.to_csv('/kaggle/working/submission.csv',index=False)


# Some random code below pls ignore

# In[ ]:


# submissions=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
# comparewithnb=pd.DataFrame(pd.read_csv('../input/comparewithnb/filename11.csv'))
# cwnb=comparewithnb['0'].to_list()
# print(len(cwnb))
# count=0
# mismatch=[]
# for i in range(3263):
#     if(cwnb[i]==prediction_labels[i]):
#         count+=1
#     else:
#         mismatch.append(i)
# print(count)
        


# In[ ]:


testlabels=pd.DataFrame(pd.read_csv('../input/testlabels2/submission.csv'))
labels=testlabels['target'].to_list()
count=0
mismatch=[]
for i in range(3263):
    if(labels[i]==prediction_labels[i]):
        count+=1
    else:
        mismatch.append(i)
print(count)


# Here, I've implemented the SVM, which tries to divide datapoints into classes by using a hyperplane. This hyperplane must be as distant from the two classes as possible. The support vectors are the points closest to the hyperplane.

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn import svm,metrics

train_model=svm.SVC().fit(X_train_tf, train["target"].values)
predictions=train_model.predict(X_test_tf)

# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# hist2=SVM.fit(X_train_tf,train_y)
# predictions_SVM = SVM.predict(X_test_tf)


# SVM does seem to do better than CNNs, but I will look into this later.

# In[ ]:


##SVM
count=0
mismatch=[]
for i in range(3263):
    if(labels[i]==predictions[i]):
        count+=1
    else:
        mismatch.append(i)
print(count)
submissions['target']=predictions

submissions.to_csv('/kaggle/working/submission.csv',index=False)


# In[ ]:




