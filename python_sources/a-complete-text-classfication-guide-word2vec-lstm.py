#!/usr/bin/env python
# coding: utf-8

# ## Text Classification on Amazon Fine Food Dataset with Google Word2Vec Word Embeddings in Gensim and training using LSTM In Keras.

# In[ ]:





# ## [Please star/upvote if u like it.]

# In[ ]:





# In[ ]:





# ### IMPORTING THE MODULES

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#nltk
import nltk

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for part-of-speech tagging
from nltk import pos_tag

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# BeautifulSoup libraray
from bs4 import BeautifulSoup 

import re # regex

#model_selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation
from sklearn.metrics import accuracy_score,roc_auc_score 
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix

#preprocessing scikit
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
 
#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))

#keras
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input,CuDNNLSTM,LSTM
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence

#gensim w2v
#word2vec
from gensim.models import Word2Vec


# In[ ]:





# ### LOADING THE DATASET

# In[ ]:


rev_frame=pd.read_csv(r'../input/Reviews.csv')


# In[ ]:


df=rev_frame.copy()


# In[ ]:


df.head()


# #### A brief description of the dataset from Overview tab on Kaggle : -
# 
# Data includes:
# - Reviews from Oct 1999 - Oct 2012
# - 568,454 reviews
# - 256,059 users
# - 74,258 products
# - 260 users with > 50 reviews

# In[ ]:





# ### DATA CLEANING AND PRE-PROCESSING

# #### Since here I am concerned with sentiment analysis I shall keep only the 'Text' and the 'Score' column.

# In[ ]:


df=df[['Text','Score']]


# In[ ]:


df['review']=df['Text']
df['rating']=df['Score']
df.drop(['Text','Score'],axis=1,inplace=True)


# In[ ]:


print(df.shape)
df.head()


# In[ ]:





# #### Let us now see if any of the column has any null values.

# In[ ]:


# check for null values
print(df['rating'].isnull().sum())
df['review'].isnull().sum()  # no null values.


# In[ ]:





# #### Note that there is no point for keeping rows with different scores or sentiment for same review text.  So I will keep only one instance and drop the rest of the duplicates.

# In[ ]:


# remove duplicates/ for every duplicate we will keep only one row of that type. 
df.drop_duplicates(subset=['rating','review'],keep='first',inplace=True) 


# In[ ]:


# now check the shape. note that shape is reduced which shows that we did has duplicate rows.
print(df.shape)
df.head()


# In[ ]:





# #### Let us now print some reviews and see if we can get insights from the text.

# In[ ]:


# printing some reviews to see insights.
for review in df['review'][:5]:
    print(review+'\n'+'\n')


# In[ ]:





# #### There is nothing much that I can figure out except the fact that there are some stray words and some punctuation that we have to remove before moving ahead.
# 
# **But note that if I remove the punctuation now then it will be difficult to break the reviews into sentences which is required by Word2Vec constructor in Gensim. So we will first break text into sentences and then clean those sentences. **

# In[ ]:





# #### Note that since we are doing sentiment analysis I will convert the values in score column to sentiment. Sentiment is 0 for ratings or scores less than 3 and 1 or  +  elsewhere.

# In[ ]:


def mark_sentiment(rating):
  if(rating<=3):
    return 0
  else:
    return 1


# In[ ]:


df['sentiment']=df['rating'].apply(mark_sentiment)


# In[ ]:


df.drop(['rating'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df['sentiment'].value_counts()


# As you can see the sentiment column now has sentiment of the corressponding product review.

# In[ ]:





# #### Pre-processing steps :
# 
# 1 ) First **removing punctuation and html tags** if any. note that the html tas may be present ast the data must be scraped from net.
# 
# 2) **Tokenize** the reviews into tokens or words .
# 
# 3) Next **remove the stop words and shorter words** as they cause noise.
# 
# 4) **Stem or lemmatize** the words depending on what does better. Herer I have yse lemmatizer.

# In[ ]:





# In[ ]:


# function to clean and pre-process the text.
def clean_reviews(review):  
    
    # 1. Removing html tags
    review_text = BeautifulSoup(review,"lxml").get_text()
    
    # 2. Retaining only alphabets.
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    
    # 3. Converting to lower case and splitting
    word_tokens= review_text.lower().split()
    
    # 4. Remove stopwords
    le=WordNetLemmatizer()
    stop_words= set(stopwords.words("english"))     
    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words]
    
    cleaned_review=" ".join(word_tokens)
    return cleaned_review


# In[ ]:





# #### Note that pre processing all the reviews is taking way too much time and so I will take only 100K reviews. To balance the class  I have taken equal instances of each sentiment.

# In[ ]:


pos_df=df.loc[df.sentiment==1,:][:50000]
neg_df=df.loc[df.sentiment==0,:][:50000]


# In[ ]:


pos_df.head()


# In[ ]:


neg_df.head()


# #### We can now combine reviews of each sentiment and shuffle them so that their order doesn't make any sense.

# In[ ]:


#combining
df=pd.concat([pos_df,neg_df],ignore_index=True)


# In[ ]:


print(df.shape)
df.head()


# In[ ]:


# shuffling rows
df = df.sample(frac=1).reset_index(drop=True)
print(df.shape)  # perfectly fine.
df.head()


# In[ ]:





# ### CREATING GOOGLE WORD2VEC WORD EMBEDDINGS IN GENSIM

# In this section I have actually created the word embeddings in Gensim. Note that I planed touse the pre-trained word embeddings like the google word2vec trained on google news corpusor the famous Stanford Glove embeddings. But as soon as I load the corressponding embeddings through Gensim the runtime dies and kernel crashes ; perhaps because it contains 30L words and which is exceeding the RAM on Google Colab.
# 
# Because of this ; for now I have created the embeddings by training on my own corpus.

# In[ ]:


# import gensim
# # load Google's pre-trained Word2Vec model.
# pre_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(r'drive/Colab Notebooks/amazon food reviews/GoogleNews-vectors-negative300.bin', binary=True) 


# In[ ]:





# #### First we need to break our data into sentences which is requires by the constructor of the Word2Vec class in Gensim. For this I have used Punk English tokenizer from the NLTK.

# In[ ]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences=[]
sum=0
for review in df['review']:
  sents=tokenizer.tokenize(review.strip())
  sum+=len(sents)
  for sent in sents:
    cleaned_sent=clean_reviews(sent)
    sentences.append(cleaned_sent.split()) # can use word_tokenize also.
print(sum)
print(len(sentences))  # total no of sentences


# #### Now let us print some sentences just to check iff they are in the correct fornat.

# In[ ]:


# trying to print few sentences
for te in sentences[:5]:
  print(te,"\n")


# In[ ]:





# ####  Now actually creating the word 2 vec embeddings.

# In[ ]:


import gensim
w2v_model=gensim.models.Word2Vec(sentences=sentences,size=300,window=10,min_count=1)


# #### Parameters: -
# 
# **sentences : ** The sentences we have obtained.
# 
# **size : ** The dimesnions of the vector used to represent each word.
# 
# **window : ** The number f words around any word to see the context.
# 
# **min_count : ** The minimum number of times a word should appear for its embedding to be formed or learnt.
# 

# In[ ]:


w2v_model.train(sentences,epochs=10,total_examples=len(sentences))


# In[ ]:





# #### Now can try some things with word2vec embeddings. Thanks to Gensim ;)

# In[ ]:


# embedding of a particular word.
w2v_model.wv.get_vector('like')


# In[ ]:


# total numberof extracted words.
vocab=w2v_model.wv.vocab
print("The total number of words are : ",len(vocab))


# In[ ]:


# words most similar to a given word.
w2v_model.wv.most_similar('like')


# In[ ]:


# similaraity b/w two words
w2v_model.wv.similarity('good','like')


# In[ ]:





# #### Now creating a dictionary with words in vocab and their embeddings. This will be used when we will be creating embedding matrix (for feeding to keras embedding layer).

# In[ ]:


print("The no of words :",len(vocab))
# print(vocab)


# In[ ]:


# print(vocab)
vocab=list(vocab.keys())


# In[ ]:


word_vec_dict={}
for word in vocab:
  word_vec_dict[word]=w2v_model.wv.get_vector(word)
print("The no of key-value pairs : ",len(word_vec_dict)) # should come equal to vocab size
  


# In[ ]:


# # just check
# for word in vocab[:5]:
#   print(word_vec_dict[word])


# In[ ]:





# ### PREPARING THE DATA FOR KERAS EMBEDDING LAYER.

# Now we have obtained the w2v embeddings. But there are a couple of steps required by Keras embedding layer before we can move on.
# 
# **Also note that since w2v embeddings have been made now ; we can preprocess our review column by using the function that we saw above.**

# In[ ]:


# cleaning reviews.
df['clean_review']=df['review'].apply(clean_reviews)


# #### We need to find the maximum lenght of any document or review in our case. WE will pad all reviews to have this same length.This will be required by Keras embedding layer. Must check [this](https://www.kaggle.com/rajmehra03/a-detailed-explanation-of-keras-embedding-layer) kernel on Kaggle for a wonderful explanation of keras embedding layer.

# In[ ]:


# number of unique words = 56379.

# now since we will have to pad we need to find the maximum lenght of any document.

maxi=-1
for i,rev in enumerate(df['clean_review']):
  tokens=rev.split()
  if(len(tokens)>maxi):
    maxi=len(tokens)
print(maxi)


# #### Now we integer encode the words in the reviews using Keras tokenizer. 
# 
# **Note that there two important variables: which are the vocab_size which is the total no of unique words while the second is max_doc_len which is the length of every document after padding. Both of these are required by the Keras embedding layer.**

# In[ ]:


tok = Tokenizer()
tok.fit_on_texts(df['clean_review'])
vocab_size = len(tok.word_index) + 1
encd_rev = tok.texts_to_sequences(df['clean_review'])


# In[ ]:


max_rev_len=1565  # max lenght of a review
vocab_size = len(tok.word_index) + 1  # total no of words
embed_dim=300 # embedding dimension as choosen in word2vec constructor


# In[ ]:


# now padding to have a amximum length of 1565
pad_rev= pad_sequences(encd_rev, maxlen=max_rev_len, padding='post')
pad_rev.shape   # note that we had 100K reviews and we have padded each review to have  a lenght of 1565 words.


# In[ ]:





# ### CREATING THE EMBEDDING MATRIX

# #### Now we need to pass the w2v word embeddings to the embedding layer in Keras. For this we will create the embedding matrix and pass it as 'embedding_initializer' parameter to the layer.
# 
# **The embedding matrix will be of dimensions (vocab_size,embed_dim) where the word_index of each word from keras tokenizer is its index into the matrix and the corressponding entry is its w2v vector ;)**
# 
# **Note that there may be words which will not be present in embeddings learnt by the w2v model. The embedding matrix entry corressponding to those words will be a vector of all zeros.**
# 
# **Also note that if u are thinkng why won't a word be present then it is bcoz now we have learnt on out own corpus but if we use pre-trained embedding then it may happen that some words specific to our dataset aren't present then in those cases we may use a fixed vector of zeros to denote all those words that earen;t present in th pre-trained embeddings. Also note that it may also happen that some words are not present ifu have filtered some words by setting min_count in w2v constructor.
#   **

# In[ ]:


# now creating the embedding matrix
embed_matrix=np.zeros(shape=(vocab_size,embed_dim))
for word,i in tok.word_index.items():
  embed_vector=word_vec_dict.get(word)
  if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
    embed_matrix[i]=embed_vector
  # if word is not found then embed_vector corressponding to that vector will stay zero.


# In[ ]:


# checking.
print(embed_matrix[14])


# In[ ]:





# ### PREPARING TRAIN AND VALIDATION SETS.

# In[ ]:


# prepare train and val sets first
Y=keras.utils.to_categorical(df['sentiment'])  # one hot target as required by NN.
x_train,x_test,y_train,y_test=train_test_split(pad_rev,Y,test_size=0.20,random_state=42)


# In[ ]:





# ### BUILDING A MODEL AND FINALLY PERFORMING TEXT CLASSIFICATION

# Having done all the pre-requisites we finally move onto make model in Keras .
# 
# **Note that I have commented the LSTM layer as including it causes the trainig loss to be stucked at a value of about 0.6932. I don;t know why ;(.**
# 
# **In case someone knows please comment below. **

# In[ ]:


from keras.initializers import Constant
from keras.layers import ReLU
from keras.layers import Dropout
model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=embed_dim,input_length=max_rev_len,embeddings_initializer=Constant(embed_matrix)))
# model.add(CuDNNLSTM(64,return_sequences=False)) # loss stucks at about 
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.50))
# model.add(Dense(16,activation='relu'))
# model.add(Dropout(0.20))
model.add(Dense(2,activation='sigmoid'))  # sigmod for bin. classification.


# In[ ]:





# #### Let us now print a summary of the model.

# In[ ]:


model.summary()


# In[ ]:


# compile the model
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3),loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


# specify batch size and epocj=hs for training.
epochs=5
batch_size=64


# In[ ]:


# fitting the model.
model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))


# #### Note that loss as well as val_loss is  is still deceasing. You can train for more no of epochs but I am not so patient ;)
# 
# **The final accuracy after 5 epochs is about 84% which is pretty decent.**

# In[ ]:





# ### FURTHER IDEAS : -
# 
# 1) ProductId and UserId can be used to track the general ratings of a given product and also to track the review patter of a particular user as if he is strict in reviwing or not.
#  
# 
# 2) Helpfulness feature may tell about the product. This is because gretare the no of people talking about reviews, the mre stronger or critical it is expected to be.
# 
# 3) Summary column can also give a hint.
# 
# 4) One can also try the pre-trained embeddings like Glove word vectors etc...
# 
# 5) Lastly tuning the n/w hyperparameters is always an option;).
# 
#  

# In[ ]:





# ## THE END!!!

# ## [Please star/upvote if it was helpful.]

# In[ ]:




