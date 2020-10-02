#!/usr/bin/env python
# coding: utf-8

# # t-SNE visualization of Amazon reviews with polarity based color-coding 
# 
# 

# In[ ]:



import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os


# In[ ]:


con = sqlite3.connect("../input/database.sqlite")


# In[ ]:


filtered_data=pd.read_sql_query("""select * from Reviews where Score !=3""",con)


# #Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
# 

# In[ ]:


def partition(x):
    if x<3:
        return 0
    return 1


# changing reviews with score less than 3 to be positive and vice-versa
# 

# In[ ]:


filtered_data['Score']=filtered_data.Score.map(partition)


# # Exploratory Data Analysis
# 
# ## Data Cleaning: Deduplication
# 

# In[ ]:


#Sorting data according to ProductId in ascending order

sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[ ]:



#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)


# In[ ]:


#Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# In[ ]:


final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[ ]:


#How many positive and negative reviews are present in our dataset?
final['Score'].value_counts()


# #  Text Preprocessing: Stemming, stop-word removal and Lemmatization.
# ### Now that we have finished deduplication our data requires some preprocessing before we go on further with analysis and making the prediction model.
# 
#  ###  Hence in the Preprocessing phase we do the following in the order below:-
# 
#  ###  Begin by removing the html tags
#  ###  Remove any punctuations or limited set of special characters like , or . or # etc.
# ### Check if the word is made up of english letters and is not alpha-numeric
# ### Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
# ### Convert the word to lowercase
# ###  Remove Stopwords
# ### Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)
# ### After which we collect the words used to describe positive and negative reviews

# In[ ]:


stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned


# In[ ]:


#Code for implementing step-by-step the checks mentioned in the pre-processing phase
# this code takes a while to run as it needs to run on 500k sentences.
if not os.path.isfile('final.sqlite'):
    final_string=[]
    all_positive_words=[] # store words from +ve reviews here
    all_negative_words=[] # store words from -ve reviews here.
    for i, sent in enumerate(tqdm(final['Text'].values)):
        filtered_sentence=[]
        #print(sent);
        sent=cleanhtml(sent) # remove HTMl tags
        for w in sent.split():
            # we have used cleanpunc(w).split(), one more split function here because consider w="abc.def", cleanpunc(w) will return "abc def"
            # if we dont use .split() function then we will be considring "abc def" as a single word, but if you use .split() function we will get "abc", "def"
            for cleaned_words in cleanpunc(w).split():
                if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                    if(cleaned_words.lower() not in stop):
                        s=(sno.stem(cleaned_words.lower())).encode('utf8')
                        filtered_sentence.append(s)
                        if (final['Score'].values)[i] == 1: 
                            all_positive_words.append(s) #list of all words used to describe positive reviews
                        if(final['Score'].values)[i] == 0:
                            all_negative_words.append(s) #list of all words used to describe negative reviews reviews
        str1 = b" ".join(filtered_sentence) #final string of cleaned words
        #print("***********************************************************************")
        final_string.append(str1)

    #############---- storing the data into .sqlite file ------########################
    final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 
    final['CleanedText']=final['CleanedText'].str.decode("utf-8")
        # store final table into an SQlLite table for future.
    conn = sqlite3.connect('final.sqlite')
    c=conn.cursor()
    conn.text_factory = str
    final.to_sql('Reviews', conn,  schema=None, if_exists='replace',                  index=True, index_label=None, chunksize=None, dtype=None)
    conn.close()
    
    
    with open('positive_words.pkl', 'wb') as f:
        pickle.dump(all_positive_words, f)
    with open('negitive_words.pkl', 'wb') as f:
        pickle.dump(all_negative_words, f)


# In[ ]:


if os.path.isfile('final.sqlite'):
    conn = sqlite3.connect('final.sqlite')
    final = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, conn)
    conn.close()
else:
    print("Please the above cell")


# ## Considering 3k Reviews.

# In[ ]:


final['Score']=final.Score.map({0:'Negative',1:'Positive'})
score=final['Score']
score=score[0:3000]
score.shape


# In[ ]:



final_data=final['CleanedText'].head(3000)
print(final_data.shape)


# # Bag of Words (BoW)

# In[ ]:


count_vect = CountVectorizer() #in scikit-learn
final_counts = count_vect.fit_transform(final_data.values)
print("the type of count vectorizer ",type(final_counts))
print("the shape of out text BOW vectorizer ",final_counts.get_shape())
print("the number of unique words ", final_counts.get_shape()[1])


# In[ ]:


from sklearn.preprocessing import StandardScaler
bow=final_counts.toarray()
std_data=StandardScaler().fit_transform(bow)
std_data.shape


# # TSNE

# In[ ]:


from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0,perplexity=50,n_iter=5000)
tsne = model.fit_transform(std_data)
tsne.shape


# In[ ]:


tsne = np.vstack((tsne.T, score)).T
tsne.shape


# In[ ]:



tsne_df = pd.DataFrame(data=tsne, columns=("Dim_1", "Dim_2", "Reviews"))

sns.FacetGrid(tsne_df, hue="Reviews", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# # TF-IDF

# In[ ]:


tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
final_tf_idf = tf_idf_vect.fit_transform(final_data.values)
print("the type of count vectorizer ",type(final_tf_idf))
final_tf_idf.shape


# In[ ]:



tf_idf=final_tf_idf.toarray()
tf_idf=StandardScaler().fit_transform(tf_idf)
tf_idf.shape


# In[ ]:


model = TSNE(n_components=2, random_state=0,perplexity=50,n_iter=5000)
tsne = model.fit_transform(tf_idf)
tsne.shape


# In[ ]:


tsne = np.vstack((tsne.T, score)).T
tsne.shape


# In[ ]:


tsne_df = pd.DataFrame(data=tsne, columns=("Dim_1", "Dim_2", "Reviews"))

sns.FacetGrid(tsne_df, hue="Reviews", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# # Word2Vec

# In[ ]:


i=0
list_of_sent=[]
for sent in final_data.values:
    list_of_sent.append(sent.split())
#print(final_data.values[0])
#print(list_of_sent[0])
len(list_of_sent)


# In[ ]:


w2v_model=Word2Vec(list_of_sent,min_count=5,size=50, workers=4)
w2v_words = list(w2v_model.wv.vocab)


# # Avg W2V

# In[ ]:


sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(list_of_sent): # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))


# In[ ]:


model = TSNE(n_components=2, random_state=0,perplexity=50,n_iter=5000)
tsne = model.fit_transform(sent_vectors)
tsne.shape


# In[ ]:


tsne = np.vstack((tsne.T, score)).T
tsne.shape


# In[ ]:


tsne_df = pd.DataFrame(data=tsne, columns=("Dim_1", "Dim_2", "Reviews"))

sns.FacetGrid(tsne_df, hue="Reviews", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# # TF-IDF weighted Word2Vec

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
model = TfidfVectorizer()
tf_idf_matrix = model.fit_transform(final_data.values)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))


# In[ ]:


type(tf_idf_matrix)


# In[ ]:


# TF-IDF weighted Word2Vec
tfidf_feat = model.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in tqdm(list_of_sent): # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]
            # to reduce the computation we are 
            # dictionary[word] = idf value of word in whole courpus
            # sent.count(word) = tf valeus of word in this review
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1


# In[ ]:


model = TSNE(n_components=2, random_state=0,perplexity=50,n_iter=5000)
tsne = model.fit_transform(tfidf_sent_vectors)
tsne.shape


# In[ ]:


tsne = np.vstack((tsne.T, score)).T
tsne.shape


# In[ ]:


tsne_df = pd.DataFrame(data=tsne, columns=("Dim_1", "Dim_2", "Reviews"))

sns.FacetGrid(tsne_df, hue="Reviews", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# # Obsevation:
# ##  * By observing plots, Most of reviews are positive.
# ##  * As we can see in above plots, BoW and TF-IDF are similar in structure. Most of the positive and nagtive points are             overlapped.
# ##  *  Avg W2V, TFIDF-W2V are similar vector representation.
# ##  *  As compared to BoW and TF-IDF,  Avg W2V and TFIDF-W2V is better in vector representation as we can see that              points are widely spreaded in Avg W2V, TFIDF-W2V.
# 

# 

# 
