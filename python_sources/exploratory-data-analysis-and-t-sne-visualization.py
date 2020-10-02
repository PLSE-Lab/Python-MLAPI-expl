#!/usr/bin/env python
# coding: utf-8

# # Amazon fine food review

# Downloaded from https://www.kaggle.com/snap/amazon-fine-food-reviews
# The amazon fine food review dataset consists of review of fine foods on amazon .
# - Number of reviews: 568,454
# - Number of users: 256,059
# - Number of products: 74,258
# - Timespan: Oct 1999 - Oct 2012
# - Number of Attributes/Columns in data: 10
# 
# Attribute Information:
# 
# - Id
# - ProductId - unique identifier for the product
# - UserId - unqiue identifier for the user
# - ProfileName
# - HelpfulnessNumerator - number of users who found the review helpful
# - HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
# - Score - rating between 1 and 5
# - Time - timestamp for the review
# - Summary - brief summary of the review
# - Text - text of the review
# ### Objective
# To determine whether the review given is positive(score rating of 4 or 5 stars) or negative(score rating of 1 or 2 stars).

# ## Loading the data
# The dataset is available in two formats i.e.
# 1. sqlite database
# 2. .csv file
# We will load data using SQLITE database as it is easier to give commands and visualise the data.
# 
# Here we only want to get the sentiment of th review(positive or negative). We will purposely remove the score of 3 and give a positive rating if the score is more than 3 and a negative rating if it is less than 3.

# In[ ]:


import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns


# # [1]. Reading the data
# 

# In[ ]:


#loading dataset from sqlite database
con = sqlite3.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite') 

filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE score != 3""", con )
#giving a postive rating to data points having score more than 3 and a negative rating for score less than 3.
#We are removing data points where score is equal to 3 as it cannot be predicted as positive or negative.
def partition(x) :
    if x < 3 :
        return 'Positive'
    else :
        return 'Negative'


# In[ ]:


#replacing score column values with positve and negative values
actualscore = filtered_data['Score'] 
positiveNegative = actualscore.map(partition)
filtered_data['Score'] = positiveNegative
print("No of data points", filtered_data.shape)
print(filtered_data.head())


# ## Exploratory Data Analysis
# ## [2] Data Cleaning(Deduplication)
# It is observed that the data has many duplicate entries. Hence it is necessary to remove dupliactes in order to get unbiased results.

# In[ ]:


#sorting data according to ProductId
sorted_data= filtered_data.sort_values('ProductId', axis =0 , ascending = True, inplace=False, kind ='quicksort', na_position='last')
print(sorted_data.head())


# In[ ]:


#dropping duplicate values
final = sorted_data.drop_duplicates(subset = {'UserId', 'ProfileName', 'Time','Text' }, keep = 'first', inplace=False)
print(final.head())
print(final.shape)


# **Observation:** It was also observed in our data that some values have more value of HelpfulnessNumerator than HelpfulnessDenominator which is not possible so we will also clean those values

# In[ ]:


#removing values with HelpfulnessNumerator>HelpfulnessDenominator
final = final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
print(final.shape)
print(final['Score'].value_counts())


# ## [3] Text Preprocessing
# Now that we have removed the deduplicated from our dataset we need to perform some preprocessing operations before we go for further analysis and make the prediction model.
# 
# Hence in the preprocessing phase we do the following preprocessing in the order below:
# 1. Begin by removing HTML tags.
# 2. Remove any punctuation or limited set of special characters like . , # etc.
# 3. Check if the word is made up of English characters and is alpha-numeric.
# 4. Remove the words having size less than 3 (as in English vocabulary no adjective has size less than 3).
# 5. Convert words to lowercase.
# 6. Remove all Stopwords.
# 7. Finally Snowball Stemming the words .(As it was observed that it was better than Porter Stemming)
# 

# In[ ]:


import re # regular expressions
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))# set of English Stopwords
sno = nltk.stem.SnowballStemmer('english')# initialising the snowball stemmer

def cleanhtml(sentence): #function to clear HTML tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

def cleanpunc(sentence): #function to remove punctuations
    cleaned = re.sub(r'[?|!|\'|"|#]', ' ', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', ' ', cleaned)
    return cleaned
print(stop)
    


# In[ ]:


#code to implement all the preprocessing
from tqdm import tqdm
i =0 
strl = ' '
final_string= []
all_postive_words = []#stores all positive words
all_negative_words = []#stores all negative words
s = ''
for sent in tqdm(final['Text'].values) :
    filtered_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split() :
        for cleaned_words in cleanpunc(w).split() :
            if ((cleaned_words.isalpha()) & ((len(cleaned_words))>2)) :
                if (cleaned_words.lower() not in stop) :
                    s = (sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'postive':
                        all_positive_words.append(s)
                    if (final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s)
                else :
                    continue
            else : 
                continue
    strl = b" ".join(filtered_sentence) #final string of all cleared words

    final_string.append(strl)
    i+=1
    


# In[ ]:


print(final_string[0:5])


# In[ ]:


final['CleanedText'] = final_string

print(final.head(3))

#store final table into a SQLlite table for future use
conn =sqlite3.connect('final.sqlite')
c = conn.cursor()
conn.text_factory = str
final.to_sql('Reviews',conn, schema = None, if_exists='replace')


# ## [4] Featurization
# ## [4.1] Bag of Words
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer #BoW
count_vect = CountVectorizer()
final_counts = count_vect.fit_transform(final['CleanedText'].values)

print(final_counts.get_shape())


# In[ ]:


print(type(final_counts))
print(final_counts[0])


# ## [4.2] Bi-Grams and N-Grams
# 

# In[ ]:


count_vect = CountVectorizer(ngram_range=(1,2), max_features = 5000) # all unigram and bigram values
final_bigram_counts = count_vect.fit_transform(final['CleanedText'].values)
print("the type of count vectorizer ",type(final_bigram_counts))
print("the shape of out text BOW vectorizer ",final_bigram_counts.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])


# In[ ]:


print(count_vect.get_feature_names()[0:10])


# ## [4.3] TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vect = TfidfVectorizer( max_features = 5000 )
tf_idf  = tf_idf_vect.fit_transform(final['CleanedText'].values)

print(tf_idf.get_shape())


# ## [4.4] Word2Vec
# Here we are training our own Word2Vec model on the dataset.
# 

# In[ ]:


from gensim.models import Word2Vec 
# making a list of all cleaned sentances.(removing HTML tags and punctuation)
list_of_sentance = []
for sentance in tqdm(final['Text'].values) :
    filtered_sentence = []
    sentance = cleanhtml(sentance)
    for w in sentance.split() :
        for cleaned_words in cleanpunc(w).split() :
            if cleaned_words.isalpha() :
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue
    list_of_sentance.append(filtered_sentence)
            
print(list_of_sentance[0:10])


# In[ ]:


w2v_model = Word2Vec(list_of_sentance, min_count = 5, size = 50 , workers = 4 ) #training Word2Vec on list_of_sentance


# In[ ]:


print(w2v_model.wv['tasty'])


# In[ ]:


print(w2v_model.wv.most_similar('tasty'))


# ## [4.4] Average Word2Vec

# In[ ]:


sentance_vector = []
w2v_words = list(w2v_model.wv.vocab)
for sentance in tqdm(list_of_sentance) :
    sent_vector = np.zeros(50)
    cnt_words = 0
    for word in sentance :
        if word in w2v_words :
            vec = w2v_model.wv[word]
            sent_vector+=vec
            cnt_words +=1
        else :
            continue
    if cnt_words != 0:
        sent_vector = sent_vector/cnt_words
    sentance_vector.append(sent_vector)

print(len(sentance_vector))
print(len(sentance_vector[0]))
            
            
    


# ## [4.4] TF-IDF weighted Word2Vec

# In[ ]:


tfidf_feat = tf_idf_vect.get_feature_names()
row = 0
tfidf_sent_vector = []
for sentance in tqdm(list_of_sentance):
    tfidf_sent_vec = np.zeros(50)
    weight_sum =0
    for word in sentance :
        if word in w2v_words and word in tfidf_feat :
            vec = w2v_model.wv[word]
            
            tfidf = tf_idf[row, tfidf_feat.index(word)]
            tfidf_sent_vec += (vec*tfidf)
            weight_sum +=tfidf
    if weight_sum != 0 :
        tfidf_sent_vec /=weight_sum
    tfidf_sent_vector.append(tfidf_sent_vec)
    row+=1

print(len(tfidf_sent_vector))


# ## [5] Data Visualization (Using t-SNE)

# ## [5.1] Applying t-SNE on Text BoW vectors

# In[ ]:


from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

bow_final_4k = final_counts[0:4000]
score_4k = (final["Score"])[0:4000]

#standardising the BoW values to lie between 0 and 1
bow_final_4k = StandardScaler(with_mean=False).fit_transform(bow_final_4k)
#converting sparse matrix to dense matrix as t-SNE accepts only dense matrix
bow_final_4k = bow_final_4k.todense()  

tsne = TSNE(n_components = 2, random_state = 0 ) 
#default perplexity = 30
#default number of iterations = 1000

bow_tsne = tsne.fit_transform(bow_final_4k)

for_bow = np.vstack((bow_tsne.T, score_4k)).T
for_bow_df = pd.DataFrame(for_bow , columns = ('dimension_1', 'dimension_2', 'Score'))
sns.set_style('whitegrid')
sns.FacetGrid(for_bow_df, hue = 'Score',height = 7).map(plt.scatter, 'dimension_1', 'dimension_2').add_legend()
plt.show()


# ## [5.2] Applying t-SNE on Text TF-IDF vectors

# In[ ]:


from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix

tf_idf_4k = tf_idf[0:4000]
score_4k = final["Score"][0:4000]
tf_idf_4k = StandardScaler(with_mean = False).fit_transform(tf_idf_4k)
tf_idf_4k = tf_idf_4k.todense()

tsne = TSNE(n_components = 2, random_state = 0, perplexity = 50, n_iter=2000 ) 
#default perplexity = 30
#default number of iterations = 1000

tfidf_tsne = tsne.fit_transform(tf_idf_4k)

for_tfidf = np.vstack((tfidf_tsne.T, score_4k )).T
for_tfidf_df = pd.DataFrame(for_tfidf , columns = ('dimension_1', 'dimension_2', 'Score'))
sns.set_style('whitegrid')
sns.FacetGrid(for_tfidf_df, hue = 'Score',height = 7).map(plt.scatter, 'dimension_1', 'dimension_2').add_legend()
plt.show()


# ## [5.3] Applying t-SNE on  Text average Word2Vec 

# In[ ]:


sentance_vector_4k = sentance_vector[0:4000]
sentance_vector_4k = StandardScaler(with_mean = False).fit_transform(sentance_vector_4k)

tsne = TSNE(n_components = 2 , random_state = 0, perplexity = 50 , n_iter=2000)

avgw2v_tsne = tsne.fit_transform(sentance_vector_4k)

for_avgw2v = np.vstack((avgw2v_tsne.T, score_4k)).T
avgw2v_df =pd.DataFrame(for_avgw2v, columns=('dimension_1', 'dimension_2', 'score'))

sns.set_style('whitegrid')
sns.FacetGrid(avgw2v_df, hue = 'score', height = 7).map(plt.scatter, 'dimension_1', 'dimension_2').add_legend()
plt.title('average Word2Vec t-SNE')
plt.show()


# ## [5.5] Applying t-SNE on Text TF-IDF weighted Word2Vec

# In[ ]:


tfidf_sent_vector_4k = tfidf_sent_vector[0:4000]
tfidf_sent_vector_4k = StandardScaler(with_mean = False).fit_transform(tfidf_sent_vector_4k)

tsne = TSNE(n_components = 2 , random_state = 0, perplexity = 50 , n_iter=2000)
tfidf_w2v_tsne = tsne.fit_transform(tfidf_sent_vector_4k)

for_tfidf_w2v = np.vstack((tfidf_w2v_tsne.T, score_4k)).T
tfidf_w2v_df = pd.DataFrame(for_tfidf_w2v, columns=('dimension_1', 'dimension_2', 'score'))

sns.set_style('whitegrid')
sns.FacetGrid(tfidf_w2v_df, hue = 'score', height = 7).map(plt.scatter, 'dimension_1', 'dimension_2').add_legend()
plt.title('tf-idf weighted Word2Vec')
plt.show()


# ### Conclusion
# 1. None of the t-SNE plotted above gives a clear separation between positive and negative values.
# 2. We would have to use alternate methods to separate the values. 

# In[ ]:


#storing values for future use
final_avgw2v = pd.DataFrame(sentance_vector)
print(final_avgw2v.shape)

final_avgw2v.to_csv(r'D:\New folder\Practicals\amazon fine food\avgw2v.csv')


# In[ ]:


final_tfidf_w2v = pd.DataFrame(tfidf_sent_vector)
print(final_tfidf_w2v.shape)

final_tfidf_w2v.to_csv(r'D:\New folder\Practicals\amazon fine food\tfidf_w2v.csv')


# In[ ]:




