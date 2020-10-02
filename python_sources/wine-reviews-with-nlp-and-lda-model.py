#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# The goal of the project was to analyze data called Wine Reviews collected from winemag.com and published on kaggle platform, consisting of 130 000 rows. Research consists of some exploratory data analysis, especially reviews, based on which, in the next step, a simple wine recommender was built.
# Chosen programming language was Python and used libraries among others are:
# - matplotlib, wordcloud, PIL and seaborn for visualization
# - pandas and numpy for data exploring
# - nltk for text analysis
# - sklearn and gensim for making predictions

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import numpy as np
from wordcloud import WordCloud
import sklearn as sk
from PIL import Image
import nltk
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
import unicodedata
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

from ggplot import *


# Variables taken into consideration here were variety of wine, country, price, description and points. After dealing with some missing prices and points (inserting mean in its place), removing few rows without variety or description and fixing the wine varieties that are in non-english languages (f. e. changing 'alvarinho' into 'albarino'), some plots were made.

# In[ ]:


df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')
df.head(10)


# In[ ]:


df.info()


# In[ ]:


df.describe(include = 'all')


# Firstly top wine varieties were check (fig. ). Three the most popular are Pinot Noir, Chardonnay, Carbenet Cauvignon. All those come from France. To see people of which nationality rate wine, a barplot of top 10 countries based on review count was made (fig. ). It can be clearly seen that most of reviewers are from USA, probably because source website is american. About 2,5 times less reviewers are from France and nextly from Italy, presumably as an effect of good wines origin.

# In[ ]:


topCountriesList = df['country'].value_counts(ascending=False).reset_index().head(10)['index'].tolist()
topCountries = df[(df['country'].isin(topCountriesList))]

mpl.rcParams["figure.figsize"] = "15, 8"
p = ggplot(aes(x='factor(country)'), data=topCountries) +      geom_bar() +      xlab('Country') +     ylab('Review Count') #+\
     #ggtitle('Top 10 countries based on review count')
p.save('review_count.png')


# In[ ]:


topVarieties = df['variety'].value_counts().head(30).to_frame()
print(topVarieties)
tuples = topVarieties.to_records(index=True).tolist()
print(tuples)
d = {x:y for x,y in tuples}
wordcloud = WordCloud(background_color="white", width=800, height=400).generate_from_frequencies(d)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('most_popular.png')
plt.show()


# In the next step, in order to observe if it is possible to buy a bottle of delicious wine without spending a lot of money, a jointplot of points dependent on price (with upper bound equal to 100 dollars) was made. The main conclusions here are that most of scored wines cost less than 20 dollars and people are usually satisfied with them, grading with 85-90 points but there is also a tendency the more expensive, the better.

# In[ ]:


mpl.rcParams["figure.figsize"] = "15, 8"
sns.jointplot(x='price', y='points', data=df[df['price'] < 100], kind = 'hex', gridsize=20)
plt.savefig('jointplot.png')


# Text analysis

# In[ ]:


#tvec = TfidfVectorizer(min_df=0.0025, max_df = .1, stop_words='english')
#tvec_weights = tvec.fit_transform(df['description'].dropna())
#weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
#weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
#weights_df.sort_values(by='weight', ascending=False).head(10)


# In[ ]:


rawText = df['description'].to_string()
raw = rawText.replace('\n',' ') 
tokens = nltk.word_tokenize(raw)

#change all tokens into lower case 
words1 = [w.lower() for w in tokens]

#only keep text words, no numbers 
words = [w for w in words1 if w.isalpha()]

#remove stopwords
stopwords_list = stopwords.words('english')
words_nostopwords = [w for w in words if w not in stopwords_list]

#remove morphological affixes from words, leave only the word stem
stemmed_words = []
for x in words_nostopwords:
    stemmed_words.append(PorterStemmer().stem(x))
stemmed_words = filter(lambda a: a != 'wine', stemmed_words)


# In[ ]:


#generate a frequency dictionary for all tokens 
freq = FreqDist(stemmed_words)

#sort the frequency list in decending order
sorted_freq = sorted(freq.items(),key = lambda k:k[1], reverse = True)
top_list = []
for x in sorted_freq[0:40]:
    top_list.append((x[0],x[1]))


# In[ ]:


wine_mask = np.array(Image.open("../input/wine_mask/wine_mask.png"))

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val
    
# Transform mask into a new one that will work with the function:
transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)

for i in range(len(wine_mask)):
    transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))


# In[ ]:


d = {x:y for x,y in top_list}
print(d)
wordcloud = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick').generate_from_frequencies(d)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


freq.plot(40,title="Top 40 Frequently Occuring Terms")


# In[ ]:


wines = df.variety.unique().tolist()

def words_nostop(column, value):
    df2 = df[(df[column]==value)]
    valueNew = value.lower()
    raw1 = df2['description'].to_string()
    raw = raw1.replace('\n',' ') 
    tokens = nltk.word_tokenize(raw)
    words1 = [w.lower() for w in tokens]
    words2 = [w for w in words1 if w.isalpha()]
    words3 = list(filter(lambda a: a != valueNew, words2))
    words4 = list(filter(lambda a: a != 'wine', words3))
    stopwords2 = stopwords.words('english') + list(punctuation) + wines
    words_nostopwords = [w for w in words4 if w not in stopwords2]
    return words_nostopwords
    
def word_counter(column,value):
    words_nostopwords = words_nostop(column, value)
    stemmed_words = []
    for x in words_nostopwords:
        stemmed_words.append(PorterStemmer().stem(x))
    freq = FreqDist(stemmed_words)
    #sort the frequency list in decending order
    sorted_freq = sorted(freq.items(),key = lambda k:k[1], reverse = True)
    top_list = []
    for x in sorted_freq[0:40]:
        top_list.append((x[0],x[1]))
    d = {x:y for x,y in top_list}
    wordcloud = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick').generate_from_frequencies(d)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig('france.png')
    plt.axis("off")
    plt.show()
    freq.plot(40,title="Top 40 Frequently Occuring Terms")


# Nextly, the descriptions were analyzed with nltk library. After some preprocessing, including removing stopwords, wine variety name and word 'wine', frequency wordmaps drawing function was created, where arguments are column name and value. For example, on the next plot characteristics of wine from four different countries are compared.

# In[ ]:


word_counter('country','France')


# In[ ]:


#fixing the wine varieties that are in non-english languages.
wine_ml = df.copy()
wine_ml['variety'] = wine_ml['variety'].replace(['weissburgunder'], 'chardonnay') 
wine_ml['variety'] = wine_ml['variety'].replace(['spatburgunder'], 'pinot noir') 
wine_ml['variety'] = wine_ml['variety'].replace(['grauburgunder'], 'pinot gris')
wine_ml['variety'] = wine_ml['variety'].replace(['garnacha'], 'grenache')
wine_ml['variety'] = wine_ml['variety'].replace(['pinot nero'], 'pinot noir')
wine_ml['variety'] = wine_ml['variety'].replace(['alvarinho'], 'albarino')

#145


# In[ ]:





# In[ ]:


wines = df.variety.unique().tolist()
stoplist = stopwords.words('english') + list(punctuation) + wines

def build_corpus(col, stoplist):
    corpus = [desc.lower() for desc in col]
    return [[word for word in str(desc).split() if word not in stoplist] for desc in corpus]

corpuss = build_corpus(wine_ml.description, stoplist)
dictionary = corpora.Dictionary(corpuss)
print(dictionary)
#tuples with words and number of occurance
corpus = [dictionary.doc2bow(text) for text in corpuss]
print(corpus)


# In the next step, the dictionary of tokens was transfered into occurance-weighted TF-IDF matrix. Then, usage of Latent Dirichlet Allocation to essentially bin words into topics in a way similar to probability distribution, where chosen topics number was equal to 32, which is number of wine varieties which occur more than 500 times. Later, the LDA model was used to return the most likely topic for each description in the dataset. This compared to analogical processed input description and price are the values applied to our recommender.
# 
# 

# In[ ]:


#Objects of this class realize the transformation between word-document co-occurrence matrix (int) 
#into a locally/globally weighted TF-IDF matrix (positive floats).
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
#print(wine_ml.variety.value_counts()[wine_ml.variety.value_counts() > 500])
#len(wine_ml.variety.value_counts()[wine_ml.variety.value_counts() > 500])      #40
#wine_ml.variety.value_counts()[wine_ml.variety.value_counts() > 500].index.str.contains('Blend').sum()  #8
total_topics = 32


# In[ ]:


#use Latent Dirichlet Allocation to essneitally bin our words 
#in to topics in a way similar to a probability distribution.
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)
print(lda.print_topics())
corpus_lda = lda[corpus_tfidf]
#wine_ml_topics = wine_ml[['title', 'variety', 'description', 'points', 'price', 'winery']]
wine_ml_topics = wine_ml[['variety', 'description', 'points', 'price', 'country']]
#print(wine_ml_topics)


# In[ ]:


#Evaluation

# Compute Perplexity
print('\nPerplexity: ', lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda, texts=corpuss, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


mallet_path = '/home/zz/mallet-2.0.8/bin/mallet' # update this path
#ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=dictionary)


# In[ ]:


import gensim
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=corpuss, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=corpuss, start=30, limit=50, step=2)

# Show graph
limit=60; start=10; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.savefig('evaluat.png')
plt.show()


# In[ ]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[ ]:


#use Latent Dirichlet Allocation to essneitally bin our words 
#in to topics in a way similar to a probability distribution.
#Num Topics = 46  has Coherence Value of 0.5355
total_topics = 25
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)
print(lda.print_topics())
corpus_lda = lda[corpus_tfidf]
#wine_ml_topics = wine_ml[['title', 'variety', 'description', 'points', 'price', 'winery']]
wine_ml_topics = wine_ml[['variety', 'description', 'points', 'price', 'country']]


# In[ ]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
vis


# In[ ]:


#use our LDA model to return the most likely topic for each description in the dataset. 
#This and price will be the values we use to recommend a few wines below.
likely_topics = [max(lda[corpus[row]], key = itemgetter(1))[0] for row in range(wine_ml_topics.shape[0])]
wine_ml_topics = wine_ml_topics.assign(topic = likely_topics)
recommend_df = wine_ml_topics.dropna()
recommend_df


# The recommender takes a short description of dreamed-of wine and price or prices interval as arguments. Based on price, mask of possible wines was created, then lda models described in the previous section, were compared by using of cosine similarity measure. The function is shown below (fig. ).

# In[ ]:


def what_wine_to_drink(desc, price_point, df=recommend_df, num_response=5):
    
    wines = recommend_df.variety.unique().tolist()
    stoplist = stopwords.words('english') + list(punctuation) + wines
    
    #narrow by our price points first
    if isinstance(price_point, (int, float)):
        price_mask = (df.price < price_point+4.5*df.price.std()) & (df.price > price_point-4.5*df.price.std())
        predict_df = df[df.price == price_point]
        price = price_point
    else:
        price_dict = {'500 - 3300': (3300, 500), 
                      '100 - 500': (500, 100),
                      '25 - 100': (100,25), 
                      '0 - 25': (25,0)}
        
        price_mask = (df.price < price_dict[price_point][0]) & (df.price > price_dict[price_point][1])
        predict_df = df[price_mask]
        #choose average price of category
        price = np.mean(price_dict[price_point])
    
    #get most likely topic
    texts = [word for word in str(desc).lower().split() if word not in stoplist]
    desc_v = dictionary.doc2bow(texts)
    topic = max(lda[desc_v], key = itemgetter(1))[0]
    predict_df = predict_df[predict_df.topic==topic]
    predict_df = predict_df[['price','topic']]
    
    user_wine = np.array([price, topic])
    try:
        predict_df = predict_df.assign(sim_score = cosine_similarity(predict_df, user_wine.reshape(1, -1)))
    except ValueError:
        predict_df = recommend_df[['price','topic']]
        predict_df = predict_df.assign(sim_score = cosine_similarity(predict_df, user_wine.reshape(1, -1)))  
    return recommend_df.loc[predict_df.nlargest(5, 'sim_score').index].drop('topic', axis=1)


# In[ ]:


def wine_input():
    desc = input('Enter a short description of what kind of wine you want to drink: \n')
    desc = list(desc)
    
    price_point = input('''Enter a price interval (in dollars) of "500 - 3300", "100 - 500", "25 - 100", "0 - 25", 
                        or an actual numerical price: \n''')
    if len(price_point) < 5:
        price_point = float(price_point)
    
    return what_wine_to_drink(desc, price_point)


# In[ ]:


wine_input()


# In[ ]:





# In[ ]:




