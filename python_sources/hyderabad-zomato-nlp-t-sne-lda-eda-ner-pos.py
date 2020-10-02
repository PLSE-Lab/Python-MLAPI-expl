#!/usr/bin/env python
# coding: utf-8

# # Hyderabad
# If you are Indian or have stayed in India for sometime, probably you have heard about Hyderabadi Biryani, its one of hallmark dishes of country and city. We will use Zomato reviews from Hyderabad locality to see how the food market is doing in Nizam City.
# 

# # Reading the files

# In[ ]:


import pandas as pd
hyd_rest=pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')
hyd_rev=pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant reviews.csv')


# ## Importing the NLTK, Scikit libraries and features

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# # EDA
# The notebook will use bokeh and plotly to see ratings, reviews and cost relationships , will use NLTK,gensim, to convert text to vectors to find relatinonships between text. We will also see wordclouds 

# ## The Ratings distribution

# In[ ]:


hyd_rev['Rating'].value_counts(normalize=True)


# 38% reviews are 5 rated,23% are 4 rated stating that people do rate good food high.

# ### creating new variable review length to see if length of review impacts the ratings
# 

# In[ ]:


hyd_rev['Review']=hyd_rev['Review'].astype(str)
hyd_rev['Review_length'] = hyd_rev['Review'].apply(len)


# #### Bokeh scatter plot for review length vs rating

# In[ ]:


import plotly.express as px
fig = px.scatter(hyd_rev, x=hyd_rev['Rating'], y=hyd_rev['Review_length'])
fig.update_layout(title_text="Rating vs Review Length")
fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=45, ticklen=10)
fig.show()


# The scatter plot confirms that length of review doesnt impact ratings

# ## Creating polarity variable to see sentiments in reviews(using textblob)
# 
# Ploarity analyzes the text ranges and search for words that express sentiments such as good or bad assignes a score to text in following manner: emotional negative (-2), rational negative (-1), neutral (0), rational positive (+1), and emotional positive (+2). In practice, neutral often means no opinion or sentiment expressed.

# In[ ]:


hyd_rev['Polarity'] = hyd_rev['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[ ]:


hyd_rev['Polarity'].plot(kind='hist', bins=100)


# The graph shows us the majority of reviews are nuetral 0,probably sugesting mixture of bad and good words in reviews, also the number of positive reviews >0 are higher than negative reviews, more than 200 odd reviews have very high positive sentiments

# # wordclouds for all reviews, positive reviews and negative reviews
# 
# I will create two datasets equal and above 3 rating for positive reviews and below 3 for negative reviews. Apart from stopwords i have removing common words used in restuarant business
# 

# In[ ]:


stop_words = stopwords.words('english')
print(stop_words)
rest_word=['order','restaurant','taste','ordered','good','food','table','place','one','also']
rest_word


# In[ ]:


import re
hyd_rev['Review']=hyd_rev['Review'].map(lambda x: re.sub('[,\.!?]','', x))
hyd_rev['Review']=hyd_rev['Review'].map(lambda x: x.lower())
hyd_rev['Review']=hyd_rev['Review'].map(lambda x: x.split())
hyd_rev['Review']=hyd_rev['Review'].apply(lambda x: [item for item in x if item not in stop_words])
hyd_rev['Review']=hyd_rev['Review'].apply(lambda x: [item for item in x if item not in rest_word])


# In[ ]:


from wordcloud import WordCloud
hyd_rev['Review']=hyd_rev['Review'].astype(str)

ps = PorterStemmer() 
hyd_rev['Review']=hyd_rev['Review'].map(lambda x: ps.stem(x))
long_string = ','.join(list(hyd_rev['Review'].values))
long_string
wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# ambience is key factor in food business, time, and variety in food is the key like starters , main course

# ### Creating two datasets for positive and negative reviews

# In[ ]:



hyd_rev['Rating']=pd.to_numeric(hyd_rev['Rating'],errors='coerce')
pos_rev = hyd_rev[hyd_rev.Rating>= 3]
neg_rev = hyd_rev[hyd_rev.Rating< 3]


# ## Positive reviews wordcloud

# In[ ]:


long_string = ','.join(list(pos_rev['Review'].values))
long_string
wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# Service,taste,time,starters are key to good review

# ## Negative reviews wordcloud

# In[ ]:


long_string = ','.join(list(neg_rev['Review'].values))
long_string
wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# Service , bad chicken , staff behavior, stale food are key reasons for neagtive reviews

# # Creating word embeddings and t-SNE plot( for positive and negative reviews)
# 

# Tsne works by taking a group of high-dimensional (100 dimensions via Word2Vec) vocabulary word feature vectors, then compresses them down to 2-dimensional x,y coordinate pairs. The idea is to keep similar words close together on the plane, while maximizing the distance between dissimilar words.

# In[ ]:


from gensim.models import word2vec
pos_rev = hyd_rev[hyd_rev.Rating>= 3]
neg_rev = hyd_rev[hyd_rev.Rating< 3]


# ## plot for negative reviews

# In[ ]:


def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['Review']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(neg_rev)        
corpus[0:2]


# In[ ]:


model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# In[ ]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[ ]:


tsne_plot(model)


# The words close together for negative reviews : service,staff,time suggest these are key factors in negative reviews

# ## plot for postive reviews

# In[ ]:


def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['Review']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(pos_rev)        
corpus[0:2]


# In[ ]:


model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# In[ ]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[ ]:


tsne_plot(model)


# Here we have lot many frequent occurences: Chicken items like biryani kebab , cheese , burger pizza , variety service staff music ambience are key too good reviews

# # POS tagging of negative rated reviews
# 
# POS tagging tags the text to grammar like if its is noun, pronoun, verb ,adjective etc are present in texts:
# 
# The complete lits of POS tags are in this website:
# 
# https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/

# In[ ]:


from nltk.tag import pos_tag
from nltk import pos_tag_sents


# Creating a loop that counts pos tags increments tag count of tags, then we plot the number pos tags frequency

# In[ ]:


neg_texts = neg_rev['Review'].str.split().map(pos_tag)
neg_texts.head()
def count_tags(title_with_tags):
    tag_count = {}
    for word, tag in title_with_tags:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1
    return(tag_count)
neg_texts.map(count_tags).head()


# In[ ]:


neg_texts = pd.DataFrame(neg_texts)
neg_texts['tag_counts'] = neg_texts['Review'].map(count_tags)
neg_texts.head()


# In[ ]:


tag_set = list(set([tag for tags in neg_texts['tag_counts'] for tag in tags]))
for tag in tag_set:
    neg_texts[tag] = neg_texts['tag_counts'].map(lambda x: x.get(tag, 0))
title = 'Frequency of POS Tags in Negative Reviews'    
neg_texts[tag_set].sum().sort_values().plot(kind='barh', logx=True, figsize=(12,8), title=title)


# # Topic Modeling using LDA
# We will plot top 10 most occuring words. Topic modeling is  a process to automatically identify topics present in a text object and to assign text corpus to one category of topic.
#  
# LDA is one of the methods to assign topic to texts. If observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's presence is attributable to one of the document's topics.
# 
# Source: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation

# In[ ]:


import numpy as np
import seaborn as sns
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
        count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words in Negative reviews')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
text2=neg_rev['Review'].values
count_vectorizer = CountVectorizer(stop_words='english')

count_data = count_vectorizer.fit_transform(text2)
plot_10_most_common_words(count_data, count_vectorizer)


import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 10
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("10 Topics found via LDA for negative reviews:")
print_topics(lda, count_vectorizer, number_words)



# # Named Entity Recoginition in negative reviews
# 
# Named-entity recognition (NER) (also known as entity identification, entity chunking and entity extraction) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.
# 
# Source: https://en.wikipedia.org/wiki/Named-entity_recognition
# 
# I will try to find negative reviews wherever they have date mentioned in specifically in the review

# In[ ]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
def extract_named_ents(text):    
    return [ ent.label_ for ent in nlp(text).ents]  
neg_rev['named_ents'] = neg_rev['Review'].apply(extract_named_ents)   


# In[ ]:


text_ents=neg_rev[['named_ents','Review','Restaurant','Rating']]
text_ents['named_ents_new']=[','.join(map(str, l)) for l in text_ents['named_ents']]
text_ents


# In[ ]:


DATE_PROBLEM=text_ents['named_ents_new'] == 'DATE'
text_ents[DATE_PROBLEM]


# # Merging two datasets and Finding North Indian, Chinese and South Indian food options and ratings

# In[ ]:


avg_rating=hyd_rev.groupby('Restaurant',as_index=False)['Rating'].mean()
merged=hyd_rest.merge(avg_rating, how='inner',left_on='Name',right_on='Restaurant')


# In[ ]:


merged.head()


# In[ ]:


merged["North_indian"]= merged["Cuisines"].str.find("North Indian")  
merged["Chinese"]=merged["Cuisines"].str.find("Chinese")
merged["South_Indian"]=merged["Cuisines"].str.find("South Indian")


# In[ ]:


merged.loc[merged['North_indian'] == -1, 'North_Indian_menu'] = 0
merged.loc[merged['North_indian'] > -1, 'North_Indian_menu'] = 1
merged.loc[merged['Chinese'] == -1, 'Chinese_menu'] = 0
merged.loc[merged['Chinese'] > -1, 'Chinese_menu'] = 1
merged.loc[merged['South_Indian'] == -1, 'South_Indian_menu'] = 0
merged.loc[merged['South_Indian'] > -1, 'South_Indian_menu'] = 1


# In[ ]:


North=merged[merged['North_Indian_menu'] == 1]
mean_rating_N=North.groupby(['Name','Cost'],as_index=False).Rating.mean()


# In[ ]:


import plotly.express as px
fig = px.bar(mean_rating_N, x="Name", y="Cost",color="Rating")
fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=45, ticklen=10)
fig.update_layout(title_text="North Indian restaurant cost vs rating")
fig.show()


# In[ ]:


South=merged[merged['South_Indian_menu'] == 1]

mean_rating_S=South.groupby(['Name','Cost'],as_index=False).Rating.mean()
fig = px.bar(mean_rating_S, x="Name", y="Cost",color="Rating")
fig.update_layout(title_text="South Indian restaurant cost vs rating")
fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=45, ticklen=10)
fig.show()


# In[ ]:


Chinese=merged[merged['Chinese_menu'] == 1]
mean_rating_C=Chinese.groupby(['Name','Cost'],as_index=False).Rating.mean()
fig = px.bar(mean_rating_C, x="Name", y="Cost",color="Rating")
fig.update_layout(title_text="Chinese restaurant cost vs rating")
fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='crimson',tickangle=45, ticklen=10)
fig.show()


# # Kmeans - 2 cluster suggesting 1000INR for two is quality bechmark
# 
# Tried K means clustering which gives 2 clear clusters when we cluster restuarant based on rating and cost for two, if its above 1000 INR the rating are are never low barring one near average rating of Hyatt

# In[ ]:


merged['Cost']=merged['Cost'].str.replace(',', '').astype(float)
merged['Cost']=merged['Cost'].astype(float)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(merged[['Cost', 'Rating']])
merged['Name'] = kmeans.labels_
with plt.style.context('bmh', after_reset=True):
    pal = sns.color_palette('Spectral', 7)
    plt.figure(figsize = (8,6))
    for i in range(2):
        ix = merged.Name == i
        plt.scatter(merged.loc[ix, 'Rating'], merged.loc[ix, 'Cost'], color = pal[i], label = str(i))
        plt.text(merged.loc[i, 'Rating'], merged.loc[i, 'Cost'], str(i) + ': '+str(merged.loc[i, 'Name'].round(2)), fontsize = 14, color = 'brown')
    plt.title('KMeans Hyderabad Restaurant for Cost and Rating')
    plt.legend()
    plt.show()


# # Multiple Cuisines gives higher ratings
# 
# we will plot : 
# 1.what type of cuisines are served most
# 2.frequency of multiple cuisines restaurant
# 3.rating vs multiple cuisines
# 

# In[ ]:


merged['Cuisines'] = merged['Cuisines'].astype(str)
merged['fusion_num'] = merged['Cuisines'].apply(lambda x: len(x.split(',')))

from collections import Counter
lst_cuisine = set()
Cnt_cuisine = Counter()
for cu_lst in merged['Cuisines']:
    cu_lst = cu_lst.split(',')
    lst_cuisine.update([cu.strip() for cu in cu_lst])
    for cu in cu_lst:
        Cnt_cuisine[cu.strip()] += 1

cnt = pd.DataFrame.from_dict(Cnt_cuisine, orient = 'index')
cnt.sort_values(0, ascending = False, inplace = True)


tmp_cnt = cnt.head(10)
tmp_cnt.rename(columns = {0:'cnt'}, inplace = True)
with plt.style.context('bmh'):
    f = plt.figure(figsize = (12, 8))
    ax = plt.subplot2grid((2,2), (0,0))
    sns.barplot(x = tmp_cnt.index, y = 'cnt', data = tmp_cnt, ax = ax, palette = sns.color_palette('Blues_d', 10))
    ax.set_title('# Cuisine')
    ax.tick_params(axis='x', rotation=70)
    ax = plt.subplot2grid((2,2), (0,1))
    sns.countplot(merged['fusion_num'], ax=ax, palette = sns.color_palette('Blues_d', merged.fusion_num.nunique()))
    ax.set_title('# Cuisine Provided')
    ax.set_ylabel('')

    ax = plt.subplot2grid((2,2), (1,0), colspan = 2)
    fusion_rate = merged[['fusion_num', 'Rating']].copy()
    fusion_rate.loc[fusion_rate['fusion_num'] > 5,'fusion_num'] = 5
    fusion_rate = fusion_rate.loc[fusion_rate.Rating != -1, :]
    pal = sns.color_palette('Oranges', 11)
    for i in range(1,6):
        num_ix = fusion_rate['fusion_num'] == i
        sns.distplot(fusion_rate.loc[num_ix, 'Rating'], color = pal[i*2], label = str(i), ax = ax)
        ax.legend()
        ax.set_title('Rating Distribution for fusion_number')

    plt.subplots_adjust(wspace = 0.5, hspace = 0.8, top = 0.85)
    plt.suptitle('Cuisine _ Rating')
    plt.show()        
print('# Unique Cuisine: ', len(lst_cuisine))


# # Extracting total reviews from metadata column

# In[ ]:


hyd_rev['total_reviews']=hyd_rev['Metadata'].str.extract('(\d+)')
hyd_rev


# ## 3d plot for finding pattern between review length , rating and number of review

# In[ ]:


import plotly.express as px
fig = px.scatter_3d(hyd_rev, x='Review_length', y='total_reviews', z='Rating')
fig.update_layout(title_text="Review Length vs Rating vs Number of Reviews ")
fig.show()


# The plot suggest that very lengthy reviews have either very high ratings or very ratings. Average reviews have very small length of review.Number of reviews do not show much impact on ratings

# # Poeple who reviewed > 300 times, Their orders and ratings

# In[ ]:


reviewer_rating=hyd_rev.groupby(['Reviewer'],as_index=False).Rating.mean()
merged2=reviewer_rating.merge(hyd_rev[['Reviewer','total_reviews']],how='left',left_on='Reviewer',right_on='Reviewer')
merged2=merged2.drop_duplicates()
merged2['total_reviews']=merged2['total_reviews'].fillna(0)
merged2['total_reviews']=merged2['total_reviews'].astype(int)
reveiwer_300=merged2[merged2['total_reviews']>300]


# In[ ]:


fig = px.scatter_matrix(reveiwer_300,dimensions=["total_reviews", "Rating"], color="Reviewer")
fig.update_layout(title_text="Total Reviews vs Ratings for 300+ reviewers ")
fig.show()


# Two people in particular have rated every restuarant very poorly, while person who have most reviews gives average ratings to every restuarant.

# Thanks for your time to review this lengthy notebook, its a long effort, your upvote will motivate me.
# 
