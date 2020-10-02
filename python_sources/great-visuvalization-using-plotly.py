#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


conda install -c plotly plotly-orca==1.2.1


# In[ ]:


from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import plotly.express as px
import plotly.figure_factory as ff
from nltk.corpus import stopwords
stop = stopwords.words('english')
from wordcloud import WordCloud
from datetime import datetime
import plotly


# In[ ]:


restarant_names = pd.read_csv("/kaggle/input/zomato-restaurants-hyderabad/Restaurant reviews.csv")
reviews = pd.read_csv("/kaggle/input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv")


# In[ ]:


restarant_names.head()


# In[ ]:


restarant_names["Restaurant"].value_counts()


# In[ ]:


rcParams["figure.figsize"] = 50,10
sns.countplot(x="Restaurant",hue="Rating",data=restarant_names[:300])


# In[ ]:


restarant_names["Rating"].value_counts()


# In[ ]:


def convert_review_rating(Rating):
    if Rating=="1.5":
        Rating =1
    elif Rating == "1":
        Rating =1
    elif Rating =="2.5":
        Rating = 2
    elif Rating =="2":
        Rating = 2
    elif Rating =="3.5":
        Rating = 3
    elif Rating =="3":
        Rating = 3
    elif Rating == "4.5":
        Rating = 4
    elif Rating =="4":
        Rating = 4
    elif Rating =="5":
        Rating = 5
    elif Rating =="Like":
        Rating = 5
    return Rating


# In[ ]:


restarant_names["Rating"] = restarant_names["Rating"].apply(convert_review_rating)


# In[ ]:


restarant_names.head()


# In[ ]:


restarant_names.isna().sum()


# In[ ]:


fig = px.histogram(restarant_names["Rating"])
fig.show()


# In[ ]:


# rcParams["figure.figsize"]  =15,10
# restarant_names["Rating"].value_counts().plot(kind="pie")
fig = px.pie(values=restarant_names["Rating"].value_counts(),hover_name=[5.0,4.0,3.0,2.0,1.0],title="Hotel Rating Distribution")
fig.show()


# In[ ]:


restarant_names.sort_values(by=['Rating'], inplace=True,ascending=False)


# In[ ]:


restarant_names.head()


# In[ ]:


Dowinng_10 = restarant_names[restarant_names["Restaurant"]=="10 Downing Street"]


# In[ ]:


rcParams["figure.figsize"] = 20,10
sns.countplot(x=Dowinng_10["Restaurant"],hue=Dowinng_10["Rating"])


# In[ ]:


len(set(restarant_names.Reviewer))


# ## There are totally 1000 data points and has almost 7447 unique reviewers and only remaining people are regular reviewers

# # Five star Hotels

# In[ ]:


five_start_hotels = restarant_names[restarant_names["Rating"]==5]


# In[ ]:


five_start_hotels.head()


# In[ ]:


rcParams["figure.figsize"] = 50,10
five_start_hotels.Restaurant.value_counts().plot(kind="bar")


# In[ ]:


restarant_names.shape


# In[ ]:


restarant_names.isna().sum()


# In[ ]:


reviews.head()


# In[ ]:


reviews.isna().sum()


# In[ ]:


def convert(text):
    return int(text.replace(',',''))
reviews["Cost"] = reviews["Cost"].apply(convert)


# In[ ]:


top_5  = reviews.sort_values(by="Cost",ascending=False).head(5)
tail_5 = reviews.sort_values(by="Cost",ascending=True).head()


# In[ ]:


final_cost = pd.concat([top_5,tail_5])


# In[ ]:


fig = px.pie(final_cost ,values="Cost")
fig.show()


# ## Top and  Bottom price distribution

# In[ ]:


restarant_names.head()


# In[ ]:


restarant_names.shape


# In[ ]:


set(restarant_names.Rating)


# In[ ]:


restarant_names = restarant_names.dropna()


# In[ ]:


restarant_names.shape


# In[ ]:


restarant_names["Rating"].fillna(round(restarant_names["Rating"].mean()))
# eplace("NaN",)   


# ## There is no NaN values in the dataframe

# In[ ]:


restarant_names.head()


# In[ ]:


restarant_names["final_text"] = restarant_names["Restaurant"]+" "+restarant_names["Review"]


# In[ ]:


reviewers = restarant_names['Reviewer'].value_counts().reset_index()
reviewers.columns = ['Reviewer', 'Reviews']

fig = px.histogram(reviewers, 'Reviews')
fig.update_layout(title = "Distribution in no of reviews:",
                 xaxis_title = "No of Reviews",
                 yaxis_title = "Given By users")
fig.show()


# In[ ]:


temp = reviewers.head()['Reviewer'].tolist()
print("People who have posted most reviews are :", temp)


# In[ ]:


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[ ]:


restarant_names['final_text'] = restarant_names['final_text'].apply(lambda x:clean_text(x))


# In[ ]:


restarant_names['temp_list'] = restarant_names['final_text'].apply(lambda x:str(x).split())
top = Counter([item for sublist in restarant_names['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')


# In[ ]:


fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', width=700, height=700,color='Common_words')
fig.show()


# In[ ]:


fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
fig.show()


# In[ ]:


def remove_stopword(x):
    return [w for w in x if not w in stop]


# In[ ]:


restarant_names['temp_list'] = restarant_names['temp_list'].apply(lambda x:remove_stopword(x))


# In[ ]:


top = Counter([item for sublist in restarant_names['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Purples')


# In[ ]:


fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', width=700, height=700,color='Common_words')
fig.show()


# In[ ]:


fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
fig.show()


# In[ ]:


def generate_word_cloud(text,img_name):
    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'black').generate(str(text))
    fig = plt.figure(
        figsize = (40, 30),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig(img_name)


# In[ ]:


text_values = restarant_names.final_text.values[:100]
generate_word_cloud(text_values,'image1.png')


# In[ ]:


restarant_names.head()


# In[ ]:


reviews.head()


# In[ ]:


cuisines = reviews['Cuisines']
cuisines = cuisines.apply(lambda x : x.lower())


# In[ ]:


all_cuisines = ', '.join(i for i in cuisines.tolist())
all_cuisines = Counter(all_cuisines.split(', '))
all_cuisines = pd.DataFrame.from_records(list(dict(all_cuisines).items()), columns=['Name','No Of Restaurents'])
all_cuisines.sort_values(by='No Of Restaurents', ascending=False, inplace=True)


# In[ ]:


fig = px.histogram(x="Name",y="No Of Restaurents",data_frame=all_cuisines,title="Total cuisine Available",color="No Of Restaurents")
fig.show()


# In[ ]:


fig = px.histogram(x="Name",y="No Of Restaurents",data_frame=all_cuisines[:10],title="Top cuisine loved by people")
fig.show()


# In[ ]:


import seaborn as sns


# In[ ]:


cusine_sort = reviews.sort_values(by="Cost",ascending=False)


# In[ ]:


cusine_sort.head()


# In[ ]:


cusine_sort_price = cusine_sort[cusine_sort["Cost"]>=1000]
cusine_sort_price_1000 = cusine_sort[cusine_sort["Cost"]<1000]


# In[ ]:


cusine_sort.head()


# In[ ]:


fig = px.bar(cusine_sort_price, x='Cost', y='Cuisines', color='Cost',title="Cost distribution based on Cusines cost greater than 1000",)
fig.show()


# In[ ]:


fig = px.bar(cusine_sort_price_1000, x='Cost', y='Cuisines', color='Cost',title="Cost distribution based on Cusines cost less than 1000")
fig.show()


# In[ ]:


cusine_text = reviews.Cuisines.values
generate_word_cloud(cusine_text,"cusines.jpeg")


# In[ ]:


cusine_sort_price_viz = cusine_sort_price_1000[cusine_sort_price_1000["Cost"]<300]


# In[ ]:


fig = px.bar(cusine_sort_price_viz, x='Cost', y='Name', color='Cost',title="Cost distribution based on Cusines cost less than 1000")
fig.show()


# In[ ]:


restarant_names_very_low = restarant_names[restarant_names["Rating"]<3.0]


# In[ ]:


restarant_names_very_low.head()


# In[ ]:


restarant_names['Time'] = restarant_names['Time'].apply(lambda x : datetime.strptime(x, '%m/%d/%Y %H:%M'))


# In[ ]:


restarant_names_very_low_values = restarant_names_very_low.Review.values
generate_word_cloud(restarant_names_very_low_values,"low_restarant.png")


# In[ ]:


fig = px.bar(restarant_names_very_low, y='Restaurant')
fig.show()


# In[ ]:


restarant_names.head()


# In[ ]:


def convert_to_three_class(Rating):
    if Rating <3.0:
        Rating = 0
    elif Rating>3.0:
        Rating = 2
    elif Rating ==3.0:
        Rating = 1
    return Rating


# # Labels
# >Label 0 says the restaraunt which are less than 3 rating
# 
# >Label 1 says the restaraunt which are equal to 3  rating
# 
# >Label 2 says the restaraunt which are greater than 3 rating

# In[ ]:


new_restarant = restarant_names


# In[ ]:


new_restarant["Rating"] = restarant_names["Rating"].apply(convert_to_three_class)


# In[ ]:


set(new_restarant["Rating"])


# In[ ]:


zero_review = new_restarant[new_restarant["Rating"]==0]


# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt


# In[ ]:


negative_text = list(zero_review.Review.values)


# In[ ]:


def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

# Convert to list
# data = df.content.values.tolist()
data_words = list(sent_to_words(negative_text))


# In[ ]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


# In[ ]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# !python3 -m spacy download en  # run in terminal once
def process_words(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out


# In[ ]:


data_ready = process_words(data_words)  # processed Text Data!


# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]


# In[ ]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

print(lda_model.print_topics())


# In[ ]:


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# In[ ]:



df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)


# In[ ]:


from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in data_ready for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
import matplotlib.colors as mcolors
# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()
plt.savefig("final.png")


# In[ ]:


reviews.head()


# In[ ]:


cusine_sort.head()


# In[ ]:




