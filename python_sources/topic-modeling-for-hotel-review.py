#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import pandas


# In[ ]:


import pandas as pd


# In[ ]:


pd.__version__


# In[ ]:


metadata=pd.read_csv('../input/Hotel_Reviews.csv', engine='python')


# In[ ]:


metadata.head(3)


# In[ ]:


metadata['Negative_Review'].head(5)


# In[ ]:


#Replace NaN with an empty string


# In[ ]:


metadata['Negative_Review'] = metadata['Negative_Review'].fillna('')


# In[ ]:


metadata['Negative_Review'].head(10)


# In[ ]:


No_of_seleced_doc=3000


# In[ ]:


df1=metadata['Negative_Review'].iloc[0:No_of_seleced_doc]


# In[ ]:


df1.head(20)


# In[ ]:


df1.shape


# In[ ]:


print("There are {} review comment from {} different nationality, such as {}... \n".format(df1.shape[0],len(metadata.Reviewer_Nationality.unique()),", ".join(metadata.Reviewer_Nationality.unique()[0:5])))


# In[ ]:


#GroupBy by type of Reviewer_Nationality


# In[ ]:


Nationality=metadata[["Reviewer_Score","Reviewer_Nationality","Negative_Review"]].iloc[0:No_of_seleced_doc].groupby("Reviewer_Nationality")


# In[ ]:


#Summary statistic of all nationality


# In[ ]:


Nationality.describe().head(10)


# In[ ]:


Nationality.mean().sort_values(by="Reviewer_Score",ascending=False).head(10)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(15,10))
Nationality.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Nationality of Reviewer")
plt.ylabel("Number of Reviewer")
plt.show()


# In[ ]:


#metadata['Negative_Review']=metadata['Negative_Review'].drop(metadata['Negative_Review'].index[10:])


# In[ ]:


#metadata['Negative_Review']


# In[ ]:


#metadata['Negative_Review'].shape


# In[ ]:


from PIL import Image


# In[ ]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


#Replace NaN with an empty string


# In[ ]:


text = " ".join(review for review in metadata['Negative_Review'])
print ("There are {} words in the combination of all review.".format(len(text)))


# In[ ]:


# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["Nan","Negative","etc"])


# In[ ]:


# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(text)


# In[ ]:


# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


# Save the image in the img folder:
#wordcloud.to_file("../first_review.png")


# In[ ]:


#Tokenization and clean up by gensim's simple preprocess


# In[ ]:


#!pip install gensim


# In[ ]:


import os


# In[ ]:


import gensim


# In[ ]:


get_ipython().run_line_magic('pinfo', 'gensim')


# In[ ]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# In[ ]:


data_words = list(sent_to_words(df1))


# In[ ]:


#print(data_words)


# In[ ]:


print(data_words[:1])


# In[ ]:


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)


# In[ ]:


# Run in terminal: python3 -m spacy download en


# In[ ]:


import re, nltk, spacy


# In[ ]:


nlp = spacy.load('en', disable=['parser', 'ner'])


# In[ ]:


# Do lemmatization keeping only Noun, Adj, Verb, Adverb


# In[ ]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


# In[ ]:


data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[ ]:


print(data_lemmatized[:2])


# In[ ]:


import sklearn


# In[ ]:


sklearn.__version__


# In[ ]:


#import TfIdfVectorizer and CountVectorizer from Scikit-Learn


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[ ]:


# NMF is able to use tf-idf


# In[ ]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english')


# In[ ]:


tfidf=tfidf_vectorizer.fit_transform(data_lemmatized)


# In[ ]:


tfidf.shape


# In[ ]:


tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# In[ ]:


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model


# In[ ]:


tf_vectorizer=CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum read occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{2,}',  # num chars > 2
                            )


# In[ ]:


tf=tf_vectorizer.fit_transform(data_lemmatized)


# In[ ]:


tf.shape


# In[ ]:


tf_feature_names = tf_vectorizer.get_feature_names()


# In[ ]:


# Materialize the sparse data


# In[ ]:


data_dense = tf.todense()


# In[ ]:


# Compute Sparsicity = Percentage of Non-Zero cells


# In[ ]:


print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")


# In[ ]:


from sklearn.decomposition import NMF, LatentDirichletAllocation


# In[ ]:


no_topics = 15


# In[ ]:


# Run NMF


# In[ ]:


nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)


# In[ ]:


# Run LDA


# In[ ]:


# Build LDA Model


# In[ ]:


lda_model = LatentDirichletAllocation(n_components=no_topics,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='batch',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )


# In[ ]:


lda_output = lda_model.fit_transform(tf)


# In[ ]:


print(lda_model)  # Model attributes


# In[ ]:


# Log Likelyhood: Higher the better


# In[ ]:


print("Log Likelihood: ", lda_model.score(tf))


# In[ ]:


# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)


# In[ ]:


print("Perplexity: ", lda_model.perplexity(tf))


# In[ ]:


# Define Search Param


# In[ ]:


search_params = {'n_components': [5, 10, 15, 20], 'learning_decay': [.5, .7, .9]}


# In[ ]:


# Init the Model


# In[ ]:


lda = LatentDirichletAllocation()


# In[ ]:


# Init Grid Search Class


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


model = GridSearchCV(lda, param_grid=search_params)


# In[ ]:


# Do the Grid Search


# In[ ]:


model.fit(tf)


# In[ ]:


# Best Model


# In[ ]:


best_lda_model = model.best_estimator_


# In[ ]:


# Model Parameters


# In[ ]:


print("Best Model's Params: ", model.best_params_)


# In[ ]:


# Log Likelihood Score


# In[ ]:


print("Best Log Likelihood Score: ", model.best_score_)


# In[ ]:


# Perplexity


# In[ ]:


print("Model Perplexity: ", best_lda_model.perplexity(tf))


# In[ ]:


# Get Log Likelyhoods from Grid Search Output


# In[ ]:


gscore=model.fit(tf).cv_results_


# In[ ]:


type(gscore)


# In[ ]:


print(gscore)


# In[ ]:


print(gscore['params'])


# In[ ]:


print(gscore['params'][3]["learning_decay"])


# In[ ]:


print(gscore['mean_test_score'])


# In[ ]:


print(gscore['mean_test_score'][3])


# In[ ]:


print(model.scorer_)


# In[ ]:


#print([gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.7])

n_components = [5, 10, 15, 20]
log_likelyhoods_5 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.5]
#print(log_likelyhoods_5)
log_likelyhoods_7 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.7]
log_likelyhoods_9 = [gscore['mean_test_score'][gscore['params'].index(v)] for v in gscore['params'] if v['learning_decay']==0.9]


# In[ ]:


#import matplotlib as plt


# In[ ]:


#%matplotlib inline


# In[ ]:


# Show graph


# In[ ]:


plt.figure(figsize=(12, 8))
plt.plot(n_components, log_likelyhoods_5, label='0.5')
plt.plot(n_components, log_likelyhoods_7, label='0.7')
plt.plot(n_components, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()


# In[ ]:


# Create Document - Topic Matrix


# In[ ]:


lda_output = best_lda_model.transform(tf)


# In[ ]:


# column names


# In[ ]:


topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]


# In[ ]:


# index names


# In[ ]:


docnames = ["Doc" + str(i) for i in range(len(data_lemmatized))]


# In[ ]:


# Make the pandas dataframe


# In[ ]:


import numpy as np


# In[ ]:


df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)


# In[ ]:


# Get dominant topic for each document


# In[ ]:


dominant_topic = np.argmax(df_document_topic.values, axis=1)


# In[ ]:


df_document_topic['dominant_topic'] = dominant_topic


# In[ ]:


# Styling


# In[ ]:


def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)


# In[ ]:


def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)


# In[ ]:


# Apply Style


# In[ ]:


df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)


# In[ ]:


df_document_topics


# In[ ]:


df_document_topic.info()


# In[ ]:


#start K-means analysis here


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


import seaborn as sns;sns.set()  # for plot styling


# In[ ]:


df_document_topic_k=df_document_topic[["Topic0","Topic1","Topic2","Topic3","Topic4"]]


# In[ ]:


df_document_topic_k.info()


# In[ ]:


K=np.array(df_document_topic_k)


# In[ ]:


# Using the elbow method to find the optimal number of clusters


# In[ ]:


wcssK=[]
distancesK=[]
j=15


# In[ ]:


import math as math


# In[ ]:


from math import sqrt


# In[ ]:


class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])
    
    def distance_to_line(self, x,y):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff*self.x - x_diff*self.y + p2.x*p1.y - p2.y*p1.x)
        den = math.sqrt(y_diff**2 + x_diff**2)
        return num / den 


# In[ ]:


for i in range (1,j):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init = 10, random_state =20)
    kmeans.fit(df_document_topic_k)
    wcssK.append(kmeans.inertia_)


# In[ ]:


print(wcssK)


# In[ ]:


for k in range (1,j):
    p1 = Point(x_init=0,y_init=wcssK[0])
    p2 = Point(x_init=j-1,y_init=wcssK[j-2])
    p = Point(x_init=k-1,y_init=wcssK[k-2])
    distancesK.append(p.distance_to_line(p1,p2))


# In[ ]:


print(distancesK)
print("The maximum distance is ",max(distancesK),"at {}th clustering".format(distancesK.index(max(distancesK))))


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(range(1,j), wcssK)
plt.plot(range(1,j),distancesK)
plt.title("The elbow method_Topic modeling for Hotel Review")
plt.xlabel("The number of clusters")
plt.ylabel("WCSS")
plt.legend(['wcss', 'distance'], loc='upper right')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters = 5, init='k-means++', max_iter = 300, n_init = 10, random_state =20)
a=kmeans.fit(df_document_topic_k).labels_
print(a[0:99])


# In[ ]:


type(kmeans.predict(df_document_topic_k))


# In[ ]:


print(kmeans.predict(df_document_topic_k)[0:299])


# In[ ]:


import matplotlib
matplotlib.__version__


# In[ ]:


plt.figure(figsize=(12, 8))
plt.scatter(df_document_topic_k['Topic0'].iloc[0:299], df_document_topic_k['Topic1'].iloc[0:299],c=kmeans.predict(df_document_topic_k)[0:299],  s=50, )
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,2], c=[0,1,2,3,4], s=500, alpha=0.5);


# In[ ]:


type(centers)


# In[ ]:


print(centers)


# In[ ]:


type(df_document_topic_k['Topic0'].iloc[0:299])


# In[ ]:


b=np.unique(a)


# In[ ]:


for n in b:
    print("Clustering {}".format(n)+" has {} Hotel Review,".format(a.tolist().count(n)))


# In[ ]:


from sklearn.metrics import silhouette_score


# In[ ]:


range_n_clusters = [3, 4, 5, 6, 7]
for nth_clusters in range_n_clusters:
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(df_document_topic_k, kmeans.fit_predict(df_document_topic_k))
    print("For n_clusters =", nth_clusters,
          "The average silhouette_score is :", silhouette_avg)


# In[ ]:


df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")


# In[ ]:


df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")


# In[ ]:


df_topic_distribution


# In[ ]:


import pyLDAvis


# In[ ]:


import pyLDAvis.sklearn


# In[ ]:


pyLDAvis.enable_notebook()


# In[ ]:


pyLDAvis.__version__


# In[ ]:


panel = pyLDAvis.sklearn.prepare(best_lda_model, tf, tf_vectorizer,mds='tsne') #no other mds function like tsne used.


# In[ ]:


panel


# In[ ]:


# Topic-Keyword Matrix


# In[ ]:


df_topic_keywords = pd.DataFrame(best_lda_model.components_/best_lda_model.components_.sum(axis=1)[:,np.newaxis])


# In[ ]:


# Assign Column and Index


# In[ ]:


df_topic_keywords.columns = tf_vectorizer.get_feature_names()


# In[ ]:


df_topic_keywords.index = topicnames


# In[ ]:


# View


# In[ ]:


df_topic_keywords.head(15)


# In[ ]:


# Show top n keywords for each topic


# In[ ]:


def show_lda_topics(lda_model=lda_model, n_words=20):
    keywords = np.array(df_topic_keywords.columns)
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


# In[ ]:


topic_keywords = show_lda_topics(lda_model=best_lda_model, n_words=15)


# In[ ]:


# Topic - Keywords Dataframe


# In[ ]:


df_topic_keywords = pd.DataFrame(topic_keywords)


# In[ ]:


df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]


# In[ ]:


df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]


# In[ ]:


df_topic_keywords


# In[ ]:


#lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)


# In[ ]:


# Topic-Keyword matrix


# In[ ]:


#tf_topic_keywords=pd.DataFrame(lda.components_/lda.components_.sum(axis=1)[:,np.newaxis])


# In[ ]:


no_top_words = 8


# In[ ]:


# Assign Columns and Index


# In[ ]:


#tf_topic_keywords.columns=tf_feature_names


# In[ ]:


#tf_topic_keywords.index=np.arange(0,no_topics)


# In[ ]:


#print(tf_topic_keywords.head())


# In[ ]:


# display topic


# In[ ]:


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


# In[ ]:


display_topics(nmf, tfidf_feature_names, no_top_words)


# In[ ]:


# display lda through weighting


# In[ ]:


# def display_topics(feature_names, no_of_words):
   # for topic_idx, topic in enumerate(tf_topic_keywords):
   #     print ("Topic %d:" % (topic_idx))
   #     print (" ".join([feature_names[i]
   #                     for i in topic.argsort()[:-no_of_words - 1:-1]]))


# In[ ]:


# type(tf_feature_names)


# In[ ]:


# tf_feature_array=np.asarray(tf_feature_names)


# In[ ]:


#display_topics(lda, tf_feature_names, no_top_words)


# In[ ]:


#doc_topic_dist = lda.transform(tf)


# In[ ]:


#print(doc_topic_dist)


# In[ ]:


# lda_perplexity=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).perplexity(tf)


# In[ ]:


# lda_score=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).score(tf)


# In[ ]:


# print("and lda score= "lda_score)


# In[ ]:


# Importing Gensim


# In[ ]:


#import matplotlib as plt


# In[ ]:


#%matplotlib inline


# In[ ]:




