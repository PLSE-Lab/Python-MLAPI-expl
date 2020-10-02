#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# !pip install --upgrade pip
# !pip install pattern --no-dependencies
get_ipython().system('pip install --upgrade pandarallel')
# !pip install swifter
get_ipython().system('pip install wordninja')
get_ipython().system('pip install LDA2Vec')


# In[ ]:


# import swifter
import wordninja
import lda2vec
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk import word_tokenize
import pickle
import heapq
from collections import OrderedDict
import pandarallel
from gensim import corpora, models
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.tokenize import word_tokenize
from gensim.models import doc2vec,Doc2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string, strip_tags,strip_punctuation, strip_multiple_whitespaces,remove_stopwords
from gensim.utils import lemmatize


# In[ ]:


pandarallel.pandarallel.initialize(progress_bar= True)
# clean_df['top_500'] = clean_df['Clean_narrative'].parallel_apply(extract_topn_from_vector)


# # step 1: Load the list of list of token from csv 

# In[ ]:


# clean_df = pd.read_csv('/kaggle/input/lda-list-of-list-of-tokens/LDA_training_corpus.csv')


# In[ ]:


# import ast
# list_of_list_of_tokens = ast.literal_eval(list_of_list_of_tokens)


# In[ ]:


# clean_df['list_of_list_of_tokens']= clean_df['list_of_list_of_tokens'].parallel_apply(ast.literal_eval)


# In[ ]:


# list_of_list_of_tokens = clean_df.list_of_list_of_tokens.to_list()


# In[ ]:


# len(list_of_list_of_tokens)


# # Step 2 : load your dictionary LDA

# In[ ]:


# pickle_in = open('/kaggle/input/lda-dictionary/dictionary_LDA.sav',"rb")
# dictionary_LDA = pickle.load(pickle_in)
# pickle_in.close()


# # Step 3: prepare your list of list of token, dictionary LDA, corpus

# In[ ]:


# # dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
# dictionary_LDA.filter_extremes(no_below=3)
# corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]


# # Step 4 : run your lda multi core model, check coherence

# In[ ]:


# dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
# dictionary_LDA.filter_extremes(no_below=3)
# corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]

# # Model by num_sub_products
# num_topics = 10
# lda_model = models.LdaMulticore(corpus, num_topics=num_topics, \
#                                   id2word=dictionary_LDA, \
#                                   passes=4, alpha=[0.01]*num_topics, \
#                                   eta=[0.01]*len(dictionary_LDA.keys()))
# for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
#     print(str(i)+": "+ topic)
#     print()
# filename = 'lda_model_10_topics.sav'
# pickle.dump(lda_model, open(filename, 'wb'))

# coherence_model_lda = CoherenceModel(model=lda_model, texts=list_of_list_of_tokens, dictionary=dictionary_LDA, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)


# In[ ]:


# # Get topic weights and dominant topics ------------
# from sklearn.manifold import TSNE
# from bokeh.plotting import figure, output_file, show
# from bokeh.models import Label
# from bokeh.io import output_notebook
# import itertools
# import matplotlib.colors as mcolors

# # Get topic weights
# topic_weights = []
# for i, row_list in enumerate(lda_model[corpus]):
#     topic_weights.append([w for i, w in row_list])

# # Array of topic weights    
# arr = pd.DataFrame(topic_weights).fillna(0).values

# # Keep the well separated points (optional)
# arr = arr[np.amax(arr, axis=1) > 0.35]

# # Dominant topic number in each doc
# topic_num = np.argmax(arr, axis=1)

# # tSNE Dimension Reduction
# tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
# tsne_lda = tsne_model.fit_transform(arr)

# # Plot the Topic Clusters using Bokeh
# output_notebook()
# n_topics = 18
# mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

# plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
#               plot_width=500, plot_height=500)
# plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
# show(plot)


# # Load LDA model with 18 topics

# In[ ]:


# pickle_in = open('/kaggle/input/lda-gensim-model-products/lda_model_by_products.sav',"rb")
# LDA_18_topics = pickle.load(pickle_in)
# pickle_in.close()


# In[ ]:


# import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
# vis


# In[ ]:


# pickle_in = open('/kaggle/input/lda-gensim-model-products/lda_model_by_products.sav',"rb")
# LDA_model = pickle.load(pickle_in)
# pickle_in.close()


# In[ ]:


# import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(LDA_model, corpus, dictionary=LDA_model.id2word)
# vis


# In[ ]:


# test_april = pd.read_csv('/kaggle/input/testapril/complaints-2020-05-07_09_29.csv')


# In[ ]:


# test_lolotokens = test_april['Consumer complaint narrative'].parallel_apply(word_tokenize)


# In[ ]:


# test_corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in test_lolotokens.to_list()]


# In[ ]:


# test_april.columns


# In[ ]:


# # def LDA_predict(text):
# #     tokens = word_tokenize(new_text)
# for i,row_list in enumerate(LDA_model[dictionary.doc2bow[tokens]])
# LDA_model[dictionary_LDA.doc2bow(tokens)]


# In[ ]:


# def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
#     # Init output
#     sent_topics_df = pd.DataFrame()

#     # Get main topic in each document
#     for i, row_list in enumerate(LDA_model[test_corpus]):
#         row = row_list[0] if ldamodel.per_word_topics else row_list            
#         # print(row)
#         row = sorted(row, key=lambda x: (x[1]), reverse=True)
#         # Get the Dominant topic, Perc Contribution and Keywords for each document
#         for j, (topic_num, prop_topic) in enumerate(row):
#             if j == 0:  # => dominant topic
#                 wp = ldamodel.show_topic(topic_num)
#                 topic_keywords = ", ".join([word for word, prop in wp])
#                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
#             else:
#                 break
#     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

#     # Add original text to the end of the output
#     contents = pd.Series(texts)
#     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
#     return(sent_topics_df)


# df_topic_sents_keywords = format_topics_sentences(ldamodel=LDA_model, corpus=corpus, texts=test_lolotokens.to_list())

# # Format
# df_dominant_topic = df_topic_sents_keywords.reset_index()
# df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
# df_dominant_topic.head(10)


# In[ ]:


# df_dominant_topic.to_csv('LDA_prediction_on_april.csv')


# In[ ]:


# coherence_model_lda = CoherenceModel(model=_model, texts=list_of_list_of_tokens, dictionary=dictionary_LDA, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)


# # Load LDA model with 5 topics

# In[ ]:


# pickle_in = open('/kaggle/input/lda-model-5-topics/lda_model_5_topics.sav',"rb")
# LDA_model_5_topics = pickle.load(pickle_in)
# pickle_in.close()


# In[ ]:


# import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(LDA_model_5_topics, corpus, dictionary=LDA_model_5_topics.id2word)
# vis


# In[ ]:


# def topics_per_document(model, corpus, start=0, end=1):
#     corpus_sel = corpus[start:end]
#     dominant_topics = []
#     topic_percentages = []
#     for i, corp in enumerate(corpus_sel):
#         topic_percs, wordid_topics, wordid_phivalues = model[corp]
#         dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
#         dominant_topics.append((i, dominant_topic))
#         topic_percentages.append(topic_percs)
#     return(dominant_topics, topic_percentages)

# dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            

# # Distribution of Dominant Topics in Each Document
# df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
# dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
# df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# # Total Topic Distribution by actual weight
# topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
# df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# # Top 3 Keywords for each Topic
# topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
#                                  for j, (topic, wt) in enumerate(topics) if j < 3]

# df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
# df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
# df_top3words.reset_index(level=0,inplace=True)


# In[ ]:


# from matplotlib.ticker import FuncFormatter

# # Plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

# # Topic Distribution by Dominant Topics
# ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
# ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
# tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
# ax1.xaxis.set_major_formatter(tick_formatter)
# ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
# ax1.set_ylabel('Number of Documents')
# ax1.set_ylim(0, 1000)


# In[ ]:


# complaints with low contri get all contris from 18 topics, hypertune with 50k complaints, runtime  5 min, clean test and run prediction, evaluate, work on 15 months data
# check if prod sub prod is same across 15 months


# ## loading doc2vec docs and model 

# In[ ]:


documents = pd.read_csv('/kaggle/input/tagged-docs-model/complaints-2020-06-13_12_31.csv')


# In[ ]:


import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

def clean_doc(x):
    x = " ".join(wordninja.split(x))
    x = nlp(x)
    # x = ' '.join([x.decode('utf-8')[:-3] for x in lemmatize(x,stopwords={'xx','xxxx'})])
    x = " ".join([token.lemma_.lower() for token in x if token.lemma_ not in ['XX','XXXX','xxxx','xx','s','00','-PRON-']])
    # x = ' '.join(list(map(lambda x: x.decode('utf-8')[:-3], lemmatize(x,stopwords={'xx','xxxx'}))))
    x = preprocess_string(x,[strip_tags,strip_punctuation,strip_multiple_whitespaces,remove_stopwords])
    return x


# In[ ]:


get_ipython().run_cell_magic('time', '', "# clean_doc(documents['Consumer complaint narrative'][2])\n# wordninja.split(documents['Consumer complaint narrative'][0])")


# In[ ]:


# %%time
# train_docs = documents['Consumer complaint narrative'].parallel_apply(clean_doc)
# train_docs = list(train_docs)
# tagged_docs = [doc2vec.TaggedDocument(train_docs[x], [x]) for x in range(0,len(train_docs))]
# tagged_docs


# In[ ]:


# pickle_out = open('train_docs.pkl','wb')
# pickle.dump(tagged_docs,pickle_out)


# In[ ]:


pickle_in = open('/kaggle/input/tagged-docs-model/train_docs(1).pkl','rb')
tagged_docs = pickle.load(pickle_in)


# In[ ]:


len(tagged_docs)


# ## choosing embedding size
# https://stackoverflow.com/questions/48479915/what-is-the-preferred-ratio-between-the-vocabulary-size-and-embedding-dimension
# https://stackoverflow.com/questions/50945820/paragraph-vector-or-doc2vec-model-size
# https://stackoverflow.com/questions/51245689/what-does-epochs-mean-in-doc2vec-and-train-when-i-have-to-manually-run-the-itera
# 
# #### Higher dimension can lead to overfitting on smaller corpuses
# #### best dimension size is vocabulary_size**0.25

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = doc2vec.Doc2Vec(vector_size=50, min_count=5, epochs=30, workers=4)\nmodel.build_vocab(tagged_docs)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)\nmodel.save('custom_Doc2vec_50d_116k.pkl')")


# In[ ]:


model = doc2vec.Doc2Vec.load('/kaggle/input/tagged-docs-model/custom_Doc2vec_50d.pkl')
# model = doc2vec.Doc2Vec.load('/kaggle/input/tagged-docs-model/custom_Doc2vec_200d.pkl')
# model = doc2vec.Doc2VecKeyedVectors.load('/kaggle/input/google-doc2vec/GoogleNews-vectors-negative300.bin')


# In[ ]:


ranks = []
second_ranks = []
for doc_id in range(len(tagged_docs)):
    inferred_vector = model.infer_vector(tagged_docs[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])


# In[ ]:


import collections

counter = collections.Counter(ranks)
print(counter)


# In[ ]:


x = model.infer_vector(tagged_docs[6].words)
model.docvecs.most_similar([x], topn=5)


# In[ ]:


documents['Consumer complaint narrative'][5],documents['Consumer complaint narrative'][91852]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from scipy import spatial\n\nvec1 = model.infer_vector(tagged_docs[900].words)\nvec2 = model.infer_vector(tagged_docs[31720].words)\nsimilarity = spatial.distance.cosine(vec1, vec2)\n1-similarity\n\n')


# In[ ]:


tagged_docs[0].words


# In[ ]:


# x = model.infer_vector(tagged_docs[0].words)
# model.docvecs.most_similar([x], topn=5)
# tagged_docs[0].tags
# pd.pivot_table(data=documents,index=['Product','Sub-product',"Issue"],values=['clean_narrative'],aggfunc=lambda x: len(x.unique())).to_csv('class_distribution.csv')
pd.pivot_table(data=documents,index=['Sub-product',"Issue",'Sub-issue'],values=['Consumer complaint narrative'],aggfunc=lambda x: len(x.unique())).sort_values(by='Consumer complaint narrative').to_csv('class_distribution.csv')


# In[ ]:


# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# kmeans_model = KMeans(n_clusters=5, init='k-means++', max_iter=100) 
# X = kmeans_model.fit(model.docvecs.doctag_syn0)
# labels=kmeans_model.labels_.tolist()

# l = kmeans_model.fit_predict(model.docvecs.doctag_syn0)
# pca = PCA(n_components=4).fit(model.docvecs.doctag_syn0)
# datapoint = pca.transform(model.docvecs.doctag_syn0)

# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.figure
# label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#FF00FF","#98FB98",'#006400' , '#008080']
# # label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#FF00FF"]
# color = [label1[i] for i in labels]
# plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

# centroids = kmeans_model.cluster_centers_
# centroidpoint = pca.transform(centroids)
# plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
# plt.show()


# In[ ]:


# PCA_components = pd.DataFrame(datapoint)


# In[ ]:


# ## elbow is between 4 to 8
# ks = range(1, 20)
# inertias = []
# for k in ks:
#     # Create a KMeans instance with k clusters: model
#     elbow_model = KMeans(n_clusters=k)
    
#     # Fit model to samples
#     elbow_model.fit(PCA_components.iloc[:,:4])
    
#     # Append the inertia to the list of inertias
#     inertias.append(elbow_model.inertia_)
    
# plt.plot(ks, inertias, '-o', color='black')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(ks)
# plt.show()


# In[ ]:


# PCA_components['labels'] = labels


# In[ ]:


# PCA_components.to_csv('doc2vec_results.csv')


# In[ ]:


# PCA_components.head()


# In[ ]:


# import plotly.graph_objs as go
# from plotly.offline import iplot, init_notebook_mode

# trace1 = go.Scatter3d(
#     x=PCA_components[0],
#     y=PCA_components[1],
#     z=PCA_components[2],
#     mode='markers',
#     marker=dict(
#         size=10,
#         color=PCA_components['labels'],                # set color to an array/list of desired values      
#     )
# )

# data = [trace1]
# layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0  
#     )
    
# )
# fig = go.Figure(data=data, layout=layout)
# iplot(fig)


# In[ ]:


# PCA_components['complaints'] = documents['Consumer complaint narrative']


# In[ ]:


# PCA_components.to_csv('Doc2Vec_similarity.csv')


# # test on april
# # precision recall/ auc score
# # confusion matrix

# In[ ]:


# documents.columns

# documents['complaint_issues'] = documents['Sub-product']+" "+documents['Issue']+' '+documents['Sub-issue']
# documents['complaint_issues'] = documents['Issue']+' '+documents['Sub-issue']
documents['complaint_issues'] = documents['Issue']


# In[ ]:


import ast
documents['issue_tokens'] = documents[['Issue']].values.tolist()
# documents['issue_tokens'] = documents[['Issue','Sub-issue']].values.tolist()
issue_tokens = [ast.literal_eval(x) for x in documents['issue_tokens'].astype(str).unique()]


# In[ ]:


full_issue = documents['complaint_issues'].unique()
all_issues = [clean_doc(x) for x in documents['complaint_issues'].unique()]


# In[ ]:


# issue_tokens
all_issues


# In[ ]:


len(all_issues)


# In[ ]:


# def normalize(word_vec):
#     norm=np.linalg.norm(word_vec)
#     if norm == 0: 
#         return word_vec
#     else:
#         return word_vec/norm

# list_of_issue_vectors = [normalize(x) for x in list_of_issue_vectors]


# In[ ]:


list_of_issue_vectors = [model.infer_vector(x) for x in all_issues]


# In[ ]:


# # test_vector = normalize(model.infer_vector(tagged_docs[0].words))
# test_vector = model.infer_vector(tagged_docs[0].words)
# ([metrics.pairwise.cosine_similarity([test_vector],[i]) for i in list_of_issue_vectors])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from scipy import spatial\ndef issue_predict(x):\n    test_vector = model.infer_vector(x)\n    score = [1 - spatial.distance.cosine(test_vector, i) for i in list_of_issue_vectors]\n    return issue_tokens[score.index(max(score))], score[score.index(max(score))]')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn import metrics\ndef issue_pred2(x):\n    test_vector = model.infer_vector(x)\n    score = [metrics.pairwise.cosine_similarity([test_vector], [i]) for i in list_of_issue_vectors]\n    # return issue_tokens[score.index(max(score))], score[score.index(max(score))]\n    return issue_tokens[score.index(max(score))]')


# In[ ]:


x=5
issue_predict(tagged_docs[x].words),issue_pred2(tagged_docs[x].words),documents['Consumer complaint narrative'][x]


# In[ ]:


issue_predict(tagged_docs[2].words),' '.join(tagged_docs[2].words)


# In[ ]:


issue_predict(tagged_docs[3].words),' '.join(tagged_docs[3].words)


# In[ ]:


issue_predict(tagged_docs[4].words),' '.join(tagged_docs[4].words)


# In[ ]:


issue_predict(tagged_docs[6].words),documents['Consumer complaint narrative'][6]


# In[ ]:


get_ipython().run_cell_magic('time', '', "issue_predict(tagged_docs[7].words),documents['Consumer complaint narrative'][7]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "issue_predict(tagged_docs[8].words),documents['Consumer complaint narrative'][8]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "issue_predict(tagged_docs[9].words),documents['Consumer complaint narrative'][9]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "issue_predict(tagged_docs[10].words),documents['Consumer complaint narrative'][10]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "issue_predict(tagged_docs[11].words),documents['Consumer complaint narrative'][11]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "issue_predict(tagged_docs[12].words),documents['Consumer complaint narrative'][12]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "issue_predict(tagged_docs[13].words),documents['Consumer complaint narrative'][13]")


# In[ ]:


# from sklearn import metrics
metrics.pairwise.cosine_similarity([test_vector],[list_of_issue_vectors[3]])
# list_of_issue_vectors

# cos_sim = dot(a, b)/(norm(a)*norm(b))


# ## stress testing

# ## bad example 1 

# In[ ]:



x = "i have late payment charges on my credit report"
issue_predict(word_tokenize(x)),x


# ## good example 1

# In[ ]:


x = "i need better answer about why I have unknown accounts on my credit report"
issue_predict(word_tokenize(x)),x


# ## can't comment example 1

# In[ ]:


x = "i'm a victim of fraud"
issue_predict(word_tokenize(x)),x


# ## bad example 2

# In[ ]:


x = "because of unknown accounts on my credit report my credit score dropped by 200 points"
issue_predict(word_tokenize(x)),x


# ## okayish example 1 incomplete match

# In[ ]:


x = "my credit score dropped by 200 points"
issue_predict(word_tokenize(x)),x


# # good, partial match example 2

# In[ ]:


x = "one customer representative harassed me over call. I'm ging to file a lawsuit"
issue_predict(word_tokenize(x)),x


# ## bad example 3

# In[ ]:


x = "I'm happy with the bank but the customer service is really bad"
issue_predict(word_tokenize(x)),x


# ## good example 3

# In[ ]:


x = "You closed my paytm account, I need my money back"
issue_predict(word_tokenize(x)),x


# ## bad example 4

# In[ ]:


x = "credit report disputes"
issue_predict(word_tokenize(x)),x


# ## good example 4

# In[ ]:


x = "will you be happy to see late payment charges in your acount?"
issue_predict(word_tokenize(x)),x


# ## bad example 5

# In[ ]:


x = "I used to like amazon prime but I need refund for my netflix account. I am facing hardships because of your service. Can I get a credit card"
issue_predict(word_tokenize(x)),x


# ## bad example 6

# In[ ]:


x = "My credit card was stolen. I closed it but now I can't pay all my bills what to do"
issue_predict(word_tokenize(x)),x


# ## partial match example 3

# In[ ]:


x = "the day is really beautiful today but I am feeling it is going to rain for sure. My house can get damaged, I'm worried if my bank will allow deferment"
issue_predict(word_tokenize(x)),x


# ## partial match example 4

# In[ ]:


x = "hey my name is John Doe today I went to a mall some one pick pocketed my wallet all my cards and cash were lost. Can you close them all"
issue_predict(word_tokenize(x)),x


# ## partial match example 5

# In[ ]:


x = "I think someone else is using my details like ssn to create fake accounts. On amazon I have lost dollar 5000"
issue_predict(word_tokenize(x)),x


# ## bad example 7

# In[ ]:


x='my husband died. I want to transfer his money to my account please help'
issue_predict(word_tokenize(x)),x


# ## bad example 8, partial match 6

# In[ ]:


x="I never openend a Digi Bank wallet. I'm getting messages for collections. Someone stole my information please report this account"
issue_predict(word_tokenize(x)),x


# ## bad example 9, partial match 7

# In[ ]:


x='i need medical assistance please send ambulance asap'
issue_predict(word_tokenize(x)),x


# ## bad example 10

# In[ ]:


x="there's a pothole on road i'm really frustrated i've been calling the sewer company since last 5 months but no one responds on time"
issue_predict(word_tokenize(x)),x


# ## bad example 11

# In[ ]:


x="i bought an item online but my bank denied payment"
issue_predict(word_tokenize(x)),x


# ## good example 5

# In[ ]:


x="I'm not going to recommend PNC bank to my friends. I raised a request to stop debt collections calls but still I'm getting way too many"
issue_predict(word_tokenize(x)),x


# ## partial match 8

# In[ ]:


x="can i stop paying emi for damaged car"
issue_predict(word_tokenize(x)),x


# ## bad example 12

# In[ ]:


x="my cheques are bouncing even when my account has huge amount of money in it"
issue_predict(word_tokenize(x)),x


# ## good match 6 taken from corpus

# In[ ]:


x = " I am using citi credit card, I'm facing problem with a purchase shown on my statement, bank isn't resolving a dispute about a purchase on my statement"
issue_predict(word_tokenize(x)),x


# # good example 7, partial match 9

# In[ ]:


x = "due to covid-19, I'm facing hardship not sure if I can repay my loan"
issue_predict(word_tokenize(x)),x


# In[ ]:


test_data = pd.read_csv('/kaggle/input/testapril/complaints-2020-05-07_09_29.csv')


# In[ ]:


test_data['Company response to consumer'].value_counts()


# ## testing on april data

# In[ ]:


test_data.head(25)


# In[ ]:


test_result = test_data['Consumer complaint narrative'].parallel_apply(word_tokenize).parallel_apply(issue_pred2)


# In[ ]:


vocab = [item for sublist in tagged_docs for item in sublist]
vocab = [item for sublist in vocab for item in sublist]
len(vocab)


# In[ ]:


test_actual = pd.Series(test_data[['Sub-product','Issue','Sub-issue']].values.tolist(), name='narrative')


# In[ ]:


final_result = test_result.to_frame().join(test_actual)
final_result


# In[ ]:


final_result[final_result['narrative']==final_result['Consumer complaint narrative']].shape


# In[ ]:


# test_result = test_data['Consumer complaint narrative'].parallel_apply(word_tokenize).parallel_apply(issue_pred2)
# test_actual = pd.Series(test_data[['Issue','Sub-issue']].values.tolist(), name='narrative')
# final_result = test_result.to_frame().join(test_actual)
# final_result


# In[ ]:


final_result[final_result['narrative']==final_result['Consumer complaint narrative']].shape


# In[ ]:


# test_result = test_data['Consumer complaint narrative'].parallel_apply(word_tokenize).parallel_apply(issue_pred2)
# test_actual = pd.Series(test_data[['Issue','Sub-issue']].values.tolist(), name='narrative')
# final_result = test_result.to_frame().join(test_actual)
# final_result


# In[ ]:


test_result = test_data['Consumer complaint narrative'].parallel_apply(word_tokenize).parallel_apply(issue_pred2)
test_actual = pd.Series(test_data[['Issue']].values.tolist(), name='narrative')
final_result = test_result.to_frame().join(test_actual)
final_result[final_result['narrative']==final_result['Consumer complaint narrative']].shape


# In[ ]:




