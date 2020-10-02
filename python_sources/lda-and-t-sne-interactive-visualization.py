#!/usr/bin/env python
# coding: utf-8

# # NIPS: Topic modeling visualization
# 
# Some main topics at NIPS according to [wikipedia](https://en.wikipedia.org/wiki/Conference_on_Neural_Information_Processing_Systems) :
# 
# 1. Machine learning, 
# 2. Statistics, 
# 3. Artificial intelligence, 
# 4. Computational neuroscience
# 
# However, the topics are within the same domain which makes it more challenging to distinguish between them. Here in this Kernel I will try to extract some topics using Latent Dirichlet allocation __LDA__. This tutorial features an end-to-end  natural language processing pipeline, starting with raw data and running through preparing, modeling, visualizing the paper. We'll touch on the following points
# 
# 
# 1. Topic modeling with **LDA**
# 1. Visualizing topic models with **pyLDAvis**
# 1. Visualizing LDA results with **t-SNE** and **bokeh**

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')

import pandas as pd
import pickle as pk
from scipy import sparse as sp


# In[ ]:


p_df = pd.read_csv('../input/Papers.csv')
docs = array(p_df['PaperText'])


# ## Pre-process and vectorize the documents

# In[ ]:


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]
    
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]
    
    # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
  
    return docs


# In[ ]:


docs = docs_preprocessor(docs)


# ### **Compute bigrams/trigrams:**
# Sine topics are very similar what would make distinguish them are phrases rather than single/individual words.

# In[ ]:


from gensim.models import Phrases
# Add bigrams and trigrams to docs (only ones that appear 10 times or more).
bigram = Phrases(docs, min_count=10)
trigram = Phrases(bigram[docs])

for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
    for token in trigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)


# ### **Remove rare and common tokens:**

# In[ ]:


from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)
print('Number of unique words in initital documents:', len(dictionary))

# Filter out words that occur less than 10 documents, or more than 20% of the documents.
dictionary.filter_extremes(no_below=10, no_above=0.2)
print('Number of unique words after removing rare and common words:', len(dictionary))


# Pruning the common and rare words, we end up with only about 6% of the words.

# ** Vectorize data:**  
# The first step is to get a back-of-words representation of each doc.

# In[ ]:


corpus = [dictionary.doc2bow(doc) for doc in docs]


# In[ ]:


print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# With the bag-of-words corpus, we can move on to learn our topic model from the documents.

# # Train LDA model...

# In[ ]:


from gensim.models import LdaModel


# In[ ]:


# Set training parameters.
num_topics = 4
chunksize = 500 # size of the doc looked at every pass
passes = 20 # number of passes through documents
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

get_ipython().run_line_magic('time', "model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize,                        alpha='auto', eta='auto',                        iterations=iterations, num_topics=num_topics,                        passes=passes, eval_every=eval_every)")


#  # How to choose the number of topics? 
# __LDA__ is an unsupervised technique, meaning that we don't know prior to running the model how many topics exits in our corpus. Topic coherence, is one of the main techniques used to deestimate the number of topics. You can read about it [here.](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)
# 
# However, I used the LDA visualization tool **pyLDAvis**, tried a few number of topics and compared the resuls. Four seemed to be the optimal number of topics that would seperate  topics the most. 

# In[ ]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[ ]:


pyLDAvis.gensim.prepare(model, corpus, dictionary)


# ** What do we see here? **
# 
# **The left panel**, labeld Intertopic Distance Map, circles represent different topics and the distance between them. Similar topics appear closer and the dissimilar topics farther.
# The relative size of a topic's circle in the plot corresponds to the relative frequency of the topic in the corpus.
# An individual topic may be selected for closer scrutiny by clicking on its circle, or entering its number in the "selected topic" box in the upper-left.
#  
# **The right panel**, include the bar chart of the top 30 terms. When no topic is selected in the plot on the left, the bar chart shows the top-30 most "salient" terms in the corpus. A term's saliency is a measure of both how frequent the term is in the corpus and how "distinctive" it is in distinguishing between different topics.
# Selecting each topic on the right, modifies the bar chart to show the "relevant" terms for the selected topic. 
# Relevence is defined as in footer 2 and can be tuned by parameter $\lambda$, smaller $\lambda$ gives higher weight to the term's distinctiveness while larger $\lambda$s corresponds to probablity of the term occurance per topics. 
# 
# Therefore, to get a better sense of terms per topic we'll use  $\lambda$=0.

# **How to evaluate our model?**  
# So again since there is no ground through here, we have to be creative in defining ways to evaluate. I do this in two steps:
# 
# 1. divide each document in two parts and see if the topics assign to them are simialr. => the more similar the better
# 2. compare randomly chosen docs with each other. => the less similar the better

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity

p_df['tokenz'] = docs

docs1 = p_df['tokenz'].apply(lambda l: l[:int0(len(l)/2)])
docs2 = p_df['tokenz'].apply(lambda l: l[int0(len(l)/2):])


# Transform the data 

# In[ ]:


corpus1 = [dictionary.doc2bow(doc) for doc in docs1]
corpus2 = [dictionary.doc2bow(doc) for doc in docs2]

# Using the corpus LDA model tranformation
lda_corpus1 = model[corpus1]
lda_corpus2 = model[corpus2]


# In[ ]:


from collections import OrderedDict
def get_doc_topic_dist(model, corpus, kwords=False):
    
    '''
    LDA transformation, for each doc only returns topics with non-zero weight
    This function makes a matrix transformation of docs in the topic space.
    '''
    top_dist =[]
    keys = []

    for d in corpus:
        tmp = {i:0 for i in range(num_topics)}
        tmp.update(dict(model[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [array(vals)]
        if kwords:
            keys += [array(vals).argmax()]

    return array(top_dist), keys


# In[ ]:


top_dist1, _ = get_doc_topic_dist(model, lda_corpus1)
top_dist2, _ = get_doc_topic_dist(model, lda_corpus2)

print("Intra similarity: cosine similarity for corresponding parts of a doc(higher is better):")
print(mean([cosine_similarity(c1.reshape(1, -1), c2.reshape(1, -1))[0][0] for c1,c2 in zip(top_dist1, top_dist2)]))

random_pairs = np.random.randint(0, len(p_df['PaperText']), size=(400, 2))

print("Inter similarity: cosine similarity between random parts (lower is better):")
print(np.mean([cosine_similarity(top_dist1[i[0]].reshape(1, -1), top_dist2[i[1]].reshape(1, -1)) for i in random_pairs]))


# ## Let's look at the terms that appear more in each topic. 

# In[ ]:


def explore_topic(lda_model, topic_number, topn, output=True):
    """
    accept a ldamodel, atopic number and topn vocabs of interest
    prints a formatted list of the topn terms
    """
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if output:
            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))
    
    return terms


# In[ ]:


topic_summaries = []
print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')
for i in range(num_topics):
    print('Topic '+str(i)+' |---------------------\n')
    tmp = explore_topic(model,topic_number=i, topn=10, output=True )
#     print tmp[:5]
    topic_summaries += [tmp[:5]]
    print


# From above, it's possible to inspect each topic and assign a human-interpretable label to it. Here I labeled them as follows:

# In[ ]:


top_labels = {0: 'Statistics', 1:'Numerical Analysis', 2:'Online Learning', 3:'Deep Learning'}


# In[ ]:


import re
import nltk

from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

def paper_to_wordlist( paper, remove_stopwords=True ):
    '''
        Function converts text to a sequence of words,
        Returns a list of words.
    '''
    lemmatizer = WordNetLemmatizer()
    # 1. Remove non-letters
    paper_text = re.sub("[^a-zA-Z]"," ", paper)
    # 2. Convert words to lower case and split them
    words = paper_text.lower().split()
    # 3. Remove stop words
    words = [w for w in words if not w in stops]
    # 4. Remove short words
    words = [t for t in words if len(t) > 2]
    # 5. lemmatizing
    words = [nltk.stem.WordNetLemmatizer().lemmatize(t) for t in words]

    return(words)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tvectorizer = TfidfVectorizer(input='content', analyzer = 'word', lowercase=True, stop_words='english',                                  tokenizer=paper_to_wordlist, ngram_range=(1, 3), min_df=40, max_df=0.20,                                  norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)

dtm = tvectorizer.fit_transform(p_df['PaperText']).toarray()


# In[ ]:


top_dist =[]
for d in corpus:
    tmp = {i:0 for i in range(num_topics)}
    tmp.update(dict(model[d]))
    vals = list(OrderedDict(tmp).values())
    top_dist += [array(vals)]


# In[ ]:


top_dist, lda_keys= get_doc_topic_dist(model, corpus, True)
features = tvectorizer.get_feature_names()


# In[ ]:


top_ws = []
for n in range(len(dtm)):
    inds = int0(argsort(dtm[n])[::-1][:4])
    tmp = [features[i] for i in inds]
    
    top_ws += [' '.join(tmp)]
    
p_df['Text_Rep'] = pd.DataFrame(top_ws)
p_df['clusters'] = pd.DataFrame(lda_keys)
p_df['clusters'].fillna(10, inplace=True)

cluster_colors = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'red', 4: 'skyblue', 5:'salmon', 6:'orange', 7:'maroon', 8:'crimson', 9:'black', 10:'gray'}

p_df['colors'] = p_df['clusters'].apply(lambda l: cluster_colors[l])


# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(top_dist)


# In[ ]:


p_df['X_tsne'] =X_tsne[:, 0]
p_df['Y_tsne'] =X_tsne[:, 1]


# In[ ]:


from bokeh.plotting import figure, show, output_notebook, save#, output_file
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource
output_notebook()


# In[ ]:


source = ColumnDataSource(dict(
    x=p_df['X_tsne'],
    y=p_df['Y_tsne'],
    color=p_df['colors'],
    label=p_df['clusters'].apply(lambda l: top_labels[l]),
#     msize= p_df['marker_size'],
    topic_key= p_df['clusters'],
    title= p_df[u'Title'],
    content = p_df['Text_Rep']
))


# In[ ]:


title = 'T-SNE visualization of topics'

plot_lda = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

plot_lda.scatter(x='x', y='y', legend='label', source=source,
                 color='color', alpha=0.8, size=10)#'msize', )

# hover tools
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips = {"content": "Title: @title, KeyWords: @content - Topic: @topic_key "}
plot_lda.legend.location = "top_left"

show(plot_lda)

#save the plot
# save(plot_lda, '{}.html'.format(title))


# **What's next?**  
# Given this topics, one may ask, which topic has been more studied or refered to since NIPS15? For this I extracted and included the citaiton informatin of NIPS'15 papers. I wasn't You can find it at [my github account](https://github.com/ykhorram/nips2015_topic_network_analysis/blob/master/NIP15_topics_citations.ipynb). 
# 
# Please let me know if you have any questions or comment.
