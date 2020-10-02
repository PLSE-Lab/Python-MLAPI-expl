#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import gensim
import pandas
import nltk.corpus
import nltk.sentiment
import sklearn.linear_model
import textblob
import random
import numpy
import sklearn.metrics
import sklearn.ensemble
import seaborn
import re
import collections

sentence_splitter=re.compile(u"""[.?!]['"]*\s+""",re.UNICODE)


# In order to classify whether articles are Fake News or not, a corpus of news articles from a reliable source is needed. For this kernel, I'm using the Reuters Corpus, which is avaialble from NLTK. Fortunately, this contains a similar number of articles to the Fake News Corpus, thus avoiding balancing issues.
# 
# In order to classify articles as `True` or `False`, I want to avoid the use of any features directly based on the vocabulary of the messages. This is because
# 1. Using vocabulary-based features risks introducing bias into the model. The Reuters Corpus covers different dates, and hence different news stories, from the Fake News corpus, and so this would be likely to mean that vocabulary terms associated with stories found in one corpus but not the other would bias the model.
# 2. Features not based on vocabulary will generalize more easily.
# 3. Features not based on vocabulary will be harder to fake.
# 
# The features I have used are
# 1. Sentence Structure Features. For each sentence in an article, create a string by concatenating the grammatical roles of each word in the sentence, as extracted with the `textblob` library. These features are very sparse, so their dimensionality will be reduced with Latent Semantic Indexing.
# 2. Sentiment features, extracted with VADER.
# 
# Classification uses Logistic Regression and Random Forests.

# In[ ]:


def sentence_structure_features(document):
    return ['_'.join((pos for (word,pos) in sentence.pos_tags))
            for sentence in textblob.blob.TextBlob(document).sentences]


# In[ ]:


class SentenceStructureCorpus(object):
    def __init__(self):
        lies=pandas.read_csv("../input/fake.csv")
        n_lies=lies.shape[0]
        self.vader=nltk.sentiment.vader.SentimentIntensityAnalyzer()
        print("Converting Fake News corpus")
        self.data=[sentence_structure_features('{0}\n{1}'.format(row['title'],row['text']))
                   for (index,row) in lies.iterrows()]
        sentiments=[self.analyse_sentiments('{0}\n{1}'.format(row['title'],row['text']))
                    for (index,row) in lies.iterrows()]
        reuters=nltk.corpus.reuters
        print('Converting Reuters corpus')
        self.data.extend([sentence_structure_features(reuters.raw(fileid))
                          for fileid in reuters.fileids()])
        sentiments.extend([self.analyse_sentiments(reuters.raw(fileid))
                           for fileid in reuters.fileids()])
        self.sentiments=numpy.array(sentiments)
        self.N=len(self.data)
        self.labels=numpy.ones(self.N)
        self.labels[:n_lies]=0
        self.test_sample=random.sample(range(self.N),self.N//10)
        print("Creating dictionary")
        self.dictionary=gensim.corpora.dictionary.Dictionary(self.data)
        
    def __iter__(self):
        return (self.dictionary.doc2bow(document) for document in self.data)
                          
    def analyse_sentiments(self,document):
        valences=numpy.array([[sent['pos'],sent['neg'],sent['neu']]
                             for sent in (self.vader.polarity_scores(sentence)
                                          for sentence in sentence_splitter.split(document))])
        return valences.sum(axis=0)
    
    def training_data(self):
        return [self.dictionary.doc2bow(document) for (i,document) in enumerate(self.data)
                if i not in self.test_sample]
                
    def training_labels(self):
        return self.labels[[i for i in range(self.N) if i not in self.test_sample]]
    
    def training_sentiments(self):
        return self.sentiments[[i for i in range(self.N) if i not in self.test_sample]]
    
    def test_sentiments(self):
        return self.sentiments[self.test_sample]
                
    def test_data(self):
        return [self.dictionary.doc2bow(self.data[i])
                for i in self.test_sample]
            
    def test_labels(self):
        return self.labels[self.test_sample]


# In[ ]:


ssf=SentenceStructureCorpus()
print("Training LSI")
lsi=gensim.models.lsimodel.LsiModel(ssf)


# First, let us train a classifier on the sentence Structure Features

# In[ ]:


vectors=gensim.matutils.corpus2dense(lsi[ssf.training_data()],lsi.num_topics).T
classifier=sklearn.linear_model.LogisticRegression()
print("Training classifier")
classifier.fit(vectors,ssf.training_labels())
print("Testing classifier")
confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),
                                           classifier.predict(gensim.matutils.corpus2dense(lsi[ssf.test_data()],
                                                                                           lsi.num_topics).T))
seaborn.heatmap(confusion,annot=True)


# The confusion matrix shows that almost all `True` articles are correctly classified, but that just over half the `False` articles are classified as `True`

# In[ ]:


def precision(cm):
    return cm[1,1]/cm[:,1].sum()

def recall(cm):
    return cm[1,1]/cm[1].sum()

def accuracy(cm):
    return (cm[0,0]+cm[1,1])/cm.sum()

def matthews(cm):
    return (cm[0,0]*cm[1,1]-cm[1,0]*cm[0,1])/numpy.sqrt(cm[0].sum()*cm[1].sum()*cm[:,0].sum()*cm[:,1].sum())


# Precision is 61%, recall is 96%, accuracy is 70% and Matthews Correlation Coefficient is 50%, ie this classifier is 50% better than guesswork.

# In[ ]:


precision(confusion)


# In[ ]:


recall(confusion)


# In[ ]:


accuracy(confusion)


# In[ ]:


matthews(confusion)


# Now let us try using the sentiments to classifiy the articles.

# In[ ]:


sentiment_classifier=sklearn.linear_model.LogisticRegression()
sentiment_classifier.fit(ssf.training_sentiments(),ssf.training_labels())
confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),
                                           sentiment_classifier.predict(ssf.test_sentiments()))
seaborn.heatmap(confusion,annot=True)


# We can see that this classifier rejects more `False` articles, at the expense of rejecting more `True` ones. Precision is 71%, recall is 88%, accuracy 79% and Matthews 59%.

# In[ ]:


precision(confusion)


# In[ ]:


recall(confusion)


# In[ ]:


accuracy(confusion)


# In[ ]:


matthews(confusion)


# Now let's try combining grammatical and sentiment features.

# In[ ]:


enhanced_vectors=numpy.hstack([vectors,ssf.training_sentiments()])
combined_classifier=sklearn.linear_model.LogisticRegression()
print("Training classifier")
combined_classifier.fit(enhanced_vectors,ssf.training_labels())
print("Testing classifier")
enhanced_test_vectors=numpy.hstack([gensim.matutils.corpus2dense(lsi[ssf.test_data()],
                                                                 lsi.num_topics).T,
                                    ssf.test_sentiments()])
confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),
                                           combined_classifier.predict(enhanced_test_vectors))
seaborn.heatmap(confusion,annot=True)


# Performance in a slight improvement on that of sentiment alone. Precision is 74%, recall is 90%, accuracy is 81% and Matthews is 64%.

# In[ ]:


precision(confusion)


# In[ ]:


recall(confusion)


# In[ ]:


accuracy(confusion)


# In[ ]:


matthews(confusion)


# While combining the two sets of features does give a better result that either set of features alone, it is only a small improvement, and sentiment is still doing most of the work. A more sophisticated means of combining the features may be able to exploit the strengths of both.
# First, let's try Random Forests with sentence structure features.

# In[ ]:


forest0=sklearn.ensemble.RandomForestClassifier(n_estimators=100)
forest0.fit(vectors,ssf.training_labels())
confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),
                                           forest0.predict(gensim.matutils.corpus2dense(lsi[ssf.test_data()],
                                                                                           lsi.num_topics).T))
seaborn.heatmap(confusion,annot=True)


# Precision is 82%, recall is 87%, accuracy is 86% and Matthews correlation is 71%

# In[ ]:


precision(confusion)


# In[ ]:


recall(confusion)


# In[ ]:


accuracy(confusion)


# In[ ]:


matthews(confusion)


# Now let's try Random Forests on sentiments

# In[ ]:


forest1=sklearn.ensemble.RandomForestClassifier(n_estimators=100)
forest1.fit(ssf.training_sentiments(),ssf.training_labels())
confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),
                                           forest1.predict(ssf.test_sentiments()))
seaborn.heatmap(confusion,annot=True)


# Random Forests does slightly worse with sentiments than with sentence structure features. Precision is 76%, recall is 75%, accuracy is 78% and Matthews correlation is 56%. 

# In[ ]:


precision(confusion)


# In[ ]:


recall(confusion)


# In[ ]:


accuracy(confusion)


# In[ ]:


matthews(confusion)


# Now let's see how Random Forests perform with combined sentence structure features and sentiments.

# In[ ]:


forest2=sklearn.ensemble.RandomForestClassifier(n_estimators=100)
forest2.fit(enhanced_vectors,ssf.training_labels())
confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),
                                           forest2.predict(enhanced_test_vectors))
seaborn.heatmap(confusion,annot=True)


# For this classifier, precision is 84%, recall is 88%, accuracy is 87% and Matthews correlation is 74% - slightly (but not significantly) better than sentence structure features alone.

# In[ ]:


precision(confusion)


# In[ ]:


recall(confusion)


# In[ ]:


accuracy(confusion)


# In[ ]:


matthews(confusion)


# One potential problem with Sentence Structure Features is that they are very sparse. Let us examine the distribution of the number of documents a given feature appears in.

# In[ ]:


keys = collections.defaultdict(int)
for doc in ssf:
    for (key,count) in doc:
        keys[key]+=1
pandas.Series(keys).value_counts()


# We see that many features appear only in a single document. Now let us examine the number of features a given document has in common with other documents in the sample.

# In[ ]:


repeated_keys = collections.defaultdict(int)
for doc in ssf:
    repeated_keys[len([key for (key,value) in doc if keys[key]>1])]+=1
pandas.Series(repeated_keys).sort_values()


# From this we can see that around half the documents in the sample have no features in common with any other documents.
# 
# If there were a set of features that only occurred in the Fake News sample, but no features that were unique to the Reuters sample, this would explain the behaviour of the Logistic Regression model with sentence structure features. Truth would be the null hypothesis, which the model would default to if features distinctive to Fake News were not detected.
# 
# Now consider stop words. These are words that mainly carry grammatical rather than semantic meaning, and thus have the same properties of content independence that made sentence structures interesting. They will also be found in all documents, so will give a much denser feature space. It should be noted that they are often used in stylometric analysis to identify the authors of disputed documents.  

# In[ ]:


stopwords = nltk.corpus.stopwords.words("english")
stopwords


# Let us not redefine the Sentence Structure Features function to include stopwords, and rerun the analysis.

# In[ ]:


def sentence_structure_features(document):
    blob = textblob.blob.TextBlob(document)
    return ['_'.join((pos for (word,pos) in sentence.pos_tags))
            for sentence in textblob.blob.TextBlob(document).sentences] +[word.lower() 
                                                                          for word in blob.words
                                                                          if  word.lower() in stopwords]
ssf2 = SentenceStructureCorpus()
transform = gensim.models.LsiModel(ssf2,id2word=ssf2.dictionary)
training_data = gensim.matutils.corpus2dense(transform[ssf2.training_data()],
                                             transform.num_topics).T
test_data = gensim.matutils.corpus2dense(transform[ssf2.test_data()],
                                         transform.num_topics).T
classifier = sklearn.linear_model.LogisticRegression()
classifier.fit(training_data,ssf2.training_labels())
confusion = sklearn.metrics.confusion_matrix(ssf2.test_labels(),
                                             classifier.predict(test_data))
seaborn.heatmap(confusion,annot=True)


# Precision is 90%, Recall is 96%, Accuracy is 93% and Matthews coefficient is 87%

# In[ ]:


precision(confusion)


# In[ ]:


recall(confusion)


# In[ ]:


accuracy(confusion)


# In[ ]:


matthews(confusion)


# This is then an simple and accurate estimate of whether an article is stylistically more similar to fake news or a genuine news source. However, it remains to be see how well it would perform on articles from different sources.
