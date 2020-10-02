#!/usr/bin/env python
# coding: utf-8

# 

# # News exploration
# 
# In this notebook we are going to work on the news dataset provided by the assignment. Our goals are two fold:
# 
# * Predict an article's category using only its headline.
# * Exlore what the data can tell us about the world's state at different points in time. 
# 
# Before diving into the specific questions, let's start with some EDA to shed some light into the dataset's properties and interesting features.
# 
# # Table of contents
# 
# 1. [EDA](#eda)
#     * [Date](#date)
#     * [Category](#category)
#     * [Publisher](#publisher)
#     
# 2. [Predicting Categories](#predict)
#     * [Classification](#classification)
#     * [Addressing Overfitting](#overfitting)
#     * [Other Metrics](#other-metrics)   
#     
# 3. [Discovering Trends](#new-trends)
#     * [Topic Modeling](#topic-modeling)
#     * [Unseen Documents](#unseen-documents)
#     * [Conclusions](#gensim-conclusions)

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import multiprocessing

news = pd.read_csv("../input/uci-news-aggregator.csv")

# Convert date
news['TIMESTAMP'] = pd.to_datetime(news['TIMESTAMP'], unit='ms')

# Let's take a look at our interesting columns.
news[["TITLE", "PUBLISHER", "CATEGORY", "HOSTNAME", "TIMESTAMP"]].head()


# # Exploratory Data Analysis <a name="eda"></a>
# 
# Let's check some descriptive statistics for each of our columns. This will help us discover outliers and focus our attention on interesting features for the rest of this study.
# 
# ## Date <a name="date"></a>
# 
# We can start with the date feature (initially extracted from the `TIMESTAMP` column. When were these articles published?

# In[ ]:


def date_printer(date):
    return "{}/{}/{}".format(date.day, date.month, date.year)

start, end = news['TIMESTAMP'].min(), news['TIMESTAMP'].max() 
print("Our dataset timeline starts at {} and ends at {}".format(date_printer(start), date_printer(end)))


# ### Perhaps more articles are published during specific months
# 
# Let's check the count of published articles per month. Since our dataset does not start on the first of March, nor does it end on the 31th of August, we ought to normalize for the number of existing days first.
# 

# In[ ]:


news['MONTH'] = news['TIMESTAMP'].apply(lambda date: date.month)
news['DAY'] = news['TIMESTAMP'].apply(lambda date: date.day)

# Some months have 30 and others have 30 days. The first and last months in our dataset and not whole.
month_days = {
    3: 21,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 27
}
articles_per_day = {}
for month in month_days:
    n_articles = len(news[news['MONTH'] == month])
    articles_per_day[month] = n_articles / month_days[month]

ax = sns.barplot(x=list(articles_per_day.keys()), y=list(articles_per_day.values()))
ax.set_title("Normalized Counts")
ax.set_xlabel("Month")
ax.set_ylabel("Articles Per Day")
plt.show()


# ### Unexpected low on August
# 
# There is a significant reduction in the number of articles published daily during the summer months which could be attributed to journalists going on vacation. The extreme low on August might be caused by the same phenomenon, or by missing data. In order to interpret this, we should check the source of data. For example, does the provider guarantee that he/she provides ALL the articles produced by the existing publishers? Or is it just a random subset? Without any further info, I would personally assume a custom scraper which has a certain chance of breaking (for example if an API breaks or the HTML layout changes)

# ## Category <a name="category"></a>
# 
# We know that there are 4 different categories as well as their real meaning. Let's explore their distribution

# In[ ]:


cat_map = {
    'b': 'Business',
    't': 'Science',
    'e': 'Entertainment',
    'm': 'Health'
}
ax = sns.countplot(news['CATEGORY'])
ax.set_title("Category Counts")
ax.set_xlabel("Category")
# Manipulate the labels to make them more readable
ax.set_xticklabels([cat_map[x.get_text()] for x in ax.get_xticklabels()], rotation=45)
plt.show()


# As we can see, the category feature is rather balanced, with the only exception of `Health` which is a bit low in comparison to other values.

# ## Publisher <a name="publisher"></a>
# 
# How many different publishers do we have, and how often do they publish?

# In[ ]:


from collections import Counter

# Byte magic to style print output
def emphasize(s):
    """Bold the string to help get the print reader's attention.
    
    Parameters
    ----------
    s : str
        String to be decorated with bold.
    
    Returns
    -------
    str
        The string in bold.
    """
    red = '\x1b[1;31m'
    stop = '\x1b[0m'
    return red + str(s) + stop

nunique = news['PUBLISHER'].nunique()
print("There are {} different publishers. Below some of the most common:".format(emphasize(nunique)))
for key, value in Counter(news['PUBLISHER']).most_common(5):
    print("   {} posted {} articles".format(emphasize(key), emphasize(value)))


# ### Conclusion
# 
# Doesn't make too much sense to plot the counts here since they are too many. It is important to note that many famous papers can be found in our dataset. Even more importantly, some of them could obviously be used to improve the category prediction in the next section (Bloomberg would be much more likely to post something related to `Business` than `Entertainment`, the opposite holds true for `Contactmusic.com`. However we will follow the instructions given and only utilize the headline free text for our predictions.

# # Let's Predict! <a name="predict"></a>
# 
# The first step is of course preprocessing - we need to create numerical features out of our raw text. We can use sklearn for that. Its classes often offer some default preprocessing and we could rely on them to do the job. However that would be:
# 
# 1. Not fun - obviously.
# 2. Not optimal since we can perform some smarter preprocessing and adapt it to our specific case. 

# In[ ]:


from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenize(s, lemmatize=True, decode=False):
    """ Split a sentence into it's words by removing case sensitivity and stopwords, as well as taking punctuation 
    and other special characters into account.
    
    Parameters
    ----------
    s : str
        The raw string
    lemmatize : bool, optional
        Optionally lemmatize the provided text
    decode : bool, optional
        Whether or not Unicode input that needs to be decoded is expected.
    
    Returns
    -------
    list of str
        The cleaned tokens
        
    """
    # Make sure the NLTK data are downloaded.
    try:
        if decode:
            s = s.decode("utf-8")
        tokens = word_tokenize(s.lower())
    except LookupError:
        nltk.download('punkt')
        tokenize(s)
    

    # Exclude punctuation only after NLTK tokenizer to ensure part of word punctuation is not removed.
    # For example only the second "." should be removed in the below string
    # "Mr. X correctly diagnozed his patient."
    ignored = stopwords.words("english") + [punct for punct in string.punctuation]
    clean_tokens = [token for token in tokens if token not in ignored]
    
    # Optionally lemmatize the output to reduce the number of unique words and address overfitting.
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in clean_tokens]
    return clean_tokens


def test_tokenize():
    """Unit test the tokenizer. """
    
    # With lemmatization
    text = "Mr. X correctly diagnosed his patients."
    expected_result = ['mr.', 'x', 'correctly', 'diagnosed', 'patient']
    assert tokenize(text) == expected_result
    
    # Without lemmatization
    expected_result = ['mr.', 'x', 'correctly', 'diagnosed', 'patients']
    assert tokenize(text, lemmatize=False) == expected_result
    
test_tokenize()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Bag of Words Representation using our own tokenizer.
vectorizer = CountVectorizer(lowercase=False, tokenizer=tokenize)
x = vectorizer.fit_transform(news['TITLE'])

# Create numerical labels.
encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])

# Let's keep this in order to interpret our results later,
encoder_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))

# Split into a training and test set. Classifiers will be trained on the former and the final
# results will be reported on the latter.
seed = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)


# ### Let's try a classifier <a name="classification"></a>
# 
# At this point it is up to us to pick a classifier. A rather convenient balance between model complexity and fitting potential is a RandomForest. In a real world setting we might also try `XGBoost`, `CatBoost` to maximize accuracy, or `Decision Trees` in case we care more about interpretability than accuracy.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

def report_accuracy(trained_clf):
    train_score = trained_clf.score(x_train, y_train)
    test_score = trained_clf.score(x_test, y_test)
    print("Training set accuracy score is: {}".format(emphasize(train_score)))
    print("Test set accuracy score is: {}".format(emphasize(test_score)))
    
# Let's use all our cores to speed things up.
n_cores = max(multiprocessing.cpu_count(), 1)

rf = RandomForestClassifier(n_jobs=n_cores)
rf.fit(x_train, y_train)
report_accuracy(rf)


# ## Overfitting <a name="overfitting"></a>
# 
# 91% test set accuracy is a pretty impressive result given that we didn't even tune the model. However we can see a high discrepancy between the training and test error. This is normally the result of overfitting: our model is partly managing to memorize the train test rather than generilize to unseen data. 
# 
# ** Addressing Overfitting **
# 
# *We can tackle this issue with different ways depending on the model (usually by applying L1 or L2 reguralization, or dropout in case of NNs). In the random forest case, we usually address this by increasing the number of trees. Alternative methods apply some form of pruning, by prematurely prohibiting the trees to grow (split). We could easily do this by controlling the `max_depth` or `min_samples_leaf` parameters. However I am personally not a fan of this approach. Instead I believe that by increasing the number of trees, we actually WANT individual trees to overfit on a particular subset of the training set - we then expect the ensembling (vote) to cancel this effect out. The only consideration is then the expected runtime since the algorithm is obviously **O(N)** w.r.t. the number of estimators. But I need to go grocery shopping anyway so that's not a huge issue.*

# ## Other metrics <a name="other-metrics"></a>
# 
# Accuracy feels like a rather "good" metric in our case, after all there is no reason why certain misclassifications would carry higher weight than others. However depending on our specific use case we might want to explore different metrics. For example if all we care about are articles in the `Health` category, then we only need to optimize the model's ability w.r.t. this specific label. Let's take a look into the confusion matrix.
# 

# In[ ]:


from sklearn.metrics import confusion_matrix
import numpy as np

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, normalize=False):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap. Based on
    shaypal5's gist: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
        
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.set_ylabel('True label')
    heatmap.set_xlabel('Predicted label')
    heatmap.set_title(title)
    return fig

conf_mat = confusion_matrix(y_test, rf.predict(x_test))

# Get some readable labels
labels = [cat_map[encoder_mapping[label]] for label in sorted(encoder_mapping.keys())]
ax = print_confusion_matrix(conf_mat, labels, normalize=True)
plt.show()


# ### Confusion Matrix Interpretation
# 
# As we can see from this heatmap - some categories are harder than others to predict. On one hand we see a **97%** score in the `Entertainment` category - my guess is that not even a human could easily beat that score. On the other hand we see a much worse score for `Health` and `Science`. Perhaps we can connect those results with the metrics we observed during the [EDA](#eda) phase: `Entertainment` is by far the richest category with around **100,000 samples**, while `Health` is by far the poorest with less than **40,000**. As a result we might assume that the category itself is not in fact harder to predict - the problem is we have much fewer samples!
# 
# ## Conclusions
# 
# Using some basic preprocessing and minimal tuning, we achieved some impressive metrics. This probably has to do with the high quality dataset we received - titles are clean and offer great predictive power into an article's category. 
# 
# ### Improving Accuracy
# If we cared about even higher metrics, we could consider a deep learning approach, for example using Keras. In fact those articles are probably hand labeled by a human which already leaks a minor error (I don't think that everyone would agree on those categories. For example what if an article discussed a new policy voted in the US congress and how this policy affects cancer research. Is that `Health` or `Politics`?) So our accuracy has an upper bound of human performance which could be around 95%. Does it make sense to use deep learning hoping to achieve a 3% improvement? 
# 
# Anyway, if we were to try deep learning, then we would be in good luck because Google has released pre-trained embeddings on a [google-news dataset](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) - which means that those embeddings would probably serve as a great first hidden layer in any kind of recurrent architecture we might consider. However the current results are reasonable, so improving upon them on those is out of score of this exercise. 

# # News trends - What is happening in the world? <a name="new-trends"></a>
# 
# We will now address a much more open ended question: What can we infer about the world's state during the timeline covered by our data? We will start with some simple time series analysis to find out whether particular patterns occur for specific categories. We will then look into certain categories with more detail and try to identify specific events using carious NLP tools.

# In[ ]:


news['WEEK'] = news['TIMESTAMP'].apply(lambda date: date.week)

# Aggregate by week
aggregated = news[['WEEK', 'MONTH', 'CATEGORY']]
grouped = aggregated.groupby(['WEEK', 'CATEGORY']).size().reset_index(name='article_count')
grouped['CATEGORY'] = grouped['CATEGORY'].apply(lambda x: cat_map[x])

plt.figure(figsize=(10, 7))
ax = sns.lineplot(x='WEEK', y='article_count', hue='CATEGORY', data=grouped, ci=None)
ax.set_xticks(list(range(11, 36)))
ax.set_xlabel("Week of the year")
ax.set_ylabel("Number of articles")
ax.set_title("Articles published per week")
plt.show()


# ## Impressive Correlation
# 
# We can see that all those categories are pretty correlated - they peak together and sink together. This relationship only breaks during the last weeks which happens to be summer so maybe it is connected to the holiday schedule of different newspapers. This correlation is a rather interesting phenomenon with at least several possible interpretations:
# 
# 1. Important events in very different sectors happen together. While I could see this happening with politics (for example pass a controversial law when the public is preoccupied with something unrelated) I can't see it in the case of our categories. Which brings us to the second and more probably possible cause:
# 
# 2. Publishers follow certain seasonality patterns: They publish more during specific days and less during other days regardless of the importance of available news and follow this strategy for every different category.
# 
# 3. Nothing of the above is relevant and this is only a bug or "feature" in the data collection process used to assemble this dataset. Since I do have any insight into this process I will silently ignore this possibility, however it is definitely worth exploring in a real world scenario.

# ## Topic Modeling <a name="topic-modeling"></a>
# 
# This particular text based dataset comes with category labels, which allowed us to perform supervised machine learning above. However this is an exception rather than the norm, as most raw text datasets of this size are not labeled. The reason is simple: manually labeling those headlines is a menial task requires human effort which would often make such a venture prohitively expensive. Manually assigning labels also results in loss of information in at least two axes:
# 
# 1. How would one decide on the number of categories? The authors of the original dataset opted for 4 categories, and I would assume this to be an educated guess. However different, and potentially equaly reasonable results could be obtained by assuming 3 or 5 categories. Perhaps there should have been a `Politics` category, or the `Entertainment` category could be split into `Celebrity & Lifestyle` and `Art`.
# 
# 2. Limiting each article to a single category or topic makes a huge assumption **which we know to be generally False**: That each article is relevant to a single category. What about articles touching a little bit of multiple worlds. For example articles explaining a new governmental policy on vaccination and the results of this policy in the health of the population. Is that `Politics` or `Heath`? Another possibility, very prevalent in the dataset at hand, is a mix of `Business` and `Science & Technology` since multiple articles of both categories revolve around products releases of technology tech giants like Samsung, Apple and Microsoft. Assigning a single category to those articles invariable introduces information loss.
# 
# Using topic modeling, we instead assume that any given article was generated from a mix of topics, or categories, with different proportions. In the our example, we would expect the headlines to be a mix of the `Business` and `Technology` categories. If the headline focuses more on the specifications of the new product then the weight of `Technology` will increase at the expense of `Business`. If the headline mostly ignores the specification and discussed the impact of this product on the company's stock, then the opposite is true.
# 
# Let's realize those concepts with the most popular open source tool in topic modeling: [Gensim](https://radimrehurek.com/gensim/)
# 
# **Disclaimer**: I have contributed to Gensim's source code in the past.
# 

# In[ ]:


from gensim import corpora
import numpy
import random

# Let's reuse the tokenizer we wrote before to clean the text.
clean_text = news['TITLE'].apply(tokenize)

# Reproducible topics.
numpy.random.seed(seed)
random.seed(seed)

# Create Dictionary and a Corpus (basic Gensim structures)
id2word = corpora.Dictionary(clean_text)
id2word.filter_extremes(no_below=5, no_above=0.05)
print(id2word)
corpus = [id2word.doc2bow(text) for text in clean_text]


# ## Applying a topic model
# 
# Gensim offers a plethora of topic models that we can choose from. A mathematically fancy one is Latent Dirichlet Allocation - [LDA](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) authored by Andrew Ng himself.  
# 
# 
# ### Picking the number of topics
# 
# A common issue for every topic model, is the choice of the number of latent topics as this cannot be inferred from the raw data alone. For now, let's assume that the original creators of the dataset were right in arbitrarily choosing the number 4.

# In[ ]:


import gensim
import re

num_topics = 4
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=seed)

# Let's inspect at the discovered topics.
def print_model(model):
    """Print a readable representation of a topic model. 
    
    Parameters
    ----------
    model : gensim.models.ldamodel.LdaModel
        A trained model.  
    """
    def print_topic(topic):
        topic_no, topic_repr = topic
        parts = topic_repr.split("+")
        words = [re.search('"(.*)"', part).group(1) for part in parts]
        return "{}: {}".format(topic_no, words)
    
    for topic in model.print_topics():
        print(print_topic(topic))
        
    
print_model(lda_model)


# Our model at first glance seems to agree with the existing categories. The first topic obviously corresponds to `Health`,  and the last one to `Entertainment`. The second and third topics seem to be a mix of the remaining two categories. This supports our initial observation that many articles in these categories are indeed hard to distinquish. Perhaps we could obtain more coprehensible results by limiting our search to three topics

# In[ ]:


num_topics = 3
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=seed)
print_model(lda_model)


# The discovered topics are now much more coherent:
# 
# 1. The first one corresponds to health
# 2. The second one to business and technology companies
# 3. The third to entertainment and lifestyle
# 
# ## Unseen documents <a name="unseen-documents"></a>
# 
# Another advantage of topic models, is that they are inherently limited to the corpus used for their training. Instead they can be used to infer the topic mix for new, previously unseen documents. This is a very powerful feature for dynamic applications where we expect to produce new data that needs to be categorized, be it in batches or in real-time using a streaming service. Let's see how our LDA model can handle this case.

# In[ ]:


topic_mapping = {
    0: "Health",
    1: "Business & Technology",
    2: "Entertainment"
}

unseen_headlines = [
    "Beyonce won a music award",
    "Samsung releases new product - stock rises",
    "New cow disease outbreaks with multiple symptoms endagers humans"
]

def rank_headline(headline):
    bow_vector = id2word.doc2bow(tokenize(headline))
    lda_vector = lda_model[bow_vector]
    top_topic = topic_mapping[max(lda_vector, key=lambda item: item[1])[0]]
    distribution = {topic_mapping[topic_no]: proportion for topic_no, proportion in lda_vector}
    return top_topic, distribution

for headline in unseen_headlines:
    top_topic, distribution = rank_headline(headline)
    print("{}: {} \n Topic Mix: {}\n".format(headline, emphasize(top_topic), distribution))


# As we can see the topic model correctly and confidently identifies the correct category for previously unseen documents. Observe that the sentences we used are clearly related to a single category -  this is reflected in the huge discrepancy between the assigned proportions assigned to each topic. As a result these examples would also be easy (but potentially time consuming) for a human to assign. What about more complex unseen documents that touch on multiple topics?

# In[ ]:


unseen_headlines = [
    "Pop star Justin Bieber to start clothes company - stock expected to skyrocket",
    "Startup develops new sustainable vaccination against Ebola developed."
]

for headline in unseen_headlines:
    top_topic, distribution = rank_headline(headline)
    print("{}: {} - {}".format(headline, emphasize(top_topic), distribution))


# Observe how the topic mixture is much more balanced than before. The first sentence is almost entirely in the middle of two categories, the last one also exhibits a mix. It should be obvious that accepting a mix of topics for examples such examples is much more reasonable than limiting ourselves to a single category.
# 
# ## Conclusions <a name="gensim-conclusions"></a>
# 
# In this section we showed how topic modeling can help us derive useful insights from a completely unlabeled dataset - in some cases matching or surpassing human labeling achieved through menial and potentially expensive effort. Our model was able to handle a massive amount of data in minimal training time, and infer the topic mix for previously unseen documents.
# 
# ### Future Work
# This is the result of fast and dirty exploration performed during a lazy Sunday afternoon. There are many ways to improve upon my results including:
# 
# * More sophisticated tokenization. Specifically our model includes many names, of either people or companies. As a result it would likely benefit from introducing bigrams so that "Justin Bieber" is a single token.
# * Instead of feeding the raw OHE input to our topic model, we could try using gensim's TFIDF transformer as a preprocessing step.
# * To boost performance, LdaMulticore can be used instead of the single core implementation I 
