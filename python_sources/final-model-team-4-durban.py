#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install comet_ml


# In[ ]:


# import comet_ml in the top of your file
# from comet_ml import Experiment

# # Initialise comet experiment
# experiment = Experiment(api_key="NWLadblAxMd1YoXYmNsNiVX36",
#                         project_name="nlp-machine-learning", workspace="pilasande")


# # TABLE OF CONTENTS
# * [Introduction](#intoduction)
# * [Import Packages](#import_packages)
# * [Import Data](#import_data)
# * [Exploratory Data Analysis](#exploratory_data)
# * [Data Preprocessing](#data_preprocessing)
# * [Model Selection](#model_selection)
# * [Insights](#insights)
# * [Conlusion](#conclusion)
# * [Kaggle Submission File](#kaggle_submission_file)
# * [References](#references)
# * [Pickled files](#pickled_files)
# 

# # Introduction <a class="anchor" id="introduction"></a>

# With the continued popularity of online social networking,companies have started focusing their marketing efforts oninteractive media such as Twitter and Facebook. These media channels, as well as others, are helping companies to better engage with their customers/consumers than traditional marketing methods can. In addition, the data science space has revolutionzed the process of market research through innovative methods. Given that there is so much corporate activity, media attention, and consumer involvement been directed toward sustaining the planet,improving the lives of people around the world, and protecting the ability of future generations to meet their own needs, the focus of this notebook is to accurately clasify belief in anthropogenic climate change, a hot topic, within a dataset of tweets.

# # Import Packages <a class="anchor" id="import_packages"></a>

# In[ ]:


# Python packages
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
# Wordcount visualizations
from wordcloud import WordCloud
# NLP
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from spacy import displacy
from bs4 import BeautifulSoup
# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
# Optimization
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score,accuracy_score
# Warnings
import warnings


# # Import Data <a class="anchor" id="import_data"></a>

# In[ ]:


train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')
test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')
sample = pd.read_csv('../input/climate-change-belief-analysis/sample_submission.csv')
data = pd.read_csv('../input/mbti-type/mbti_1.csv')
mbti = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 'J':'Judging', 'P': 'Perceiving'}


# In[ ]:


# view data
train.head()


# In[ ]:


# Data summary and checking for nulls
train.info()


# **Meaning of Clases**<br>
# <table align='left' style='width:50%'>
#     <tr>
#         <th style='width:10%' align="center">Class</th>
#         <th style='width:30%' align="center">Sentiment</th>
#         <th style='width:5%' align='center'>Stance</th>
#     </tr>
#     <tr>
#         <td>-1</td>
#         <td>The tweet does not believe in man made climate change</td>
#         <td>Agnostic</td>
#     </tr>
#     <tr>
#         <td>0</td>
#         <td>The tweet neither supports nor refutes the belief of man made climate change</td>
#         <td>Neutral</td>
#     </tr>
#     <tr>
#         <td>1</td>
#         <td>The tweet supports the belief of man made climate change</td>
#         <td>Believer</td>
#     </tr>
#     <tr>
#         <td>2</td>
#         <td>The tweet links to factual news on climate change</td>
#         <td>News</td>
#     </tr>
# </table>

# In[ ]:


# label the stance indicated by the class
labels_dict = {-1: 'Agnostic',0: 'Neutral',1: 'Believer',2: 'News'}
# Replace class values with tweet stance:
train.replace({'sentiment': labels_dict}, inplace=True)


# In[ ]:


# Document Corpus
raw_corpus = [statement.lower() for statement in train.message]


# # Exploratory Data Analysis <a class="anchor" id="exploratory data analysis"></a>

# The following section sets out to do analysis on the raw data and prime non essential elements for improved perfomance of the model(s).

# ## Class WordCount <a class="anchor" id="class_word_count"></a>
# The aim is to calculate the frequency of words in each class and find the most used words for each class.

# In[ ]:


def word_count(df,Corpus):
    """Output graph of most frequent words in each class
       given a dataframe with a class column and a corpus """
    fig, axs = plt.subplots(2,2, figsize=(16,8),)
    fig.subplots_adjust(hspace = 0.5, wspace=.2)
    axs = axs.ravel()
    for index, stance in enumerate(df.sentiment.unique()):
        corpus = np.array(Corpus)[df[df.sentiment == stance].index.values]
        corpus = ' '.join(corpus).split(' ')
        word_counts = {}
        for word in corpus:
            if word in word_counts.keys():
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        word_val_pair = []
        for word,word_freq in word_counts.items():
            word_val_pair.append((word,word_freq))
        word_val_pair.sort(key = lambda x: x[1],reverse=True)
        words = []
        frequency = []
        for word_val in word_val_pair[:10]:
            words.append(word_val[0])
            frequency.append(word_val[1])
        axs[index].set_title(f'{stance}',fontsize=15)
        axs[index].bar(x=words,height=frequency,edgecolor='k')
    


# In[ ]:


word_count(train,raw_corpus)


# The plots between classes show the 10 most used words in each class, it can be seen that climate is the most used word in the data sets in 3 of the four clases, the plots also consist of common english words which offer little insights, futher cleaning must be done to remove these words.

# In[ ]:


# Word Cloud
def word_cloud(input_df,Corpus):
    """Function output the wordcloud of a class given
       a dataframe with a sentiment column and a corpus"""
    df = input_df.copy()
    fig, axs = plt.subplots(2,2, figsize=(16,8))
    fig.subplots_adjust(hspace = 0.5, wspace=.2)
    axs = axs.ravel()
    for index, stance in enumerate(df.sentiment.unique()):
        corpus = np.array(Corpus)[df[df.sentiment == stance].index.values]
        corpus = ' '.join(corpus)
        word_cloud = WordCloud(background_color='white', max_font_size=80).generate(corpus)
        axs[index].set_title(f'{stance}',fontsize=15)
        axs[index].imshow(word_cloud,interpolation='bilinear')
        axs[index].axis('off')


# In[ ]:


word_cloud(train,raw_corpus)


# ## Hashtag analysis <a class="anchor" id="hashtag_analysis"></a>
# In this section we look at the hashtags association between the classes, a word count is perfomed and the data visalized in the form of a wordcloud.

# In[ ]:


def hashtags(input_df,Corpus):
    """Function output the wordcloud of a class given
       a dataframe with a sentiment column and a corpus"""
    df = input_df.copy()
    fig, axs = plt.subplots(2,2, figsize=(16,8))
    fig.subplots_adjust(hspace = 0.5, wspace=.2)
    axs = axs.ravel()
    for index, stance in enumerate(df.sentiment.unique()):
        corpus = list(np.array(Corpus)[df[df.sentiment == stance].index.values])
        for line in range(len(corpus)):
            corpus[line] = ' '.join([word for word in corpus[line].split() if word.startswith('#')])
        corpus = ' '.join([word for word in corpus if word])
        corpus = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", corpus)
        word_cloud = WordCloud(background_color='black', max_font_size=80).generate(corpus)
        axs[index].set_title(f'{stance}',fontsize=15)
        axs[index].imshow(word_cloud,interpolation='bilinear')
        axs[index].axis('off')


# In[ ]:


hashtags(train,raw_corpus)


# ## Green Speak Terms <a class="anchor" id="green_speak"></a>

# In[ ]:


green_terms = ['biofuels','photovoltaic',
               'cap-and-trade','pollution',
               'carbon dioxide','renewable energy',
               'carbon footprint','solar',
               'carbon offsets','wind energy',
               'carbon tax','carcinogen',
               'clean energy','clean tech', 
               'climate bill','climate change',
               'corporate social responsibility',
               'cradle to cradle','ecolabel',
               'energy','fossil fuels',
               'green economy','green roof',
               'green-collar','greenhouse',
               'cycle assessment','wind power','green',
               'carbon','dioxide']
def green_speak(input_df,Corpus):
    """Function output the wordcloud of a class given
       a dataframe with a sentiment column and a corpus"""
    df = input_df.copy()
    fig, axs = plt.subplots(2,2, figsize=(16,8))
    fig.subplots_adjust(hspace = 0.5, wspace=.2)
    axs = axs.ravel()
    for index, stance in enumerate(df.sentiment.unique()):
        corpus = np.array(Corpus)[df[df.sentiment == stance].index.values]
        corpus = ' '.join(corpus)
        corpus = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", corpus)
        word_dict = {}
        for term in green_terms:
            if term in corpus:
                word_dict[term] = corpus.count(term)
        word_cloud = WordCloud(background_color='black', max_font_size=80).generate_from_frequencies(word_dict)
        axs[index].set_title(f'{stance}',fontsize=15)
        axs[index].imshow(word_cloud,interpolation='bilinear')
        axs[index].axis('off')


# In[ ]:


green_speak(train,raw_corpus)


# ## Class Imbalance <a class="anchor" id="class imbalance"></a>
# This section aims to look at class imbalance of the given dataset

# In[ ]:


imbalance = sns.barplot(x = train.sentiment.value_counts(), y= train.sentiment.value_counts().index)
imbalance.set(title = 'Class distribution in data',xlabel='sentiment counts')
plt.show()


# The data shows imbalanced data with 59% of the sample being in the Believer class and only 8% being in the Agnostics class, this may lead to a biased model. This hypothesis was tested by oversampling the minority classes to half the number of the Believers class and further undersampling the believers class to half its size in the train set, the results yeilded a decrease in model perfomance.

# ## Tweet length distribution <a class="anchor" id="tweet_lenght"></a>

# In[ ]:


def boxplot(input_df):
    df = input_df.copy()
    df.tweet_length = df.message.apply(lambda x: len(x))
    plot = sns.boxplot(x=df.tweet_length,y=df.sentiment)
    plot.set(xlabel='tweet_lenght')
    return(plot)


# In[ ]:


boxplot(train)
plt.show()


# The plot indicates that news and neutral tweeps tend to write shorter messages as compared to the agnostics and believers, neutral tweets have a higher variation (with some being very short and others very long), it is also quite possibly harder to extract sentiment from shorter tweets as they do not give enough context.

# # Data Preprocessing <a class="anchor" id="data_preprocessing"></a>

# This section aims to clean the raw data into the most important text articles, after cleaning the raw data a visualization similar to the one on the EDA step is shown. The data is cleaned using the <code>cleaning_fun</code> function which applies the use of regular expressions, list comprehensions and nltk packages (tokenization and stop words) to remove text which is deemed non-significant for sentiment analysis.

# In[ ]:


# Set stopwords
added_stop_words = ['rt','dm']
stop_words = set(stopwords.words("english")+added_stop_words)
removed_stop_words = ['not','do']
for i in removed_stop_words:
    stop_words.remove(i)


# In[ ]:


# Define Cleanig function
def cleaning_fun(tweet):
    """This function takes a tweet and extracts important text"""
    tweet = tweet.lower()
    tweet = re.sub(r'https?://\S+|www\.\S+','',tweet) # Remove URLs
    tweet = re.sub(r'<.*?>','',tweet) # Remove html tags
    tweet = re.sub(r'abc|cnn|fox|sabc','news',tweet) # Replace tags with news
    tweet = re.sub(r'climatechange','climate change',tweet)
#   Tokenize tweet
    tokenizer = TreebankWordTokenizer()
    tweet = tokenizer.tokenize(tweet)
    tweet = [word for word in tweet if word.isalnum()] #Remove punctuations
#   Remove numbers
    tweet = [word for word in tweet if not any(c.isdigit() for c in word)]
#   Replace News if news is in the words
    tweet = ['news' if 'news' in word else word for word in tweet]
#   Replace word with trump if trump is in the word
    tweet = ['trump' if 'trump' in word else word for word in tweet]
#   Remove stop words
    tweet = ' '.join([word for word in tweet if word not in stop_words])
    return(tweet)


# In[ ]:


# Add clean tweets column to train data
train['clean_tweets'] = train.message.apply(lambda x: cleaning_fun(x))
train.head(3)


# In[ ]:


# cleaned corpus
clean_corpus = [cleaning_fun(tweet) for tweet in raw_corpus]


# In[ ]:


word_count(train,clean_corpus)


# In[ ]:


word_cloud(train,clean_corpus)


# In[ ]:


def NER(corpus):
    nlp = spacy.load('en_core_web_sm')
    seperator=','
    y=[]
    doc=nlp(seperator.join(clean_corpus[:90]))
    for entity in doc.ents:
        y.append(entity.text)
    word_cloud = WordCloud(background_color='white', max_font_size=80).generate(seperator.join(y))
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(word_cloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()
NER(clean_corpus)


# In[ ]:


def MBTI(input_df,train):
    df = input_df.copy()
    for i in range(len(df)):
        df.posts[i] = BeautifulSoup(df.posts[i], "lxml").text
        df.posts[i] = re.sub(r'\|\|\|', r' ', df.posts[i])
        df.posts[i] = re.sub(r'http\S+', r'<URL>', df.posts[i])
    np.random.seed(1)
    tfidf2 = CountVectorizer(ngram_range=(1, 1), stop_words='english',lowercase = True, max_features = 5000)
    model_lr = Pipeline([('tfidf1', tfidf2), ('lr', LogisticRegression(class_weight="balanced", C=0.005,max_iter=300))])
    warnings.filterwarnings("ignore")
    model_lr.fit(df.posts, df.type)
    separator = ', '
    a=separator.join(train.query("sentiment=='News'")['clean_tweets'][:1000].values.tolist())
    b=separator.join(train.query("sentiment=='Believer'")['clean_tweets'][:1000].values.tolist())
    c=separator.join(train.query("sentiment=='Neutral'")['clean_tweets'][:1000].values.tolist())
    d=separator.join(train.query("sentiment=='Agnostic'")['clean_tweets'][:1000].values.tolist())
    k=[a,b,c,d]
    pred_all = model_lr.predict(k)
    return (pred_all)
MBTI(data,train)


# From initial preprocessing the following lines were added to the cleaning function:<br>
# <code>tweet = re.sub(r'climatechange','climate change',tweet)</code><br>
# <code>tweet = re.sub(r'abc|cnn|fox|sabc','news',tweet)</code><br>
# <code>tweet = ['news' if 'news' in word else word for word in tweet]</code><br>
# <code>tweet = ['trump' if 'trump' in word else word for word in tweet]</code><br>

# ## Vectorize Corpus and Split Data <a class="anchor" id="vectorize_corpus"></a>

# In[ ]:


#instantiate and vectorize corpus
count_vectorizer = CountVectorizer(ngram_range=(1,2))
count_vectorizer.fit(clean_corpus)


# In[ ]:


#Included for pickling to web app (not used in actual model) 
tfvectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True)
tfvectorizer.fit(clean_corpus)


# In[ ]:


# Define feature and target variables
X = train.clean_tweets
y = train.sentiment


# In[ ]:


# Tran test split data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[ ]:


# Vectorize test and train set
X_train = count_vectorizer.transform(X_train)
X_test = count_vectorizer.transform(X_test)


# # Model Selection <a class="anchor" id="model_selection"></a>

# The following models were tested:
# - Logistic Regression Model <br>
# Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. The idea in logistic regression is to cast the problem in the form of a generalized linear regression model.
# Multiclass classification with logistic regression can be done either through the one-vs-rest scheme in which for each class a binary classification problem of data belonging or not to that class is done, or changing the loss function to cross-entropy loss.
# - Linear Support Vector Classifier<br>
# Support vector machine algorithm in which each data item is plotted as a point in n-dimensional space (where n is number of features) with the value of each feature being the value of a particular coordinate. Then, classification is perfomed by finding the hyper-plane that differentiates the two classes. Linear SVC then uses linear support vectors as the between classes.
# - Random Forest<br>
# Randomforest is an ensemle method using decision trees, bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.

# In[ ]:


from sklearn.linear_model import SGDClassifier
# Define Classification Models to be tested with default parameters
models = {'LogisticReg': LogisticRegression(multi_class='ovr',
                                            class_weight='balanced',
                                            max_iter=1000),
          'LinearSVC': LinearSVC(),
          'RandomForest': RandomForestClassifier(n_estimators=5)}


# In[ ]:


perfomance_df = pd.DataFrame()
for name in models.keys():
    scores = cross_val_score(models[name], X_train, y_train, cv=5, scoring='f1_weighted')
    mean_score = round(scores.mean(),2)
    mean_stddev = round(scores.std(),3)
    temp = pd.DataFrame({'weighted_f1_avg':mean_score,'deviation':mean_stddev}, index=[name])
    perfomance_df = pd.concat([perfomance_df, temp])
print(perfomance_df.sort_values('weighted_f1_avg', ascending=False))


# In[ ]:


# Validation of Models
val_df = pd.DataFrame()
for name in models.keys():
    models[name].fit(X_train,y_train)
    y_pred = models[name].predict(X_test)
    eval_score = f1_score(y_test,y_pred,average='weighted')
    eval_score = round(eval_score,2)
    temp = pd.DataFrame({'weighted_f1_avg':eval_score}, index=[name])
    val_df = pd.concat([val_df, temp])
print(val_df.sort_values('weighted_f1_avg', ascending=False))


# ## Model Training and Validation

# Based on perfomance on cross validation the logistic Regression model was chosen

# In[ ]:


chosen_model = 'LogisticReg'


# In[ ]:


y_pred = models[chosen_model].predict(X_test)


# In[ ]:


def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,5)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%' % (p)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % (p)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
# Code source : https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7


# In[ ]:


cm_analysis(y_test,y_pred,models[chosen_model].classes_)


# In[ ]:


classification_report(y_test, y_pred)


# In[ ]:


# Comet log parameters
# experiment.log_metric('weighted_f1',f1_score(y_test,y_pred,average='weighted'))
# experiment.log_confusion_matrix(y_test, y_pred)


# It can be seen from the confission matrix that the model perfoms relatively well for the classes of Believers and news tweets, Our model appears to be biased towards the believers class with the class containing the highest numbers of false positives/false negatives (i.e 29.5% of agnostic,32.9% of Neutral and 15.3% of news tweets were all classified as Believers)

# ## Parameter Tuning

# A grid search over five folds was used on the model to tune the C parameter for an optimized model.

# In[ ]:


parameters = {'C':[10,5,1],
              'multi_class': ['ovr','multinomial']}
grid_search = GridSearchCV(models[chosen_model], parameters,scoring='f1_weighted')
grid_search.fit(X_train, y_train)
search_params = grid_search.best_params_


# Once best parameters are found from gridsearch, the model is then initialised with the best parameters and a cross validation is perfomed once again.

# In[ ]:


best_logistic_model = LogisticRegression(multi_class=search_params['multi_class'],
                                         class_weight='balanced',
                                         max_iter=1000,
                                         C = search_params['C'])
score = cross_val_score(best_logistic_model, X_train, y_train, cv=5, scoring='f1_weighted')
mean_score = round(score.mean(),2)
print(mean_score)


# **Model Metrics on test data**

# In[ ]:


best_logistic_model.fit(X_train,y_train)
y_pred = best_logistic_model.predict(X_test)


# In[ ]:


# Confussion Matrix
cm_analysis(y_test,y_pred,best_logistic_model.classes_)


# In[ ]:


#classifiaction report
print(classification_report(y_test, y_pred))


# In[ ]:


# Comet log parameters
experiment.log_metric('weighted_f1',f1_score(y_test,y_pred,average='weighted'))
experiment.log_confusion_matrix(y_test, y_pred)


# In[ ]:


#End Comet experiment
experiment.end()


# # Insights and Observations <a class="anchor" id="6th-bullet"></a>

# ## Effect of lemmitazation
# The following packages were used for lemmitazation:<br>
# <code>from nltk.corpus import wordnet</code><br>
# <code>from nltk import pos_tag</code><br>
# <code>from nltk.stem import WordNetLemmatizer</code><br>
# #Define tagging function to be used with lemmatization<br>
# <code>def get_wordnet_pos(word):
#     """Map POS tag to first character lemmatize() accepts"""
#     tag = pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ,
#                 "N": wordnet.NOUN,
#                 "V": wordnet.VERB,
#                 "R": wordnet.ADV}
#     return tag_dict.get(tag, wordnet.NOUN)</code><br>
# with the following in the cleaning_fun function:<br>
# <code>lemm = WordNetLemmatizer()
# tweet = [lemm.lemmatize(word, get_wordnet_pos(word)) for word in tweet]
# </code>
# Overall it was found that lemmatization offered no improvement in model perfomance and lead to poorer recall values in 'Neutral' and 'Agnostic' classes.<br>
# Code source[https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#textbloblemmatizerwithappropriatepostag].

# # Conclusion

# The model could be especially useful to businesses looking into the Lifestyles Of Health and Sustainability (LOHAS) market.Exploratory data analysis revealed that tweets classified as Believers or Agnostics were strategic and logical individuals. Carefully constructed arguments as well as objective evidence is the way to go when marketing green products to these groups.

# # Submission File Preparation <a class="anchor" id="8th-bullet"></a>

# In[ ]:


# Clean test file
test.message = test.message.apply(lambda x: cleaning_fun(x))


# In[ ]:


test_X = count_vectorizer.transform(test.message)


# In[ ]:


test_pred = best_logistic_model.predict(test_X)


# In[ ]:


output = pd.DataFrame({'tweetid':test.tweetid,'sentiment':test_pred})


# In[ ]:


# Replace original labels
new_dict = {'Agnostic':-1,'Neutral':0,'Believer':1,'News':2}
output.replace({'sentiment':new_dict},inplace=True)


# In[ ]:


output.head()


# In[ ]:


output.to_csv('my_submission.csv', index=False)


# # References <a class="anchor" id="9th-bullet"></a>

# In[ ]:


1. https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#textbloblemmatizerwithappropriatepostag <br>
2. https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7<br>
3. https://www.kaggle.com/lbronchal/what-s-the-personality-of-kaggle-users


# # Pickle Files Preparation <a class="anchor" id="10th-bullet"></a>

# In[ ]:


# Pickle Vectorizers
import pickle
Vectorizers = {'countvec':count_vectorizer,'tfidf':tfvectorizer}
for filename,item in Vectorizers.items():
    outfile = open(f'{filename}.pkl','wb')
    pickle.dump(item,outfile)
    outfile.close()


# In[ ]:


#Pickle Models
for filename,item in models.items():
    outfile = open(f'{filename}.pkl','wb')
    pickle.dump(item,outfile)
    outfile.close()


# In[ ]:




