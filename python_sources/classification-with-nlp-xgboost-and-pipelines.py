#!/usr/bin/env python
# coding: utf-8

# # Grape variety classification based on wine description
# *We build a classifier for predicting the grape variety based on the expert description of wines. We go through all the preprocessing steps, show the elements of the classifier model and finally present its performance. Having 35 different grape type, the model manages to predict their variety with 60% accuracy. *
# 
# ## Introduction
# A while ago we started to work on a wine project that tried to imitate blind tasting of wines. We have built a classification model that, based on a wine description given by an expert or semi-expert, is able to tell what grape was used to produce that wine. Unfortunately it is not easy to find such descriptions, so we decided to collect data ourselves. We managed to tell the grape type of four different grapes with 85% accuracy. All this is described in 3 studies at: [our webpage](https://diveki.github.io/projects/wine/index.html) and the code can be found on [our GitHub repository](https://github.com/diveki/WineSommelier). Only later on we realised that there is this huge collection of wine descriptions on [Kaggle](https://www.kaggle.com/zynicide/wine-reviews). 
# 
# Here we will work with 35 different grapes (or collection of grapes) that has more than 200 samples in the data base. Relying on our data cleaning, preprocessing, text vectorization and an XGBoost classifier we manage to achieve 65% accuracy in telling what grape was used to produce the wine under inspection. 
# 
# We will start by importing the data then we **drop the features we are not interested in, remove duplicates and NAs, remove accents from text and make everything lower case**. We will process the target feature (grape variety) too. **One grape type may appear under different names**. One reason for that is that in different country they have different names. We try to bring them under the same name. Furthermore, we **introduce colours of the wine as a new input feature**, since during blind tasting, the expert can actually look at the colour of the wine. 
# 
# For the classification model we define **stop words, a class for lemma tokenization, a tf-idf vectorizer and an XGBoost classifier**. We use **pipelines** to put all these things together.  
# 
# Finally we show the implementation of the model, present **hyperparameter tuning and cross validation testing**. At the end we will show a little discussion about the effect of having **inbalanced number of target features** and options to deal with it. 
# 
# ## Table of contents
# 1. [Importing data](#ch1)
# 2. [Data preprocessing](#ch2)
# 
#  2.1 [Selecting features](#ch2.1) 
# 
#  2.2 [Removing duplicates, NAs and text formatting](#ch2.2) 
# 
#  2.3 [Target feature processing](#ch2.3) 
# 
#  2.4 [Introduction of new features](#ch2.4) 
# 
# 3. [Building the classification model](#ch3)
# 
#  3.1 [Stop words](#ch3.1)
#  
#  3.2 [POS tagging and Lemma tokenization](#ch3.2)
#  
#  3.3 [Text vectorizer](#ch3.3)
#  
#  3.4 [XGBoost classifier](#ch3.4)
#  
#  3.5 [Pipeline with helper functions](#ch3.5)
# 
# 4. [Hyperparameter tuning](#ch4)
# 
# 5. [Notes on imbalanced target features](#ch5)
# 
# 6. [Conclusion](#ch6)
# 
# <a id="ch1"></a>
# ## 1. Importing data

# First thing to do, we import useful packages and load the wine database. We will rely on many python libraries maninly pandas and numpy for processing dataframes, seaborn and matplotlib for plotting, sklearn, imblearn and xgboost for machine learning techniques and nltk for natural language processing tools. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import warnings  
warnings.filterwarnings('ignore')

# importing packages
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# sklearn packages
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

# nltk packages
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from string import punctuation
import unidecode


# Let us look at the content of the wine database.

# In[ ]:


data = pd.read_csv('../input/winemag-data_first150k.csv')
data.head(5)


# Each line corresponds to a wine and each column is an infomration about that wine. Since our aim is to classify grape variety, our target feature will be the *variety* column. We will use the description column, which is a characterisation by an expert, to determine to which target feature it belongs to.
# Another interesting study could be to identify wineries, country or province based on the description but we leave that for another time. The concepts laid down here could be applied there too.

# <a id="ch2"></a>
# ## 2. Data preprocessing
# We do not performa detailed feature analysis of the data, since we have a clear objective, using descriptions and colours to determine grape types. But if someone is interested in such kinf of study visit [this page](https://www.kaggle.com/carkar/classifying-wine-type-by-review).
# 
# <a id="ch2.1"></a>
# ### 2.1 Selecting features
# First we drop some of the columns we are certainly not interested in. The ones not droped will be helpful to clean the data.

# In[ ]:


data_sel = data.drop(['Unnamed: 0','designation','points','region_2',], axis = 1)


# In[ ]:


data_sel.shape


# Now we have 7 columns and 150 930 rows. Let us see if all the samples are relevent to us.

# <a id="ch2.2"></a>
# ### 2.2 Removing duplicates, NAs and text formatting
# We do not want any duplication in the description column since that might falsely over emphasize certain features. Therefore, we start by dropping any description duplicates.

# In[ ]:


data_single = data_sel.drop_duplicates('description')


# In[ ]:


data_single.shape


# Interestingly there were quite a few duplicates. There were either duplicates of some wines or the different vintages of the same wine had the same description. Now we have 97 821 unique samples. 
# Most of the times, these kind of data bases are not fully filled, therefore we might have some NAs in features we are interested in: description, variety (and maybe price). In the case of price we could estimate a value based on other similar inputs, but we are not able to come up with new descriptions or varieties, therefore we just simply drop any row that contains NAs in these columns.

# In[ ]:


data_single = data_single.dropna(subset=['description', 'variety', 'price'])


# In[ ]:


data_single.head()


# In[ ]:


data_single.describe(include='all')


# We have left country, province, region_1 and winery in as features, because they will serve as stop words to be removed from the description column, to try to decrease the chance of having any hints in the description about grape type. The price could be used as a feature to predict grape types, but we will not use it here. You could change our code and try it if you wish.
# We have printed a summary of the data frame too. We can see that the number of samples in description, variety and price match, which is a good sign, meaning they should be unique inputs. It seems there are NAs in country, province and region_1. There are 619 unique grape varieties, pinot noir being the most frequent, having 8802 inputs. The price range is quite wide. Would you pay $2300 for a bottle of wine? 
# The next thing we do is to transform all text into lower case letters and remove any accents from letters.

# In[ ]:


for col in ['variety', 'description', 'province', 'region_1', 'winery', 'country']:
    data_single[col] = data_single[col].str.lower()


# In[ ]:


def unidecode_text(text):
    try:
        #pdb.set_trace()
        text = unidecode.unidecode(text)
    except:
        pass
    return text


# In[ ]:


for col in ['description', 'variety', 'province', 'winery']:
    data_single[col] = data_single.apply(lambda row: unidecode_text(row[col]), axis=1)


# <a id="ch2.3"></a>
# ### 2.3 Target feature processing
# By looking at the feature target, varieties, we can find out that there are 619 varieties, each one having different number of inputs, many of them having only one. Certainly, the latter ones will not be of use, because we will not be able to train and test on one sample. Later on we will set a limit on how many samples a variety has to have. 

# In[ ]:


data_single.variety.value_counts()


# We can see many blend wines. A blended wine refers to a mixture of different grapes (not necessarily having a concensus on what grapes are included). We find it too ambitious to have a classifier that could correctly identify these wines, therefore we simply remove those samples that are labelled blends. Also, there are roses too, without any reference to the type of the grape it was produced. That is why we will remove roses too. 
# Below you find a list of labels we will remove from the variety column. 

# In[ ]:


filtered_name = ['red blend', 'portuguese red', 'white blend', 'sparkling blend', 'champagne blend', 
                 'portuguese white', 'rose', 'bordeaux-style red blend', 'rhone-style red blend',
                 'bordeaux-style white blend', 'alsace white blend', 'austrian red blend',
                 'austrian white blend', 'cabernet blend', 'malbec blend', 'portuguese rose',
                 'portuguese sparkling', 'provence red blend', 'provence white blend',
                 'rhone-style white blend', 'tempranillo blend', 'grenache blend',
                 'meritage' # beaurdaux blend
                ]


# In[ ]:


data_filtered = data_single.copy()
data_filtered = data_filtered[~data_filtered['variety'].isin(filtered_name)]


# Our next task is to bring the variety names on a common ground. Different countries have different names for the same grape, so we should categorize them in the same way. For example *trebbiano* and *ugni blanc* is the same grape. Therefore, we make a mapping between certain names and their common name.

# In[ ]:


def correct_grape_names(row):
    regexp = [r'shiraz', r'ugni blanc', r'cinsaut', r'carinyena', r'^ribolla$', r'palomino', r'turbiana', r'verdelho', r'viura', r'pinot bianco|weissburgunder', r'garganega|grecanico', r'moscatel', r'moscato', r'melon de bourgogne', r'trajadura|trincadeira', r'cannonau|garnacha', r'grauburgunder|pinot grigio', r'pinot noir|pinot nero', r'colorino', r'mataro|monastrell', r'mourv(\w+)']
    grapename = ['syrah', 'trebbiano', 'cinsault', 'carignan', 'ribolla gialla', 'palomino','verdicchio', 'verdejo','macabeo', 'pinot blanc', 'garganega', 'muscatel', 'muscat', 'muscadet', 'treixadura', 'grenache', 'pinot gris', 'pinot noir', 'lambrusco', 'mourvedre', 'mourvedre']
    f = row
    for exsearch, gname in zip(regexp, grapename):
        f = re.sub(exsearch, gname, f)
    return f

name_pairs = [('spatburgunder', 'pinot noir'), ('garnacha', 'grenache'), ('pinot nero', 'pinot noir'),
              ('alvarinho', 'albarino'), ('assyrtico', 'assyrtiko'), ('black muscat', 'muscat hamburg'),
              ('kekfrankos', 'blaufrankisch'), ('garnacha blanca', 'grenache blanc'),
              ('garnacha tintorera', 'alicante bouschet'), ('sangiovese grosso', 'sangiovese')
             ]


# In[ ]:


data_corrected = data_filtered.copy()
data_corrected['variety'] = data_corrected['variety'].apply(lambda row: correct_grape_names(row))
for start, end in name_pairs:
    data_corrected['variety'] = data_corrected['variety'].replace(start, end) 


# In[ ]:


len(data_corrected.variety.value_counts())


# Visibly, we have now only 557 different grape types (and still there might be several references to the same grape under different names).
# Now we filter out any grape variety that has less than 200 samples in the data base. 

# In[ ]:


data_reduced = data_corrected.groupby('variety').filter(lambda x: len(x) > 200)
data_reduced.shape


# Now we have only 62 087 inputs and 35 grape labels. As you can see the number of samples for each grape is very imbalanced which actually has a great implication on the performance of the classifier. We will discuss this in [section 6](#ch6).

# In[ ]:


grapes = list(np.unique(data_reduced.variety.value_counts().index.tolist()))


# In[ ]:


len(grapes)


# In[ ]:


data_reduced.variety.value_counts()


# <a id="ch2.4"></a>
# ### 2.4 Introduction of new features
# During blind tasting wine experts are allowed to look and expect the wine, but the wine label is not exposed to them. Therefore, we introduce the colour feature of the selected grapes.

# In[ ]:


colour_map = {'aglianico': 'red', 'albarino': 'white', 'barbera': 'red', 'cabernet franc': 'red',
              'cabernet sauvignon': 'red', 'carmenere': 'red', 'chardonnay': 'white', 'chenin blanc': 'white',
              'corvina, rondinella, molinara': 'red', 'gamay': 'red', 'garganega': 'white', 
              'gewurztraminer': 'white', 'glera': 'white', 'grenache': 'red', 'gruner veltliner': 'white',
              'malbec': 'red', 'merlot': 'red', 'mourvedre': 'red', 'muscat': 'white', 'nebbiolo': 'red',
              "nero d'avola": 'red', 'petite sirah': 'red', 'pinot blanc': 'white', 'pinot gris': 'white',
              'pinot noir': 'red', 'port': 'red', 'prosecco': 'white', 'riesling': 'white', 'sangiovese': 'red',
              'sauvignon blanc': 'white', 'syrah': 'red', 'tempranillo': 'red', 'torrontes': 'white', 
              'verdejo': 'white', 'viognier': 'white', 'zinfandel': 'white'
             }


# We create two more columns with names *red* and *white*, and their values can be 1 if the wine has that colour or 0 if it does not. In *sklearn* this corresponds to *one hot incoding*, we transform categorical data into vectors.

# In[ ]:


kaggle_input = data_reduced.copy()
kaggle_input['colour'] = kaggle_input.apply(lambda row: colour_map[row['variety']], axis=1)
colour_dummies = pd.get_dummies(kaggle_input['colour'])
kaggle_input = kaggle_input.merge(colour_dummies, left_index=True, right_index=True)


# In[ ]:


kaggle_input.reset_index(inplace=True)
kaggle_input.head()


# <a id="ch3"></a>
# ## 3. Building the classification model
# Now that we are ready with the data preprocessing wwe start to build up the model for classification. We start with steps to build a text vectorizer, define a XGBoost classifier and put everything into a pipeline which deals with selecting text based and numerical based inputs separately.

# <a id="ch3.1"></a>
# ### 3.1 Stop words
# These are words that we do not want to come accross in the text and do not want to analyse them. That is why we remove any occurance of the country, province and winery from the text. 

# In[ ]:


# stop words for countries
stop_country = list(np.unique(kaggle_input.country.dropna().str.lower().tolist()))

#stop words for province
stop_province = list(np.unique(kaggle_input.province.dropna().str.lower().tolist()))

#stop words for winery
stop_winery = list(np.unique(kaggle_input.winery.dropna().str.lower().tolist()))


# We load in the stop words of *nltk* too. That has a good collection of filling words that are usually not descriptive for a text. We additionally extend that with our self defined list of punctuations and a few words. 

# In[ ]:


# defining stopwords: using the one that comes with nltk + appending it with words seen from the above evaluation
stop_words = stopwords.words('english')
stop_append = ['.', ',', '`', '"', "'", '!', ';', 'wine', 'fruit', '%', 'flavour', 'aromas', 'palate']
stop_words1 = stop_words + stop_append + grapes + stop_country + stop_province + stop_winery


# <a id="ch3.2"></a>
# ### 3.2 POS tagging and Lemma tokenization
# In the description text a word might appear in different forms while actually representing the same word, like *good* and *best*. To reduce noise we try to find the basic form (*lemma*) of each word in the text. We will rely on the *nltk* package to achieve this, but first we have to type tag each word, wether they are nouns, verbs, adjectives or adverbs. For that we built up some helper functions.

# In[ ]:


# list of word types (nouns and adjectives) to leave in the text
defTags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']#, 'RB', 'RBS', 'RBR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# functions to determine the type of a word
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

# transform tag forms
def penn_to_wn(tag):
    if is_adjective(tag):
        return nltk.stem.wordnet.wordnet.ADJ
    elif is_noun(tag):
        return nltk.stem.wordnet.wordnet.NOUN
    elif is_adverb(tag):
        return nltk.stem.wordnet.wordnet.ADV
    elif is_verb(tag):
        return nltk.stem.wordnet.wordnet.VERB
    return nltk.stem.wordnet.wordnet.NOUN
    
# lemmatizer + tokenizer (+ stemming) class
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        # we define (but not use) a stemming method, uncomment the last line in __call__ to get stemming tooo
        self.stemmer = nltk.stem.SnowballStemmer('english') 
    def __call__(self, doc):
        # pattern for numbers | words of length=2 | punctuations | words of length=1
        pattern = re.compile(r'[0-9]+|\b[\w]{2,2}\b|[%.,_`!"&?\')({~@;:#}+-]+|\b[\w]{1,1}\b')
        # tokenize document
        doc_tok = word_tokenize(doc)
        #filter out patterns from words
        doc_tok = [x for x in doc_tok if x not in stop_words1]
        doc_tok = [pattern.sub('', x) for x in doc_tok]
        # get rid of anything with length=1
        doc_tok = [x for x in doc_tok if len(x) > 1]
        # position tagging
        doc_tagged = nltk.pos_tag(doc_tok)
        # selecting nouns and adjectives
        doc_tagged = [(t[0], t[1]) for t in doc_tagged if t[1] in defTags]
        # preparing lemmatization
        doc = [(t[0], penn_to_wn(t[1])) for t in doc_tagged]
        # lemmatization
        doc = [self.wnl.lemmatize(t[0], t[1]) for t in doc]
        # uncomment if you want stemming as well
        #doc = [self.stemmer.stem(x) for x in doc]
        return doc


# We created a class for lemmatization, *LemmaTokenizer*. Besides finding the lemmas it does some text filtering and at the end only selects nouns and adjectives, since these are the most descriptive words in the life of a wine. 

# <a id="ch3.3"></a>
# ### 3.3 Text vectorizer
# To transform the text based description column into a number based object, we use term frequency-inverse document frequency (tf-idf) vectorization. First it counts the occurance of a word in one sample and than down-weighs it with the occurance of the same word over all the samples. It repeats this process for each word and outputs a vector of numbers, where each number represents a word. The actual vectorizer performes some firther normalization too.
# We have decided to analyse the text word by word (1-grams), but you could choose 2-grams or n-grams, meaning selecting the first n words to represent the first element of the vector. Then each consequent element would be constructed by shifting the n-range by one. 
# The vectorizer accepts a tokenizer as an input, that is why we have defined one previously.
# The vectorizer gives a lot of opportunities for different setups and tests. If you want to see a smaller experiment with it, visit [our wine study](https://diveki.github.io/projects/wine/tfidf.html).

# In[ ]:


vec_tdidf = TfidfVectorizer(ngram_range=(1,1), analyzer='word', #stop_words=stop_words1, 
                                               norm='l2', tokenizer=LemmaTokenizer())


# <a id="ch3.4"></a>
# ### 3.4 XGBoost classifier
# XGBoost is a version of gradient boosted decision tree classifier. In boosting, the trees are built sequentially such that each subsequent tree aims to reduce the errors of the previous tree. These subsequent trees are called base or weak learners. Each of these weak learners contributes some vital information for prediction, enabling the boosting technique to produce a strong learner by effectively combining these weak learners. The power of XGBoost lies in its scalability, which drives fast learning through parallel and distributed computing and offers efficient memory usage.  
# 
# We will not do a very detailed hyperparameter optimization becuase it is very time consuming, but rather predifine certain input arguments that control the performance of the classifier and use one argument to otpimize.  *subsample* controls the ratio of the randomly selected training samples before growing the tree. It ranges between 0 and 1. Higher values tend to cause overfitting. *colsample_bytree* denotes the fraction of columns to be randomly sampled for each tree. *n_estimators* controls the number of trees to be constructed during the classification process.  We will optimize that parameter.

# In[ ]:


clf = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7)


# <a id="ch3.5"></a>
# ### 3.5 Pipeline with helper functions
# Since we have combined input features, text and numeric based features, we cannot treat them in the same way. We create helper classes that will return either the text column (*TextSelector*) or the number based column (*NumberSelector*). They will be the part of the pipeline. You initialize them with the column name of the input data. They need to have a *fit* and *transform* method. In our case the *fit* returns the class itself, while the *transform* method returns the column of the input data with the given column name.  

# In[ ]:


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None, *parg, **kwarg):
        return self

    def transform(self, X):
        # returns the input as a string
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # returns the input as a dataframe
        return X[[self.key]]


# We also define a function that prints out the accuracy, the classification report (including precision, recall and f1 scores) and a confusion matrix expressed in percentages of the number of each variety.

# In[ ]:


def print_stats(preds, target, labels, sep='-', sep_len=40, fig_size=(10,8)):
    print('Accuracy = %.3f' % metrics.accuracy_score(target, preds))
    print(sep*sep_len)
    print('Classification report:')
    print(metrics.classification_report(target, preds))
    print(sep*sep_len)
    print('Confusion matrix')
    cm=metrics.confusion_matrix(target, preds)
    cm = cm / np.sum(cm, axis=1)[:,None]
    sns.set(rc={'figure.figsize':fig_size})
    sns.heatmap(cm, 
        xticklabels=labels,
        yticklabels=labels,
           annot=True, cmap = 'YlGnBu')
    plt.pause(0.05)


# Now it is time to build up the pipeline. We start by creating a pipeline object for each input feature. 
# 
# Let us start with a text pipeline, that selects the text column from the input and applies the tf-idf vectorizer on it to transform it into a matrix of scores. 

# In[ ]:


text = Pipeline([
                ('selector', TextSelector(key='description')),
                ('vectorizer', vec_tdidf)
                ])


# Next we create pipeline objects to select the colour columns of the input data.

# In[ ]:


#pipelines of colour features
red = Pipeline([
                ('selector', NumberSelector(key='red')),
                ])
white = Pipeline([
                ('selector', NumberSelector(key='white')),
                ])


# Here comes the tricky part. To combine all feature, we use the *FeatureUnion* object. That makes sure there will not be any errors from combining text and number based inputs.

# In[ ]:


feats = FeatureUnion([('description', text),
                      ('red', red),
                      ('white', white)
                      ])


# Finally, we are ready to combine the input features with the classifier into one single pipeline .  

# In[ ]:


pipe = Pipeline([('feats', feats),
                 ('clf',clf)
                 ])


# <a id="ch4"></a>
# ## 4. Hyperparameter tuning
# As we have mentioned earlier we will the do a hyperparameter tuning on the *n_estimators*. In normal cases you would do it on more parameters, depending on how big is your data. Normally one would start with a random search of parameters to target the location of each paremeter where it should be more investigated and as a second step you would do a more detailed grid search. 
# 
# One can check what parameters can be finetuned for the classifier with the following expression:

# In[ ]:


pipe.named_steps['clf'].get_params()


# To save some space we do just the latter, with *GridSearchCV*. It works on different folds and splits each fold into a test and train range. We will see how to retrieve information from those sets. Because of this, we first split our input data into a train and test set, so that the test will actually serve as a validation set. A totally unseen data during the process. It is a good habit to have a validation set, to check the deterioration of the performance of the model. If it is big, it means you are rather overfitting your model.
# 
# Now let us create the train and test sets:

# In[ ]:


# split the data into train and test
combined_features = ['description', 'white', 'red']
target = 'variety'

X_train, X_test, y_train, y_test = train_test_split(kaggle_input[combined_features], kaggle_input[target], 
                                                    test_size=0.33, random_state=42, stratify=kaggle_input[target])


# To do the hyperparameter optimization, we have to define the parameter grid too. In our case this parameter is the classifiers *n_estimators* parameter. If someone is interested, there are two commented lines, they are other parameters that can be optimized. By simply uncommenting them, you enlarge the grid of parameter search. Keep in mind that doing so will extremely extend the duration of the grid search.

# In[ ]:


# definition of parameter grid to scan through
param_grid = {
     'clf__n_estimators': [50,100,300]
#    'clf__colsample_bytree': [0.6,0.8,1]
#    'clf__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
}


# Finally we define the grid search object. Our *pipe* variable will serve as the estimator and it needs the parameter grid too. We define 3 cross validation folds (in principle the more the better, but it takes more time too), and to exploit parallel computing we use 3 cores with the *n_jobs* argument. Setting the *verbose* argument to 0 prevents the algorithm to print our messages to the console. After the grid search object construction we can use the *fit* method to train our model:

# In[ ]:


# grid search cross validation instantiation
grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid, 
                          cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)


# In[ ]:


#hyperparameter fitting
grid_search.fit(X_train, y_train)


# After finishing with the grid search we can look at many features of the results. One interesting point to look at is the accuracy of the model on the cross validations train and test sets. Usually the train accuracy is larger than the test accuracy. If the degradation is large, it means the model is overfitting, picking up target characteristic features in the train set. To look at these, we call the *mean_train_score* and the *mean_test_score*.

# In[ ]:


grid_search.cv_results_['mean_train_score']


# In[ ]:


grid_search.cv_results_['mean_test_score']


# In each case we have 3 mean scores, each corresponding to the relevant *n_estimators* value. We can also see that by increasing the *n_estimators* the accuracy increases and the difference between train and set accuracies increase. One could check how far more can be the accuracy increased by increasing this hyperparameter, but for now we stick with the best result:

# In[ ]:


grid_search.best_params_


# Now we create a new classifier that has the best parameters from this grid search and use it on the validation set to really make sure that we are not over or underfitting. 

# In[ ]:


clf_test = grid_search.best_estimator_


# In[ ]:


# test stats
preds = clf_test.predict(X_test)


# In[ ]:


print_stats(y_test, preds, clf_test.classes_)


# We still get an accuracy very close to the mean test accuracy on the validation set. 63% with a very simple classifier is not a bad deal. The preciosion for the target values that has not many samples are poor, but certainly there is hope for improvement. One could do a more thorough hyperparameter tuning or change to neural network classifiers. 

# <a id="ch5"></a>
# ## 5. Notes on imbalanced target features
# It is very easy to come accross a dataset with imbalanced targets (meaning class 1 is far more represented than class 2). Some of the classifiers like Random Forest, by their design will classify a new item into a class that has more samples in the train set. This behaviour decreases the precision of the classifier.  
# 
# Depending on the exact issue there may be different solutions to this problem. It can happen that one is not even interested in a good precision at all. In some cases there can be lot of samples of one class and some in the other class, but in other cases the whole database can be very small. One can choose wether to decrease the number of samples of the large class (undersample) or try to increase the number of the small class (oversample). 
# 
# The *imblearn* package provides different methods to perform both cases. The *RandomUnderSampler* reduces the number of the samples in each class to the number of samples in the smallest class. It is done by randomly selecting samples. The *RandomOverSampler* does the some thing but increases the sample of the class where there are only a few samples. These are very simple approaches to balance the sample numbers in classes and they do not really introduce new features.
# 
# Obviously the best would be to find new samples, but it is not always easy. One could create new samples from existing ones by synthetically contructing them. Such technique is the Synthetic Minority Oversampling Technique (SMOTE). It first selects an element in the minority class (the class that needs new samples) and  searches for its n-nearest neighbours. Then these neighbours are connected with the element with a line and random dots are generated on those lines. These will be the synthetically constructed new elements of the class.
# 
# Let us use the SMOTE technique and the optimized parameters we learned previously and classify the wines again.

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline as imbPipeline


# The *imblearn* package is not compatible with the *sklearn* Pipelines, therefore we have to use their ones. We define a *SMOTE* object that is an input for the pipeline. It will oversample the prepared data.

# In[ ]:


clf = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7, 
                    n_estimators=300)
sm = SMOTE(random_state=42, sampling_strategy='auto')
pipe = imbPipeline([('feats', feats),
                    ('smote', sm),
                    ('clf',clf)
                 ])


# Now we are ready to fit the pipeline to the train set and then test it and print the statistics.

# In[ ]:


pipe.fit(X_train, y_train)


# In[ ]:


preds = pipe.predict(X_test)
print_stats(y_test, preds, pipe.classes_)


# From the result we can see that the accuracy has decreased a little bit, to 61%, while the precision has  improved. This is expected in a way. We have increased the samples for each class, therefore giving a chance for better precision, while the created new samples are synthetic, therefore probably not necessarily in line with real data, resulting in the decrease of the accuracy.

# <a id="ch6"></a>
# ## 6. Conclusion
# We have presented a simple model that is capable of classifying grape varieties based on wine descriptions and their colour. During the process we have cleaned the input data, created objects to turn text documents into numeric matrices while counting the frequency of the words in the text. Then we optimized a hyperparameter in the classifier to achieve better performance. At the end, we highlighted the importance of having balanced number of input classes. 
# 
# Although the achieved accuracy is only 62% for 35 different grapes, this can be improved. The first way of doing that is to getting more data (one of the classes had only 200 samples). Second, one could build up better classifiers either with more thorough hyperparameter tuning or maybe using neural networks. Finally, one can come up with a more sofisticated text vectorizer that relies on maybe information extraction and grammar rules. 
# 
# For a more detailed study on a different (smaller) database please visit [my website](https://diveki.github.io/projects/wine/index.html).
