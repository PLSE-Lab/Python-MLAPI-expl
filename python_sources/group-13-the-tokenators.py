#!/usr/bin/env python
# coding: utf-8

# # **Classification using NLP on MBTI Personality Types** 
# 
# ![image](https://cdn.shopify.com/s/files/1/0100/5392/articles/Mouthpiece_VdayMeyersBriggs1.progressive.jpg?v=1549578121)
# 
# As a species, we have developed high level of cultural and social constructs to keep us evolutionary marvels. To create such constructs, the human brain compartmentalises other humans into categories. This makes absolute sense in terms of survival as we can gauge threats, especially since humans are intellectually complex... and hierarchical.
# 
# One such psychological classification technique we developed as humans is the MBTI personality type predictor. Designed as a set of questions which essentially try scale you into 16 different personality types.
# 
# Being able to predict someone's personality type just from the words they have written has application in sales, marketting, recruitment as well as human resources. 
# 
# Problem statement: build a classifier model that can predict a persons personality type from their twitter posts but is sufficiently generalised that it will be able to do so from any other source of text.

# ## Table of Contents
# ***
# 1. [Explore the data](#eda)
#     * [Insights from the Training set](#training)
#     * [Establishing a Baseline](#baseline)
#     * [Investigating the imbalance](#imbalance)
#         * [Applying Oversampling](#oversampling)
#         * [Applying Undersampling](#undersampling)
# ***    
# 2. [Data Cleaning](#datacleaning)
#     * [Remove Personality Types](#nopers)
#     * [Remove Punctuation](#nopunc)
#     * [Stemmatising and Lemmatising](#stemlem)
#     * [Remove Stop Words](#nostopwords)
# ***
# 3. [Feature Engineering](#featureeng)
#     * [Counting Post Properties](#countpostprop)
#     * [Parts of Speech Analysis](#pos)
#     * [Sentiment Analysis](#sentiment)
#     * [Insights](#insights)
#     * [Dimensionality Reduction](#dimension)
# ***
# 4. [Applying Insights to Full Dataset](#applyinsight)
# ***
# 5. [Conclusion](#conclusion)
# ***
# 6. [Future Work](#futurework)
# 

# In[ ]:


# import libraries

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# For visualising our data
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from wordcloud import WordCloud, STOPWORDS
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

# to deal wrangle text data
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import FreqDist
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import string
import re

# models used
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.decomposition import TruncatedSVD


# # Explore the data
# <a id="eda"></a>

# In[ ]:


# import the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Function to customise table display
def multi_table(table_list):
    from IPython.core.display import HTML
    return HTML(
        '<table><tr style="background-color:white;">' +
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )


# View both tables simultaneously

# In[ ]:


multi_table([train.head(), test.head()])


# ### Explore some of the training data to get insights on each personality
# <a id='training'></a>

# In[ ]:


##Lets examine how many representatives there are for each personality type in the dataset
ax = sns.countplot(y=train.type, order = train['type'].value_counts().index)


# With majority of the posts coming from six personality types, this measn the data is unbalanced, we will see how this affects the acuracy of our prediction later on.

# Each of the four sub-categories in a personality type fall into two letters (explained in the next section), let us see the representation of each.

# In[ ]:


list1 = []
list2 = []
list3 = []
list4 = []

for cat in list(train['type']):

    
    list1.append(cat[0])
    list2.append(cat[1])
    list3.append(cat[2])
    list4.append(cat[3])

fig, ax =plt.subplots(1,4,figsize=(20,10))

sns.countplot(x=list1, ax=ax[0], palette='Set1')
sns.countplot(x=list2, ax=ax[1], palette='Set2')
sns.countplot(x=list3, ax=ax[2], palette='colorblind')
sns.countplot(x=list4, ax=ax[3], palette='Set3')
fig.show()


# Not only are the number of representatives for the 16 personalities imbalanced, the same trend is seen for the four sub-categories. 

# # Baseline
# <a id='baseline'></a>

# Now that we have looked at the dependent variables, let's establish a baseline model. This approach was adapted from Fast.AI  (lessons 1 to 4: http://course18.fast.ai/ml.html) and relies on using random forests to get not only a prediction, but extract insight on feature importance and how the decisions based on the data are made by the model. Random forests does not require the data to be clean and is not affected by distribution of dependent variable.

# ### Combine the train and test set to preprocess together

# In[ ]:


##The test dataset lacks the type column, an artificial one will be created 
test['type'] = 'test'
test_id = test['id']
test.drop("id", axis=1, inplace=True)


# In[ ]:


mbti = pd.concat((train, test)).reset_index(drop=True)
mbti.shape


# ### Step 1

# In[ ]:


##Remove the separator
##mbti['seperated_post'] = mbti['posts'].apply(lambda x: x.replace("|||",''))


# ### Step 2

# In[ ]:


##Obtain the index for the train and test subsets so this concatenated document can be separated later
test_ind = mbti[mbti['type'] == 'test'].index.tolist()
train_ind = mbti[mbti['type'] != 'test'].index.tolist()


# ### Step 3

# In[ ]:


##Vectorise the text
##tfd = TfidfVectorizer()
##X_tf = tfd.fit_transform(mbti['seperated_post'])


# ### Terminology:

# Rather than providing 16 labels for the classifiers to predict, each of these can be broken down into four sets of binary labels. This format is required for submissions to this Kaggle challenge. Besides submission requirements, this format will allow us to break down each aspect of the personality individually

# Mind: Introverted (0), Extroverted (1)

# Energy: Sensing (0) , Intuitive (1)

# Nature: Feeling (0), Thinking (1)

# Tactics: Perceiving (0), Judging (1)

# In[ ]:


## This function will allow for a rapid conversion of the 16 personality types into four binary columns.

def pers_conv(df):
    df['mind'] = df['type'].apply(lambda x: x[0] == 'E').astype('int')
    df['energy'] = df['type'].apply(lambda x: x[1] == 'N').astype('int')
    df['nature'] = df['type'].apply(lambda x: x[2] == 'T').astype('int')
    df['tactics'] = df['type'].apply(lambda x: x[3] == 'J').astype('int')
    return df


# ### Step 4

# In[ ]:


##Convert the single type column to four binary columns, using the pers_conv function defined above
##y_train = mbti.iloc[train_ind][['type']]
##y_train = pers_conv(y_train)
##y_train.drop("type", axis=1, inplace=True)


# ### Step 5

# In[ ]:


##Separate the train and test data from each other
##X_train = X_tf[train_ind,:]
##X_test = X_tf[test_ind,:]


# ### Step 6

# In[ ]:


##Initiate the random forest classifier
##rfr = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features= 0.5, random_state=0, n_jobs=-2)


# ### Step 7

# In[ ]:


##Trained the model
##rfr.fit(X_train, y_train)


# ### Step 8

# In[ ]:


##Obtained a multilabel prediction, renamed the  columns to match those used in the pers_conv function, added the ids 
##and saved to csv
##rfr_pred = rfr.predict(X_test)
##multiclass_rfr = pd.DataFrame(rfr_pred, columns=y_train.columns)
##multiclass_rfr_id = pd.concat([test_id, multiclass_rfr], axis=1)
##
##multiclass_rfr_id.to_csv('Rfr_first_prediction.csv', index=False)


# Random forest classifier also has a feature importance function which allows you to investigate which features 
# are most important in explaining the target variable.
# So when training a tree we can compute how much each feature contributes to decreasing the weighted impurity
# in classification, Gini impurity / information gain (entropy) is the metric used. The higher the score, the more informative
# the feature.

# In[ ]:


##We will keep this sorted dictionary full of feature importance for later after we experiment with approaches of obtaining 
##predictions for our model.
##feat_imp = sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd.get_feature_names()), reverse=True)


# Multilabel classification using the random forest classifier results in a score of 5.62375. But what if we did each prediciton individually? How would the score change?

# In[ ]:


##Rather than a multilabel prediction, each of the four letters of the personality are predicted individually and concatenated
##to a four column dataframe.
##types = y_train.columns.tolist()
##concat_pred = pd.DataFrame()
##for t in types:
##    rfr.fit(X_train, y_train[t])
##    temp_var = pd.DataFrame(rfr.predict(X_test), columns=[t])
##    concat_pred = pd.concat([concat_pred,temp_var], axis=1)


# In[ ]:


##The predictions were concatenated to the test_ids and saved to a csv so a submission to Kaggle could be made.
##comb_rfr_id = pd.concat([test_id, concat_pred], axis=1)
##comb_rfr_id.to_csv('Rfr_first_prediction_separated.csv', index=False)


# ![](http://)This results in a score of 5.65562. Many estimators are not capable of processing a multilabel input, so from now on, we will be using this method rather than multiclass prediction since the score is very similar.

# Since the number of estimators determines how many decision trees are generated and average, the higher the n_estimators, the better the score, we tried this out.

# In[ ]:


##Initiate the random forest classifier with 100 estimators
##rfr = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features= 0.5, random_state=0, n_jobs=-2)


# In[ ]:


##Rather than a multilabel prediction, each of the four letters of the personality are predicted individually and concatenated
##to a four column dataframe.
##types = y_train.columns.tolist()
##concat_pred = pd.DataFrame()
##for t in types:
##    rfr.fit(X_train, y_train[t])
##    temp_var = pd.DataFrame(rfr.predict(X_test), columns=[t])
##    concat_pred = pd.concat([concat_pred,temp_var], axis=1)
##    
####The predictions were concatenated to the test_ids and saved to a csv so a submission to Kaggle could be made.
##comb_rfr_id = pd.concat([test_id, concat_pred], axis=1)
##comb_rfr_id.to_csv('RFR_pred_100_est.csv', index=False)


# Our assumption ran true, this approach gave us a score 5.50437.

# # Investigating the data imbalance
# <a id='imbalance'></a>

# From investigating the type data, we find that there are imbalances in the number of representatives per personality group, there are several approaches to deal with this problem. Here, we compare our baseline model with two oversampling approaches and one undersampling approach.

# # Oversampling
# <a id='oversampling'></a>

# Random over-sampling increases the under-represented data by drawing from existing data points. Synthetic Minority Over-sampling Technique (SMOTE) creates new data points based on the information of those that already exist. It picks a point at random and computes the new data point using k-nearest neighbours. This is explained in greater detail in the following kernel:https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

# In[ ]:


##Import the oversamplers from the imbalanced learn toolkit
##from imblearn.over_sampling import RandomOverSampler
##from imblearn.over_sampling import SMOTE


# In[ ]:


##Initiate the over-samplers
##ros = RandomOverSampler(random_state=9000)
##oversampler = SMOTE(random_state=0)


# In[ ]:


##Obtain the dependent variable
##Y = train['type']


# In[ ]:


##Performed random over-sampling
##X_tfidf_resampled, Y_tfidf_resampled = ros.fit_sample(X_train, Y)


# In[ ]:


##Performed a SMOTE
##os_X, os_y = oversampler.fit_sample(X_train,Y)


# In[ ]:


##The transformed dependent variable is transformed to a dataframe and the number of representatives per personality type is
##investigated. The same was done for the SMOTE experiment
##y_train = pd.DataFrame(Y_tfidf_resampled, columns=['type']) 
##y_train['type'].value_counts()


# From this point, steps 4 to 8 from the above baseline protocol were performed, we just replaced the respective variables with the ones used in this section.

# It was found that both random oversampling (6.74692) and SMOTE (6.39643) were not able to provide a better prediction that the default dataset

# # Undersampling
# <a id='undersampling'></a>

# Under-sampling involves reducing the number of samples from each category to that of the group with the least representatives. 

# In[ ]:


##Get the number of representatives for the minority class (30) and each of the personality types.
##type_summary = pd.DataFrame(train['type'].value_counts())
##smallest = type_summary.loc[type_summary['type'].idxmin()]['type']
##pers_id = train['type'].unique().tolist()


# In[ ]:


##This function can be used to create a smaller dataset where the number of representatives per personality profile are reduced
##to that of the minorty class. The posts from each personality profile are shuffled randomly and a subset of 30 is selected.
##def reduce_size(df, state = 42):
##    b = pd.DataFrame()
##    for per in pers_id:
##        a = df[df['type'] == per]
##        a = shuffle(a, random_state=state)
##       a = a[:smallest]
##        b = pd.concat([b,a], axis=0)
##    return b


# In[ ]:


##Obtain a reduced datase
##small_set = reduce_size(train)


# In[ ]:


##Check that each personality profile has the same number of representatives
##small_set['type'].value_counts()


# In[ ]:


##Making a new smaller dataset
##mbti_s = pd.concat((small_set, test)).reset_index(drop=True)
##mbti_s.shape


# From this point on, we performed steps 1 to 8 from the above baseline protocol were performed, we just replaced the respective variables with the ones used in this section.

# We see that this approach also does not perform as well as the default dataset with a score of 11.43870. However, it was found that the random forest classifier ran very quickly as compared to the three datasets above, so from this point on, we will be using this smaller dataset to mine insight from the text data and troubleshoot. We will then apply these findings at the end to the full dataset. We will see how these findings will affect the score and the accuracy of our model.

# # Data clean-up
# <a id='datacleaning'></a>

# # Removing personality types
# <a id='nopers'></a>

# In[ ]:


##feat_imp


# Are you seeing what I am seeing? The best predictors for personality types are the personality types themselves, since these posts were obtained from a twitter page dedicated to discussions about personality, this makes sense. You will know what personality type a person is if they tell you directly. However, this does not help in predicting personality types from text that does not contain personality type information, so we need to remove these so other words that predict personality types can be identified.

# ### Splitting the posts

# Since the we are after insight, having only 30 representatives per personality type is a bit small, so we decided to split the posts using the '|||' separator in order to get a larger representative dataset 

# In[ ]:


##Small_set will be converted to a longer format which we will refer to as small set long
##small_set_long = []
##for i, row in small_set.iterrows():
##    for post in row['posts'].split('|||'):
##        small_set_long.append([row['type'], post])
##small_set_long = pd.DataFrame(small_set_long, columns=['type', 'posts'])
##print(small_set_long.shape)


# Before we start, let us set a new baseline for the reduced dataset to gauge the effect that the data cleaning is having on our ability to predict personality types. We will also be using three additional classifiers to identify whether another classifier produces better predictions than random forests and what factors may influence the classifiers ability to make a prediction.

# In[ ]:


##Since we are tokenising the posts, we will encounter a problem when we perform TfidfVectorisation as the vectoriser has a 
##built-in tokeniser and pre-processor. One of the preprocessing steps is making text lowercase, we will do this so long.
##small_set_long['posts'] = small_set_long['posts'].str.lower()


# In[ ]:


##This dataframe will contain all the scores of our troubleshooting efforts.
##Testlogger = pd.DataFrame()


# In[ ]:


##Lets set all these classifiers to default but allow them to run as quickly as possible with the exception of random forests
##as we want to identifiy the most informative features.
##NB = MultinomialNB()
##ada = AdaBoostClassifier()
##lorg = LogisticRegression(n_jobs=-1)
##rfr = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features= 0.5, random_state=0, n_jobs=-2)


# In[ ]:


def get_scores(df, logdf, name, x_col, y_col):
    
    ##TfidfVectorizer performs tokenisation and preprocessing automatically, we have already performed some preprocessing and
    ##tokenisation. So this function is a work around to use the vectorizer 
    def dummy(doc):
        return doc
    ##We want to use the get_feature_names command for this vectoriser when investigating feature importance using random 
    ##forests classifier, but in it's current form, tfd_2 is a local variable and won't be accessible outside of the function.
    ##This can be circumvented by declaring tfd_2 as a global variable.
    ##Later, we will also be extracting the word vector, so X_tf_2 will also need to be a global variable.
    global tfd_2
    global X_tf_2
    tfd_2 = TfidfVectorizer(lowercase=False, tokenizer=dummy, preprocessor=dummy)
    X_tf_2 = tfd_2.fit_transform(df[x_col])
    
    y = df[['type']]
    y = pers_conv(y)
    y.drop("type", axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tf_2,y[y_col], shuffle=True, random_state=42, test_size=0.2)
    
    ##Naive Bayes
    NB.fit(X_train,y_train)
    nb_pred = NB.predict(X_test)

    ##Adaboost
    ada.fit(X_train,y_train)
    ada_pred = ada.predict(X_test)

    ##Logisitic regression
    lorg.fit(X_train, y_train)
    lorg_pred = lorg.predict(X_test)
    
    ##Random forests
    rfr.fit(X_train, y_train)
    rfr_pred = rfr.predict(X_test)

    tempvar = pd.DataFrame([{'Description': name}])
    tempvar['NaiveBayesAcc'] = NB.score(X_test,y_test)
    tempvar['NBLogLoss'] = log_loss(y_test,nb_pred)
    tempvar['AdaboostAcc'] = ada.score(X_test,y_test)
    tempvar['AdaBLoss'] = log_loss(y_test,ada_pred)
    tempvar['LogisticRegAcc'] = lorg.score(X_test,y_test)
    tempvar['LogRegLogLoss'] = log_loss(y_test,lorg_pred)
    tempvar['RandomForestsAcc'] = rfr.score(X_test,y_test)
    tempvar['RandomForestsLogLoss'] = log_loss(y_test,rfr_pred)
    
    
    logdf = pd.concat([logdf, tempvar], axis=0).reset_index(drop=True)    
    
    return logdf


# Before we do this comparison, the posts need to be tokenised to make removing the personality types possible

# In[ ]:


##Tokenised the posts. We will use the TreeBankWordTokenizer since it is quicker than the word_tokenise function.
##tokeniser = TreebankWordTokenizer()
##small_set_long['tokens'] = small_set_long['posts'].apply(tokeniser.tokenize)


# From examining the feature importance list, we need a list a the personality types, as well as their plural. Since punctuation was not removed at this stage, will add the single and plurals with a period to the list as well.

# In[ ]:


##pers_id_lower = list(map(str.lower, pers_id))
##Added an s to the end of the personality type
##plur_pers = [c + "s" for c in pers_id_lower]
##Added a period to the end of the personality types
##dot_pers = [c + "." for c in pers_id_lower]
##Added a period to the end of the plural version of the personality types
##dot_plur_pers = [c + "." for c in plur_pers]
##pers_total = pers_id_lower + plur_pers + dot_pers + dot_plur_pers


# In[ ]:


##This function loops through the tokens and only keeps those that are not in the pers_total list
def remove_personality(tokens):    
    return [t for t in tokens if t not in pers_total]


# In[ ]:


##The personality types were removed from the dataset
##small_set_long['no_pers'] = small_set_long['tokens'].apply(remove_personality)


# In[ ]:


##Investigated the effect of removing personality types from the text data by running the logger.
##Testlogger = get_scores(small_set_long, Testlogger, 'tokenised posts','tokens','mind')
##Testlogger = get_scores(small_set_long, Testlogger, 'personality type removed','no_pers','mind')
##Testlogger


# It seems that the score gets worse when the personality types are removed, but this makes sense as they were the most informative features. We can once again check what features are informative.

# In[ ]:


##sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True)


# We can see that there is punctuation present in the data. We will remove these and see how our score is impacted.

# ### Remove punctation
# <a id='nopunc'></a>

# In[ ]:


def remove_punctuation(post):
    return ''.join([l for l in post if l not in string.punctuation])


# Need to generate new tokens and new columns to examine the effect of removing punctuation.

# In[ ]:


#small_set_long['post_rp'] = small_set_long['posts'].apply(remove_punctuation)
#small_set_long['tokens_rp'] = small_set_long['post_rp'].apply(tokeniser.tokenize)
##small_set_long['no_pers_rp'] = small_set_long['tokens_rp'].apply(remove_personality)


# In[ ]:


##Testlogger = get_scores(small_set_long, Testlogger, 'punctuation removed','tokens_rp','mind')
##Testlogger = get_scores(small_set_long, Testlogger, 'personality type and punct removed','no_pers_rp','mind')
##Testlogger


# Some results became better when punctuation was removed while others got worse. Additionally, punctuation was among the more informative features. We will remove them for now, but use this insight to do feature engineering such as counting punctuation.

# In[ ]:


##sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True)


# The main predicting words seem to be stopwords, this is strange since they are among the first features of text to be removed in NLP practises. However, it seems that stopwords may actually be good predictors of personality, and it seems that this article seems to agree : https://www.scientificamerican.com/article/you-are-what-you-say/?redirect=1. We will remove stopwords later to identify other words that may be of interest.

# ## Stemming vs lemmatizing
# <a id='stemlem'></a>

# Another way to clean data is to employ stemming or lemmatising. Words such as run, ran, running all come from the word run. Stemming removes the endings of words, this will work for running but not ran. While lemmatising finds the root of the word called lemma, so it can convert all the example words to run.

# Let's test which of these performs better.

# In[ ]:


##Define the stemmer
##stemmer = SnowballStemmer('english')


# In[ ]:


##This function loops through all the words in the list and applies the stemmer.
def mbti_stemmer(words, stemmer):
    return [stemmer.stem(word) for word in words]


# In[ ]:


##Apply the stemmer function to the data with punctuation removed.
##small_set_long['stem'] = small_set_long['no_pers_rp'].apply(mbti_stemmer, args=(stemmer, ))


# In[ ]:


##Define the lemmatizer. 
##lemmatizer = WordNetLemmatizer()


# In[ ]:


##This function loops through all the words in the list and applies the lemmatiser.
def mbti_lemma(words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in words] 


# In[ ]:


##Apply the lemmatiser function to the data with punctuation removed.
##small_set_long['lemma'] = small_set_long['no_pers_rp'].apply(mbti_lemma, args=(lemmatizer, ))


# In[ ]:


##Testlogger = get_scores(small_set_long, Testlogger, 'stemmatized','stem','mind')
##Testlogger = get_scores(small_set_long, Testlogger, 'lemmatized','lemma','mind')
##Testlogger


# Seems like despite the lemmatizer being better in theory, it seems that the model performed better when a stemmer was applied to the tokens.

# ## Removing stopwords
# <a id='nostopwords'></a>

# In[ ]:


##stop_words = stopwords.words('english')


# In[ ]:


def remove_stop_words(tokens):    
    return [t for t in tokens if t not in stopwords.words('english')]


# In[ ]:


#small_set_long['no_stop_stem'] = small_set_long['stem'].apply(remove_stop_words)
##small_set_long['no_stop_lemma'] = small_set_long['lemma'].apply(remove_stop_words)


# In[ ]:


##Testlogger = get_scores(small_set_long, Testlogger, 'stemmatized_stoprem','no_stop_stem','mind')
##Testlogger = get_scores(small_set_long, Testlogger, 'lemmatized_stoprem','no_stop_lemma','mind')
##Testlogger


# In[ ]:


##sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True)##


# It seems that there are still a few stopwords in the data, and it seems that there are a few spelling mistakes. 

# # Feature engineering
# <a id='featureeng'></a>

# ## Counting post properties
# <a id='countpostprop'></a>

# We found that removing punctuation affected the performance of the model in its ability to predict personality types both positively and negatively. This may be an indication that the words themselves may not be the only predictors of personality. Perhaps we need to look a little deeper into the posts themselves and see what insights we can find . To do this, we will create common text summaries on the reduced data set and see how they effect the models perfomance.

# In[ ]:


def post_stats(data):
    # create column with number of words 
    data['num_words'] = data['posts'].apply(lambda x:len(x.split()))
    
     # create column with number of different images   
    data['num_jpg_images'] = data['posts'].apply(lambda x:x.count('jpg'))
    data['num_png_images'] = data['posts'].apply(lambda x:x.count('png'))
    data['num_gif_images'] = data['posts'].apply(lambda x:x.count('gif'))
    
    # create a column with number of urls
    data['num_links'] = data['posts'].apply(lambda x:x.count('http'))

    # create a column with number of question and exclamation marks
    data['num_qmarks'] = data['posts'].apply(lambda x:x.count('?'))
    data['num_excls'] = data['posts'].apply(lambda x:x.count('!'))
    
    # create a column with number of periods and commas
    data['num_period'] = data['posts'].apply(lambda x:x.count('.'))
    data['num_comma'] = data['posts'].apply(lambda x:x.count(','))
    
    ##Difficult to count emojis, but the colon represent the eyes
    data['num_colon'] = data['posts'].apply(lambda x:x.count(':')) 
    
    # create a column with number of hashtags used 
    data['num_hashtags'] = data['posts'].apply(lambda x: x.count('#'))
    return data


# Create new columns with the summaries of the different elements making up the posts and print them out

# In[ ]:


##mbti_with_stats_long = post_stats(small_set_long)
##post_stat_cols = ['num_words',
                  'num_jpg_images', 
                  'num_png_images', 
                  'num_gif_images', 
                  'num_links', 
                  'num_qmarks', 
                  'num_excls', 
                  'num_period', 
                  'num_comma', 
                  'num_colon', 
                  'num_hashtags']
##mbti_with_stats_long[post_stat_cols].head()


# We saw that many of the columns had 0 in them quite often, so we wanted to investigate how these features would perform in predicting personality types when we compare the long vs wide format of the text data.

# In[ ]:


##Add these engineered features.
##mbti_with_stats_wide = post_stats(small_set)


# In[ ]:


##feat_eng_log = pd.DataFrame()


# In[ ]:


def get_scores_feat_eng(df, logdf, name, x_col, y_col):
    
    
    y = df[['type']]
    y = pers_conv(y)
    y.drop("type", axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df[x_col],y[y_col], shuffle=True, random_state=42, test_size=0.2)
    
    ##Naive Bayes
    NB.fit(X_train,y_train)
    nb_pred = NB.predict(X_test)

    ##Adaboost
    ada.fit(X_train,y_train)
    ada_pred = ada.predict(X_test)

    ##Logisitic regression
    lorg.fit(X_train, y_train)
    lorg_pred = lorg.predict(X_test)
    
    ##Random forests
    rfr.fit(X_train, y_train)
    rfr_pred = rfr.predict(X_test)

    tempvar = pd.DataFrame([{'Description': name}])
    tempvar['NaiveBayesAcc'] = NB.score(X_test,y_test)
    tempvar['NBLogLoss'] = log_loss(y_test,nb_pred)
    tempvar['AdaboostAcc'] = ada.score(X_test,y_test)
    tempvar['AdaBLoss'] = log_loss(y_test,ada_pred)
    tempvar['LogisticRegAcc'] = lorg.score(X_test,y_test)
    tempvar['LogRegLogLoss'] = log_loss(y_test,lorg_pred)
    tempvar['RandomForestsAcc'] = rfr.score(X_test,y_test)
    tempvar['RandomForestsLogLoss'] = log_loss(y_test,rfr_pred)
    
    
    logdf = pd.concat([logdf, tempvar], axis=0).reset_index(drop=True)    
    
    return logdf


# In[ ]:


##feat_eng_log = get_scores_feat_eng(mbti_with_stats_long, feat_eng_log, ##'Posts long format_counts', post_stat_cols, 'mind')
##feat_eng_log = get_scores_feat_eng(mbti_with_stats_wide, feat_eng_log, 'Posts wide format_counts', post_stat_cols, 'mind')
##feat_eng_log


# The scores are not particularily good, but they do show some predictive power. Interestingly, it seems that half of the algorithms perform better with the long format while the other half perform better with long format.

# Lets examine what features may be important in this prediction.

# In[ ]:


##sorted(zip(map(lambda x: x, rfr.feature_importances_), post_stat_cols), reverse=True)


# It seems as if punctuation does have an effect on predicting personality types, while weblink related counts do not. It is interesting that the number of words used is a predictor. Let us have a look into that.

# In[ ]:


# Display the lengths of the posts per personality type
##plt.figure(figsize=(15,10))
##small_set["text_size"] = small_set["posts"].apply(len)
##sns.swarmplot("type", "text_size", data=small_set)


# The differences may not be too pronounced, but it seems like the majority of the extroverted personality types tend to have longer posts.

# Besides just punctuation, sentence structure may have some predictive value. Let us investigate this further.

# # Part of speech analysis
# <a id='pos'></a>

# As with the example above, removing stopwords affected the performance of the the different algorithms differently. We also saw that stopwords were among the features that had the highest importance. Perhaps other aspects of sentence structure have the ability to predict personality types.

# In[ ]:


# Fraction of unique words per post
def unique_word_fraction(row):
    """function to calculate the fraction of unique words on total words of the posts"""
    post = row['posts']
    post_split = post.split(' ')
    post_split = [''.join(c for c in s if c not in string.punctuation) for s in post_split]
    post_split = [s for s in post_split if s]
    word_count = post_split.__len__()
    unique_count = list(set(post_split)).__len__()
    return (unique_count / word_count)

# stop words count
eng_stopwords = set(stopwords.words("english"))
def stopwords_count(row):
    """ Number of stopwords fraction in a posts"""
    text = row['posts'].lower()
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    stopwords_count = len([w for w in text_splited if w in eng_stopwords])
    return (stopwords_count/word_count)

# Fraction of punctuations per post
def punctuations_fraction(row):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given posts """
    text = row['posts']
    char_count = len(text)
    punctuation_count = len([c for c in text if c in string.punctuation])
    return (punctuation_count/char_count)

# Character counts per post
def char_count(row):
    """function to return number of chracters """
    return len(row['posts'])

# Fraction of Nouns per post
def fraction_noun(row):
    """function to give us fraction of noun over total words """
    text = row['posts']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    return (noun_count/word_count)

# Fraction of Adjectives per post
def fraction_adj(row):
    """function to give us fraction of adjectives over total words in given posts"""
    text = row['posts']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    return (adj_count/word_count)

# Fraction of Verbs per post
def fraction_verbs(row):
    """function to give us fraction of verbs over total words in given text"""
    text = row['posts']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return (verbs_count/word_count)


# In[ ]:


##Create new columns in the long format dataset by applying the above functions
#mbti_with_stats_long['unique_word_fraction'] = mbti_with_stats_long.apply(lambda row: unique_word_fraction(row), axis=1)
#mbti_with_stats_long['stopwords_count'] = mbti_with_stats_long.apply(lambda row: stopwords_count(row), axis=1)
#mbti_with_stats_long['punctuations_fraction'] = mbti_with_stats_long.apply(lambda row: punctuations_fraction(row), axis=1)
#mbti_with_stats_long['char_count'] = mbti_with_stats_long.apply(lambda row: char_count(row), axis=1)
#mbti_with_stats_long['fraction_noun'] = mbti_with_stats_long.apply(lambda row: fraction_noun(row), axis=1)
#mbti_with_stats_long['fraction_adj'] = mbti_with_stats_long.apply(lambda row: fraction_adj(row), axis=1)
#mbti_with_stats_long['fraction_verbs'] = mbti_with_stats_long.apply(lambda row: fraction_verbs(row), axis=1)


##Create new columns in the wide format dataset by applying the above functions
##mbti_with_stats_wide['unique_word_fraction'] = mbti_with_stats_wide.apply(lambda row: unique_word_fraction(row), axis=1)
##mbti_with_stats_wide['stopwords_count'] = mbti_with_stats_wide.apply(lambda row: stopwords_count(row), axis=1)
####mbti_with_stats_wide['punctuations_fraction'] = mbti_with_stats_wide.apply(lambda row: punctuations_fraction(row), axis=1)
##mbti_with_stats_wide['char_count'] = mbti_with_stats_wide.apply(lambda row: char_count(row), axis=1)
##mbti_with_stats_wide['fraction_noun'] = mbti_with_stats_wide.apply(lambda row: fraction_noun(row), axis=1)
##mbti_with_stats_wide['fraction_adj'] = mbti_with_stats_wide.apply(lambda row: fraction_adj(row), axis=1)
##mbti_with_stats_wide['fraction_verbs'] = mbti_with_stats_wide.apply(lambda row: fraction_verbs(row), axis=1)

##Create a list of all the new columns to allow them to be subsetted
##POS_cols = ['unique_word_fraction', 
##            'stopwords_count', 
##            'punctuations_fraction', 
##            'char_count',
##            'fraction_noun', 
##            'fraction_adj', 
##            'fraction_verbs']


# Since these functions mostly relate to fractions of parts of speech present, only the wide format was used to determine whether any of these features had any predictive power. The code for the long format data remains but it seems that certain posts had nothing in them since errors related to dividing by zero were returned. 

# In[ ]:


##feat_eng_log = get_scores_feat_eng(mbti_with_stats_wide, feat_eng_log, 'Posts wide format POS', POS_cols, 'mind')


# In[ ]:


##feat_eng_log


# In[ ]:


##sorted(zip(map(lambda x: x, rfr.feature_importances_), POS_cols), reverse=True)


# The scores are pretty bad, perhaps in combination with other features or more samples, these features may have more predictive power.

# # Sentiment analysis
# <a id='sentiment'></a>

# We want to explore how the different personality types communicate. Using TextBlob's sentiment analyser we can do this with ease. 
# - First we will simply rank the personality types according to how positive or negative their posts average sentiments are.
# - Next we will use graphs to visualise the sentiments

# In[ ]:


def ranked_sentiments(data):   
    from IPython.display import display_html
    
    # retrieve the sentiments from all the posts in the dataset
    data[['polarity', 'subjectivity']] = data['posts'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

    # Ranking positive sentiment according to the polarity
    pos_senti_df = data.sort_values('polarity', ascending=False).drop_duplicates(['type'])
    pos_senti_df = pos_senti_df[['type', 'polarity']].groupby(by='type').mean()
    pos_senti_df = pos_senti_df.sort_values(by='polarity', ascending=False)
    pos_senti_df = pos_senti_df.reset_index()
    # Ranking negative sentiments according to the polarity
    neg_senti_df = data.sort_values('polarity', ascending=True).drop_duplicates(['type'])
    neg_senti_df = neg_senti_df[['type', 'polarity']].groupby(by='type').mean()
    neg_senti_df = neg_senti_df.sort_values(by='polarity')
    neg_senti_df = neg_senti_df.reset_index()
    
    # prepare the display side by side
    df1_styler = pos_senti_df.style.set_table_attributes("style='display:inline'").set_caption('Ranked Positive Sentiments')
    df2_styler = neg_senti_df.style.set_table_attributes("style='display:inline'").set_caption('Ranked Negative Sentiments')
    
    return display_html(df1_styler._repr_html_()+df2_styler._repr_html_(), raw=True)
    
##ranked_sentiments(small_set)


# From the tables above we it looks like ENFJ's exibit more positive sentiments in their posts as oppose to the other type. INTP's display express the most negative sentiments in their posts/comments. This gives a bit of insight into how we can expect when we communicate with these personality types

# In[ ]:


##small_set_long[['polarity', 'subjectivity']] = small_set_long['posts'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))


# In[ ]:


##plotting the average polarity per post for each personality type.
##plt.figure(figsize=(15,10))
##sns.barplot(data=small_set_long, x='type', y="polarity")
##x=plt.xticks(rotation=90)


# In[ ]:


##Plotting the average subjectivity per post
##plt.figure(figsize=(15,10))
##sns.barplot(data=small_set_long, x='type', y="subjectivity")
##x=plt.xticks(rotation=90)


# Polarity seems to differ slightly between personality types, while subjectivity remains relatively constant. Despite this, we will see whether sentiment can be used to predict personality types.

# In[ ]:





# # Insight
# <a id='insights'></a>

# ## Mind

# Eventhough the mind predictor has been the test in the data cleaning and EDA section, we still need to extract some insight on
# what words are associated with introverts and extroverts. For this section, we will run the model with and without stopwords included. We will extract features with a gini score above 0.005 for both experiments. We will extract them from the word vector and combine them with the engineered features to see how this affects predictions. Finally, we will produce a mind map of the most common words in introverts and extroverts as well as a countplot that shows the count of the informative words between introverts and extroverts.

# In[ ]:


##Testlogger = get_scores(small_set_long,  Testlogger, 'Mind insight stop included','lemma','mind')


# In[ ]:


##Extract the feature importance from random forest classifier, but display only the features with a gini >= 0.005
##feat_imp_mind_stop = pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True))
##feat_imp_mind_stop was too long to write out so shortened to fims
##fims_subset = feat_imp_mind_stop[feat_imp_mind_stop[0] >= 0.005]
##fims_subset


# In[ ]:


##Convert the word vector to a datatframe
##inside_vect_mind = pd.DataFrame(X_tf_2.A, columns=tfd_2.get_feature_names())


# In[ ]:


##Testlogger = get_scores(small_set_long, Testlogger, 'Mind insight no stop','no_stop_lemma','mind')


# In[ ]:


##Extract the feature importance from random forest classifier, but display only the feature with a gini >= 0.005
##feat_imp_mind_no_stop = pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True))
##fimns_subset = feat_imp_mind_no_stop[feat_imp_mind_no_stop[0] >= 0.005]
##fimns_subset


# In[ ]:


##df = pers_conv(small_set_long)
##fig, ax = plt.subplots(len(df['mind'].unique()), sharex=True, figsize=(15,10*len(df['mind'].unique())))

##k = 0
##for i in df['mind'].unique():
##    df_4 = df[df['mind'] == i]
##    wordcloud = WordCloud().generate(df_4['lemma'].to_string())
##    ax[k].imshow(wordcloud)
##    ax[k].set_title(i)
##    ax[k].axis("off")
##    k+=1


# In[ ]:


##mind_pred_words = fims_subset[1].tolist() + fimns_subset[1].tolist()
##mind_pred_words_unique = list(set(mind_pred_words))
##len(mind_pred_words_unique)


# 64 main predicting words for identifying between introverts and extroverts.

# In[ ]:


##vector_subset = inside_vect_mind[mind_pred_words_unique]
##vector_subset_type_mind = pd.concat([df[['mind']],vector_subset], axis=1)
##vector_subset_type_mind.groupby(['mind']).mean()


# In[ ]:


##rfr = RandomForestClassifier(n_estimators=40, max_depth=4, max_features=0.5, random_state=0, n_jobs=-2)


# In[ ]:


##X_train, X_test, y_train, y_test = train_test_split(X_tf_2,df['mind'], shuffle=True, random_state=42, test_size=0.2)


# In[ ]:


##rfr.fit(X_train, y_train)


# In[ ]:


##pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True)).head(10)


# Code for generating the decision tree obtained from https://www.kaggle.com/willkoehrsen/visualize-a-decision-tree-w-python-scikit-learn

# In[ ]:


#Draw the decision tree at estimator 0
##dot_data = StringIO()
##estimator = rfr.estimators_[0]
##export_graphviz(estimator, out_file='tree_limited.dot',feature_names=tfd_2.get_feature_names(),
##                class_names=np.asarray(['introvert', 'extrovert']),
##                filled=True, rounded=True,
##                special_characters=True)
##!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600 
##Image(filename='tree_limited.png')


# ## Energy

# In[ ]:


#Reset the random forests classifier
##rfr = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-2)


# In[ ]:


#Get the predictions for the energy subcategory
##Testlogger = get_scores(small_set_long, Testlogger, 'Energy insight stop included','lemma','energy')


# In[ ]:


##Extract the feature importance from random forest classifier, but display only the features with a gini >= 0.005
##feat_imp_energy_stop = pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True))
##feat_imp_energy_stop was too long to write out so shortened to fies
##fies_subset = feat_imp_energy_stop[feat_imp_energy_stop[0] >= 0.005]
##fies_subset


# In[ ]:


##Convert the word vector to a datatframe
##inside_vect_energy = pd.DataFrame(X_tf_2.A, columns=tfd_2.get_feature_names())


# In[ ]:


##run the model on the posts without stop words
##Testlogger = get_scores(small_set_long,  Testlogger, 'Energy insight no stop','no_stop_lemma','energy')


# In[ ]:


##Extract the feature importance from random forest classifier, but display only the feature with a gini >= 0.005
##feat_imp_energy_no_stop = pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True))
##fiens_subset = feat_imp_energy_no_stop[feat_imp_energy_no_stop[0] >= 0.005]
##fiens_subset


# In[ ]:


#df = pers_conv(small_set_long)
##fig, ax = plt.subplots(len(df['energy'].unique()), sharex=True, figsize=(15,10*len(df['energy'].unique())))

##k = 0
##for i in df['energy'].unique():
##    df_4 = df[df['energy'] == i]
##    wordcloud = WordCloud().generate(df_4['lemma'].to_string())
##    ax[k].imshow(wordcloud)
##    ax[k].set_title(i)
##    ax[k].axis("off")
##    k+=1


# In[ ]:


##energy_pred_words = fies_subset[1].tolist() + fiens_subset[1].tolist()
##energy_pred_words_unique = list(set(energy_pred_words))
##len(energy_pred_words_unique)


# In[ ]:


##vector_subset_energy = inside_vect_energy[energy_pred_words_unique]
##vector_subset_type_energy = pd.concat([df[['energy']],vector_subset_energy], axis=1vector_subset_type_energy.groupby(['energy']).mean()


# In[ ]:


##rfr = RandomForestClassifier(n_estimators=40, max_depth=4, max_features=0.5, random_state=0, n_jobs=-2)


# In[ ]:


##X_train, X_test, y_train, y_test = train_test_split(X_tf_2,df['energy'], shuffle=True, random_state=42, test_size=0.2)


# In[ ]:


##rfr.fit(X_train, y_train)


# In[ ]:


##pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True)).head(10)


# In[ ]:


#Draw the decision tree at estimator 0
##dot_data = StringIO()
##estimator = rfr.estimators_[0]
##export_graphviz(estimator, out_file='tree_limited.dot',feature_names=tfd_2.get_feature_names(),
##                class_names=np.asarray(['sensing', 'intuitive']),
##                filled=True, rounded=True,
##                special_characters=True)
##!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600 
##Image(filename='tree_limited.png')


# ## Nature

# In[ ]:


#Reset the random forests classifier
##rfr = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-2)


# In[ ]:


#Get the predictions for the nature subcategory
##Testlogger = get_scores(small_set_long, Testlogger, 'Nature insight stop included','lemma','nature')


# In[ ]:


##Extract the feature importance from random forest classifier, but display only the features with a gini >= 0.005
##feat_imp_nature_stop = pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True))
##feat_imp_nature_stop was too long to write out so shortened to fins
##fins_subset = feat_imp_nature_stop[feat_imp_nature_stop[0] >= 0.005]
##fins_subset


# In[ ]:


##Convert the word vector to a datatframe
##inside_vect_nature = pd.DataFrame(X_tf_2.A, columns=tfd_2.get_feature_names())


# In[ ]:


##Testlogger = get_scores(small_set_long, Testlogger, 'Nature insight no stop','no_stop_lemma','nature')


# In[ ]:


##Extract the feature importance from random forest classifier, but display only the feature with a gini >= 0.005
##feat_imp_nature_no_stop = pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True))
##finns_subset = feat_imp_nature_no_stop[feat_imp_nature_no_stop[0] >= 0.005]
##finns_subset


# In[ ]:


##fig, ax = plt.subplots(len(df['nature'].unique()), sharex=True, figsize=(15,10*len(df['nature'].unique())))

##k = 0
##for i in df['nature'].unique():
##    df_4 = df[df['nature'] == i]
##    wordcloud = WordCloud().generate(df_4['lemma'].to_string())
##    ax[k].imshow(wordcloud)
##    ax[k].set_title(i)
##    ax[k].axis("off")
##    k+=1


# In[ ]:


##nature_pred_words = fins_subset[1].tolist() + finns_subset[1].tolist()
##nature_pred_words_unique =  list(set(nature_pred_words))
##len(nature_pred_words_unique)


# In[ ]:


##vector_subset_nature = inside_vect_nature[nature_pred_words_unique]
##vector_subset_type_nature = pd.concat([df[['nature']],vector_subset_nature], axis= 1)
##vector_subset_type_nature.groupby(['nature']).mean()


# In[ ]:


##rfr = RandomForestClassifier(n_estimators=40, max_depth=4, max_features=0.5, random_state=0, n_jobs=-2)


# In[ ]:


##X_train, X_test, y_train, y_test = train_test_split(X_tf_2,df['nature'], shuffle=True, random_state=42, test_size=0.2)


# In[ ]:


##rfr.fit(X_train, y_train)


# In[ ]:


##pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True)).head(10)


# In[ ]:


#Draw the decision tree at estimator 0
##dot_data = StringIO()
##estimator = rfr.estimators_[0]
##export_graphviz(estimator, out_file='tree_limited.dot',feature_names = tfd_2.get_feature_names(),
##                class_names = np.asarray(['feeling', 'thinking']),
##                filled=True, rounded=True,
##                special_characters=True)
##!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600 
##Image(filename='tree_limited.png')


# ## Tactics

# In[ ]:


#Reset the random forests classifier
##rfr = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-2)


# In[ ]:


#Get the predictions for the tactics subcategory
##Testlogger = get_scores(small_set_long, Testlogger, 'Tactics insight stop included','lemma','tactics')


# In[ ]:


##Extract the feature importance from random forest classifier, but display only the features with a gini >= 0.005
##feat_imp_tactics_stop = pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True))
##feat_imp_tactics_stop was too long to write out so shortened to fies
##fits_subset = feat_imp_tactics_stop[feat_imp_tactics_stop[0] >= 0.005]
##fits_subset


# In[ ]:


##Convert the word vector to a datatframe
##inside_vect_tactics = pd.DataFrame(X_tf_2.A, columns=tfd_2.get_feature_names())


# In[ ]:


##Testlogger = get_scores(small_set_long, Testlogger, 'Tactics insight no stop','no_stop_lemma','tactics')


# In[ ]:


##Extract the feature importance from random forest classifier, but display only the feature with a gini >= 0.005
##feat_imp_tactics_no_stop = pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True))
##fitns_subset = feat_imp_tactics_no_stop[feat_imp_tactics_no_stop[0] >= 0.005]
##fitns_subset


# In[ ]:


##fig, ax = plt.subplots(len(df['tactics'].unique()), sharex=True, figsize=(15,10*len(df['tactics'].unique())))

##k = 0
##for i in df['tactics'].unique():
##    df_4 = df[df['tactics'] == i]
##    wordcloud = WordCloud().generate(df_4['lemma'].to_string())
##    ax[k].imshow(wordcloud)
##    ax[k].set_title(i)
##    ax[k].axis("off")
##    k+=1


# In[ ]:


##tactics_pred_words = fits_subset[1].tolist() + fitns_subset[1].tolist()
##tactics_pred_words_unique =  list(set(tactics_pred_words))
##len(tactics_pred_words_unique)


# In[ ]:


##vector_subset_tactics = inside_vect_tactics[tactics_pred_words_unique]
##vector_subset_type_tactics = pd.concat([df[['tactics']],vector_subset_tactics], axis=1)
##vector_subset_type_tactics.groupby(['tactics']).mean()


# In[ ]:


##rfr = RandomForestClassifier(n_estimators=40, max_depth=4, max_features=0.5, random_state=0, n_jobs=-2)


# In[ ]:


##X_train, X_test, y_train, y_test = train_test_split(X_tf_2,df['tactics'], shuffle=True, random_state=42, test_size=0.2)


# In[ ]:


##rfr.fit(X_train, y_train)


# In[ ]:


##pd.DataFrame(sorted(zip(map(lambda x: x, rfr.feature_importances_), tfd_2.get_feature_names()), reverse=True)).head(10)


# In[ ]:


#Draw the decision tree at estimator 0
##dot_data = StringIO()
##estimator = rfr.estimators_[0]
##export_graphviz(estimator, out_file='tree_limited.dot',feature_names=tfd_2.get_feature_names(),
##                class_names = np.asarray(['feeling', 'thinking']),
##                filled=True, rounded=True,
##                special_characters=True)
##!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600 
##Image(filename = 'tree_limited.png')


# ### Core set of predicting words

# In[ ]:


##Predicting_pers_words = list(set(mind_pred_words_unique + energy_pred_words_unique + nature_pred_words_unique + tactics_pred_words_unique))##
##len(Predicting_pers_words)


# Seems like 74 words mainly composed of articles, personal pronouns and prepositions are the strongest predictors of personality type.

# # Dimension reduction
# <a id='dimension'></a>

# Allows us to investigate the features and their replationships in two dimensional space.

# We will investigate with and without stopwords

# In[ ]:


def dummy(doc):
        return doc
##tfd_3 = TfidfVectorizer(lowercase=False, tokenizer=dummy, preprocessor=dummy, max_features=50)
##X_tf_3 = tfd_3.fit_transform(df['no_stop_lemma'])


# In[ ]:


##svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=100, random_state=122)


# In[ ]:


##No_stop_lemma = svd_model.fit_transform(X_tf_3.T)


# In[ ]:


##Initialise the bokeh 
##output_notebook()


# In[ ]:


##df_svd = pd.DataFrame(columns=['x', 'y', 'word'])
##df_svd['x'], df_svd['y'], df_svd['word'] = No_stop_lemma[:,0], No_stop_lemma[:,1], tfd_3.get_feature_names()

##Makes an interactive scatter plot with text labels for each point
##Due to the interactivity of the plot, namely the ability to zoom using either selecting over an area or using the scrolling 
##fun, bokeh was chosen over matplotlib.

##source = ColumnDataSource(ColumnDataSource.from_df(df_svd))
##labels = LabelSet(x="x", y="y", text="word", y_offset=8,
##                  text_font_size="8pt", text_color="#555555",
##                  source=source, text_align='center')
 
##plot = figure(plot_width=600, plot_height=600)
##plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
##plot.add_layout(labels)
##show(plot, notebook_handle=True)


# With stopwords removed, we see very slight clustering, such as the words 'don't' and  'think'. But with the main informative features removed, the words do not show any sort of structure. 

# In[ ]:


##terms = tfd_3.get_feature_names()

##for i, comp in enumerate(svd_model.components_):
##    terms_comp = zip(terms, comp)
##    sorted_terms = sorted(terms_comp, key=lambda x:x[1], reverse=True)[:7]
##    print("Topic "+str(i)+": ")
##    for t in sorted_terms:
##        print(t[0])
 ##       print(" ")


# Some of these topics make sense that the words would be grouped together. Some examples include topic 0 and 1.

# In[ ]:


##tfd_4 = TfidfVectorizer(lowercase = False, tokenizer=dummy, preprocessor=dummy, max_features=50)
##X_tf_4 = tfd_4.fit_transform(df['lemma'])


# In[ ]:


##Stop_lemma = svd_model.fit_transform(X_tf_4.T)##


# In[ ]:


##df_svd_2 = pd.DataFrame(columns=['x', 'y', 'word'])
##df_svd_2['x'], df_svd_2['y'], df_svd_2['word'] = Stop_lemma[:,0], Stop_lemma[:,1], tfd_4.get_feature_names()

##Makes an interactive scatter plot with text labels for each point
##Due to the interactivity of the plot, namely the ability to zoom using either selecting over an area or using the scrolling wheel on the mouse 
##, bokeh was chosen over matplotlib.

##source = ColumnDataSource(ColumnDataSource.from_df(df_svd_2))
##labels = LabelSet(x="x", y="y", text="word", y_offset=8,
##                  text_font_size="8pt", text_color="#555555",
##                  source=source, text_align='center')
 
##plot = figure(plot_width=600, plot_height=600)
##plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
##plot.add_layout(labels)
##show(plot, notebook_handle=True)


# Now that the stopwords are back, we can see that there is some clear patterns, such as I and you clustering away from each other as well as my and your. 

# In[ ]:


##terms_stop = tfd_4.get_feature_names()

##for i, comp in enumerate(svd_model.components_):
##    terms_comp = zip(terms_stop, comp)
##    sorted_terms = sorted(terms_comp, key=lambda x:x[1], reverse=True)[:7]
##    print("Topic "+str(i)+": ")
##    for t in sorted_terms:
 ##       print(t[0])
##        print(" ")


# Topics 0 to 3 seem to make sense why these words would be clustered together. 

# # Applying insights to full dataset
# <a id='applyinsight'></a>

# In[ ]:


##mbti['sep_post_rp'] = mbti['posts'].apply(remove_punctuation)
##mbti['tokens_rp'] = mbti['sep_post_rp'].apply(tokeniser.tokenize)
##mbti['no_pers_rp'] = mbti['tokens_rp'].apply(remove_personality)
##mbti['lemma'] = mbti['no_pers_rp'].apply(mbti_lemma, args=(lemmatizer, ))


# In[ ]:


##tfd_5 = TfidfVectorizer(lowercase=False, tokenizer=dummy, preprocessor=dummy)
##X_tf_5 = tfd_5.fit_transform(mbti['lemma'])
##Full_vector = pd.DataFrame(X_tf_5.A, columns=tfd_5.get_feature_names())
##Full_subset = Full_vector[Predicting_pers_words]
##Full_subset.shape


# # Hyperparameter tuning

# In[ ]:


##X_train = Full_subset[train_ind,:]
##X_test = Full_subset[test_ind,:]


# Naive Bayes: alpha determines the level of smoothing the model does, with 0 indicating no smoothing. It is a form of regularisation for Naive Bayes. 
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

# Adaboost: Number of estimators has to do with how many trees are created by the algorithm. The learning rate indicates how each tree contributes to the overall results.
# https://educationalresearchtechniques.com/2019/01/02/adaboost-classification-in-python/

# Logistic regression: C determines how much regularisation is done on the data, while penalty will provide the type of regularisation to perform. L1 represents Lasso regularisation while L2 represents ridge regularisation.
# https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search/

# For a random forest classifier, important factors to consider is how many samples should be in each terminal leaf before stopping. This will be determined by setting min_samples_leaf. Everytime a split is made by the regressor, needs to consider number of features when looking for the best split. Eg, 0.1 indicates that a random subset of 10% of the features will be available to consider for the best split. While log2 is the log2 of the number of features whole sqrt is the square root of the number of features.
# http://course18.fast.ai/ml.html

# In[ ]:


## Reset all algorithms
##NB = MultinomialNB()
##ada = AdaBoostClassifier()
##lorg = LogisticRegression(n_jobs=-1)
##rfr = RandomForestClassifier(n_estimators=40, n_jobs=-1)


# In[ ]:


##param_grid_NB = dict(alpha=[1,0.1,0.01,0.001])
##param_grid_AB = dict(n_estimators=[[50,100,200]], learning_rate=[.001,0.01,.1])
##param_grid_logreg = dict(C=[10,1,0.1,0.01,0.001], penalty=["l1","l2"])
##param_grid_rfr = dict(min_samples_leaf=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], max_features=[0.1,0.3, 0.5,0.7,0.9,"log2","sqrt"])


# In[ ]:



##def get_best_params(iterations = 30):
##    pg = {"NB": param_grid_NB, "adaboost": param_grid_AB, "logreg": param_grid_logreg, "rfr" : param_grid_rfr}
##    md = {"NB" : NB, "adaboost": ada, "logreg": lorg, "rfr" : rfr}
##    b = pd.DataFrame()
    #This loops through the dictionary that contains the three regressors.
##    for key,val in pg.items():
##        random = RandomizedSearchCV(estimator=md[key], param_distributions=val, cv=5, n_iter=iterations, random_state=2)
##        random_result = random.fit(X_train, y_train)
##        a = pd.DataFrame([{'model': key, 'BestScore' : random_result.best_score_, 'BestParameters' : random_result.best_params_}])
##        b = b.append(a)
##    return b


# In[ ]:


#get_best_params(100)


# In[ ]:


##Tuned hyper paramters
##NB = MultinomialNB(alpha=0.1)
##ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.01)
##lorg = LogisticRegression(C=0.001, penalty='l1', n_jobs=-1)
##rfr = RandomForestClassifier(min_samples_leaf=3, max_features=0.5, n_estimators=40,n_jobs=-1)


# In[ ]:


## Run a prediction using a tuned Naive Bayes classifier on our full dataset with only informative features.
##types = y_train.columns.tolist()
##concat_pred = pd.DataFrame()
##for t in types:
##    NB.fit(X_train, y_train[t])
##    temp_var = pd.DataFrame(NB.predict(X_test), columns=[t])
##    concat_pred = pd.concat([concat_pred,temp_var], axis=1)

##comb_NB_id = pd.concat([test_id, concat_pred], axis=1)
#comb_NB_id.to_csv('Naive_bayes_full.csv', index=False)


# In[ ]:


## Run a prediction using a tuned Adaboost classifier on our full dataset with only informative features.
##types = y_train.columns.tolist()
##concat_pred = pd.DataFrame()
##for t in types:
##    ada.fit(X_train, y_train[t])
##    temp_var = pd.DataFrame(ada.predict(X_test), columns=[t])
##    concat_pred = pd.concat([concat_pred,temp_var], axis=1)

##comb_ada_id = pd.concat([test_id, concat_pred], axis=1)
#comb_ada_id.to_csv('Adaboost_full.csv', index=False)


# In[ ]:


## Run a prediction using a tuned Logistic regression classifier on our full dataset with only informative features. 
##types = y_train.columns.tolist()
##concat_pred = pd.DataFrame()
##for t in types:
##    lorg.fit(X_train, y_train[t])
 ##   temp_var = pd.DataFrame(lorg.predict(X_test), columns=[t])
 ##   concat_pred = pd.concat([concat_pred,temp_var], axis=1)

##comb_lorg_id = pd.concat([test_id, concat_pred], axis=1)
#comb_lorg_id.to_csv('lorg_full.csv', index=False)


# In[ ]:


## Run a prediction using a tuned Random forest classifier on our full dataset with only informative features.
##types = y_train.columns.tolist()
##concat_pred = pd.DataFrame()
##for t in types:
##    rfr.fit(X_train, y_train[t])
##    temp_var = pd.DataFrame(rfr.predict(X_test), columns=[t])
##    concat_pred = pd.concat([concat_pred,temp_var], axis=1)
##
##comb_rfr_id = pd.concat([test_id, concat_pred], axis=1)
#comb_rfr_id.to_csv('rfr_full.csv', index=False)


# # Conclusions
# <a id='conclusion'></a>

# -Best prediction score was obtained by just removing separators from the posts, performing tf-idf vectorization, and running random forest classifier with 100 estimators, min_samples_leaf of 3 and max_features of 0.5. This gave us a score of 5.50437

# -Stopwords such as articles, personal pronouns and prepositions seem to be good predictors of personality types.

# -The relatively similar group of words seem to be used by random forests to predict personality types, perhaps the model is sufficiently general to be applicable to other forms of writing, not just tweets.

# -Slightly longer posts by extroverts

# -From our analysis, the fraction of the sentence that a part of speech occupies is not an informative feature for predicting personality types.

# # Future work
# <a id='futurework'></a>

# Due to time and technical restraints, there are still a few questions we wanted to address. We will work on these at a later date.

# -Perform this analysis on the full dataset rather than a small subset, however due to the time taken to complete operations, it was not feasible at this time.

# -We see that the model prediction accuracy improved as the number of estimators increased in our baseline set, perhaps the larger datasets produced by Random oversampling and SMOTE will provide better results if more estimators were used, such as 500. Due to the size of the dataset, when more estimators were added, the Kaggle kernel timed out.

# -Visit the twitter feed where this data was obtained and obtain data for the under-represented personality types

# -More in-depth part of speech analysis where each category of adjective, noun and verb are investigated. Normal counts rather than fractions would also be investigated.

# -Make custom made preprocessor to fix all spelling mistakes and remove stopwords that didn't have apostrophes in them that were skipped when using NLTKs stopword list.

# -Investigate the effect of adding bi-grams and tri-grams to the model

# -Apply a recursive neural network in Keras on the data to see whether a better prediction can be obtained.

# In[ ]:




