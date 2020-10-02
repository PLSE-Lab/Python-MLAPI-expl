#!/usr/bin/env python
# coding: utf-8

# #Importing libraries

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
np.random.seed(500)
import warnings
from sklearn.utils.testing import ignore_warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(ngram_range=(1,3))
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.svm import LinearSVC
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# #EDA

# In[ ]:


df = pd.read_json(os.path.join(dirname, filename),lines=True)
df.head()


# In[ ]:


df.info()


# There are no NULL entries in the dataset, which is good!

# In[ ]:


labels = list(df.category.unique())
labels.sort()
print(labels)


# In[ ]:


plt.figure(figsize=(14,6))
df.category.value_counts().plot(kind='bar')
plt.show()


# These are all the categories in the dataset. 
# Some of these seem to be repeated, like ARTS, CULTURE & ARTS, and ARTS & CULTURE. We will merge these together

# In[ ]:


df.category[(df['category']=='ARTS') | (df['category']=='CULTURE & ARTS')]='ARTS & CULTURE'
df.category[df['category']=='PARENTS']='PARENTING'
df.category[df['category']=='STYLE']='STYLE & BEAUTY'
df.category[df['category']=='THE WORLDPOST']='WORLDPOST'


# In[ ]:


labels = list(df.category.unique())
labels.sort()
print(labels)
plt.figure(figsize=(14,6))
df.category.value_counts().plot(kind='bar')
plt.show()


# #Preprocessing

# In[ ]:


def preprocessing(col,h_pct=1,l_pct=1):
    '''
    Cleans the text in the input column

    Parameters
    ----------
    col : pandas.core.series.Series
        The column which needs to be processed
    h_pct : float (default = 1)
        The percentage of high frequency words to remove from the corpus
    l_pct : float (default = 1)
        The percentage of low frequency words to remove from the corpus
    
    Returns
    -------
    cleaned text series
    '''
    #Lower case
    lower = col.apply(str.lower)
    
    #Removing HTML tags
    import html
    rem_html = lower.apply(lambda x: x.replace('#39;', "'").replace('amp;', '&')
                             .replace('#146;', "'").replace('nbsp;', ' ').replace('#36;', '$')
                             .replace('\\n', "\n").replace('quot;', "'").replace('<br />', " ")
                             .replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.')
                             .replace(' @-@ ','-').replace('\\', ' \\ ').replace('&lt;','<')
                             .replace('&gt;', '>'))
    
    #Lemmatizing
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    
    #Stemming
    from nltk.stem import SnowballStemmer
    stem = SnowballStemmer('english')
    stemmed = rem_html.apply(lambda x: ' '.join(stem.stem(word) for word in str(x).split()))
    
    #removing punctuation
    import re
    rem_punc = stemmed.apply(lambda x: re.sub(r'[^\w\s]',' ',x))
    
    #removing stopwords and extra spaces
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    rem_stopwords = rem_punc.apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    
    #removing numbers
    rem_num = rem_stopwords.apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
    
    #remove words having length=1
    rem_lngth1 = rem_num.apply(lambda x: re.sub(r'[^\w\s]',' ',x))
    
    if h_pct != 0:
        #removing the top $h_pct of the most frequent words 
        high_freq = pd.Series(' '.join(rem_lngth1).split()).value_counts()[:int(pd.Series(' '.join(rem_lngth1).split()).count()*h_pct/100)]
        rem_high = rem_lngth1.apply(lambda x: " ".join(x for x in x.split() if x not in high_freq))
    else:
        rem_high = rem_lngth1
    
    if l_pct != 0:
        #removing the top $l_pct of the least frequent words
        low_freq = pd.Series(' '.join(rem_high).split()).value_counts()[:-int(pd.Series(' '.join(rem_high).split()).count()*l_pct/100):-1]
        rem_low = rem_high.apply(lambda x: " ".join(x for x in x.split() if x not in low_freq))
    else:
        rem_low = rem_high
    
    return rem_low


# This is a standard text preprocessing UDF which I use.
# Most of the functions here are self-explanatory, except the h_pct and l_pct parts.
# Let me elaborate a bit on this...

# In[ ]:


counts = pd.Series(' '.join(df.short_description).split()).value_counts()
counts


# These are the counts of all the words/tokens in the dataset.
# We will remove a percentage of the most and least frequent words.
# (This is the last step of the preprocessing function, so stopwords, and punctuation will already be removed by then)

# In[ ]:


high_freq = counts[:int(pd.Series(' '.join(df.short_description).split()).count()*1/100)]
high_freq


# These are the top 1% of the most frequent words.

# In[ ]:


low_freq = counts[:-int(pd.Series(' '.join(df.short_description).split()).count()*1/100):-1]
low_freq


# These are the top 1% of the least frequent words. All of these words occur only once in the vocabulary, and thus don't have much significance and can be removed

# In[ ]:


df.loc[df.short_description.str.len() == df.short_description.str.len().max()]


# In[ ]:


df.loc[58142]['short_description']


# This is the longest story in our dataset. We will use this as reference to see how our preprocessing function works

# #Model building

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
def prep_fit_pred(df, h_pct, l_pct, model, verbose=False):
    '''
    Takes the dataframe, and returns asset tag predictions for the stories

    Parameters
    ----------
    col : pandas.core.frame.DataFrame
    h_pct : float
        The percentage of high frequency words to remove from the corpus
    l_pct : float
        The percentage of low frequency words to remove from the corpus
    model : the model which will be used for predictions
    verbose : boolean (default: False)
        Verbosity of the output. True = all outputs, False = no outputs
            
    Returns
    -------
    preds : pandas.core.series.Series
        Column with the predicted asset class
    acc : float
        Accuracy of the predictions on the test set
    model : the trained model
    '''
    
    df['short_description_processed'] = preprocessing(df['short_description'],h_pct,l_pct)
    df['concatenated'] = df['headline'] + '\n' + df['short_description_processed']
    #not removing high and low frequency words from headline
    #this is because the headline carries more significance in determining the classification of the news
    df['concat_processed'] = preprocessing(df['concatenated'],0,0)
    
    if verbose:
        print('Number of words in corpus before processing: {}'
              .format(df['short_description'].apply(lambda x: len(x.split(' '))).sum()))
        print('Number of words in corpus after processing: {} ({}%)'
              .format(df['short_description_processed'].apply(lambda x: len(x.split(' '))).sum()
                     , round(df['short_description_processed'].apply(lambda x: len(x.split(' '))).sum()*100\
                             /df['short_description'].apply(lambda x: len(x.split(' '))).sum())))
        print('Number of words in final corpus: {} ({}%)'
              .format(df['concat_processed'].apply(lambda x: len(x.split(' '))).sum()
                     , round(df['concat_processed'].apply(lambda x: len(x.split(' '))).sum()*100\
                             /df['short_description'].apply(lambda x: len(x.split(' '))).sum())))

        print('\nRaw story:\n{}'.format(df['short_description'][58142]))
        print('\nProcessed story:\n{}'.format(df['short_description_processed'][58142]))
        print('\nAdding additional columns to story:\n{}'.format(df['concatenated'][58142]))
        print('\nFinal story:\n{}'.format(df['concat_processed'][58142]))

    X = df['concat_processed']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, 
                                                    stratify=y) 
    
    bow_xtrain = bow.fit_transform(X_train)
    bow_xtest = bow.transform(X_test)

    model.fit(bow_xtrain,y_train)
    preds = model.predict(bow_xtest)

    acc = accuracy_score(y_test,preds)*100
    
    if verbose:
        print('\nPredicted class: {}'.format(preds[58142]))
        print('Actual class: {}\n'.format(y_test.iloc[58142]))
        plt.figure(figsize=(14,14))
        sns.heatmap(confusion_matrix(y_test,preds),cbar=False,annot=True,square=True
               ,xticklabels=labels,yticklabels=labels,fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.yticks(rotation=0)
        plt.show()
        print(classification_report(y_test,preds))
        print('Accuracy: {0:.2f}%'.format(acc))

    return preds, acc, model


# This one function takes the dataframe, the high and low percet words to remove, the model and a verbosity flag as inputs, and returns the  predictions, model accuracy, and the trained model

# **The below cell has been converted to markdown because of the time it takes to run.
# Change to code for reference**

# #Checking for optimum h_pct and l_pct combination
# %matplotlib inline
# 
# import warnings
# from sklearn.utils.testing import ignore_warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# from sklearn.exceptions import ConvergenceWarning
# warnings.simplefilter(action='ignore', category=ConvergenceWarning)
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# bow = CountVectorizer(ngram_range=(1,3))
# from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
# from sklearn.svm import LinearSVC
# 
# pct=[]
# svc_max, svc_max_pct, acc_svc = 0, '', []
# 
# for i in tqdm_notebook(np.linspace(0,10,11)):
#     for j in tqdm_notebook(np.linspace(0,10,11)):
#         
#         pct.append(str(i)+'|'+str(j))
#         
#         preds, acc, _ = prep_fit_pred(df, i, j, LinearSVC())
#         acc_svc.append(acc)
#         if acc > svc_max:
#             svc_max = acc
#             svc_max_pct = str(i)+'|'+str(j)
# 
# print('SVC max: {}%, pct:{}'.format(svc_max,svc_max_pct))
# 
# plt.figure(figsize=(18,10))
# ax = plt.axes()
# plt.plot(pct,acc_svc,label='SVC',marker='v')
# plt.xlabel('pct')
# plt.ylabel('accuracy')
# plt.legend()
# plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='x-small')
# plt.show()

# This cell finds the optimal values of h_pct and l_pct. We start with integer values from 0 to 10, and based on the results, can further tune the percentages to 0.5% steps.
# 
# Its takes a while to run (ok, more than a while), so I'll just paste the results I achieved in my local.
# 
# The first iteration returned values of 0.0 and 1.0 for h_pct and l_pct respectivaly, the below are the results when running for vallues between 0.0 to 0.5% for h_pct and 0.5 to 1.5% for l_pct

# > SVC max: 63.79560061555173%, pct:0.0|1.0
# ![image.png](attachment:image.png)![](http://)

# *Apologies for the labels (this is my first kernel on Kaggle).*
# 
# The optinal values are still 0.0 and 1.0 for h_pct and l_pct respectively. We'll go ahead with these values

# #Putting it together

# In[ ]:


preds_abc, acc_abc, abc = prep_fit_pred(df, 0, 1, LinearSVC(), verbose=True)


# **This is my first kernel on kaggle. Any feedback will be GREATLY appreciated.
# Thanks a lot! :)**
