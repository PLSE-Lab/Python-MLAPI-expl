#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
get_ipython().system('unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
get_ipython().system('unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
get_ipython().system('unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import math
import gc
from sklearn.metrics import roc_auc_score
import joblib # for saving models
import warnings
warnings.filterwarnings('ignore')


# ## Creating Small Train and Test Sets (10:6)

# In[ ]:


df_original=pd.read_csv('/kaggle/working/train.csv')
df_original.head(3)


# In[ ]:


df=pd.concat([df_original[df_original['toxic']!=1][:5],df_original[df_original['toxic']==1][:5]]).reset_index(drop=True)
df=df.sample(frac=1).reset_index(drop=True)
df


# In[ ]:


df_test=pd.concat([df_original[df_original['toxic']!=1][10:13],df_original[df_original['toxic']==1][10:13]]).reset_index(drop=True)
df_test=df_test.sample(frac=1).reset_index(drop=True)
df_test


# ## Data Cleaning

# In[ ]:


def print_comment(df,column):
    for index,text in enumerate(df[column]):
        print('Comment %d:\n'%(index+1),text)


# In[ ]:


print_comment(df,'comment_text')


# In[ ]:


# Lowercasing the text
df['cleaned']=df['comment_text'].apply(lambda x:x.lower())


# In[ ]:


print_comment(df,'cleaned')


# In[ ]:


# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not","can't": "can not","can't've": "cannot have",
"'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
"didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
"hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
"he'll've": "he will have","how'd": "how did","how'd'y": "how do you","how'll": "how will","i'd": "i would",
"i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have",
"isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have",
"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
"mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
"needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
"oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
"shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will",
"she'll've": "she will have","should've": "should have","shouldn't": "should not",
"shouldn't've": "should not have","so've": "so have","that'd": "that would","that'd've": "that would have",
"there'd": "there would","there'd've": "there would have",
"they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
"they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",
"we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
"weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
"what've": "what have","when've": "when have","where'd": "where did",
"where've": "where have","who'll": "who will","who'll've": "who will have","who've": "who have",
"why've": "why have","will've": "will have","won't": "will not","won't've": "will not have",
"would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you would","you'd've": "you would have","you'll": "you will","you'll've": "you will have",
"you're": "you are","you've": "you have"}


# In[ ]:


# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)


# In[ ]:


df['cleaned']=df['cleaned'].apply(lambda x:expand_contractions(x))


# In[ ]:


print_comment(df,'cleaned')


# In[ ]:


def clean_text(text):
    # removing word with digits
    text=re.sub('\w*\d\w*','', text)
    # removing \n from comments
    text=re.sub('\n',' ',text)
    # removing anything which is not an alphabet
    text=re.sub('[^a-z]',' ',text)
    return text


# In[ ]:


df['cleaned']=df['cleaned'].apply(lambda x: clean_text(x))


# In[ ]:


print_comment(df,'cleaned')


# In[ ]:


# Removing extra spaces
df['cleaned']=df['cleaned'].apply(lambda x: re.sub(' +',' ',x))


# In[ ]:


print_comment(df,'cleaned')


# In[ ]:


# Stopwords removal & Lemmatizing tokens using SpaCy
import spacy
nlp = spacy.load('en_core_web_sm',disable=['parser','ner'])


# In[ ]:


# Removing Stopwords and Lemmatizing words
df['lemmatized']=df['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))


# In[ ]:


print_comment(df,'lemmatized')


# In[ ]:


# Cleaning Test Set
# Lowercasing the text
df_test['cleaned']=df_test['comment_text'].apply(lambda x:x.lower())
# Expanding contractions
df_test['cleaned']=df_test['cleaned'].apply(lambda x:expand_contractions(x))
# Cleaning the text
df_test['cleaned']=df_test['cleaned'].apply(lambda x: clean_text(x))
# Removing extra spaces
df_test['cleaned']=df_test['cleaned'].apply(lambda x: re.sub(' +',' ',x))
# Removing Stopwords and Lemmatizing words
df_test['lemmatized']=df_test['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))


# In[ ]:


print_comment(df_test,'lemmatized')


# ## EDA

# In[ ]:


text_word_count = []

#populate the lists with comments lengths
for i in df['lemmatized']:
      text_word_count.append(len(i.split()))

length_df = pd.DataFrame({'Word Count Distribution':text_word_count})
length_df.hist(bins = 3, range=(0,length_df['Word Count Distribution'].max()),figsize=(10,8))
plt.show()


# In[ ]:


df.groupby('toxic')['lemmatized'].apply(lambda x: ' '.join(x))[1]


# In[ ]:


# Preparing data for document term matrix
temp=[]
temp.append(df.groupby('toxic')['lemmatized'].apply(lambda x: ' '.join(x))[0])
temp.append(df.groupby('toxic')['lemmatized'].apply(lambda x: ' '.join(x))[1])

df_for_dtm=pd.DataFrame(columns=['type','text'])
df_for_dtm['type']=['non-toxic','toxic']
df_for_dtm['text']=temp
df_for_dtm=df_for_dtm.set_index('type',drop=True)
df_for_dtm


# In[ ]:


# Creating Document Term Matrix for generating Word Cloud
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(analyzer='word')

data=cv.fit_transform(df_for_dtm['text'])

df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index=df_for_dtm.index
df_dtm=df_dtm.transpose()
df_dtm


# In[ ]:


# Generating Word Clouds
from textwrap import wrap
from wordcloud import WordCloud
def generate_wordcloud(data,title):
    wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title('\n'.join(wrap(title,60)),fontsize=13)
    plt.show()


# In[ ]:


for index,type_of_text in enumerate(df_dtm.columns):
    generate_wordcloud(df_dtm[type_of_text].sort_values(ascending=False),type_of_text)


# ## Feature Engineering (TF-IDF)

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


# initialize TFIDF
vec = TfidfVectorizer(ngram_range=(1,3), stop_words='english',analyzer='word',dtype=np.float32)


# In[ ]:


# create TFIDF for train
tfidf = vec.fit_transform(df['lemmatized'])


# In[ ]:


tfidf


# In[ ]:


# create TFIDF for test
tfidf_test = vec.transform(df_test['lemmatized'])


# In[ ]:


tfidf_test


# ## Modeling

# In[ ]:


X_train=tfidf.toarray()
X_test=tfidf_test.toarray()


# In[ ]:


Y_train=df['toxic'].values
Y_test=df_test['toxic'].values


# In[ ]:


# Naive Bayes Model
class NaiveBayes:
    def fit(self,X,Y):
        n_samples,n_features=X.shape
        self.classes=np.unique(Y)
        n_class=len(self.classes)
        
        #initializing mean, variance, prior probabilities
        self.mean=np.zeros((n_class,n_features),dtype=np.float64)
        self.var=np.zeros((n_class,n_features),dtype=np.float64)
        self.prior_prob=np.zeros(n_class,dtype=np.float64)
        
        #calculating mean, variance and prior probabilities
        for cls in self.classes:
            X_for_cls=X[Y==cls]
            self.mean[cls,:]=X_for_cls.mean(axis=0)
            self.var[cls,:]=X_for_cls.var(axis=0)
            self.prior_prob[cls]= X_for_cls.shape[0]/float(n_samples)
            
    
    def predict_for_one(self,x):
        posteriors=[]
        
        for index,cls in enumerate(self.classes):
            prior=np.log(self.prior_prob[index])
            mean=self.mean[index]
            var=self.var[index]
            likelihood=np.log(np.exp(- (x-mean)**2 / (2 * var))/np.sqrt(2 * np.pi * var))
            print(likelihood)
            class_conditional=np.sum(likelihood)
            posterior=prior+class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    
    def predict_class(self,X):
        Y_pred=[self.predict_for_one(x) for x in X]
        return Y_pred


# #### My scratch implementation

# In[ ]:


model=NaiveBayes()
model.fit(X_train,Y_train)


# In[ ]:


#In-sample Evaluation
train_pred=model.predict_class(X_train)
#Out-sample Evaluation
test_pred=model.predict_class(X_test)


# In[ ]:


print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(Y_train,train_pred))
print('Out-sample Evaluation ROC-AUC Score\n',roc_auc_score(Y_test,test_pred))


# #### Sklearn GaussianNB

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


model=GaussianNB()
model.fit(X_train,Y_train)


# In[ ]:


#In-sample Evaluation
train_pred=model.predict(X_train)
#Out-sample Evaluation
test_pred=model.predict(X_test)


# In[ ]:


print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(Y_train,train_pred))
print('Out-sample Evaluation ROC-AUC Score\n',roc_auc_score(Y_test,test_pred))


# In[ ]:




