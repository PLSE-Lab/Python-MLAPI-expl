#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Utils
from sklearn.utils.multiclass import unique_labels
# Text Processing
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2 ,RFECV
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from collections import defaultdict,Counter
import string
import re

#Spacy NLP
from spacy import displacy
import en_core_web_sm


# Model Evaluation
from sklearn.model_selection import train_test_split,StratifiedKFold ,cross_val_score
from sklearn.metrics import f1_score,classification_report
from sklearn.metrics import confusion_matrix
#Graphics
import matplotlib.pyplot as plt
import seaborn as sns
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn import svm

#disabel annoying sklearn warnings:
import warnings
warnings.filterwarnings('ignore') 

print(os.listdir("../input"))
pd.set_option('display.max_rows',20)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Input data files are available in the "../input/" directory.
train_df=pd.read_csv('../input/cp_challenge_train.csv')
competition_test_df=pd.read_csv('../input/challenge_testset.csv')


# # Exploratory Data Analysis

# In[ ]:


print(train_df.shape,competition_test_df.shape)
counts= train_df['label'].value_counts()
pd.DataFrame({'Sample_Count': counts,'Percentages':counts/train_df.shape[0]}).style.format({'Percentages': '{:.2%}'})


# Imbalanced Multi-Class ,top 3 make 70% of samples!

# In[ ]:


# Try to predict it yourself (get a sense of how hard it is)
train_df['num_sents']= train_df['Plot'].apply(lambda x: len(x.split('.')))
rnd_sample=train_df[train_df['num_sents']==1].sample()
rnd_sample['Plot'].tolist()


# In[ ]:


rnd_sample['Title'].tolist()


# In[ ]:


rnd_sample['label'].tolist()


# # Data Processing

# In[ ]:


# Use Title 
train_df['Plot']=train_df['Plot']+' '+train_df['Title']
competition_test_df['Plot']=competition_test_df['Plot']+' '+competition_test_df['Title']


# In[ ]:


# Split the train_df into train & test sets 
y = train_df['label']
X = train_df['Plot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0,shuffle=True,stratify=y )
print(y_train.shape[0],y_test.shape[0])
# print("X_test sample counts:")
# counts=y_test.value_counts()
# pd.DataFrame({'Sample_Count': counts,'Percentages':counts/y_test.shape[0]}).style.format({'Percentages': '{:.2%}'})


# # BOW Vectorization

# Shortcomings:
# 1. Lost information: Words Order,meaning(semantics) & Context.
# 2. Sparse, number of features is case-specific and usually vast.

# In[ ]:


# Vectorize 1:
nlp = en_core_web_sm.load()
def spacy_tokenizer(text):
    tokenlist=[]
    mytokens = nlp(text)
    for token in mytokens:
        # 1.Remove stop words ,punctuations and pronouns,2.lemmatize
        if not token.is_stop and not token.is_punct:
            if token.lemma_ != "-PRON-":
                tokenlist.append(token.lemma_)
#                 tokenlist.append(token.text)
    return tokenlist
vect = CountVectorizer(tokenizer = spacy_tokenizer,min_df=4,binary=True,ngram_range=(1,1) )
train_sprse = vect.fit_transform(X_train)
test_sprse = vect.transform(X_test)
# In dataframe form makes it easier later on
X_train_df = pd.DataFrame(train_sprse.toarray(),columns=vect.get_feature_names())
X_test_df = pd.DataFrame(test_sprse.toarray(),columns=vect.get_feature_names())
print("Number of features(unique tokens):",len(vect.get_feature_names()))


# In[ ]:


# Vectorize 2:
vect = CountVectorizer(stop_words='english',min_df=4,binary=True,ngram_range=(1,1) )
# Using the out-of-the-box tfidf vectorizer doesn't make sense
# vect = TfidfVectorizer(stop_words='english',min_df=1,binary=True,sublinear_tf =True)
train_sprse = vect.fit_transform(X_train)
test_sprse = vect.transform(X_test)
# In dataframe form makes it easier later on
X_train_df = pd.DataFrame(train_sprse.toarray(),columns=vect.get_feature_names())
X_test_df = pd.DataFrame(test_sprse.toarray(),columns=vect.get_feature_names())
print("Number of features(unique tokens):",len(vect.get_feature_names()))


# In[ ]:


for genre in unique_labels(y_train):
    y_binary= pd.Series(np.where(y_train.values == genre, 1, 0),y_train.index)
    print(genre)
    res = chi2(train_sprse, y_binary)
    df = pd.DataFrame({'chi2':res[0],'pval':res[1]},index=vect.get_feature_names())
    df.sort_values(by='chi2' ,axis=0,ascending=False, inplace=True)
    print(df.head(10),end='\n\n')
    
# ch2 = SelectKBest(chi2,k=100)
# train_sprse= ch2.fit_transform(train_sprse,y_train)


# In[ ]:


clf = MultinomialNB()
# StratifiedKFold CV F1 score:
print(cross_val_score(clf,train_sprse,y_train,cv=5,scoring='f1_micro'))

# Fit and predict
model = clf.fit(train_sprse, y_train)
y_pred = model.predict(test_sprse)


# In[ ]:


# Investigate model predictions
print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False))
print("predictions counter:",Counter(y_pred))

#Sorted confusion_matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=unique_labels(y_test) ,columns = unique_labels(y_test))
sorted_index=df_cm.sum(axis=0).sort_values(ascending=False).index
df_cm = df_cm.reindex(sorted_index, axis=1)
df_cm = df_cm.reindex(sorted_index, axis=0)
plt.figure(figsize = (12,10))
sns.heatmap(df_cm,annot=True,cmap='Blues', fmt='g',cbar=False)


# **** To predict on the competition_test_df , the model is retrained on all the train samples setting test_size=0. it somewhat improves the score.****

# ## What is Naive Bayes ?
# * **Generative** algorithm , learns P(X|y) & P(y)  *the class prior*
# * Under the independency 'Naive' assumption we can use Bayes Rule:
# P(y|X)=P(X|y)*P(y)/P(X)
# * Use argmax(P) to predict the class

# ## Why Naive Bayes ? 
# #### Generally, as a baseline model:
# * Fast to run , fast to implement
# * Irrelevant features tend to cancel each other out.
# * If naivety holds, it is proved optimal (for the specific set of features)
# 
# #### Specifically, for our problem:
# * Inherently Multinomial (as oppesed to binary classifiers where ovr,ova is required)
# * Embrace and enhance the imbalance 
# * works well for many **equally important** features (as opposed to desicion trees for example)
# 

# ## *I hear ya*. Still, how such a simple baseline model is in the same ballpark with state of the art models?
# * Diminishing returns as we're approching the Bayes optimal error ,judging by human Level performance( it is **Natural** language processing).
# * There's not enough data to learn such complicated embeddings/context patterns,many of the samples consist of only one line. 
# * Heterogeneous data. The plots were written by different people over many decades, long enough that even slang has changed. 

# ## Possible Improvements 
# * Optimizing the vectorizer and alpha parameter of the NB
# * Ensemble using meta classifiers focused on the minority classes(Trees?).
# * over/under sampling (recommended python library: imblearn ) 
# 
