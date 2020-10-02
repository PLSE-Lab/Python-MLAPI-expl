#!/usr/bin/env python
# coding: utf-8

# # Intoduction

# Being anonymous over the internet can sometimes make people say nasty things that they normally would not in real life. Let's filter out the hate from our platforms one comment at a time.

# # Objective
#  

# To create an EDA/ feature-engineering starter notebook for toxic comment classification.

# # Data Overview

# The dataset here is from wiki corpus dataset which was rated by human raters for toxicity. The corpus contains 63M comments from discussions relating to user pages and articles dating from 2004-2015.
# 
# Different platforms/sites can have different standards for their toxic screening process. Hence the comments are tagged in the following five categories

# **EDA**

# > I will visulaize and analysis the comments in the training data before moving on to the modeling section.

# **Install and import necessary packages**

# In[ ]:


get_ipython().system('pip install -q pyicu')
get_ipython().system('pip install -q pycld2')
get_ipython().system('pip install -q polyglot')
get_ipython().system('pip install -q textstat')
get_ipython().system('pip install -q googletrans')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import os
import gc
import re
import folium
from scipy import stats
from colorama import Fore, Back, Style, init
import string
import math
import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats as ss

import random
import networkx as nx
from pandas import Timestamp

from PIL import Image
from IPython.display import SVG
from keras.utils import model_to_dot
import string
import requests
from IPython.display import HTML

import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt

tqdm.pandas()

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import transformers
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from tensorflow.keras.models import Model
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.optimizers import Adam
from tokenizers import BertWordPieceTokenizer
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding
from tensorflow.keras.layers import LSTM, GRU, Conv1D, SpatialDropout1D

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.constraints import *
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import *



from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer

from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer  

import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from googletrans import Translator
from nltk import WordNetLemmatizer
from polyglot.detect import Detector
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
stopword=set(STOPWORDS)
#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

np.random.seed(0)


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


valid1=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')
valid1.head()


# In[ ]:


validate=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv')
validate.head()


# In[ ]:


comment=pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv")
display(comment.head())
print("Shape:",comment.shape)


# In[ ]:


validation=pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")
display(validation.head())
print('shape:',validation.shape)


# In[ ]:


test_processed=pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test-processed-seqlen128.csv")
display(test_processed.head())
print('shape:',test_processed.shape)


# In[ ]:


train=pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
display(train.head())
print("Shape:",train.shape)


# In[ ]:


test=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
display(test.head())
print("shape:",test.shape)


# In[ ]:


test['lang'].unique()


# In[ ]:


sns.countplot(test['lang'])


# In[ ]:


train_bias=pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train-processed-seqlen128.csv")
display(train_bias.head())
print("shape:",train_bias.shape)


# In[ ]:


train_bias.columns


# In[ ]:


submission=pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
submission.head()


# In[ ]:


nrow_train=train.shape[0]
nrow_test=test.shape[0]
sum=nrow_train+nrow_test
print("       : train : test")
print("rows   :",nrow_train,":",nrow_test)
print("perc   :",round(nrow_train*100/sum),"   :",round(nrow_test*100/sum))


# There is 70:30 train:test split and test might be changed in future .
# but in this case, train:test 78:22.it indicates that class is balanced in dataset.

# In[ ]:


x=train.iloc[:,2:].sum()
#marking comments without any tags as "clean"
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)
#count number of clean entries
train['clean'].sum()
print("Total comments = ",len(train))
print("Total clean comments = ",train['clean'].sum())
print("Total tags =",x.sum())


# In[ ]:


print("Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)

print("filling NA with \"unknown\"")
train["comment_text"].fillna("unknown", inplace=True)


# In[ ]:


print("Check for missing values in Test dataset")
null_check=test.isnull().sum()
print(null_check)
print("filling NA with \"unknown\"")


# In[ ]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()


# **Example of Commen**ts:

# In[ ]:


print("toxic:")
print(train[train.severe_toxic==1].iloc[3,1])
#print(train[train.severe_toxic==1].iloc[5,1])


# In[ ]:


print("severe_toxic:")
print(train[train.severe_toxic==1].iloc[4,1])
#print(train[train.severe_toxic==1].iloc[4,1])


# In[ ]:


print("Threat:")
print(train[train.threat==1].iloc[1,1])


# In[ ]:


print("Obscene:")
print(train[train.obscene==1].iloc[1,1])


# In[ ]:


print("identity_hate:")
print(train[train.identity_hate==1].iloc[4,1])


# In[ ]:


x=train.iloc[:,2:].sum()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[ ]:


x=rowsums.value_counts()

#plot
plt.figure(figsize=(8,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Multiple tags per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of tags ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')


# In[ ]:


fig = go.Figure(data=[
    go.Pie(labels=train.columns[2:7],
           values=train.iloc[:, 2:7].sum().values, marker=dict(colors=px.colors.qualitative.Plotly))
])
fig.update_traces(textposition='outside', textfont=dict(color="black"))
fig.update_layout(title_text="Pie chart of labels")
fig.show()


# The toxicity is  highly spread out across classes.

# In[ ]:


temp_df=train.iloc[:,2:-1]
# filter temp by removing clean comments
# temp_df=temp_df[~train.clean]

corr=temp_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)


# In[ ]:


# https://pandas.pydata.org/pandas-docs/stable/style.html
def highlight_min(data, color='orange'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_max = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),index=data.index, columns=data.columns)


# In[ ]:


#Crosstab
# Since technically a crosstab between all 6 classes is impossible to vizualize, lets take a 
# look at toxic with other tags
main_col="toxic"
corr_mats=[]
for other_col in temp_df.columns[1:]:
    confusion_matrix = pd.crosstab(temp_df[main_col], temp_df[other_col])
    corr_mats.append(confusion_matrix)
out = pd.concat(corr_mats,axis=1,keys=temp_df.columns[1:])

#cell highlighting
out = out.style.apply(highlight_min,axis=0)
out


# The above table represents the Crosstab/ consufion matix of Toxic comments with the other classes.
# 
# Some interesting observations:
# 
# A Severe toxic comment is always toxic
# 
# Other classes seem to be a subset of toxic barring a few exceptions

# **Wordcloud of all comments**

# In[ ]:


def nonan(x):
    if type(x)==str:
        return x.replace("\n","")
    else:
        return ""
text=''.join([nonan(abstract)for abstract in train["comment_text"]])
wordcloud=WordCloud(max_font_size=None,background_color="black",collocations=False,width=1240,height=800).generate(text)
fig=px.imshow(wordcloud)
fig.update_layout(title="Common words in comments")


# In the wordcloud above, we can see the most common words in the comments. These words include "wikipedia", "page","Thank", and "article" among other words. More offensive words like "f**k" seem to occur less often, indicating that toxic, insulting comments are seen less frequently than non-toxic comments

# **Wordclouds-Frequent words:**

# Now,let's take a look a t words that are associated with these classes.
# 
# Chart Desc:The visuals here are word clouds (ie0 more frequent words appear bigger
# 

# A simpler way would be to create a new kaggle dataset and import images from there.

# In[ ]:


get_ipython().system('du -l ../input/*')


# In[ ]:


get_ipython().system('ls ../input/imagesforkernal/')
stopword=set(STOPWORDS)


# In[ ]:


#clean comments
clean_mask=np.array(Image.open("../input/imagesforkernal/safe-zone.png"))
clean_mask=clean_mask[:,:,1]
#wordcloud for clean comments
subset=train[train.clean==True]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=clean_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Words frequented in Clean Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# In[ ]:


toxic_mask=np.array(Image.open("../input/imagesforkernal/toxic-sign.png"))
toxic_mask=toxic_mask[:,:,1]
#wordcloud for clean comments
subset=train[train.toxic==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=4000,mask=toxic_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(20,20))
plt.subplot(221)
plt.axis("off")
plt.title("Words frequented in Toxic Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)

#Severely toxic comments
plt.subplot(222)
severe_toxic_mask=np.array(Image.open("../input/imagesforkernal/bomb.png"))
severe_toxic_mask=severe_toxic_mask[:,:,1]
subset=train[train.severe_toxic==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=severe_toxic_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Severe Toxic Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Reds' , random_state=244), alpha=0.98)

#Threat comments
plt.subplot(223)
threat_mask=np.array(Image.open("../input/imagesforkernal/anger.png"))
threat_mask=threat_mask[:,:,1]
subset=train[train.threat==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=threat_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Threatening Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'summer' , random_state=2534), alpha=0.98)

#insult
plt.subplot(224)
insult_mask=np.array(Image.open("../input/imagesforkernal/swords.png"))
insult_mask=insult_mask[:,:,
                        1]
subset=train[train.insult==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=insult_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in insult Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)


plt.show()


# In[ ]:


#identity_hate
subset=train[train.toxic==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=4000,mask=toxic_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.subplot(111)
identity_hate=np.array(Image.open("../input/imagesforkernal/megaphone.png"))
identity_hate=identity_hate[:,:,1]
subset=train[train.identity_hate==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=identity_hate,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in identity_hate Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)
#obsence




# In[ ]:


#obsence
plt.subplot(122)
obscene=np.array(Image.open("../input/imagesforkernal/biohazard-symbol.png"))
obscene=obscene[:,:,1]
subset=train[train.obscene==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=obscene,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in obscene Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)


# **Languages**

# Now,We will go for analysis of language in the dataset and detect the language present in the comments.
# 

# In[ ]:


def get_language(text):
    return Detector("".join(x for x in text if x.isprintable()), quiet=True).languages[0].name

train["lang"] = train["comment_text"].progress_apply(get_language)


# **English Vs Non English**

# In[ ]:


lang_list=sorted(list(set(train["lang"])))
counts=[list(train["lang"]).count(cont) for cont in lang_list]
df=pd.DataFrame(np.transpose([lang_list,counts]))
df.columns=["Language","Count"]
df["Count"]=df["Count"].apply(int)

display(df["Language"].unique())
display(sns.countplot(df["Language"]))

fig=px.bar(df,x="Language",y="Count",title="Language of comments",color='Language',text="Count",width=700)
display(fig)


# In[ ]:


df["Count"].sum()


# In[ ]:


fig = px.bar(df.query(" Language!='un' & Language != 'en'").query("Count >= 50"),y="Language", x="Count", title="Language of non-English comments",template="plotly_white", color="Language", text="Count", orientation="h")
fig.update_traces(marker=dict(line=dict(width=0.75,color='black')),  textposition="outside")
fig.update_layout(showlegend=False)
fig


# We can see in bar chart .German(de),Scots(sc) and Danish(da) are the most common non-english languages featuring in the dataset,with more than 100 comments in each language.remaining non english language are not far behind.

# In[ ]:


fig = go.Figure([go.Pie(labels=df.query("Language!='un' & Language != 'en'").query("Count >= 50")["Language"],
values=df.query(" Language!='un' & Language !='en'").query("Count >= 50")["Count"])])
fig.update_layout(title_text="Pie chart of non-English languages", template="plotly_white")
fig.data[0].marker.colors = [px.colors.qualitative.Plotly[2:]]
fig.data[0].textfont.color = "black"
fig.data[0].textposition = "outside"
fig.show()


# from the Pie chart above, we can once see that German,Danish ,and Scots with more than 15% of the pie belonging to each of these three languages.

# In[ ]:


def get_country(Language):
    if Language=="de":
        return "Germany"
    if Language=="sco":
        return "Scotland"
    if Language=="da":
        return "Denmark"
    if Language=="ar":
        return "Saudi Arabia"
    if Language=="es":
        return "Spain"
    if Language=="fa":
        return "Iran"
    if Language=="el":
        return "Greece"
    if Language=="pt":
        return "Portugal"
    
    if Language=="en":
        return "United Kingdom"
    if Language=="ht":
        return "India"
    if Language=="aa":
        return "Albania"
    if Language=="bn":
        return "Bosnia and Herzegovina"
    if Language=="crs":
        return "Croatia"
    if Language=="de":
        return"Netherlands"
    if Language=="sr":
        return "Russia"
    if Language=='vi':
        return "Vietnam"
    if Language=='sm':
        return "Somalia"
    if Language=="sr":
        return "Serbia"
    if Language=="ie":
        return "Indonesia"
    if Language=="mk":
        return "Ireland"
    if Language=="iv":
        return "Holy See (Vatican City State)"
    if Language=="af":
        return "South Africa"
    return "None"
df["country"]=df["Language"].progress_apply(get_country)


# **World plot of non-English languages**

# In[ ]:



fig = px.choropleth(df.query("Language != 'en'& Language != 'un' & country != 'None'").query("Count >= 5"), locations="country", hover_name="country",projection="natural earth", locationmode="country names", title="Countries of non-English languages", color="Count",template="plotly", color_continuous_scale="agsunset")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()


# from the world plot above .we can see that western Europe and the middle-east are the most represented regions in the dataset .Africa,Asia and eastern Europe are relatively under-represented.
# 

# **Non-English European**

# In[ ]:


fig = px.choropleth(df.query("Language != 'en' & Language != 'un' & country != 'None'"), locations="country", hover_name="country", projection="natural earth", locationmode="country names", title="Non-English European countries", color="Count",template="plotly", color_continuous_scale="aggrnyl", scope="europe")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()


# We can see that German and English are the most common European languages to feature in the dataset,although Spanish and Greek are not far behind.

# In[ ]:


fig = px.choropleth(df.query("Language != 'en' & Language != 'un' "), locations="country", hover_name="country",
                     projection="natural earth", locationmode="country names", title="Asian countries", color="Count",
                     template="plotly", color_continuous_scale="spectral", scope="asia")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()


# This plot shows that middle-eastern languages such as Arabic and Persian are represented more than languages from the Indian Subcontinent or south-east Asia,such as Hindi,Vietnamese and Indonesian.There is not a single Comment In amndarin,Korean or Japanese!

# In[ ]:


fig = px.choropleth(df.query("Language != 'English' & Language != 'un' & country != 'None'").query("Count >= 5"), locations="country", hover_name="country",
                     projection="natural earth", locationmode="country names", title="African countries", color="Count",
                     template="plotly", color_continuous_scale="agsunset", scope="africa")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()


# We can see that Africa is not as well as other continents in the dataset.The two most common African languages in the dataset are Afrikans and Somali

# # Feature engineering

# I've broadly classified my feature engineering ideas into the following three groups.

# **Direct Features**
# 
# features which are a directly due to word/content .We would exploring the following techniques.
# * Word frequency features
#     1. Count features
#     1. Bigrams
#     1. Trigrams
# * Vector distance mapping of words(Word2Vec)
# * Sentiment scores
# 

# **Indirect features**

# Some more experimental features.
# * count of sentences
# * count of words
# * count of unique words
# * count of letters
# * count of punctuations
# * count of uppercase words/letters
# * count of stop words
# * Avg length of each word
# 
# 

# **Leaky features**
# from the example ,we know that the comments contain identifier information (eg:IP,username etc.).We can create features out of them but ,it will certainly lead to overfitting to this specific wikipedia use-case.
# * toxic IP scores
# * toxic users
# **Note:**Creating the indirect and leaky features first .There are two reasons for this,
# * Count features(Direct features)are usefu; only if they are created from a clean corpus
# * Also the indirect features help compensate for the loss of information when cleaning the dataset.

# In[ ]:


merge=pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
df=merge.reset_index(drop=True)


# In[ ]:


## Indirect features
# sentence count in each comment:
#'\n 'can be used to count the number of sentences in each comment
df['count_sent']=df["comment_text"].apply(lambda x:len(re.findall("\n",str(x)))+1)
#word count in each comment
df['count_word']=df["comment_text"].apply(lambda x:len(str(x).split()))
#Unique word count
df["count_unique_word"]=df["comment_text"].apply(lambda x:len(set(str(x).split())))
#Letter count
df['count_letters']=df["comment_text"].apply(lambda x:len(str(x)))
#punctuation count


# In[ ]:


#upper case words count

df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

df["count_word_title"]=df["comment_text"].apply(lambda x:len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
df["count_stopwords"]=df["comment_text"].apply(lambda x:len([w for w in str(x).lower().split() if w in stopword])) 
#Average length of the words
df["mean_word_len"]=df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[ ]:


#derived features
#Word count percent in each comment:
df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
#derived features


# In[ ]:


#serperate train and test features
train_feats=df.iloc[0:len(train),]
test_feats=df.iloc[len(train):,]
#join the tags
train_tags=train.iloc[:,2:]
train_feats=pd.concat([train_feats,train_tags],axis=1)


# In[ ]:


train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10 
plt.figure(figsize=(12,6))
## sentenses
plt.subplot(121)
plt.suptitle("Are longer comments more toxic?",fontsize=20)
sns.violinplot(y='count_sent',x='clean', data=train_feats,split=True)
plt.xlabel('Clean?', fontsize=12)
plt.ylabel('# of sentences', fontsize=12)
plt.title("Number of sentences in each comment", fontsize=15)
# words
train_feats['count_word'].loc[train_feats['count_word']>200] = 200
plt.subplot(122)
sns.violinplot(y='count_word',x='clean', data=train_feats,split=True,inner="quart")
plt.xlabel('Clean?', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()


# Long sentences or more words do not seem to be a significant indicator of toxicity.
# 
# Chart desc:Violin plot is an alternative to the traditional box plot.The inner markings show the percentiles whi;e the width of the "violin" shows the volume of comments at that level

# In[ ]:


train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200
#prep for split violin plots
#For the desired plots , the data must be in long format
temp_df = pd.melt(train_feats, value_vars=['count_word', 'count_unique_word'], id_vars='clean')
#spammers - comments with less than 40% unique words
spammers=train_feats[train_feats['word_unique_percent']<30]


# In[ ]:


import matplotlib.gridspec as gridspec 
plt.figure(figsize=(16,12))
plt.suptitle("What's so unique ?",fontsize=20)
gridspec.GridSpec(2,2)
plt.subplot2grid((2,2),(0,0))
sns.violinplot(x='variable', y='value', hue='clean', data=temp_df,split=True,inner='quartile')
plt.title("Absolute wordcount and unique words count")
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.subplot2grid((2,2),(0,1))
plt.title("Percentage of unique words of total words in comment")
#sns.boxplot(x='clean', y='word_unique_percent', data=train_feats)
ax=sns.kdeplot(train_feats[train_feats.clean == 0].word_unique_percent, label="Bad",shade=True,color='r')
ax=sns.kdeplot(train_feats[train_feats.clean == 1].word_unique_percent, label="Clean")
plt.legend()
plt.ylabel('Number of occurances', fontsize=12)
plt.xlabel('Percent unique words', fontsize=12)



# **Corpus cleaning**

# Its important to use a clean dataset before creating count features.

# In[ ]:


corpus=merge.comment_text


# In[ ]:


#https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}


# In[ ]:


def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    
    
    comment=re.sub("\\n","",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    # (')aphostophe  replacement (ie)   you're --> you are  
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)


# In[ ]:


corpus.iloc[12235]


# In[ ]:


clean(corpus.iloc[12235])


# # Building Model

# In[ ]:


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


# In[ ]:


def regular_encode(texts,tokenizer,maxlen=512):
    enc_di=tokenizer.batch_encode_plus(texts,return_attention_masks=False,return_token_type_ids=False,pad_to_max_length=True,max_langth=maxlen)
    return np.array(enc_di['input_ids'])


# In[ ]:


def build_model(transformer, loss='binary_crossentropy', max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    x = tf.keras.layers.Dropout(0.3)(cls_token)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss=loss, metrics=[tf.keras.metrics.AUC()])
    
    return model


# # TPU Configs

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:



import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors


# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE



# Configuration
EPOCHS = 2
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
MODEL = 'jplu/tf-xlm-roberta-large'
#GCS_DS_PATH = KaggleDatasets().get_gcs_path('kaggle/input/')


# **Create fast tokenizer**

# In[ ]:


# First load the real tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# In[ ]:


train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train2.toxic=train2.toxic.round().astype(int)
valid = pd.read_csv('/kaggle/input/val-en-df/validation_en.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')


# In[ ]:


# Combine train1 with a subset of train2
train = pd.concat([
    train[['comment_text', 'toxic']],
    train2[['comment_text', 'toxic']].query('toxic==1'),
    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)
])


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nx_train = regular_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)\nx_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)\nx_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)\n\ny_train = train.toxic.values\ny_valid = valid.toxic.values')


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)
test_dataset = [(
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


# ****

# In[ ]:


get_ipython().run_cell_magic('time', '', "with strategy.scope():\n    transformer_layer = transformers.TFBertModel.from_pretrained('bert-base-uncased')\n    model = build_model(transformer_layer, loss=focal_loss(gamma=1.5), max_len=512)\nmodel.summary()")


# **Train Model**

# First,we train in the subset of taining set,which is completely in English.

# In[ ]:


n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


# Now that we have pretty much saturated the learning potential of the model on english only data,we train it one more epoch on the validation set,which is significantly smaller but contains a mixture of diffferent languages.
# 

# In[ ]:


n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS
)


# And finally, create the submission file.

# In[ ]:


sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)


# In[ ]:


sub.head()


# * # Please UPVOTE me and comments for any suggestion regarding this notebook

# In[ ]:




