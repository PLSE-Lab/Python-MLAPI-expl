#!/usr/bin/env python
# coding: utf-8

# ![](https://m.popkey.co/ddb315/LmGYG_s-200x150.gif?c=popkey-web&p=funny_or_die&i=funny-or-die-ent&l=search&f=.gif)
# If you use parts of this notebook in your own scripts, please give some sort of credit (for example link back to this). Thanks!
# 
# **Please upvote to encourage me to do more.**
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# In[ ]:


datadir = '../input/'
train = pd.read_csv(datadir+'train.csv')
train.head()


# In[ ]:


datadir = '../input/'
test = pd.read_csv(datadir+'test.csv')
test.head()


# Now we will examine the train dataset

# In[ ]:


train.tail()


# In[ ]:


train.shape


# In[ ]:


train['comment_text'][0]


# It is multi class classification problem having 6 classes

# In[ ]:


train.columns[3:]
#these are the 6 classes which we need to decide to which class a message belongs to


# In[ ]:


train.head()


# If you can see in the above 6th message belongs to 4 classes, let us see what the message is

# In[ ]:


train['comment_text'][6]
#it is really bad,isnt it?


# Now we have to clean the text which we call preprocessing of data, so that we can know which words causing to determine particular class

# In[ ]:


train.comment_text.str.len().describe()
#here the minimum length of a message is 6 as you can see the below table


# if you remember some of the message dont belong either of 6 classes that means  all classes are marked as 0, now we have to find  how many messages are there.

# Now we'll create a seventh class and if a messge belongs to none of the 6 classes then it belongs to seventh class.

# In[ ]:


train['seventh'] = 1 - train[train.columns[2:]].max(axis =1)
train.head()


# In[ ]:


#we will see are there any null values 
train.isnull().any()


# In[ ]:


print('you can find null values in train set like this also')
print(train.isnull().sum())
print('for test set null values are')
print(test.isnull().sum())


# In[ ]:


# Here is the total number of samples belongs to each class
x = train.iloc[:,2:].sum()
print('total number of comment:',len(train),'\n','samples belongs to each class','\n',x)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x.index,x.values)
plt.xticks(rotation=90)
plt.title('class distribution')
plt.show()


# Do you remember there are some messages which belongs to multiple classes and as you can see in the above image classes are also not evenlt spread,that means class imbalance also there, let us check how one class is correlated with other class with the help of heapmaps
# 

# In[ ]:


y = train.corr()
plt.figure(figsize=(8,8))
sns.heatmap(y,annot=True,center=True,square=True)
plt.title('heatmap showing correlation between classes')
plt.show()
#Here i intentionally included seventh class which we created


# In[ ]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from scipy.sparse import hstack


# **Actually i thought to finish it directly without exploring whole content but, it will be much helpful to newbies who want to learn if i do EDA from basic level
# **

# In[ ]:


merge=pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
df=merge.reset_index(drop=True)
df.head()


# In[ ]:


print('train shape',train.shape,'\n','test shape',test.shape,'\n','df shape',df.shape,'\n')


# In[ ]:


import re
import string
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
# start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
import warnings
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Sentense count in each comment:
    #  '\n' can be used to count the number of sentences in each comment
df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
#Word count in each comment:
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
#Unique word count
df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
#Letter count
df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))
#punctuation count
df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#upper case words count
df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#title case words count
df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
#Average length of the words

df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[ ]:


df['ip'] = df['comment_text'].apply(lambda x: re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", str(x)))
df['count_ip']=df["ip"].apply(lambda x: len(x))


# I'll update more in coming future so watch this space.
# 
# # Please upvote to encourage me.
# 
# ![](https://media3.giphy.com/media/l4FGpdrSHIcgR7Mli/200.webp#5-grid1)
# 
# ## Thank you all :)
# 

# In[ ]:




