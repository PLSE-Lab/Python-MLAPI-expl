#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, we will be representing our documents as a TF-IDF vector , which is a way to model your text data into vectors , because all the libraries which you will generally use, tends to represent the data  into mathematical models. Then we will train a classifer(naiveBayes, SVM) on the dataset and then will evaluate our model. The tutorial will progress as follows:
# 
# 
# 1. **Preprocess your Data with Gensim** - We will be using gensim library along with nltk, to quickly and efficiently preprocess our data. 
# 
# 2. **Building the Model** - We will be training Naive Bayes classifier on our data
# 
# 3. **Train the Model** - We will be training Naive Bayes classifier on our data
# 
# 4. **Prediction**
# 
# 5. **Evaluating the Model** - We will be evaluating our model
# 
# **Note:** I won't be doing a lot of Exploratory Data analysis as there are already public kernels available. Please go through **Spooky NLP and Topic Modelling tutorial** for data visualization purpose

# In[2]:


# read in some helpful libraries
import nltk                       # the natural langauage toolkit, open-source NLP
import pandas as pd               # pandas dataframe
import re                         # regular expression
from nltk.corpus import stopwords  
from gensim import parsing        # Help in preprocessing the data, very efficiently
import gensim
import numpy as np

# Loading in the training data with Pandas
df_train = pd.read_csv("../input/train.csv")


# In[3]:


# look at the first few rows and how the text looks like
print (df_train['text'][2]) , '\n'
df_train.head()


# In[4]:


## check the dimensions of the table
print ("Shape:", df_train.shape, '\n')

## Check if there is any NULL values inside the dataset
print ("Null Value Statistics:", '\n \n', df_train.isnull().sum()) ## Sum will tell the total number of NULL values inside the dataset
print ('\n')

## Explore the data types of your dataset
print ("Data Type of All Columns:" '\n \n', df_train.dtypes)


# In[5]:


## Collect all unique author names from author column
author_names = df_train['author'].unique()
print (author_names)


# For performance reason it is good to convert the target variable(author name) in **some sort of coding format** 

# In[6]:


"""
MWS 2
EAP 0
HPL 1
""" 
authorname_to_id = {}
assign_id = 0
for name in author_names:
    authorname_to_id[name] = assign_id
    assign_id += 1  ## Get a new id for new author
    
##  Print the dictionary created
for key, values in authorname_to_id.items():
    print (key, values)


# In[7]:


## convert the author name to id --> So when we predict the result humans can understand
"""
0 EAP
1 HPL
2 MWS
""" 
id_to_author_name = {v: k for k, v in authorname_to_id.items()}
for key, values in id_to_author_name.items():
    print (key, values)


# In[8]:


## Add a new column to pandas dataframe, with the author name mapping
def get_author_id(author_name):
    return authorname_to_id[author_name]

df_train['author_id'] = df_train['author'].map(get_author_id)


# In[9]:


df_train.head()


# ## 1. Preprocessing the Data

# In[10]:


def transformText(text):
    
    stops = set(stopwords.words("english"))
    
    # Convert text to lower
    text = text.lower()
    # Removing non ASCII chars    
    text = re.sub(r'[^\x00-\x7f]',r' ',text)
    
    # Strip multiple whitespaces
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    
    # Removing all the stopwords
    filtered_words = [word for word in text.split() if word not in stops]
    
    # Removing all the tokens with lesser than 3 characters
    filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=3)
    
    # Preprocessed text after stop words removal
    text = " ".join(filtered_words)
    
    # Remove the punctuation
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    
    # Strip all the numerics
    text = gensim.parsing.preprocessing.strip_numeric(text)
    
    # Strip multiple whitespaces
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    
    # Stemming
    return gensim.parsing.preprocessing.stem_text(text)


# In[11]:


df_train['text'] = df_train['text'].map(transformText)


# In[12]:


## Print a couple of rows after the preprocessing of the data is done

print (df_train['text'][0] , '\n')
print (df_train['text'][1] , '\n')
print (df_train['text'][2])


# ### We will be dividing our training data into **(Train, Test)**. So that we can evaluate the model

# In[13]:


## Split the data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train['text'], df_train['author_id'], 
                                                    test_size=0.33, random_state=42)


# In[14]:


print ("Training Sample Size:", len(X_train), ' ', "Test Sample Size:" ,len(X_test))


# ## 2. Building the Model

# In[15]:


## Get the word vocabulary out of the data
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

## Count of 'mistak' in corpus (mistake -> mistak after stemming)
print ('mistak appears:', count_vect.vocabulary_.get(u'mistak') , 'in the corpus')


# In[16]:


## Get the TF-IDF vector representation of the data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print ('Dimension of TF-IDF vector :' , X_train_tfidf.shape)


# ## 3. Training a classifier

# In[17]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# ## 4. Prediction

# In[18]:


## Prediction part

X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)


# In[19]:


## predictions for first 10 test samples

counter  = 0
for doc, category in zip(X_test, predicted):
    print('%r => %s' % (doc, id_to_author_name[category]))
    if(counter == 10):
        break
    counter += 1    


# ## 5. Evaluation 
# 
# We will be doing simple evaluation scheme and will conclude  mean of the **correct predictions** as our accuracy

# In[20]:


np.mean(predicted == y_test) ## 80% sounds good only 


# # Where to go from Here:
# Here are my couple of ideas to try:
# 1. Use a better classifier (SVM)
# 2. Go with word2vec representation of the data rather than TF-IDF
# 
# **Note:** These are my possible ideas, feel free to comment and let me know the sections of the code which 
#                     can be improved.
