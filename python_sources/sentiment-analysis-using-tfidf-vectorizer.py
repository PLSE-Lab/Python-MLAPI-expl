#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
#             This notebook is on classification on the basis of sentiment analysis based on imdb users reviews. Its tottaly at beginner 
#             level.tf-idf vectorizer is used for the transformation of text to numbers and logistic regression model is used for the
#             classification.

# **Loading the all required models**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #interact withe file sysytem
import re #used for data cleaning
import nltk #used for data cleaing
from nltk.stem.porter import PorterStemmer #used for finding the root word of a word 
from nltk.corpus import stopwords #used fro removing the stop words
from bs4 import BeautifulSoup #used for removing the html tags in the text
from nltk.tokenize import TreebankWordTokenizer#it t
from sklearn.linear_model import LogisticRegression #for classification
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report #for testing the model accuracy


# **Importing the datasets**

# importing the train dataset first

# In[ ]:


#importing all the negetive file first
train_neg=os.listdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/neg')#listing all the filenames present in the directory
os.chdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/neg')#setting the directory path
train_set=[]
for i in train_neg:
    fp=open(i,"r",encoding="utf8")
    train_set.append(fp.read())
    fp.close()


# In[ ]:


#now importing the all positive files
train_pos=os.listdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/pos')
os.chdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/pos')
for i in train_pos:
    fp=open(i,"r",encoding="utf8")
    train_set.append(fp.read())
    fp.close()
train_set[12500]


# In[ ]:


len(train_set)


# now we have a list of negetive and positive reviews.in the list negetive reviews comes first followed by positive reviews.now we will convert this list to a dataframe and which will contain 2 columns Review and rating column.so lets make the datframe. 

# In[ ]:


train_set=pd.DataFrame(train_set,columns=["Reviews"])#converting train_set from list to a datframe
train_set["Rating"]=np.zeros([len(train_set),1],dtype=int)#adding rating column for all the rows to zero then we will change it to 1 for the positive reviews
train_set.loc[12500:25000,"Rating"]=1#changing the rating value to 1 for positive reviews
y=train_set.loc[:,"Rating"]


# now lets check first 5 of each negetive and positive reviews and their ratings

# In[ ]:


train_set[0:5]


# importing the test files

# In[ ]:


train_set[12500:12505]


# now we have preapred our train dataset successfully now lets do the same operations for the test set

# In[ ]:


test_neg=os.listdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/neg')
test_pos=os.listdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/pos')


# In[ ]:


os.chdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/neg')
test_set=[]
for i in test_neg:
    fp=open(i,"r",encoding="utf8")
    test_set.append(fp.read())
    fp.close()
test_set[0]


# In[ ]:


os.chdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/pos')
for i in test_pos:
    fp=open(i,"r",encoding="utf8")
    test_set.append(fp.read())
    fp.close()
test_set[12500]


# In[ ]:


test_set=pd.DataFrame(test_set,columns=["Reviews"])#converting test_set from list to a datframe
test_set["Rating"]=np.zeros([len(test_set),1],dtype=int)#adding rating column for all the rows to zero then we will change it to 1 for the positive reviews
test_set.loc[12500:25000,"Rating"]=1#changing the rating value to 1 for positive reviews
y_test=test_set.loc[ : ,"Rating"]


# In[ ]:


test_set[0:5]


# In[ ]:


test_set[12500:12505]


# Now we have prepared the train and test datasets lets go for the data cleaning process.

# **DATA CLEANING**
# In sentiment analysis data cleaning means text cleaning.because the texts we have contain different articles,pontuation marks,unnecessery spaces,and html tags along with the sentiment words,for the sentiment ananlysis we only need sentiment words not other words.training a model without cleaning the text is a bad practice.so we will remove those words from our text and will prepare our corpse whcih will contain only sentiment words.and we will also stem similar type of words for example enjoying and enjoy they refer to a same sentiment so we dont need that extra ing term.And one thing ,while we are removing pontuation marks there may be a chance we can lose a meaning of  word by doing so for example don't if we will tokenize it the it will become don and t and both of the words nake no sense do we will use the treebankwordtokenizer so that it will tokenize and the words wont lose their meaning.so lets start the cleaning process.

# In[ ]:


train_corpus=[]
tokenizer=TreebankWordTokenizer()
for i in range(0,len(train_set)):
    soup=BeautifulSoup(train_set["Reviews"][i],"html.parser")
    review=soup.get_text()#removes all html tags
    review.lower()#converting all words to lower case
    review=tokenizer.tokenize(review)#tokenizing using trebankwordtokenizer
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review]#stemming the words
    review=' '.join(review)
    train_corpus.append(review)
train_corpus[0:10]


# as you can see we have made the corpus for the train set,there is no html tags ,all the words are lower case and stemmed but the articles and pontuation marks are still there .Tf-Idf vectorier will take care of this.

# now lets preapre the corpus for test set

# In[ ]:


test_corpus=[]
for i in range(0,len(test_set)):
    soup=BeautifulSoup(test_set["Reviews"][i],"html.parser")
    review=soup.get_text()#removes all html tags
    review.lower()#converting all words to lower case
    review=tokenizer.tokenize(review)#tokenizing using trebankwordtokenizer
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review]#stemming the words
    review=' '.join(review)
    test_corpus.append(review)


# In[ ]:


test_corpus[0:5]


# **** PREPARING THE VECTORIZER****
# now we have preapred the corpuses lets prepare our vectorizer.tfidf vectorizer will remove the words which will appear morethan 50 percent of the documents and which will appear less than 5 documents.And we have considered 1gram and 2gram words for better accuracy.more accuracy can be achieved by considering bigger ngrams.

# > from sklearn.feature_extraction.text import TfidfVectorizer
# 
# > tfidf=TfidfVectorizer(min_df=5,max_df=0.5,ngram_range=(1,2))
# 
# > x=tfidf.fit_transform(train_corpus).toarray()
# 
# > x.shape()

# ![image.png](attachment:image.png)

# now lets fit the vectorizer to the test_set

# > from sklearn.feature_extraction.text import TfidfVectorizer
# 
# > tfidf=TfidfVectorizer(min_df=5,max_df=0.5,ngram_range=(1,2))
# 
# > x_test=tfidf.fit(train_corpus).toarray()
# 
# > x_test.shape()

# **TRAINING THE MODEL:**
training the model with trainset
# > logi_classifier = LogisticRegression(random_state = 0)
# 
# > logi_classifier.fit(x, y)
# 

# prdicting the vectorizer with testset

# > y_pred_logi = logi_classifier.predict(x_test)
# 
# > y_pred_logi
# 

# ![image.png](attachment:image.png)

# ****Accuracy****

# as our dataset is equally devided,means we have equal number of positive and negetive records we will consider the. we will caluclate the confusion matrix using confusion_matrix model and accuracy_score model to get the accuracy score for the model and accuracy_report model for the accuracy report.

# > cm_logi = confusion_matrix(y_test, y_pred_logi)
# 
# > cm_logi
# 

#  ![image.png](attachment:image.png)

# > acc_logi=accuracy_score(y_test, y_pred_logi)
# 
# > acc_logi
# 

# ![image.png](attachment:image.png)

# > a=classification_report(y_test, y_pred_logi)
# 
# > print(a)

# ![image.png](attachment:image.png)

# **CONCLUSION**

# logistic regression model produces 89.29 percentage of accuracy which is acceptable .we can achive higher accuracy using random forest model. we can achieve upto 93 percentage of accuracy using random forest but it will take a large amount of memory which may not be possible in low end systems.We used tfidf vectorizer model we can use other models.
