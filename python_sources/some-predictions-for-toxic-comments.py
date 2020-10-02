#!/usr/bin/env python
# coding: utf-8

# ### Welcome to my 2nd kernel for Toxic Comments Classification Challenge
# 
# Here I will test and benchmark some algorithm used in Natural Language Processing. Namely :
# 1. Logistic Regression
# 2. Naive Bayes Algorithm
# 3. Long Short Term Memory Neural networks
# 
# My approach will be " KIS" : Keep It Simple, as long as I can. I will make no assumption when testing algorithm in the beginning and then analyze the results before new test
# 
# ![](http://www.elpoderdelasideas.com/wp-content/uploads/google-jigsaw-2016.png)

# In[101]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer

#NLP tools
import re
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
stopwords = nltk.corpus.stopwords.words('english')

# plot tools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")

path="../input/"

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[84]:


train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")


# In[85]:


replacement_patterns = [
 (r'won\'t', 'will not'),
 (r'can\'t', 'cannot'),
 (r'i\'m', 'i am'),
 (r'ain\'t', 'is not'),
 (r'(\w+)\'ll', '\g<1> will'),
 (r'(\w+)n\'t', '\g<1> not'),
 (r'(\w+)\'ve', '\g<1> have'),
 (r'(\w+)\'s', '\g<1> is'),
 (r'(\w+)\'re', '\g<1> are'),
 (r'(\w+)\'d', '\g<1> would')
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
         self.patterns = [(re.compile(regex), repl) for (regex, repl) in
         patterns]
     
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
             s = re.sub(pattern, repl, s)
        return s


# In[86]:


from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
from nltk.tokenize import TweetTokenizer
#from replacers import RegexpReplacer
replacer = RegexpReplacer()
tokenizer=TweetTokenizer()


def comment_process(comment):
        comment=tokenizer.tokenize(replacer.replace(comment))
        comment= [word for word in comment if ( word.lower() not in stopwords 
                              and word.lower() not in list(string.punctuation) )]
        comment=[lemmer.lemmatize(word, 'v') for word in comment]
        comment.extend(list(comment))
        comment=" ".join(comment)
        return comment
    

cleaned_train=train.comment_text.apply(comment_process)
#cleaned_test=test.comment_text.apply(comment_process)



# In[87]:


tf = TfidfVectorizer( strip_accents='unicode',analyzer='word', max_features= 50000, ngram_range=(4,4),
            use_idf=True,smooth_idf=True,sublinear_tf=True,
            stop_words = 'english')


# In[88]:


cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y=train[cols]

xtrain, xvalid, ytrain, yvalid = train_test_split(cleaned_train, y, 
                                                  random_state=42, 
                                                  test_size=0.3, shuffle=True)

xtraintf=tf.fit_transform(xtrain)
xvalidtf=tf.fit_transform(xvalid)

#xtest=tf.transform(cleaned_test)





# ### Logistic Regression

# In[89]:


prd_valid = np.zeros((xvalidtf.shape[0],yvalid.shape[1]))
prd_train = np.zeros((xtraintf.shape[0],ytrain.shape[1]))
train_loss = []
valid_loss = []
bnb = LogisticRegression(penalty='l2')
for i,col in enumerate(cols):
    print('Building {} model for column:{''}'.format(i,col)) 
    bnb.fit(xtraintf,ytrain[col])
    prd_valid[:,i] = bnb.predict_proba(xvalidtf)[:,1]
    prd_train[:,i] = bnb.predict_proba(xtraintf)[:,1]
    train_loss_class=log_loss(ytrain[col],prd_train[:,i])
    valid_loss_class=log_loss(yvalid[col],prd_valid[:,i])
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))


# .143 mean column-wise log loss... No so bad for this simple model.
# ## Let's try Naive Bayes Algorithm and compare

# In[90]:


prd_valid = np.zeros((xvalidtf.shape[0],yvalid.shape[1]))
prd_train = np.zeros((xtraintf.shape[0],ytrain.shape[1]))
train_loss = []
valid_loss = []
bnb = BernoulliNB()
for i,col in enumerate(cols):
    print('Building {} model for column:{''}'.format(i,col)) 
    bnb.fit(xtraintf,ytrain[col])
    prd_valid[:,i] = bnb.predict_proba(xvalidtf)[:,1]
    prd_train[:,i] = bnb.predict_proba(xtraintf)[:,1]
    train_loss_class=log_loss(ytrain[col],prd_train[:,i])
    valid_loss_class=log_loss(yvalid[col],prd_valid[:,i])
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))


# 10.23 mean column-wise log loss! Much worse than logistic regression. 
# ### My guess:
# My guess is that the Naive assumption of the Naive Bayes Algorithm is not valid. This assumption is that the features ( the words) are independent with each other. And obviously it's not valid here. But it would be "less unvalid" if the algorithm 2 or 3 words ( or more) instead of just one.  
# This picture illustrates the concept of n_grams. Instead of considering only one word we can consider every pair of words or every tree words. 
# ![](https://i.stack.imgur.com/8ARA1.png)
# 
# Of course it will change the TF-IDF scores. The score will now be calculated for 2grams, 3grams... Ngrams.

# In[91]:


def test_model(model,xtraintf, xvalidtf, ytrain, yvalid ):    
    prd_valid = np.zeros((xvalidtf.shape[0],yvalid.shape[1]))
    prd_train = np.zeros((xtraintf.shape[0],ytrain.shape[1]))
    train_loss = []
    valid_loss = []
    
    if model=="lr":
        model= LogisticRegression(penalty="l2")
    if model=="nb":
        model=BernoulliNB()
    for i,col in enumerate(cols):
        model.fit(xtraintf,ytrain[col])
        
        prd_valid[:,i] = model.predict_proba(xvalidtf)[:,1]
        prd_train[:,i] = model.predict_proba(xtraintf)[:,1]
        
        train_loss_class=log_loss(ytrain[col],prd_train[:,i])
        valid_loss_class=log_loss(yvalid[col],prd_valid[:,i])
        
        train_loss.append(train_loss_class)
        valid_loss.append(valid_loss_class)
    return(np.mean(train_loss), np.mean(valid_loss))


    


# In[93]:


train_lr, valid_lr=[],[]
train_nb, valid_nb=[],[]
ngram_list=[x for x in range (1,6)]

for ngram in ngram_list:
    tf = TfidfVectorizer( strip_accents='unicode',analyzer='word', 
                         max_features= 50000, ngram_range=(ngram,ngram),
            use_idf=True,smooth_idf=True,sublinear_tf=True,
            stop_words = 'english')
    xtraintf=tf.fit_transform(xtrain)
    xvalidtf=tf.fit_transform(xvalid)
    print("testing logistic regression with "+ str(ngram)+"grams")
    score_lr=test_model('lr', xtraintf, xvalidtf, ytrain, yvalid )
    train_lr.append(score_lr[0])
    valid_lr.append(score_lr[1])
    
    print("testing naive bayes with "+ str(ngram)+"grams")
    score_nb=test_model('nb', xtraintf, xvalidtf, ytrain, yvalid )
    train_nb.append(score_nb[0])
    valid_nb.append(score_nb[1])
    


# In[104]:


plt.figure(figsize=(16,12))
plt.suptitle("Ngrams comparison",fontsize=20)

plt.subplot2grid((2,1),(0,0))
plt.title("Logistic Regression")
plt.plot(ngram_list, train_lr,'xkcd:crimson', label='train', linewidth=3 )
plt.plot(ngram_list, valid_lr, 'xkcd:azure',label='validation', linewidth=3)
plt.legend(fontsize=14)
plt.ylabel('mean column-wise log loss', fontsize=20)
plt.xlabel('Ngrams', fontsize=20)

plt.subplot2grid((2,1),(1,0))
plt.title("Naive Bayes")
plt.plot(ngram_list, train_nb, 'xkcd:crimson',label='train', linewidth=3 )
plt.plot(ngram_list, valid_nb,'xkcd:azure',label='validation', linewidth=3)
plt.legend(fontsize=14)
plt.ylabel('mean column-wise log loss', fontsize=20)
plt.xlabel('Ngrams', fontsize=20)


# ## As we can see, the 2 models react differently to Ngrams size
# 
# * The Naive Bayes Algorithm seems to give better results with bigger Ngrams. It validate smy intuition about the non-valid independency hypothesis of this algorithm. Again, the bigger the Ngrams, the less "unvalid" this hypothesis is.
# * Logistic regression's validation score does not seem to depend on the ngram size
# 
# * The increasing training loss show that bigger ngrams makes the model underfit. Indeed, bigger ngrams makes the model less complex. The bigger the ngrams, the lesser ngrams the model will learn from ==> the lesser complex our model will be.
# 
# ## Next step :
# * keep on analyzing these results
# * trying long short term memory neural network.
# I know I could try SVM or Decision trees before LSTM because LSTM  are much more complex. But I have more knowledge on LSTM and I can't wait to try them ahah !
