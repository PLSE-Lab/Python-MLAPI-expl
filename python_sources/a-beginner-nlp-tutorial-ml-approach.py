#!/usr/bin/env python
# coding: utf-8

# # NLP: Real or Not? NLP with Disaster Tweets
# ## AI Open Sessions 
# 
# 

# ### Our goal is to build a model able to predict if a tweet really concern a disaster or it does not

# ### 1) Load the librairies and the dataset
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### 2) Load the dataset from Kaggle

# In[ ]:


df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')


# Let's have a look at our data

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.target.value_counts()


# Some investigation on 'keyword' and 'location'

# In[ ]:


print('Number of unique keywords : ', df.keyword.nunique())


# In[ ]:


df.keyword.value_counts()


# In[ ]:


print('Number of unique locations :' , df.location.nunique())


# #### Dealing with Categorical missing values : 
# 
# 
# 1.   Delete whole column
# 2.   Delete rows with missing values
# 3.   Replace with the most frequent values
# 4.   Apply a Classification Algorithm (Supervised)
# 5.   Apply a Clustering Algorithm  (Unsupervised)
# 

# We will go with the lazy method.<br>
# We will delete the 'location' column, an delete rows with null keyword value

# In[ ]:


df.drop('location',axis = 1 , inplace=  True)
df.dropna(inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# A good habit, is to reset index after dropping some rows

# In[ ]:


df.reset_index(drop=True, inplace=True)


# Check the Target column

# In[ ]:


sns.countplot(data = df, x='target')


# ### 3) NLP Preprocessing

# The NLP wokflow is different, we will go step by step here <br>
# We will define functions that we can use later

# First we will import libraries that we will need

# In[ ]:


import nltk
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from wordcloud import WordCloud


# In[ ]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# #### 1) Change all the text to lower case

# In[ ]:


def Lower (text):
  return text.lower()


# #### 2) Word Tokenisation : Basically, tokenisation is the process of breaking a stream of text up into words , for example : "This is a text" => 'This' , 'is' , 'a' ,'text'

# In[ ]:


def Tokenisation (text):
  return nltk.word_tokenize(text)


# In[ ]:


# Let's test it :
test = Tokenisation('Hello there. How! are you ? this super notebook is about nlp')


# #### 3) Remove Stop Words and Non Alpha Text : Nltk provides as with some stop words so here we define a function that returns as a list with those words and othersfrom our business understanding

# In[ ]:


#Create a list of stopwords 
Stpwrd_List=stopwords.words('english')


# In[ ]:


def StopWordsAlphaText(tokenized_text):
  filtred_text=[]
  for word in tokenized_text:
  #strip punctuation
    word = word.strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    #check if the word starts with an alphabet
    val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
    #ignore if it is a stop word or val is none
    if((word not in Stpwrd_List) and (val is not None)):
      filtred_text.append(word)
  return filtred_text


# In[ ]:


StopWordsAlphaText(test)


# ### 4) a-  Word Lemmatization : it's the process during wich we reduce the inflected words properly ensuring that the root word belongs to the language. <br> A little explanation : we use somethong called "pos_tagger", this tool is used to identify the nature of the word given. We do this process to provide the word's nature to the lemmatizer to obtain the best result possible
# 

# In[ ]:


from nltk.corpus import wordnet


# In[ ]:


tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}


# In[ ]:


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return tag_dict.get(tag, wordnet.NOUN)


# In[ ]:


lemmatizer = WordNetLemmatizer()


# In[ ]:


def Lemmetizer(tokens):
  lemmetized_text=[]
  for word in tokens:
    word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    lemmetized_text.append(word)
  return lemmetized_text


# ### 4) b- Stemming is the process of reducing inflected words to their word stem, base or root form

# In[ ]:


PosStem = PorterStemmer ()


# In[ ]:


def Stemmer (tokens):
  stemmed_text=[]
  for word in tokens :
    word = PosStem.stem(word)
    stemmed_text.append(word)
  return stemmed_text


# 

# In[ ]:


df.text = df.text.apply(Lower)


# In[ ]:


df.text = df.text.apply(Tokenisation)


# In[ ]:


df.text = df.text.apply(StopWordsAlphaText)


# In[ ]:


df.text = df.text.apply(Lemmetizer)


# In[ ]:


df.head()


# ### Wordcloud and word count

# In[ ]:


df.head()


# In[ ]:


real=""
fake=""
for index,row in df.iterrows():
  text = " ".join(row["text"])
  if(row["target"]==1):
    real=real+" "+text
  else:
    fake=fake+" "+text


# In[ ]:


#Create a real_tweets wordcloud
wordcloud_real=WordCloud(max_font_size=100, max_words=100, background_color="white").generate(real)
#Create a real_tweets wordcloud
wordcloud_fake=WordCloud(max_font_size=100, max_words=100, background_color="white").generate(fake)


# In[ ]:


plt.figure(figsize=(15,15))
plt.imshow(wordcloud_real, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud of positive Reviews")
plt.show()


# In[ ]:


plt.figure(figsize=(15,15))
plt.imshow(wordcloud_fake, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud of positive Reviews")
plt.show()


# The next steps need a string format (not a list), that is why, we have to join all the words in a single sentence 

# In[ ]:


df.text = df.text.apply(lambda x: " ".join(x))


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# #### First seperate features from target, then train/test split

# In[ ]:


X = df.drop(['target','id'],axis = 1)
y = df.target


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)


# #### We need to take care of the categorical data, so we will use an ordinal encode

# #### We will try CountVectorizer() provided by SKlearn to get our document-term matrix (n_samples,n_features) 

# <img src="https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/vectorchart.PNG" alt="drawing" width="600"/>

# In[ ]:


count_vect = CountVectorizer()


# In[ ]:


#Let's see what it does exactly
Example = [ 'this is the first line.',
           'this line is the second line.',
           'and this is the third.',
           'is this the fifth line?']


# In[ ]:


Vect_Cv = count_vect.fit_transform(Example)


# In[ ]:


count_vect.get_feature_names()


# In[ ]:


Vect_Cv


# In[ ]:


Vect_Cv.toarray()


# ### Term ferquency - Inverse document frequency : <br> Reduce the weightage of more common words like (the, is, an etc.) which occur in all document. <br> Wiki : TF_IDF is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus

# <img src="https://miro.medium.com/max/3604/1*qQgnyPLDIkUmeZKN2_ZWbQ.png" alt="drawing" width="500"/><img src="https://miro.medium.com/max/876/1*_OsV8gO2cjy9qcFhrtCdiw.jpeg" alt="drawing" width="455"/>
# 

# In[ ]:


Tfidf = TfidfTransformer()


# In[ ]:


Vect_Tfidf = Tfidf.fit_transform(Vect_Cv)


# In[ ]:


Vect_Tfidf


# In[ ]:


Vect_Tfidf.toarray()


# #### These two functions are already combined in another one called 'TfidfVectorizer'

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfvect = TfidfVectorizer()
tfvect.fit_transform(Example).toarray()


# ### Now to the pipeline

# First we create a list with sperated feature types

# We create the text feature pipeline

# In[ ]:


text_transformer = Pipeline([
    ('CountVectorizer', CountVectorizer()),
    ('TfidfTransformer',TfidfTransformer())
          ])


# In[ ]:


Preprocessing = ColumnTransformer([
    ("text features", text_transformer, 'text'),
    ('categorical_features', OrdinalEncoder(),['keyword'])
                                   ])


# In[ ]:


X_train = Preprocessing.fit_transform(x_train)


# In case we have many features, we can put them on a list like this : <br>
# 
# 
# ```
# Numerical_features = ['num1','num2']
# categorical_features = ['cat1','cat2']
# ```
# 
# 

# ### Choosing our model
# ![Image](https://miro.medium.com/max/1400/1*tpOZa-wju9pD-5vFd-Es-g.png)

# ### A quick explanation about the Bayes Theorem.<br>
# #### Bayes theorem is a famous equation that allows us to make predictions based on data. Here is the classic version of the Bayes theorem:
# ![image.png](data:image/png;base64,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)
# <br>
# so, if we want to apply it to our data, it will be in this form : 
# ![image.png](data:image/png;base64,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)

# For the varieties that we will use, we have : 
# 
# *   **Gaussian Naive Bayes :** It uses the previous equation . Because of the assumption of the normal distribution, Gaussian Naive Bayes is best used in cases when all our features are continuous.
# *   **Bernoulli Naive Bayes :** The Bernoulli naive Bayes classifier assumes that all our features are binary such that they take only two values
# *   **Multinomial Naive Bayes :** Multinomial naive Bayes works similar to Gaussian naive Bayes, however the features are assumed to be multinomially distributed. In practice, this means that this classifier is commonly used when we have discrete data 
# 
# 
# 
# 

# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import time


# In[ ]:


names = ["Gaussian Naive Bayes", "Multinomial Naive Bayes","Bernoulli Naive Bayes" , "Random Forest Classifier"]


# In[ ]:


classifiers = [GaussianNB(),
               MultinomialNB(),
               BernoulliNB(),
               RandomForestClassifier()
               ]


# In[ ]:


for name, clf in zip(names, classifiers):
  #Cross validation prediction, and we measure fitting time 
  start = time.time()
  preds = cross_val_predict(clf,X_train.toarray(),y_train,cv=3)
  end = time.time()
  #Metrics
  acc = accuracy_score(y_train,preds)
  precision = precision_score(y_train,preds)
  recall = recall_score(y_train,preds)
  f1 = f1_score(y_train,preds)
  cm = confusion_matrix(y_train,preds)
  #Printing results
  print (name, 'Accuracy  :  ', "%.2f" %(acc*100),'%', ', Precision',"%.3f" %precision, 'Recall :' , "%.3f" %recall ,'F1-Score : ',"%.3f" %f1)
  print('The confusion Matrix : ' )
  print(cm)
  #Now we check how long did it take
  print('Time used :', "%.3f" %(end - start), 'seconds')
  print(' *-----------------------------------------------------------------------------------------------------*')


# As you can see, Random Forest takes a kot of time, that is way we use NaiveBayes based Algorithms

# Let's move to a gridsearch, to look for optimals parameters

# In[ ]:


from sklearn.model_selection import GridSearchCV


# Now we define our parameters that we want to test. <br>
# you ca always check [Sklearn website](https://scikit-learn.org/) to see the model parameters

# In[ ]:


Grid_par = [
            {'alpha' : [0,0.5,1,1.5], 'fit_prior' : [True, False]}
            ]
model = BernoulliNB()


# In[ ]:


GridSearch = GridSearchCV(estimator= model , param_grid=Grid_par, cv = 5,
                         scoring='accuracy', return_train_score=True)


# In[ ]:


GridSearch.fit(X_train,y_train)


# In[ ]:


results = GridSearch.cv_results_


# In[ ]:


for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print ("%.3f" %(mean_score*100),'% | Parameters : ',params) 


# In[ ]:


GridSearch.best_estimator_


# In[ ]:


final_model = GridSearch.best_estimator_


# Create our final pipeline, including the model

# In[ ]:


Final_Pipeline = Pipeline([
     ('Preprocessing', Preprocessing),
     ('clf', final_model)
])


# In[ ]:


Final_Pipeline.fit(x_train,y_train)


# ### Now we move to the testing phase
# #### We can directly apply our pipeline, to predict the test dataset

# In[ ]:


Preds = Final_Pipeline.predict(x_test)


# In the end, we measure our accuracy

# In[ ]:


print('Final model accuracy : ', "%.3f" %(accuracy_score(y_test,Preds)*100), '%')

