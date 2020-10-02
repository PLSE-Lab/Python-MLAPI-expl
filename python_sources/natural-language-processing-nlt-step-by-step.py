#!/usr/bin/env python
# coding: utf-8

# Created by: Sangwook Cheon
# 
# Date: Dec 24, 2018
# 
# This is step-by-step guide to Natural Language Processing (NLP) using scikit-learn, which I created for reference. I added some useful notes along the way to clarify things. This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of Natural Language Processing (NLP).
# 
# ## Content:
# ### 1. Natural Language Processing step-by-step 
# 
# This algorithm will predict if reviews are positive or negative. This will be done by using Bag of Words model. 
# 
# ## Difference between .tsv and .csv
# For text analysis, we need to use .tsv, as it is Tab Separated Values, which allows commas to be ignored when using each value. If .csv is used, as there are already many commas within sentences, the program will not know where it should separate between columns. Therefore, .tsv should be used.
# 
# ### 1) Preparing data

# In[ ]:


#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#can use read_csv but include parameters.
dataset = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #\t --> tab
#by putting quoting = 3, we ignore double quotes

print(dataset.shape[0])
dataset.head(10)



# ### 2) Cleaning Data
# We need to get rid of all unnecessary words and numbers, and apply stemming (converting different tenses to the same tense). This is necessary because the machine needs only important words to determine its positivity or negativity.

# In[ ]:


#import cleaning library
import re
dataset["Review"][0]

#Let's clean the first entry as an example

review = re.sub('[^a-zA-Z]', ' ', dataset["Review"][0], )
#first parameter --> ^ shows that we should not remove the following characters
#second parameter --> prevents the program from removing all the spaces.
#third parameter --> what is being processed
review


# In[ ]:


review = review.lower()
#makes all the letters lowercase
review


# In[ ]:


#further processing
import nltk #famous NLP library
nltk.download('stopwords') #download words that are not significant


# In[ ]:


#spliting the sentence
review = review.split()
review


# In[ ]:


#updating the list, and stemming (taking the root of each word)
from nltk.corpus import stopwords #importing the downloaded list
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer() 

review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
# 'english' is specified so that we are only looking at english words.
#use set, as it has faster searching algorithm than a list. This is useful for long text.
review


# In[ ]:


#joining the words into a sentence again
review = ' '.join(review)
review


# This changed "loved" to "love". Now, let's do this for all the sentences. We can use a for loop to go over all the indexes.
# 
# 
# 

# In[ ]:


#can use read_csv but include parameters.
dataset = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #\t --> tab
#by putting quoting = 3, we ignore double quotes

corpus = []
# curpus is a common word for collection of texts.

for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
pd_corpus = pd.DataFrame(corpus)
print(corpus[999])
print('')
print(dataset.loc[999, "Review"])
#just to check the last item in both of the lists match


# In[ ]:


#Creating the bag of words model --> will create a sparse matrix (a lot of zeros)
#create one column for each unique word. In the rows, if the word is not there, it is 0, and it is, it is 1. 
#This allows each word to be an independent variable leading to a dependent variable (Yes or No) classification

#Tokenization --> 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #restrict sparsity by removing irrelevant words
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
print(x.shape)
print(x[[1,2,3], :])
print(y[1:5])


# ### 3) Training the model
# We can apply the same classification algorithm to text, as now text became an independent variable. If not familiar with classification, please [refer to this kernel](https://www.kaggle.com/sangwookchn/classification-techniques-using-scikit-learn)
# #Most commonly used: Decision Tree, Naive Bayes

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling is not needed as we only have small integers: 0, 1, 2, 3, ---

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm


# As we only have 800 results to train on, this result is pretty good. We made (55 + 91) = 146 correct guesses and 54 incorrect ones. Now, let's try predicting on arbitrary text.

# In[ ]:


y1 = "not bad at all"
y2 = "if you think it's bad, you're bad"
y3 = "just wanted to vomit after eating this"
y4 = "it is incredibly bitter and therefore terrible"

list1 = [y1, y2, y3, y4]
corpus2 = []

for i in range(0, len(list1)):
    review = re.sub('[^a-zA-Z]', ' ', list1[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus2.append(review)

x2 = cv.transform(corpus2).toarray()

print(x2)


# The pretrained CountVectorizer is used, as the classifier is trained on this dataset. In the second sentence, as "bad" appears twice, the cell corresponding to this sentence and the word "bad" will show up as 2.

# In[ ]:


y_pred2 = classifier.predict(x2)
y_pred2


# Hm, the first and second sentence were positive reviews... Let's test some simple ones then.

# In[ ]:


y1 = "it's bad"
y2 = "it's unbelievably bad."
y3 = "tastes so good"
y4 = "I cannot forget this enjoyable taste"
y5 = 'terrible. not tasty at all'

list1 = [y1, y2, y3, y4, y5]
corpus2 = []

for i in range(0, len(list1)):
    review = re.sub('[^a-zA-Z]', ' ', list1[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus2.append(review)

x2 = cv.transform(corpus2).toarray()

print(x2)

y_pred2 = classifier.predict(x2)
y_pred2


# It seems clear that the algorithm confuses with words that may seem positive on its own, such as "incredible" and "unbelievable". This means the algorithm wasn't able to catch advanced English styles. Possibly, training on larger dataset will work better. Also, I need to research more advanced NLP algorithms out there. I will write another kernel once I find this. 

# In[ ]:




