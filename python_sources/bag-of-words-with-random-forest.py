#!/usr/bin/env python
# coding: utf-8

# ##  Importing all necessary libraries

# In[ ]:


import pandas as pd     
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords # Import the stop word list
from sklearn.model_selection import train_test_split


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Reading the data (Training & Testing data)

# In[ ]:


df_train = pd.read_csv("/kaggle/input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/labeledTrainData.tsv", 
                              header=0, 
                              delimiter="\t", 
                              quoting=3)

df_test = pd.read_csv("/kaggle/input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/testData.tsv",
                             header=0, 
                             delimiter="\t", 
                             quoting=3)


# In[ ]:


# X_train, X_test, Y_train, Y_test = train_test_split(pd.DataFrame(main_train_data["review"]), \
#                                                     pd.Series(main_train_data["sentiment"]), \
#                                                     test_size = 0.3, \
#                                                     random_state=42)

# train_data = pd.DataFrame(X_train).join(pd.DataFrame(Y_train))
# test_data = pd.DataFrame(X_test).join(pd.DataFrame(Y_test))

# train_data = train_data.reset_index(drop=True)
# test_data = test_data.reset_index(drop=True)

# print(train_data.shape)
# print(test_data.shape)
# print(main_test_data.shape)


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# In[ ]:


df_train.info()


# In[ ]:


print(df_train.columns.values)
print(df_test.columns.values)


# In[ ]:


df_train['review'][0]


# ## PreProcessing data for one item. 
# ###### beautifying the text of HTML and XML data
# 

# In[ ]:



bs_data = BeautifulSoup(df_train["review"][0])
print(bs_data.get_text())


# In[ ]:


letters_only = re.sub("[^a-zA-Z]", " ", bs_data.get_text() )
print(letters_only)


# In[ ]:


lower_case = letters_only.lower()  
words = lower_case.split()  
print(words)


# In[ ]:


print(stopwords.words("english") )


# In[ ]:


words = [w for w in words if not w in stopwords.words("english")]
print(words)


# ##  PreProcessing data for all of the training data

# In[ ]:


training_data_size = df_train["review"].size
testing_data_size = df_test["review"].size

print(training_data_size)
print(testing_data_size)


# In[ ]:


def clean_text_data(data_point, data_size):
    review_soup = BeautifulSoup(data_point)
    review_text = review_soup.get_text()
    review_letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    review_lower_case = review_letters_only.lower()  
    review_words = review_lower_case.split() 
    stop_words = stopwords.words("english")
    meaningful_words = [x for x in review_words if x not in stop_words]
    
    if( (i)%2000 == 0 ):
        print("Cleaned %d of %d data (%d %%)." % ( i, data_size, ((i)/data_size)*100))
        
    return( " ".join( meaningful_words)) 
    


# In[ ]:


# clean_train_data_list = []
# clean_test_data_list = []


# ##### cleaning training data.

# In[ ]:


df_train.head()


# In[ ]:


for i in range(training_data_size):
    df_train["review"][i] = clean_text_data(df_train["review"][i], training_data_size)
print("Cleaning training completed!")


# ##### cleaning testing data.

# In[ ]:


for i in range(testing_data_size):
    df_test["review"][i] = clean_text_data(df_test["review"][i], testing_data_size)
print("Cleaning validation completed!")


# ## Getting the features ready to be trained 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 


# In[ ]:


X_train, X_cv, Y_train, Y_cv = train_test_split(df_train["review"], df_train["sentiment"], test_size = 0.3, random_state=42)


# ### Converting the train, validation and test data to vectors

# In[ ]:


X_train = vectorizer.fit_transform(X_train)
X_train = X_train.toarray()
print(X_train.shape)


# In[ ]:


X_cv = vectorizer.transform(X_cv)
X_cv = X_cv.toarray()
print(X_cv.shape)


# In[ ]:


X_test = vectorizer.transform(df_test["review"])
X_test = X_test.toarray()
print(X_test.shape)


# In[ ]:


vocab = vectorizer.get_feature_names()
print(vocab)


# In[ ]:


distribution = np.sum(X_train, axis=0)

for tag, count in zip(vocab, distribution):
    print(count, tag)


# ## Training Random Forest model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest = RandomForestClassifier() 
forest = forest.fit( X_train, Y_train)


# ## Testing the model

# In[ ]:


predictions = forest.predict(X_cv) 
print("Accuracy: ", accuracy_score(Y_cv, predictions))


# ## Creating the output submission file

# In[ ]:


result = forest.predict(X_test) 
output = pd.DataFrame( data={"id":df_test["id"], "sentiment":result} )
output.to_csv( "submission.csv", index=False, quoting=3 )


# ### This was a beginners lesson to Bag of Words.

# **Please upvote if you find this useful!**
