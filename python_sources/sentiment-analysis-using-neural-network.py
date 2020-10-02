#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing all necessary libraries to run the code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import re
# using the variable sw to hold all stopwords that are in English
sw = stopwords.words('english')


# In[ ]:


# reading csv file with the data for analyse
ds = pd.read_csv('../input/googleplaystore_user_reviews.csv')


# In[ ]:


# checking to see how the data are formatted
ds.head()


# It is possible to realize that the data have NaN values, so we need to remove them, since this data will not add anything to our analysis. We are trying to predict the Sentiment of a person, based on their app review. So, the columns 'Translated_Review' and 'Sentiment' will be used to get our result.

# In[ ]:


# the method info of a dataframe shows us the number of null coluns of our data
ds.info()


# In[ ]:


# Number of elements before removing the NaN values
print('Size before removing Nan: %s'% len(ds))

# Number of elements after removing the NaN values
ds.dropna(axis=0, inplace=True)
print('Size before removing Nan: %s'% len(ds))


# This function was used to clean the data. The stopwords were removed from the data since it also does not add too much to the analysis. To detect the words that have negative meaning was used a regex. This regex find the words ends in "n't" or is the word "not" or "no" or "never". More words can be add later and it may increase the result. When this words are found, the next word will have a "not_" before the word. <br>
# I.E: <lo> <li><b>Input:</b> 'This app is not good' <br> </li>
#      <li><b>Output:</b> ['app', 'not_good'] </li> </lo> <br>
# This is the return of the cleaning_data function to the input given above

# In[ ]:


def cleaning_data(data):
    aux_list = []
    flag = False
    for phase_word in data:
        word_list = []
        for word in phase_word.split():
            word = word.lower()
            if flag and not word in sw:
                flag = False
                word_list.append('not_'+word)
                continue
            if re.search('(n\'t)$|(not)|(no)|(never)', word):
                flag = True
                continue
            if not word in sw:
                word = re.sub('[\W_0-9]', ' ', word)
                word_list.append(word)
        aux_list.append(' '.join(word_list))
    return aux_list


# After removing the null values, let's split the data in, training and test case, so we can train our model and test to check the model accuracy.

# In[ ]:


X = cleaning_data(ds['Translated_Review'])
y = ds['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# To use a NN is necessary to apply an label encode in the output. Since the output is 'Positive','Neutral' or 'Negative' with the enconder applied the result is 0,1 or 2

# In[ ]:


# This CountVectorizer is used to represent the words as a list of values, instead of text
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

vectorizer.fit(X)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y_train = le.transform(y_train)
y_test = le.transform(y_test)


# The keras library was used to create a NN. The NN uses the relu as activation function in the hidden layer and the sigmoid in the output layer. 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=3, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, 
          epochs=2, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy:", scores[1])


# The result of the analysis can be seen above, the NN created was capable of predict correctly about 92% of the test cases.
