#!/usr/bin/env python
# coding: utf-8

# ![The Irish News](https://www.nova.ie/wp-content/uploads/2017/12/The-Irish-Times.jpg)

# Before dive-in to the NLP, Here are some information about the Irish Times
# 
# ## Irish Times
# 
# The Irish Times is an Irish daily broadsheet newspaper launched on 29 March 1859. The editor is Paul O'Neill who succeeded Kevin O'Sullivan on 5 April 2017; the deputy editor is Deirdre Veldon. The Irish Times is published every day except Sundays. It employs 420 people. More info in https://en.wikipedia.org/wiki/The_Irish_Times
# 
# ### Contents of the NLP
# The following process will be undertaken in the NLP
# 1. Importing the libraries needed for the process
# 2. Downloading the StopWords and Lemmatizer
# 3. Perform Tokenization and Removal of StopWords
# 4. Initialization of Count Vectorizer
# 5. Split of Train and Test dataset
# 6. Creation of Model
# 7. Performing Predictions and analyze the performance

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# #### 1. Importing the libraries needed for the process
# Let's have a quick walkthrough about the libraries that gonna be used.
# 
# -> **Pandas** - Handling and Manipulation of Data Frame
# 
# -> **NLTK (Natural Language Toolkit)** - Powerful NLP libraries which contains packages to make machines understand human language
# 
# -> **RE (Regular Expression)** - Special sequence of characters that helps you match or find other strings or sets of strings.
# 
# -> **SKLearn (Scikit-learn)** - Machine learning library for the Python programming language. It features various classification, regression and clustering algorithms.
# 

# In[ ]:


import pandas as pd
import re

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# ### Analysis of the Dataset
# We use **read_csv** in Pandas to read the dataset

# In[ ]:


irish_data = pd.read_csv('../input/ireland-historical-news/irishtimes-date-text.csv')


# The dataset consists of 3 columns and 1425460 rows.
# 
# The dataset consists of three columns:
#     1. publish_date - The date when the news gets published. The format of the date is YYYYMMDD
#     2. headline_category - The category of the news
#     3. headline_text - The headline information of the news

# In[ ]:


print("The shape of the data is ",irish_data.shape)
irish_data.head()


# #### 2. Downloading the StopWords and Lemmatizer
# Now, Let's download the stopwords and initialize the Lemmatizer. Note that, NLTK has a various language version of StopWords. You can have a look at the documentation of NLTK.
# 
# We will initialize the WordNetLemmatizer and also make a call to the function with a random words, because the lemmatizer will take some time to load the words into the workspace initially. So, it's my practice to run the function using initialization.

# In[ ]:


stop_words = set(stopwords.words('english')) 
lem = WordNetLemmatizer()
lem.lemmatize('Ready')


# Here, we will use the below function to remove the stopwords and also special characters from the words using **re**

# #### 3. Perform Tokenization and Removal of StopWords

# In[ ]:


def remove_stopwords(line):
    word_tokens = word_tokenize(line)  
    filtered_words = [re.sub('[^A-Za-z]+', '', w.lower()) for w in word_tokens if not w in stop_words]
    return filtered_words


# Now, we will use the above function and create a new column called **tokenized** which will contain the filtered words.

# In[ ]:


irish_data['tokenized'] = irish_data['headline_text'].apply(lambda x: remove_stopwords(x))


# Below is the function used to lemmatize the filtered words and also we will create a new column called 'lemmatized' which will contain the lemmatized words.

# In[ ]:


get_words = []

def lemmatize(line):
    lem_words = []
    
    for word in line:
        lem_word = lem.lemmatize(word,"v")
        if len(lem_word) > 1:
            lem_words.append(lem_word)
            get_words.append(lem_word)
    
    return lem_words


# In[ ]:


irish_data['lemmatized'] = irish_data['tokenized'].apply(lambda x: lemmatize(x))


# In[ ]:


freq_words = nltk.FreqDist(get_words)
less_words = []

for word in freq_words:
    if freq_words[word]<=2:
        less_words.append(word)


# In[ ]:


def remove_less_words(lists):
    filters = []
    
    for word in lists:
        if word not in less_words:
            filters.append(word)
            
    return ' '.join(filters)


# In[ ]:


irish_data['filtered'] = irish_data['lemmatized'].apply(lambda x: remove_less_words(x))


# So, now the data is filtered and lemmatized and stored in a column as you can see below

# In[ ]:


irish_data.head()


# #### 4. Initialization of Count Vectorizer
# Let's initialize the CountVectorizer. Below you can see that some parameters are initialized. We will go thorugh the parameters.
# 
# - ngram_range = (1,2) => We are specifying the Count Vectorizer to use both unigrams and bigrams. Setting it to (2, 2) means only bigrams and (1, 1) means only unigrams.
# 
# - lowercase=True => We are transforming the text into lowercase (**Note**: We already did this process before in Step 3, but just to show that we can do it via Count Vectorizer :) )
# 
# - stop_words='english' => We are specifying the stopwords as english to remove from the sentence. (**Note**: We already did this process before in Step 3, but just to show that we can do it via Count Vectorizer :) )
# 
# - tokenizer = token.tokenize => We are removing the special characters from the text using **RegexpTokenizer**. (**Note**: We already did this process before in Step 3, but just to show that we can do it via Count Vectorizer :) )
# 
# 

# In[ ]:


#token = RegexpTokenizer(r'[a-zA-Z0-9]+')
#CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,2),tokenizer = token.tokenize)
cv = CountVectorizer(ngram_range = (1,1))
text_counts= cv.fit_transform(irish_data['filtered'])


# #### 5. Split of Train and Test dataset
# Now, the dataset is ready for training. So,Let's split the data into train and test data. We will use **train_test_split** to perform the step.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(text_counts, irish_data['headline_category'], test_size=0.3, random_state=101)


# #### 6. Creation of Model
# Now, Let's create the model. We will select **LogisticRegression** as our model for NLP. Below, is the model creation and training the model with the train input and train output.

# In[ ]:


logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)


# #### 7. Performing Predictions and analyze the performance
# Now, the model is trained with the dataset. Let's use the test dataset (X_test) to predict the outputs

# In[ ]:


y_pred = logistic_model.predict(X_test)


# Now, the prediction is performed and we will use the **accuracy score** function to calculate the accuracy level

# In[ ]:


accuracy_per = accuracy_score(y_test, y_pred)

print("Accuracy on the dataset: {:.2f}".format(accuracy_per*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




