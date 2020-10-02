#!/usr/bin/env python
# coding: utf-8

# First we import important packages like **pandas,nltk,re,os** we use pandas to handle our dataset it is used to take input of test and training data then we import stopwords to remove usnecessary words like is,are,names etc from the dataset we use re to keep only words i will explain this in details where we use re. then we import os for setting directory
# #if you dont have any of these files then you can download these files from command prompt by using pip install module name
# for pandas --- pip install pandas
# for nltk ---- pip install nltk then you have to download stopwords by going to python editor and import nltk then nltk.download() select all from gui or you can make custom download i suggest you to download all.
# Rest are inbuilt in python(excluding keras i explained thoses below) just import and enjoy.

# In[ ]:


import pandas as pd
from nltk.corpus import stopwords
import re
import os
print(os.listdir("../input"))


# we use pd.read_csv file to create to test and training data set then we use train.head() to take a look at our dataset so that we will know that which colunm contain what values.

# In[ ]:


df_train = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
df_train.head()


# here we divided our training dataset into x and y where x is review and y is its curresponding sentiment size of x is (25000,1) and size of y is (25000,1) you can check size by x.shape command.

# In[ ]:


X = df_train.iloc[:, 2].values
y = df_train.iloc[:, 1].values


# Here read our test_data and store it in varaible df_test and we store the reviews of test data as X_1 

# In[ ]:


df_test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", quoting=3)
X_1 = df_test.iloc[:, 1].values


# Here we have created a function review_to_words to clean the review words from our review section we remove stop words then we remove all special characters and keep only words. line by line explanation
# 
# **line1** ----  *"re.sub("[^a-zA-Z]"," ", raw_review)"* in this line we will keep all the alphabetical words which are present in the file name raw_review all special characters are replaced by a space. 
# 
# **line2** ---- * letters_only.lower().split()* convert the string into lowercase string then we use split() which will split the string and return a list of words.
# 
# **line3** ----  *set(stopwords.words("english"))* create a touple of stop words which are present in nltk stopword library
# 
# **line4** ----  * [word for word in lower_words if word.isalpha()]* if any special character is left we will remove that by creating a  list comprehension and checking ever word.
# 
# **line 4** ---- * [ w for w in words if not w in stops]*  here we keep only those words which are not  present in the stop word touple.
# 
# **line 5** ----  *" ".join(meaningful_words) * joining all the words back and making a string again.
# 

# In[ ]:


def review_to_words(raw_review):
    letters_only = re.sub("[^a-zA-Z]"," ", raw_review)
    lower_words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    words = [word for word in lower_words if word.isalpha()] #removing special character and numbers
    meaningful_words = [ w for w in words if not w in stops]
    return(" ".join(meaningful_words))


# created a empty list filtered_x and stored the size of X which is out training dataset in total_review, then we apply a for loop to filtere all the reviews present in the training data. 

# In[ ]:


filtered_x = []
total_reviews = X.size  #total number of reviews present or number of rows
for i in range(0,total_reviews):
    filtered_x.append(review_to_words(X[i]))


# Importe train_test_split from sklearn.model_selection so that we can split our training and test data. here we choose the ratio is 80:20 80% training set and 20% validation set.
# if you don't have sklearn then use pip install sklearn and you are good to go. 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split( filtered_x, y, test_size = 0.2, random_state = 0)
x_test = df_test["review"].map(review_to_words)


# we import Tokenizer and pad_sequence from keras.preprocessig Tokenizer is used for text preprocessing.
# we tokenize the words in numeric values here we can choose how many words we want to tokenize so we choose 2000 so most frequenty comming 2000 words will be tokenized.
# 
# **how tokenizer works**
# The Tokenizer stores everything in the word_index during fit_on_texts. Then, when calling the texts_to_sequences method, only the top num_words are considered.
# 
# Then we pad the sequence by importing pad_sequence it is  used to ensure that all sequences in a list have the same length. here wh choose maxlen = 400 you can choose any other value 

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Here we converted all our words to numbers so that our model can understand

# In[ ]:


tokenizer = Tokenizer(num_words=2000) #tokenised to 2000 most frequent words
tokenizer.fit_on_texts(filtered_x)
# padding sequence to the limit is 500 words so it will look 500 words back 
train_reviews_tokenized = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(train_reviews_tokenized, maxlen=400)
val_review_tokenized = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(val_review_tokenized, maxlen=400)
test_review_tokenized = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(test_review_tokenized, maxlen=400)


# It's time to build our RNN model we use sequential model and for layers we use Dense and Embedding and LSTM layers we can import all these from keras 
# if you dont have all these then use pip install keras and you are good to go.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


# **Embedding** --- The weights of the Embedding layer are of the shape (vocabulary_size, embedding_dimension). For each training sample, its input are integers,so here our vocabulary size is 2000 and we choose 128 embedding_dimensions we can also call thses as hidden neurons.
# 
# **LSTM dropout explanation:**
# dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
# recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
# in recurrent dropout the connections between the recurrent units will be dropped we we have choose 20% of the linear and recurrent connection will dropout every iteration.
# 
# we have used sigmoid activation because its a binary classification function.and loss is calculated by binary_crossentropy function and here we have used adam optimizer.it is one of the best optimizer present in keras to tackle classification problems.

# In[ ]:


model = Sequential()
model.add(Embedding(20000,128)) #20000 words and funneling them into 128 hidden neurons
model.add(LSTM(128,dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = "sigmoid"))
#compiling model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# now we fit our model with a batch size of 32 and number of epoch is 5 after this training will start.

# In[ ]:


model.fit(X_train, Y_train, batch_size = 32, epochs = 5, validation_data=[X_val, Y_val])


# In[ ]:


prediction = model.predict(x_test)
y_pred = (prediction > 0.5)


# In[ ]:


df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = df_test["sentiment"]


# In[ ]:


from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)


# In[ ]:




