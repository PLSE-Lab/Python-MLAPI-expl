#!/usr/bin/env python
# coding: utf-8

# # Roman Urdu Sentiment Analysis with Deep Learning

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### **GOAL**
# * Train a sentiment classifier (Positive, Negative, Neutral) on a corpus of the provided dataset below. 
# * Maximize accuracy of the classifier with a special interest in being able to accurately detect negative sentiment.
# 
# ### **Data Set Description** 
# Classifier model developed below uses a dataset obtained form UCI Machine Learning Repository located <a href="http://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set">here</a>
# 
# This dataset is authored by Zareen Sharf, from Shaheed Zulfiqar Ali Bhutto Institute of Science and Technology (SZABIST). Data set contains sentences in Urdu languate and it is tagged for sentiements either, Positive, Negative, or Neutral.
# 
# Sentences in Urdu are written in plain English for word processing rather than native Urdu fonts. Data includes documents from a wide variety of sources, not merely social media, and some of it may be inconsistently
# labeled.

# ### **Plan of Work** 
# 
# Sentiment analysis involves the following stages. 
# 
# * Loading and cleaning the data
# * Exploratory data analysis
# * Ommiting stop words or common words
# * Creating new features
# * Plotting most frequently used words
# * Vectorizing sentences to build a matrix
# * Training and testing the machine learning model
# * Evaluating the machine learning model accuracy

# In[ ]:


# Loading required python packages and libraries
import nltk
import pandas as pd
import re
import string
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from wordcloud import WordCloud,STOPWORDS


# In[ ]:


# Load the csv file using Pandas and print first 5 lines
data = pd.read_csv("../input/urduromansentiment/Roman Urdu DataSet.csv",header=None)
data.head()


# ### **Data Preprocessing**
# 
# In this step, data is preprocessed or cleaned for missing column names, incorrect values of sentiments, null values present in the text body. 

# In[ ]:


# Adding column names
data.columns =['body_text','sentiment','unknown']


# In[ ]:


# Print unique values in column 1 and 2
print ('Unique values of the sentiments are', data.sentiment.unique())
print ('Unique values of the unknonwn column are', data.unknown.unique())


# In[ ]:


# 'Neative' sentiment will be most likely Negative, so it is replaced accordingly. 
data[data['sentiment']=='Neative']='Negative'


# In[ ]:


# Verify we replaced all the 'Neative'
print ('Unique values of the sentiments are', data.sentiment.unique())


# In[ ]:


# Checking Null values in the data
data.isna().sum()


# In[ ]:


# Dropping the text body row which has a null value
data = data.dropna(subset=['body_text'])


# In[ ]:


# Last column can be dropped as it does not contain any useful information. Here axis=1, means column. 
data = data.drop('unknown', axis=1)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# ### **Data Exploration**
# 
# From the data of sentiments, it looks like it is not balanced. There are more neutral comments than positive and negative comments. When the class distribution is unbalanced, accuracy is considered a poor choice of evaluation of classifier. This is due the fact that, it gives high scores to models which just predict the most frequent class.

# In[ ]:


print ('Number of sentiments in each category are as below')
print (data.sentiment.value_counts())

print ('\nPerecentage sentiments in each category are as below')
print (data.sentiment.value_counts()/data.shape[0]*100)

data.sentiment.value_counts().plot(kind='bar')


# In[ ]:


# Dropping neutral sentiment sentences. 
data = data[data.sentiment != 'Neutral']


# In[ ]:


data = data.reset_index(drop=True)


# In[ ]:


data.head()


# In[ ]:


data.sentiment.value_counts().plot(kind='bar')


# In[ ]:


data.describe()


# ### **Data Cleaning**
# 
# * Make all words lowercase
# * Replace any stopwords
# 
# **Frequently Used Words**
# 
# It is important to check for frequency of words. This helps to identify any missing stopwods or its variation in spelling. Stopwords are then updated if required. 
# 
# * First we will create a long list containing all words in the text provided. 
# * Unique words are then counted and words with significant number of occurances are reported. 
# * Stopwords list is updated.
# * Remaining high frequency words are then visually inspected using wordcloud by each sentiment.
# 
# From words in worldcloud of each sentiment, it can be noted that there is some overlapp of words between three classes. Though most of the words are different and correct. 

# In[ ]:


text_wordcloud = " ".join(word.lower() for word in data.body_text)
print ('There are total {} words in text provided.'.format(len(text_wordcloud)))


# In[ ]:


str2 = [] 
def freq(str): 
  
    # Break the string into list of words  
    str = str.split()          
    #str2 = [] 
  
    # Loop till string values present in list str 
    for i in str:              
  
        # Checking for the duplicacy 
        if i not in str2: 
  
            # Append value in str2 
            str2.append(i)  
              
    for i in range(0, len(str2)): 
        if(str.count(str2[i])>100): 
            print('Frequency of word,', str2[i],':', str.count(str2[i]))
            
freq(text_wordcloud)


# In[ ]:


print ('Number of unique words in vocabulary are',len(str2))


# In[ ]:


UrduStopWordList = [line.rstrip('\n') for line in open('../input/urdustopwords/stopwords.txt')]

print (UrduStopWordList)


# In[ ]:


stopwords_with_urdu = set(STOPWORDS)
stopwords_with_urdu.update(UrduStopWordList)


wordcloud = WordCloud(stopwords=stopwords_with_urdu,
                      background_color='white',
                      width=3000,
                      height=2500
                     ).generate(text_wordcloud)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


neg_text_wordcloud = " ".join(word.lower() for word in data[data['sentiment']=='Negative']['body_text'])
print ('There are total {} words in sentences with negative sentiments.'.format(len(neg_text_wordcloud)))


# In[ ]:


# Plotting Plotting words in setences with negative sentiment
wordcloud = WordCloud(stopwords=stopwords_with_urdu,
                      background_color='white',
                      width=3000,
                      height=2500
                     ).generate(neg_text_wordcloud)
plt.figure(1,figsize=(12, 12))
plt.title('Negative Sentiment Words',fontsize = 20)
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


pos_text_wordcloud = " ".join(word.lower() for word in data[data['sentiment']=='Positive']['body_text'])
print ('There are total {} words in text with positive sentements.'.format(len(pos_text_wordcloud)))


# In[ ]:


# Plotting words in positive sentiment sentences

wordcloud = WordCloud(stopwords=stopwords_with_urdu,
                      background_color='white',
                      width=3000,
                      height=2500
                     ).generate(pos_text_wordcloud)
plt.figure(1,figsize=(12, 12))
plt.title('Positive Sentiment Words',fontsize = 20)
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()


# # Data Preprocessing

# In[ ]:


def clean_text(text):
    #Change each character to lowercase and avoid any punctuation. Finally join word back. 
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    
    # Use non word characters to split the sentence
    tokens = re.split('\W+', text)

    # Remove the stop words - commonly used words such as I, we, you, am, is etc in Urdu language 
    # that do not contribute to sentiment. 
    text = [word for word in tokens if word not in stopwords_with_urdu]
    return text


# In[ ]:


data['body_text'] = data['body_text'].apply(lambda x: clean_text(x))


# In[ ]:


data.head()


# In[ ]:


y = data['sentiment']
y = np.array(list(map(lambda x: 1 if x=="Positive" else 0, y)))


# ### Split into train/test
# 

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['body_text'],y, test_size=0.2,random_state=42,stratify=data['sentiment'])


# In[ ]:


from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


# In[ ]:


X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)


# In[ ]:


X_train.shape,X_test.shape


# ### Building Machine Learning Models

# In[ ]:


n_words = X_train.shape[1]
print (n_words)


# In[ ]:


vocab_size = len(tokenizer.word_index) + 1
print (vocab_size)


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense

def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(100, input_shape=(n_words,), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

model = define_model(n_words)
# fit network
model.fit(X_train, y_train, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))


# In[ ]:


y_pred = model.predict(X_test)


# **Classification Accuracy**
# 
# When classifying the comments by sentiments into three classes (Negative, Neutral, positive), it is important that negative comments are reported with higher accuracy. Precision and Recall allow us to check the performance of a classifier. 
# 
# *Precision*:
# * Precision checks out of all reported negative comments, how many are actually negative. So it checks for percentage of correct Negative Sentiments among all reported Negative sentiments
# * General definition of precision for a class is,
#     * Precision = TP/(TP+FP)
# * If we think of it for Negative sentiments class, 
#     * Precision = True Negative Sentiments/(True Negative Sentiments + Falsely Reported Negative Sentiments)
# 
# 
# *Recall*:
# * Recall checks for missed positive prediction which are misclassified as false negative. 
# * General definition of precision for a class is,
#     * Recall = TP/(TP+FN)
# * For Negative sentiments class it is, 
#     * Recall = True Negative Sentiments/(True Negative Sentiments + Missed Negative Sentiments in Reported)
# 
# When classification model improves, precision of reporting negative sentiments goes high. Other class sentiments, which are falsely reported Negative goes down. 
# 
# For satisfactory performance of model, it is desired to have both high precision and high recall to get a final high accuracy. F1 score considers both precision and recall and gives a single number to compare. We need to select parameters of the model for which F1 score is high for Negative class.  
# 
# For overall accuracy of classifier, micro score is important. 
# 
# *How much accuracy is enough?*
# 
# Generally precision of nearly 70% is considered as a good classifier performance. This due to the fact that subtle meaning of words is perceived differently[](https://mashable.com/2010/04/19/sentiment-analysis/) by humans and also can not be captured by machine learning models. 
# 
# As per this Wikipedia [source](https://en.wikipedia.org/wiki/Sentiment_analysis): 
# > The accuracy of a sentiment analysis system is, in principle, how well it agrees with human judgments. This is usually measured by variant measures based on precision and recall over the two target categories of negative and positive texts. However, according to [research](https://mashable.com/2010/04/19/sentiment-analysis/) human raters typically only agree about 80% of the time (see Inter-rater reliability). Thus, a program which achieves 70% accuracy in classifying sentiment is doing nearly as well as humans, even though such accuracy may not sound impressive
# 

# **Classification report for Random Forest classifier**

# In[ ]:


from sklearn.metrics import precision_recall_fscore_support as score, roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from mlxtend.plotting import plot_confusion_matrix

print ('Classification Report for Classifier:\n',classification_report(y_test, y_pred.round(),digits=3))
print ('\nConfussion matrix for Classifier:\n'),confusion_matrix(y_test, y_pred.round())


# In[ ]:


cm = confusion_matrix(y_test,y_pred.round())
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)
plt.xticks(range(2), ['Negative','Positive'], fontsize=20)
plt.yticks(range(2),['Negative','Positive'] , fontsize=20)
plt.show()


# **Predicting with Model**

# In[ ]:


k = random.randint(0,data.shape[0])
message = data['body_text'][k]
message


# In[ ]:


data['sentiment'][k]


# In[ ]:


X_predict_vect = tokenizer.texts_to_matrix(message)


# In[ ]:


y_message =max(model.predict(X_predict_vect))
y_message_rnd = np.round(y_message)

if y_message_rnd==1:
    print ("Positive")
else:
    print ("Negative")


# References:
# 
# * [The Basics of Sentiment Analysis](https://monkeylearn.com/sentiment-analysis/)
# * [Multiclass Classification Reporting and Interpretation](https://stackoverflow.com/questions/30746460/how-to-interpret-scikits-learn-confusion-matrix-and-classification-report)
# * <a href ="https://8kmiles.com/blog/benchmarking-sentiment-analysis-systems/">Benchmarking Sentiment Analysis</a>
# * <a href="https://www.lynda.com/Python-tutorials/NLP-Python-Machine-Learning-Essential-Training/622075-2.html">NLP with Python for Machine Learning - Lynda.com Course</a>
# * <a href ="https://github.com/haseebelahi/roman-urdu-stopwords">Roman Urdu Stopwords</a>
# * <a href="https://www.datacamp.com/community/tutorials/wordcloud-python">Wordcloud in Python</a>
# * <a href="https://www.kaggle.com/parthsharma5795/comprehensive-twitter-airline-sentiment-analysis">Airline Twitter Sentiment Analysis</a>
# * <a href="https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/">Hands on Guide to Twitter Sentiment Analysis</a>
# * <a href="https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case">Stackoverflow</a>
# * <a href="https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1">Multiclass Metric</a>
