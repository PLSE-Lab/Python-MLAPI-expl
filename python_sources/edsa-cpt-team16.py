#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[ ]:


import nltk
import string
from nltk.corpus import stopwords
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Let's read in our MBTI Headset with 'read_csv'

# In[ ]:


#train  =  pd.read_csv('/content/gdrive/My Drive/Explore/train.csv' )
#test = pd.read_csv('/content/gdrive/My Drive/Explore/test.csv' )

train  =  pd.read_csv('train.csv' )
test = pd.read_csv('test.csv' )


# #### Now let's view this data
# - We will just view the first three columns just for saving space

# In[ ]:


train.head(3)


# #### Now let's check if there are any missing values in our dataset just for a double-check

# In[ ]:


train.isnull().sum()


# - As we were told the dataset is indeed without missing values
# - Now let's confirm the size of this dataset

# In[ ]:


train.shape


# #### EDA let's explore the 'posts' column:

# In[ ]:


train.iloc[1,1].split('|||')[:10]


# - now let's confirm if the length has 50 posts as promised by the competition designers

# In[ ]:



print('We have : ',len(train.iloc[1,1].split('||')), ' posts in each row.')


# - Now that the number of posts is confirmed let's check if we have exactly 16 personalities in the 'type' column

# In[ ]:


train['type'].unique()


# - Again as promised we have only 16 personalities in alignment with MBTI personality types
# - Now let' explore the dataset further by looking at the number of posts per personality type
# - and then plot the distribution of the total number of posts per type
# - we will achieve this by grouping the train dataframe by type and count that and then multiplying by 50
# - we multiply by 50 because as we have realized each post row consist of 50 posts

# In[ ]:


total = train.groupby(['type']).count()*50
total #### show the total dataframe


# #### now lets plot that distribution for a better view perspective

# In[ ]:



plt.figure(figsize = (12,6))

plt.bar(np.array(total.index), height = total['posts'],)
plt.xlabel('Personality types', size = 14)
plt.ylabel('Number of posts available', size = 14)
plt.title('Total posts for each personality type')


# - as we can see with the distribution graph above
# - Introverted types like INFP, ITNJ, ITNP and INFJ post way more than Extroverted type
# - and this is viewed in contrast with the distribution by the Extroverted types
# - and these include ENFJ, ENFP, ENTP and ENTJ
# - Now we have explored our dataset comfortably let's create our binary columns
# - and these columns are based on the MBTI traits : mind (I/E), enery(N/S),nature(T/F) and tactics(J/P)
# 
# - This we achieve by applying a lambda to create binary columns for traits and assign 0 and 1 appropriately

# In[ ]:



train['mind'] = train['type'].apply(lambda x: x[0] == 'E').astype('int')
train['energy'] = train['type'].apply(lambda x: x[1] == 'N').astype('int')
train['nature'] = train['type'].apply(lambda x: x[2] == 'T').astype('int')
train['tactics'] = train['type'].apply(lambda x: x[3] == 'J').astype('int')


# #### now let's view our changes 

# In[ ]:


train.head()


# In[ ]:


messages=pd.concat([train,test],join='inner')


# In[ ]:


messages.info()


# - Replace the urls with 'url-web'
# - We achieve this by using pd.replace() with a regex

# In[ ]:


pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
messages['posts'] = messages['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)


# In[ ]:


messages.head()


# - We are going to remove punctuations using string 'maketrans'
# - return the df with no punctuations

# In[ ]:


def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


# In[ ]:


messages['posts'] = messages['posts'].apply(remove_punctuation)


# In[ ]:


messages.head()


# In[ ]:


#### let's confirm if we correctly got stopwords
stopwords.words('english')[0:10] 


# In[ ]:


sw = stopwords.words('english')


# - Now let define a function to remove stop words and 
# - and then appy that to messages 'posts' column

# In[ ]:


def remove_stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)


# In[ ]:


messages['posts'] = messages['posts'].apply(remove_stopwords)


# In[ ]:


messages.head(3)


# - Let's use snowball stemmer to take words to their root form before vectorization
# - And then create a fuunction that we will appy to messages 'posts' column to perform stemming

# In[ ]:


#### let's take the words to their root source
stemmer = SnowballStemmer("english")

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 


# In[ ]:


messages['posts'] = messages['posts'].apply(stemming)


# 
# ### Declare both a CountVectorizer and TFIDF vectorizer
# - after applying TFDIFVectorizer we saw and improvement in our model and decided to stick with that
# - for our vectorization

# In[ ]:


#count_vectorizer = CountVectorizer()


# In[ ]:


tfid_vectorizer = TfidfVectorizer("english")


# In[ ]:



#print(np.mean(train.mind==1))
#print(np.mean(train.energy==1))
#print(np.mean(train.nature==1))
#print(np.mean(train.tactics==1))


# #### Now let's do our train test split for all traits

# In[ ]:




msg_train1, msg_test1, label_train1, label_test1 =train_test_split(messages['posts'].iloc[0:6506], train['mind'])
msg_train2, msg_test2, label_train2, label_test2 =train_test_split(messages['posts'].iloc[0:6506], train['energy'])
msg_train3, msg_test3, label_train3, label_test3 =train_test_split(messages['posts'].iloc[0:6506], train['nature'])
msg_train4, msg_test4, label_train4, label_test4 =train_test_split(messages['posts'].iloc[0:6506], train['tactics'])


# #### Let's create a pipeline to use with out bow, vectorizer and models
# - We Tried RandomForest, SVC, BernnouliNB, And simple Naive Bayse
# - RandomFores performs worse than all the above models 
# - Our assumption is that it is because of the binary nature of our classifier
# - Random Forest performs well with multi classifiers
# 
# 
# - SVC,Bernnouli and simple Naive Bayes also performs bad
# - this is because they also in design are not appropriate for this type of challenge
# - our conclusion was to drop all of them except for logistic regression which seem to perform well at this
# - Now let's begin our modelling

# In[ ]:


#from sklearn.pipeline import Pipeline

#pipeline = Pipeline([
 #   ('bow', CountVectorizer(ngram_range=(1,3),min_df=2,max_df=1.0)),#,binary=True)),  # strings to token integer counts
  #  ('tfidf', TfidfTransformer()),
   # ('model',LogisticRegression(solver='lbfgs',multi_class='ovr',C=5,class_weight='balanced') ),# integer counts to weighted TF-IDF scores,# train on TF-IDF vectors w/ Naive Bayes classifier
#])


# - Now we fit and train our models for all traits
# - We then print the Results for each train
# - We use the threshold for each traint to get our y_thresh

# In[ ]:


#### fit the train for the mind

pipeline.fit(msg_train1,label_train1)
predictions1 = pipeline.predict(msg_test1)
y_prob=pipeline.predict_proba(msg_test1)
y_thresh = np.where(y_prob[:,1] > 0.77, 1, 0)


# In[ ]:


#### print the results for mind

print(accuracy_score(predictions1,label_test1)*100)
print(log_loss(predictions1,label_test1))
pipeline.fit(messages['posts'].iloc[0:6506],train['mind'])
predictions1=pipeline.predict(messages['posts'].iloc[6506:])
y_thresh1=pd.Series(predictions1)
y_thresh1[:10]


# In[ ]:


#### fit for the energy trait

pipeline.fit(msg_train2,label_train2)
predictions2 = pipeline.predict(msg_test2)
y_prob=pipeline.predict_proba(msg_test2)
y_thresh2 = np.where(y_prob[:,1] > 0.14, 1, 0)


# In[ ]:


#### print the Results for energy
print(accuracy_score(predictions2,label_test2)*100)
print(log_loss(predictions2,label_test2))
pipeline.fit(messages['posts'].iloc[0:6506],train['energy'])
predictions2=pipeline.predict(messages['posts'].iloc[6506:])
y_thresh2=pd.Series(predictions2)
y_thresh2[:10]


# In[ ]:


#### fit for the nature trait

pipeline.fit(msg_train3,label_train3)
predictions3 = pipeline.predict(msg_test3)
y_prob=pipeline.predict_proba(msg_test3)
y_thresh3 = np.where(y_prob[:,1] > 0.54, 1, 0)


# In[ ]:


#### print the Results for energy
print(accuracy_score(predictions3,label_test3)*100)
print(log_loss(predictions3,label_test3))
pipeline.fit(messages['posts'].iloc[0:6506],train['nature'])
predictions3=pipeline.predict(messages['posts'].iloc[6506:])
y_thresh3=pd.Series(predictions3)
y_thresh3[:10]


# In[ ]:



#### fit for the tactics trait

pipeline.fit(msg_train4,label_train4)
predictions4 = pipeline.predict(msg_test4)
y_prob=pipeline.predict_proba(msg_test4)
y_thresh4 = np.where(y_prob[:,1] > 0.60, 1, 0)


# In[ ]:


#### print the Results for nature
print(accuracy_score(predictions4,label_test4)*100)
print(log_loss(predictions4,label_test4))
pipeline.fit(messages['posts'].iloc[0:6506],train['tactics'])
predictions4=pipeline.predict(messages['posts'].iloc[6506:])
y_thresh4=pd.Series(predictions4)
y_thresh4[:10]


# - Finally we save our predictions into a dataframe and sumbit to kaggle
# - There is a lot of exploration that can still be performed with this problem 
# - We did not remove any emojis, hashtags and mentions ...which we will definitely pursue
# - Shoooo!

# In[ ]:



#### Save our results for kaggle

values=pd.concat([y_thresh1,y_thresh2,y_thresh3,y_thresh4],axis=1)
values.index=test.index
values=values.rename(columns={0:'mind',1:'energy',2:'nature',3:'tactics'})
values.index=test.id
values.head()
values.info()
values.to_csv('/content/gdrive/My Drive/Explore/Final_Output.csv')



