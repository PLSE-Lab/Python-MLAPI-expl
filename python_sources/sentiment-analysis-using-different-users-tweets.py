#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Airline users using thier tweets.

# **Breakdown of this notebook:**
# 
# 1. Loading the dataset: Load the data and import the libraries.
# 2. Data Preprocessing:
#      - Analysing missing data. 
#      - Removing redundant columns.
# 3. Visualising and counting sentiments of tweets for each airline.
# 4. Wordcloud plots for **positive** and **negative** tweets to visualise most frequent words for each.
# 5. Analysing the reasons for **negative tweets** for each airline.
# 6. Visualising negative tweet-sentiment relationship with dates.
# 7. Predicting the tweet sentiments with tweet text data with:
#       - SVM(Support Vector Machine)
#       - Decision Tree Classifier
#       - Random Forest Classifier
# 8. Calculating accuracies, plotting the confusion matrix and comparing the models.

# ### References:- 
# I learnt a lot from this blog which shows you how to handle nlp data and implement data preprocessing and explanatory visualization for better understanding.
# 
# https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/

# ### Importing the libraries and loading the data

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #data visualisation

import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("All libraries are imported")


# In[ ]:


df= pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")
df.head()


# ### Data Preprocessing

# The first step should be to check the shape of the dataframe and then check the number of null values in each column.
# 
# In this way we can get an idea of the redundant columns in the data frame depending on which columns have the highest number of null values.

# In[ ]:


print("Shape of the dataframe is",df.shape)
print("The number of nulls in each column are \n", df.isna().sum())


# To get a better idea, lets calculate the percentage of nulls or NA values in each column

# In[ ]:


print("Percentage null or na values in df")
((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2)


# Here **tweet_coord , airline_sentiment_gold, negativereason_gold**  have more than 90% missing data. It will be better to delete these columns as they will not provide any constructive information.
# 
# 

# In[ ]:


del df['tweet_coord']
del df['airline_sentiment_gold']
del df['negativereason_gold']
df.head()


# ### Airline sentiments for each airline
# 

# - Firstly lets calculate the total number of tweets for each airline
# - Then, we are going to get the barplots for each airline with respect to sentiments of tweets (positive,negative or neutral).
# - This will give us a clear idea about the airline sentiments and airlines relationship. 

# In[ ]:


print("Total number of tweets for each airline \n ",df.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False))
airlines= ['US Airways','United','American','Southwest','Delta','Virgin America']
plt.figure(1,figsize=(15,15))
for i in airlines:
    indices= airlines.index(i)
    plt.subplot(2,3,indices+1)
    new_df=df[df['airline']==i]
    count=new_df['airline_sentiment'].value_counts()
    Index = [1,2,3]
    plt.bar(Index,count, color=['blue', 'green', 'red'])
    plt.xticks(Index,['negative','neutral','positive'])
    plt.ylabel('Mood Count')
    plt.xlabel('Mood')
    plt.title('Count of Moods of '+i)


#  - United, US Airways, American substantially get negative reactions.
#  - Tweets for Virgin America are the most balanced.

# ### Most used words in Positive and Negative tweets 

# In[ ]:


from wordcloud import WordCloud,STOPWORDS


# - The goal is to firstly get an idea of the most frequent words in negative tweets.
# - Get idea about most frequent words in positive tweets.

# ### Wordcloud for Negative sentiments of tweets

# Wordcloud is a great tool for visualizing nlp data. The larger the words in the wordcloud image , the more is the frequency of that word in our text data.

# In[ ]:


new_df=df[df['airline_sentiment']=='negative']
words = ' '.join(new_df['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ### Wordcloud for positive reasons

# The code for getting positive sentiments is completely same with the one for negative sentiments. Just replace negative with positive in the first line.

# In[ ]:


new_df=df[df['airline_sentiment']=='positive']
words = ' '.join(new_df['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
plt.figure(1,figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ### Lets try and calculate the highest frequency words in postive sentimental tweets

# In[ ]:


# Calculate highest frequency words in positive tweets
def freq(str): 
  
    # break the string into list of words  
    str = str.split()          
    str2 = [] 
  
    # loop till string values present in list str 
    for i in str:              
  
        # checking for the duplicacy 
        if i not in str2: 
  
            # insert value in str2 
            str2.append(i)  
              
    for i in range(0, len(str2)): 
        if(str.count(str2[i])>50): 
            print('Frequency of', str2[i], 'is :', str.count(str2[i]))
        
print(freq(cleaned_word))


# * Words like **Thanks**, **best**, **customer** , **love**, **flying** , **good** are understandably present in the **most frequent** words of positive tweets. 
# * However, other than these, most of the words are stop words and need to be filtered. We will do so later.
# * Lets try and visualize the reasons for negative tweets first !!

# ### What are the reasons for negative sentimental tweets for each airline ?

# #### We will explore the **negative reason** column of our dataframe to extract conclusions about negative sentiments in the tweets by the customers 

# In[ ]:


#get the number of negative reasons
df['negativereason'].nunique()


NR_Count=dict(df['negativereason'].value_counts(sort=False))
def NR_Count(Airline):
    if Airline=='All':
        a=df
    else:
        a=df[df['airline']==Airline]
    count=dict(a['negativereason'].value_counts())
    Unique_reason=list(df['negativereason'].unique())
    Unique_reason=[x for x in Unique_reason if str(x) != 'nan']
    Reason_frame=pd.DataFrame({'Reasons':Unique_reason})
    Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x: count[x])
    return Reason_frame



def plot_reason(Airline):
    
    a=NR_Count(Airline)
    count=a['count']
    Index = range(1,(len(a)+1))
    plt.bar(Index,count, color=['blue','yellow','red','orange','black','brown','gray','cyan','purple','green'])
    plt.xticks(Index,a['Reasons'],rotation=90)
    plt.ylabel('Count')
    plt.xlabel('Reason')
    plt.title('Count of Reasons for '+Airline)
    
plot_reason('All')  
plt.figure(2,figsize=(15,15))
for i in airlines:
    indices= airlines.index(i)
    plt.subplot(2,3,indices+1)
    plt.subplots_adjust(hspace=0.9)
    plot_reason(i)


# - **Customer Service Issue** is the main neagtive reason for US Airways,United,American,Southwest,Virgin America
# - **Late Flight** is the main negative reason for Delta  
# - Interestingly, Virgin America has the least count of negative reasons (all less than 60)
# - Contrastingly to Virgin America, airlines like US Airways,United,American have more than 500 negative reasons (Late flight, Customer Service Issue)

# ### Is there a relationship between negative sentiments and date ?

# Our dataframe has data from **2015-02-17** to **2015-02-24**
# 
# It will be interesting to see if the date has any effect on the sentiments of the tweets(*especially negative !*). We can draw various conclusions by visualizing this.

# In[ ]:


date = df.reset_index()
#convert the Date column to pandas datetime
date.tweet_created = pd.to_datetime(date.tweet_created)
#Reduce the dates in the date column to only the date and no time stamp using the 'dt.date' method
date.tweet_created = date.tweet_created.dt.date
date.tweet_created.head()
df = date
day_df = df.groupby(['tweet_created','airline','airline_sentiment']).size()
# day_df = day_df.reset_index()
day_df


# This shows the sentiments of tweets for each date from **2015-02-17** to **2015-02-24** for every airline in our dataframe.
# 
# Our next step will be to plot this and get better visualization for negative tweets.

# In[ ]:


day_df = day_df.loc(axis=0)[:,:,'negative']

#groupby and plot data
ax2 = day_df.groupby(['tweet_created','airline']).sum().unstack().plot(kind = 'bar', color=['blue', 'green', 'red','yellow','orange','purple'], figsize = (15,8), rot = 70)
labels = ['American','Delta','Southwest','US Airways','United','Virgin America']
ax2.legend(labels = labels)
ax2.set_xlabel('Date')
ax2.set_ylabel('Negative Tweets')
plt.show()


# - Interestingly, **American** has a sudden upsurge in negative sentimental tweets on **2015-02-23**, which reduced to half the very next day **2015-02-24**. (*I hope American is doing better these days and resolved their Customer Service Issue as we saw before*)
# - **Virgin America** has the least number of negative tweets throughout the weekly data that we have. It should be noted that the total number of tweets for **Virgin America** was also significantly less as compared to the rest airlines, and hence the least negative tweets.
# - The negative tweets for all the rest airlines is slightly skewed towards the end of the week !

# ### Preprocessing the tweet text data

# Now, we will clean the tweet text data and apply classification algorithms on it

# In[ ]:


def tweet_to_words(tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 


# In[ ]:


df['clean_tweet']=df['text'].apply(lambda x: tweet_to_words(x))


# **The data is split in the standard 80,20 ratio.**

# In[ ]:


train,test = train_test_split(df,test_size=0.2,random_state=42)


# In[ ]:


train_clean_tweet=[]
for tweet in train['clean_tweet']:
    train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in test['clean_tweet']:
    test_clean_tweet.append(tweet)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(analyzer = "word")
train_features= v.fit_transform(train_clean_tweet)
test_features=v.transform(test_clean_tweet)


# ### Prediciting sentiments from tweet text data 

# - SVM(Support Vector Machine)
# - Decision Tree Classifier
# - Random Forest Classifier

# In[ ]:


Classifiers = [
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200)]


# In[ ]:


dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['airline_sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['airline_sentiment'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['airline_sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))
    print(classification_report(pred,test['airline_sentiment']))
    cm=confusion_matrix(pred , test['airline_sentiment'])
    plt.figure()
    
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)
    plt.xticks(range(3), ['Negative', 'Neutral','Positive'], fontsize=16,color='blue')
    plt.yticks(range(3), ['Negative', 'Neutral','Positive'], fontsize=16,color='blue')
    plt.show()


# - As we you can see above we have plotted the **Confusion Matrix** for predicted sentiments and actual sentiments (negative,neutral and positive)
# - **Random Forest Classifier** gives us the best accuracy score, precision scores according to the classification report.
# - The confusion matrix shows the TP,TN,FP,FN for all the 3 sentiments(negative,neutral and positive)
#   Here also **Random Forest Classifier** gives **better** results than the **Decision Tree Classifier** and **SVM**.
#   
#   

# In[ ]:


Index = [1,2,3]
plt.figure(figsize=(15,10))
plt.bar(Index,Accuracy,color=['orange','red','blue'])
plt.xticks(Index, Model,rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracies of Models')


# ## Thanks for stay tuned....
