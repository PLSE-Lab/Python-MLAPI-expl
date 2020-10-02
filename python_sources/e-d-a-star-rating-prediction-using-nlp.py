#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# # Approach

# Our data is of mixed type so we will try these two things
# 1. Firstly we will try to find out some insights using some basic statistics and graphs.
# 2. Secondly we will try to create some new meaningful variables for analysis.
# 3. At last we wll try to find out some insights using text data by forming wordcloud.

# # Importing necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 


# # Importing Dataset

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Let's check what is inside the train data

# In[ ]:


train.head()


# # Let's check what is inside the test data

# In[ ]:


test.head()


# # Checking number of rows and coloumns in train and test data

# In[ ]:


print(train.shape)
print(test.shape)


# # Checking if there are some missing values or not in train and test data.

# In[ ]:


print(train.isnull().sum())
print(test.isnull().sum())


# 1. It can be easily seen that App version is not available of many users.
# 2. And also many users havn't written Title for their review.

# # Is there any duplicate values?

# In[ ]:


dup = train.duplicated().sum()
dup1 = test.duplicated().sum()
print("Number of duplicate values in train data : " + str(dup))
print("Number of duplicate values in test data : " + str(dup))


# # Let's figure out how many users give title to their reviews.

# In[ ]:


rev = train['Review Title'].count()
rev1 = test['Review Title'].count()
print(" Total no. of people who gave Review title in train data are " + str(rev))
print(" Total no. of people who gave Review title in test data are " + str(rev1))


# # Let's see the Average Star rating

# In[ ]:


avg = train['Star Rating'].mean()
Avg = round(avg,1)
print("Average rating given by users is " + str(Avg))


# # Checking App Version and creating new variable of version for analysis.

# In[ ]:


oldest = train['App Version Name'].min()
latest = train['App Version Name'].max()

print("Oldest App Version is " + str(oldest))
print("Latest App version is " + str(latest))

train.loc[(train['App Version Name']<2),'App Version'] = 1      #Version1
train.loc[(train['App Version Name']>=2) ,'App Version'] = 2    #Version2


# Since niki has created only 2 versions of app so i have created a new variable with 2 discrete versions which is easy to analyse

# # Checking if new variable is added succesfully or not

# In[ ]:


train.head()


# # Number of users based on app version.
# (I AM NOT CONSIDERING THE NAN VALUES NEITHER I AM REMOVING THEM FOR NOW)

# In[ ]:


sns.countplot("App Version", data = train)
plt.ylabel('Number of Users')
plt.show()


# App Version 1 was not popular but in version 2 significant number of customers joined niki.

# #  Number of Lovers and haters which belongs to niki

# In[ ]:


sns.countplot(x="Star Rating" ,data=train)
plt.show()


# Oh... We do have a lot of unsatisfied users who are not loving us. It's time to improve.

# # Haters and lovers based on App Version

# In[ ]:


sns.countplot(x="Star Rating", hue = 'App Version', data=train)
plt.show()


# Ahhh.. During App Version 1 niki was having somehow similar number of haters and lovers.
# 
# In App version 2 niki is still having alot of unsatisfied customers. But a slight improvement is there from Version 1

# # Number of people who have spent their time on writing "Review Title"

# In[ ]:


df = train[['Review Title','Star Rating']]  #Creating a new dataframe with Review title and rating variable
df1 = df.dropna()                           #Removing null values
sns.countplot('Star Rating', data = df1)
plt.show()


# 1. We can clearly see that users who gave 1 star as well as 5 stars always write "Review Title".
# 2. This can affect our review page alot on website because title fits into eyes of reader first.

# # Here we will see how much long review and title users have given. 
# # Also we will find it's relation with Star Rating

# 1. Firstly we will create the length of review, Title and store it into new variable.
# 2. Secondly we will analyse the length of review, title and also analyse w.r.t. Star Rating.

# In[ ]:


train["Review_Length"]= train["Review Text"].str.len()     #Calculating and storing review's length
train["Title_Length"] = train["Review Title"].str.len()    #Calculating and storing title's review


# # Review Length Frequency

# In[ ]:


sns.distplot(train['Review_Length'].dropna())
plt.show()


# 1. Most of the people write review under 200 words.
# 2. We can see that generally users write review of about 50-100 words.

# # Title Length Frequency

# In[ ]:


sns.distplot(train['Title_Length'].dropna())
plt.show()


# 1. Most of the user write Title under 50 words.
# 2. We can see that generally users write Title of about 20-25 words.

# # Review Length vs Star Rating

# In[ ]:


plt.scatter(train['Review_Length'], train['Star Rating'])
plt.title('Review_Length vs Star Rating')
plt.xlabel('Review Length')
plt.ylabel('Star Rating')
plt.show()
print("Review Length to Star Rating Correlation:",train['Star Rating'].corr(train['Review_Length']))


# This shows us that if our length of review increases then our star rating will decrease because of negative correlation.

# # Title Length vs Star Rating

# In[ ]:


plt.scatter(train['Title_Length'], train['Star Rating'])
plt.title('Title_Length vs Star Rating')
plt.xlabel('Title Length')
plt.ylabel('Star Rating')
plt.show()
print("Title Length to Star Rating Correlation:",train['Star Rating'].corr(train['Title_Length']))


# This shows us that if our length of title increases then our star rating will decrease because of negative correlation but not too much like review length

# # Now it's time to see what users say about niki in their reviews.

# Finding out what most of the people say in review of training and testing data.
# For this a little cleaning is done as per follow:
# 1. Each review is converted into string.
# 2. Each review is broken down into small words.
# 3. Each word is converted to lower case.
# 4. Removing stopwords with the help of library (stopword example:- not, the, in, etc) and creating a wordcloud

# # Training Data

# In[ ]:


comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in train["Review Text"]: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (9, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# According to this worcloud it can be seen that most of the people maybe use it for
# 1. recharge
# 2. payment
# 3. cashbacks
# 4. electricity bill etc.
# 
# Some good words are used very frequently like:
# Nice, Good, awesome which shows that overall experience among the customers is descent.

# # Testing Data

# In[ ]:


comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in test["Review Text"]: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (9, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# 1. This worcloud is also somehow similar to training data's wordcloud.
# 2. Most of the people are talking about (offer, recharge, payment, good, cashback etc)

# # Let's figure out what users have written down in the "Review Title"

# We have already seen that there are many people who gave 1 star. To confirm it let us create a wordcloud which will also create negative comments.

# In[ ]:


f = train['Review Title'].dropna() #Extracting coloumn and removing null values from train data.
g = test['Review Title'].dropna()  #Extracting coloumn and removing null values from test data.


# # Training Data

# In[ ]:


comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in f: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (9, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# It is clearly visible that there are negative as well as positive titles available
# Positive words :- Good, Awesome, Helpful, Nice app etc
# 
# Negative words :- Worst, Useless, Fake App etc.
# 
# These negative title can affect new customers and they are large in number so have to do something about this.

# # Test Data

# In[ ]:


comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in g: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (9, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# It is clearly visible that there are negative as well as positive titles available
# Positive words :- Good, Great, Nice app etc
# 
# Negative words :- Worst, Useless, Fake, bakwas etc.
# 
# These negative title can aafect new customers and they are large in number so have to do something about this.

# # Rating Prediction using Naive Bayes Algorithm

# # Approach

# The whole process is divided into 4 parts.
# 1. Data Prepration
# 2. Feature Engineering
# 3. Model Training
# 4. Model Evaluation & Testing
# 
# First of all we will import our data and then we will create a new dataframe with only 2 useful variables i.e. 'Review Text & 'Star Rating'. We will not take 'Review Title' because more than 80% data is not available for that variable. Now we will process data so that it can ingest into ML model. So, firstly we will convert our reviews into lower form then we will remove punctuation then stopwards and then digits. Now we will lemmatize the data which will convert each word into it's root form.
# (eg. studies :- study).
# 
# Now we will create a tf - idf vector of each review because machine can only understand numerical data. After that we will ingest this data to Naive Bayes Algorithm for prediction.

# # Pros in data

# 1. All the review's are available which are useful in prediction.

# # Cons in data

# 1. The distribution of 'Star Rating' is not equal. (1 and 5 Star are more in number and others are less so it will affect predictions).
# 2. Review Titles are not available of all the users.
# 3. Some reviews are in native language.
# 4. Spelling Mistakes

# # Importing Necessary Libraries

# In[ ]:


import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.metrics import f1_score, accuracy_score
from sklearn import model_selection, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")


# # Importing Dataset

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Data Overview

# In[ ]:


train.head()


# # Creating new dataframe with useful variables

# In[ ]:


df = train[['Review Text', 'Star Rating']]
df.head()


# # Checking and removing null values

# In[ ]:


print(df.isnull().sum())
print('---------')
df1 = df.dropna()   # Creating new dataframe without null values
print(df1.isnull().sum())


# # Conerting to lower case

# In[ ]:


df1['Cleaned'] = df1['Review Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df1.head()


# # Removing punctuation

# In[ ]:


df1['Cleaned'] = df1['Cleaned'].str.replace('[^\w\s]','')
df1.head()


# # Removing Stopwords

# In[ ]:


df1['Cleaned'] = df1['Cleaned'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df1.head()


# # Removing Digits

# In[ ]:


df1['Cleaned'] = df1['Cleaned'].str.replace('\d+', '')
df1.head()


# # Lemmatizing

# In[ ]:


lemmatizer = WordNetLemmatizer()
df1['Cleaned'] = [lemmatizer.lemmatize(row) for row in df1['Cleaned']]
df1.head()


# # Creating dependent and independent variable

# In[ ]:


x = df1['Cleaned']       # Independent Variable
y = df1['Star Rating']   # Dependent Variable


# # Splitting data into training and validation

# In[ ]:


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(x,y, random_state = 2)


# # Creating tf-idf vector with ngram

# In[ ]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=5000)
tfidf_vect_ngram.fit(x)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


# # Building and validating model

# In[ ]:


nb = naive_bayes.MultinomialNB(alpha = 0.6)

model = nb.fit(xtrain_tfidf_ngram, train_y)

pred = model.predict(xvalid_tfidf_ngram)

acc = accuracy_score(valid_y,pred)

print('Accuracy of validation set is :', acc)


# # Weighted F score

# In[ ]:


score = f1_score(valid_y, pred, average='weighted')
print("Weighted F score is ",score)


# # Testing Data

# # Creating data frame with id, review and no null values

# In[ ]:


df2 = test[['id','Review Text']]
df3 = df2.dropna()              #Removing null values
df3.head()


# # Cleaning of testing data

# In[ ]:


#To lower case
df3['Cleaned'] = df3['Review Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#Removing punctuation
df3['Cleaned'] = df3['Cleaned'].str.replace('[^\w\s]','')

#Removing Stopwords
df3['Cleaned'] = df3['Cleaned'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Removing digits
df3['Cleaned'] = df3['Cleaned'].str.replace('\d+', '')

#Lemmatizing
lemmatizer = WordNetLemmatizer()
df3['Cleaned'] = [lemmatizer.lemmatize(row) for row in df3['Cleaned']]
df3.head()


# # Creating independent variable

# In[ ]:


x1 = df3['Cleaned']


# # Creating tf-idf vector with ngram

# In[ ]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=5000)
tfidf_vect_ngram.fit(x1)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(x1)


# # Predicting and merging Values to dataest

# In[ ]:


test_pred = model.predict(xtest_tfidf_ngram)
df3['Star Rating'] = test_pred
df4 = df3[['id','Star Rating']]
df4.to_csv("predictions.csv")

