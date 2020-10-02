#!/usr/bin/env python
# coding: utf-8

# # Drug Sentiment Analysis

# ## Problem Statement
# The dataset provides patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient satisfaction. We have to create a target feature out of ratings and predict the sentiment of the reviews.

# ### Data Description :
# The data is split into a train (75%) a test (25%) partition.
# 
# * drugName (categorical): name of drug
# * condition (categorical): name of condition
# * review (text): patient review
# * rating (numerical): 10 star patient rating
# * date (date): date of review entry
# * usefulCount (numerical): number of users who found review useful
# 
# The structure of the data is that a patient with a unique ID purchases a drug that meets his condition and writes a review and rating for the drug he/she purchased on the date. Afterwards, if the others read that review and find it helpful, they will click usefulCount, which will add 1 for the variable.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Import all the necessary packages
# Here we have imported the basic packages that are required to do basic processing. Feel free to use any library that you think can be useful here.

# In[ ]:


#import all the necessary packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')


# ### Load Data

# In[ ]:


#read the train and test data

test = pd.read_csv('/kaggle/input/kuc-hackathon-winter-2018/drugsComTest_raw.csv') #train data
train = pd.read_csv('/kaggle/input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv') #test data


# ### Checking Out The Data

# In[ ]:


#check the head of train data
train.head(10)


# In[ ]:


#check the head of test data
test.head(10)


# By looking at the head of train and test data we see that there are 7 features in our Dataset but we don't have any sentiment feature which can serve as our target variable. We will make a target feature out of rating. If Rating is greater than 5 we will assign it as positive else we will assign it as negative.

# In[ ]:


#check the shape of the given dataset
print(f'train has {train.shape[0]} number of rows and {train.shape[1]} number of columns')
print(f'train has {test.shape[0]} number of rows and {test.shape[1]} number of columns')


# In[ ]:


#check the columns in train
train.columns


# ## Exploratory Data Analysis

# The purpose of EDA is to find out interesting insights and irregularities in our Dataset. We will look at Each feature and try to find out interesting facts and patterns from them. And see whether there is any relationship between the variables or not.

# Merge the train and test data as there are no target labels. We will perform our EDA and Pre-processing on merged data. Then we will dive the data into 70 : 30 ratio for training and testing

# In[ ]:


#merge train and test data

merge = [train,test]
merged_data = pd.concat(merge,ignore_index=True)

merged_data.shape   #check the shape of merged_data


# ### Check number of uniqueIds to see if there's any duplicate record in our dataset

# In[ ]:


#check uniqueID
merged_data['uniqueID'].nunique()


# There are 215063 uniqueIds meaning that every record is unique.

# ### Check information of the merged data

# In[ ]:


merged_data.info()


# ### Check the Description

# In[ ]:


merged_data.describe(include='all')


# **Following things can be noticed from the description**
# * Top **drugName** is **Levonorgestrel**, It will be intresting to see for what condition it is used.
# * Top **condition** is **Birth Control**.
# * Top **review** is just a single word "Good", but it has very small count - 39. May be lazy people like me have written that comment.
# * Most single day review came on 1-Mar-16, it will be interesting to investigate this date and see for which drugName and which conditions these reviews were for.

# ### Check the percentage of null values in each column

# In[ ]:


merged_data.isnull().sum()/merged_data.shape[0]


# We just have null values in just 1 column i.e **condition** . We will leave the null values in that column for now as the null values are very small.

# ### Check number of unique values in drugName and condition

# In[ ]:


#check number of unique values in drugName
print(merged_data['drugName'].nunique())

#check number of unique values in condition
print(merged_data['condition'].nunique())


# We can see that there are 3671 drugName and only 916 conditions. So there are conditions which has multiple drugs.

# Now the time is to plot some beautiful graphs and find some interesting insights from our Data. **Here your detective skills are needed so be ready and interrogate the data as much as you can ** 

# ### Check the top 20 conditions

# In[ ]:


#plot a bargraph to check top 20 conditions
plt.figure(figsize=(12,6))
conditions = merged_data['condition'].value_counts(ascending = False).head(20)

plt.bar(conditions.index,conditions.values)
plt.title('Top-20 Conditions',fontsize = 20)
plt.xticks(rotation=90)
plt.ylabel('count')
plt.show()


# **From above graph we can see that the :**
# * Birth control is twice as big as anyone, around 38,000.
# * Most of the conditions for top 20 conditions are between 5000 - 10000 

# ### Plot the bottom 20 conditions

# In[ ]:


#plot a bargraph to check bottom 20 conditions
plt.figure(figsize=(12,6))
conditions_bottom = merged_data['condition'].value_counts(ascending = False).tail(20)

plt.bar(conditions_bottom.index,conditions_bottom.values)
plt.title('Bottom-20 Conditions',fontsize = 20)
plt.xticks(rotation=90)
plt.ylabel('count')
plt.show()


# * Bottom 20 conditions have just single counts in our dataset. They may be the rare conditions.
# * And if we look at our plot we see that there are conditions whose name are strange starting with **"61<_/span_>users found this comment helpful"** , these are the noise present in our data. We will deal with these noise later.

# ### Check top 20 drugName

# In[ ]:


#plot a bargraph to check top 20 drugName
plt.figure(figsize=(12,6))
drugName_top = merged_data['drugName'].value_counts(ascending = False).head(20)

plt.bar(drugName_top.index,drugName_top.values,color='blue')
plt.title('drugName Top-20',fontsize = 20)
plt.xticks(rotation=90)
plt.ylabel('count')
plt.show()


# * The top drugName is Levonorgestrel, which we had seen in description as well.
# * The top 3 drugName has count around 4000 and above. 
# * Most of the drugName counts are around 1500 if we look at top 20

# ### Check bottom 20 drugName

# In[ ]:


#plot a bargraph to check top 20 drugName
plt.figure(figsize=(12,6))
drugName_bottom = merged_data['drugName'].value_counts(ascending = False).tail(20)

plt.bar(drugName_bottom.index,drugName_bottom.values,color='blue')
plt.title('drugName Bottom-20',fontsize = 20)
plt.xticks(rotation=90)
plt.ylabel('count')
plt.show()


# * The bottom 20 drugName has count 1. These might be the drugs used of rare conditions or are new in market.

# ### Checking Ratings Distribution

# In[ ]:


ratings_ = merged_data['rating'].value_counts().sort_values(ascending=False).reset_index().                    rename(columns = {'index' :'rating', 'rating' : 'counts'})
ratings_['percent'] = 100 * (ratings_['counts']/merged_data.shape[0])
print(ratings_)


# In[ ]:


# Setting the Parameter
sns.set(font_scale = 1.2, style = 'darkgrid')
plt.rcParams['figure.figsize'] = [12, 6]

#let's plot and check
sns.barplot(x = ratings_['rating'], y = ratings_['percent'],order = ratings_['rating'])
plt.title('Ratings Percent',fontsize=20)
plt.show()


# We notice that most of the ratings are high with ratings 10 and 9.
# rating 1 is also high which shows the extreme ratings of the user. We can say that the users mostly prefer to rate when the drugs are either very useful to them or the drugs fails, or there is some side effects. About 70% of the values have rating greater than 7.

# ### Check the distribution of usefulCount

# In[ ]:


#plot a distplot of usefulCount
sns.distplot(merged_data['usefulCount'])
plt.show()


# * usefulCount is positively-skewed.
# * Most of the usefulCounts are distributed between 0 and 200.
# * There are extreme outliers present in our usefulCounts. We either have to remove them or transform them.

# In[ ]:


#check the descriptive summary
sns.boxplot(y = merged_data['usefulCount'])
plt.show()


# We can see that there are huge outliers present in our dataset. Some drugs have extreme useful counts.

# ### Check number of Drugs per condition

# In[ ]:


#lets check the number of drugs/condition
merged_data.groupby('condition')['drugName'].nunique().sort_values(ascending=False).head(20)


# If we look above the top value is not listed/othe. 
# * It might be possible that the user didn't mentioned his/her condition as sometimes people doesn't want to reveal thier disorders. We can look up the drug names and fill up the conditions for which that drug is used.
# 
# * Another point to note here is that there are values is condition like **'3 <_/span_> user found this comment helpful'**, **4<_/span_> users found this comment helpful**. These are the noises present in our dataset. The dataset appears to have been extracted through webscraping, the values are wrongly fed in here.

# ##### Let's look at ''3 <_/span_> user found this comment helpful'

# In[ ]:


span_data = merged_data[merged_data['condition'].str.contains('</span>',case=False,regex=True) == True]
print('Number of rows with </span> values : ', len(span_data))
noisy_data_ = 100 * (len(span_data)/merged_data.shape[0])
print('Total percent of noisy data {} %  '.format(noisy_data_))


#  There are only 0.54 % values with  </span  type data. We can remove these from our dataset as we won't lose much information by removing them.

# In[ ]:


#drop the nosie 
merged_data.drop(span_data.index, axis = 0, inplace=True)


# ### Now let's look at the not listed/other

# In[ ]:


#check the percentage of 'not listed / othe' conditions
not_listed = merged_data[merged_data['condition'] == 'not listed / othe']
print('Number of not_listed values : ', len(not_listed))
percent_not_listed = 100 * len(not_listed)/merged_data.shape[0]
print('Total percent of noisy data {} %  '.format(percent_not_listed))


# There are 592 unique drugs for "not / listed othe "  values. There are 2 options  to deal with these values  
# 1. Check the condition associated with the drugs and replace the values.
# 2. We can drop the values as these only accounts for 0.27 % of total data. To save our time we will drop the nosiy data.

# In[ ]:


# drop noisy data
merged_data.drop(not_listed.index, axis = 0, inplace=True)


# In[ ]:


# after removing the noise, let's check the shape
merged_data.shape[0]


# ### Now Check number of drugs present per condition after removing noise

# In[ ]:


#lets check the number of drugs present in our dataset condition wise
conditions_gp = merged_data.groupby('condition')['drugName'].nunique().sort_values(ascending=False)

#plot the top 20
# Setting the Parameter
condition_gp_top_20 = conditions_gp.head(20)
sns.set(font_scale = 1.2, style = 'darkgrid')
plt.rcParams['figure.figsize'] = [12, 6]
sns.barplot(x = condition_gp_top_20.index, y = condition_gp_top_20.values)
plt.title('Top-20 Number of drugs per condition',fontsize=20)
plt.xticks(rotation=90)
plt.ylabel('count',fontsize=10)
plt.show()


# * Most of the drugs are for pain, birth control and high blood pressure which are common conditions.
# * In top- 20 each condition has above 50 drugs.

# ### Check bottom 20 drugs per conditions

# In[ ]:


#bottom-20
condition_gp_bottom_20 = conditions_gp.tail(20)
#plot the top 20

sns.barplot(x = condition_gp_bottom_20.index, y = condition_gp_bottom_20.values,color='blue')
plt.title('Bottom-20 Number of drugs per condition',fontsize=20)
plt.xticks(rotation=90)
plt.ylabel('count',fontsize=10)
plt.show()


# Bottom-20 conditions just have single drugs. These are the rare conditions.

# ### Now let's check if a single drug can be used for Multiple conditions

# In[ ]:


#let's check if a single drug is used for multiple conditions
drug_multiple_cond = merged_data.groupby('drugName')['condition'].nunique().sort_values(ascending=False)
print(drug_multiple_cond.head(10))


# There are many drugs which can be used for multiple conditions. 

# ### Check the number of drugs with rating 10

# In[ ]:


#Let's check the Number of drugs with rating 10.
merged_data[merged_data['rating'] == 10]['drugName'].nunique()


# We have 2907 drugs with rating 10.

# ### Plot top-20 drugs with rating 10

# In[ ]:


#Check top 20 drugs with rating=10/10
top_20_ratings = merged_data[merged_data['rating'] == 10]['drugName'].value_counts().head(20)
sns.barplot(x = top_20_ratings.index, y = top_20_ratings.values )
plt.xticks(rotation=90)
plt.title('Top-20 Drugs with Rating - 10/10', fontsize=20)
plt.ylabel('count')
plt.show()


# * We can see that Levonorgestrel has most of the ratings 10/10. It seems it is used for the common condition and, it would be the most effective one.
# * Other drugs have ratings between 1000 and 500 from top-20 10/10.

# ### Check for what condition Levonorgestrel is used for

# In[ ]:


merged_data[merged_data['drugName'] == 'Levonorgestrel']['condition'].unique()


# Levonorgestrel is used for 3 different conditions. 
# * emergency contraception
# * birth control
# * abnormal uterine bleeding

# ### Top 10 drugs with 1/10 Rating

# In[ ]:


#check top 20 drugs with 1/10 rating

top_20_ratings_1 = merged_data[merged_data['rating'] == 1]['drugName'].value_counts().head(20)
sns.barplot(x = top_20_ratings_1.index, y = top_20_ratings_1.values )
plt.xticks(rotation=90)
plt.title('Top-20 Drugs with Rating - 1/10', fontsize=20)
plt.ylabel('count')
plt.show()


# Top-3 of 1/10 ratings have almost 700 counts. Which means they are not so useful drugs.

# ### Now we will look at the Date column

# In[ ]:


# convert date to datetime and create year andd month features

merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data['year'] = merged_data['date'].dt.year  #create year
merged_data['month'] = merged_data['date'].dt.month #create month


# ### Check Number of reviews per year

# In[ ]:


#plot number of reviews year wise
count_reviews = merged_data['year'].value_counts().sort_index()
sns.barplot(count_reviews.index,count_reviews.values,color='blue')
plt.title('Number of reviews Year wise')
plt.show()


# The year 2015, 2016 and 2017 accounts for the most reviews. Almost 60% of the reviews are from these years.

# ### Check average rating per year

# In[ ]:


#check average rating per year
yearly_mean_rating = merged_data.groupby('year')['rating'].mean()
sns.barplot(yearly_mean_rating.index,yearly_mean_rating.values,color='green')
plt.title('Mean Rating Yearly')
plt.show()


# * Rating has been almost constant from year 2009 - 2014 but after 2014 the ratings has started to decrease.
# * As the number of reviews has increased for last 3 years, the rating has decreased.

# ### Per year drug count and Condition count

# In[ ]:


#check year wise drug counts and year wise conditions counts

year_wise_condition = merged_data.groupby('year')['condition'].nunique()
sns.barplot(year_wise_condition.index,year_wise_condition.values,color='green')
plt.title('Conditions Year wise',fontsize=20)
plt.show()


# * Condition has increased in last 3 years. Which means the new conditions has been coming up.
# * Starting year 2008 had lowest number of conditions. 

# **We expect that as the the conditions has increased. Drugs should have also increased. Let's check that out.**

# In[ ]:


#check drugs year wise

year_wise_drug = merged_data.groupby('year')['drugName'].nunique()
sns.barplot(year_wise_drug.index,year_wise_drug.values,color='green')
plt.title('Drugs Year Wise',fontsize=20)
plt.show()


# As expected number of drugs has also increased in last three years.

# ## Data Pre-Processing

# Data Pre-processing is a vital part in model building. **"Garbage In Garbage Out"**, we all have heard this statement. But what does it mean. It means if we feed in garbage in our data like missing values, and different features which doesn't have any predictive power and provides the same information in our model. Our model will be just making a random guess and it won't be efficient enough for us to use it for any predictions.

# In[ ]:


# check the null values
merged_data.isnull().sum()


# We only have null values in condition. We will drop the records with null values as it only accounts for 0.5 % of total data.

# In[ ]:


# drop the null values
merged_data.dropna(inplace=True, axis=0)


# ### Pre-Processing Reviews

# **Check the first few reviews**

# In[ ]:


#check first three reviews
for i in merged_data['review'][0:3]:
    print(i,'\n')


# ### Steps for reviews pre-processing.
# * **Remove HTML tags**
#      * Using BeautifulSoup from bs4 module to remove the html tags. We have already removed the html tags with pattern "64</_span_>...", we will use get_text() to remove the html tags if there are any.
# * **Remove Stop Words**
#      * Remove the stopwords like "a", "the", "I" etc.
# * **Remove symbols and special characters**
#      * We will remove the special characters from our reviews like '#' ,'&' ,'@' etc.
# * **Tokenize**
#      * We will tokenize the words. We will split the sentences with spaces e.g "I might come" --> "I", "might", "come"
# * **Stemming**
#      * Remove the suffixes from the words to get the root form of the word e.g 'Wording' --> "Word"

# In[ ]:


#import the libraries for pre-processing
from bs4 import BeautifulSoup
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stops = set(stopwords.words('english')) #english stopwords

stemmer = SnowballStemmer('english') #SnowballStemmer

def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stops]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(stemming_words))


# In[ ]:


#apply review_to_words function on reviews
merged_data['review'] = merged_data['review'].apply(review_to_words)


# ### Now we will create our target variable "Sentiment" from rating

# In[ ]:


#create sentiment feature from ratings
#if rating > 5 sentiment = 1 (positive)
#if rating < 5 sentiment = 0 (negative)
merged_data['sentiment'] = merged_data["rating"].apply(lambda x: 1 if x > 5 else 0)


# We will predict the sentiment using the reviews only. So let's start building our model.

# ## Building Model

# In[ ]:


#import all the necessary packages

from sklearn.model_selection import train_test_split #import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #import TfidfVectorizer 
from sklearn.metrics import confusion_matrix #import confusion_matrix
from sklearn.naive_bayes import MultinomialNB #import MultinomialNB
from sklearn.ensemble import RandomForestClassifier  #import RandomForestClassifier


# We all know that we cannot pass raw text features in our model. We have to convert them into numeric values. We will use TfidfVectorizer to convert our reviews in numeric features.

# ### TfidfVectorizer (Term frequency - Inverse document frequency)
# **TF - Term Frequency** :- 
# 
# How often a term t occurs in a document d.
# 
# TF = (_Number of occurences of a word in document_) / (_Number of words in that document_)
# 
# **Inverse  Document Frequency**
# 
# IDF = log(Number of sentences / Number of sentence containing word)
# 
# **Tf - Idf = Tf * Idf**
# 

# In[ ]:


# Creates TF-IDF vectorizer and transforms the corpus
vectorizer = TfidfVectorizer()
reviews_corpus = vectorizer.fit_transform(merged_data.review)
reviews_corpus.shape


# We have built reviews_corpus which are the independent feature in our model. 

# ### **Store Dependent feature in sentiment and split the Data into train and test**

# In[ ]:


#dependent feature
sentiment = merged_data['sentiment']
sentiment.shape


# In[ ]:


#split the data in train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(reviews_corpus,sentiment,test_size=0.33,random_state=42)
print('Train data shape ',X_train.shape,Y_train.shape)
print('Test data shape ',X_test.shape,Y_test.shape)


# ### Apply Multinomial Naive Bayes

# In[ ]:


#fit the model and predicct the output

clf = MultinomialNB().fit(X_train, Y_train) #fit the training data

pred = clf.predict(X_test) #predict the sentiment for test data

print("Accuracy: %s" % str(clf.score(X_test, Y_test))) #check accuracy
print("Confusion Matrix") 
print(confusion_matrix(pred, Y_test)) #print confusion matrix


# We have got accuracy score of 75.8% by using NaiveBayes

# ### Apply RandomForestClassifier

# In[ ]:


#fit the model and predicct the output

clf = RandomForestClassifier().fit(X_train, Y_train)

pred = clf.predict(X_test)

print("Accuracy: %s" % str(clf.score(X_test, Y_test)))
print("Confusion Matrix")
print(confusion_matrix(pred, Y_test))


# ## Parameter Tuning
# Try different sets of parameters for RandomForestClassifier using RandomSearchCV and check which sets of parameters gives the best accuracy. *A task for you to try*

# ## Conclusion 
# After applying the TfidfVectorizer to transform our reviews in Vectors and applying NaiveBayes and RandomForestClassifier we see that RandomForestClassifier outperforms MulinomialNB. We have achieved accuracy of 89.7 % after applying RandomForestClassifier without any parameter tuning. We can tune the parameters of our classifier and improve our accuracy.

# In[ ]:


#Refrences - https://www.kaggle.com/chocozzz/recommendation-medicines-by-using-a-review
            #https://www.kaggle.com/sumitm004/eda-and-sentiment-analysis


# 
