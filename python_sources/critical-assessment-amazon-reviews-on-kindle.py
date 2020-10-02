#!/usr/bin/env python
# coding: utf-8

# # Amazon Reviews on kindle products

# ![alt text](https://i.guim.co.uk/img/media/f5da07b449b6dfe3891a8462b44f4050d272880b/0_0_3200_2360/master/3200.jpg?w=620&q=20&auto=format&usm=12&fit=max&dpr=2&s=211109be6ff35dad62c27fae8ff20797)
# 
# 

# ### The scope of the project is to explore the reviews submitted by users and understand in depth about the recommendations

# The outline for the project will be as follows- 
# 1. Understand and clean the data
#     - Check for null values
#     - Drop columns which arent useful
# 2. Speculate whether ratings are genuine ?
#     - what if the one user is trying to give all rating ?
#     - How will the distribution look for bulk users ?
#     - How many users are bulk ?
# 3. Find the NPS net promoter score of amazon
#     - What's NPS score ?
#     - How do we calculated for amazon  ?
# 4. Pick a product and deep dive
#     - We will pick one variation of kindle product drill & analyse its characteristics
# 5. [Paper white kindle] - NPS score  ? 
# 6. [Paper white kindle] - Plot time series for review
#     - How to handle date time text ?
#     - How to plot time series on a graph ? 
#     - How does the graph look like in small intervals of 5 days or 10 days or 30 days ?
#     - Did the performance (NPS) go up or down with time ?
# 7. [Paper white kindle] Predict Recommendations based on reviews content
#     - Make a clean function
#         - Remove punctuations
#         - Remove stopwords
#         - Stem vs Lemmatize
#     - Create a TFIDF vectorizer
#     - Create Features
#     - Understand and explore sentiment analysis
#         - Use compound feature
#     - Use RandomForestClassifier
#     - Check the score 
#         

# # 1. Understand and clean the data
# 
# - Check for null values
# - Drop columns which arent useful
# 

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


add = "../input/1429_1.csv"

reviews = pd.read_csv(add,low_memory=False)
reviews.columns = ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer','date', 'dateAdded', 'dateSeen',
       'didPurchase', 'doRecommend', 'id','numHelpful', 'rating', 'sourceURLs','text', 'title', 'userCity',
       'userProvince', 'username']


# In[ ]:


reviews.nunique()


# There are lot of null values and irrelevant columns 

# In[ ]:


reviews.isnull().sum()
#lets drop usernames, userProvince,id,didPurchase


# In[ ]:


reviews.drop(labels=['didPurchase','id','userCity','userProvince'],axis=1,inplace=True)


# In[ ]:


reviews.isnull().sum()


# # 2. Speculate whether ratings are genuine ?
# - what if the one user is trying to give all rating ?
# - How will the distribution look for bulk users ?
# - How many users are bulk ?

# ### Are the all the reviews given by same group of users ? 

# In[ ]:


rating_perperson=reviews.username.value_counts()
#ratings 
print ("Total ratings : " + str(sum(rating_perperson)))
print ("Total users : " + str(len(rating_perperson)))
print("Users giving bulk ratings (more than 10) : " + str(sum(rating_perperson >10)))
bulk = rating_perperson[rating_perperson >10]
bulk_rating = sum(bulk)
print ("Bulk ratings : " + str(bulk_rating))
print ("Populations of bulk ratings : " + str(bulk_rating*100/sum(rating_perperson)))
print ("Populations of bulk users : " + str(sum(rating_perperson >10)*100/len(rating_perperson)))
rating_perperson.value_counts().plot(kind='pie',figsize=(10,10), title='Ratings Per User')


# #### Although the pie chart reveals that most of the users have given single rating but its interesting to note following fact
#  #### 1 : Only 0.55 % of the users are bulk users
#  #### 2 : Around 9 % of the ratings have been submitted by just 0.55% users - Does it seem odd to you ?

# In[ ]:


reviews['bulk']= reviews['username'].apply(lambda x : 1 if x in bulk.index else 0)
#gives us the category whether a rating is bulk or not
from matplotlib import pyplot
#series.hist(by=series)
print(reviews.rating.hist(by=reviews.bulk))
print(reviews[reviews.bulk==1].rating.describe())
print(reviews[reviews.bulk==0].rating.describe())


# ### Well ! that picture says it all. Now we dont think that bulk users are spam since the have the same rating distribution as others

# ### Distribution of User rating

# In[ ]:


from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

star = reviews.rating.value_counts()
print("*** Rating distribution ***")
print(star)
star.sort_index(inplace=True)
star.plot(kind='bar',title='Amazon customer ratings',figsize=(6,6),style='Solarize_Light2')


# # 3. Find the NPS net promoter score of amazon
# - What's NPS score ?
# - How do we calculated for amazon  ?

# #### Looks like amazon is really good 

# # NPS Score ( Net promoter score ) 

# #### Net Promoters Score helps us evaluate customer satisfaction and loyalty
# 
# Rating 1,2,3 - Detractors <br>
# Rating 4   - Passive <br>
# Rating 5 - Promoters <br>
# 
# NPS = (Promoters - Detractors)/Total ratings * 100

# In[ ]:


NPS_score = round (100*((star.loc[5])-sum(star.loc[1:3]))/sum(star.loc[:]),2)
print (" NPS score of Amazon is : "  + str(NPS_score))


# # 4. Pick a product and deep dive
# - We will pick one variation of kindle product drill & analyse its characteristics

# Lets deep dive and pick product to analyse

# In[ ]:


kindle = reviews[reviews.name=='Amazon Kindle Paperwhite - eBook reader - 4 GB - 6 monochrome Paperwhite - touchscreen - Wi-Fi - black,,,']


# In[ ]:


kindle.isnull().sum()
# The dataset looks good to go


# # 5. [Paper white kindle] - NPS score  ? 

# In[ ]:


kindle_s = kindle.rating.value_counts()
kindle_s.sort_index(inplace=True)

Kindle_NPS_score = round (100*(kindle_s[5]-sum(kindle_s[1:3]))/sum(kindle_s),2)
print (" NPS score of Kindle is : "  + str(Kindle_NPS_score))
#better NPS than overall amazon
kindle_s.plot(kind='bar',title='Amazon customer ratings',figsize=(6,6),style='Solarize_Light2')


# ### What about recommendations ? How is rating related to recommendation ?

# In[ ]:


kindle.doRecommend.value_counts()


# In[ ]:


kindle.rating.hist(by=kindle.doRecommend,figsize=(12,6))


# In[ ]:


plus_kindle = kindle[kindle.doRecommend==True].rating.value_counts()
plus_kindle.sort_index(inplace=True)
recomm_NPS = round(100*(sum(plus_kindle[4:5])-sum(plus_kindle[1:2]))/sum(plus_kindle),2)
minus_kindle = kindle[kindle.doRecommend==False].rating.value_counts()
minus_kindle.sort_index(inplace=True)
notrecomm_NPS = round(100*(sum(minus_kindle[4:5])-sum(minus_kindle[1:2]))/sum(minus_kindle),2)
print("Those who recommend amazon kindle generate high NPS score of " + str(recomm_NPS))
print("Those who DO NOT recommend kindle produce a NPS score of " + str(notrecomm_NPS))
print(" ~ pretty much correct definition of NPS score")


# # 6. [Paper white kindle] - Plot time series for reviews 
# - How to handle date time text ?
# - How to plot time series on a graph ? 
# - How does the graph look like in small intervals of 5 days or 10 days or 30 days ?
# - Did the performance (NPS) go up or down with time ?
# 

# In[ ]:


kindle['temp'] = kindle.date.apply(lambda x : pd.to_datetime(x))
kindle_review_dates = kindle.date.value_counts()
kindle_review_dates.sort_index(inplace=True)
kindle_review_dates.plot(kind='area',figsize=(12,6))


# In[ ]:


rating_perdate = kindle_review_dates.sort_values(ascending=False)
peakrating = rating_perdate[:20]
peak_month=[]
for x in peakrating.index:
    peak_month.append(pd.to_datetime(x).month)
pd.Series(peak_month).value_counts()


# #### Insight 
# 1. January month has the highest number of peaks >> Activity is high >> More Sales during Jan ( We all know)
# 2. There is high degree of variance in reviews added over time
# 
# 

# In[ ]:


rating_series = pd.DataFrame(kindle.date)
dforms=[]
for x in rating_series.date:
    dforms.append((pd.to_datetime(x)).value)
# now we have dforms which has dates transformed to numeric values
rating2 = rating_series.assign(date_min = dforms)
rating2.reset_index(inplace=True)
#rating2.set_index('date_min')
#rating2.columns=['timestamp_string','review_count','date_min']
bins = np.linspace(min(rating2.date_min),max(rating2.date_min),num=50)
rating2.hist(column='date_min', bins=20,figsize=(10,6),)
rating2.hist(column='date_min', bins=30,figsize=(10,6))
rating2.hist(column='date_min', bins=50,figsize=(10,6))


# In[ ]:


def NPS_eval (A):
    score =0
    for x in A[:]:
        if (x>4) :
            score+=1
        elif (x<4) :
            score-=1
    return 100*score/len(A)    


# In[ ]:


NPS_overtime = kindle[['temp','rating']]
NPS_overtime.groupby(by='temp').agg(NPS_eval).plot(figsize=(15,10))


# In[ ]:


NPS_overtime['timeline']= NPS_overtime['temp'].apply(lambda x : (x.month+(12*(x.year-2015))))
NPS_by_month= NPS_overtime.groupby(by='timeline').agg(NPS_eval)
print(NPS_by_month.plot())
NPS_by_month.sort_values(by='rating')


# # 7. [Paper white kindle] Predict Recommendations based on reviews content
# - Make a clean function
#   - Remove punctuations
#   - Remove stopwords
#   - Stem vs Lemmatize
# - Create a TFIDF vectorizer
# - Create Features
# - Understand and explore sentiment analysis
#     - Use compound feature
# - Use RandomForestClassifier
# - Check the score 
#         

# ## Can we predict Recommendations with given comments on product ?

# In[ ]:


comments = pd.concat([kindle['text']+". "+ kindle['title'],kindle['rating'],kindle['doRecommend']],axis=1)
comments.columns=['text','rating','recommend']


# In[ ]:


import string
import nltk
from nltk import PorterStemmer
import re 

stopwords = nltk.corpus.stopwords.words('english')
ps = PorterStemmer()
wn = nltk.WordNetLemmatizer()


def clean_stem (sent): 
    temp1 ="".join(x for x in sent if x not in string.punctuation)
    temp2 = re.split('\W+',temp1.lower())
    temp3 = [ps.stem(x) for x in temp2 if x not in stopwords]
    return temp3

def clean_lemma (sent): 
    temp1 ="".join(x for x in sent if x not in string.punctuation)
    temp2 = re.split('\W+',temp1.lower())
    temp3 = [wn.lemmatize(x) for x in temp2 if x not in stopwords]
    return temp3

text="Hello this is, my happiest place. organize, organizes, and organizing in Happy world ! with happiness ..so much of happy!! "

print("Stemmed " + "-".join(clean_stem(text)))
print("Lemmatized " + "-".join(clean_lemma(text)))


# ### Lets create vectors from the text columns

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectstem = TfidfVectorizer(analyzer=clean_stem)
vectlemm = TfidfVectorizer(analyzer=clean_lemma)

textfeatures=vectstem.fit_transform(comments['text'])
print("Stemmed - " + str(len(vectstem.get_feature_names())))

vectlemm.fit_transform(comments['text'])
print("Lemmatized - " + str(len(vectlemm.get_feature_names())))


# Stemmed has features 18 % lower than that of Lemmatized. 
#     - In the above example in happy line, you can see how ineffective lemmatization can be 
#     - Thus we will be applying cleanstem algo here
#     - Lower features means more information density in the compressed columns

# ### Lets have a look on our stemmed data

# In[ ]:


pd.DataFrame(textfeatures.toarray()).head(15)


# The column names dont make sense - Need to update them with real words
#     - for this we use vectstem.vocabulary_ to modify the columns

# In[ ]:


textmatrix = pd.DataFrame(textfeatures.toarray(),columns=vectstem.vocabulary_)
textmatrix.head(5)


# In[ ]:


sum_scores = pd.DataFrame(textmatrix.sum(),columns=['sum_scores_TFIDF'])
sum_scores.head(10)


# In[ ]:


# Need to see most important words in the reviews
# words used by many people or less frequent in sentences
sum_scores.sort_values(by='sum_scores_TFIDF',ascending=True)[:5] 


# In[ ]:


#high usage of words in reviews
sum_scores.sort_values(by='sum_scores_TFIDF',ascending=False)[:5]


# ### Lets build features on our data
# 

# In[ ]:


pd.set_option('display.max_colwidth', 0)
comments.head()


# **What is sentiment analysis ? No idea ? 
# **Read the next code block

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

happy = "I am Happy. this is so awesome. I love life. I will be in heaven"
#when you find free food in university
print("happy " + str(sid.polarity_scores(text)))


sad = "i hate this. I am mad this is stupid. I will kill you"
#when your professor gives you a ZERO in assignment
print("sad " + str(sid.polarity_scores(sad)))

neut = "I will come. You should go. This is blue color"
#when you state facts and nothing else
print("dont care - " + str(sid.polarity_scores(neut)))

srishti = "money"
print("dss - " + str(sid.polarity_scores(srishti)))


# #### Understand the output
#     - sid.polarity is a dictionary
#     - pos and neg indicates - positive and negative emotions in sentence
#     - we should be interested in compund score which calculates the final effect
#    

# In[ ]:


# Feature 1 : Sentiment compound value
def sentiment(x):
    score = sid.polarity_scores(x)
    return score['compound']
    
#sentiment(happy)
comments['sentiment']= comments['text'].apply(lambda x : sentiment(x))


# In[ ]:


# Feature 2 : Length of string

comments['length'] = comments['text'].apply(lambda x : len(re.split('\W+',x)))
comments[comments['rating']==5].head(10)

# before we proceed - we need to convert all true >> 1 and false as 0
def convert(x):
    
    if x==True:
        return 1
    else :
        return 0
    
print(convert("False"))

comments['target_rec'] = comments['recommend'].apply(lambda x : convert(x))
comments.head(5)


# In[ ]:


comments[comments['rating']==1].head(5)


# ### Lets predict recommendation !

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

# need to reset index of the comments column to match with textfeatures
new_sentiment = comments.sentiment.reset_index()['sentiment']
new_length = comments.length.reset_index()['length']

x_features = pd.concat([new_sentiment,new_length,
                        pd.DataFrame(textfeatures.toarray(),
                        columns=vectstem.vocabulary_)],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_features,comments.target_rec,test_size=0.2)

rf = RandomForestClassifier(n_jobs=-1,n_estimators=50,max_depth=90)
rfmodel=rf.fit(x_train,y_train)

y_pred = rfmodel.predict(x_test)
sorted(zip(rfmodel.feature_importances_,x_train.columns),reverse=True)[0:10]


# In[ ]:


precision, recall, fscore , support = score(y_test,y_pred,average='binary')
print('Precision: {} / Recall :{} / Accuracy {} '.format(round(precision,3),
                                                         round(recall,3),
                                                         round((y_pred==y_test).sum()/len(y_test),3)))


# ## CHEERS !
