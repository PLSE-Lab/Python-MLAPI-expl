#!/usr/bin/env python
# coding: utf-8

# # Kindle PaperWhite 

# ![alt text](https://i.imgflip.com/2defis.jpg)

# This project focusses on the following areas :
#     
#    - Analysis of the dataset
#    - Understanding of the User's Rating Distribution
#    - Predict Recommend Status based on the subjective review provided by the user

# ### Approach

# 1. Clean the Dataset
#     - Clean Column names 
#     - Clean Categories
#     - Clean Keys
# 2. Analysis of Data
# 3. Transforming Date Time
#     - Parse ReviewDate to [Date and Time]
#     - Parse ReviewDateAdded to [Date and Time]
#     - Parse ReviewDateSeen to [Date and Time]
# 4. Likert Scale Analysis : 
#     - 5 Point NPS Breakdown
#     - Ratings from 0-3  :  Detractors
#     - Ratings from 4  : Passive
#     - Ratings from 5  : Promoter
# 5. Feature Engineering
# 6. Apply NLTK - Sentiment Analysis to find Compound Score
# 7. Understanding the Fearures Added
# 8. Using TF-IDF and Random Forest to predict Recommendation Status

# # 1. Cleaning the Dataset
# 

# In[ ]:



import pandas as pd
add = "../input/1429_1.csv"
df = pd.read_csv(add)
df1_oasis = df.iloc[2816:3482,]
df2_fire_16gb = df.iloc[14448:15527,]
df3_paperwhite_4gb = df.iloc[17216:20392,]
df4_voyage = df.iloc[20392:20972,]
df5_paperwhite = df.iloc[20989:21019,]
#df.head(5)


# #### Exploring individual Dataframes

# In[ ]:


print(df1_oasis.shape)
print(df2_fire_16gb.shape)
print(df3_paperwhite_4gb.shape)
print(df4_voyage.shape)
print(df5_paperwhite.shape)


# We focus our attention on Kindle Paperwhite (df3 and df5), while saving other dataframes for later use

# In[ ]:


df1_oasis.to_csv('Oasis.csv')
df2_fire_16gb.to_csv('Fire.csv')
df4_voyage.to_csv('Voyage.csv')


# #### Combining df3 and df5 for Kindle Paperwhite Edition Data Only

# In[ ]:


frames = [df3_paperwhite_4gb,df5_paperwhite]
df4_voyage.to_csv('Voyage.csv')
kp = pd.concat(frames)
print(kp.head(5))
print(kp.tail(5))
kp = kp.reset_index()
print(kp.columns)
print(kp['reviews.rating'].describe())
kp.columns = ['Index','ID','Name','ASINS','Brand','Categories','Keys','Manufacturer','ReviewDate','ReviewDateAdded','ReviewDateSeen','PurchasedOnReview','RecommendStatus','ReviewID','ReviewHelpful','Rating','SourceURL','Comments','Title','UserCity','UserProvince','Username']
kp.columns
kp.head(5)
print(kp.columns.nunique())
kp = kp.drop(['ReviewID' , 'UserCity' , 'UserProvince','PurchasedOnReview'],axis = 1)
print(kp.columns.nunique())
print(kp.Rating.value_counts())
kp.Rating.value_counts()


# # 2. Analysis of Data
# 
#     User Rating Count Distribution
#     Rating          No            Percentage
#     5                   2564         79.97% 
#     4                   545           16.99%
#     3                   60             1.87%
#     2                   22             0.686%
#     1                   15             0.467%
# 
# 
# 

# In[ ]:


kp.RecommendStatus.nunique()
import matplotlib.pyplot as plt
kp.hist(column = 'Rating', by = 'RecommendStatus', color = 'Red')
plt.show()
print(kp.info())


# From the above chart we can see that there are people with 4 and 5 rating who have still not recommended the product
# Let's try to explore the review provided by these 1o individuals

# In[ ]:


#slice for rating 5
# slice for recommended
#slice for comments


# In[ ]:


kp['Categories'] = 'Tablets'
kp['Name'] = 'Amazon Kindle Paperwhite'
print(kp.head(5))
print(kp.ReviewHelpful.value_counts())


# Let's see the comments of people have given a 5 star rating and have still Not Recommended the Product

# In[ ]:


pd.DataFrame(kp[(kp.Rating==5)&(kp.RecommendStatus==False)]['Comments'])


#  Let's see the Unique User Names / Shape of the Dataset and the the number of unique Usernames

# In[ ]:


print(kp.Username.nunique())
print(kp.shape)
sum(kp['Username'].value_counts()>1)


# ### NOTES : 
# Understanding the Rating distribution
# 
# 
# There are total of 2890 unique users , however the total ratings received are 3206, so that means some people are giving their ratings more than once, so we need to figure out if the extra 316 ratings are from people who have already provided ratings, and if yes are they from the same date and how many products
# 
# there are 188 people who had given more than 1 rating

# In[ ]:


len(kp['Username'].value_counts()>1)


# We now have names of people who provided more than one comment we now need to figure out the dates when people with THESE NAMES provided their comments. Select dates on which these people added reviews, sort by Username 

# In[ ]:


kp.head(2)
kp = kp.drop('Keys',axis = 1)
print(kp.columns.nunique())
kp =kp.reset_index()
print(kp.head(2))


# # 3. Transforming Date Time
#     - Parse ReviewDate to [Date and Time]
#     - Parse ReviewDateAdded to [Date and Time]
#     - Parse ReviewDateSeen to [Date and Time]

# In[ ]:


kp.ReviewDate = pd.to_datetime(kp['ReviewDate'], dayfirst= True)
kp.ReviewDateAdded =pd.to_datetime(kp.ReviewDateAdded , dayfirst= True)
#kp.ReviewDateSeen = pd.to_datetime(kp.ReviewDateSeen, dayfirst = True)


# In[ ]:


kp['ReviewDateSeen'] = kp['ReviewDateSeen'].str.split(',',expand = True).apply(lambda x:x.str.strip())
kp.ReviewDateSeen = pd.to_datetime(kp.ReviewDateSeen,dayfirst= True)   
print(kp.head(4))


# # 4. Likert Scale Analysis
# 
# ### Net Promoter Score

# In[ ]:


import numpy as np
promoters = sum(kp.Rating==5)
passive = sum(kp.Rating == 4)
detractors = sum(np.logical_and(kp.Rating >= 1, kp.Rating <=3))
respondents = promoters+passive+detractors
NPS_P = ((promoters - detractors)/respondents )*100
print(NPS_P)


# Promters had a NPS of 80.324.
# 
# 
# This is the overall NPS of the product, however, let's visualize how the rating of KindlePaperWhite has changed over time

# In[ ]:


print(kp.tail(2))


# All dates are different, so we hae to calculate NPS for interval of 2 months Plot a line chart for the same

# In[ ]:


kp.plot(x = 'ReviewDate',y = 'Rating', kind = 'line',  figsize=(10,10))


# ###  Pivot Table for Promoter Score by Date

# In[ ]:


review_date = kp.ReviewDate
rating = kp.Rating
df_dr = pd.concat([review_date,rating],axis = 1)
print(df_dr.tail(5))
print(df_dr.shape)


# In[ ]:


df_dr = df_dr.groupby(['ReviewDate','Rating']).size().unstack(fill_value = 0)
print(df_dr.loc['2017-02-04'])


# In[ ]:


print(df_dr.head(5))


# So we  now have rating distribution by date, let's now calculate the sum of ratings for 1,2 and 3 for each date,and finally add a new column to the df, while deleting 1,2,3

# In[ ]:


df_dr.columns = ['A','B','C','Passive','Promoters']
df_dr['Detractors'] = df_dr['A'] + df_dr['B'] + df_dr['C']
df_dr.head(5)


# In[ ]:


df_dr = df_dr.drop(labels = ['A','B','C'],axis = 1)
print(df_dr.head(5))


# In[ ]:


df_dr['NPS'] = (df_dr['Promoters'] - df_dr['Detractors']) * 100 / (df_dr['Passive'] + df_dr['Promoters'] + df_dr['Detractors'])
print(df_dr.head(5))


# In[ ]:


df_dr = df_dr.reset_index()
df_dr.plot( x = 'ReviewDate', y = 'NPS',kind = 'line', figsize=(10,10))


# In[ ]:


df_dr.shape


# # 5. Feature Engineering  
# # 6. Sentiment Analysis - NLTK to find Compound Score

# In[ ]:


kp.Name.nunique()
kp.head(2)
    


# #### Columns to Remove : Index - ID - Name - ASINS - Brand - ReviewDateAdded - ReviewDateSeen - SourceURL

# In[ ]:


data =  kp.drop(['Index','ID','Name','ASINS','Brand','Categories','Manufacturer','ReviewDateAdded','ReviewDateSeen','SourceURL'], axis = 1)
# Cleaned Dataset Now becomes
data = data.reset_index()
data.head(5)


# In[ ]:


data = data.drop(['ReviewDate'], axis = 1)
data.columns


# #### Changing RecommendStatus from True/False to Recommend/Not Recommend

# In[ ]:


def status(data):
    if(data == True):
        data = "Recommend"
        return data
    else:
        data = "Not Recommend"
        return data
    
data['RecommendStatus'] = pd.DataFrame(data['RecommendStatus'].apply(lambda x : status(x)))
data.head(5)
    


# In[ ]:


dsa = data
dsa['feedback'] = dsa['Comments'] + dsa['Title']
dsa = dsa.drop(['Comments','Title'], axis = 1)


# In[ ]:


dsa.head(5)


# #### Feature for Compound Score (using Sentiment Analysis)

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def polar_score(text):
    score = sid.polarity_scores(text)
    x = score['compound']
    return x


dsa['Compound_Score'] = dsa['feedback'].apply(lambda x : polar_score(x))


# In[ ]:


dsa.head(5)


# #### Feature for Text Length

# In[ ]:


dsa['length'] = dsa['feedback'].apply(lambda x: len(x) - x.count(" "))
dsa.head(2)


# # 7. Understanding the  Features added

# #### Ideally people who'll Not Recommend the product, would have a lot to say against the features of the product

# In[ ]:


import numpy as np
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


bins = np.linspace(0,200,40)
pyplot.hist(dsa[dsa['RecommendStatus'] == 'Not Recommend']['length'],bins,alpha  = 0.5,normed = True, label = 'Not Recommend')
pyplot.hist(dsa[dsa['RecommendStatus'] == 'Recommend']['length'],bins,alpha = 0.5,normed = True, label = 'Recommend')
pyplot.legend(loc = 'upper right')
pyplot.show()


# #### Above graph shows that our original hypothesis was correct

# # 8. Using TF-IDF and Random Forest to predict Recommendation Status

# In[ ]:


import string
import nltk
import re
stopword =  nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


# In[ ]:


def clean(text):
    no_punct = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',no_punct)
    text_stem = ([ps.stem(word) for word in tokens if word not in stopword])
    return text_stem


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(analyzer= clean)
Xtf_idfVector = tf_idf.fit_transform(dsa['feedback'])


# #### New DataFrame having all the required features , the label we want to predict and the tf_idf vectorizer

# In[ ]:


import pandas as pd

Xfeatures_data = pd.concat([dsa['Compound_Score'], dsa['length'], pd.DataFrame(Xtf_idfVector.toarray())], axis = 1)
Xfeatures_data.head(5)


# #### We finally have the dataframe we would be applying Machine Learning to

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(Xfeatures_data, dsa['RecommendStatus'], test_size = 0.2)

rf = RandomForestClassifier(n_estimators= 50, max_depth= 20, n_jobs= -1)
rf_model = rf.fit(X_train,y_train)
sorted(zip(rf.feature_importances_,X_train.columns), reverse = True)[0:10]


# So our original assumption about Compound Score being a major indicator in classifcation values was correct

# #### Applying Grid Search to change hyper parameters and then applying RF
# 

# In[ ]:


def compute(n_est, depth):
    rf = RandomForestClassifier(n_estimators= n_est, max_depth= depth)
    rf_model = rf.fit(X_train, y_train)
    y_pred  = rf_model.predict(X_test)
    precision,recall,fscore,support  = score(y_test,y_pred, pos_label= 'Recommend', average = 'binary')
    print('Est: {}\ Depth: {}\ Precision: {}\ Recall: {}\ Accuracy: {}'.format(n_est, depth, round(precision,3), round(recall,3), (y_pred == y_test).sum()/ len(y_pred)))


# In[ ]:


for n_est in [10,30,50,70]:
    for depth in [20,40,60,80,90]:
        compute(n_est,depth)
    


# # Conclusion :  
# 
# Feature Engineering played a key role in boosting the model's performance matrix. The length of the text
# and calculation of compound_score using sentiment analysis served as a basis to strike a balance between Precision & Recall (0.975 vs 1.0) and further made the model robust enough to predict user's recommend status to 97.5%
# 
# This concludes our Analysis of the Kindle Paperwhite.
