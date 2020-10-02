#!/usr/bin/env python
# coding: utf-8

# # Analysis of Google Play Store Apps

# Google's Android OS is the most popular and most widely used in all the smartphones . Every day ,the demand for Android developers is growing with a number of android apps being created and made available in the play store , the official download site for all the android apps . This dataset provides information on the android apps available in the play store along with the user reviews . The analysis will focus on broadly providing insights on the variety of apps available , user rating , sentiment of the customers .

# If you like my kernel , pls leave your comments/upvote.

# ### Loading the required libraries and the data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import os
import re
import functools

pd.options.display.float_format = "{:.2f}".format

# Standard plotly imports
#import plotly_express as px
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

os.listdir()


# In[ ]:


kaggle=1
if kaggle==0:
    store=pd.read_csv('googleplaystore.csv')
    review=pd.read_csv('googleplaystore_user_reviews.csv')
else:
    store=pd.read_csv('../input/googleplaystore.csv')
    review=pd.read_csv('../input/googleplaystore_user_reviews.csv')


# ### Data Cleaning

# The number of rows and columns of the dataset is given as follows,

# In[ ]:


print(f'Shape of the google playstore data:{store.shape}')
print(f'Shape of the user reviews data:{review.shape}')


# Thus we have 10,841 rows and 13 columns in the playstore dataset and 64,295 rows and 5 columns in the userreviews dataset.The columns and their datatype is obtained as follows,

# In[ ]:


review.info()


# In[ ]:


store.info()


# * From the information , we find that there are null values in columns rating ,type,current ver and android ver in play store data where as there are null values in columns translated review,sentiment , sentiment polarity and sentiment subjectivity in reviews data.
# 
# * From the datatype , we understand that even numeric columns like reviews , size,installs , price are all of object datatype .On manually looking at the data , we understand that there is one row which had a column jump and hence the contents changed . Lets check that row.

# In[ ]:


store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame"]


# It is obvious that the category column should have been missing and the preceding column values have jumped ahead.Lets define a category for this app and set things right .

# In[ ]:


store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Category']='PHOTOGRAPHY'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Rating']='1.9'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Reviews']='19'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Size']='3.0M'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Installs']='1,000+'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Type']='Free'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Price']='0'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Content Rating']='Everyone'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Genres']='Photography'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Last Updated']='February 11, 2018'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Current Ver']='1.0.19'
store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Android Ver']='4.0 and up'


# In[ ]:


store.loc[store['App']=="Life Made WI-Fi Touchscreen Photo Frame"]


# Now that we have corrected that row , lets convert the reviews column into integer datatype . Also the size and installs have "M" and "+" in them which makes them an object datatype . We replace those two and convert it into a integer datatype.

# In[ ]:


store['Size']=store['Size'].apply(lambda x:str(x).replace("M","") if 'M' in str(x) else x)
store['Size']=store['Size'].apply(lambda x:float(str(x).replace("k",""))/1000 if 'k' in str(x) else x)
store['Price']=store['Price'].apply(lambda x:str(x).replace("$","") if "$" in str(x) else x)
store['Installs']=store['Installs'].apply(lambda x:str(x).replace("+","") if "+" in str(x) else x)
store['Installs']=store['Installs'].apply(lambda x:str(x).replace(",","") if "," in str(x) else x)


# In[ ]:


store=store.astype({'Rating':'float32','Reviews':'float32','Installs':'float64','Price':'float64'})


# Lets turn our focus on the missing values.

# In[ ]:


store.isnull().sum()


# From this we understand that the columns rating,type,current ver and android ver have null values.Lets check one by one and decide how to impute the null values.

# In[ ]:


store.loc[store['Type'].isna()]


# We find that there is one null value for Type . A quick google check indicates that it is a purchased product and hence should be imputed with "Paid" though the price value is 0.We impute that column to the app price $1.14 .

# In[ ]:


store.loc[store['Type'].isna(),'Price']=1.14
store.loc[store['Type'].isna(),'Type']='Paid'


# In[ ]:


store.loc[store['Rating'].isna()]


# Since it would be difficult to impute each of the apps with the corresponding rating though google search , we impute NA with 0 meaning not rated yet.

# In[ ]:


store['Rating'].fillna(0,inplace=True)


# In[ ]:


store.loc[store['Current Ver'].isna()]


# We impute the rows as "varies with device" in this case

# In[ ]:


store['Current Ver'].fillna('Varies with device',inplace=True)


# In[ ]:


store.loc[store['Android Ver'].isna()]


# In this case also , we impute "Varies with device"

# In[ ]:


store['Android Ver'].fillna('Varies with device',inplace=True)


# Now that all the datacleaning is done , we are ready to begin our analysis.

# In[ ]:


store.head()


# In[ ]:


review.head()


# ### Exploratory Data Analysis

# ### Distribution of Rating , Review ,Size and Installs

# In[ ]:


store['Rating'].describe()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.distplot(store['Rating'],bins=40,color="green")
ax.set_xlabel("Rating")
ax.set_ylabel("Distribution Frequency")
ax.set_title("Rating - Distribution")


# From the plot , we understand that the distribution is multimodal having peaks at rating 0 and at 4 .There are more apps having a rating value between 3 to 4.5 .

# In[ ]:


store['Reviews'].describe()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.distplot(store['Reviews'],color="green")
ax.set_xlabel("Review")
ax.set_ylabel("Distribution Frequency")
ax.set_title("Review - Distribution")


# It is not surprising at all to see a distribution of the reviews tailed towards the right since the descriptive statistics gave a clear indication where we find that almost 75 % of the data take up values of magnitude till 10^4.The standard deviation of the variable is very high - in the magnitude of 10^6 .Taking the log of the variable and plotting the distribution,

# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.distplot(np.log1p(store['Reviews']),color="green")
ax.set_xlabel("Review")
ax.set_ylabel("Log - Distribution Frequency")
ax.set_title("Review -Log  Distribution")


# The distribution is not very symmetric but far better than the previous graph without the log transformation.

# In[ ]:


store.loc[store['Size']!="Varies with device",['Size']].astype('float32').describe()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.distplot(store.loc[store['Size']!="Varies with device",['Size']].astype('float32'),color="green")
ax.set_xlabel("Size")
ax.set_ylabel("Distribution Frequency")
ax.set_title("Size -Distribution")


# The distribution for the size seems to be skewed towards the right .The mode is between 0 - 20 MB .

# In[ ]:


store['Installs'].describe()


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.distplot(store['Installs'],color="green")
ax.set_xlabel("Installs")
ax.set_ylabel("Distribution Frequency")
ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.set_title("Installs -Distribution")


# The plot is highly skewed and provides an insight from the descriptive statistics that there have been only few apps which has seen the maximum number of downloads . 

# ### Apps and Category

# Lets check how many apps are present in the dataset.

# In[ ]:


store['App'].nunique()


# There are 9660 unique apps present in the dataset.Lets check out the apps that are repeated more than once.

# In[ ]:


pd.concat(g for _,g in store.groupby('App') if len(g)>1)


# In[ ]:


pd.concat(g for _,g in store.groupby('App') if len(g)>1)['App'].nunique()


# In[ ]:


pd.concat(g for _,g in store.groupby('App') if len(g)>1)['App'].shape


# From the output of the data, we understand that there are 798 apps out of 9660 apps which seems to have been duplicated ( eg. 10 best food for you , 1800 contacts-Lens store) whereas there are also rows for the same app but having different review count(eg.365Scores-Live Scores,8 ball pool) or category(eg.A&E - Watch full episodes of TV shows) .To simplify things for further analysis , I remove the duplicates , keep the row for which the review count has been higher in case if there are different review counts .

# In[ ]:


## Removing the duplicates:
store = store.drop_duplicates()


# In[ ]:


## Check the number of rows,
store.shape


# In[ ]:


## Get the rows having the maximum review count ,
#temp=store.loc[store.groupby(['App'])['Reviews'].idxmax()]
store=store.loc[store.groupby(['App','Category'])['Reviews'].idxmax()]


# In[ ]:


store.shape


# In[ ]:


## Check the apps having assigned in more than 1 category 
## Taken from my kernel on Olympics - https://www.kaggle.com/gsdeepakkumar/gold-hunters
multi_cat=store.groupby('App').apply(lambda x:x['Category'].unique()).to_frame().reset_index()
multi_cat.columns=['App','Categories']
multi_cat['Count']=[len(c) for c in multi_cat['Categories']]


# In[ ]:


multi_cat[multi_cat['Count']>1].sort_values('Count',ascending=False)


# Lets check the different type of categories represented in the dataset.

# In[ ]:


store['Category'].nunique()


# There are 33 unique categories . What are they?

# In[ ]:


store['Category'].unique()


# ### How many apps are represented for each category ?

# In[ ]:


category_app=store.groupby('Category')['App'].nunique().sort_values(ascending=False).to_frame().reset_index()
category_app.columns=['Category','Total']
category_app['Perc']=category_app['Total']/sum(category_app['Total'])
category_app


# Family apps dominate the store dataset with 1909 apps which contributes to 19 % of the total apps represented in the dataset . 9 % of the apps belong to game category followed closely by the tools category . Beauty ,comics apps are represented in small numbers .

# ### What is the average rating for each of the categories ?

# In[ ]:


store.groupby('Category')['Rating'].mean().sort_values(ascending=False)


# The average rating for apps in education category is 4.32 which is closely followed by art and design , entertainment .Lets check the average rating by the type of app - free and paid.

# In[ ]:


avg_rating=store.groupby(['Category','Type'])['Rating'].mean().sort_values(ascending=False).to_frame().reset_index()
plt.figure(figsize=(12,7))
plt.subplot(211)
ax=sns.barplot(x='Category',y='Rating',data=avg_rating.loc[avg_rating['Type']=='Free'],color="blue")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Category')
ax.set_ylabel('Rating')
ax.set_title("Category and Average Rating for Free Apps")
plt.subplot(212)
ax=sns.boxplot(x='Category',y='Rating',data=store.loc[store['Type']=='Free'],order=avg_rating.loc[avg_rating['Type']=='Free','Category'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Category')
ax.set_ylabel('Rating')
ax.set_title("Boxplot of Category and Rating for Free Apps")

plt.subplots_adjust(wspace = 0.8, hspace = 1.2,top = 1.3)

plt.show()


# In[ ]:


plt.figure(figsize=(12,7))
plt.subplot(211)
ax=sns.barplot(x='Category',y='Rating',data=avg_rating.loc[avg_rating['Type']=='Paid'],color="lightblue")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Category')
ax.set_ylabel('Rating')
ax.set_title("Category and Average Rating for Paid Apps")

plt.subplot(212)
ax=sns.boxplot(x='Category',y='Rating',data=store.loc[store['Type']=='Paid'],order=avg_rating.loc[avg_rating['Type']=='Paid','Category'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Category')
ax.set_ylabel('Rating')
ax.set_title("Boxplot of Category and Rating for Paid Apps")

plt.subplots_adjust(wspace = 0.8, hspace = 1.2,top = 1.3)

plt.show()


# * From the two plots, we see that there is a significant difference in the ratings and the category . While news and magazines is top rated in paid category ,education is the top rated in free apps . There is no difference in the 2nd and 3rd positions though.
# * Education is the top rated category in free apps 
# * While the free app category rating tends to atleast have a rating of above ~2.5 for all the categories ,for paid apps the rating is also 0 for the categories events , libraries and demo.
# * From the boxplot , it is understood that while there are many categories for which the apps have been rated 0 in free apps , there are only few such apps in paid category.
# * In the paid apps category , the boxplot distibution is very thin for top apps . 

# ### What is the average download for each of the categories ?

# In the similar lines as that of category ratings , we analyse the average installs for the app categories by the type .

# In[ ]:



store.groupby('Category')['Installs'].mean().sort_values(ascending=False)


# The average downloads is higher for Communication with 3.4 Mn downloads .Video players stands second with 2.3 Mn downloads .While the difference between first and second downloads is high ,the difference is not much between the second and third categories.

# In[ ]:


avg_installs=store.groupby(['Category','Type'])['Installs'].mean().sort_values(ascending=False).to_frame().reset_index()
plt.figure(figsize=(12,7))
plt.subplot(211)
ax=sns.barplot(x='Category',y='Installs',data=avg_installs.loc[avg_installs['Type']=='Free'],color="blue")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Category')
ax.set_ylabel('Installs')
ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.set_title("Category and Average Installs for Free Apps")

plt.subplot(212)
ax=sns.boxplot(x='Category',y='Installs',data=store.loc[store['Type']=='Free'],order=avg_installs.loc[avg_installs['Type']=='Free','Category'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Category')
ax.set_ylabel('Installs')
ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.set_title("Boxplot of Category and Installs for Free Apps")

plt.subplots_adjust(wspace = 0.8, hspace = 1.2,top = 1.3)

plt.show()


# In[ ]:



plt.figure(figsize=(12,7))
plt.subplot(211)
ax=sns.barplot(x='Category',y='Installs',data=avg_installs.loc[avg_installs['Type']=='Paid'],color="blue")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Category')
ax.set_ylabel('Installs')
ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.set_title("Category and Average Installs for Paid Apps")

plt.subplot(212)
ax=sns.boxplot(x='Category',y='Installs',data=store.loc[store['Type']=='Paid'],order=avg_installs.loc[avg_installs['Type']=='Paid','Category'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Category')
ax.set_ylabel('Installs')
ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.set_title("Boxplot of Category and Installs for Paid Apps")

plt.subplots_adjust(wspace = 0.8, hspace = 1.2,top = 1.3)

plt.show()


# * The average number of installs for a free app is higher than the average number of installs for a paid app.While the avg download is about 3.5 Mn for a free app , for a paid app it is 0.25 Mn .
# * There is a significant difference in the downloads between the free and a paid app . While Communication , video players and social are top downloaded in free app category , game ,education and family categories are most downloaded in the paid category.
# * The free type of downloads is most dominated by outliers for each of the categories where as there are only few outliers for the categories in paid type of downloads

# ### Which apps are top rated , most downloaded in each of the top categories - FAMILY,GAME,TOOLS,BUSINESS,MEDICAL ?

# For this analysis , inorder to get accurate results , we consider only the rating of the apps which have exceeded the average download for that category .

# In[ ]:


family_app=store[store['Installs']>=4654605].loc[store['Category']=='FAMILY'].sort_values(by='Rating',ascending=False)[0:9]
family_app[['App','Rating','Installs','Type']]


# In[ ]:


game_app=store[store['Installs']>=14550962].loc[store['Category']=='GAME'].sort_values(by='Rating',ascending=False)[0:9]
game_app[['App','Rating','Installs','Type']]


# In[ ]:


tool_app=store[store['Installs']>=9774151].loc[store['Category']=='TOOLS'].sort_values(by='Rating',ascending=False)[0:9]
tool_app[['App','Rating','Installs','Type']]


# In[ ]:


business_app=store[store['Installs']>=1659916].loc[store['Category']=='BUSINESS'].sort_values(by='Rating',ascending=False)[0:9]
business_app[['App','Rating','Installs','Type']]


# In[ ]:


medical_app=store[store['Installs']>=99224].loc[store['Category']=='MEDICAL'].sort_values(by='Rating',ascending=False)[0:9]
medical_app[['App','Rating','Installs','Type']]


# From the output of this analysis , we understand that all the top rated apps in each category belong to free type of downloads .

# ### Is there a relation between the size of the app and rating ?

# While there may not seem to be a relation between the size and rating , there are cases where the app might hang due to lot of customisation and background process (which may have a huge file size) and this might make the users give a lower rating.Let us check this relation . I will use interactive plots from plotly for this .

# In[ ]:


store.loc[store['Size']!='Varies with device'].iplot(
    x='Rating',
    y='Size',
    # Specify the category
    categories='Type',
    xTitle='Rating',
    yTitle='Size',
    title='Rating Vs Size by Type')


# From the plot, it is understood that there is no clear distinction or boundary between the rating and size . There are apps with a higher rating with app size on the higher size .

# Lets check the relation between the reviews and rating .

# In[ ]:


store.iplot(
    x='Rating',
    y='Reviews',
    # Specify the category
    categories='Type',
    xTitle='Rating',
    yTitle='Reviews',
    title='Rating Vs Reviews by Type')


# The most number of reviews come for free apps.Clearly it is seen that free apps having rating between 4 to 5 have the maximum reviews .

# In[ ]:


# px.scatter(store, x="Installs", y="Rating", color="Rating", facet_col="Type",
#            color_continuous_scale=px.colors.sequential.Viridis, render_mode="webgl")


# The following conclusions were made after the output of the plot .Unfortunately,I was not able to install plotly_express package in Kaggle kernels.So I ran it in my local machine .
# * From the plot it is seen that paid apps have not seen more than 10 Mn downloads .
# * There are free apps which have high ratings but the downloads do not exceed 10 Mn . On the other hand , there are some free apps which have an rating of ~3.7 but has seen 1Bn downloads . Clearly this makes us to ponder over the fact that  whether people download the apps based on only their needs and requirements or is it some other factor which is dominating the downloads ?

# ### Content of the app

# Lets check the content rating of the publisher of the apps.

# In[ ]:


content_rating=store.groupby('Content Rating')['App'].nunique().sort_values(ascending=False).to_frame().reset_index()
content_rating.columns=['Content Rating','Apps']
content_rating['Perc']=content_rating['Apps']/sum(content_rating['Apps'])
content_rating


# 82 % of the apps can be used by everyone where as 11% are for the teens .

# ### Expensive Apps

# Lets check the price distribution of the paid apps.

# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.distplot(store.loc[store['Type']=='Paid',['Price']],color='green')
ax.set_xlabel("Price of app")
ax.set_ylabel("Freq Distribution")
ax.set_title("Distribution of Price")


# The distribution is skewed towards right with most of the price of the apps falling between $ 0 - $ 50 .Lets check the most expensive apps.

# In[ ]:


exp_apps=store.loc[store['Type']=='Paid'].sort_values(by='Price',ascending=False)[0:10]
exp_apps[['App','Price']]


# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.barplot(x='App',y='Price',data=exp_apps)
ax.set_xlabel('App')
ax.set_ylabel('Price')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title("Most Expensive Apps")


# All the top 10 apps are the same with different names - I am Rich !!! .

# ### Wordcloud of Reviews

# Now that we have covered extensively on the app data lets turn out focus on the reviews for each app .On having a glimpse of the app earlier , we find that there are certain rows having NA values .For our analysis we remove those columns.

# In[ ]:


review.head()


# In[ ]:


review=review.dropna()


# In[ ]:


review.head()


# Lets get a summary of the sentiments.

# In[ ]:


review['Sentiment'].value_counts()


# We have an overwhelming positive sentiments in the app reviews.Lets create a wordcloud for each of those sentiments.

# In[ ]:


pos=review.loc[review['Sentiment']=='Positive']
neg=review.loc[review['Sentiment']=='Negative']
neu=review.loc[review['Sentiment']=='Neutral']


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import string
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import word_tokenize
eng_stopwords=set(stopwords.words('english'))


# In[ ]:


### Inspired from https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
plt.figure(figsize=(16,13))
wc = WordCloud(background_color="white", max_words=10000, 
            stopwords=STOPWORDS)
wc.generate(" ".join(pos['Translated_Review'].values))
plt.title("Wordcloud for Positive Reviews", fontsize=20)
plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98)
plt.axis('off')


# Great,love,time,good,work are most repeated words in the positive reviews.

# In[ ]:


plt.figure(figsize=(16,13))
wc = WordCloud(background_color="white", max_words=10000, 
            stopwords=STOPWORDS)
wc.generate(" ".join(neg['Translated_Review'].values))
plt.title("Wordcloud for Negative Reviews", fontsize=20)
plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98)
plt.axis('off')


# game,update,problem,eve,time are some of the words most common in the negative review.

# In[ ]:


plt.figure(figsize=(16,13))
wc = WordCloud(background_color="white", max_words=10000, 
            stopwords=STOPWORDS)
wc.generate(" ".join(neu['Translated_Review'].values))
plt.title("Wordcloud for Neutral Reviews", fontsize=20)
plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98)
plt.axis('off')


# Neutral reviews have a mix of both words from positive and negative reviews.

# ### Difference between Positive ,Negative and Neutral Reviews

# Lets find out the difference between positive,negative and neutral reviews by extracting the basic metafeatures - number of words,number of stopwords,number of unique words and number of punctuations in the sentence.

# In[ ]:


review['num_words']=review['Translated_Review'].apply(lambda x:len(str(x).split()))
review['num_stopwords']=review['Translated_Review'].apply(lambda x:len([w for w in str(x).lower().split() if w in eng_stopwords]))
review['num_punctuations']=review['Translated_Review'].apply(lambda x:len([w for w in str(x) if w in string.punctuation]))


# To differentiate between the sentiment of the reviews , lets plot a violin plot .

# In[ ]:


## For better visuals truncate words greater than 120 to 120
review['num_words'].loc[review['num_words']>120]=120

plt.figure(figsize=(8,8))
ax=sns.boxplot(x='Sentiment',y='num_words',data=review)
ax.set_xlabel('Sentiment')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_ylabel('Number of Words')
ax.set_title("Difference between the Number of Words Vs Sentiment")


# The plot shows that the median number of words for a negative review is higher than the median for positive and neutral reviews . Also the mode for positive review is closer to 0 -20 range.

# In[ ]:



plt.figure(figsize=(8,8))
ax=sns.boxplot(x='Sentiment',y='num_stopwords',data=review)
ax.set_xlabel('Sentiment')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_ylabel('Number of Stop Words')
ax.set_title("Difference between the Number of Stop Words Vs Sentiment")


# There is no significant difference between the setiment and number of stopwords used to express each sentiment.

# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.boxplot(x='Sentiment',y='num_punctuations',data=review)
ax.set_xlabel('Sentiment')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_ylabel('Number of Punctuations')
ax.set_title("Difference between the Number of Punctuations Vs Sentiment")


# Similarly , the number of punctuations is not significantly different between the sentiments expressed .

# ### Review Analysis

# In[ ]:


### Inspired from - https://www.kaggle.com/residentmario/exploring-elon-musk-tweets


# Let us define a function to tokenize a word and check the related reviews where that token appears .From this analysis , we would be able to see what people have to say on a particular token for an app . The tokens we would consider is - performance,battery , productivity,memory .For simplicity , we take the first 5 hits and output it.

# This analysis could prove to be useful for app developers to understand the concerns that would be faced by the user of the app and modify the app accordingly

# In[ ]:


tokens=review['Translated_Review'].map(word_tokenize)


# In[ ]:


tokens.head()


# In[ ]:


def get_reviews_on_token(x):
    x_l = x.lower()
    x_t = x.title()
    return review.loc[tokens.map(lambda sent: x_l in sent or x_t in sent).values]


# In[ ]:


get_reviews_on_token('Performance')[['App','Translated_Review']][0:5].values.tolist()


# The output of the function is converted into a list with first index referring to the app and the second index is the review for that app containing the token 'performance'

# In[ ]:


get_reviews_on_token('memory')[['App','Translated_Review']][0:5].values.tolist()


# In[ ]:


get_reviews_on_token('battery')[['App','Translated_Review']][0:5].values.tolist()


# In[ ]:


get_reviews_on_token('productivity')[['App','Translated_Review']][0:5].values.tolist()


# # Conclusion

# Thus through the analysis of the dataset ,the following insights were mined :
# 
# * The average rating for the app stands at 4.5 and the number of reviews for the app is heavily biased for the apps which has seen the top ratings.
# * There is no relationship between the rating and downloads .There is some other factor which makes people download and use the app.
# * There are more free apps which has seen ~1 Bn downloads when compared to paid apps which has not seen the downloads crossing more than 10Mn.
# * Average rating for Education , Art and Design,Entertainment,Game,Comic categories are above 4 whereas for Business the average rating has been less than 3.
# * Average downloads for Communication , Video players and Social has been above 2 Mn whereas the average download for Events and Medical has been less than 0.5 Mn.
# * While wordcloud gives a glimpse of most used words in the review , some words have been repeated in all the three sentiment polarities thereby we cant conclude anything on the reviews.
# * An attempt was made to get the reviews which had a particular token to understand whether people have given positive or negative reviews for that app .
# 
