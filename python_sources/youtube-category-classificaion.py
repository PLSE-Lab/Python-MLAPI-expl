#!/usr/bin/env python
# coding: utf-8

# # Youtube Category Classification
# By : Hesham Asem
# ____
# 
# 
# in this dataset , we have a dataset for viewership habits for specific most famous videos on Youtube , for Great Britain on 2017 , 2018
# 
# we need to make a classification model , so it can detect which category of the video depend on viewership habits & video details
# 
# dataset here : 
# https://www.kaggle.com/datasnaek/youtube-new#GBvideos.csv
# 
# ____
# 
# 
# let's first import needed libraries
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid")


# then we can open the dataset

# In[ ]:


data = pd.read_csv('/kaggle/input/youtube-new/GBvideos.csv') 


# let's have a look at it 

# In[ ]:


data.head()


# ____
# 
# ok , several important & less important features , how is the dimension ? 

# In[ ]:


data.shape


# 
# ______
# 
# # Important Functions 
# 
# so we'll need to define important functions which will be used here

# In[ ]:


def make_label_encoder(original_feature , new_feature) : 
    enc  = LabelEncoder()
    enc.fit(data[original_feature])
    data[new_feature] = enc.transform(data[original_feature])
    data.drop([original_feature],axis=1, inplace=True)


# In[ ]:


def make_countplot(feature) :
    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("prism", 3)) 


# In[ ]:


def make_kdeplot(feature) : 
    sns.kdeplot(data[feature], shade=True)


# In[ ]:


def make_pie(feature) : 
    plt.pie(data[feature].value_counts(),labels=list(data[feature].value_counts().index),
        autopct ='%1.2f%%' , labeldistance = 1.1,explode = [0.05 for i in range(len(data[feature].value_counts()))] )
    plt.show()


# 
# _______
# 
# 
# # Data Processing 
# 
# so we'll need to start with processing the features one after one , the list of features is : 

# In[ ]:


data.columns


# unique data is an important thing to know about each feature , so let;s know the umber of unique values for each feature 

# In[ ]:


for col in data.columns : 
    print('Unique values for Column {0}     is      {1}'.format(col ,len(data[col].unique())))


# _____
# 
# as shown here , some features will be used with its original values , & some will need some processing
# 
# 
# let's start with video_id . 
# 
# since video_id generally refer to something useless , which will not be helpful in training , but here we can see that there is 3272 unique values from almost 39K sample size , which mean there is a huge number of repeating video_id , so this feature will be important in training
# 
# we'll have to convert it using labelencoder from sklearn , to use code instead of it 
# 

# In[ ]:


make_label_encoder('video_id' , 'video_id Code')


# now let's have a look to the data

# In[ ]:


data.head()


# ____
# 
# now how about trending_date , we need to know more about it

# In[ ]:


data['trending_date'].unique()


# ok , how many unique values ? 
# 

# In[ ]:


len(data['trending_date'].unique())


# from 39K row , there is only 205 unique value , so we'll need it 
# 
# now let's make a smart step , we can extract new features from treding_date , by getting the year , month & date of it to use them as separate 3 new features 

# In[ ]:


year_list = []
month_list = []
day_list = []
for x in range(data.shape[0]) :
    year_list.append(data['trending_date'][x][:2])
    month_list.append(data['trending_date'][x][6:])
    day_list.append(data['trending_date'][x][3:5])


# now we can add them to the data

# In[ ]:


data.insert(16,'Year',year_list)
data.insert(17,'Month',month_list)
data.insert(18,'Day',day_list)


# and have a look now

# In[ ]:


data.head()


# ____
# 
# 
# cool  , now let's see how countplot of different years , months & days

# In[ ]:


ax = sns.countplot(x="Year", data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark",3))


# In[ ]:


ax = sns.countplot(x="Month", data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark",3))


# In[ ]:


ax = sns.kdeplot(data['Day'], shade=True)


# as we get all what we need from trending_date , we can drop it now

# In[ ]:


data.drop(['trending_date'],axis=1, inplace=True)


# 
# also we can handle the Year value , to make it either 0 or 1 , instead of 17 & 18 , so it may be easier for the model in calculation 

# In[ ]:


year_dict = {'17': 0,'18':1}
data['Year'] = data['Year'].map(year_dict)


# how it looks now

# In[ ]:


data.head()


# ______
# 
# 
# # Words Processing
# 
# we have here a big point which related to the title of the video 
# 
# the title contain several different words , which might be important for training , but also we can never just use labelencoder to convert unique values , we need to make something more smart
# 
# first let's know the amount of unique values

# In[ ]:


len(data['title'].unique())


# 
# ____
# 
# OK , we'll make a HUGE list contain all words separately of all titles , then we'll focus only on most repeated words , as they are important keywords

# In[ ]:


all_words = []
for x in range(data.shape[0]) : 
    all_words = all_words  +  data['title'][x].split()


# great , how many words we have now  ? 

# In[ ]:


len(all_words)


# so we need to convert it to Series so we can easily count repeated words 

# In[ ]:


all_words_series = pd.Series(all_words)


# so we'll make a new Series which is the most repeated 30 words here

# In[ ]:


top_words =  all_words_series.value_counts()[:30] 
top_words


# ok , we need to get only the index (the words themselves)

# In[ ]:


top_words = list(top_words.index)


# In[ ]:


top_words


# as we see , there are many meaningless words , which looks like stopwords , so we have to drop them to avoid any misleading in training as they refer to nothing in the meaning 

# In[ ]:


top_words.remove('-')
top_words.remove('|')
top_words.remove('The')
top_words.remove('&')
top_words.remove('the')
top_words.remove('of')
top_words.remove('and')
top_words.remove('in')
top_words.remove('on')
top_words.remove('a')
top_words.remove('with')
top_words.remove('In')
top_words.remove('To')
top_words.remove('A')


# great , now let's have another look

# In[ ]:


top_words


# here we go they are 16 words 
# 
# but since there are repeating words here which are Video) , Video] , also 3 different types of official , so they are 13 words
# 
# now we'll make 13 lists , each one will be either 1 if the title contain this word , or 0 if it doesn't , then we'll insert them in the dataset later

# In[ ]:


l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []
l9 = []
l10 = []
l11 = []
l12 = []
l13 = []

this_list = []
for x in range(data.shape[0]) : 
    this_list  =  data['title'][x].split()
    
    if ('Video)' in this_list) or ('Video]' in this_list)  : 
        l1.append(1)
    else : 
        l1.append(0)
        
    if ('(Official' in this_list) or ( 'Official'in this_list)  or ( '[Official' in this_list)  : 
        l2.append(1)
    else : 
        l2.append(0)
        
    if 'Trailer' in this_list : 
        l3.append(1)
    else : 
        l3.append(0)

    if 'You' in this_list : 
        l4.append(1)
    else : 
        l4.append(0)
    if 'Last' in this_list : 
        l5.append(1)
    else : 
        l5.append(0)
    if 'Star' in this_list : 
        l6.append(1)
    else : 
        l6.append(0)
    if 'My' in this_list : 
        l7.append(1)
    else : 
        l7.append(0)
    if 'Me' in this_list : 
        l8.append(1)
    else : 
        l8.append(0)
    if  'I'  in this_list : 
        l9.append(1)
    else : 
        l9.append(0)
    if 'to' in this_list : 
        l10.append(1)
    else : 
        l10.append(0)
    if '2018' in this_list : 
        l11.append(1)
    else : 
        l11.append(0)
    if 'Music' in this_list : 
        l12.append(1)
    else : 
        l12.append(0)
 
    if 'ft.' in this_list : 
        l13.append(1)
    else : 
        l13.append(0)


# perfect , let's check the length of any of these lists

# In[ ]:


len(l12)


# exactly the length of our dataset , now le'ts insert them in the dataset

# In[ ]:


data.insert(18,'word 1',l1)
data.insert(19,'word 2',l2)
data.insert(20,'word 3',l3)
data.insert(21,'word 4',l4)
data.insert(22,'word 5',l5)
data.insert(23,'word 6',l6)
data.insert(24,'word 7',l7)
data.insert(25,'word 8',l8)
data.insert(26,'word 9',l9)
data.insert(27,'word 10',l10)
data.insert(28,'word 11',l11)
data.insert(29,'word 12',l12)
data.insert(30,'word 13',l13)


# now we can safely drop the title feature

# In[ ]:


data.drop(['title'],axis=1, inplace=True)


# let's see how it looks like now

# In[ ]:


data.head()


# _______
# 
# 
# # More Data Processing
# 
# ok , let's continue in the rest of features
# 
# how about the channel_title

# In[ ]:


len(data['channel_title'].unique())


# ok , we might not need to make that word processing as we did in the title , let's just convert it using labelencoder

# In[ ]:


make_label_encoder('channel_title' , 'channel Code')


# how the data looks like

# In[ ]:


data.head()


# how about category_id

# In[ ]:


data['category_id'].unique()


# so this is the output , which we'll need to classify the data depend on it , how many categories we have here

# In[ ]:


len(data['category_id'].unique())


# let's draw it

# In[ ]:


make_countplot('category_id')


# majority of them in 10th & 24th category , ok , nothing to be done with it
# 
# _____
# 
# how about publish time

# In[ ]:


len(data['publish_time'].unique())


# more than 3 thousand unique values , how this data looks like ? 

# In[ ]:


data['publish_time'][:10]


# so it's year them month then day then housr , minute , second . .
# 
# I think the most important part of data here is the House of publishing , since the day , month & year is somehow related to trending date which we already included
# 
# so let's make something useful with it , we'll use the publishing hour to know which quarter of the day it published , & use it as an important feature

# In[ ]:


publish_quarter = []

for x in range(data.shape[0]):
    publish_hour = int(str(data['publish_time'][x])[11:13])
    if publish_hour >=0 and publish_hour < 6  : 
        publish_quarter.append(1)
    elif publish_hour >=6 and publish_hour < 12  : 
        publish_quarter.append(2)
    elif publish_hour >=12 and publish_hour < 18  : 
        publish_quarter.append(3)
    else: 
        publish_quarter.append(4)


# great , let's be sure the length of the list is equal to data length

# In[ ]:


len(publish_quarter)


# now to insert it

# In[ ]:


data.insert(30,'Publish Quarter',publish_quarter)


# how data looks like

# In[ ]:


data.head()


# so we can encode publish time now

# In[ ]:


make_label_encoder('publish_time' , 'publish time code')


# In[ ]:


data.head()


# _____
# 
# ok how about tags ? 

# In[ ]:


len(data['tags'].unique())


# how it looks like ? 

# In[ ]:


data['tags'][:10]


# let's see a random tag

# In[ ]:


data['tags'][10]


# we can encode it directly now

# In[ ]:


make_label_encoder('tags' , 'tags code')


# In[ ]:


data.head()


# ___
# 
# # Viewership Numbers
# 
# ow we need to focus in important numbers like : views, likes & dislikes
# 
# let's start with views

# In[ ]:


len(data['views'].unique())


# almost there is no repeated values here , how about min & max values for views of these videos

# In[ ]:


data['views'].min()


# In[ ]:


data['views'].max()


# the different is HUGE , from 800 to 400 Million , even drawing it will not be easy

# In[ ]:


make_kdeplot('views')


# so it's clear that majority of videos within 800 to 30 Million
# 
# 
# so we'll make an important function to make 10 sectors of this feature (or any feature) , depend on the max value we define

# In[ ]:


def feature_sectors(data , maxx , feature , new_featre):
    step = (maxx- data[feature].min())/10
    new_list = []
    minn = data[feature].min()
    for x in range(data.shape[0]) : 
        if data[feature][x] <= (minn + step):
            new_list.append(1)
        elif data[feature][x] <= (minn + (2*step)):
            new_list.append(2)            
        elif data[feature][x] <= (minn + (3*step)):
            new_list.append(3)            
        elif data[feature][x] <= (minn + (4*step)):
            new_list.append(4)            
        elif data[feature][x] <= (minn + (5*step)):
            new_list.append(5)            
        elif data[feature][x] <= (minn + (6*step)):
            new_list.append(6)            
        elif data[feature][x] <= (minn + (7*step)):
            new_list.append(7)            
        elif data[feature][x] <= (minn + (8*step)):
            new_list.append(8)            
        elif data[feature][x] <= (minn + (9*step)):
            new_list.append(9)            
        else:
            new_list.append(10)            
    data.insert(data.shape[1], new_featre , new_list)   
    


# ___
# 
# 
# great , now let's use it to make 10 sectors for the feature views , with max 50 Million , & check how it looks like

# In[ ]:


feature_sectors(data ,50000000 , 'views' , 'views sector')


# how data looks like now

# In[ ]:


data.head()


# let's graph it

# In[ ]:


make_countplot('views sector')


# well, not very representative m since almost all sectors less than 50 Million , lets drop this sector , & make a new sector with only 1 million views as max

# In[ ]:


data.drop(['views sector'],axis=1, inplace=True)


# In[ ]:


feature_sectors(data ,1000000 , 'views' , 'views sector')


# how data looks like

# In[ ]:


data.head()


# and graph

# In[ ]:


make_countplot('views sector')


# not very well , 1 Million is too small for that , let's use 10 million instead , but first drop that sector feature

# In[ ]:


data.drop(['views sector'],axis=1, inplace=True)


# In[ ]:


feature_sectors(data ,10000000 , 'views' , 'views sector')


# In[ ]:


data.head()


# ok the graph 

# In[ ]:


make_countplot('views sector')


# ok , I think this is kinda good as a representative for the data 
# 
# ______
# 
# 
# we need to make something similar in likes
# 

# In[ ]:


make_kdeplot('likes')


# I think 500 thousands is a suitable number to divide on it

# In[ ]:


feature_sectors(data ,500000 , 'likes' , 'likes sectors')


# how it looks 

# In[ ]:


make_countplot('likes sectors')


# kinda suitable , let's do it in dislikes

# In[ ]:


make_kdeplot('dislikes')


# may be 50 thousand is suitable

# In[ ]:


feature_sectors(data ,50000 , 'dislikes' , 'dislikes sectors')


# how it looks now

# In[ ]:


make_countplot('dislikes sectors')


# I think we can reduce it a little bit , let's drop it first

# In[ ]:


data.drop(['dislikes sectors'],axis=1, inplace=True)


# & make it with 10 thousands

# In[ ]:


feature_sectors(data ,10000 , 'dislikes' , 'dislikes sectors')


# how it looks

# In[ ]:


make_countplot('dislikes sectors')


# ok looks better, now how looks like

# In[ ]:


data.head()


# ____
# 
# how about comment count

# In[ ]:


make_kdeplot('comment_count')


# let's repeat it with 10 thousand

# In[ ]:


feature_sectors(data ,10000 , 'comment_count' , 'comment_count sectors')


# how it looks

# In[ ]:


make_countplot('comment_count sectors')


# ____
# 
# ow the thubnail link will be useless for us

# In[ ]:


data.drop(['thumbnail_link'],axis=1, inplace=True)


# also we have to convert comment_disabled , rating_disabled , video_error_or_removed , using label encoder to be either 1 or 0

# In[ ]:


make_label_encoder('comments_disabled','comments_disabled code')
make_label_encoder('ratings_disabled','ratings_disabled code')
make_label_encoder('video_error_or_removed','video_error_or_removed')


# how data looks now

# In[ ]:


data.head()


# ____
# 
# now the description , how many unique values here ? 
# 

# In[ ]:


len(data['description'].unique())


# so we'll not be able to encode it , since it contains many unusual charracters , so we'll have to drop it

# In[ ]:


data.drop(['description'],axis=1, inplace=True)


# now the last feature : category_id

# In[ ]:


make_countplot('category_id')


# ok looks fine , how many unique values are there ? 

# In[ ]:


len(data['category_id'].unique())


# great , now we are ready , let's have a final look 

# In[ ]:


data.info()


# we are lucky , no nulls 
# 
# _____
# 
# # Data Splitting
# 
# first we'll need to define X & y
# 

# In[ ]:


X = data.drop(['category_id'], axis=1, inplace=False)
y = data['category_id']


# what are their shapes ? 

# In[ ]:


X.shape


# In[ ]:


y.shape


# let's split it using sklearn

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# ____
# 
# # Building the Model
# 
# 
# how about using Gradient Boosting Classifier , with 100 estimators & 3 depth 

# In[ ]:


GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=33) 
GBCModel.fit(X_train, y_train)


# great. we need to have a look for the test score

# In[ ]:


print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))


# ok , not a very great accuracy , how about Decision Tree Classifier , it might be helpful

# In[ ]:


DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='entropy',max_depth=20,random_state=33) #criterion can be entropy
DecisionTreeClassifierModel.fit(X_train, y_train)

#Calculating Details
print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))
print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))


# great accuracy , let's use it in prediction 

# In[ ]:


#Calculating Prediction
y_pred = DecisionTreeClassifierModel.predict(X_test)
y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)
print('Predicted Value for DecisionTreeClassifierModel is : ' , y_pred[:20])

