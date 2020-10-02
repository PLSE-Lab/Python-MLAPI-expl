#!/usr/bin/env python
# coding: utf-8

# # Toxic Text Recognition 
# by : Hesham Asem
# 
# ________________
# 
# here we have a huge database with about 1.8 million sample size , which got a variety comments & texts , some of them are classified as offensive & toxic 
# 
# we need to build a model able to train from this database , so he can classify the test data
# 
# database  : https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
# 
# 
# so let's import needed libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
from wordcloud import WordCloud
import collections
import spacy
nlp = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile , f_classif 


# then we'll read the data

# In[ ]:


data = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv' )  
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv' )  


# and since the database is huge , we'll just got 10K rows from it to save time in training , but you can deactivate this line if you need to take the whole data

# In[ ]:


data = data[:10000]


# ____
# 
# now it's time to define needed functions
# 

# In[ ]:


def unique(feature) : 
    global data
    print(f'Number of unique vaure are {len(list(data[feature].unique()))} which are : \n {list(data[feature].unique())}')

def count_nulls() : 
    global data
    for col in data.columns : 
        if not data[col].isna().sum() == 0 : 
            print(f'Column   {col}    got   {data[col].isna().sum()} nulls  ,  Percentage : {round(100*data[col].isna().sum()/data.shape[0])} %')

def cplot(feature) : 
    global data
    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))

def encoder(feature , new_feature, drop = True) : 
    global data
    enc  = LabelEncoder()
    enc.fit(data[feature])
    data[new_feature] = enc.transform(data[feature])
    if drop == True : 
        data.drop([feature],axis=1, inplace=True)
    
def MakeCloud(text , title = 'Word Clouds' , w = 15 , h = 15):
    plt.figure(figsize=(w,h))
    plt.imshow(WordCloud(background_color="white",stopwords=set(stopwords.words('english')))
               .generate(" ".join([i for i in text.str.lower()])))
    plt.axis("off")
    plt.title(title)
    plt.show()


def SelectedData(data , feature , value , operation, selected_feature ):
    if operation==0 : 
        result = data[data[feature]==value][selected_feature]
    elif operation==1 : 
        result = data[data[feature] > value][selected_feature]
    elif operation==2 : 
        result = data[data[feature]< value][selected_feature]
    
    return result 



def CommonWords(text ,show = True , kk=10) : 
    all_words = []

    for i in range(text.shape[0]) : 
        this_phrase = list(text)[i]
        for word in this_phrase.split() : 
            all_words.append(word)
    common_words = collections.Counter(all_words).most_common()
    k=0
    word_list =[]
    for word, i in common_words : 
        if not word.lower() in  nlp.Defaults.stop_words :
            if show : 
                print(f'The word is   {word}   repeated   {i}  times')
            word_list.append(word)
            k+=1
        if k==kk : 
            break
            
    return word_list

def RemoveWords(data , feature , new_feature, words_list ) : 
    new_column = []
    for i in range(data.shape[0]) : 
        this_phrase = data[feature][i]
        new_phrase = []
        for word in this_phrase.split() : 
            if not word.lower() in words_list : 
                new_phrase.append(word)
        new_column.append(' '.join(new_phrase))
    
    data.insert(data.shape[1],new_feature,new_column)
    

    
def CountWords(text) :  
    
    all_words = []

    for i in range(text.shape[0]) : 
        this_phrase = list(text)[i]
        for word in this_phrase.split() : 
            all_words.append(word)

    print(f'Total words are {len(all_words)} words')   
    print('')
    print(f'Total unique words are {len(set(all_words))} words')   

def SlicedData(feature_list, dropna = False) : 
    global data
    if dropna :
        return data.loc[:, feature_list ].dropna()
    else : 
        return data.loc[:, feature_list ]


# 
# ____
# 
# 
# # Data Processing
# 
# now how data looks like

# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.describe()


# ______
# 
# so we need to have a close look to some features

# In[ ]:


SlicedData(['obscene','identity_attack', 'insult' , 'thread']).describe()


# In[ ]:


SlicedData(['asian','atheist', 'bisexual' , 'black']).describe()


# # Treating Output
# 
# since output 'target' is a numerical value & the model shuld be regression , we'll concert the output into 4 classes , depend on how toxic is the text 
# 
# so instead have a huge numbers in the output , it will be either 0 , 1, 2 or 3 , & this to make the model easier , specially that we have a huge sample size & will use a big number of features (due to TF Vectorizer)
# 
# so let's see max & min value for the output

# In[ ]:


data['target'].min() , data['target'].max()


# great , now if we multiply 3 time all numbers then round it into integers , we'll have all outputs either 0 , 1 , 2 or 3 , depend on how toxic is the text

# In[ ]:


data['target sector'] = round(data['target']*3)


# let's check it

# In[ ]:


unique('target sector')


# now how is the distribution of output

# In[ ]:


cplot('target sector')


# almost 90% of text is non-toxic , now how about the toxic texts , lets temporarly drop all non-toxic text , to see distribution of ther categories

# In[ ]:


temp_data = data[data['target sector'] > 0]['target sector']


# now to pie chart it

# In[ ]:


plt.figure(figsize=(10,10))
plt.pie(temp_data.value_counts(),labels=list(temp_data.value_counts().index),autopct ='%1.2f%%',labeldistance = 1.1)
plt.show()


# almost two-third of text are kinda toxic , then moderate & little bit which is very toxic
# 
# now how about nulls in the database 

# In[ ]:


count_nulls()


# _____
# 
# # Treating Text
# 
# now it's time to focus in text its self before we build the model 
# 
# first we need to lowercase all words , to avoid any misleading in training

# In[ ]:


data['comments']  =  data['comment_text'].str.lower()


# now how it looks like

# In[ ]:


SlicedData(['comment_text' , 'comments']).head(20)


# then we need to know the number of total words & unique words in the whole text 

# In[ ]:


CountWords(data['comments'])


# _____
# 
# almost a 600K word in only 10K row , which based on 54K unqiue words
# 
# now we need to know most common words in the whole text , & we'll use the feature 'comments' , which is lowered case , not the original feature 'comment_text'

# In[ ]:


common = CommonWords(data['comments'])


# it looks like these 10 words are non-leading words , which might appear in toxic or non-toxic words , so it'll be a good idea to remove it from all phrases , to ust leave the most important words
# 
# let's create a new feature called 'filtered comments' , which contain all phrases exclude these common words

# In[ ]:


RemoveWords(data , 'comments' , 'filtered comments', common)
SlicedData(['comments' , 'filtered comments']).head(20)


# great , now phrases become more extinctive 
# 
# _____
# 
# # Cloud Words
# 
# it's useful to use wordcloud tool , to know most repeated words 
# 
# let's see most repeated words now in all filtered comments , after we eliminate those 10 common words

# In[ ]:


MakeCloud(data['filtered comments'])


# how about to see the could words for each category . . 
# 
# we'll make a function now , which will show cloud words for only rows which got offensive value more than 0.1 in each category , of the 24 category we have here 
# 
# & to make it easier for us to read it , we'll show it 3 by 3 
# 

# In[ ]:


def showclouds(n) : 
    this_list = ['asian', 'atheist', 'bisexual','black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
                 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability','jewish', 'latino', 'male', 'muslim',
                 'other_disability','other_gender', 'other_race_or_ethnicity', 'other_religion','other_sexual_orientation', 
                 'physical_disability','psychiatric_or_mental_illness', 'transgender', 'white' ]

    for item in this_list[n*3:(n*3)+3] : 
        this_data =  SelectedData(data ,item , 0.1 , 1 , 'filtered comments')
        print(f'for item    {item}')
        print(f'Number of selected rows {this_data.shape[0]}')
        print('common words : ')
        _ = CommonWords(this_data)
        if this_data.shape[0] >0 : 
            MakeCloud(this_data , str(f'Word Cloud for {item}'), 8 ,8)
        print('--------------------------')


# ____
# 
# 
# now start with :   'asian', 'atheist', 'bisexual'

# In[ ]:


showclouds(0)


# _______
# 
# then :'black', 'buddhist', 'christian'

# In[ ]:


showclouds(1)


# _____
# 
# then :  'female', 'heterosexual', 'hindu'

# In[ ]:


showclouds(2)


# ______
# 
# now : 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability','jewish'

# In[ ]:


showclouds(3)


# ______
# 
# now  :  'latino', 'male', 'muslim'

# In[ ]:


showclouds(4)


# _____
# 
# then : 'other_disability','other_gender', 'other_race_or_ethnicity'

# In[ ]:


showclouds(5)


# ______
# 
# then : 'other_religion','other_sexual_orientation'  ,'physical_disability'

# In[ ]:


showclouds(6)


# ____
# 
# and last : 'psychiatric_or_mental_illness', 'transgender', 'white'

# In[ ]:


showclouds(7)


# it was so obvious how distinctive words are appear in each category , & that will help us in the classification
# 
# _____
# 
# # Data Preparing
# 
# now we are ready to define X & y 

# In[ ]:


X = data['filtered comments']
y = data['target sector']


# how X looks like ? 

# In[ ]:


X.head(10)


# & y ? 

# In[ ]:


y.head(10)


# we need to be sure there is no nulls in both of them 

# In[ ]:


X.isnull().sum() , y.isnull().sum()


# then we'll use TF Vectorizer tool , to create sparse matrix for all words

# In[ ]:


VecModel = TfidfVectorizer()
X = VecModel.fit_transform(X)
print(f'The new shape for X is {X.shape}')


# ok , almost 25K feature , which is so much & will consume a huge amount of time , specially that we have 10K sample size , so we'll use sklearn to only select 1% of those features

# In[ ]:


FeatureSelection = SelectPercentile(score_func = f_classif, percentile=1)
X = FeatureSelection.fit_transform(X, y)


# now how X looks like

# In[ ]:


print('X Shape is ' , X.shape)


# great , then split them 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# now we are ready for traning
# 
# _____
# 
# # Build the Model
# 
# let's use GBC since it's more suitable for big amout of data . 
# 
# we'll use 500 estimators with max_depth =5
# 
# let's start the train & check accuracy for train & test data

# In[ ]:


GBCModel = GradientBoostingClassifier(n_estimators=500,max_depth=5,random_state=33) 
GBCModel.fit(X_train, y_train)


print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))


# ok 84% is not a bad accuracy , might be better if we increase the data more than 100K but it will need more time
# 
# now let's predict the Test Data. . 

# In[ ]:


test.head()


# let's lower case it as we did in trainging data

# In[ ]:


test['comments']  =  test['comment_text'].str.lower()
test.head()


# and put it in the variable X_test

# In[ ]:


X_test = test['comments']


# what is the shape 

# In[ ]:


X_test.shape


# then we have to transform it with the Vectorize Model , which fitted in the training data

# In[ ]:


X_test = VecModel.transform(X_test)


# the data now should have 25K feature
# 

# In[ ]:


X_test.shape


# erfect , now we have to apply again the feature selection model , to only pick 1% of same features 

# In[ ]:


X_test = FeatureSelection.transform(X_test)


# now it should have only 257 feature

# In[ ]:


X_test.shape


# now it's ready for predicting using the same GBC Model

# In[ ]:


y_pred = GBCModel.predict(X_test)
y_pred_prob = GBCModel.predict_proba(X_test)
print('Predicted Value for GBCModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob[:10])


# great , well done !
# 
# hope you like it & found it useful . . 

# In[ ]:




