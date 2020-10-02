#!/usr/bin/env python
# coding: utf-8

# # Spooky Classification for 3 Authors
# By : Hesham Asem
# 
# _____
# 
# here we have 3 kinds of phrases , for 3 fammous authors (Edgar Allan Poe  , HP Lovecraft , Mary Wollstonecraft Shelley) , & we need to build a model which is able to know the author depend on the phrase 
# 
# let's first import needed libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from wordcloud import WordCloud
import collections
import spacy
nlp = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile , chi2
from sklearn.naive_bayes import MultinomialNB


# then to read the training data

# In[ ]:


data = pd.read_csv('../input/spooky-author-identification/train/train.csv')  
print(f'Data Shape is {data.shape}')
data.head()


# now to define needed functions

# In[ ]:


def show_details() : 
    global data
    for col in data.columns : 
        print(f'for feature : {col}')
        print(f'Number of Nulls is   {data[col].isna().sum()}')
        print(f'Number of Unique values is   {len(data[col].unique())}')
        print(f'random Value {data[col][0]}')
        print(f'random Value {data[col][10]}')
        print(f'random Value {data[col][20]}')
        print('--------------------------')

def CountWords(text) :  
    
    all_words = []

    for i in range(text.shape[0]) : 
        this_phrase = list(text)[i]
        for word in this_phrase.split() : 
            all_words.append(word)

    print(f'Total words are {len(all_words)} words')   
    print('')
    print(f'Total unique words are {len(set(all_words))} words')   
    
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

def SelectedData(feature , value , operation, selected_feature ):
    global data
    if operation==0 : 
        result = data[data[feature]==value][selected_feature]
    elif operation==1 : 
        result = data[data[feature] > value][selected_feature]
    elif operation==2 : 
        result = data[data[feature]< value][selected_feature]
    
    return result 

def LowerCase(feature , newfeature) : 
    global data
    def ApplyLower(text) : 
        return text.lower()
    data[newfeature] = data[feature].apply(ApplyLower)
    
def Drop(feature) :
    global data
    data.drop([feature],axis=1, inplace=True)
    data.head()
def Unique(feature) : 
    global data
    print(f'Number of unique vaure are {len(list(data[feature].unique()))} which are : \n {list(data[feature].unique())}')
    
def Encoder(feature , new_feature, drop = False) : 
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
def BPlot(feature_1,feature_2) :
    global data
    sns.barplot(x=feature_1, y=feature_2 , data=data)
    
def CPlot(feature) : 
    global data
    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))


# ____
# 
# # Text Processing
# 
# so let's start make basic processes in our Text here 
# 
# first to drop the feature id

# In[ ]:


Drop('id')


# then to lowercase all texts , & to drop original text feature

# In[ ]:


LowerCase('text' , 'lower text')
Drop('text')


# now how data looks like

# In[ ]:


data.head(20)


# let's be sure no Nulls exists

# In[ ]:


show_details()


# who are authors we have here ? 

# In[ ]:


Unique('author')


# so we need to labelencode our authors here , so EAP will be 0 , HPL will be 1 & MWS will be 2 , we need to memorize these numbers to use in predicting test data

# In[ ]:


Encoder('author' , 'author code')
data.head()


# now how many words we have in all phrases  ? 

# In[ ]:


CountWords(data['lower text'])


# about half million word , which depend on about 45K unique words
# 
# ____
# 
# 
# and we need to be sure that we have balances about of phrases for each category in the output
# 
# 

# In[ ]:


BPlot(data['author'].value_counts().index , data['author'].value_counts().values )


# great , balanced amount
# 
# _____
# 
# # Common Words
# 
# 
# let's have a look to most common used words ( stopwords will be excluded automatically from this function)

# In[ ]:


AllCommon = CommonWords(data['lower text'])


# & it will be better to make wordcloud for all words

# In[ ]:


MakeCloud(data['lower text'] , 'All Words')


# how about common words in phrases written by EAP ? 

# In[ ]:


ECommon = CommonWords(SelectedData('author','EAP',0,'lower text'))


# also we can make it in cloud form 

# In[ ]:


MakeCloud(SelectedData('author','EAP',0,'lower text') , 'EAP Words')


# then repeat same process for HPL

# In[ ]:


HCommon = CommonWords(SelectedData('author','HPL',0,'lower text'))


# In[ ]:


MakeCloud(SelectedData('author','HPL',0,'lower text') , 'HPL Words')


# and for MWS

# In[ ]:


MCommon = CommonWords(SelectedData('author','MWS',0,'lower text'))


# In[ ]:


MakeCloud(SelectedData('author','MWS',0,'lower text') , 'MWS Words')


# ____
# 
# # More Data
# 
# we might need to know more features about these phrases , which might be helpful in our training
# 
# let's make a new feature about number of words in each phrase , & check if it will be a helpful feature

# In[ ]:


data['number of words'] = data['lower text'].apply(lambda x : len(x.split()))
print('mean words for EAP is  ' , SelectedData('author', 'EAP' , 0 , 'number of words').mean())  
print('mean words for HPL is  ' , SelectedData('author', 'HPL' , 0 , 'number of words').mean())  
print('mean words for MWS is  ' , SelectedData('author', 'MWS' , 0 , 'number of words').mean())  


# ok , the 3 means are very close to each other 
# 
# how about the number of charachters

# In[ ]:


data['number of chars'] = data['lower text'].apply(lambda x : len(x))
print('mean chars for EAP is  ' , SelectedData('author', 'EAP' , 0 , 'number of chars').mean())  
print('mean chars for HPL is  ' , SelectedData('author', 'HPL' , 0 , 'number of chars').mean())  
print('mean chars for MWS is  ' , SelectedData('author', 'MWS' , 0 , 'number of chars').mean())  


# HPL prefer to write more letters , but still close to each other 
# 
# so how about number of punctuations ? 

# In[ ]:


data['number of punctuations'] = data['lower text'].apply(lambda x : len([k for k in  x if k in r'.,;:!?|\#$%^&*/']))
print('mean punctuations for EAP is  ' , SelectedData('author', 'EAP' , 0 , 'number of punctuations').mean())  
print('mean punctuations for HPL is  ' , SelectedData('author', 'HPL' , 0 , 'number of punctuations').mean())  
print('mean punctuations for MWS is  ' , SelectedData('author', 'MWS' , 0 , 'number of punctuations').mean())  


# still not a big difference . so how about number of stop words ? 

# In[ ]:


data['number of stop'] = data['lower text'].apply(lambda x : len([k for k in  x if k in nlp.Defaults.stop_words ]))
print('mean stop for EAP is  ' , SelectedData('author', 'EAP' , 0 , 'number of stop').mean())  
print('mean stop for HPL is  ' , SelectedData('author', 'HPL' , 0 , 'number of stop').mean())  
print('mean stop for MWS is  ' , SelectedData('author', 'MWS' , 0 , 'number of stop').mean())  


# almost equal each other . . 
# 
# so it might not be helpful to use any of those features
# 
# _____
# 
# # Training the Data
# 
# so let's start prepare our data to be ready for training 

# In[ ]:


data.head()


# let's define X & y , since we'll not use any of those new features 

# In[ ]:


X = data['lower text']
y = data['author code']


# then we'll have to vectorize the text & check its shape

# In[ ]:


VecModel = TfidfVectorizer()
XVec = VecModel.fit_transform(X)

print(f'The new shape for X is {XVec.shape}')


# about 25K features . . 
# 
# how about reducing them using SelectPercentile from sklearn , to its half , using chi2 function

# In[ ]:


FeatureSelection = SelectPercentile(score_func = chi2, percentile=50)
X_data = FeatureSelection.fit_transform(XVec, y)

print('X Shape is ' , X_data.shape)


# ok looks fine , not to split it into training & testing data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.33, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# now to use MultiNomial Naive Bayes model for classification , with alpha only 0.05 to get the best score

# In[ ]:


MultinomialNBModel = MultinomialNB(alpha=0.05)
MultinomialNBModel.fit(X_train, y_train)

print('MultinomialNBModel Train Score is : ' , MultinomialNBModel.score(X_train, y_train))
print('MultinomialNBModel Test Score is : ' , MultinomialNBModel.score(X_test, y_test))


# well , 87% accuracy in test score is not the best but it's acceptable , may be we can increase it by more tuning for the parameters
# 
# _____
# 
# # Predicting Test Data
# 
# now let's moving to predicting test data

# In[ ]:


data = pd.read_csv('../input/spooky-author-identification/test/test.csv')  
print(f'Test data Shape is {data.shape}')
data.head()


# we have to lower case it as well 

# In[ ]:


LowerCase('text' , 'lower text')
Drop('text')
data.head()


# then define X

# In[ ]:


X = data['lower text']


# now to apply Vectorizing Model to it , & it have bring the same 25K features

# In[ ]:


XVec = VecModel.transform(X)
print(f'The new shape for X is {XVec.shape}')


# perfect  . again we have to apply SelectPercentile Model to it , to select same half features

# In[ ]:


X_data = FeatureSelection.transform(XVec)
print('X Shape is ' , X_data.shape)


# now it's ready for predicting

# In[ ]:


y_pred = MultinomialNBModel.predict(X_data)
y_pred_prob = MultinomialNBModel.predict_proba(X_data)
print('Predicted Value for MultinomialNBModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for MultinomialNBModel is : ' , y_pred_prob[:10])


# great , now to open the submission file , to insert the answer

# In[ ]:


data = pd.read_csv('../input/spooky-author-identification/sample_submission/sample_submission.csv')  
print(f'Test data Shape is {data.shape}')
data.head()


# here they don't need the argmax prediction , but the probabilities , so we'll use predict probability method from same model , then transform it into dataframe , & concatenate to the original id feature

# In[ ]:


idd = data['id']
FinalResults = pd.DataFrame(y_pred_prob  ,columns= ['EAP','HPL','MWS'])
FinalResults.insert(0,'id',idd)


# how it looks now ? 

# In[ ]:


FinalResults.head()


# great, now submission 

# In[ ]:


FinalResults.to_csv("sample_submission.csv",index=False)


# hope you enjoyed it !
# 
