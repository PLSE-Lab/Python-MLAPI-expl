#!/usr/bin/env python
# coding: utf-8

# # Loading Necessary Modules

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib


# # Loading Data

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


PATH='/kaggle/input/fake-and-real-news-dataset'
TRUE_FILE_PATH=os.path.join(PATH,'True.csv')
FAKE_FILE_PATH=os.path.join(PATH,'Fake.csv')


# In[ ]:


true_data_df=pd.read_csv(TRUE_FILE_PATH)
true_class=['True' for index in range(true_data_df.shape[0])]
fake_data_df=pd.read_csv(FAKE_FILE_PATH)
fake_class=['Fake' for index in range(fake_data_df.shape[0])]


# # Exploratory Data Analysis and pre processing

# In[ ]:


labels=['True','Fake']
class_wise_counts=[true_data_df.shape[0],fake_data_df.shape[0]]


# In[ ]:


matplotlib.rcParams['figure.figsize']=(10,10)
plt.bar(labels,class_wise_counts,align='center', alpha=0.5,color='r')
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Count vs Classes')
plt.show()
print ("Ratio of fake is to real news:",(fake_data_df.shape[0]/true_data_df.shape[0]))


# In[ ]:


true_data_df['class']=true_class
fake_data_df['class']=fake_class


# In[ ]:


fake_data_df['class']=fake_class


# In[ ]:


true_data_df.head()


# In[ ]:


fake_data_df.head()


# In[ ]:


data_frame=pd.concat([true_data_df,fake_data_df],axis='rows')


# In[ ]:


data_frame.isnull().sum()


# In[ ]:


data_frame.head()


# In[ ]:


data_frame.date.value_counts()


# ## Date contains a lot of unique values so not much value can be extracted from it hence dropping it for now

# In[ ]:


data_frame.drop('date',axis='columns',inplace=True)


# In[ ]:


data_frame.head()


# ## Looking in subject feature 

# In[ ]:


data_frame.subject.unique()


# In[ ]:


real_news_df=data_frame[data_frame.subject=='politicsNews']


# In[ ]:


real_news_df.shape


# In[ ]:


(fake_subject_keys,fake_counts)=np.unique(data_frame[data_frame['class']=='Fake'].subject,return_counts=True)
(true_subject_keys,true_counts)=np.unique(data_frame[data_frame['class']=='True'].subject,return_counts=True)


# In[ ]:


matplotlib.rcParams['figure.figsize']=(10,10)
plt.bar(fake_subject_keys,fake_counts,align='center', alpha=0.5,color='g')
plt.xlabel('Subjects')
plt.ylabel('Counts')
plt.title('FakeNewsCounts vs Subjects')
plt.show()


# In[ ]:


matplotlib.rcParams['figure.figsize']=(10,7)
plt.bar(true_subject_keys,true_counts,align='center', alpha=0.5,color='b')
plt.xlabel('Subjects')
plt.ylabel('Counts')
plt.title('TrueNewsCounts vs Subjects')
plt.show()


# ## So only politicalNews and worldnews are giving true news remaning all of them are giving fake news

# ## Converting the subject feature into one hot encoded features

# In[ ]:


subject_dummies=pd.get_dummies(data_frame.subject)


# In[ ]:


data_frame2=pd.concat([data_frame,subject_dummies],axis='columns')


# ## Cleaning the title and text seperatly

# In[ ]:


title_column=list(data_frame2.title)
text_column=list(data_frame2.text)


# In[ ]:


title_column[0]


# ## Cleaning the title and text columns using NLTK

# In[ ]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


# In[ ]:


stop_words=stopwords.words('english')
stop_words.extend(string.punctuation)


# In[ ]:


from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[ ]:


lemmatizer=WordNetLemmatizer()

def clean_data(text):
    
    clean_words=[]
    words=word_tokenize(text)
    for word in words:
        if (word.lower() not in stop_words and word.isdigit()==False):
            curr_word_pos_tag=pos_tag([word])
            
            simple_pos_tag=get_wordnet_pos(curr_word_pos_tag[0][1])
            clean_words.append(lemmatizer.lemmatize(word,simple_pos_tag))
    return clean_words

clean_title_column=[clean_data(current_column) for current_column in title_column]


# In[ ]:


clean_title_column[0]


# In[ ]:


clean_text_column=[clean_data(current_column) for current_column in text_column]


# ## Now we have a list of list where each item contains the words that are not stop words in the current text.

# ## Vectorising them so that important words can be extracted from it for converting into features

# In[ ]:


clean_title_column_list=[" ".join(list_words) for list_words in clean_title_column]
clean_text_column_list=[" ".join(list_words) for list_words in clean_text_column]


# In[ ]:


data_frame2['title']=clean_title_column_list
data_frame2['text']=clean_text_column_list


# ## Shuffling the dataframe so that we can split into train and test sets

# In[ ]:


from sklearn.utils import shuffle
data_frame3 = shuffle(data_frame2)


# In[ ]:


data_frame3.reset_index(inplace=True, drop=True)


# ## Splitting the data into 75% for training and 25% for testing

# In[ ]:


train_dataframe=data_frame3.loc[:int(0.75*data_frame3.shape[0]),:]


# In[ ]:


test_dataframe=data_frame3.loc[int(0.75*data_frame3.shape[0]):,:]


# In[ ]:


yTrain=list(train_dataframe['class'])
yTest=list(test_dataframe['class'])


# ## Since the subject features 'class' and 'subject' have already been taken care of hence dropping them. 

# In[ ]:


train_dataframe.drop(['class','subject'],axis=1,inplace=True)
test_dataframe.drop(['class','subject'],axis=1,inplace=True)


# In[ ]:


test_dataframe.reset_index(inplace=True,drop=True)
test_dataframe.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# In[ ]:


train_title_column=list(train_dataframe['title'])
train_text_column=list(train_dataframe['text'])
test_title_column=list(test_dataframe['title'])
test_text_column=list(test_dataframe['text'])


# In[ ]:


train_dataframe.drop(['title','text'],axis=1,inplace=True)
test_dataframe.drop(['title','text'],axis=1,inplace=True)


# ## Vectorisation for 'title' feature

# In[ ]:


count_vec=CountVectorizer(max_features=5000,ngram_range=(1,2))


# In[ ]:


train_title_sparse_matrix=count_vec.fit_transform(train_title_column)


# In[ ]:


test_title_sparse_matrix=count_vec.transform(test_title_column)


# In[ ]:


test_title_sparse_matrix.shape


# ## Converting the sparse matrix to dataframe for train and test set

# In[ ]:


train_dataframe_title = pd.DataFrame.sparse.from_spmatrix(train_title_sparse_matrix,columns=count_vec.get_feature_names())


# In[ ]:


test_dataframe_title=pd.DataFrame.sparse.from_spmatrix(test_title_sparse_matrix,columns=count_vec.get_feature_names())


# In[ ]:


train_dataframe_title.head()


# In[ ]:


test_dataframe_title.head()


# ## Adding the features extracted from the 'title' column as features to the train and test set

# In[ ]:


train_dataframe1=pd.concat([train_dataframe,train_dataframe_title],axis='columns')


# In[ ]:


train_dataframe1.head()


# In[ ]:


test_dataframe1=pd.concat([test_dataframe,test_dataframe_title],axis='columns')


# In[ ]:


test_dataframe1.head()


# ## Vectorisation for 'text' Column

# In[ ]:


count_vec_text=CountVectorizer(max_features=5000,ngram_range=(1,2))


# In[ ]:


train_text_sparse_matrix=count_vec_text.fit_transform(train_text_column)


# In[ ]:


test_text_sparse_matrix=count_vec_text.transform(test_text_column)


# ## Converting the sparse matrix to train and test set

# In[ ]:


train_dataframe_text = pd.DataFrame.sparse.from_spmatrix(train_text_sparse_matrix,columns=count_vec_text.get_feature_names())


# In[ ]:


train_dataframe_text.head()


# In[ ]:


test_dataframe_text=pd.DataFrame.sparse.from_spmatrix(test_text_sparse_matrix,columns=count_vec_text.get_feature_names())


# In[ ]:


test_dataframe_text.head()


# ## Adding the features extracted from 'text' column to the train and test set

# In[ ]:


train_dataframe2=pd.concat([train_dataframe1,train_dataframe_text],axis='columns')


# In[ ]:


test_dataframe2=pd.concat([test_dataframe1,test_dataframe_text],axis='columns')


# In[ ]:


train_dataframe2.head()


# In[ ]:


test_dataframe2.head()


# In[ ]:


train_dataframe2.isnull().sum()


# In[ ]:


test_dataframe2.isnull().sum()


# In[ ]:


train_dataframe2.shape


# In[ ]:


test_dataframe2.shape


# In[ ]:


train_dataframe2.shape,test_dataframe2.shape,yTrain.shape,yTest.shape


# In[ ]:


xTrain=train_dataframe2.values
xTest=test_dataframe2.values


# In[ ]:


xTrain.shape,xTest.shape,yTrain.shape,yTest.shape


# # Training model

# ## Logisitc Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
lr.fit(xTrain,yTrain)


# In[ ]:


yPredicted=lr.predict(xTest)


# In[ ]:


lr.score(xTest,yTest)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print (confusion_matrix(yTest,yPredicted))


# In[ ]:


print (classification_report(yTest,yPredicted))


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf_rf=RandomForestClassifier()
clf_rf.fit(xTrain,yTrain)


# In[ ]:


clf_rf.score(xTest,yTest)


# In[ ]:


yPredicted_rf=clf_rf.predict(xTest)


# In[ ]:


print (confusion_matrix(yTest,yPredicted_rf))


# In[ ]:


print (classification_report(yTest,yPredicted_rf))


# ## Multinomial Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


clf_mnb=MultinomialNB()


# In[ ]:


clf_mnb.fit(xTrain,yTrain)


# In[ ]:


clf_mnb.score(xTest,yTest)


# In[ ]:


yPredicted_mnb=clf_mnb.predict(xTest)


# In[ ]:


confusion_matrix(yTest,yPredicted_mnb)


# In[ ]:


print (classification_report(yTest,yPredicted_mnb))

