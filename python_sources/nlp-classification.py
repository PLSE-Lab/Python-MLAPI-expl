#!/usr/bin/env python
# coding: utf-8

# # NLP Classification
# By : Hesham Asem
# 
# _____
# 
# we have 2 files sheet 1 & sheet 2
# 
# sheet 1 contain 80 replies from users to chatbot , & it either classied as offensive (flagged) or non-offensive (not flagged)
# 
# and sheet 2 contain 125 resumes , some of them looks unreal so it flagged & some are real : not flagged
# 
# we need to use NLP techniques to train our model , so he can be able to diffrentiate between them 
# 
# 
# Data File : https://www.kaggle.com/samdeeplearning/deepnlp
# 
# let's first import needed libraries
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import collections
import spacy
nlp = spacy.load('en_core_web_sm')


# and read both files

# In[ ]:


ResponseData = pd.read_csv("../input/deepnlp/Sheet_1.csv",encoding='latin-1')
ResumeData = pd.read_csv("../input/deepnlp/Sheet_2.csv",encoding='latin-1')


# so we'll start witjh responses file 
# 
# _____
# 
# 
# # Response File
# 
# let's have a look to the file

# In[ ]:


ResponseData.head()


# we'll just need 2 columns , which is response_text as X & class as y , let's drop the rest 

# In[ ]:


ResponseData.drop(['response_id','Unnamed: 3', 'Unnamed: 4','Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'],axis=1, inplace=True)
ResponseData.head()


# what is the shape ? 

# In[ ]:


ResponseData.shape


# does it contain any nulls ? 

# In[ ]:


ResponseData.info()


# great , now how many flagged & non-flagged we have here ? 

# In[ ]:


sns.countplot(x='class', data=ResponseData ,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))


# suitable ratios , now let's define the cloud function , to show most repeated words in each sector

# In[ ]:


def cloud(text):
    plt.figure(figsize=(15,15))
    plt.imshow(WordCloud(background_color="white",stopwords=set(stopwords.words('english')))
               .generate(" ".join([i for i in text.str.lower()])))
    plt.axis("off")
    plt.title("Response could words")


# now we can show most repeated words in flagged reponses

# In[ ]:


cloud(ResponseData[ResponseData['class']=='flagged']['response_text'])


# many related words appear like : suicide , anxiety , addiction 
# 
# now how not-flagged looks like

# In[ ]:


cloud(ResponseData[ResponseData['class']=='not_flagged']['response_text'])


# many normal words appear . 
# 
# but since several words appeared here , so it might mislead the training , so we have to know most common words , then remove them since they are like stop words
# 
# so we we'll define a function to know most common words

# In[ ]:


def CommonWords(text , kk=10) : 

    all_words = []

    for i in range(text.shape[0]) : 
        this_phrase = list(text)[i]
        for word in this_phrase.split() : 
            all_words.append(word)

    print(f'Total words are {len(all_words)} words')   
    print('')

    common_words = collections.Counter(all_words).most_common()
    k=0
    word_list =[]
    for word, i in common_words : 
        if not word.lower() in  nlp.Defaults.stop_words :
            print(f'The word is   {word}   repeated   {i}  times')
            word_list.append(word)
            k+=1
        if k==kk : 
            break
            
    return word_list


# then here . we'll get most common 5 words in not flagged responses

# In[ ]:


words1 = CommonWords(ResponseData[ResponseData['class']=='not_flagged']['response_text'],5)


# and here most common 5 words in flagged

# In[ ]:


words2 = CommonWords(ResponseData[ResponseData['class']=='flagged']['response_text'],5)


# now we camn add the two lists 

# In[ ]:


filtered_words = words1+words2
filtered_words


# then define the removal function 

# In[ ]:


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


# now to remove these words & make a new column call filtered_text

# In[ ]:


RemoveWords(ResponseData , 'response_text' , 'filtered_text' , filtered_words)


# now how data looks like

# In[ ]:


ResponseData.head()


# & even we can make cloud again for flagged responses

# In[ ]:


cloud(ResponseData[ResponseData['class']=='flagged']['filtered_text'])


# now words are more representative 
# 
# & cloud for nonflagged responses

# In[ ]:


cloud(ResponseData[ResponseData['class']=='not_flagged']['filtered_text'])


# great , now we need to label encode the output

# In[ ]:


enc  = LabelEncoder()
enc.fit(ResponseData['class'])
ResponseData['class'] = enc.transform(ResponseData['class'])


# how data looks like ? 

# In[ ]:


ResponseData.head()


# then we define X & y

# In[ ]:


X = ResponseData['filtered_text']
y = ResponseData['class']


# how is X & y shapes ? 

# In[ ]:


X.shape


# In[ ]:


y.shape


# then apply count vectorizer to make the sparse matrix to X

# In[ ]:


VecModel = TfidfVectorizer()
X = VecModel.fit_transform(X)

print(f'The new shape for X is {X.shape}')


# and split the data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=402)


# In[ ]:


print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# let's use Decision Tree Classifier , with gini criterion& depth 10

# In[ ]:


DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',max_depth=10,random_state=33) 
DecisionTreeClassifierModel.fit(X_train, y_train)


# how is scores ? 

# In[ ]:


print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))
print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))
print('DecisionTreeClassifierModel Classes are : ' , DecisionTreeClassifierModel.classes_)


# ok 90% is fine enough , let's predict some result

# In[ ]:


y_pred = DecisionTreeClassifierModel.predict(X_test)
y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)
print('Predicted Value for DecisionTreeClassifierModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for DecisionTreeClassifierModel is : ' , y_pred_prob[:10])


# also we can use the model to predict new phrases we just invent now
# 
# let's form a normal phrase , it should classified as not-flagged

# In[ ]:


phrase = ['I went to my friend to talk about normal issues']
enc.inverse_transform(DecisionTreeClassifierModel.predict(VecModel.transform(phrase)))


# great , now to form a weired phrase which looks like offensive

# In[ ]:


phrase = ['I know a Friend was thinking about suicide']
enc.inverse_transform(DecisionTreeClassifierModel.predict(VecModel.transform(phrase)))


# good job , not let's move to Resume Data , to apply same steps
# 
# ______
# 
# # Resume Data
# 
# we'll apply almost same steps here , as we did in responses 

# In[ ]:


ResumeData.head()


# In[ ]:


ResumeData.shape


# In[ ]:


ResumeData.info()


# In[ ]:


sns.countplot(x='class', data=ResumeData ,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))


# In[ ]:


cloud(ResumeData[ResumeData['class']=='flagged']['resume_text'])


# In[ ]:


cloud(ResumeData[ResumeData['class']=='not_flagged']['resume_text'])


# In[ ]:


words1 = CommonWords(ResumeData[ResumeData['class']=='flagged']['resume_text'],10)


# In[ ]:


words2 = CommonWords(ResumeData[ResumeData['class']=='not_flagged']['resume_text'],10)


# In[ ]:


filtered_words = words1+words2
filtered_words


# In[ ]:


RemoveWords(ResumeData , 'resume_text' , 'filtered_text' , filtered_words)
ResumeData.head()


# In[ ]:


cloud(ResumeData[ResumeData['class']=='flagged']['filtered_text'])


# In[ ]:


cloud(ResumeData[ResumeData['class']=='not_flagged']['filtered_text'])


# In[ ]:


enc.fit(ResumeData['class'])
ResumeData['class'] = enc.transform(ResumeData['class'])
ResumeData.head()


# In[ ]:


X = ResumeData['filtered_text']
y = ResumeData['class']


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


X = VecModel.fit_transform(X)

print(f'The new shape for X is {X.shape}')


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102)
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# here we'll use SVC since it will make better accuracy 

# In[ ]:


SVCModel = SVC(kernel= 'linear',# it can be also linear,poly,sigmoid,precomputed
               max_iter=10000,C=10,gamma='auto')
SVCModel.fit(X_train, y_train)


# In[ ]:


print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))


# In[ ]:


y_pred = SVCModel.predict(X_test)
print('Predicted Value for SVCModel is : ' , y_pred[:10])


# 
