#!/usr/bin/env python
# coding: utf-8

# # A Simple XGBoost Approach to solve DonorsChoose Application Screening

# In[187]:


# importing nesessary Libraries
import os
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import re
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ## Reading Data

# In[ ]:


# Reading Data
project_data = pd.read_csv('../input/train.csv')
resource_data = pd.read_csv('../input/resources.csv')
test_data=pd.read_csv('../input/test.csv')


# merging two dataframers in order to prepare train data

# In[ ]:


# Merging two dataframes 

price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
price_data.head(2)
project_data = pd.merge(project_data, price_data, on='id', how='left')
test_data=pd.merge(test_data,price_data,on='id',how='left')


# Now Let's take a first look of our data
# 

# In[ ]:


# Data Overview

print("Number of data points in train data", project_data.shape)
print('-'*50)
print("The attributes of data :", project_data.columns.values)


# As we have 'project_submitted_datetime available' with us we are sorting the data by date of project submission

# In[ ]:



cols = ['Date' if x=='project_submitted_datetime' else x for x in list(project_data.columns)]
project_data['Date'] = pd.to_datetime(project_data['project_submitted_datetime'])
project_data.drop('project_submitted_datetime', axis=1, inplace=True)
project_data.sort_values(by=['Date'], inplace=True)
project_data = project_data[cols]


# In[ ]:


cols = ['Date' if x=='project_submitted_datetime' else x for x in list(test_data.columns)]
test_data['Date'] = pd.to_datetime(test_data['project_submitted_datetime'])
test_data.drop('project_submitted_datetime', axis=1, inplace=True)
test_data.sort_values(by=['Date'], inplace=True)
test_data = test_data[cols]


# we are merging essay columns in to one to preprocess easily

# In[ ]:



project_data["essay"] = project_data["project_essay_1"].map(str) +                        project_data["project_essay_2"].map(str) +                         project_data["project_essay_3"].map(str) +                         project_data["project_essay_4"].map(str)
project_data.drop(['project_essay_1'], axis=1, inplace=True)
project_data.drop(['project_essay_2'], axis=1, inplace=True)
project_data.drop(['project_essay_3'], axis=1, inplace=True)
project_data.drop(['project_essay_4'], axis=1, inplace=True)

test_data["essay"] = test_data["project_essay_1"].map(str) +                        test_data["project_essay_2"].map(str) +                         test_data["project_essay_3"].map(str) +                         test_data["project_essay_4"].map(str)
test_data.drop(['project_essay_1'], axis=1, inplace=True)
test_data.drop(['project_essay_2'], axis=1, inplace=True)
test_data.drop(['project_essay_3'], axis=1, inplace=True)
test_data.drop(['project_essay_4'], axis=1, inplace=True)


# we need to check for NaN values in our data , as it may lead to inconsistancy 

# In[ ]:


project_data.isna().sum()


# we are getting 4 'Na' values in teacher prefix column. let's replace those with 'undefined'

# In[ ]:


project_data.fillna(value='undefined',inplace=True)
test_data.fillna(value='undefined',inplace=True)
project_data.isna().sum()


# ## Spliting Data

# In[ ]:


y = project_data['project_is_approved'].values
project_data.drop(['project_is_approved'], axis=1, inplace=True)
project_data.head(1)
x=project_data


# let's split our data in to Train and CV to cross validate performance of our model before actual submission 

# In[ ]:


x_train,x_cv,y_train,y_cv=train_test_split(x,y,test_size=0.33,stratify=y)


# ## Data Preprocessing

# ## preprocessing of `project_subject_categories`

# In[ ]:


# Function to Pre Process project subject Categories

def clean_categories(df,col='project_subject_categories'):
    catogories = list(df[col].values)
    cat_list = []
    for i in catogories:
        temp = ""
        for j in i.split(','): 
            if 'The' in j.split(): 
                j=j.replace('The','') 
            j = j.replace(' ','') 
            temp+=j.strip()+" " 
            temp = temp.replace('&','_')
        cat_list.append(temp.strip())
    
    df['clean_categories'] = cat_list
    df.drop([col], axis=1, inplace=True)

    from collections import Counter
    my_counter = Counter()
    for word in df['clean_categories'].values:
        my_counter.update(word.split())

    cat_dict = dict(my_counter)
    sorted_cat_dict = dict(sorted(cat_dict.items(), key=lambda kv: kv[1]))
    return sorted_cat_dict


# In[ ]:


sorted_dict_key_x_train=clean_categories(x_train)
sorted_dict_key_x_cv=clean_categories(x_cv)
sorted_dict_key_test=clean_categories(test_data)


# ## preprocessing of `project_subject_subcategories`

# In[ ]:


# Function to Pre Process project subject Sub Categories

def clean_subcategories(df,col='project_subject_subcategories'):
    catogories = list(df[col].values)
    sub_cat_list = []
    for i in catogories:
        temp = ""
        for j in i.split(','): 
            if 'The' in j.split(): 
                j=j.replace('The','')
            j = j.replace(' ','') 
            temp+=j.strip()+" " 
            temp = temp.replace('&','_') 
        sub_cat_list.append(temp.strip())
    
    df['clean_subcategories'] = sub_cat_list
    df.drop([col], axis=1, inplace=True)

    from collections import Counter
    my_counter = Counter()
    for word in df['clean_subcategories'].values:
        my_counter.update(word.split())

    sub_cat_dict = dict(my_counter)
    sorted_sub_cat_dict = dict(sorted(sub_cat_dict.items(), key=lambda kv: kv[1]))
    return sorted_sub_cat_dict


# In[ ]:


sorted_sub_dict_key_x_train=clean_subcategories(x_train)
sorted_sub_dict_key_x_cv=clean_subcategories(x_cv)
sorted_sub_dict_key_test=clean_subcategories(test_data)


# ## Text preprocessing

# In[ ]:


# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[ ]:


# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]


# In[ ]:


# Preprocessing Essay Column
from nltk.stem.snowball import SnowballStemmer
#from tqdm import tqdm_notebook as tqdm
stemmer=SnowballStemmer('english')
def preprocess_essay(data):
    preprocessed_data=[]
    for sentance in (data.values):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent=' '.join(stemmer.stem(word) for word in sent.split() if word not in stopwords)
        preprocessed_data.append(sent.lower().strip())
    return preprocessed_data

preprocessed_essays_x_train=preprocess_essay(x_train['essay'])
preprocessed_essays_x_cv=preprocess_essay(x_cv['essay'])
preprocessed_essays_test=preprocess_essay(test_data['essay'])


# In[ ]:


# Preprocessing project title column

def preprocess_title(data):
    preprocessed_data=[]
    for sentance in (data.values):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        sent=' '.join(stemmer.stem(word) for word in sent.split() if word not in stopwords)
        preprocessed_data.append(sent.lower().strip())
    return preprocessed_data

preprocessed_title_x_train=preprocess_title(x_train['project_title'])
preprocessed_title_x_cv=preprocess_title(x_cv['project_title'])
preprocessed_title_test=preprocess_title(test_data['project_title'])


# ## Project Grade Category

# In[ ]:


# data overview of project grade category
project_data['project_grade_category'].tail(3)


# In[ ]:


# Preprocessing Grade

def preprocess_grade(data):
    preprocessed_data=[]
    for sentence in (data.values):
        sentence=sentence.replace('Grades','')
        sentence=sentence.replace('-','to')
        preprocessed_data.append(sentence)
    return preprocessed_data
    
preprocessed_grade_x_train=preprocess_grade(x_train['project_grade_category'])
preprocessed_grade_x_cv=preprocess_grade(x_cv['project_grade_category'])
preprocessed_grade_test=preprocess_grade(test_data['project_grade_category'])




# In[ ]:


x_train['clean_grade']=preprocessed_grade_x_train
x_cv['clean_grade']=preprocessed_grade_x_cv
test_data['clean_grade']=preprocessed_grade_test


# In[ ]:


test_data.columns


# ## Vectorizing Categorical data

# As we are using Tree based Classifier we can not use One Hot Encoding to vectorize our text data ,instead we will use Label Encoding of Sckit Learn . It also comes with the limitation that it can not handle efficiently if a new category occurs at test data that was not seen in train data. To remove that probability we will consolidate the Train , Cross Validation and Test Data in order to get all categorical values.

# In[ ]:


consolidated=pd.concat([x_train,x_cv,test_data],axis=0)


# In[ ]:


# Label Encoder
cols = [
    'teacher_prefix', 
    'school_state', 
    'clean_categories', 
    'clean_subcategories', 
    'clean_grade'
]

for c in cols:
    le = LabelEncoder()
    le.fit(consolidated[c])
    x_train[c] = le.transform(x_train[c].astype(str))
    x_cv[c] = le.transform(x_cv[c].astype(str))
    test_data[c] = le.transform(test_data[c].astype(str))


# In[ ]:


# Encoding categorical features of Train Data
cat_encoded_x_train=x_train['clean_categories'].values.reshape(-1,1)
sub_cat_encoded_x_train=x_train['clean_subcategories'].values.reshape(-1,1)
state_encoded_x_train=x_train['school_state'].values.reshape(-1,1)
prefix_encoded_x_train=x_train['teacher_prefix'].values.reshape(-1,1)
grade_encoded_x_train=x_train['clean_grade'].values.reshape(-1,1)
print(cat_encoded_x_train.shape)


# In[ ]:


# Encoding categorical features of Cross Validation Data
cat_encoded_x_cv=x_cv['clean_categories'].values.reshape(-1,1)
sub_cat_encoded_x_cv=x_cv['clean_subcategories'].values.reshape(-1,1)
state_encoded_x_cv=x_cv['school_state'].values.reshape(-1,1)
prefix_encoded_x_cv=x_cv['teacher_prefix'].values.reshape(-1,1)
grade_encoded_x_cv=x_cv['clean_grade'].values.reshape(-1,1)
print(cat_encoded_x_cv.shape)


# In[ ]:


# Encoding categorical features of Test Data
cat_encoded_test=test_data['clean_categories'].values.reshape(-1,1)
sub_cat_encoded_test=test_data['clean_subcategories'].values.reshape(-1,1)
state_encoded_test=test_data['school_state'].values.reshape(-1,1)
prefix_encoded_test=test_data['teacher_prefix'].values.reshape(-1,1)
grade_encoded_test=test_data['clean_grade'].values.reshape(-1,1)
print(cat_encoded_test.shape)


# ## Vectorizing Text data

# #### TFIDF vectorizer

# We will use TF-IDF Vectorizer to vectorize our text data

# In[ ]:


# TFIDF Encoding of essay text

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(min_df=10,max_features=5000)
vectorizer.fit(preprocessed_essays_x_train)
X_train_essay_tfidf = vectorizer.transform(preprocessed_essays_x_train)
X_cv_essay_tfidf = vectorizer.transform(preprocessed_essays_x_cv)
test_essay_tfidf = vectorizer.transform(preprocessed_essays_test)
print("After vectorizations")
print(X_train_essay_tfidf.shape, y_train.shape)
print(X_cv_essay_tfidf.shape, y_cv.shape)
print(test_essay_tfidf.shape)


# In[ ]:


# TFIDF encoding of project_title ,We are considering only the words which appeared in at least 10 documents(rows or projects).
vectorizer=TfidfVectorizer(min_df=10)
vectorizer.fit(preprocessed_title_x_train)
X_train_title_tfidf = vectorizer.transform(preprocessed_title_x_train)
X_cv_title_tfidf = vectorizer.transform(preprocessed_title_x_cv)
test_title_tfidf = vectorizer.transform(preprocessed_title_test)
print("After vectorizations")
print(X_train_title_tfidf.shape, y_train.shape)
print(X_cv_title_tfidf.shape, y_cv.shape)
print(test_title_tfidf.shape)


# ### Vectorizing Numerical features

# In[ ]:


# Normalizing price
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(x_train['price'].values.reshape(-1,1))

X_train_price_norm = normalizer.transform(x_train['price'].values.reshape(-1,1))
X_cv_price_norm = normalizer.transform(x_cv['price'].values.reshape(-1,1))
test_price_norm = normalizer.transform(test_data['price'].values.reshape(-1,1))
print("After vectorizations")
print(X_train_price_norm.shape, y_train.shape)
print(X_cv_price_norm.shape, y_cv.shape)


# In[ ]:


# Normalizing teacher_number_of_previously_posted_projects
normalizer = Normalizer()
normalizer.fit(x_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_train_teacher_number_of_previously_posted_projects_norm = normalizer.transform(x_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_cv_teacher_number_of_previously_posted_projects_norm = normalizer.transform(x_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
test_teacher_number_of_previously_posted_projects_norm = normalizer.transform(test_data['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
print("After vectorizations")
print(X_train_teacher_number_of_previously_posted_projects_norm.shape, y_train.shape)
print(X_cv_teacher_number_of_previously_posted_projects_norm.shape, y_cv.shape)


#  ### Merging all the above features

# Let's merge categorical,Text and Numerical Data to prepare final the evaluation data matrixes

# In[ ]:


from scipy.sparse import hstack
x_train=hstack((cat_encoded_x_train,sub_cat_encoded_x_train,state_encoded_x_train,prefix_encoded_x_train,grade_encoded_x_train,X_train_title_tfidf,
X_train_essay_tfidf,X_train_price_norm,X_train_teacher_number_of_previously_posted_projects_norm)).tocsr()
x_cv=hstack((cat_encoded_x_cv,sub_cat_encoded_x_cv,state_encoded_x_cv,prefix_encoded_x_cv,grade_encoded_x_cv,X_cv_title_tfidf,
X_cv_essay_tfidf,X_cv_price_norm,X_cv_teacher_number_of_previously_posted_projects_norm)).tocsr()
test_eval=hstack((cat_encoded_test,sub_cat_encoded_test,state_encoded_test,prefix_encoded_test,grade_encoded_test,test_title_tfidf,
test_essay_tfidf,test_price_norm,test_teacher_number_of_previously_posted_projects_norm)).tocsr()


# Take a look at the shapes of our final matrixes

# In[ ]:


print(x_train.shape)
print(y_train.shape)
print(x_cv.shape)
print(y_cv.shape)
print(test_eval.shape)


# # XGBoost Classifier

# Here we will use XGBoost Classifier which is a Sckit Learn Wrapper for original XGBoost algorithm . For simplicity let's take 1000 estimators with learning rate as 0.01 .For better result we can use KFoldCrossValidation using GridSearch.

# In[ ]:


# Trainning XGBoost Model
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve

clf = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,   
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=5, 
                      )
clf.fit(x_train,y_train)



# In[ ]:


# Predicting Train and AUC Scores using our model
y_pred_train=clf.predict(x_train)
train_fpr,train_tpr,train_threshold=roc_curve(y_pred_train,y_train)
y_pred_cv=clf.predict(x_cv)
cv_fpr,cv_tpr,cv_threshold=roc_curve(y_pred_cv,y_cv)


# In[ ]:


# Plot results obtained from the model 
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(cv_fpr, cv_tpr, label="CV AUC ="+str(auc(cv_fpr, cv_tpr)))
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('FPR vs TPR')
plt.legend()


# ## Submission

# In[ ]:


submission = pd.DataFrame()
test_pred=clf.predict_proba(test_eval)[:,1]
submission['id'] = test_data['id']
submission['project_is_approved'] = test_pred
submission.to_csv('submission.csv', index=False)


# ## Endnotes:

# The main intention of this kernel is to get familier with the XGBoost algorithm with simple featurization techniques .I am adding some interesting reads  on the topic :
# 
# [1] https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 
# 
# [2] https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest
