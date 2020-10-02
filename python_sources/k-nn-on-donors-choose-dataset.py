#!/usr/bin/env python
# coding: utf-8

# # DonorsChoose

# <p>
# DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. Right now, a large number of volunteers is needed to manually screen each submission before it's approved to be posted on the DonorsChoose.org website.
# </p>
# <p>
#     Next year, DonorsChoose.org expects to receive close to 500,000 project proposals. As a result, there are three main problems they need to solve:
# <ul>
# <li>
#     How to scale current manual processes and resources to screen 500,000 projects so that they can be posted as quickly and as efficiently as possible</li>
#     <li>How to increase the consistency of project vetting across different volunteers to improve the experience for teachers</li>
#     <li>How to focus volunteer time on the applications that need the most assistance</li>
#     </ul>
# </p>    
# <p>
# The goal of the competition is to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school. DonorsChoose.org can then use this information to identify projects most likely to need further review before approval.
# </p>

# ## About the DonorsChoose Data Set
# 
# The `train.csv` data set provided by DonorsChoose contains the following features:
# 
# Feature | Description 
# ----------|---------------
# **`project_id`** | A unique identifier for the proposed project. **Example:** `p036502`   
# **`project_title`**    | Title of the project. **Examples:**<br><ul><li><code>Art Will Make You Happy!</code></li><li><code>First Grade Fun</code></li></ul> 
# **`project_grade_category`** | Grade level of students for which the project is targeted. One of the following enumerated values: <br/><ul><li><code>Grades PreK-2</code></li><li><code>Grades 3-5</code></li><li><code>Grades 6-8</code></li><li><code>Grades 9-12</code></li></ul>  
#  **`project_subject_categories`** | One or more (comma-separated) subject categories for the project from the following enumerated list of values:  <br/><ul><li><code>Applied Learning</code></li><li><code>Care &amp; Hunger</code></li><li><code>Health &amp; Sports</code></li><li><code>History &amp; Civics</code></li><li><code>Literacy &amp; Language</code></li><li><code>Math &amp; Science</code></li><li><code>Music &amp; The Arts</code></li><li><code>Special Needs</code></li><li><code>Warmth</code></li></ul><br/> **Examples:** <br/><ul><li><code>Music &amp; The Arts</code></li><li><code>Literacy &amp; Language, Math &amp; Science</code></li>  
#   **`school_state`** | State where school is located ([Two-letter U.S. postal code](https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations#Postal_codes)). **Example:** `WY`
# **`project_subject_subcategories`** | One or more (comma-separated) subject subcategories for the project. **Examples:** <br/><ul><li><code>Literacy</code></li><li><code>Literature &amp; Writing, Social Sciences</code></li></ul> 
# **`project_resource_summary`** | An explanation of the resources needed for the project. **Example:** <br/><ul><li><code>My students need hands on literacy materials to manage sensory needs!</code</li></ul> 
# **`project_essay_1`**    | First application essay<sup>*</sup>  
# **`project_essay_2`**    | Second application essay<sup>*</sup> 
# **`project_essay_3`**    | Third application essay<sup>*</sup> 
# **`project_essay_4`**    | Fourth application essay<sup>*</sup> 
# **`project_submitted_datetime`** | Datetime when project application was submitted. **Example:** `2016-04-28 12:43:56.245`   
# **`teacher_id`** | A unique identifier for the teacher of the proposed project. **Example:** `bdf8baa8fedef6bfeec7ae4ff1c15c56`  
# **`teacher_prefix`** | Teacher's title. One of the following enumerated values: <br/><ul><li><code>nan</code></li><li><code>Dr.</code></li><li><code>Mr.</code></li><li><code>Mrs.</code></li><li><code>Ms.</code></li><li><code>Teacher.</code></li></ul>  
# **`teacher_number_of_previously_posted_projects`** | Number of project applications previously submitted by the same teacher. **Example:** `2` 
# 
# <sup>*</sup> See the section <b>Notes on the Essay Data</b> for more details about these features.
# 
# Additionally, the `resources.csv` data set provides more data about the resources required for each project. Each line in this file represents a resource required by a project:
# 
# Feature | Description 
# ----------|---------------
# **`id`** | A `project_id` value from the `train.csv` file.  **Example:** `p036502`   
# **`description`** | Desciption of the resource. **Example:** `Tenor Saxophone Reeds, Box of 25`   
# **`quantity`** | Quantity of the resource required. **Example:** `3`   
# **`price`** | Price of the resource required. **Example:** `9.95`   
# 
# **Note:** Many projects require multiple resources. The `id` value corresponds to a `project_id` in train.csv, so you use it as a key to retrieve all resources needed for a project:
# 
# The data set contains the following label (the value you will attempt to predict):
# 
# Label | Description
# ----------|---------------
# `project_is_approved` | A binary flag indicating whether DonorsChoose approved the project. A value of `0` indicates the project was not approved, and a value of `1` indicates the project was approved.

# ### Notes on the Essay Data
# 
# <ul>
# Prior to May 17, 2016, the prompts for the essays were as follows:
# <li>__project_essay_1:__ "Introduce us to your classroom"</li>
# <li>__project_essay_2:__ "Tell us more about your students"</li>
# <li>__project_essay_3:__ "Describe how your students will use the materials you're requesting"</li>
# <li>__project_essay_3:__ "Close by sharing why your project will make a difference"</li>
# </ul>
# 
# 
# <ul>
# Starting on May 17, 2016, the number of essays was reduced from 4 to 2, and the prompts for the first 2 essays were changed to the following:<br>
# <li>__project_essay_1:__ "Describe your students: What makes your students special? Specific details about their background, your neighborhood, and your school are all helpful."</li>
# <li>__project_essay_2:__ "About your project: How will these materials make a difference in your students' learning and improve their school lives?"</li>
# <br>For all projects with project_submitted_datetime of 2016-05-17 and later, the values of project_essay_3 and project_essay_4 will be NaN.
# </ul>
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#from gensim.models import Word2Vec
#from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os

#from plotly import plotly
#import plotly.offline as offline
#import plotly.graph_objs as go
#offline.init_notebook_mode()
from collections import Counter


# ## 1.1 Reading Data

# In[ ]:


project_data = pd.read_csv('../input/train_data.csv')
resource_data = pd.read_csv('../input/resources.csv')


# In[ ]:


print("Number of data points in train data", project_data.shape)
print('-'*50)
print("The attributes of data :", project_data.columns.values)


# In[ ]:


# how to replace elements in list python: https://stackoverflow.com/a/2582163/4084039
cols = ['Date' if x=='project_submitted_datetime' else x for x in list(project_data.columns)]


#sort dataframe based on time pandas python: https://stackoverflow.com/a/49702492/4084039
project_data['Date'] = pd.to_datetime(project_data['project_submitted_datetime'])
project_data.drop('project_submitted_datetime', axis=1, inplace=True)
project_data.sort_values(by=['Date'], inplace=True)


# how to reorder columns pandas python: https://stackoverflow.com/a/13148611/4084039
project_data = project_data[cols]


project_data.head(2)


# In[ ]:


print("Number of data points in train data", resource_data.shape)
print(resource_data.columns.values)
resource_data.head(2)


# ## 1.2 preprocessing of `project_subject_categories`

# In[ ]:


catogories = list(project_data['project_subject_categories'].values)
# remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python
cat_list = []
for i in catogories:
    temp = ""
    # consider we have text like this "Math & Science, Warmth, Care & Hunger"
    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]
        if 'The' in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"
            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')
        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"
        temp+=j.strip()+" " #" abc ".strip() will return "abc", remove the trailing spaces
        temp = temp.replace('&','_') # we are replacing the & value into 
    cat_list.append(temp.strip())
    
project_data['clean_categories'] = cat_list
project_data.drop(['project_subject_categories'], axis=1, inplace=True)

from collections import Counter
my_counter = Counter()
for word in project_data['clean_categories'].values:
    my_counter.update(word.split())

cat_dict = dict(my_counter)
sorted_cat_dict = dict(sorted(cat_dict.items(), key=lambda kv: kv[1]))


# ## 1.3 preprocessing of `project_subject_subcategories`

# In[ ]:


sub_catogories = list(project_data['project_subject_subcategories'].values)
# remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python

sub_cat_list = []
for i in sub_catogories:
    temp = ""
    # consider we have text like this "Math & Science, Warmth, Care & Hunger"
    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]
        if 'The' in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"
            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')
        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"
        temp +=j.strip()+" "#" abc ".strip() will return "abc", remove the trailing spaces
        temp = temp.replace('&','_')
    sub_cat_list.append(temp.strip())

project_data['clean_subcategories'] = sub_cat_list
project_data.drop(['project_subject_subcategories'], axis=1, inplace=True)

# count of all the words in corpus python: https://stackoverflow.com/a/22898595/4084039
my_counter = Counter()
for word in project_data['clean_subcategories'].values:
    my_counter.update(word.split())
    
sub_cat_dict = dict(my_counter)
sorted_sub_cat_dict = dict(sorted(sub_cat_dict.items(), key=lambda kv: kv[1]))


# ## 1.3 Text preprocessing

# In[ ]:


# merge two column text dataframe: 
project_data["essay"] = project_data["project_essay_1"].map(str) +                        project_data["project_essay_2"].map(str) +                         project_data["project_essay_3"].map(str) +                         project_data["project_essay_4"].map(str)


# In[ ]:


project_data.head(2)


# In[ ]:


# printing some random reviews
print(project_data['essay'].values[0])
print("="*50)
print(project_data['essay'].values[150])
print("="*50)
print(project_data['essay'].values[1000])
print("="*50)
print(project_data['essay'].values[20000])
print("="*50)
print(project_data['essay'].values[99999])
print("="*50)


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


sent = decontracted(project_data['essay'].values[20000])
print(sent)
print("="*50)


# In[ ]:


# \r \n \t remove from string python: http://texthandler.com/info/remove-line-breaks-python/
sent = sent.replace('\\r', ' ')
sent = sent.replace('\\"', ' ')
sent = sent.replace('\\n', ' ')
print(sent)


# In[ ]:


#remove spacial character: https://stackoverflow.com/a/5843547/4084039
sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
print(sent)


# In[ ]:


# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]


# In[ ]:


# Combining all the above stundents 
from tqdm import tqdm
preprocessed_essays = []
# tqdm is for printing the status bar
for sentance in tqdm(project_data['essay'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
    preprocessed_essays.append(sent.lower().strip())


# In[ ]:


# after preprocesing
preprocessed_essays[20000]


# <h2><font color='red'> 1.4 Preprocessing of `project_title`</font></h2>

# In[ ]:


project_data["project_title"]
# copy pasted the code above and changed the column
preprocessed_titles = []
for sentance in tqdm(project_data['project_title'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_titles.append(sent.lower().strip())


# In[ ]:


len(preprocessed_titles)


# <h2><font color='red'> 1.4 Preprocessing of `project_teacher_prefix`</font></h2>

# In[ ]:


# filling the nan values with "Mr." because the average value of the prefixes is closer to the category "Mr."

project_data["teacher_prefix"]= project_data["teacher_prefix"].fillna("Mr.")

project_data["teacher_prefix"]
# copy pasted the code above and changed the column
preprocessed_teacher_prefix = []
for sentance in tqdm(project_data['teacher_prefix'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_teacher_prefix.append(sent.lower().strip())


# In[ ]:


len(preprocessed_teacher_prefix)


# ## 1.5 Preparing data for models

# project_data.columns

# we are going to consider
# 
#        - school_state : categorical data
#        - clean_categories : categorical data
#        - clean_subcategories : categorical data
#        - project_grade_category : categorical data
#        - teacher_prefix : categorical data
#        
#        - project_title : text data
#        - text : text data
#        - project_resource_summary: text data (optinal)
#        
#        - quantity : numerical (optinal)
#        - teacher_number_of_previously_posted_projects : numerical
#        - price : numerical

# <h1>2. K Nearest Neighbor</h1>

# <h2>2.1 Splitting data into Train and cross validation(or test): Stratified Sampling</h2>

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# go through documentations and blogs before you start coding
# first figure out what to do, and then think about how to do.
# reading and understanding error messages will be very much helpfull in debugging your code
# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label


# In[ ]:


price_data = resource_data.groupby('id').agg({'price':'sum'}).reset_index()
project_data3 = pd.merge(project_data, price_data, on='id', how='left')

len(project_data3['price'])


# In[ ]:


final_data = project_data[["id",
                                  "school_state",
                                  "Date",
                                  "project_grade_category",
                                  "clean_categories",
                                  "clean_subcategories",
                                  "teacher_number_of_previously_posted_projects",
                                  "project_is_approved"]]

final_data["price"]= project_data3["price"]
final_data["preprocessed_teacher_prefix"]=preprocessed_teacher_prefix
final_data["preprocessed_titles"]=preprocessed_titles
final_data["preprocessed_essays"]=preprocessed_essays

final_data.shape
final_data


# In[ ]:


"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( final_data, data_label, test_size=0.20, random_state=0 ,shuffle = 0)
X_train.shape   # (87398, 11)
X_test.shape    # (21850, 11)
y_train.shape   # (87398,)
y_test.shape     # (21850,)"""


# In[ ]:


final_data1 = final_data.sample(n=50000)
final_data1.shape


# In[ ]:


final_data1.sort_values(by=['Date'], inplace=True)
li =final_data1["project_is_approved"].value_counts()
li
for val in li:
    print("percentage of values for class labels are: ",(val/len(final_data1["project_is_approved"])))


# In[ ]:


# time based splitting 
# since the data in the final_data is alredy sorted according to the time, we can use this fact
# for splitting our data into train, cv and test.
# we will divide the 109248 into the ratio 6:2:2
# 
X_train = final_data1[:40000]
X_train = X_train.drop(["project_is_approved"], axis =1 )
y_train = final_data1["project_is_approved"][:40000]


X_test = final_data1[40000:]
X_test = X_test.drop(["project_is_approved"], axis =1 )
y_test = final_data1["project_is_approved"][40000:]


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_test.shape


# In[ ]:


X_train.columns


# <h2>2.2 Make Data Model Ready: encoding numerical, categorical features</h2>

# ### 1.5.1 Vectorizing Categorical data

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# go through documentations and blogs before you start coding 
# first figure out what to do, and then think about how to do.
# reading and understanding error messages will be very much helpfull in debugging your code
# make sure you featurize train and test data separatly

# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label


# - https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/handling-categorical-and-numerical-features/
# we use count vectorizer to convert the values into one hot encoded features 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=list(sorted_cat_dict.keys()), lowercase=False, binary=True)
categories_one_hot = vectorizer.fit_transform(project_data['clean_categories'].values)
print(vectorizer.get_feature_names())
print("Shape of matrix after one hot encodig ",categories_one_hot.shape)# we use count vectorizer to convert the values into one 
vectorizer = CountVectorizer(vocabulary=list(sorted_sub_cat_dict.keys()), lowercase=False, binary=True)
sub_categories_one_hot = vectorizer.fit_transform(project_data['clean_subcategories'].values)
print(vectorizer.get_feature_names())
print("Shape of matrix after one hot encodig ",sub_categories_one_hot.shape)
# In[ ]:


# one hot encoding for  clean_categories

# we use count vectorizer to convert the values into one hot encoded features 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=False, binary=True)
vectorizer.fit(X_train['clean_categories'].values)
#print(vectorizer.get_feature_names())
#print("Shape of matrix after one hot encodig ",categories_one_hot.shape)
X_train_clean_categories_OHE = vectorizer.transform(X_train["clean_categories"].values)
#X_train_cv_clean_categories_OHE = vectorizer.transform(X_train_cv["clean_categories"].values)
X_test_clean_categories_OHE = vectorizer.transform(X_test["clean_categories"].values)

print("After Vectorizations")
print(X_train_clean_categories_OHE.shape)
#print(X_train_cv_clean_categories_OHE.shape)
print(X_test_clean_categories_OHE.shape)
print(vectorizer.get_feature_names())


# In[ ]:


# one hot encoding for  clean_subcategories

# we use count vectorizer to convert the values into one hot encoded features 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=False, binary=True)
vectorizer.fit(X_train['clean_subcategories'].values)
#print(vectorizer.get_feature_names())
#print("Shape of matrix after one hot encodig ",categories_one_hot.shape)
X_train_clean_subcategories_OHE = vectorizer.transform(X_train["clean_subcategories"].values)
#X_train_cv_clean_subcategories_OHE = vectorizer.transform(X_train_cv["clean_subcategories"].values)
X_test_clean_subcategories_OHE = vectorizer.transform(X_test["clean_subcategories"].values)

print("After Vectorizations")
print(X_train_clean_subcategories_OHE.shape)
#print(X_train_cv_clean_subcategories_OHE.shape)
print(X_test_clean_subcategories_OHE.shape)
print(vectorizer.get_feature_names())


# In[ ]:


#school_state 
# one hot encoding for  school_state

# we use count vectorizer to convert the values into one hot encoded features 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=False, binary=True)
vectorizer.fit(X_train['school_state'].values)
#print(vectorizer.get_feature_names())
#print("Shape of matrix after one hot encodig ",categories_one_hot.shape)
X_train_school_state_OHE = vectorizer.transform(X_train["school_state"].values)
#X_train_cv_school_state_OHE = vectorizer.transform(X_train_cv["school_state"].values)
X_test_school_state_OHE = vectorizer.transform(X_test["school_state"].values)

print("After Vectorizations")
print(X_train_school_state_OHE.shape)
#print(X_train_cv_school_state_OHE.shape)
print(X_test_school_state_OHE.shape)
print(vectorizer.get_feature_names())


# In[ ]:


#preprocessed_teacher_prefix

# one hot encoding for  preprocessed_teacher_prefix

# we use count vectorizer to convert the values into one hot encoded features 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=False, binary=True)
vectorizer.fit(X_train['preprocessed_teacher_prefix'].values)
#print(vectorizer.get_feature_names())
#print("Shape of matrix after one hot encodig ",categories_one_hot.shape)
X_train_preprocessed_teacher_prefix_OHE = vectorizer.transform(X_train["preprocessed_teacher_prefix"].values)
#X_train_cv_preprocessed_teacher_prefix_OHE = vectorizer.transform(X_train_cv["preprocessed_teacher_prefix"].values)
X_test_preprocessed_teacher_prefix_OHE = vectorizer.transform(X_test["preprocessed_teacher_prefix"].values)

print("After Vectorizations")
print(X_train_preprocessed_teacher_prefix_OHE.shape)
#print(X_train_cv_preprocessed_teacher_prefix_OHE.shape)
print(X_test_preprocessed_teacher_prefix_OHE.shape)
print(vectorizer.get_feature_names())


# In[ ]:


#project_grade_category 

# one hot encoding for  project_grade_category

# we use count vectorizer to convert the values into one hot encoded features 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary= list(X_train["project_grade_category"].unique()),
                             lowercase=False,
                             binary=True)

vectorizer.fit(X_train['project_grade_category'].values)
#print(vectorizer.get_feature_names())
#print("Shape of matrix after one hot encodig ",categories_one_hot.shape)
X_train_project_grade_category_OHE = vectorizer.transform(X_train["project_grade_category"].values)
#X_train_cv_project_grade_category_OHE = vectorizer.transform(X_train_cv["project_grade_category"].values)
X_test_project_grade_category_OHE = vectorizer.transform(X_test["project_grade_category"].values)

print("After Vectorizations")
print(X_train_project_grade_category_OHE.shape)
#print(X_train_cv_project_grade_category_OHE.shape)
print(X_test_project_grade_category_OHE.shape)
print(vectorizer.get_feature_names())


# In[ ]:





# ### 1.5.3 Vectorizing Numerical features
price_data = resource_data.groupby('id').agg({'price':'sum'}).reset_index()
project_data = pd.merge(project_data, price_data, on='id', how='left')
# check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# price_standardized = standardScalar.fit(project_data['price'].values)
# this will rise the error
# ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# Reshape your data either using array.reshape(-1, 1)

price_scalar = StandardScaler()
price_scalar.fit(project_data['price'].values.reshape(-1,1)) # finding the mean and standard deviation of this data
print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now standardize the data with above maen and variance.
price_standardized = price_scalar.transform(project_data['price'].values.reshape(-1, 1))
# In[ ]:


from sklearn.preprocessing import StandardScaler
price_scalar = StandardScaler()
price_scalar.fit(X_train['price'].values.reshape(-1,1))# finding the mean and standard deviation of this data
print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

X_train_price_standardized = price_scalar.transform(X_train['price'].values.reshape(-1, 1))
#X_train_cv_price_standardized = price_scalar.transform(X_train_cv['price'].values.reshape(-1, 1))
X_test_price_standardized = price_scalar.transform(X_test['price'].values.reshape(-1, 1))

print("After vectorizations")
print(X_train_price_standardized.shape, y_train.shape)
#print(X_train_cv_price_standardized.shape, y_train_cv.shape)
print(X_test_price_standardized.shape, y_test.shape)
print("="*100)

len(price_standardized)
# In[ ]:


from sklearn.preprocessing import StandardScaler
price_scalar = StandardScaler()
price_scalar.fit(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))# finding the mean and standard deviation of this data
print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

X_train_previous_projects_standardized = price_scalar.transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))
#X_train_cv_previous_projects_standardized = price_scalar.transform(X_train_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))
X_test_previous_projects_standardized = price_scalar.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))

print("After vectorizations")
print(X_train_previous_projects_standardized.shape, y_train.shape)
#print(X_train_cv_previous_projects_standardized.shape, y_train_cv.shape)
print(X_test_previous_projects_standardized.shape, y_test.shape)
print("="*100)


# In[ ]:





# <h2>2.3 Make Data Model Ready: encoding eassay, and project_title</h2>

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# go through documentations and blogs before you start coding
# first figure out what to do, and then think about how to do.
# reading and understanding error messages will be very much helpfull in debugging your code
# make sure you featurize train and test data separatly

# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label


# ### 1.5.2 Vectorizing Text data

# #### 1.5.2.1 Bag of words
# We are considering only the words which appeared in at least 10 documents(rows or projects).
vectorizer = CountVectorizer(min_df=10)
text_bow = vectorizer.fit_transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_bow.shape)
# In[ ]:


#vectorizing text data 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=10)
vectorizer.fit(X_train['preprocessed_essays'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_essay_bow = vectorizer.transform(X_train['preprocessed_essays'].values)
#X_train_cv_essay_bow = vectorizer.transform(X_train_cv['preprocessed_essays'].values)
X_test_essay_bow = vectorizer.transform(X_test['preprocessed_essays'].values)

print("After vectorizations")
print(X_train_essay_bow.shape, y_train.shape)
#print(X_train_cv_essay_bow.shape, y_train_cv.shape)
print(X_test_essay_bow.shape, y_test.shape)
print("="*100)


# In[ ]:


# you can vectorize the title also 
# before you vectorize the title make sure you preprocess it

#vectorizing text data 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=10)
vectorizer.fit(X_train['preprocessed_titles'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_title_bow = vectorizer.transform(X_train['preprocessed_titles'].values)
#X_train_cv_title_bow = vectorizer.transform(X_train_cv['preprocessed_titles'].values)
X_test_title_bow = vectorizer.transform(X_test['preprocessed_titles'].values)

print("After vectorizations")
print(X_train_title_bow.shape, y_train.shape)
#print(X_train_cv_title_bow.shape, y_train_cv.shape)
print(X_test_title_bow.shape, y_test.shape)
print("="*100)


# #### 1.5.2.2 TFIDF vectorizer

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# text_tfidf = vectorizer.fit_transform(preprocessed_essays)
# print("Shape of matrix after one hot encodig ",text_tfidf.shape)

# In[ ]:


# TFIDF for  preprocessed_essays

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
vectorizer.fit(X_train['preprocessed_essays'].values)

X_train_essays_tfidf = vectorizer.transform(X_train['preprocessed_essays'].values)
X_test_essays_tfidf = vectorizer.transform(X_test['preprocessed_essays'].values)


print("Shape of matrix after one hot encodig ",X_train_essays_tfidf.shape)
print("Shape of matrix after one hot encodig ",X_test_essays_tfidf.shape)


# In[ ]:


# TFIDF for  preprocessed_titles

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
vectorizer.fit(X_train['preprocessed_titles'].values)

X_train_title_tfidf = vectorizer.transform(X_train['preprocessed_titles'].values)
X_test_title_tfidf = vectorizer.transform(X_test['preprocessed_titles'].values)


print("Shape of matrix after one hot encodig ",X_train_title_tfidf.shape)
print("Shape of matrix after one hot encodig ",X_test_title_tfidf.shape)


# In[ ]:





# #### 1.5.2.3 Using Pretrained Models: Avg W2V

# In[ ]:


# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
# make sure you have the glove_vectors file
with open('../input/glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words =  set(model.keys())


# In[ ]:


# average Word2Vec preprocessed_essays
# compute average word2vec for each review.
X_train_preprocessed_essays_avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_train["preprocessed_essays"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    X_train_preprocessed_essays_avg_w2v_vectors.append(vector)

print(len(X_train_preprocessed_essays_avg_w2v_vectors))
print(len(X_train_preprocessed_essays_avg_w2v_vectors[0]))


# In[ ]:


# average Word2Vec preprocessed_essays
# compute average word2vec for each review.
X_test_preprocessed_essays_avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_test["preprocessed_essays"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    X_test_preprocessed_essays_avg_w2v_vectors.append(vector)

print(len(X_test_preprocessed_essays_avg_w2v_vectors))
print(len(X_test_preprocessed_essays_avg_w2v_vectors[0]))


# In[ ]:


# average Word2Vec preprocessed_titles
# compute average word2vec for each review.
X_train_preprocessed_titles_avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_train["preprocessed_titles"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    X_train_preprocessed_titles_avg_w2v_vectors.append(vector)

print(len(X_train_preprocessed_titles_avg_w2v_vectors))
print(len(X_train_preprocessed_titles_avg_w2v_vectors[0]))


# In[ ]:


# average Word2Vec preprocessed_titles
# compute average word2vec for each review.
X_test_preprocessed_titles_avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_test["preprocessed_titles"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    X_test_preprocessed_titles_avg_w2v_vectors.append(vector)

print(len(X_test_preprocessed_titles_avg_w2v_vectors))
print(len(X_test_preprocessed_titles_avg_w2v_vectors[0]))


# #### 1.5.2.3 Using Pretrained Models: TFIDF weighted W2V

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
tfidf_model = TfidfVectorizer()
tfidf_model.fit(X_train["preprocessed_essays"])
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
tfidf_words = set(tfidf_model.get_feature_names())


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
X_train_essays_tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_train["preprocessed_essays"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in glove_words) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    X_train_essays_tfidf_w2v_vectors.append(vector)

print(len(X_train_essays_tfidf_w2v_vectors))
print(len(X_train_essays_tfidf_w2v_vectors[0]))


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
X_test_essays_tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_test["preprocessed_essays"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in glove_words) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    X_test_essays_tfidf_w2v_vectors.append(vector)

print(len(X_test_essays_tfidf_w2v_vectors))
print(len(X_test_essays_tfidf_w2v_vectors[0]))


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
X_train_titles_tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_train["preprocessed_titles"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in glove_words) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    X_train_titles_tfidf_w2v_vectors.append(vector)

print(len(X_train_titles_tfidf_w2v_vectors))
print(len(X_train_titles_tfidf_w2v_vectors[0]))


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
X_test_titles_tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_test["preprocessed_titles"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in glove_words) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    X_test_titles_tfidf_w2v_vectors.append(vector)

print(len(X_test_titles_tfidf_w2v_vectors))
print(len(X_test_titles_tfidf_w2v_vectors[0]))


# ## 1.5.4 Merging all the above features

# - we need to merge all the numerical vectors i.e catogorical, text, numerical vectors

# In[ ]:



print( "clean_categories")
print(X_train_clean_categories_OHE.shape)
#print(X_train_cv_clean_categories_OHE.shape)
print(X_test_clean_categories_OHE.shape)

print( "clean_subcategories")
print(X_train_clean_subcategories_OHE.shape)
#print(X_train_cv_clean_subcategories_OHE.shape)
print(X_test_clean_subcategories_OHE.shape)

print( "school_state")
print(X_train_school_state_OHE.shape)
#print(X_train_cv_school_state_OHE.shape)
print(X_test_school_state_OHE.shape)

print("preprocessed_teacher_prefix")
print(X_train_preprocessed_teacher_prefix_OHE.shape)
#print(X_train_cv_preprocessed_teacher_prefix_OHE.shape)
print(X_test_preprocessed_teacher_prefix_OHE.shape)

print("project_grade_category")
print(X_train_project_grade_category_OHE.shape)
#print(X_train_cv_project_grade_category_OHE.shape)
print(X_test_project_grade_category_OHE.shape)

print("essay_bow")
print(X_train_essay_bow.shape, y_train.shape)
#print(X_train_cv_essay_bow.shape, y_train_cv.shape)
print(X_test_essay_bow.shape, y_test.shape)

print("title_bow")
print(X_train_title_bow.shape, y_train.shape)
#print(X_train_cv_title_bow.shape, y_train_cv.shape)
print(X_test_title_bow.shape, y_test.shape)

print("price_standardized")
print(X_train_price_standardized.shape, y_train.shape)
#print(X_train_cv_price_standardized.shape, y_train_cv.shape)
print(X_test_price_standardized.shape, y_test.shape)

print("previous_projects_standardized")
print(X_train_previous_projects_standardized.shape, y_train.shape)
#print(X_train_cv_previous_projects_standardized.shape, y_train_cv.shape)
print(X_test_previous_projects_standardized.shape, y_test.shape)

################################################################################################################################################
# TFIDF vectors         SET_2

print("title_tfidf")
print("Shape of matrix after one hot encodig ",X_train_title_tfidf.shape)
print("Shape of matrix after one hot encodig ",X_test_title_tfidf.shape)

print("essays_tfidf")
print("Shape of matrix after one hot encodig ",X_train_essays_tfidf.shape)
print("Shape of matrix after one hot encodig ",X_test_essays_tfidf.shape)


###################################################################################################################################################
# average word2vec            SET_3

print("essays_avg_w2v_vectors")
print(len(X_train_preprocessed_essays_avg_w2v_vectors))
#print(len(X_train_preprocessed_essays_avg_w2v_vectors[0]))
print(len(X_test_preprocessed_essays_avg_w2v_vectors))
#print(len(X_test_preprocessed_essays_avg_w2v_vectors[0]))

print("titles_avg_w2v_vectors")
print(len(X_train_preprocessed_titles_avg_w2v_vectors))
#print(len(X_train_preprocessed_titles_avg_w2v_vectors[0]))
print(len(X_test_preprocessed_titles_avg_w2v_vectors))
#print(len(X_test_preprocessed_titles_avg_w2v_vectors[0]))



##################################################################################################################################################
# TFIDF weighted word2vec       SET_4
print("essays_tfidf_w2v_vectors")
print(len(X_train_essays_tfidf_w2v_vectors))
#print(len(X_train_essays_tfidf_w2v_vectors[0]))
print(len(X_test_essays_tfidf_w2v_vectors))
#print(len(X_test_essays_tfidf_w2v_vectors[0]))

print("titles_tfidf_w2v_vectors")
print(len(X_train_titles_tfidf_w2v_vectors))
#print(len(X_train_titles_tfidf_w2v_vectors[0]))
print(len(X_test_titles_tfidf_w2v_vectors))
#print(len(X_test_titles_tfidf_w2v_vectors[0]))


# In[ ]:





# In[ ]:


# SET_1 
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_train_SET_1 = hstack((X_train_clean_categories_OHE,
            X_train_clean_subcategories_OHE,
            X_train_school_state_OHE,
            X_train_preprocessed_teacher_prefix_OHE,
            X_train_project_grade_category_OHE,
            X_train_essay_bow,
            X_train_title_bow,
            X_train_price_standardized,
            X_train_previous_projects_standardized))
X_train_SET_1.shape


# In[ ]:


y_train_SET_1 = y_train
y_train_SET_1.shape


# In[ ]:


X_test_SET_1 = hstack((X_test_clean_categories_OHE,
            X_test_clean_subcategories_OHE,
            X_test_school_state_OHE,
            X_test_preprocessed_teacher_prefix_OHE,
            X_test_project_grade_category_OHE,
            X_test_essay_bow,
            X_test_title_bow,
            X_test_price_standardized,
            X_test_previous_projects_standardized))
X_test_SET_1.shape


# In[ ]:


y_test_SET_1 = y_test
y_test_SET_1.shape


# In[ ]:





# In[ ]:


# SET_2 
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_train_SET_2 = hstack((X_train_clean_categories_OHE,
            X_train_clean_subcategories_OHE,
            X_train_school_state_OHE,
            X_train_preprocessed_teacher_prefix_OHE,
            X_train_project_grade_category_OHE,
            X_train_essays_tfidf,
            X_train_title_tfidf,
            X_train_price_standardized,
            X_train_previous_projects_standardized))
X_train_SET_2.shape


# In[ ]:


y_train_SET_2 = y_train
y_train_SET_2.shape


# In[ ]:


# SET_2 
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_test_SET_2 = hstack((X_test_clean_categories_OHE,
            X_test_clean_subcategories_OHE,
            X_test_school_state_OHE,
            X_test_preprocessed_teacher_prefix_OHE,
            X_test_project_grade_category_OHE,
            X_test_essays_tfidf,
            X_test_title_tfidf,
            X_test_price_standardized,
            X_test_previous_projects_standardized))
X_test_SET_2.shape


# In[ ]:


y_test_SET_2 = y_test
y_test_SET_2.shape


# In[ ]:


# SET_3
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_train_SET_3 = hstack((X_train_clean_categories_OHE,
            X_train_clean_subcategories_OHE,
            X_train_school_state_OHE,
            X_train_preprocessed_teacher_prefix_OHE,
            X_train_project_grade_category_OHE,
            X_train_preprocessed_essays_avg_w2v_vectors,
            X_train_preprocessed_titles_avg_w2v_vectors,
            X_train_price_standardized,
            X_train_previous_projects_standardized))
X_train_SET_3.shape


# In[ ]:


y_train_SET_3 = y_train
y_train_SET_3.shape


# In[ ]:


# SET_3
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_test_SET_3 = hstack((X_test_clean_categories_OHE,
            X_test_clean_subcategories_OHE,
            X_test_school_state_OHE,
            X_test_preprocessed_teacher_prefix_OHE,
            X_test_project_grade_category_OHE,
            X_test_preprocessed_essays_avg_w2v_vectors,
            X_test_preprocessed_titles_avg_w2v_vectors,
            X_test_price_standardized,
            X_test_previous_projects_standardized))
X_test_SET_3.shape


# In[ ]:


y_test_SET_3 = y_test
y_test_SET_3.shape


# In[ ]:


# SET_4
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_train_SET_4 = hstack((X_train_clean_categories_OHE,
            X_train_clean_subcategories_OHE,
            X_train_school_state_OHE,
            X_train_preprocessed_teacher_prefix_OHE,
            X_train_project_grade_category_OHE,
            X_train_essays_tfidf_w2v_vectors,
            X_train_titles_tfidf_w2v_vectors,
            X_train_price_standardized,
            X_train_previous_projects_standardized))
X_train_SET_4.shape


# In[ ]:


y_train_SET_4 = y_train
y_train_SET_4.shape


# In[ ]:


# SET_4
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_test_SET_4 = hstack((X_test_clean_categories_OHE,
            X_test_clean_subcategories_OHE,
            X_test_school_state_OHE,
            X_test_preprocessed_teacher_prefix_OHE,
            X_test_project_grade_category_OHE,
            X_test_essays_tfidf_w2v_vectors,
            X_test_titles_tfidf_w2v_vectors,
            X_test_price_standardized,
            X_test_previous_projects_standardized))
X_test_SET_4.shape


# In[ ]:


y_test_SET_4 = y_test
y_test_SET_4.shape


# # Assignment 3: Apply KNN

# <ol>
#     <li><strong>[Task-1] Apply KNN(brute force version) on these feature sets</strong>
#         <ul>
#             <li><font color='red'>Set 1</font>: categorical, numerical features + project_title(BOW) + preprocessed_essay (BOW)</li>
#             <li><font color='red'>Set 2</font>: categorical, numerical features + project_title(TFIDF)+  preprocessed_essay (TFIDF)</li>
#             <li><font color='red'>Set 3</font>: categorical, numerical features + project_title(AVG W2V)+  preprocessed_essay (AVG W2V)</li>
#             <li><font color='red'>Set 4</font>: categorical, numerical features + project_title(TFIDF W2V)+  preprocessed_essay (TFIDF W2V)</li>
#         </ul>
#     </li>
#     <br>
#     <li><strong>Hyper paramter tuning to find best K</strong>
#         <ul>
#     <li>Find the best hyper parameter which results in the maximum <a href='https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/receiver-operating-characteristic-curve-roc-curve-and-auc-1/'>AUC</a> value</li>
#     <li>Find the best hyper paramter using k-fold cross validation (or) simple cross validation data</li>
#     <li>Use gridsearch-cv or randomsearch-cv or  write your own for loops to do this task</li>
#         </ul>
#     </li>
#     <br>
#     <li>
#     <strong>Representation of results</strong>
#         <ul>
#     <li>You need to plot the performance of model both on train data and cross validation data for each hyper parameter, as shown in the figure
#     <img src='train_cv_auc.JPG' width=300px></li>
#     <li>Once you find the best hyper parameter, you need to train your model-M using the best hyper-param. Now, find the AUC on test data and plot the ROC curve on both train and test using model-M.
#     <img src='train_test_auc.JPG' width=300px></li>
#     <li>Along with plotting ROC curve, you need to print the <a href='https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/confusion-matrix-tpr-fpr-fnr-tnr-1/'>confusion matrix</a> with predicted and original labels of test data points
#     <img src='confusion_matrix.png' width=300px></li>
#         </ul>
#     </li>
#     <li><strong> [Task-2] </strong>
#         <ul>
#             <li>Select top 2000 features from feature <font color='red'>Set 2</font> using <a href='https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html'>`SelectKBest`</a>
# and then apply KNN on top of these features</li>
#             <li>
#                 <pre>
#                 from sklearn.datasets import load_digits
#                 from sklearn.feature_selection import SelectKBest, chi2
#                 X, y = load_digits(return_X_y=True)
#                 X.shape
#                 X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
#                 X_new.shape
#                 ========
#                 output:
#                 (1797, 64)
#                 (1797, 20)
#                 </pre>
#             </li>
#             <li>Repeat the steps 2 and 3 on the data matrix after feature selection</li>
#         </ul>
#     </li>
#     <br>
#     <li><strong>Conclusion</strong>
#         <ul>
#     <li>You need to summarize the results at the end of the notebook, summarize it in the table format. To print out a table please refer to this prettytable library<a href='http://zetcode.com/python/prettytable/'> link</a> 
#         <img src='summary.JPG' width=400px>
#     </li>
#         </ul>
# </ol>

# <h4><font color='red'>Note: Data Leakage</font></h4>
# 
# 1. There will be an issue of data-leakage if you vectorize the entire data and then split it into train/cv/test.
# 2. To avoid the issue of data-leakag, make sure to split your data first and then vectorize it. 
# 3. While vectorizing your data, apply the method fit_transform() on you train data, and apply the method transform() on cv/test data.
# 4. For more details please go through this <a href='https://soundcloud.com/applied-ai-course/leakage-bow-and-tfidf'>link.</a>

# pd.DataFrame(grid.cv_results_)[["mean_test_score","std_test_score","params"]]

# plt.plot(k_range, pd.DataFrame(grid.cv_results_)[["mean_test_score"]])
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated ROC Score')

# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# 
# knn = KNeighborsClassifier(weights="distance")
# k_range = list(range(200,510,10))# k_best = 2000 
# print(k_range)
# param_grid = dict(n_neighbors=k_range)
# #print(param_grid)
# 

# 
# grid = GridSearchCV(knn, param_grid, cv=10, scoring='roc_auc', return_train_score=False, n_jobs= -1)
# grid.fit(X_train_SET_2, y_train_SET_2)

# pd.DataFrame(grid.cv_results_)[["mean_test_score","std_test_score","params"]]

# plt.plot(k_range, pd.DataFrame(grid.cv_results_)[["mean_test_score"]])
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated ROC Score')

# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# 
# knn = KNeighborsClassifier()
# k_range = list(range(100,1100,100)) # k_best = 2000 
# print(k_range)
# param_grid = dict(n_neighbors=k_range)
# #print(param_grid)
# 

# 
# grid = GridSearchCV(knn, param_grid, cv=10, scoring='roc_auc', return_train_score=False, n_jobs= -1)
# grid.fit(X_train_SET_3, y_train_SET_3)

# pd.DataFrame(grid.cv_results_)[["mean_test_score","std_test_score","params"]]

# plt.plot(k_range, pd.DataFrame(grid.cv_results_)[["mean_test_score"]])
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated ROC Score')

# <h2>2.4 Appling KNN on different kind of featurization as mentioned in the instructions</h2>
# 
# <br>Apply KNN on different kind of featurization as mentioned in the instructions
# <br> For Every model that you work on make sure you do the step 2 and step 3 of instructions

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# go through documentations and blogs before you start coding
# first figure out what to do, and then think about how to do.
# reading and understanding error messages will be very much helpfull in debugging your code

# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label


# ### 2.4.1 Applying KNN brute force on BOW,<font color='red'> SET 1</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:


#########################################################################BOW###############################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
k_range = [1,5,10,15,20,25,30,40,45,51]
print(k_range)
param_grid = dict(n_neighbors=k_range)
#print(param_grid)


# In[ ]:



grid = GridSearchCV(knn, param_grid, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
grid.fit(X_train_SET_1, y_train_SET_1)


# In[ ]:


pd.DataFrame(grid.cv_results_)


# In[ ]:



train_auc= grid.cv_results_['mean_train_score']
train_auc_std= grid.cv_results_['std_train_score']
cv_auc = grid.cv_results_['mean_test_score'] 
cv_auc_std= grid.cv_results_['std_test_score']


# In[ ]:


# best K according to gridsearchCV 
best_k = grid.best_params_.get("n_neighbors")
print(best_k)


# In[ ]:



plt.plot(param_grid['n_neighbors'], train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['n_neighbors'],train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(param_grid['n_neighbors'], cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['n_neighbors'],cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(param_grid['n_neighbors'], train_auc, label='Train AUC points')
plt.scatter(param_grid['n_neighbors'], cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs K ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()


# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 291
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

neigh = KNeighborsClassifier(n_neighbors=15,n_jobs=-1)
neigh.fit(X_train_SET_1, y_train_SET_1)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:


# predicted probablities 
y_train_pred_prob = neigh.predict_proba( X_train_SET_1)    
y_test_pred_prob = neigh.predict_proba( X_test_SET_1)


# In[ ]:


# predicted values 
y_train_pred = neigh.predict( X_train_SET_1)    
y_test_pred = neigh.predict( X_test_SET_1)


# In[ ]:


# checking the model using performace metric : accuracy 

from sklearn.metrics import accuracy_score

accuracy_score_train = accuracy_score(y_train, y_train_pred)
accuracy_score_test = accuracy_score(y_test, y_test_pred)

print("accuracy on Training dataset: {0} ".format(accuracy_score_train ))
print("accuracy on Test dataset: {0} ".format(accuracy_score_test ))


# In[ ]:


# checking the model using performace metric : f1_score 

from sklearn.metrics import f1_score

f1_score_train = f1_score(y_train, y_train_pred)
f1_score_test = f1_score(y_test, y_test_pred)

print("f1_score on Training dataset: {0} ".format(f1_score_train ))
print("f1_score on Test dataset: {0} ".format(f1_score_test ))


# In[ ]:


# checking the model using performace metric : AUC 

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred_prob[:,1])
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred_prob[:,1])


print("AUC on train data : {0} ".format(auc(train_fpr, train_tpr)))
print("AUC on test data : {0} ".format(auc(test_fpr, test_tpr)))


# In[ ]:



plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_train = confusion_matrix(y_train, y_train_pred) 
print ('Confusion Matrix Train :')
print(results_train)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_train

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_test = confusion_matrix(y_test, y_test_pred) 
print ('Confusion Matrix test :')
print(results_test)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_test

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# In[ ]:





# ### 2.4.2 Applying KNN brute force on TFIDF,<font color='red'> SET 2</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:



from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
k_range = [x for x in range(1,26) if(x%2!=0)]
print(k_range)
param_grid = dict(n_neighbors=k_range)
#print(param_grid)


# In[ ]:



grid = GridSearchCV(knn, param_grid, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
grid.fit(X_train_SET_2, y_train_SET_2)


# In[ ]:


pd.DataFrame(grid.cv_results_)


# In[ ]:



train_auc= grid.cv_results_['mean_train_score']
train_auc_std= grid.cv_results_['std_train_score']
cv_auc = grid.cv_results_['mean_test_score'] 
cv_auc_std= grid.cv_results_['std_test_score']


# In[ ]:


# best K according to gridsearchCV 
best_k = grid.best_params_.get("n_neighbors")
print(best_k)


# In[ ]:



plt.plot(param_grid['n_neighbors'], train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['n_neighbors'],train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(param_grid['n_neighbors'], cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['n_neighbors'],cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(param_grid['n_neighbors'], train_auc, label='Train AUC points')
plt.scatter(param_grid['n_neighbors'], cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs K ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()


# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 15,9
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

KNN_SET_2 = KNeighborsClassifier(n_neighbors=7,n_jobs=-1)
KNN_SET_2.fit(X_train_SET_2, y_train_SET_2)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:


# predicted probablities 
y_train_pred_prob = KNN_SET_2.predict_proba( X_train_SET_2)    
y_test_pred_prob = KNN_SET_2.predict_proba( X_test_SET_2)


# In[ ]:


y_train_pred_prob[:,1].shape


# In[ ]:


# predicted values 
y_train_pred = KNN_SET_2.predict( X_train_SET_2)    
y_test_pred = KNN_SET_2.predict( X_test_SET_2)


# In[ ]:


y_train_pred.shape


# In[ ]:


# checking the model using performace metric : accuracy 

from sklearn.metrics import accuracy_score

accuracy_score_train = accuracy_score(y_train, y_train_pred)
accuracy_score_test = accuracy_score(y_test, y_test_pred)

print("accuracy on Training dataset: {0} ".format(accuracy_score_train ))
print("accuracy on Test dataset: {0} ".format(accuracy_score_test ))


# In[ ]:


# checking the model using performace metric : f1_score 

from sklearn.metrics import f1_score

f1_score_train = f1_score(y_train, y_train_pred)
f1_score_test = f1_score(y_test, y_test_pred)

print("f1_score on Training dataset: {0} ".format(f1_score_train ))
print("f1_score on Test dataset: {0} ".format(f1_score_test ))


# In[ ]:


# checking the model using performace metric : AUC 

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred_prob[:,1], drop_intermediate=False)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred_prob[:,1])


print("AUC on train data : {0} ".format(auc(train_fpr, train_tpr)))
print("AUC on test data : {0} ".format(auc(test_fpr, test_tpr)))


# In[ ]:


tr_thresholds


# In[ ]:



plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_train = confusion_matrix(y_train, y_train_pred) 
print ('Confusion Matrix Train :')
print(results_train)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_train

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_test = confusion_matrix(y_test, y_test_pred) 
print ('Confusion Matrix test :')
print(results_test)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_test

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# ### 2.4.3 Applying KNN brute force on AVG W2V,<font color='red'> SET 3</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:



from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

KNN_SET_3 = KNeighborsClassifier()
k_range = [x for x in range(1,26) if(x%2!=0)]
print(k_range)
param_grid = dict(n_neighbors=k_range)
#print(param_grid)


# In[ ]:



grid = GridSearchCV(knn, param_grid, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
grid.fit(X_train_SET_3, y_train_SET_3)


# In[ ]:


pd.DataFrame(grid.cv_results_)


# In[ ]:



train_auc= grid.cv_results_['mean_train_score']
train_auc_std= grid.cv_results_['std_train_score']
cv_auc = grid.cv_results_['mean_test_score'] 
cv_auc_std= grid.cv_results_['std_test_score']


# In[ ]:


# best K according to gridsearchCV 
best_k = grid.best_params_.get("n_neighbors")
print(best_k)


# In[ ]:



plt.plot(param_grid['n_neighbors'], train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['n_neighbors'],train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(param_grid['n_neighbors'], cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['n_neighbors'],cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(param_grid['n_neighbors'], train_auc, label='Train AUC points')
plt.scatter(param_grid['n_neighbors'], cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs K ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()


# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 15
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

KNN_SET_3 = KNeighborsClassifier(n_neighbors=15,n_jobs=-1)
KNN_SET_3.fit(X_train_SET_3, y_train_SET_3)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:


# predicted probablities 
y_train_pred_prob = KNN_SET_3.predict_proba( X_train_SET_3)    
y_test_pred_prob = KNN_SET_3.predict_proba( X_test_SET_3)


# In[ ]:


# predicted values 
y_train_pred = KNN_SET_3.predict( X_train_SET_3)    
y_test_pred = KNN_SET_3.predict( X_test_SET_3)


# In[ ]:


# checking the model using performace metric : accuracy 

from sklearn.metrics import accuracy_score

accuracy_score_train = accuracy_score(y_train, y_train_pred)
accuracy_score_test = accuracy_score(y_test, y_test_pred)

print("accuracy on Training dataset: {0} ".format(accuracy_score_train ))
print("accuracy on Test dataset: {0} ".format(accuracy_score_test ))


# In[ ]:


# checking the model using performace metric : f1_score 

from sklearn.metrics import f1_score

f1_score_train = f1_score(y_train, y_train_pred)
f1_score_test = f1_score(y_test, y_test_pred)

print("f1_score on Training dataset: {0} ".format(f1_score_train ))
print("f1_score on Test dataset: {0} ".format(f1_score_test ))


# In[ ]:


# checking the model using performace metric : AUC 

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred_prob[:,1])
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred_prob[:,1])


print("AUC on train data : {0} ".format(auc(train_fpr, train_tpr)))
print("AUC on test data : {0} ".format(auc(test_fpr, test_tpr)))


# In[ ]:



plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_train = confusion_matrix(y_train, y_train_pred) 
print ('Confusion Matrix Train :')
print(results_train)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_train

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_test = confusion_matrix(y_test, y_test_pred) 
print ('Confusion Matrix test :')
print(results_test)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_test

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# In[ ]:





# ### 2.4.4 Applying KNN brute force on TFIDF W2V,<font color='red'> SET 4</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:



from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

KNN_SET_4 = KNeighborsClassifier()
k_range = [x for x in range(1,26) if(x%2!=0)]
print(k_range)
param_grid = dict(n_neighbors=k_range)
#print(param_grid)


# In[ ]:



grid = GridSearchCV(knn, param_grid, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
grid.fit(X_train_SET_4, y_train_SET_4)


# In[ ]:


pd.DataFrame(grid.cv_results_)


# In[ ]:



train_auc= grid.cv_results_['mean_train_score']
train_auc_std= grid.cv_results_['std_train_score']
cv_auc = grid.cv_results_['mean_test_score'] 
cv_auc_std= grid.cv_results_['std_test_score']


# In[ ]:


# best K according to gridsearchCV 
best_k = grid.best_params_.get("n_neighbors")
print(best_k)


# In[ ]:



plt.plot(param_grid['n_neighbors'], train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['n_neighbors'],train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(param_grid['n_neighbors'], cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['n_neighbors'],cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(param_grid['n_neighbors'], train_auc, label='Train AUC points')
plt.scatter(param_grid['n_neighbors'], cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs K ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()


# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 15
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

KNN_SET_4 = KNeighborsClassifier(n_neighbors=15,n_jobs=-1)
KNN_SET_4.fit(X_train_SET_4, y_train_SET_4)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:


# predicted probablities 
y_train_pred_prob = KNN_SET_4.predict_proba( X_train_SET_4)    
y_test_pred_prob = KNN_SET_4.predict_proba( X_test_SET_4)


# In[ ]:


# predicted values 
y_train_pred = KNN_SET_4.predict( X_train_SET_4)    
y_test_pred = KNN_SET_4.predict( X_test_SET_4)


# In[ ]:


# checking the model using performace metric : accuracy 

from sklearn.metrics import accuracy_score

accuracy_score_train = accuracy_score(y_train, y_train_pred)
accuracy_score_test = accuracy_score(y_test, y_test_pred)

print("accuracy on Training dataset: {0} ".format(accuracy_score_train ))
print("accuracy on Test dataset: {0} ".format(accuracy_score_test ))


# In[ ]:


# checking the model using performace metric : f1_score 

from sklearn.metrics import f1_score

f1_score_train = f1_score(y_train, y_train_pred)
f1_score_test = f1_score(y_test, y_test_pred)

print("f1_score on Training dataset: {0} ".format(f1_score_train ))
print("f1_score on Test dataset: {0} ".format(f1_score_test ))


# In[ ]:


# checking the model using performace metric : AUC 

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred_prob[:,1])
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred_prob[:,1])


print("AUC on train data : {0} ".format(auc(train_fpr, train_tpr)))
print("AUC on test data : {0} ".format(auc(test_fpr, test_tpr)))


# In[ ]:



plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_train = confusion_matrix(y_train, y_train_pred) 
print ('Confusion Matrix Train :')
print(results_train)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_train

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_test = confusion_matrix(y_test, y_test_pred) 
print ('Confusion Matrix test :')
print(results_test)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_test

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# 
# knn = KNeighborsClassifier()
# k_range = list(range(100,1100,100)) # k_best = 2000 
# print(k_range)
# param_grid = dict(n_neighbors=k_range)
# #print(param_grid)
# 

# 
# grid = GridSearchCV(knn, param_grid, cv=10, scoring='roc_auc', return_train_score=False, n_jobs= -1)
# grid.fit(X_train_SET_3, y_train_SET_3)

# pd.DataFrame(grid.cv_results_)[["mean_test_score","std_test_score","params"]]

# plt.plot(k_range, pd.DataFrame(grid.cv_results_)[["mean_test_score"]])
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated ROC Score')

# <h2>2.5 Feature selection with `SelectKBest` </h2>

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# go through documentations and blogs before you start coding
# first figure out what to do, and then think about how to do.
# reading and understanding error messages will be very much helpfull in debugging your code

# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label


# <h1>3. Conclusions</h1>

# In[ ]:


# Please compare all your models using Prettytable library


# In[ ]:





# #########################################################################BOW###############################################################################
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# 
# knn = KNeighborsClassifier()
# k_range = list(range(1,101,10)) # k_best = 2433
# print(k_range)
# param_grid = dict(n_neighbors=k_range)
# #print(param_grid)
# 

# 
# grid = GridSearchCV(knn, param_grid, cv=10, scoring='roc_auc', return_train_score=True, n_jobs= -1)
# grid.fit(X_train_SET_1, y_train_SET_1)

# pd.DataFrame(grid.cv_results_)

# 
# train_auc= grid.cv_results_['mean_train_score']
# train_auc_std= grid.cv_results_['std_train_score']
# cv_auc = grid.cv_results_['mean_test_score'] 
# cv_auc_std= grid.cv_results_['std_test_score']
# 

# 
# plt.plot(parameters['n_neighbors'], train_auc, label='Train AUC')
# # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# plt.gca().fill_between(param_grid['n_neighbors'],train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')
# 
# plt.plot(param_grid['n_neighbors'], cv_auc, label='CV AUC')
# # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# plt.gca().fill_between(param_grid['n_neighbors'],cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')
# 
# plt.scatter(param_grid['n_neighbors'], train_auc, label='Train AUC points')
# plt.scatter(param_grid['n_neighbors'], cv_auc, label='CV AUC points')
# 
# 
# plt.legend()
# plt.xlabel("K: hyperparameter")
# plt.ylabel("AUC")
# plt.title("ERROR PLOTS")
# plt.grid()
# plt.show()

# #########################################################################BOW###############################################################################
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# 
# knn = KNeighborsClassifier()
# k_range = list(range(1,101,10)) # k_best = 2433
# print(k_range)
# param_grid = dict(n_neighbors=k_range)
# #print(param_grid)
# 

# 
# grid = GridSearchCV(knn, param_grid, cv=10, scoring='roc_auc', return_train_score=True, n_jobs= -1)
# grid.fit(X_train_SET_1, y_train_SET_1)

# pd.DataFrame(grid.cv_results_)

# 
# train_auc= grid.cv_results_['mean_train_score']
# train_auc_std= grid.cv_results_['std_train_score']
# cv_auc = grid.cv_results_['mean_test_score'] 
# cv_auc_std= grid.cv_results_['std_test_score']
# 

# # this is for plot 1
# 
# plt.plot(parameters['n_neighbors'], train_auc, label='Train AUC')
# # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# plt.gca().fill_between(param_grid['n_neighbors'],train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')
# 
# plt.plot(param_grid['n_neighbors'], cv_auc, label='CV AUC')
# # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# plt.gca().fill_between(param_grid['n_neighbors'],cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')
# 
# plt.scatter(param_grid['n_neighbors'], train_auc, label='Train AUC points')
# plt.scatter(param_grid['n_neighbors'], cv_auc, label='CV AUC points')
# 
# 
# plt.legend()
# plt.xlabel("K: hyperparameter")
# plt.ylabel("AUC")
# plt.title("ERROR PLOTS")
# plt.grid()
# plt.show()

# #########################################################################BOW###############################################################################
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# 
# knn1 = KNeighborsClassifier()
# k_range1 = list(range(1,7000,1000)) # k_best = 2433
# print(k_range1)
# param_grid1 = dict(n_neighbors=k_range1)
# #print(param_grid)
# 

# 
# grid1 = GridSearchCV(knn1, param_grid1, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
# grid1.fit(X_train_SET_1, y_train_SET_1)

# pd.DataFrame(grid1.cv_results_)
