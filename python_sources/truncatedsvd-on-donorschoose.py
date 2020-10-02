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


project_data = pd.read_csv('../input/donors-choose-elysian/train_data.csv')
resource_data = pd.read_csv('../input/donors-choose-elysian/resources.csv')


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
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't", "nannan"]


# In[ ]:


from nltk.tokenize import word_tokenize 

def remove_stopwords(senta):
    # tokenizing the string ,and removing the stop words.
    example_sent = senta
    stop_words = set(stopwords) 
    word_tokens = word_tokenize(example_sent) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    # concatinating the list to form a string again because the decontracted() accepts strings.
    s = " "
    for word in filtered_sentence:

        s = s + str(word) + str(" ")
    return s


# In[ ]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
# import nltk

sid = SentimentIntensityAnalyzer()


sentiment_score_of_essays_neg = []
sentiment_score_of_essays_neu = []
sentiment_score_of_essays_pos = []
sentiment_score_of_essays_compound = []

def store_sentiment_score(string):

    ss = sid.polarity_scores(string)
    sentiment_score_of_essays_neg.append(ss.get("neg"))
    sentiment_score_of_essays_neu.append(ss.get("neu"))
    sentiment_score_of_essays_pos.append(ss.get("pos"))
    sentiment_score_of_essays_compound.append(ss.get("compound"))
    
#for k in ss:
#    print('{0}: {1}, '.format(k, ss[k]), end='')

# we can use these 4 things as features/attributes (neg, neu, pos, compound)
# neg: 0.0, neu: 0.753, pos: 0.247, compound: 0.93


# In[ ]:


from nltk.tokenize import word_tokenize 

number_of_words_in_essays = [] 
def count_words_essays(string):
    word_tokens = word_tokenize(string)
    number_of_words_in_essays.append(len(word_tokens))
    
 


# In[ ]:


number_of_words_in_title = [] 
def count_words_title(string):
    word_tokens = word_tokenize(string)
    number_of_words_in_title.append(len(word_tokens))
 


# In[ ]:


# Combining all the above stundents 
from tqdm import tqdm
preprocessed_essays = []
# tqdm is for printing the status bar
for sentance in tqdm(project_data['essay'].values):
    sentance = sentance.lower()
    count_words_essays(sentance)
    store_sentiment_score(sentance)
    sentance = remove_stopwords(sentance)
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


# In[ ]:


len(number_of_words_in_essays)


# <h2><font color='red'> 1.4 Preprocessing of `project_title`</font></h2>

# In[ ]:


project_data["project_title"]
# copy pasted the code above and changed the column
preprocessed_titles = []
for sentance in tqdm(project_data['project_title'].values):
    sentance = sentance.lower()
    count_words_title(sentance)
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_titles.append(sent.lower().strip())


# In[ ]:


len(number_of_words_in_title)


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

# <h1>2.TruncatedSVD </h1>

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


quantity_data = resource_data.groupby('id').agg({'quantity':'sum'}).reset_index()
project_data4 = pd.merge(project_data, quantity_data, on='id', how='left')

len(project_data4['quantity'])


# In[ ]:


final_data = project_data[[       "id",
                                  "school_state",
                                  "Date",
                                  "project_grade_category",
                                  "clean_categories",
                                  "clean_subcategories",
                                  "teacher_number_of_previously_posted_projects",
                                  "project_is_approved"]]

final_data["price"]= project_data3["price"]
final_data["quantity"] = project_data4["quantity"]
final_data["preprocessed_teacher_prefix"]=preprocessed_teacher_prefix
final_data["preprocessed_titles"]=preprocessed_titles
final_data["preprocessed_essays"]=preprocessed_essays

final_data["sentiment_score_of_essays_neg"]=sentiment_score_of_essays_neg
final_data["sentiment_score_of_essays_neu"]=sentiment_score_of_essays_neu
final_data["sentiment_score_of_essays_pos"]=sentiment_score_of_essays_pos
final_data["sentiment_score_of_essays_compound"]=sentiment_score_of_essays_compound

final_data["number_of_words_in_essays"]=number_of_words_in_essays
final_data["number_of_words_in_title"]=number_of_words_in_title


final_data.shape


# In[ ]:


final_data['title_and_essay'] = final_data['preprocessed_essays'] +' '+ final_data['preprocessed_titles']
final_data


# In[ ]:


"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( final_data, data_label, test_size=0.20, random_state=0 ,shuffle = 0)
X_train.shape   # (87398, 11)
X_test.shape    # (21850, 11)
y_train.shape   # (87398,)
y_test.shape     # (21850,)"""


# In[ ]:


final_data1 = final_data
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
X_train = final_data1[:87399]
X_train = X_train.drop(["project_is_approved"], axis =1 )
y_train = final_data1["project_is_approved"][:87399]


X_test = final_data1[87399:]
X_test = X_test.drop(["project_is_approved"], axis =1 )
y_test = final_data1["project_is_approved"][87399:]


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

feature_names_clean_categories = vectorizer.get_feature_names()
feature_names_clean_categories


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

print(vectorizer.get_feature_names())
feature_names_clean_subcategories = vectorizer.get_feature_names()
feature_names_clean_subcategories


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
feature_names_school_state = vectorizer.get_feature_names()
feature_names_school_state


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
feature_names_preprocessed_teacher_prefix = vectorizer.get_feature_names()
feature_names_preprocessed_teacher_prefix


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
feature_names_project_grade_category = vectorizer.get_feature_names()
feature_names_project_grade_category


# In[ ]:





# ### 1.5.3 Vectorizing Numerical features

# In[ ]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['price'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['price'].values.reshape(-1,1))

X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
#X_cv_price_norm = normalizer.transform(X_cv['price'].values.reshape(-1,1))
X_test_price_norm = normalizer.transform(X_test['price'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_price_norm.shape, y_train.shape)
#print(X_cv_price_norm.shape, y_cv.shape)
print(X_test_price_norm.shape, y_test.shape)
print("="*100)
feature_names_price = ["price"]


# In[ ]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['teacher_number_of_previously_posted_projects'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))

X_train_teacher_number_of_previously_posted_projects_norm = normalizer.transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
#X_cv_teacher_number_of_previously_posted_projects_norm = normalizer.transform(X_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_test_teacher_number_of_previously_posted_projects_norm = normalizer.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_teacher_number_of_previously_posted_projects_norm.shape, y_train.shape)
#print(X_cv_teacher_number_of_previously_posted_projects_norm.shape, y_cv.shape)
print(X_test_teacher_number_of_previously_posted_projects_norm.shape, y_test.shape)
print("="*100)
feature_names_teacher_number_of_previously_posted_projects = ["teacher_number_of_previously_posted_projects"]


# In[ ]:


len(X_test_teacher_number_of_previously_posted_projects_norm)


# sentiment_score_of_essays_neg',
#        'sentiment_score_of_essays_neu', 'sentiment_score_of_essays_pos',
#        'sentiment_score_of_essays_compound'

# In[ ]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['sentiment_score_of_essays_neg'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['sentiment_score_of_essays_neg'].values.reshape(-1,1))

X_train_sentiment_score_of_essays_neg_norm = normalizer.transform(X_train['sentiment_score_of_essays_neg'].values.reshape(-1,1))
#X_cv_sentiment_score_of_essays_neg_norm = normalizer.transform(X_cv['sentiment_score_of_essays_neg'].values.reshape(-1,1))
X_test_sentiment_score_of_essays_neg_norm = normalizer.transform(X_test['sentiment_score_of_essays_neg'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_sentiment_score_of_essays_neg_norm.shape, y_train.shape)
#print(X_cv_sentiment_score_of_essays_neg_norm.shape, y_cv.shape)
print(X_test_sentiment_score_of_essays_neg_norm.shape, y_test.shape)
print("="*100)
#feature_names_sentiment_score_of_essays_neg = ["sentiment_score_of_essays_neg"]


# In[ ]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['sentiment_score_of_essays_neu'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['sentiment_score_of_essays_neu'].values.reshape(-1,1))

X_train_sentiment_score_of_essays_neu_norm = normalizer.transform(X_train['sentiment_score_of_essays_neu'].values.reshape(-1,1))
#X_cv_sentiment_score_of_essays_neu_norm = normalizer.transform(X_cv['sentiment_score_of_essays_neu'].values.reshape(-1,1))
X_test_sentiment_score_of_essays_neu_norm = normalizer.transform(X_test['sentiment_score_of_essays_neu'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_sentiment_score_of_essays_neu_norm.shape, y_train.shape)
#print(X_cv_sentiment_score_of_essays_neu_norm.shape, y_cv.shape)
print(X_test_sentiment_score_of_essays_neu_norm.shape, y_test.shape)
print("="*100)
#feature_names_sentiment_score_of_essays_neu = ["sentiment_score_of_essays_neu"]


# In[ ]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['sentiment_score_of_essays_pos'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['sentiment_score_of_essays_pos'].values.reshape(-1,1))

X_train_sentiment_score_of_essays_pos_norm = normalizer.transform(X_train['sentiment_score_of_essays_pos'].values.reshape(-1,1))
#X_cv_sentiment_score_of_essays_pos_norm = normalizer.transform(X_cv['sentiment_score_of_essays_pos'].values.reshape(-1,1))
X_test_sentiment_score_of_essays_pos_norm = normalizer.transform(X_test['sentiment_score_of_essays_pos'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_sentiment_score_of_essays_pos_norm.shape, y_train.shape)
#print(X_cv_sentiment_score_of_essays_pos_norm.shape, y_cv.shape)
print(X_test_sentiment_score_of_essays_pos_norm.shape, y_test.shape)
print("="*100)
#feature_names_sentiment_score_of_essays_pos = ["sentiment_score_of_essays_pos"]


# In[ ]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['sentiment_score_of_essays_compound'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['sentiment_score_of_essays_compound'].values.reshape(-1,1))

X_train_sentiment_score_of_essays_compound_norm = normalizer.transform(X_train['sentiment_score_of_essays_compound'].values.reshape(-1,1))
#X_cv_sentiment_score_of_essays_compound_norm = normalizer.transform(X_cv['sentiment_score_of_essays_compound'].values.reshape(-1,1))
X_test_sentiment_score_of_essays_compound_norm = normalizer.transform(X_test['sentiment_score_of_essays_compound'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_sentiment_score_of_essays_compound_norm.shape, y_train.shape)
#print(X_cv_sentiment_score_of_essays_compound_norm.shape, y_cv.shape)
print(X_test_sentiment_score_of_essays_compound_norm.shape, y_test.shape)
print("="*100)
#feature_names_sentiment_score_of_essays_compound = ["sentiment_score_of_essays_compound"]


# In[ ]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['number_of_words_in_essays'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['number_of_words_in_essays'].values.reshape(-1,1))

X_train_number_of_words_in_essays_norm = normalizer.transform(X_train['number_of_words_in_essays'].values.reshape(-1,1))
#X_cv_number_of_words_in_essays_norm = normalizer.transform(X_cv['number_of_words_in_essays'].values.reshape(-1,1))
X_test_number_of_words_in_essays_norm = normalizer.transform(X_test['number_of_words_in_essays'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_number_of_words_in_essays_norm.shape, y_train.shape)
#print(X_cv_number_of_words_in_essays_norm.shape, y_cv.shape)
print(X_test_number_of_words_in_essays_norm.shape, y_test.shape)
print("="*100)
#feature_names_number_of_words_in_essays = ["number_of_words_in_essays"]


# In[ ]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['number_of_words_in_title'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['number_of_words_in_title'].values.reshape(-1,1))

X_train_number_of_words_in_title_norm = normalizer.transform(X_train['number_of_words_in_title'].values.reshape(-1,1))
#X_cv_number_of_words_in_title_norm = normalizer.transform(X_cv['number_of_words_in_title'].values.reshape(-1,1))
X_test_number_of_words_in_title_norm = normalizer.transform(X_test['number_of_words_in_title'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_number_of_words_in_title_norm.shape, y_train.shape)
#print(X_cv_number_of_words_in_title_norm.shape, y_cv.shape)
print(X_test_number_of_words_in_title_norm.shape, y_test.shape)
print("="*100)
#feature_names_number_of_words_in_title = ["number_of_words_in_title"]


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

# #### 1.5.2.2 TFIDF vectorizer

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# text_tfidf = vectorizer.fit_transform(preprocessed_essays)
# print("Shape of matrix after one hot encodig ",text_tfidf.shape)

# # TFIDF for  title_and_essays
# 
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# vectorizer.fit(X_train['title_and_essay'].values)
# 
# X_train_essays_tfidf = vectorizer.transform(X_train['title_and_essay'].values)
# X_test_essays_tfidf = vectorizer.transform(X_test['title_and_essay'].values)
# 
# 
# print("Shape of matrix after one hot encodig ",X_train_essays_tfidf.shape)
# print("Shape of matrix after one hot encodig ",X_test_essays_tfidf.shape)
# 
# feature_names_essays_tfidf = vectorizer.get_feature_names()
# #feature_names_essays_tfidf
# 

# vectorizer.idf_

# # TFIDF for  preprocessed_titles
# 
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# vectorizer.fit(X_train['preprocessed_titles'].values)
# 
# X_train_title_tfidf = vectorizer.transform(X_train['preprocessed_titles'].values)
# X_test_title_tfidf = vectorizer.transform(X_test['preprocessed_titles'].values)
# 
# 
# print("Shape of matrix after one hot encodig ",X_train_title_tfidf.shape)
# print("Shape of matrix after one hot encodig ",X_test_title_tfidf.shape)
# 
# feature_names_title_tfidf = vectorizer.get_feature_names()
# feature_names_title_tfidf
# 

# In[ ]:





# ## 1.5.4 Merging all the above features

# # Applying TruncatedSVD on tfidf vectors

print("essays_tfidf")
print("Shape of matrix after one hot encodig ",X_train_essays_tfidf.shape)
print("Shape of matrix after one hot encodig ",X_test_essays_tfidf.shape)
from sklearn.decomposition import TruncatedSVD

#best_componets = np.linspace(1,5000,50)
best_componets = 2000
print(best_componets)
svd = TruncatedSVD(n_components=best_componets, n_iter=7, random_state=42)
svd.fit(X_train_essays_tfidf)
X_train_essays_tfidf_svd = svd.transform(X_train_essays_tfidf)
X_test_essays_tfidf_svd = svd.transform(X_test_essays_tfidf)print(svd.explained_variance_ratio_)  print(X_train_essays_tfidf_svd.shape)
print(X_test_essays_tfidf_svd.shape)
# ## 1.5.4 Merging all the above features

# - we need to merge all the numerical vectors i.e catogorical, text, numerical vectors

# # making a list of all the vectorized features for SET_1 
# list_of_vectorized_features_SET_1 =  (feature_names_clean_categories +
#                                 feature_names_clean_subcategories +
#                                 feature_names_school_state + 
#                                 feature_names_preprocessed_teacher_prefix +
#                                 feature_names_project_grade_category +
#                                 feature_names_essay_bow +
#                                 feature_names_title_bow +
#                                 feature_names_price+
#                                 feature_names_teacher_number_of_previously_posted_projects)
#                                 
# len(list_of_vectorized_features_SET_1)                                

# In[ ]:





# # making a list of all the vectorized features for SET_2
# list_of_vectorized_features_SET_2 =  (feature_names_clean_categories +
#                                     feature_names_clean_subcategories +
#                                     feature_names_school_state + 
#                                     feature_names_preprocessed_teacher_prefix +
#                                     feature_names_project_grade_category +
#                                     feature_names_essays_tfidf +
#                                     feature_names_title_tfidf +
#                                     feature_names_price+
#                                     feature_names_teacher_number_of_previously_posted_projects)
#                                 
# len(list_of_vectorized_features_SET_2)                                

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

            X_train_price_norm,
            X_train_teacher_number_of_previously_posted_projects_norm,
            X_train_sentiment_score_of_essays_neg_norm,
            X_train_sentiment_score_of_essays_neu_norm,
            X_train_sentiment_score_of_essays_pos_norm,
            X_train_sentiment_score_of_essays_compound_norm,
            X_train_number_of_words_in_essays_norm,
            X_train_number_of_words_in_title_norm))

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

            X_test_price_norm,
            X_test_teacher_number_of_previously_posted_projects_norm,
            X_test_sentiment_score_of_essays_neg_norm,
            X_test_sentiment_score_of_essays_neu_norm,
            X_test_sentiment_score_of_essays_pos_norm,
            X_test_sentiment_score_of_essays_compound_norm,
            X_test_number_of_words_in_essays_norm,
            X_test_number_of_words_in_title_norm))

X_test_SET_2.shape


# In[ ]:


y_test_SET_2 = y_test
y_test_SET_2.shape


# In[ ]:





# # Assignment 11: TruncatedSVD

# - <font color='red'>step 1</font> Select the top 2k words from essay text and project_title (concatinate essay text with project title and then find the top 2k words) based on their <a href='https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html'>`idf_`</a> values 
# - <font color='red'>step 2</font> Compute the co-occurance matrix with these 2k words, with window size=5 (<a href='https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/'>ref</a>)
#     <img src='cooc.JPG' width=300px>
# - <font color='red'>step 3</font> Use <a href='http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html'>TruncatedSVD</a> on calculated co-occurance matrix and reduce its dimensions, choose the number of components (`n_components`) using <a href='https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/pca-code-example-using-non-visualization/'>elbow method</a>
#  >- The shape of the matrix after TruncatedSVD will be 2000\*n, i.e. each row represents a vector form of the corresponding word. <br>
#  >- Vectorize the essay text and project titles using these word vectors. (while vectorizing, do ignore all the words which are not in top 2k words)
# - <font color='red'>step 4</font> Concatenate these truncatedSVD matrix, with the matrix with features
# <ul>
#     <li><strong>school_state</strong> : categorical data</li>
#     <li><strong>clean_categories</strong> : categorical data</li>
#     <li><strong>clean_subcategories</strong> : categorical data</li>
#     <li><strong>project_grade_category</strong> :categorical data</li>
#     <li><strong>teacher_prefix</strong> : categorical data</li>
#     <li><strong>quantity</strong> : numerical data</li>
#     <li><strong>teacher_number_of_previously_posted_projects</strong> : numerical data</li>
#     <li><strong>price</strong> : numerical data</li>
#     <li><strong>sentiment score's of each of the essay</strong> : numerical data</li>
#     <li><strong>number of words in the title</strong> : numerical data</li>
#     <li><strong>number of words in the combine essays</strong> : numerical data</li>
#     <li><strong>word vectors calculated in</strong> <font color='red'>step 3</font> : numerical data</li>
# </ul>
# - <font color='red'>step 5</font>: Apply GBDT on matrix that was formed in <font color='red'>step 4</font> of this assignment, <font color='blue'><strong>DO REFER THIS BLOG: <a href='https://www.kdnuggets.com/2017/03/simple-xgboost-tutorial-iris-dataset.html'>XGBOOST DMATRIX<strong></a></font>
# <li><font color='red'>step 6</font>:Hyper parameter tuning (Consider any two hyper parameters)<ul><li>Find the best hyper parameter which will give the maximum <a href='https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/receiver-operating-characteristic-curve-roc-curve-and-auc-1/'>AUC</a> value</li>
#     <li>Find the best hyper paramter using k-fold cross validation or simple cross validation data</li>
#     <li>Use gridsearch cv or randomsearch cv or you can also write your own for loops to do this task of hyperparameter tuning</li> 
#         </ul>
#     </li>
# 
# 

# ## 

# import sys
# import math
#  
# import numpy as np
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import roc_auc_score
# 
# # you might need to install this one
# import xgboost as xgb
# 
# class XGBoostClassifier():
#     def __init__(self, num_boost_round=10, **params):
#         self.clf = None
#         self.num_boost_round = num_boost_round
#         self.params = params
#         self.params.update({'objective': 'multi:softprob'})
#  
#     def fit(self, X, y, num_boost_round=None):
#         num_boost_round = num_boost_round or self.num_boost_round
#         self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
#         dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
#         self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round, verbose_eval=1)
#  
#     def predict(self, X):
#         num2label = {i: label for label, i in self.label2num.items()}
#         Y = self.predict_proba(X)
#         y = np.argmax(Y, axis=1)
#         return np.array([num2label[i] for i in y])
#  
#     def predict_proba(self, X):
#         dtest = xgb.DMatrix(X)
#         return self.clf.predict(dtest)
#  
#     def score(self, X, y):
#         Y = self.predict_proba(X)[:,1]
#         return roc_auc_score(y, Y)
#  
#     def get_params(self, deep=True):
#         return self.params
#  
#     def set_params(self, **params):
#         if 'num_boost_round' in params:
#             self.num_boost_round = params.pop('num_boost_round')
#         if 'objective' in params:
#             del params['objective']
#         self.params.update(params)
#         return self
#     
# 
# clf = XGBoostClassifier(eval_metric = 'auc', num_class = 2, nthread = 4,)
# ###################################################################
# #               Change from here                                  #
# ###################################################################
# parameters = {
#     'num_boost_round': [100, 250, 500],
#     'eta': [0.05, 0.1, 0.3],
#     'max_depth': [6, 9, 12],
#     'subsample': [0.9, 1.0],
#     'colsample_bytree': [0.9, 1.0],
# }
# 
# clf = GridSearchCV(clf, parameters)
# X = np.array([[1,2], [3,4], [2,1], [4,3], [1,0], [4,5]])
# Y = np.array([0, 1, 0, 1, 0, 1])
# clf.fit(X, Y)
# 
# # print(clf.grid_scores_)
# best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
# print('score:', score)
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))

# <h1>2. TruncatedSVD </h1>

# <h2>2.1 Selecting top 2000 words from `essay` and `project_title`</h2>

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


# #### 1.5.2.2 TFIDF vectorizer

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# text_tfidf = vectorizer.fit_transform(preprocessed_essays)
# print("Shape of matrix after one hot encodig ",text_tfidf.shape)

# In[ ]:


get_ipython().run_cell_magic('time', '', '# TFIDF for  title_and_essays\n\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nvectorizer = TfidfVectorizer(min_df=10)\nvectorizer.fit(X_train[\'title_and_essay\'].values)\n\nX_train_essays_tfidf = vectorizer.transform(X_train[\'title_and_essay\'].values)\nX_test_essays_tfidf = vectorizer.transform(X_test[\'title_and_essay\'].values)\n\n\nprint("Shape of matrix after one hot encodig ",X_train_essays_tfidf.shape)\nprint("Shape of matrix after one hot encodig ",X_test_essays_tfidf.shape)\n\nfeature_names_essays_tfidf = vectorizer.get_feature_names()\n#feature_names_essays_tfidf')


# In[ ]:


df = pd.DataFrame(list(zip( feature_names_essays_tfidf, vectorizer.idf_)), 
               columns =['feature_names_essays_tfidf', 'idf_scores']) 

df_2k = df.sort_values("idf_scores", ascending = True)


# In[ ]:


df_2k = df_2k.iloc[5000:7000]


# In[ ]:


top_2k_words = list(df_2k["feature_names_essays_tfidf"])


# <h2>2.2 Computing Co-occurance matrix</h2>

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# go through documentations and blogs before you start coding 
# first figure out what to do, and then think about how to do.
# reading and understanding error messages will be very much helpfull in debugging your code
# make sure you featurize train and test data separatly

# when you plot any graph make sure you use i
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label


# In[ ]:


len(X_train["title_and_essay"][0])


# In[ ]:


from nltk.tokenize import word_tokenize
from tqdm import tqdm

#top_2k_words
list_for_storing_everything = []
list_of_sentances = list(X_train["title_and_essay"])


for sentance in tqdm(list_of_sentances):
    list_of_words = word_tokenize(sentance)
    string = ""
    for word in list_of_words:
        if word in top_2k_words:
            string += " " +word
            #print(list_of_words,string)
    list_for_storing_everything.append(string)
    
print(list_for_storing_everything)


# In[ ]:


len(list_for_storing_everything[1])


# In[ ]:


list_for_storing_everything[1]


# In[ ]:


len(set(list_for_storing_everything))


# In[ ]:


# https://datascience.stackexchange.com/questions/40038/how-to-implement-word-to-word-co-occurence-matrix-in-python
get_ipython().run_line_magic('time', '')

from nltk.tokenize import word_tokenize
from itertools import combinations
from collections import Counter

sentences = list_for_storing_everything
vocab = set(word_tokenize(' '.join(sentences)))
print('Vocabulary:\n',vocab,'\n')
token_sent_list = [word_tokenize(sen) for sen in sentences]
print('Each sentence in token form:\n',token_sent_list,'\n')

co_occ = {ii:Counter({jj:0 for jj in vocab if jj!=ii}) for ii in vocab}
k=5

for sen in tqdm(token_sent_list):
    for ii in range(len(sen)):
        if ii < k:
            c = Counter(sen[0:ii+k+1])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c
        elif ii > len(sen)-(k+1):
            c = Counter(sen[ii-k::])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c
        else:
            c = Counter(sen[ii-k:ii+k+1])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c

# Having final matrix in dict form lets you convert it to different python data structures
co_occ = {ii:dict(co_occ[ii]) for ii in vocab}
display(co_occ)


# In[ ]:


co_occ


# In[ ]:


dm = pd.DataFrame(co_occ).fillna(0)


# In[ ]:


dm.as_matrix()


# In[ ]:


import scipy
sparse_matrix = scipy.sparse.csr_matrix(dm.as_matrix())


# import numpy as np
# import pandas as pd
# 
# ctxs = [
#     'krayyem like candy crush more then coffe',
#     'krayyem plays candy crush all days',
#     'krayyem do not invite his friends to play candy crush',
#     'krayyem is smart',
# ]
# 
# l_unique = list(set((' '.join(ctxs)).split(' ')))
# mat = np.zeros((len(l_unique), len(l_unique)))
# 
# nei = []
# nei_size = 3
# 
# for ctx in ctxs:
#     words = ctx.split(' ')
# 
#     for i, _ in enumerate(words):
#         nei.append(words[i])
# 
#         if len(nei) > (nei_size * 2) + 1:
#             nei.pop(0)
# 
#         pos = int(len(nei) / 2)
#         for j, _ in enumerate(nei):
#            if nei[j]  in l_unique and words[i] in l_unique:
#               mat[l_unique.index(nei[j]), l_unique.index(words[i])] += 1
# 
# mat = pd.DataFrame(mat)
# mat.index = l_unique
# mat.columns = l_unique
# display(mat)

# <h2>2.3 Applying TruncatedSVD and Calculating Vectors for `essay` and `project_title`</h2>

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


# In[ ]:


from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

n_components_range = [x for x in range(0,2000,10)]
list_for_explained_variance = []

for n in tqdm(n_components_range):
    svd = TruncatedSVD(n_components=n, n_iter=3)
    svd.fit(sparse_matrix)
    list_for_explained_variance.append(svd.explained_variance_ratio_.sum())  


# In[ ]:


# graph for elbow plot 

import matplotlib.pyplot as plt

plt.plot(n_components_range , list_for_explained_variance , label='variance')

#plt.legend()
plt.xlabel("n_components_range")
plt.ylabel("list_for_explained_variance")
plt.title(" n_components_range Vs list_for_explained_variance ")
plt.grid()
plt.show()


# ### graph interpretation:
# - from the graph 300 can be interpreted as a good inflection point
# - hence the word vectors will be vectors of length 300 

# In[ ]:


svd = TruncatedSVD(n_components=300, n_iter=3)
svd.fit(sparse_matrix)


# In[ ]:


X = svd.transform(sparse_matrix)
len(X[0][:])


# In[ ]:


words_2000 = list(dm.index.values)


# In[ ]:


dictonary_for_vectorization = {}
for i in range(2000):
    dictonary_for_vectorization.__setitem__(words_2000[i], X[0][:])


# In[ ]:


len(dictonary_for_vectorization["101"])


# <h2>2.4 Merge the features from <font color='red'>step 3</font> and <font color='red'>step 4</font></h2>

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


glove_words = set(dictonary_for_vectorization.keys())


# In[ ]:


# average Word2Vec preprocessed_essays
# compute average word2vec for each review.
X_train_title_essays_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_train["title_and_essay"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += dictonary_for_vectorization[word]
    X_train_title_essays_vectors.append(vector)

print(len(X_train_title_essays_vectors))
print(len(X_train_title_essays_vectors[0]))


# In[ ]:


len(X_train_title_essays_vectors[0])


# In[ ]:


# average Word2Vec preprocessed_essays
# compute average word2vec for each review.
X_test_title_essays_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_test["title_and_essay"]): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += dictonary_for_vectorization[word]
    X_test_title_essays_vectors.append(vector)

print(len(X_test_title_essays_vectors))
print(len(X_test_title_essays_vectors[0]))


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
            X_train_title_essays_vectors,
            X_train_price_norm,
            X_train_teacher_number_of_previously_posted_projects_norm,
            X_train_sentiment_score_of_essays_neg_norm,
            X_train_sentiment_score_of_essays_neu_norm,
            X_train_sentiment_score_of_essays_pos_norm,
            X_train_sentiment_score_of_essays_compound_norm,
            X_train_number_of_words_in_essays_norm,
            X_train_number_of_words_in_title_norm))

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
            X_test_title_essays_vectors,
            X_test_price_norm,
            X_test_teacher_number_of_previously_posted_projects_norm,
            X_test_sentiment_score_of_essays_neg_norm,
            X_test_sentiment_score_of_essays_neu_norm,
            X_test_sentiment_score_of_essays_pos_norm,
            X_test_sentiment_score_of_essays_compound_norm,
            X_test_number_of_words_in_essays_norm,
            X_test_number_of_words_in_title_norm))

X_test_SET_2.shape


# In[ ]:


y_test_SET_2 = y_test
y_test_SET_2.shape


# <h2>2.5 Apply XGBoost on the Final Features from the above section</h2>

# https://xgboost.readthedocs.io/en/latest/python/python_intro.html

# In[ ]:


# No need to split the data into train and test(cv)
# use the Dmatrix and apply xgboost on the whole data
# please check the Quora case study notebook as reference

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


# this is for plot 2 
# best k from the above plot is K = 291
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
from xgboost.sklearn import XGBClassifier

params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': 1,
    'n_estimators': 5
}


RF_SET_2_best = XGBClassifier(**params)
RF_SET_2_best.fit(X_train_SET_2, y_train_SET_2)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:


# predicted probablities 
y_train_pred_prob = RF_SET_2_best.predict_proba( X_train_SET_2)    
y_test_pred_prob = RF_SET_2_best.predict_proba( X_test_SET_2)
y_train_pred_prob


# In[ ]:


# predicted values 
y_train_pred = RF_SET_2_best.predict( X_train_SET_2)    
y_test_pred = RF_SET_2_best.predict( X_test_SET_2)


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


# <h1>3. Conclusion</h1>

# In[ ]:


# Please write down few lines about what you observed from this assignment. 


# - Using Co-occurance matrix the relationships between the words present inside the corpus can be expressed.
# - Co-occurance matrix is an NxN matrix showing the relationship between the words.
# - This NxN matrix can be reduced to NxR Using singular value decomposition(SVD), where R is the number of reduced columns.
# - The NxR matrix can now be used as a word vector representation, having N words and every word corresponds to a vector of length R.
# - Using this word vector representation, text data can be vectorized.
