#!/usr/bin/env python
# coding: utf-8

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

# # DonorsChoose

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


# In[ ]:


get_ipython().system('pip install WordCloud')


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

# <h1>2. Decision Tree </h1>

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
# from sklearn.preprocessing import StandardScaler
# price_scalar = StandardScaler()
# price_scalar.fit(X_train['price'].values.reshape(-1,1))# finding the mean and standard deviation of this data
# print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")
# 
# X_train_price_standardized = price_scalar.transform(X_train['price'].values.reshape(-1, 1))
# #X_train_cv_price_standardized = price_scalar.transform(X_train_cv['price'].values.reshape(-1, 1))
# X_test_price_standardized = price_scalar.transform(X_test['price'].values.reshape(-1, 1))
# 
# print("After vectorizations")
# print(X_train_price_standardized.shape, y_train.shape)
# #print(X_train_cv_price_standardized.shape, y_train_cv.shape)
# print(X_test_price_standardized.shape, y_test.shape)
# print("="*100)
# 
len(price_standardized)
# from sklearn.preprocessing import StandardScaler
# price_scalar = StandardScaler()
# price_scalar.fit(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))# finding the mean and standard deviation of this data
# print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")
# 
# X_train_previous_projects_standardized = price_scalar.transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))
# #X_train_cv_previous_projects_standardized = price_scalar.transform(X_train_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))
# X_test_previous_projects_standardized = price_scalar.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))
# 
# print("After vectorizations")
# print(X_train_previous_projects_standardized.shape, y_train.shape)
# #print(X_train_cv_previous_projects_standardized.shape, y_train_cv.shape)
# print(X_test_previous_projects_standardized.shape, y_test.shape)
# print("="*100)
# 

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

# #### 1.5.2.1 Bag of words
# We are considering only the words which appeared in at least 10 documents(rows or projects).
vectorizer = CountVectorizer(min_df=10)
text_bow = vectorizer.fit_transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_bow.shape)
# In[ ]:


#vectorizing text data 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=10, max_features=5000)
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

feature_names_essay_bow = vectorizer.get_feature_names()
feature_names_essay_bow


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

feature_names_title_bow = vectorizer.get_feature_names()
feature_names_title_bow


# #### 1.5.2.2 TFIDF vectorizer

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# text_tfidf = vectorizer.fit_transform(preprocessed_essays)
# print("Shape of matrix after one hot encodig ",text_tfidf.shape)

# In[ ]:


# TFIDF for  preprocessed_essays

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=10, max_features=5000)
vectorizer.fit(X_train['preprocessed_essays'].values)

X_train_essays_tfidf = vectorizer.transform(X_train['preprocessed_essays'].values)
X_test_essays_tfidf = vectorizer.transform(X_test['preprocessed_essays'].values)


print("Shape of matrix after one hot encodig ",X_train_essays_tfidf.shape)
print("Shape of matrix after one hot encodig ",X_test_essays_tfidf.shape)

feature_names_essays_tfidf = vectorizer.get_feature_names()
feature_names_essays_tfidf


# In[ ]:


# TFIDF for  preprocessed_titles

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
vectorizer.fit(X_train['preprocessed_titles'].values)

X_train_title_tfidf = vectorizer.transform(X_train['preprocessed_titles'].values)
X_test_title_tfidf = vectorizer.transform(X_test['preprocessed_titles'].values)


print("Shape of matrix after one hot encodig ",X_train_title_tfidf.shape)
print("Shape of matrix after one hot encodig ",X_test_title_tfidf.shape)

feature_names_title_tfidf = vectorizer.get_feature_names()
feature_names_title_tfidf


# In[ ]:





# #### 1.5.2.3 Using Pretrained Models: Avg W2V

# In[ ]:


# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
# make sure you have the glove_vectors file
with open('../input/donors-choose-elysian/glove_vectors', 'rb') as f:
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

# In[ ]:


# making a list of all the vectorized features for SET_1 
list_of_vectorized_features_SET_1 =  (feature_names_clean_categories +
                                feature_names_clean_subcategories +
                                feature_names_school_state + 
                                feature_names_preprocessed_teacher_prefix +
                                feature_names_project_grade_category +
                                feature_names_essay_bow +
                                feature_names_title_bow +
                                feature_names_price+
                                feature_names_teacher_number_of_previously_posted_projects)
                                
len(list_of_vectorized_features_SET_1)                                


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
            X_train_price_norm,
            X_train_teacher_number_of_previously_posted_projects_norm))
X_train_SET_1.shape


# In[ ]:


y_train_SET_1 = y_train
y_train_SET_1.shape


# In[ ]:


# SET_1 
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_test_SET_1 = hstack((X_test_clean_categories_OHE,
            X_test_clean_subcategories_OHE,
            X_test_school_state_OHE,
            X_test_preprocessed_teacher_prefix_OHE,
            X_test_project_grade_category_OHE,
            X_test_essay_bow,
            X_test_title_bow,
            X_test_price_norm,
            X_test_teacher_number_of_previously_posted_projects_norm))
X_test_SET_1.shape


# In[ ]:


y_test_SET_1 = y_test
y_test_SET_1.shape


# In[ ]:





# In[ ]:


# making a list of all the vectorized features for SET_2
list_of_vectorized_features_SET_2 =  (feature_names_clean_categories +
                                    feature_names_clean_subcategories +
                                    feature_names_school_state + 
                                    feature_names_preprocessed_teacher_prefix +
                                    feature_names_project_grade_category +
                                    feature_names_essays_tfidf +
                                    feature_names_title_tfidf +
                                    feature_names_price+
                                    feature_names_teacher_number_of_previously_posted_projects)
                                
len(list_of_vectorized_features_SET_2)                                


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
            X_train_price_norm,
            X_train_teacher_number_of_previously_posted_projects_norm))
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
            X_test_price_norm,
            X_test_teacher_number_of_previously_posted_projects_norm))
X_test_SET_2.shape


# In[ ]:


y_test_SET_2 = y_test
y_test_SET_2.shape


# In[ ]:





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
            X_train_price_norm,
            X_train_teacher_number_of_previously_posted_projects_norm))
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
            X_test_price_norm,
            X_test_teacher_number_of_previously_posted_projects_norm))
X_test_SET_3.shape


# In[ ]:


y_test_SET_3 = y_test
y_test_SET_3.shape


# In[ ]:





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
            X_train_price_norm,
            X_train_teacher_number_of_previously_posted_projects_norm))
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
            X_test_price_norm,
            X_test_teacher_number_of_previously_posted_projects_norm))
X_test_SET_4.shape


# In[ ]:


y_test_SET_4 = y_test
y_test_SET_4.shape


# In[ ]:





# In[ ]:





# # SET_5
# # merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
# from scipy.sparse import hstack
# # with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
# X_train_SET_5 = hstack((X_train_clean_categories_OHE,
#             X_train_clean_subcategories_OHE,
#             X_train_school_state_OHE,
#             X_train_preprocessed_teacher_prefix_OHE,
#             X_train_project_grade_category_OHE,
#             X_train_sentiment_score_of_essays_neg_norm,
#             X_train_sentiment_score_of_essays_neu_norm,
#             X_train_sentiment_score_of_essays_pos_norm,
#             X_train_sentiment_score_of_essays_compound_norm,
#             X_train_price_norm,
#             X_train_teacher_number_of_previously_posted_projects_norm,
#             X_train_number_of_words_in_essays_norm,
#             X_train_number_of_words_in_title_norm))
# X_train_SET_5.shape
# 

# In[ ]:



y_train_SET_5 = y_train
y_train_SET_5.shape


# # SET_5
# # merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
# from scipy.sparse import hstack
# # with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
# X_test_SET_5 = hstack((X_test_clean_categories_OHE,
#             X_test_clean_subcategories_OHE,
#             X_test_school_state_OHE,
#             X_test_preprocessed_teacher_prefix_OHE,
#             X_test_project_grade_category_OHE,
#             X_test_sentiment_score_of_essays_neg_norm,
#             X_test_sentiment_score_of_essays_neu_norm,
#             X_test_sentiment_score_of_essays_pos_norm,
#             X_test_sentiment_score_of_essays_compound_norm,
#             X_test_price_norm,
#             X_test_teacher_number_of_previously_posted_projects_norm,
#             X_test_number_of_words_in_essays_norm,
#             X_test_number_of_words_in_title_norm))
# 
# X_test_SET_5.shape
# 

# In[ ]:


y_test_SET_5 = y_test
y_test_SET_5.shape


# In[ ]:


# making a list of all the vectorized features for SET_1 
list_of_vectorized_features_SET_1 =  (feature_names_clean_categories +
                                feature_names_clean_subcategories +
                                feature_names_school_state + 
                                feature_names_preprocessed_teacher_prefix +
                                feature_names_project_grade_category +
                                feature_names_essay_bow +
                                feature_names_title_bow +
                                feature_names_price+
                                feature_names_teacher_number_of_previously_posted_projects)
                                
len(list_of_vectorized_features_SET_1)                                


# # __ Computing Sentiment Scores__

# In[ ]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

# import nltk

sid = SentimentIntensityAnalyzer()

for_sentiment = 'a person is a person no matter how small dr seuss i teach the smallest students with the biggest enthusiasm for learning my students learn in many different ways using all of our senses and multiple intelligences i use a wide rangeof techniques to help all my students succeed students in my class come from a variety of different backgrounds which makesfor wonderful sharing of experiences and cultures including native americans our school is a caring community of successful learners which can be seen through collaborative student project based learning in and out of the classroom kindergarteners in my class love to work with hands on materials and have many different opportunities to practice a skill before it ismastered having the social skills to work cooperatively with friends is a crucial aspect of the kindergarten curriculummontana is the perfect place to learn about agriculture and nutrition my students love to role play in our pretend kitchenin the early childhood classroom i have had several kids ask me can we try cooking with real food i will take their idea and create common core cooking lessons where we learn important math and writing concepts while cooking delicious healthy food for snack time my students will have a grounded appreciation for the work that went into making the food and knowledge of where the ingredients came from as well as how it is healthy for their bodies this project would expand our learning of nutrition and agricultural cooking recipes by having us peee will all our own apples to make homemade applesauce make our own bread and mix up healthy plants from our classroom garden in the spring wso create our own cookbooks to be printed and shared with families students will gain math and literature skills as well as a life long enjoyment for healthy cooking nannan'
ss = sid.polarity_scores(for_sentiment)
ss
#for k in ss:
#    print('{0}: {1}, '.format(k, ss[k]), end='')

# we can use these 4 things as features/attributes (neg, neu, pos, compound)
# neg: 0.0, neu: 0.753, pos: 0.247, compound: 0.93


# # Assignment 8: DT

# <ol>
#     <li><strong>Apply Decision Tree Classifier(DecisionTreeClassifier) on these feature sets</strong>
#         <ul>
#             <li><font color='red'>Set 1</font>: categorical, numerical features + project_title(BOW) + preprocessed_eassay (BOW)</li>
#             <li><font color='red'>Set 2</font>: categorical, numerical features + project_title(TFIDF)+  preprocessed_eassay (TFIDF)</li>
#             <li><font color='red'>Set 3</font>: categorical, numerical features + project_title(AVG W2V)+  preprocessed_eassay (AVG W2V)</li>
#             <li><font color='red'>Set 4</font>: categorical, numerical features + project_title(TFIDF W2V)+  preprocessed_eassay (TFIDF W2V)</li>        </ul>
#     </li>
#     <br>
#     <li><strong>Hyper paramter tuning (best `depth` in range [4,6, 8, 9,10,12,14,17] , and the best `min_samples_split` in range [2,10,20,30,40,50])</strong>
#         <ul>
#     <li>Find the best hyper parameter which will give the maximum <a href='https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/receiver-operating-characteristic-curve-roc-curve-and-auc-1/'>AUC</a> value</li>
#     <li>Find the best hyper paramter using k-fold cross validation or simple cross validation data</li>
#     <li>Use gridsearch cv or randomsearch cv or you can also write your own for loops to do this task of hyperparameter tuning</li> 
#         </ul>
#     </li>
#     <br>
#     <li><strong>Graphviz</strong>
#         <ul>
#     <li>Visualize your decision tree with Graphviz. It helps you to understand how a decision is being made, given a new vector.</li>
#     <li>Since feature names are not obtained from word2vec related models, visualize only BOW & TFIDF decision trees using Graphviz</li>
#     <li>Make sure to print the words in each node of the decision tree instead of printing its index.</li>
#     <li>Just for visualization purpose, limit max_depth to 2 or 3 and either embed the generated images of graphviz in your notebook, or directly upload them as .png files.</li>                
#         </ul>
#     </li>
#     <br>
#     <li>
#     <strong>Representation of results</strong>
#         <ul>
#     <li>You need to plot the performance of model both on train data and cross validation data for each hyper parameter, like shown in the figure
#     <img src='https://i.imgur.com/Gp2DQmh.jpg' width=500px> with X-axis as <strong>min_sample_split</strong>, Y-axis as <strong>max_depth</strong>, and Z-axis as <strong>AUC Score</strong> , we have given the notebook which explains how to plot this 3d plot, you can find it in the same drive <i>3d_scatter_plot.ipynb</i></li>
#             <p style="text-align:center;font-size:30px;color:red;"><strong>or</strong></p> <br>
#     <li>You need to plot the performance of model both on train data and cross validation data for each hyper parameter, like shown in the figure
#     <img src='https://i.imgur.com/fgN9aUP.jpg' width=300px> <a href='https://seaborn.pydata.org/generated/seaborn.heatmap.html'>seaborn heat maps</a> with rows as <strong>min_sample_split</strong>, columns as <strong>max_depth</strong>, and values inside the cell representing <strong>AUC Score</strong> </li>
#     <li>You choose either of the plotting techniques out of 3d plot or heat map</li>
#     <li>Once after you found the best hyper parameter, you need to train your model with it, and find the AUC on test data and plot the ROC curve on both train and test.
#     <img src='train_test_auc.JPG' width=300px></li>
#     <li>Along with plotting ROC curve, you need to print the <a href='https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/confusion-matrix-tpr-fpr-fnr-tnr-1/'>confusion matrix</a> with predicted and original labels of test data points
#     <img src='confusion_matrix.png' width=300px></li>
#     <li>Once after you plot the confusion matrix with the test data, get all the `false positive data points`
#         <ul>
#             <li> Plot the WordCloud <a href='https://www.geeksforgeeks.org/generating-word-cloud-python/' with the words of eassy text of these `false positive data points`>WordCloud</a></li>
#             <li> Plot the box plot with the `price` of these `false positive data points`</li>
#             <li> Plot the pdf with the `teacher_number_of_previously_posted_projects` of these `false positive data points`</li>
#         </ul>
#         </ul>
#     </li>
#     <br>
#     <li><strong>[Task-2]</strong>
#         <ul>
#     <li> Select 5k best features from features of <font color='red'>Set 2</font> using<a href='https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html'>`feature_importances_`</a>, discard all the other remaining features and then apply any of the model of you choice i.e. (Dession tree, Logistic Regression, Linear SVM), you need to do hyperparameter tuning corresponding to the model you selected and procedure in step 2 and step 3</li>
#         </ul>
#     <br>
#     <li><strong>Conclusion</strong>
#         <ul>
#     <li>You need to summarize the results at the end of the notebook, summarize it in the table format. To print out a table please refer to this prettytable library<a href='http://zetcode.com/python/prettytable/'>  link</a> 
#         <img src='summary.JPG' width=400px>
#     </li>
#         </ul>
# </ol>

# <h1>2. Decision Tree </h1>

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


# <h2>2.2 Make Data Model Ready: encoding numerical, categorical features</h2>

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


# <h2>2.4 Appling  Decision Tree on different kind of featurization as mentioned in the instructions</h2>
# 
# <br>Apply  Decision Tree on different kind of featurization as mentioned in the instructions
# <br> For Every model that you work on make sure you do the step 2 and step 3 of instrucations

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


# ### 2.4.1 Applying DecisionTreeClassifier on BOW,<font color='red'> SET 1</font>

# In[ ]:


# c = 1/lambda 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

DT_SET_1 = DecisionTreeClassifier(class_weight = 'balanced')
max_depth_range = [4,6,8,9,10,12,14,17]
min_samples_split_range = [2,10,20,30,40,50]

#alpha_range = [10**x for x in range(-4,4) ]
#alpha_range = [ x for x in np.linspace(0.001,0.1,10)]
#print(alpha_range)
param_grid = dict(max_depth = max_depth_range, 
                  min_samples_split = min_samples_split_range)
print(param_grid) 


# In[ ]:



grid = GridSearchCV(DT_SET_1, param_grid, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
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
#best_alpha = grid.best_params_.get("alpha")
print(grid.best_params_)
best_max_depth = grid.best_params_.get('max_depth')
best_min_samples_split =  grid.best_params_.get('min_samples_split')
print(best_max_depth)
print(best_min_samples_split)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
import numpy as np


# In[ ]:


def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  init_notebook_mode(connected=False)


# In[ ]:


max_depth_range = [4,6,8,9,10,12,14,17]
min_samples_split_range = [2,10,20,30,40,50]


# In[ ]:


x1 = [4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,12,12,12,12,12,12,14,14,14,14,14,14,17,17,17,17,17,17]
y1 = [2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50]
z1 = train_auc

x2 = [4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,12,12,12,12,12,12,14,14,14,14,14,14,17,17,17,17,17,17]
y2 = [2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50]
z2 = cv_auc


# In[ ]:


# https://plot.ly/python/3d-axes/
trace1 = go.Scatter3d(x=x1,y=y1,z=z1, name = 'train')
trace2 = go.Scatter3d(x=x2,y=y2,z=z2, name = 'Cross validation')
data = [trace1, trace2]
enable_plotly_in_cell()

layout = go.Layout(scene = dict(
        xaxis = dict(title='max_depth'),
        yaxis = dict(title='min_sample_split'),
        zaxis = dict(title='AUC'),))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename='3d-scatter-colorscale')


# In[ ]:





plt.plot(np.log(param_grid['alpha']), train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.log(param_grid['alpha']),train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(np.log(param_grid['alpha']), cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.log(param_grid['alpha']),cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(np.log(param_grid['alpha']), train_auc, label='Train AUC points')
plt.scatter(np.log(param_grid['alpha']), cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs alpha ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()
# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 291
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

DT_SET_1_best = DecisionTreeClassifier(max_depth = best_max_depth,min_samples_split = best_min_samples_split,class_weight= 'balanced')
DT_SET_1_best.fit(X_train_SET_1, y_train_SET_1)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:





# In[ ]:


# predicted probablities 
y_train_pred_prob = DT_SET_1_best.predict_proba( X_train_SET_1)    
y_test_pred_prob = DT_SET_1_best.predict_proba( X_test_SET_1)


# In[ ]:


y_train_pred_prob


# In[ ]:


# predicted values 
y_train_pred = DT_SET_1_best.predict( X_train_SET_1)    
y_test_pred = DT_SET_1_best.predict( X_test_SET_1)


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
sn.heatmap(df_cm, annot=True, fmt ="d")


# In[ ]:


# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
results_test = confusion_matrix(y_test, y_test_pred) 
print ('Confusion Matrix test :')
print(results_test)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
print("TRUE NEGATIVES = ", tn)
print("FALSE POSITIVES = ", fp)
print("FALSE NEGATIVES = ", fn)
print("TRUE POSITIVES = ", tp)


# In[ ]:


# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = results_test

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt ="d")


# ## Wordcloud

# In[ ]:


# making a dataframe of y_test & y_test_pred

df_y_test_y_pred = df = pd.DataFrame(list(zip(y_test, y_test_pred )), columns =['y_test', 'y_pred'])
df_y_test_y_pred.shape


# In[ ]:


df_false_positives =  df_y_test_y_pred[(df_y_test_y_pred['y_test'] == 0) & (df_y_test_y_pred['y_pred'] ==1)]
df_false_positives.shape


# In[ ]:


data_false_positives = X_test
data_false_positives.reset_index(inplace=True)
data_false_positives.shape


# In[ ]:


#data_false_positives = data_false_positives.loc[data_false_positives.index & df_y_test_y_pred.index]
data_false_positives = data_false_positives[data_false_positives.index.isin(df_false_positives.index)]
#data_false_positives = data_false_positives.loc[data_false_positives.index.intersection(df_false_positives.index)]
data_false_positives.shape


# In[ ]:


l1 = list(data_false_positives["preprocessed_essays"])
len(l1)


# In[ ]:


giant_list = [] # for storing every word in all of the sentances
for sentance in l1:
    temp_list = sentance.split()
    giant_list = giant_list + temp_list     


# In[ ]:


from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(str(giant_list)) 


# In[ ]:


# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:


box_plot_price = data_false_positives.boxplot(column = ["price"])


# In[ ]:


pdf_previous_submisssions = data_false_positives["teacher_number_of_previously_posted_projects"].plot.kde()


# #### 2.4.1.1 Graphviz visualization of Decision Tree on BOW,<font color='red'> SET 1</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(list_of_vectorized_features_SET_1)
features


# In[ ]:


dot_data = StringIO()  
export_graphviz(DT_SET_1_best, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


# In[ ]:


len(DT_SET_1_best.feature_importances_)


# ### 2.4.2 Applying Decision Trees on TFIDF,<font color='red'> SET 2</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:


# c = 1/lambda 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

DT_SET_2 = DecisionTreeClassifier(class_weight = 'balanced')
max_depth_range = [4,6,8,9,10,12,14,17]
min_samples_split_range = [2,10,20,30,40,50]

#alpha_range = [10**x for x in range(-4,4) ]
#alpha_range = [ x for x in np.linspace(0.001,0.1,10)]
#print(alpha_range)
param_grid = dict(max_depth = max_depth_range, 
                  min_samples_split = min_samples_split_range)
print(param_grid) 


# In[ ]:



grid = GridSearchCV(DT_SET_2, param_grid, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
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
#best_alpha = grid.best_params_.get("alpha")
print(grid.best_params_)
best_max_depth = grid.best_params_.get('max_depth')
best_min_samples_split =  grid.best_params_.get('min_samples_split')
print(best_max_depth)
print(best_min_samples_split)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
import numpy as np


# In[ ]:


def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  init_notebook_mode(connected=False)


# In[ ]:


max_depth_range = [4,6,8,9,10,12,14,17]
min_samples_split_range = [2,10,20,30,40,50]


# In[ ]:


x1 = [4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,12,12,12,12,12,12,14,14,14,14,14,14,17,17,17,17,17,17]
y1 = [2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50]
z1 = train_auc

x2 = [4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,12,12,12,12,12,12,14,14,14,14,14,14,17,17,17,17,17,17]
y2 = [2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50]
z2 = cv_auc


# In[ ]:


# https://plot.ly/python/3d-axes/
trace1 = go.Scatter3d(x=x1,y=y1,z=z1, name = 'train')
trace2 = go.Scatter3d(x=x2,y=y2,z=z2, name = 'Cross validation')
data = [trace1, trace2]
enable_plotly_in_cell()

layout = go.Layout(scene = dict(
        xaxis = dict(title='max_depth'),
        yaxis = dict(title='min_sample_split'),
        zaxis = dict(title='AUC'),))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename='3d-scatter-colorscale')


# In[ ]:





plt.plot(np.log(param_grid['alpha']), train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.log(param_grid['alpha']),train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(np.log(param_grid['alpha']), cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.log(param_grid['alpha']),cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(np.log(param_grid['alpha']), train_auc, label='Train AUC points')
plt.scatter(np.log(param_grid['alpha']), cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs alpha ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()
# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 291
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

DT_SET_2_best = DecisionTreeClassifier(max_depth = best_max_depth ,min_samples_split = best_min_samples_split,class_weight= 'balanced')
DT_SET_2_best.fit(X_train_SET_2, y_train_SET_2)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:





# In[ ]:


# predicted probablities 
y_train_pred_prob = DT_SET_2_best.predict_proba( X_train_SET_2)    
y_test_pred_prob = DT_SET_2_best.predict_proba( X_test_SET_2)


# In[ ]:


y_train_pred_prob


# In[ ]:


# predicted values 
y_train_pred = DT_SET_2_best.predict( X_train_SET_2)    
y_test_pred = DT_SET_2_best.predict( X_test_SET_2)


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
sn.heatmap(df_cm, annot=True, fmt ="d")


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
sn.heatmap(df_cm, annot=True, fmt ="d")


# In[ ]:





# ## Wordcloud

# In[ ]:


# making a dataframe of y_test & y_test_pred

df_y_test_y_pred = df = pd.DataFrame(list(zip(y_test, y_test_pred )), columns =['y_test', 'y_pred'])
df_y_test_y_pred.shape


# In[ ]:


df_false_positives =  df_y_test_y_pred[(df_y_test_y_pred['y_test'] == 0) & (df_y_test_y_pred['y_pred'] ==1)]
df_false_positives.shape


# In[ ]:


data_false_positives = X_test
data_false_positives.reset_index(inplace=True)
data_false_positives.shape


# In[ ]:


#data_false_positives = data_false_positives.loc[data_false_positives.index & df_y_test_y_pred.index]
data_false_positives = data_false_positives[data_false_positives.index.isin(df_false_positives.index)]
#data_false_positives = data_false_positives.loc[data_false_positives.index.intersection(df_false_positives.index)]
data_false_positives.shape


# In[ ]:


l1 = list(data_false_positives["preprocessed_essays"])
len(l1)


# In[ ]:


giant_list = [] # for storing every word in all of the sentances
for sentance in l1:
    temp_list = sentance.split()
    giant_list = giant_list + temp_list     


# In[ ]:


from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(str(giant_list)) 


# In[ ]:


# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:


box_plot_price = data_false_positives.boxplot(column = ["price"])


# In[ ]:


pdf_previous_submisssions = data_false_positives["teacher_number_of_previously_posted_projects"].plot.kde()


# #### 2.4.2.1 Graphviz visualization of Decision Tree on TFIDF,<font color='red'> SET 2</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(list_of_vectorized_features_SET_2)
features


# In[ ]:


dot_data = StringIO()  
export_graphviz(DT_SET_2_best, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


# ### 2.4.3 Applying Decision Trees on AVG W2V,<font color='red'> SET 3</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:


# c = 1/lambda 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

DT_SET_3 = DecisionTreeClassifier(class_weight = 'balanced')
max_depth_range = [4,6,8,9,10,12,14,17]
min_samples_split_range = [2,10,20,30,40,50]

#alpha_range = [10**x for x in range(-4,4) ]
#alpha_range = [ x for x in np.linspace(0.001,0.1,10)]
#print(alpha_range)
param_grid = dict(max_depth = max_depth_range, 
                  min_samples_split = min_samples_split_range)
print(param_grid) 


# In[ ]:



grid = GridSearchCV(DT_SET_3, param_grid, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
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
#best_alpha = grid.best_params_.get("alpha")
print(grid.best_params_)
best_max_depth = grid.best_params_.get('max_depth')
best_min_samples_split =  grid.best_params_.get('min_samples_split')
print(best_max_depth)
print(best_min_samples_split)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
import numpy as np


# In[ ]:


def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  init_notebook_mode(connected=False)


# In[ ]:


max_depth_range = [4,6,8,9,10,12,14,17]
min_samples_split_range = [2,10,20,30,40,50]


# In[ ]:


x1 = [4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,12,12,12,12,12,12,14,14,14,14,14,14,17,17,17,17,17,17]
y1 = [2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50]
z1 = train_auc

x2 = [4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,12,12,12,12,12,12,14,14,14,14,14,14,17,17,17,17,17,17]
y2 = [2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50]
z2 = cv_auc


# In[ ]:


# https://plot.ly/python/3d-axes/
trace1 = go.Scatter3d(x=x1,y=y1,z=z1, name = 'train')
trace2 = go.Scatter3d(x=x2,y=y2,z=z2, name = 'Cross validation')
data = [trace1, trace2]
enable_plotly_in_cell()

layout = go.Layout(scene = dict(
        xaxis = dict(title='max_depth'),
        yaxis = dict(title='min_sample_split'),
        zaxis = dict(title='AUC'),))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename='3d-scatter-colorscale')


# In[ ]:





plt.plot(np.log(param_grid['alpha']), train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.log(param_grid['alpha']),train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(np.log(param_grid['alpha']), cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.log(param_grid['alpha']),cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(np.log(param_grid['alpha']), train_auc, label='Train AUC points')
plt.scatter(np.log(param_grid['alpha']), cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs alpha ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()
# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 291
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

DT_SET_3_best = DecisionTreeClassifier(max_depth = best_max_depth,min_samples_split = best_min_samples_split,class_weight= 'balanced')
DT_SET_3_best.fit(X_train_SET_3, y_train_SET_3)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:





# In[ ]:


# predicted probablities 
y_train_pred_prob = DT_SET_3_best.predict_proba( X_train_SET_3)    
y_test_pred_prob = DT_SET_3_best.predict_proba( X_test_SET_3)


# In[ ]:


y_train_pred_prob


# In[ ]:


# predicted values 
y_train_pred = DT_SET_3_best.predict( X_train_SET_3)    
y_test_pred = DT_SET_3_best.predict( X_test_SET_3)


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
sn.heatmap(df_cm, annot=True, fmt ="d")


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
sn.heatmap(df_cm, annot=True, fmt ="d")


# In[ ]:





# ## Wordcloud

# In[ ]:


# making a dataframe of y_test & y_test_pred

df_y_test_y_pred = df = pd.DataFrame(list(zip(y_test, y_test_pred )), columns =['y_test', 'y_pred'])
df_y_test_y_pred.shape


# In[ ]:


df_false_positives =  df_y_test_y_pred[(df_y_test_y_pred['y_test'] == 0) & (df_y_test_y_pred['y_pred'] ==1)]
df_false_positives.shape


# In[ ]:


data_false_positives = X_test
#data_false_positives.reset_index(inplace=True)
data_false_positives.shape


# In[ ]:


#data_false_positives = data_false_positives.loc[data_false_positives.index & df_y_test_y_pred.index]
data_false_positives = data_false_positives[data_false_positives.index.isin(df_false_positives.index)]
#data_false_positives = data_false_positives.loc[data_false_positives.index.intersection(df_false_positives.index)]
data_false_positives.shape


# In[ ]:


l1 = list(data_false_positives["preprocessed_essays"])
len(l1)


# In[ ]:


giant_list = [] # for storing every word in all of the sentances
for sentance in l1:
    temp_list = sentance.split()
    giant_list = giant_list + temp_list     


# In[ ]:


from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(str(giant_list)) 


# In[ ]:


# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:


box_plot_price = data_false_positives.boxplot(column = ["price"])


# In[ ]:


pdf_previous_submisssions = data_false_positives["teacher_number_of_previously_posted_projects"].plot.kde()


# ### 2.4.4 Applying Decision Trees on TFIDF W2V,<font color='red'> SET 4</font>

# In[ ]:


# Please write all the code with proper documentation


# In[ ]:


# c = 1/lambda 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

DT_SET_4 = DecisionTreeClassifier(class_weight = 'balanced')
max_depth_range = [4,6,8,9,10,12,14,17]
min_samples_split_range = [2,10,20,30,40,50]

#alpha_range = [10**x for x in range(-4,4) ]
#alpha_range = [ x for x in np.linspace(0.001,0.1,10)]
#print(alpha_range)
param_grid = dict(max_depth = max_depth_range, 
                  min_samples_split = min_samples_split_range)
print(param_grid) 


# In[ ]:



grid = GridSearchCV(DT_SET_4, param_grid, cv=3, scoring='roc_auc', return_train_score=True, n_jobs= -1)
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
#best_alpha = grid.best_params_.get("alpha")
print(grid.best_params_)
best_max_depth = grid.best_params_.get('max_depth')
best_min_samples_split =  grid.best_params_.get('min_samples_split')
print(best_max_depth)
print(best_min_samples_split)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
import numpy as np


# In[ ]:


def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  init_notebook_mode(connected=False)


# In[ ]:


max_depth_range = [4,6,8,9,10,12,14,17]
min_samples_split_range = [2,10,20,30,40,50]


# In[ ]:


x1 = [4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,12,12,12,12,12,12,14,14,14,14,14,14,17,17,17,17,17,17]
y1 = [2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50]
z1 = train_auc

x2 = [4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,12,12,12,12,12,12,14,14,14,14,14,14,17,17,17,17,17,17]
y2 = [2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50,2,10,20,30,40,50]
z2 = cv_auc


# In[ ]:


# https://plot.ly/python/3d-axes/
trace1 = go.Scatter3d(x=x1,y=y1,z=z1, name = 'train')
trace2 = go.Scatter3d(x=x2,y=y2,z=z2, name = 'Cross validation')
data = [trace1, trace2]
enable_plotly_in_cell()

layout = go.Layout(scene = dict(
        xaxis = dict(title='max_depth'),
        yaxis = dict(title='min_sample_split'),
        zaxis = dict(title='AUC'),))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename='3d-scatter-colorscale')


# In[ ]:





plt.plot(np.log(param_grid['alpha']), train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.log(param_grid['alpha']),train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(np.log(param_grid['alpha']), cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(np.log(param_grid['alpha']),cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(np.log(param_grid['alpha']), train_auc, label='Train AUC points')
plt.scatter(np.log(param_grid['alpha']), cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs alpha ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()
# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 291
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

DT_SET_4_best = DecisionTreeClassifier(max_depth = best_max_depth ,min_samples_split = best_min_samples_split,class_weight= 'balanced')
DT_SET_4_best.fit(X_train_SET_4, y_train_SET_4)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:





# In[ ]:


# predicted probablities 
y_train_pred_prob = DT_SET_4_best.predict_proba( X_train_SET_4)    
y_test_pred_prob = DT_SET_4_best.predict_proba( X_test_SET_4)


# In[ ]:


y_train_pred_prob


# In[ ]:


# predicted values 
y_train_pred = DT_SET_4_best.predict( X_train_SET_4)    
y_test_pred = DT_SET_4_best.predict( X_test_SET_4)


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
sn.heatmap(df_cm, annot=True, fmt ="d")


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
sn.heatmap(df_cm, annot=True, fmt ="d")


# In[ ]:





# ## Wordcloud

# In[ ]:


# making a dataframe of y_test & y_test_pred

df_y_test_y_pred = df = pd.DataFrame(list(zip(y_test, y_test_pred )), columns =['y_test', 'y_pred'])
df_y_test_y_pred.shape


# In[ ]:


df_false_positives =  df_y_test_y_pred[(df_y_test_y_pred['y_test'] == 0) & (df_y_test_y_pred['y_pred'] ==1)]
df_false_positives.shape


# In[ ]:


data_false_positives = X_test
#data_false_positives.reset_index(inplace=True)
data_false_positives.shape


# In[ ]:


#data_false_positives = data_false_positives.loc[data_false_positives.index & df_y_test_y_pred.index]
data_false_positives = data_false_positives[data_false_positives.index.isin(df_false_positives.index)]
#data_false_positives = data_false_positives.loc[data_false_positives.index.intersection(df_false_positives.index)]
data_false_positives.shape


# In[ ]:


l1 = list(data_false_positives["preprocessed_essays"])
len(l1)


# In[ ]:


giant_list = [] # for storing every word in all of the sentances
for sentance in l1:
    temp_list = sentance.split()
    giant_list = giant_list + temp_list     


# In[ ]:


from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(str(giant_list)) 


# In[ ]:


# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:


box_plot_price = data_false_positives.boxplot(column = ["price"])


# In[ ]:


pdf_previous_submisssions = data_false_positives["teacher_number_of_previously_posted_projects"].plot.kde()


# <h2>2.5 [Task-2]Getting top 5k features using `feature_importances_`  on SET_2 </h2>

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


print(DT_SET_2_best.feature_importances_)
importance = list(DT_SET_2_best.feature_importances_)


# In[ ]:


feature_importance = pd.DataFrame(list(zip(list_of_vectorized_features_SET_2,importance)), columns =['features', 'importance']) 
feature_importance


# In[ ]:


top_5k_feature =  feature_importance.sort_values(by = 'importance', ascending = False).head(5000)
index_of_best = list(top_5k_feature.index)


# In[ ]:


X_test_SET_5 = X_test_SET_2[:, index_of_best]
X_train_SET_5 = X_train_SET_2[:, index_of_best]


# ## Logisitic Regression

# In[ ]:


# c = 1/lambda 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

LR_SET_5 = LogisticRegression(class_weight = 'balanced')
c_range = [10**x for x in range(-4,4) ]
#c_range = [ x for x in np.linspace(0.001,0.1,50)]
print(c_range)
param_grid = dict(C = c_range)
#print(param_grid) 


# In[ ]:


grid = GridSearchCV(LR_SET_5, param_grid, cv=10, scoring='roc_auc', return_train_score=True, n_jobs= -1)
grid.fit(X_train_SET_5, y_train_SET_5)


# In[ ]:


pd.DataFrame(grid.cv_results_)


# In[ ]:


# best K according to gridsearchCV 
best_c = grid.best_params_.get("C")
print(best_c)


# In[ ]:


train_auc= grid.cv_results_['mean_train_score']
train_auc_std= grid.cv_results_['std_train_score']
cv_auc = grid.cv_results_['mean_test_score'] 
cv_auc_std= grid.cv_results_['std_test_score']


# In[ ]:


plt.plot(param_grid['C'], train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['C'],train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(param_grid['C'], cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(param_grid['C'],cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(param_grid['C'], train_auc, label='Train AUC points')
plt.scatter(param_grid['C'], cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title(" Train & test plot of  AUC Vs K ") # try changing socring = "roc_auc" to  socring = "accuracy " so as to get the real ERROR plots 
plt.grid()
plt.show()


# In[ ]:


# this is for plot 2 
# best k from the above plot is K = 291
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

LR_SET_5_best = LogisticRegression(C=best_c,n_jobs=-1,class_weight = 'balanced')
LR_SET_5_best.fit(X_train_SET_5, y_train_SET_5)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs


# In[ ]:


# predicted probablities 
y_train_pred_prob = LR_SET_5_best.predict_proba( X_train_SET_5)    
y_test_pred_prob = LR_SET_5_best.predict_proba( X_test_SET_5)


# In[ ]:


# predicted values 
y_train_pred = LR_SET_5_best.predict( X_train_SET_5)    
y_test_pred = LR_SET_5_best.predict( X_test_SET_5)


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
sn.heatmap(df_cm, annot=True, fmt =" d" )


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
sn.heatmap(df_cm, annot=True,fmt = "d")


# In[ ]:





# <h1>3. Conclusion</h1>

# In[ ]:


# Please compare all your models using Prettytable library

