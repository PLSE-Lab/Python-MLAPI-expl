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

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
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


project_data = pd.read_csv('../input/appliedai-donorchoose/train_data.csv')
resource_data = pd.read_csv('../input/appliedai-donorchoose/resources.csv')


# In[ ]:


print("Number of data points in train data", project_data.shape)
print('-'*50)
print("The attributes of data :", project_data.columns.values)


# In[ ]:


print("Number of data points in train data", resource_data.shape)
print(resource_data.columns.values)
resource_data.head(2)


# In[ ]:


project_data.head(2)


# # Mergeing both project_data and resource_data based on id

# In[ ]:


price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
project_data = pd.merge(project_data, price_data, on='id', how='left')
project_data.head(2)


# # Seperating Negative and Positive datapoints and selecting 10000 from each class,then combining them.

# In[ ]:


Negative_Class = project_data.loc[project_data['project_is_approved'] == 0]
Positive_Class = project_data.loc[project_data['project_is_approved'] == 1]
project_data= Negative_Class[:10000].append(Positive_Class[:10000])
project_data= Negative_Class[:10000].append(Positive_Class[:10000])

#shuffle a dataframe pandas: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
project_data = project_data.sample(frac=1).reset_index(drop=True)


# In[ ]:


project_data.head(2)


# # Splitting Train,Test,CV data before vectorizing to avoid Memory Leak

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
y=project_data['project_is_approved']
project_data.drop(['project_is_approved'], axis=1)

# split the data set into train and test
X_train, X_test, y_train, y_test = train_test_split(project_data, y,test_size=0.3, random_state=0)

# split the train data set into cross validation train and cross validation test
X_tr, X_cv, y_tr, y_cv = train_test_split(X_train, y_train, test_size=0.3)


# ## 1.2 preprocessing of `project_subject_categories`

# In[ ]:


catogories = list(X_tr['project_subject_categories'].values)
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
    
X_tr['clean_categories'] = cat_list
X_tr.drop(['project_subject_categories'], axis=1, inplace=True)


# ## 1.3 preprocessing of `project_subject_subcategories`

# In[ ]:


sub_catogories = list(X_tr['project_subject_subcategories'].values)
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

X_tr['clean_subcategories'] = sub_cat_list
X_tr.drop(['project_subject_subcategories'], axis=1, inplace=True)


# ## 1.3 Text preprocessing

# In[ ]:


# merge two column text dataframe: 
X_tr["essay"] = X_tr["project_essay_1"].map(str) +                        X_tr["project_essay_2"].map(str) +                         X_tr["project_essay_3"].map(str) +                         X_tr["project_essay_4"].map(str)


# In[ ]:


X_tr.head(2)


# In[ ]:


#### 1.4.2.3 Using Pretrained Models: TFIDF weighted W2V


# In[ ]:


# printing some random reviews
print(X_tr['essay'].values[0])
print("="*50)
print(X_tr['essay'].values[150])
print("="*50)
print(X_tr['essay'].values[1000])
print("="*50)
print(X_tr['essay'].values[2000])
print("="*50)
print(X_tr['essay'].values[99])
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


sent = decontracted(X_tr['essay'].values[2000])
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
for sentance in tqdm(X_tr['essay'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_essays.append(sent.lower().strip())


# In[ ]:


# after preprocesing
preprocessed_essays[2000]


# In[ ]:


preprocessed_essays_train=preprocessed_essays


# <h2><font color='red'> 1.4 Preprocessing of `project_title`</font></h2>

# In[ ]:


# similarly you can preprocess the titles also


# In[ ]:


# similarly you can preprocess the titles also
# similarly you can preprocess the titles also
# Combining all the above stundents 
from tqdm import tqdm
preprocessed_titles = []
# tqdm is for printing the status bar
for sentance in tqdm(X_tr['project_title'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
    preprocessed_titles.append(sent.lower().strip())


# In[ ]:


preprocessed_titles_train = preprocessed_titles


# In[ ]:


# count of all the words in corpus python: https://stackoverflow.com/a/22898595/4084039
from collections import Counter
my_counter = Counter()
for word in X_tr['clean_subcategories'].values:
    my_counter.update(word.split())
    
sub_cat_dict = dict(my_counter)
sorted_sub_cat_dict = dict(sorted(sub_cat_dict.items(), key=lambda kv: kv[1]))

my_counter = Counter()
for word in X_tr['clean_categories'].values:
    my_counter.update(word.split())

cat_dict = dict(my_counter)
sorted_cat_dict = dict(sorted(cat_dict.items(), key=lambda kv: kv[1]))


# ## 1.5 Preparing data for models

# In[ ]:


project_data.columns


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
# 
# - quantity : numerical (optinal)
#        - teacher_number_of_previously_posted_projects : numerical
#        - price : numerical

# ### 1.5.1 Vectorizing Categorical data

# - https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/handling-categorical-and-numerical-features/

# In[ ]:


def Categorical_Probabilities(ColName,alpha):
        Cat_Counter=dict(X_tr[ColName].value_counts())
        probs={}
        for i in Cat_Counter.keys():
            df_0=  np.round(len(X_tr[(X_tr["project_is_approved"] == 0) & (X_tr[ColName]==i)])+alpha*10/(Cat_Counter[i]+20*alpha),4)
            df_1=  np.round(len(X_tr[(X_tr["project_is_approved"] == 1) & (X_tr[ColName]==i)])+alpha*10/(Cat_Counter[i]+20*alpha),4)
            probs[i] = [ np.round(df_0/(df_0+df_1),3), np.round(df_1/(df_0+df_1),3)]
            
        return probs    
    
def Categorical_Hack(a,b):
    if a in b.keys():
        return b[a]
    else:
        return [0.5,0.5]      


# In[ ]:


Categorical_Probs_cat = Categorical_Probabilities("clean_categories",1)
clean_categories_cat_hack = X_tr.apply(lambda x: Categorical_Hack(x["clean_categories"],Categorical_Probs_cat), axis=1)  
clean_categories_cat_hack=np.asarray(list(clean_categories_cat_hack))


# In[ ]:


features_final=list(set(X_tr["clean_categories"]))
len(features_final)


# In[ ]:


Categorical_Probs_subcat = Categorical_Probabilities("clean_subcategories",1)
clean_sub_categories_cat_hack = X_tr.apply(lambda x: Categorical_Hack(x["clean_subcategories"],Categorical_Probs_subcat), axis=1)  
len(clean_sub_categories_cat_hack)
clean_sub_categories_cat_hack=np.asarray(list(clean_sub_categories_cat_hack))


# In[ ]:


features_final=list(set(X_tr["clean_subcategories"]))
len(features_final)


# In[ ]:


clean_sub_categories_cat_hack.shape


# In[ ]:


# you can do the similar thing with state, teacher_prefix and project_grade_category also


# In[ ]:


Categorical_Probs_school_state = Categorical_Probabilities("school_state",1)
school_state_cat_hack = X_tr.apply(lambda x: Categorical_Hack(x["school_state"], Categorical_Probs_school_state), axis=1)  
school_state_cat_hack=np.asarray(list(school_state_cat_hack))
school_state_cat_hack.shape


# In[ ]:


Categorical_Probs_project_grade_category = Categorical_Probabilities("project_grade_category",1)
project_grade_category_cat_hack = X_tr.apply(lambda x: Categorical_Hack(x["project_grade_category"], Categorical_Probs_project_grade_category), axis=1)  
project_grade_category_cat_hack=np.asarray(list(project_grade_category_cat_hack))
project_grade_category_cat_hack.shape


# In[ ]:


X_tr['teacher_prefix']=X_tr['teacher_prefix'].fillna("Not Specefied")
Categorical_Probs_teacher_prefix = Categorical_Probabilities("teacher_prefix",1)
teacher_prefix_cat_hack = X_tr.apply(lambda x: Categorical_Hack(x["teacher_prefix"], Categorical_Probs_teacher_prefix), axis=1)  
teacher_prefix_cat_hack=np.asarray(list(teacher_prefix_cat_hack))
teacher_prefix_cat_hack.shape


# ### 1.5.2 Vectorizing Text data

# #### 1.5.2.1 Bag of words

# In[ ]:


# We are considering only the words which appeared in at least 10 documents(rows or projects).
vectorizer_essays_bow = CountVectorizer(min_df=10)
text_bow = vectorizer_essays_bow.fit_transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_bow.shape)


# In[ ]:


tr_features_bow=vectorizer_essays_bow.get_feature_names()


# In[ ]:


# you can vectorize the title also 
# before you vectorize the title make sure you preprocess it


# In[ ]:


vectorizer_titles_bow = CountVectorizer(min_df=10)
titles_bow = vectorizer_titles_bow.fit_transform(preprocessed_titles)
print("Shape of matrix after one hot encodig ",titles_bow.shape)


# #### 1.5.2.2 TFIDF vectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_essays_tfidf = TfidfVectorizer(min_df=10)
text_tfidf = vectorizer_essays_tfidf.fit_transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_tfidf.shape)
text_tfidf_train =text_tfidf


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_titles_tfidf = TfidfVectorizer(min_df=10)
title_tfidf = vectorizer_titles_tfidf.fit_transform(preprocessed_titles)
print("Shape of matrix after one hot encodig ",title_tfidf.shape)


# #### 1.5.2.3 Using Pretrained Models: Avg W2V

# In[ ]:


import os
os.listdir("../input/glove-vectors-applied-ai")


# In[ ]:


# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
# make sure you have the glove_vectors file
with open('../input/glove-vectors-applied-ai/glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words =  set(model.keys())


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_essays): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors.append(vector)

print(len(avg_w2v_vectors))
print(len(avg_w2v_vectors[0]))


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
avg_w2v_titles_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_titles): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_titles_vectors.append(vector)

print(len(avg_w2v_titles_vectors))
print(len(avg_w2v_titles_vectors[0]))


# #### 1.5.2.3 Using Pretrained Models: TFIDF weighted W2V

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
tfidf_model_essays = TfidfVectorizer()
tfidf_model_essays.fit(preprocessed_essays)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model_essays.get_feature_names(), list(tfidf_model_essays.idf_)))
tfidf_words = set(tfidf_model_essays.get_feature_names())


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_essays): # for each review/sentence
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
    tfidf_w2v_vectors.append(vector)

print(len(tfidf_w2v_vectors))
print(len(tfidf_w2v_vectors[0]))


# In[ ]:


# Similarly you can vectorize for title also


# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
tfidf_title_model = TfidfVectorizer()
tfidf_title_model.fit(preprocessed_titles)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary_title = dict(zip(tfidf_title_model.get_feature_names(), list(tfidf_title_model.idf_)))
tfidf_title_words = set(tfidf_title_model.get_feature_names())

# average Word2Vec
# compute average word2vec for each review.
tfidf_title_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_titles): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in glove_words) and (word in tfidf_title_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary_title[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_title_w2v_vectors.append(vector)

print(len(tfidf_title_w2v_vectors))
print(len(tfidf_title_w2v_vectors[0]))


# ### 1.5.3 Vectorizing Numerical features

# In[ ]:


# check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# price_standardized = standardScalar.fit(project_data['price'].values)
# this will rise the error
# ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# Reshape your data either using array.reshape(-1, 1)

price_scalar = StandardScaler()
price_scalar.fit(X_tr['price'].values.reshape(-1,1)) # finding the mean and standard deviation of this data
print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now standardize the data with above maen and variance.
price_standardized = price_scalar.transform(X_tr['price'].values.reshape(-1, 1))
price_standardized_train = price_standardized


# In[ ]:


price_standardized_train


# In[ ]:


# check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# price_standardized = standardScalar.fit(project_data['price'].values)
# this will rise the error
# ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# Reshape your data either using array.reshape(-1, 1)

quantity_scalar = StandardScaler()
quantity_scalar.fit(X_tr['quantity'].values.reshape(-1,1)) # finding the mean and standard deviation of this data
print(f"Mean : {quantity_scalar.mean_[0]}, Standard deviation : {np.sqrt(quantity_scalar.var_[0])}")

# Now standardize the data with above maen and variance.
quantity_standardized = quantity_scalar.transform(X_tr['quantity'].values.reshape(-1, 1))
quantity_standardized_train =quantity_standardized


# In[ ]:


# check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# price_standardized = standardScalar.fit(project_data['price'].values)
# this will rise the error
# ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# Reshape your data either using array.reshape(-1, 1)

teacher_number_of_previously_posted_projects_scalar = StandardScaler()
teacher_number_of_previously_posted_projects_scalar.fit(X_tr['teacher_number_of_previously_posted_projects'].values.reshape(-1,1)) # finding the mean and standard deviation of this data
print(f"Mean : {teacher_number_of_previously_posted_projects_scalar.mean_[0]}, Standard deviation : {np.sqrt(teacher_number_of_previously_posted_projects_scalar.var_[0])}")

# Now standardize the data with above maen and variance.
teacher_number_of_previously_posted_projects_standardized = teacher_number_of_previously_posted_projects_scalar.transform(X_tr['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))
teacher_number_of_previously_posted_projects_standardized_train =teacher_number_of_previously_posted_projects_standardized


# ### 1.5.4 Merging all the above features

# - we need to merge all the numerical vectors i.e catogorical, text, numerical vectors

# In[ ]:


print(clean_categories_cat_hack.shape)
print(clean_sub_categories_cat_hack.shape)
print(school_state_cat_hack.shape)
print(project_grade_category_cat_hack.shape)
print(teacher_prefix_cat_hack.shape)
print(teacher_number_of_previously_posted_projects_standardized.shape)
print(quantity_standardized.shape)
print(price_standardized.shape)
print(text_bow.shape)
print(titles_bow.shape)


# In[ ]:


#clean_categories_cat_hack = clean_categories_cat_hack.ravel().reshape(-1,1)
#clean_sub_categories_cat_hack = clean_sub_categories_cat_hack.ravel().reshape(-1,1)
#school_state_cat_hack =school_state_cat_hack.ravel().reshape(-1,1)
#project_grade_category_cat_hack = project_grade_category_cat_hack.ravel().reshape(-1,1)
#teacher_prefix_cat_hack = teacher_prefix_cat_hack.ravel().reshape(-1,1)


# In[ ]:


clean_categories_cat_hack[0]


# In[ ]:


print(X_tr['teacher_prefix'].fillna("Not Specefied").shape)
print(X_tr['clean_categories'].shape)
print(X_tr['clean_subcategories'].shape)
print(X_tr['school_state'].shape)
print(X_tr['school_state'].shape)


# In[ ]:


#Preparing Data for set-1
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx 
# X1 Data for set1
X1_tr = hstack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,text_bow,titles_bow))
# X2 Data for set2
X2_tr = hstack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,text_tfidf,title_tfidf))


# In[ ]:


print(X1_tr.shape)
print(X2_tr.shape)


# In[ ]:


avg_w2v_vectors[0].shape


# In[ ]:


avg_w2v_vectors = np.asarray(avg_w2v_vectors)
print(avg_w2v_vectors.shape)
avg_w2v_titles_vectors = np.asarray(avg_w2v_titles_vectors)
print(avg_w2v_titles_vectors.shape)


# In[ ]:


print(clean_categories_cat_hack.shape)
print(clean_sub_categories_cat_hack.shape)
print(school_state_cat_hack.shape)
print(project_grade_category_cat_hack.shape)
print(teacher_prefix_cat_hack.shape)
print(teacher_number_of_previously_posted_projects_standardized.shape)
print(quantity_standardized.shape)
print(price_standardized.shape)
print(avg_w2v_vectors.shape)
print(avg_w2v_titles_vectors.shape)


# In[ ]:


#https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.column_stack.html
X3_tr=np.column_stack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,avg_w2v_vectors,avg_w2v_titles_vectors))
X3_tr.shape


# In[ ]:


# X4 Data for set4
X4_tr = np.column_stack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,tfidf_w2v_vectors,tfidf_title_w2v_vectors))

#Preparing labels
y_tr=X_tr['project_is_approved']


# In[ ]:


print(X1_tr.shape)
print(X2_tr.shape)
print(X3_tr.shape)
print(X4_tr.shape)
print(y_tr.shape)


# __ Computing Sentiment Scores__

# In[ ]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# import nltk
# nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

for_sentiment = 'a person is a person no matter how small dr seuss i teach the smallest students with the biggest enthusiasm for learning my students learn in many different ways using all of our senses and multiple intelligences i use a wide rangeof techniques to help all my students succeed students in my class come from a variety of different backgrounds which makesfor wonderful sharing of experiences and cultures including native americans our school is a caring community of successful learners which can be seen through collaborative student project based learning in and out of the classroom kindergarteners in my class love to work with hands on materials and have many different opportunities to practice a skill before it ismastered having the social skills to work cooperatively with friends is a crucial aspect of the kindergarten curriculummontana is the perfect place to learn about agriculture and nutrition my students love to role play in our pretend kitchenin the early childhood classroom i have had several kids ask me can we try cooking with real food i will take their idea and create common core cooking lessons where we learn important math and writing concepts while cooking delicious healthy food for snack time my students will have a grounded appreciation for the work that went into making the food and knowledge of where the ingredients came from as well as how it is healthy for their bodies this project would expand our learning of nutrition and agricultural cooking recipes by having us peel our own apples to make homemade applesauce make our own bread and mix up healthy plants from our classroom garden in the spring we will also create our own cookbooks to be printed and shared with families students will gain math and literature skills as well as a life long enjoyment for healthy cooking nannan'
ss = sid.polarity_scores(for_sentiment)

for k in ss:
    print('{0}: {1}, '.format(k, ss[k]), end='')

# we can use these 4 things as features/attributes (neg, neu, pos, compound)
# neg: 0.0, neu: 0.753, pos: 0.247, compound: 0.93


# # Preprocessing and Vectorising Test and CV data

# ## preprocessing of project_subject_categories

# In[ ]:


catogories_cv = list(X_cv['project_subject_categories'].values)
# remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python
cat_list_cv = []
for i in catogories_cv:
    temp = ""
    # consider we have text like this "Math & Science, Warmth, Care & Hunger"
    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]
        if 'The' in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"
            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')
        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"
        temp+=j.strip()+" " #" abc ".strip() will return "abc", remove the trailing spaces
        temp = temp.replace('&','_') # we are replacing the & value into 
    cat_list_cv.append(temp.strip())
    
X_cv['clean_categories'] = cat_list_cv
X_cv.drop(['project_subject_categories'], axis=1, inplace=True)

from collections import Counter
my_counter = Counter()
for word in X_cv['clean_categories'].values:
    my_counter.update(word.split())

cat_dict = dict(my_counter)
sorted_cat_dict = dict(sorted(cat_dict.items(), key=lambda kv: kv[1]))


# ## preprocessing of project_subject_subcategories

# In[ ]:


sub_catogories = list(X_cv['project_subject_subcategories'].values)
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

X_cv['clean_subcategories'] = sub_cat_list
X_cv.drop(['project_subject_subcategories'], axis=1, inplace=True)

# count of all the words in corpus python: https://stackoverflow.com/a/22898595/4084039
my_counter = Counter()
for word in X_cv['clean_subcategories'].values:
    my_counter.update(word.split())
    
sub_cat_dict = dict(my_counter)
sorted_sub_cat_dict = dict(sorted(sub_cat_dict.items(), key=lambda kv: kv[1]))


# 
# 
# 
# ## Text preprocessing

# In[ ]:


# merge two column text dataframe: 
X_cv["essay"] = X_cv["project_essay_1"].map(str) +                        X_cv["project_essay_2"].map(str) +                         X_cv["project_essay_3"].map(str) +                         X_cv["project_essay_4"].map(str)


# In[ ]:


# Combining all the above stundents 
from tqdm import tqdm
preprocessed_essays = []
# tqdm is for printing the status bar
for sentance in tqdm(X_cv['essay'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_essays.append(sent.lower().strip())
# similarly you can preprocess the titles also
# similarly you can preprocess the titles also
# Combining all the above stundents 
from tqdm import tqdm
preprocessed_titles = []
# tqdm is for printing the status bar
for sentance in tqdm(X_cv['project_title'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
    preprocessed_titles.append(sent.lower().strip()) 
    
preprocessed_essays_cv=preprocessed_essays   
preprocessed_titles_cv = preprocessed_titles


# # Categorical Features

# In[ ]:


clean_categories_cat_hack = X_cv.apply(lambda x: Categorical_Hack(x["clean_categories"],Categorical_Probs_cat), axis=1)  
clean_categories_cat_hack=np.asarray(list(clean_categories_cat_hack))
clean_categories_cat_hack.shape


# In[ ]:


clean_sub_categories_cat_hack = X_cv.apply(lambda x: Categorical_Hack(x["clean_subcategories"],Categorical_Probs_subcat), axis=1)  
clean_sub_categories_cat_hack=np.asarray(list(clean_sub_categories_cat_hack))
clean_sub_categories_cat_hack.shape


# In[ ]:


school_state_cat_hack = X_cv.apply(lambda x: Categorical_Hack(x["school_state"], Categorical_Probs_school_state), axis=1)  
school_state_cat_hack=np.asarray(list(school_state_cat_hack))
school_state_cat_hack.shape


# In[ ]:


project_grade_category_cat_hack = X_cv.apply(lambda x: Categorical_Hack(x["project_grade_category"], Categorical_Probs_project_grade_category), axis=1)  
project_grade_category_cat_hack=np.asarray(list(project_grade_category_cat_hack))
project_grade_category_cat_hack.shape


# In[ ]:


X_cv['teacher_prefix']=X_cv['teacher_prefix'].fillna("Not Specefied")
teacher_prefix_cat_hack = X_cv.apply(lambda x: Categorical_Hack(x["teacher_prefix"], Categorical_Probs_teacher_prefix), axis=1)  
teacher_prefix_cat_hack=np.asarray(list(teacher_prefix_cat_hack))
teacher_prefix_cat_hack.shape


# # Text Data

# In[ ]:


#BOW
text_bow = vectorizer_essays_bow.transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_bow.shape)


titles_bow = vectorizer_titles_bow.transform(preprocessed_titles)
print("Shape of matrix after one hot encodig ",titles_bow.shape)

bow_words_tr=vectorizer_essays_bow.get_feature_names()+vectorizer_titles_bow.get_feature_names()
#print(bow_words_tr)


# In[ ]:


k=["6"]
j=["5"]
k+j


# In[ ]:


#TFIDF
text_tfidf = vectorizer_essays_tfidf.transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_tfidf.shape)

title_tfidf = vectorizer_titles_tfidf.transform(preprocessed_titles)
print("Shape of matrix after one hot encodig ",title_tfidf.shape)
text_tfidf_cv = text_tfidf

tfidf_words_tr=vectorizer_essays_tfidf.get_feature_names()+vectorizer_titles_tfidf.get_feature_names()


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_essays): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors.append(vector)

print(len(avg_w2v_vectors))
print(len(avg_w2v_vectors[0]))

# average Word2Vec
# compute average word2vec for each review.
avg_w2v_titles_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_titles): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_titles_vectors.append(vector)

print(len(avg_w2v_titles_vectors))
print(len(avg_w2v_titles_vectors[0]))


# In[ ]:


#TFIDF weighted W2V

tfidf_model_essays.transform(preprocessed_essays)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model_essays.get_feature_names(), list(tfidf_model_essays.idf_)))
tfidf_words = set(tfidf_model_essays.get_feature_names())

# average Word2Vec
# compute average word2vec for each review.
tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_essays): # for each review/sentence
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
    tfidf_w2v_vectors.append(vector)

print(len(tfidf_w2v_vectors))
print(len(tfidf_w2v_vectors[0]))


tfidf_title_model.transform(preprocessed_titles)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary_title = dict(zip(tfidf_title_model.get_feature_names(), list(tfidf_title_model.idf_)))
tfidf_title_words = set(tfidf_title_model.get_feature_names())

# average Word2Vec
# compute average word2vec for each review.
tfidf_title_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_titles): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in glove_words) and (word in tfidf_title_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary_title[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_title_w2v_vectors.append(vector)

print(len(tfidf_title_w2v_vectors))
print(len(tfidf_title_w2v_vectors[0]))


# In[ ]:


# Numerical Features
price_standardized = price_scalar.transform(X_cv['price'].values.reshape(-1, 1))
quantity_standardized = quantity_scalar.transform(X_cv['quantity'].values.reshape(-1, 1))
teacher_number_of_previously_posted_projects_standardized = teacher_number_of_previously_posted_projects_scalar.transform(X_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))

price_standardized_cv= price_standardized
quantity_standardized_cv =quantity_standardized
teacher_number_of_previously_posted_projects_standardized_cv =teacher_number_of_previously_posted_projects_standardized


# # Mearging CV features

# In[ ]:


print(clean_categories_cat_hack.shape)
print(clean_sub_categories_cat_hack.shape)
print(school_state_cat_hack.shape)
print(project_grade_category_cat_hack.shape)
print(teacher_prefix_cat_hack.shape)
print(teacher_number_of_previously_posted_projects_standardized.shape)
print(quantity_standardized.shape)
print(price_standardized.shape)
print(text_bow.shape)
print(titles_bow.shape)


# In[ ]:


#Preparing Data for set-1
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx 
# X1 Data for set1
X1_cv = hstack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,text_bow,titles_bow))
# X2 Data for set2
X2_cv = hstack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,text_tfidf,title_tfidf))


# In[ ]:


avg_w2v_vectors[0].shape


# In[ ]:


avg_w2v_vectors = np.asarray(avg_w2v_vectors)
print(avg_w2v_vectors.shape)
avg_w2v_titles_vectors = np.asarray(avg_w2v_titles_vectors)
print(avg_w2v_titles_vectors.shape)


# In[ ]:



print(clean_categories_cat_hack.shape)
print(clean_sub_categories_cat_hack.shape)
print(school_state_cat_hack.shape)
print(project_grade_category_cat_hack.shape)
print(teacher_prefix_cat_hack.shape)
print(teacher_number_of_previously_posted_projects_standardized.shape)
print(quantity_standardized.shape)
print(price_standardized.shape)
print(avg_w2v_vectors.shape)
print(avg_w2v_titles_vectors.shape)


# In[ ]:


#https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.column_stack.html
X3_cv=np.column_stack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,avg_w2v_vectors,avg_w2v_titles_vectors))
X3_tr.shape


# In[ ]:


# X4 Data for set4
X4_cv = np.column_stack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,tfidf_w2v_vectors,tfidf_title_w2v_vectors))


# In[ ]:


X4_cv.shape


# In[ ]:


#Preparing labels
y_cv=X_cv['project_is_approved']


# In[ ]:


print(X1_cv.shape)
print(X2_cv.shape)
print(X3_cv.shape)
print(X4_cv.shape)
print(y_cv.shape)


# ## Featurization for test Data

# # Preprocessing of Test Data

# In[ ]:



catogories_test = list(X_test['project_subject_categories'].values)
# remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python
cat_list_test = []
for i in catogories_test:
    temp = ""
    # consider we have text like this "Math & Science, Warmth, Care & Hunger"
    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]
        if 'The' in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"
            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')
        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"
        temp+=j.strip()+" " #" abc ".strip() will return "abc", remove the trailing spaces
        temp = temp.replace('&','_') # we are replacing the & value into 
    cat_list_test.append(temp.strip())
    
X_test['clean_categories'] = cat_list_test
X_test.drop(['project_subject_categories'], axis=1, inplace=True)

from collections import Counter
my_counter = Counter()
for word in X_test['clean_categories'].values:
    my_counter.update(word.split())

cat_dict = dict(my_counter)
sorted_cat_dict = dict(sorted(cat_dict.items(), key=lambda kv: kv[1]))

sub_catogories = list(X_test['project_subject_subcategories'].values)
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

X_test['clean_subcategories'] = sub_cat_list
X_test.drop(['project_subject_subcategories'], axis=1, inplace=True)

# count of all the words in corpus python: https://stackoverflow.com/a/22898595/4084039
my_counter = Counter()
for word in X_test['clean_subcategories'].values:
    my_counter.update(word.split())
    
sub_cat_dict = dict(my_counter)
sorted_sub_cat_dict = dict(sorted(sub_cat_dict.items(), key=lambda kv: kv[1]))


# In[ ]:


# merge two column text dataframe: 
X_test["essay"] = X_test["project_essay_1"].map(str) +                        X_test["project_essay_2"].map(str) +                         X_test["project_essay_3"].map(str) +                         X_test["project_essay_4"].map(str)


# Combining all the above stundents 
from tqdm import tqdm
preprocessed_essays = []
# tqdm is for printing the status bar
for sentance in tqdm(X_test['essay'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_essays.append(sent.lower().strip())
# similarly you can preprocess the titles also
# similarly you can preprocess the titles also
# Combining all the above stundents 
from tqdm import tqdm
preprocessed_titles = []
# tqdm is for printing the status bar
for sentance in tqdm(X_test['project_title'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
    preprocessed_titles.append(sent.lower().strip())    
preprocessed_essays_test = preprocessed_essays
preprocessed_titles_test = preprocessed_titles


# # Categorical Data

# In[ ]:


clean_categories_cat_hack = X_test.apply(lambda x: Categorical_Hack(x["clean_categories"],Categorical_Probs_cat), axis=1)  
clean_categories_cat_hack=np.asarray(list(clean_categories_cat_hack))
print(clean_categories_cat_hack.shape)

clean_sub_categories_cat_hack = X_test.apply(lambda x: Categorical_Hack(x["clean_subcategories"],Categorical_Probs_subcat), axis=1)  
clean_sub_categories_cat_hack=np.asarray(list(clean_sub_categories_cat_hack))
print(clean_sub_categories_cat_hack.shape)

school_state_cat_hack = X_test.apply(lambda x: Categorical_Hack(x["school_state"], Categorical_Probs_school_state), axis=1)  
school_state_cat_hack=np.asarray(list(school_state_cat_hack))
print(school_state_cat_hack.shape)

project_grade_category_cat_hack = X_test.apply(lambda x: Categorical_Hack(x["project_grade_category"], Categorical_Probs_project_grade_category), axis=1)  
project_grade_category_cat_hack=np.asarray(list(project_grade_category_cat_hack))
print(project_grade_category_cat_hack.shape)

X_test['teacher_prefix']=X_test['teacher_prefix'].fillna("Not Specefied")
teacher_prefix_cat_hack = X_test.apply(lambda x: Categorical_Hack(x["teacher_prefix"], Categorical_Probs_teacher_prefix), axis=1)  
teacher_prefix_cat_hack=np.asarray(list(teacher_prefix_cat_hack))
print(teacher_prefix_cat_hack.shape)


# # Text data and numerical data

# In[ ]:


#BOW
text_bow = vectorizer_essays_bow.transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_bow.shape)


titles_bow = vectorizer_titles_bow.transform(preprocessed_titles)
print("Shape of matrix after one hot encodig ",titles_bow.shape)

#TFIDF
text_tfidf = vectorizer_essays_tfidf.transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_tfidf.shape)
text_tfidf_test = text_tfidf

title_tfidf = vectorizer_titles_tfidf.transform(preprocessed_titles)
print("Shape of matrix after one hot encodig ",title_tfidf.shape)

# average Word2Vec
# compute average word2vec for each review.
avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_essays): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors.append(vector)

print(len(avg_w2v_vectors))
print(len(avg_w2v_vectors[0]))

# average Word2Vec
# compute average word2vec for each review.
avg_w2v_titles_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_titles): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_titles_vectors.append(vector)

print(len(avg_w2v_titles_vectors))
print(len(avg_w2v_titles_vectors[0]))

#TFIDF weighted W2V

tfidf_model_essays.transform(preprocessed_essays)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model_essays.get_feature_names(), list(tfidf_model_essays.idf_)))
tfidf_words = set(tfidf_model_essays.get_feature_names())

# average Word2Vec
# compute average word2vec for each review.
tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_essays): # for each review/sentence
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
    tfidf_w2v_vectors.append(vector)

print(len(tfidf_w2v_vectors))
print(len(tfidf_w2v_vectors[0]))


tfidf_title_model.transform(preprocessed_titles)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary_title = dict(zip(tfidf_title_model.get_feature_names(), list(tfidf_title_model.idf_)))
tfidf_title_words = set(tfidf_title_model.get_feature_names())

# average Word2Vec
# compute average word2vec for each review.
tfidf_title_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_titles): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in glove_words) and (word in tfidf_title_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary_title[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_title_w2v_vectors.append(vector)

print(len(tfidf_title_w2v_vectors))
print(len(tfidf_title_w2v_vectors[0]))

# Numerical Features
price_standardized = price_scalar.transform(X_test['price'].values.reshape(-1, 1))
quantity_standardized = quantity_scalar.transform(X_test['quantity'].values.reshape(-1, 1))
teacher_number_of_previously_posted_projects_standardized = teacher_number_of_previously_posted_projects_scalar.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))

price_standardized_test = price_standardized
quantity_standardized_test =quantity_standardized
teacher_number_of_previously_posted_projects_standardized_test =teacher_number_of_previously_posted_projects_standardized


# # Mearging Test features

# In[ ]:


print(clean_categories_cat_hack.shape)
print(clean_sub_categories_cat_hack.shape)
print(school_state_cat_hack.shape)
print(project_grade_category_cat_hack.shape)
print(teacher_prefix_cat_hack.shape)
print(teacher_number_of_previously_posted_projects_standardized.shape)
print(quantity_standardized.shape)
print(price_standardized.shape)
print(text_bow.shape)
print(titles_bow.shape)


# In[ ]:



#Preparing Data for set-1
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx 
# X1 Data for set1
X1_test = hstack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,text_bow,titles_bow))
# X2 Data for set2
X2_test = hstack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,text_tfidf,title_tfidf))


# In[ ]:


avg_w2v_vectors = np.asarray(avg_w2v_vectors)
print(avg_w2v_vectors.shape)
avg_w2v_titles_vectors = np.asarray(avg_w2v_titles_vectors)
print(avg_w2v_titles_vectors.shape)


# In[ ]:



#https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.column_stack.html
X3_test=np.column_stack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,avg_w2v_vectors,avg_w2v_titles_vectors))


# X4 Data for set4
X4_test = np.column_stack((clean_categories_cat_hack, clean_sub_categories_cat_hack,school_state_cat_hack,project_grade_category_cat_hack,teacher_prefix_cat_hack,teacher_number_of_previously_posted_projects_standardized,            quantity_standardized,price_standardized,tfidf_w2v_vectors,tfidf_title_w2v_vectors))


# In[ ]:


print(X1_test.shape)
print(X2_test.shape)
print(X3_test.shape)
print(X4_test.shape)
print(y_test.shape)


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
#     <li><strong>Hyper paramter tuning (best `depth` in range [1, 5, 10, 50, 100, 500, 100], and the best `min_samples_split` in range [5, 10, 100, 500])</strong>
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
#     <img src='train_cv_auc.JPG' width=300px></li>
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

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
depth_alpha=[1, 5, 10, 50, 100, 500, 100]
min_sample_split_alpha=[5,10,100,500]


# ### 2.4.1 Applying Decision Trees on BOW,<font color='red'> SET 1</font>

# In[ ]:


aucTotal_cv=[]
aucTotal_train=[]
HayperParametesDictAUC_cv=dict()
for i in min_sample_split_alpha:
    for j in depth_alpha:
        print("for min sample split =", i,"and max depth = ", j)
        clf = DecisionTreeClassifier(min_samples_split=i ,criterion='gini', max_depth=j,class_weight = 'balanced',random_state=42)
        clf.fit(X1_tr, y_tr)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X1_tr, y_tr)
        # predict the response on the crossvalidation train
        
        probs_cv = sig_clf.predict_proba(X1_cv)
    
        # predict the response on the train
        probs_train = sig_clf.predict_proba(X1_tr)
        
        # keep probabilities for the positive outcome only
        probs_cv = probs_cv[:, 1]
        probs_train = probs_train[:,1]
        
        # calculate AUC
        auc_cv = roc_auc_score(y_cv, probs_cv)
        auc_train = roc_auc_score(y_tr, probs_train)
        # evaluate CV accuracy
        probs_cv_acc=sig_clf.predict(X1_cv)
        acc_cv = accuracy_score(y_cv, probs_cv_acc, normalize=True) * float(100)
        print('\nCV accuracy for alpha = %f is %d%% and auc= %.3f' % (i, acc_cv,auc_cv))
        aucTotal_cv.append(auc_cv)
        aucTotal_train.append(auc_train) 
        HayperParametesDictAUC_cv["{},{}".format(i,j)]=auc_cv


# In[ ]:


aucTotal_cv[np.argmax(aucTotal_cv)]


# In[ ]:


#https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
import operator
alpha=max(HayperParametesDictAUC_cv.items(), key=operator.itemgetter(1))[0]
min_samples_split_cv,max_depth_cv = alpha.split(",")
min_samples_split_cv=int(min_samples_split_cv)
max_depth_cv = int(max_depth_cv)
hp_bow = alpha


# In[ ]:


clf = DecisionTreeClassifier(min_samples_split=min_samples_split_cv ,criterion='gini', max_depth=max_depth_cv,class_weight = 'balanced' ,random_state=42)
clf.fit(X1_tr, y_tr)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(X1_tr, y_tr)






probs_test   =  sig_clf.predict_proba(X1_test)
probs_train  = sig_clf.predict_proba(X1_tr)
y_pred_test  = sig_clf.predict(X1_test)
y_pred_train = sig_clf.predict(X1_tr)

# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
probs_train = probs_train[:,1]
# calculate AUC
auc = roc_auc_score(y_test, probs_test)






#y_pred_test = sig_clf.predict(X1_test)
#auc = roc_auc_score(y_test, y_pred_test)
auc_bow =auc
print("Test AUC: ",auc)
#y_pred_tr =  sig_clf.predict(X1_tr)
auc = roc_auc_score(y_tr, probs_train)
print("Train AUC: ",auc)


# In[ ]:



import matplotlib.pyplot as plt
probs_test = sig_clf.predict_proba(X1_test)[:,1]
probs_train = sig_clf.predict_proba(X1_tr)[:,1]
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs_test)
fpr2, tpr2, thresholds2 = roc_curve(y_tr, probs_train)
# plot no skill
plt.plot([0, 1], [0, 1], color='C0')
# plot the roc curve for the model
plt.plot(fpr, tpr, color='C1',label='cv')
plt.plot(fpr2, tpr2, color='g',label='train')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ERROR PLOT")
# show the plot
plt.show()


# ## Results based on the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,sig_clf.predict(X1_test) )
TPR=np.round(cm[1][1]/(cm[0][1]+cm[1][1])*100,2)
FPR=np.round(cm[0][0]/(cm[0][0]+cm[1][0])*100,2)
Precision= np.round(cm[1][1]/(cm[1][0]+cm[1][1])*100,2)
Recall = TPR
F1_Score= np.round((2*Precision*Recall)/(Recall+Precision),3)
F1_Score_l2=F1_Score


# In[ ]:


import seaborn as sns; sns.set()
ax = sns.heatmap(cm,annot=True,fmt="d",cmap="YlGnBu")


# In[ ]:


print("TPR: {}%\nFPR: {}%\nPrecision: {}%\nRecall: {}%\nF1 Score: {}% ".format(TPR,FPR,Precision,Recall,F1_Score))


# #### 2.4.1.1 Graphviz visualization of Decision Tree on BOW,<font color='red'> SET 1</font>

# In[ ]:


pre=["clean_categories_0","clean_categories_1","clean_sub_categories_0","clean_sub_categories_1","school_state_0","school_state_1","project_grade_0","project_grade_1","teacher_prefix_0","teacher_prefix_1"]
bow_words_tr=vectorizer_essays_bow.get_feature_names()+vectorizer_titles_bow.get_feature_names()
post=["TeacherPrevProjects","quantity_standardized","price_standardized" ]
final_features_bow=pre+bow_words_tr+post


# In[ ]:


len(final_features_bow)


# In[ ]:


#https://charleshsliao.wordpress.com/2017/05/20/decision-tree-in-python-with-graphviz-to-visualize/
###visualize and analyze the tree model###
###build a file to visualize 
from sklearn.tree import export_graphviz
export_graphviz(clf,out_file="mytree.dot",feature_names=final_features_bow)
###visualize the .dot file. Need to install graphviz seperately at first 
import graphviz
with open("mytree.dot") as f:
    dot_graph=f.read()
graphviz.Source(dot_graph)


# ## Collecting False Negative rows indexs

# In[ ]:


#https://stackoverflow.com/questions/43193880/how-to-get-row-number-in-dataframe-in-pandas
test=[]
FN=pd.DataFrame({"Actual":y_test, "Predicted":y_pred_test})
FN_indexs=list(FN[(FN["Actual"]==0) & (FN["Predicted"]==1)].index)
FN_2=FN[(FN["Actual"]==0) & (FN["Predicted"]==1)]


# # WordCloud based on essay text of false Nagative features :

# In[ ]:


Essay_Tests=""
for i in FN_indexs:
    Essay_Tests+=" {}".format(X_test.loc[i]['essay'])

from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(background_color="white", max_words=len(Essay_Tests), stopwords=stopwords)
wc.generate(Essay_Tests)
print ("Word Cloud for Duplicate Question pairs")
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()  


# ## Box Plot of 'Price' in FN

# In[ ]:


price=[]
for i in FN_indexs:
    price.append(X_test.loc[i]['price'])
import seaborn as sns
sns.set(style="whitegrid")
ax = sns.barplot(y="price", hue=FN_2["Predicted"], data=X_test)    


# In[ ]:


ax = sns.barplot(y="price", hue=FN_2["Actual"], data=X_test)


# # Distribution plot of 'teacher_number_of_previously_posted_projects' in FN
# 

# In[ ]:


teacher_number_of_previously_posted_projects=[]
for i in FN_indexs:
    teacher_number_of_previously_posted_projects.append(X_test.loc[i]['teacher_number_of_previously_posted_projects'])
import seaborn as sns, numpy as np
ax = sns.distplot(teacher_number_of_previously_posted_projects)  


# ### 2.4.2 Applying Decision Trees on TFIDF,<font color='red'> SET 2</font>

# In[ ]:


aucTotal_cv=[]
aucTotal_train=[]
HayperParametesDictAUC_cv=dict()
for i in min_sample_split_alpha:
    for j in depth_alpha:
        print("for min sample split =", i,"and max depth = ", j)
        clf = DecisionTreeClassifier(min_samples_split=i ,criterion='gini', max_depth=j,class_weight = 'balanced', random_state=42)
        clf.fit(X2_tr, y_tr)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X2_tr, y_tr)
        # predict the response on the crossvalidation train
        
        probs_cv = sig_clf.predict_proba(X2_cv)
    
        # predict the response on the train
        probs_train = sig_clf.predict_proba(X2_tr)
        
        # keep probabilities for the positive outcome only
        probs_cv = probs_cv[:, 1]
        probs_train = probs_train[:,1]
        
        # calculate AUC
        auc_cv = roc_auc_score(y_cv, probs_cv)
        auc_train = roc_auc_score(y_tr, probs_train)
        # evaluate CV accuracy
        probs_cv_acc=sig_clf.predict(X2_cv)
        acc_cv = accuracy_score(y_cv, probs_cv_acc, normalize=True) * float(100)
        print('\nCV accuracy for alpha = %f is %d%% and auc= %.3f' % (i, acc_cv,auc_cv))
        aucTotal_cv.append(auc_cv)
        aucTotal_train.append(auc_train) 
        HayperParametesDictAUC_cv["{},{}".format(i,j)]=auc_cv


# In[ ]:


#https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
import operator
alpha=max(HayperParametesDictAUC_cv.items(), key=operator.itemgetter(1))[0]
min_samples_split_cv,max_depth_cv = alpha.split(",")
min_samples_split_cv=int(min_samples_split_cv)
max_depth_cv = int(max_depth_cv)
hp_tfidf=alpha


# In[ ]:


clf_tfidf = DecisionTreeClassifier(min_samples_split=min_samples_split_cv ,criterion='gini',class_weight = 'balanced',max_depth=max_depth_cv, random_state=42)
clf_tfidf.fit(X2_tr, y_tr)
sig_clf = CalibratedClassifierCV(clf_tfidf, method="sigmoid")
sig_clf.fit(X2_tr, y_tr)
##########################

probs_test   =  sig_clf.predict_proba(X2_test)
probs_train  = sig_clf.predict_proba(X2_tr)
y_pred_test  = sig_clf.predict(X2_test)
y_pred_train = sig_clf.predict(X2_tr)

# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
probs_train = probs_train[:,1]
# calculate AUC
auc = roc_auc_score(y_test, probs_test)





auc_tfidf=auc
print("Test AUC: ",auc)
#y_pred_tr =  sig_clf.predict(X2_tr)
auc = roc_auc_score(y_tr, probs_train)
print("Train AUC: ",auc)


# In[ ]:


import matplotlib.pyplot as plt
probs_test = sig_clf.predict_proba(X2_test)[:,1]
probs_train = sig_clf.predict_proba(X2_tr)[:,1]
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs_test)
fpr2, tpr2, thresholds2 = roc_curve(y_tr, probs_train)
# plot no skill
plt.plot([0, 1], [0, 1], color='C0')
# plot the roc curve for the model
plt.plot(fpr, tpr, color='C1',label='cv')
plt.plot(fpr2, tpr2, color='g',label='train')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ERROR PLOT")
# show the plot
plt.show()


# ## Results based on the confusion matrix

# In[ ]:



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred_test)
TPR=np.round(cm[1][1]/(cm[0][1]+cm[1][1])*100,2)
FPR=np.round(cm[0][0]/(cm[0][0]+cm[1][0])*100,2)
Precision= np.round(cm[1][1]/(cm[1][0]+cm[1][1])*100,2)
Recall = TPR
F1_Score= np.round((2*Precision*Recall)/(Recall+Precision),3)
F1_Score_l2=F1_Score


# In[ ]:


import seaborn as sns; sns.set()
ax = sns.heatmap(cm,annot=True,fmt="d",cmap="YlGnBu")

print("TPR: {}%\nFPR: {}%\nPrecision: {}%\nRecall: {}%\nF1 Score: {}% ".format(TPR,FPR,Precision,Recall,F1_Score))


# #### 2.4.2.1 Graphviz visualization of Decision Tree on TFIDF,<font color='red'> SET 2</font>

# In[ ]:


pre=["clean_categories_0","clean_categories_1","clean_sub_categories_0","clean_sub_categories_1","school_state_0","school_state_1","project_grade_0","project_grade_1","teacher_prefix_0","teacher_prefix_1"]
tfidf_words_tr=vectorizer_essays_tfidf.get_feature_names()+vectorizer_titles_tfidf.get_feature_names()
post=["TeacherPrevProjects","quantity_standardized","price_standardized" ]
final_features_tfidf=pre+tfidf_words_tr+post


# In[ ]:


len(final_features_tfidf)


# In[ ]:


#https://charleshsliao.wordpress.com/2017/05/20/decision-tree-in-python-with-graphviz-to-visualize/
###visualize and analyze the tree model###
###build a file to visualize 
from sklearn.tree import export_graphviz
export_graphviz(clf,out_file="mytree.dot",feature_names=final_features_tfidf)
###visualize the .dot file. Need to install graphviz seperately at first 
import graphviz
with open("mytree.dot") as f:
    dot_graph=f.read()
graphviz.Source(dot_graph)


# ## Collecting False Negative rows indexs

# In[ ]:


#https://stackoverflow.com/questions/43193880/how-to-get-row-number-in-dataframe-in-pandas
test=[]
FN=pd.DataFrame({"Actual":y_test, "Predicted":y_pred_test})
FN_indexs=list(FN[(FN["Actual"]==0) & (FN["Predicted"]==1)].index)
FN_2=FN[(FN["Actual"]==0) & (FN["Predicted"]==1)]


# # WordCloud based on essay text of false Nagative features :

# In[ ]:


Essay_Tests=""
for i in FN_indexs:
    Essay_Tests+=" {}".format(X_test.loc[i]['essay'])

from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(background_color="white", max_words=len(Essay_Tests), stopwords=stopwords)
wc.generate(Essay_Tests)
print ("Word Cloud for Duplicate Question pairs")
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()    


# ## Box Plot of 'Price' in FN

# In[ ]:


price=[]
for i in FN_indexs:
    price.append(X_test.loc[i]['price'])
import seaborn as sns
sns.set(style="whitegrid")
ax = sns.barplot(y="price", hue=FN_2["Predicted"], data=X_test)    


# In[ ]:


ax = sns.barplot(y="price", hue=FN_2["Actual"], data=X_test)


# # Distribution plot of 'teacher_number_of_previously_posted_projects' in FN

# In[ ]:


teacher_number_of_previously_posted_projects=[]
for i in FN_indexs:
    teacher_number_of_previously_posted_projects.append(X_test.loc[i]['teacher_number_of_previously_posted_projects'])
import seaborn as sns, numpy as np
ax = sns.distplot(teacher_number_of_previously_posted_projects)    


# ### 2.4.3 Applying Decision Trees on AVG W2V,<font color='red'> SET 3</font>

# In[ ]:


aucTotal_cv=[]
aucTotal_train=[]
HayperParametesDictAUC_cv=dict()
for i in min_sample_split_alpha:
    for j in depth_alpha:
        print("for min sample split =", i,"and max depth = ", j)
        clf = DecisionTreeClassifier(min_samples_split=i ,criterion='gini', max_depth=j, class_weight = 'balanced',random_state=42)
        clf.fit(X3_tr, y_tr)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X3_tr, y_tr)
        # predict the response on the crossvalidation train
        
        probs_cv = sig_clf.predict_proba(X3_cv)
    
        # predict the response on the train
        probs_train = sig_clf.predict_proba(X3_tr)
        
        # keep probabilities for the positive outcome only
        probs_cv = probs_cv[:, 1]
        probs_train = probs_train[:,1]
        
        # calculate AUC
        auc_cv = roc_auc_score(y_cv, probs_cv)
        auc_train = roc_auc_score(y_tr, probs_train)
        # evaluate CV accuracy
        probs_cv_acc=sig_clf.predict(X3_cv)
        acc_cv = accuracy_score(y_cv, probs_cv_acc, normalize=True) * float(100)
        print('\nCV accuracy for alpha = %f is %d%% and auc= %.3f' % (i, acc_cv,auc_cv))
        aucTotal_cv.append(auc_cv)
        aucTotal_train.append(auc_train) 
        HayperParametesDictAUC_cv["{},{}".format(i,j)]=auc_cv


# In[ ]:


#https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
import operator
alpha=max(HayperParametesDictAUC_cv.items(), key=operator.itemgetter(1))[0]
min_samples_split_cv,max_depth_cv = alpha.split(",")
min_samples_split_cv=int(min_samples_split_cv)
max_depth_cv = int(max_depth_cv)
hp_w2vec = alpha


# In[ ]:


clf_tfidf = DecisionTreeClassifier(min_samples_split=min_samples_split_cv ,criterion='gini',class_weight = 'balanced',max_depth=max_depth_cv, random_state=42)
clf_tfidf.fit(X3_tr, y_tr)
sig_clf = CalibratedClassifierCV(clf_tfidf, method="sigmoid")
sig_clf.fit(X3_tr, y_tr)
##########################

probs_test   =  sig_clf.predict_proba(X3_test)
probs_train  = sig_clf.predict_proba(X3_tr)
y_pred_test  = sig_clf.predict(X3_test)
y_pred_train = sig_clf.predict(X3_tr)

# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
probs_train = probs_train[:,1]
# calculate AUC
auc = roc_auc_score(y_test, probs_test)





auc_w2vec=auc
print("Test AUC: ",auc)
#y_pred_tr =  sig_clf.predict(X2_tr)
auc = roc_auc_score(y_tr, probs_train)
print("Train AUC: ",auc)


# In[ ]:


import matplotlib.pyplot as plt
probs_test = sig_clf.predict_proba(X3_test)[:,1]
probs_train = sig_clf.predict_proba(X3_tr)[:,1]
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs_test)
fpr2, tpr2, thresholds2 = roc_curve(y_tr, probs_train)
# plot no skill
plt.plot([0, 1], [0, 1], color='C0')
# plot the roc curve for the model
plt.plot(fpr, tpr, color='C1',label='cv')
plt.plot(fpr2, tpr2, color='g',label='train')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ERROR PLOT")
# show the plot
plt.show()


# ## Results based on the confusion matrix

# In[ ]:




from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred_test)
TPR=np.round(cm[1][1]/(cm[0][1]+cm[1][1])*100,2)
FPR=np.round(cm[0][0]/(cm[0][0]+cm[1][0])*100,2)
Precision= np.round(cm[1][1]/(cm[1][0]+cm[1][1])*100,2)
Recall = TPR
F1_Score= np.round((2*Precision*Recall)/(Recall+Precision),3)
F1_Score_l2=F1_Score


import seaborn as sns; sns.set()
ax = sns.heatmap(cm,annot=True,fmt="d",cmap="YlGnBu")

print("TPR: {}%\nFPR: {}%\nPrecision: {}%\nRecall: {}%\nF1 Score: {}% ".format(TPR,FPR,Precision,Recall,F1_Score))


# ## Collecting False Negative rows indexs

# In[ ]:


#https://stackoverflow.com/questions/43193880/how-to-get-row-number-in-dataframe-in-pandas
test=[]
FN=pd.DataFrame({"Actual":y_test, "Predicted":y_pred_test})
FN_indexs=list(FN[(FN["Actual"]==0) & (FN["Predicted"]==1)].index)
FN_2=FN[(FN["Actual"]==0) & (FN["Predicted"]==1)]


# # WordCloud based on essay text of false Nagative features :

# In[ ]:


Essay_Tests=""
for i in FN_indexs:
    Essay_Tests+=" {}".format(X_test.loc[i]['essay'])

from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(background_color="white", max_words=len(Essay_Tests), stopwords=stopwords)
wc.generate(Essay_Tests)
print ("Word Cloud for Duplicate Question pairs")
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()    


# ## Box Plot of 'Price' in FN

# In[ ]:


price=[]
for i in FN_indexs:
    price.append(X_test.loc[i]['price'])
import seaborn as sns
sns.set(style="whitegrid")
ax = sns.barplot(y="price", hue=FN_2["Predicted"], data=X_test)    


# In[ ]:


ax = sns.barplot(y="price", hue=FN_2["Actual"], data=X_test)


# # Distribution plot of 'teacher_number_of_previously_posted_projects' in FN

# In[ ]:



teacher_number_of_previously_posted_projects=[]
for i in FN_indexs:
    teacher_number_of_previously_posted_projects.append(X_test.loc[i]['teacher_number_of_previously_posted_projects'])
import seaborn as sns, numpy as np
ax = sns.distplot(teacher_number_of_previously_posted_projects)   


# ### 2.4.4 Applying Decision Trees on TFIDF W2V,<font color='red'> SET 4</font>

# In[ ]:


aucTotal_cv=[]
aucTotal_train=[]
HayperParametesDictAUC_cv=dict()
for i in min_sample_split_alpha:
    for j in depth_alpha:
        print("for min sample split =", i,"and max depth = ", j)
        clf = DecisionTreeClassifier(min_samples_split=i ,criterion='gini', max_depth=j, class_weight = 'balanced',random_state=42)
        clf.fit(X4_tr, y_tr)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X4_tr, y_tr)
        # predict the response on the crossvalidation train
        
        probs_cv = sig_clf.predict_proba(X4_cv)
    
        # predict the response on the train
        probs_train = sig_clf.predict_proba(X4_tr)
        
        # keep probabilities for the positive outcome only
        probs_cv = probs_cv[:, 1]
        probs_train = probs_train[:,1]
        
        # calculate AUC
        auc_cv = roc_auc_score(y_cv, probs_cv)
        auc_train = roc_auc_score(y_tr, probs_train)
        # evaluate CV accuracy
        probs_cv_acc=sig_clf.predict(X4_cv)
        acc_cv = accuracy_score(y_cv, probs_cv_acc, normalize=True) * float(100)
        print('\nCV accuracy for alpha = %f is %d%% and auc= %.3f' % (i, acc_cv,auc_cv))
        aucTotal_cv.append(auc_cv)
        aucTotal_train.append(auc_train) 
        HayperParametesDictAUC_cv["{},{}".format(i,j)]=auc_cv


# In[ ]:


#https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
import operator
alpha=max(HayperParametesDictAUC_cv.items(), key=operator.itemgetter(1))[0]
min_samples_split_cv,max_depth_cv = alpha.split(",")
min_samples_split_cv=int(min_samples_split_cv)
max_depth_cv = int(max_depth_cv)
hp_tfidf_w2vec = alpha


# In[ ]:


clf_tfidf = DecisionTreeClassifier(min_samples_split=min_samples_split_cv ,criterion='gini',class_weight = 'balanced',max_depth=max_depth_cv, random_state=42)
clf_tfidf.fit(X4_tr, y_tr)
sig_clf = CalibratedClassifierCV(clf_tfidf, method="sigmoid")
sig_clf.fit(X4_tr, y_tr)
##########################

probs_test   =  sig_clf.predict_proba(X4_test)
probs_train  = sig_clf.predict_proba(X4_tr)
y_pred_test  = sig_clf.predict(X4_test)
y_pred_train = sig_clf.predict(X4_tr)

# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
probs_train = probs_train[:,1]
# calculate AUC
auc = roc_auc_score(y_test, probs_test)





auc_tfidf_w2vec=auc
print("Test AUC: ",auc)
#y_pred_tr =  sig_clf.predict(X2_tr)
auc = roc_auc_score(y_tr, probs_train)
print("Train AUC: ",auc)


# In[ ]:


import matplotlib.pyplot as plt
probs_test = sig_clf.predict_proba(X4_test)[:,1]
probs_train = sig_clf.predict_proba(X4_tr)[:,1]
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs_test)
fpr2, tpr2, thresholds2 = roc_curve(y_tr, probs_train)
# plot no skill
plt.plot([0, 1], [0, 1], color='C0')
# plot the roc curve for the model
plt.plot(fpr, tpr, color='C1',label='cv')
plt.plot(fpr2, tpr2, color='g',label='train')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ERROR PLOT")
# show the plot
plt.show()


# ## Results based on the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred_test)
TPR=np.round(cm[1][1]/(cm[0][1]+cm[1][1])*100,2)
FPR=np.round(cm[0][0]/(cm[0][0]+cm[1][0])*100,2)
Precision= np.round(cm[1][1]/(cm[1][0]+cm[1][1])*100,2)
Recall = TPR
F1_Score= np.round((2*Precision*Recall)/(Recall+Precision),3)
F1_Score_l2=F1_Score


# In[ ]:


import seaborn as sns; sns.set()
ax = sns.heatmap(cm,annot=True,fmt="d",cmap="YlGnBu")

print("TPR: {}%\nFPR: {}%\nPrecision: {}%\nRecall: {}%\nF1 Score: {}% ".format(TPR,FPR,Precision,Recall,F1_Score))


# ## Collecting False Negative rows indexs

# In[ ]:


#https://stackoverflow.com/questions/43193880/how-to-get-row-number-in-dataframe-in-pandas
test=[]
FN=pd.DataFrame({"Actual":y_test, "Predicted":y_pred_test})
FN_indexs=list(FN[(FN["Actual"]==0) & (FN["Predicted"]==1)].index)
FN_2=FN[(FN["Actual"]==0) & (FN["Predicted"]==1)]


# # WordCloud based on essay text of false Nagative features :

# In[ ]:


Essay_Tests=""
for i in FN_indexs:
    Essay_Tests+=" {}".format(X_test.loc[i]['essay'])

from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(background_color="white", max_words=len(Essay_Tests), stopwords=stopwords)
wc.generate(Essay_Tests)
print ("Word Cloud for Duplicate Question pairs")
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()    


# ## Box Plot of 'Price' in FN

# In[ ]:


price=[]
for i in FN_indexs:
    price.append(X_test.loc[i]['price'])
import seaborn as sns
sns.set(style="whitegrid")
ax = sns.barplot(y="price", hue=FN_2["Predicted"], data=X_test)    


# In[ ]:


ax = sns.barplot(y="price", hue=FN_2["Actual"], data=X_test)


# # Distribution plot of 'teacher_number_of_previously_posted_projects' in FN

# In[ ]:


teacher_number_of_previously_posted_projects=[]
for i in FN_indexs:
    teacher_number_of_previously_posted_projects.append(X_test.loc[i]['teacher_number_of_previously_posted_projects'])
import seaborn as sns, numpy as np
ax = sns.distplot(teacher_number_of_previously_posted_projects)    


# <h2>2.5 [Task-2]Getting top 5k features using `feature_importances_`</h2>

# In[ ]:


top_5000_features=np.argsort(clf_tfidf.feature_importances_)
top_5000_features=top_5000_features[:5000]


# In[ ]:


#pick multiple csr columns based on index: https://stackoverflow.com/questions/13352280/slicing-sparse-matrices-in-scipy-which-types-work-best
indices = np.where(top_5000_features)[0]
X2_tr_5000 = X2_tr.tocsr()[:,indices]
X2_cv_5000 = X2_cv.tocsr()[:,indices]
X2_test_5000 = X2_test.tocsr()[:,indices]

#Check the shape of the new Data's

print(X2_tr_5000.shape)
print(X2_cv_5000.shape)
print(X2_test_5000.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
alpha = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000] 

aucTotal_cv=[]
aucTotal_train=[]
HayperParametesDictAUC_cv=dict()
for i in alpha:
        clf = LogisticRegression(C=i)
        clf.fit(X2_tr_5000, y_tr)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X2_tr_5000, y_tr)
        # predict the response on the crossvalidation train
        
    
        # predict the response on the crossvalidation train

        probs_cv = sig_clf.predict_proba(X2_cv_5000)

        # predict the response on the train
        probs_train = sig_clf.predict_proba(X2_tr_5000)

        # keep probabilities for the positive outcome only
        probs_cv = probs_cv[:, 1]
        probs_train = probs_train[:,1]

        # calculate AUC
        auc_cv = roc_auc_score(y_cv, probs_cv)
        auc_train = roc_auc_score(y_tr, probs_train)

        # $$$$$$$$$$
        # evaluate CV accuracy
        probs_cv_acc=sig_clf.predict(X2_cv_5000)
        acc_cv = accuracy_score(y_cv, probs_cv_acc, normalize=True) * float(100)
        print('\nCV accuracy for alpha = %f is %d%% and auc= %.3f' % (i, acc_cv,auc_cv))
        aucTotal_cv.append(auc_cv)
        aucTotal_train.append(auc_train) 
    
    
    
    
    
        ###############$$$$$$$$$$$$$%%%%%%%%%%%%%%%
    

        # evaluate CV accuracy
        #acc_cv = accuracy_score(y_cv, pred_cv, normalize=True) * float(100)
        #print('\nCV accuracy for alpha = %f is %d%% and auc= %.3f' % (i, acc_cv,auc_cv))
        #aucTotal_cv.append(auc_cv)
        #aucTotal_train.append(auc_train) 
       


# In[ ]:


hp_log_5000=alpha[np.argmax(aucTotal_cv)]


# In[ ]:


clf_tfidf = DecisionTreeClassifier(min_samples_split=min_samples_split_cv ,criterion='gini',class_weight = 'balanced',max_depth=max_depth_cv, random_state=42)
clf_tfidf.fit(X4_tr, y_tr)
sig_clf = CalibratedClassifierCV(clf_tfidf, method="sigmoid")
sig_clf.fit(X4_tr, y_tr)
##########################

probs_test   =  sig_clf.predict_proba(X4_test)
probs_train  = sig_clf.predict_proba(X4_tr)
y_pred_test  = sig_clf.predict(X4_test)
y_pred_train = sig_clf.predict(X4_tr)

# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
probs_train = probs_train[:,1]
# calculate AUC
auc = roc_auc_score(y_test, probs_test)





auc_tfidf_w2vec=auc
print("Test AUC: ",auc)
#y_pred_tr =  sig_clf.predict(X2_tr)
auc = roc_auc_score(y_tr, probs_train)
print("Train AUC: ",auc)


# In[ ]:


clf = LogisticRegression(C=alpha[np.argmax(aucTotal_cv)])
clf.fit(X2_tr_5000, y_tr)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(X2_tr_5000, y_tr)



probs_test   =  sig_clf.predict_proba(X2_test_5000)
probs_train  = sig_clf.predict_proba(X2_tr_5000)

# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
probs_train = probs_train[:,1]
# calculate AUC
auc = roc_auc_score(y_test, probs_test)




auc_logistic_5000=auc
print("Test AUC: ",auc)
auc = roc_auc_score(y_tr, probs_train)
print("Train AUC: ",auc)


# In[ ]:


import matplotlib.pyplot as plt
probs_test = sig_clf.predict_proba(X2_test_5000)[:,1]
probs_train = sig_clf.predict_proba(X2_tr_5000)[:,1]
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs_test)
fpr2, tpr2, thresholds2 = roc_curve(y_tr, probs_train)
# plot no skill
plt.plot([0, 1], [0, 1], color='C0')
# plot the roc curve for the model
plt.plot(fpr, tpr, color='C1',label='cv')
plt.plot(fpr2, tpr2, color='g',label='train')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ERROR PLOT")
# show the plot
plt.show()


# ## Results based on the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred_test)
TPR=np.round(cm[1][1]/(cm[0][1]+cm[1][1])*100,2)
FPR=np.round(cm[0][0]/(cm[0][0]+cm[1][0])*100,2)
Precision= np.round(cm[1][1]/(cm[1][0]+cm[1][1])*100,2)
Recall = TPR
F1_Score= np.round((2*Precision*Recall)/(Recall+Precision),3)
F1_Score_l2=F1_Score

import seaborn as sns; sns.set()
ax = sns.heatmap(cm,annot=True,fmt="d",cmap="YlGnBu")

print("TPR: {}%\nFPR: {}%\nPrecision: {}%\nRecall: {}%\nF1 Score: {}% ".format(TPR,FPR,Precision,Recall,F1_Score))


# <h1>3. Conclusion</h1>

# In[ ]:


# Please compare all yfrom prettytable import PrettyTable
from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["Vectorizer", "Model", "Hyper Parameter", "AUC"]
x.add_row(["BOW", "DecisionTreeClassifier", hp_bow, np.round(auc_bow,5)])
x.add_row(["Tfidf","DecisionTreeClassifier", hp_tfidf, np.round(auc_tfidf,5)])
x.add_row(["Word2Vec", "DecisionTreeClassifier", hp_w2vec, np.round(auc_w2vec,5)])
x.add_row(["TfidfWord2Vec","DecisionTreeClassifier", hp_tfidf_w2vec, np.round(auc_tfidf_w2vec,5)])
x.add_row(["Tfidf","LogisticRegression", hp_log_5000, np.round(auc_logistic_5000,5)])


# In[ ]:


print(x)


# In[ ]:




