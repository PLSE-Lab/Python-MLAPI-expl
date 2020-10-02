#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes: DonorsChoose Dataset
# 
# ### A. DATA INFORMATION
# ### B. OBJECTIVE
# ## 1. READING THE DATASET
# - **1.1 Removing Nan**
# 
# ## 2. PREPROCESSING 
# - **2.1 Preprocessing: Project Subject Categories**
# - **2.2 Preprocessing: Project Subject Sub Categories**
# - **2.3 Preprocessing: Project Grade**
# 
# ## 3. TEXT PROCESSING
# - **3.1 Text Preprocessing: Essays**
# - **3.2 Text Preprocessing: Title**
# 
# ## 4. SAMPLING
# - **4.1 Taking Sample from the complete dataset.**
# - **4.2 Splitting the dataset into Train, CV and Test datasets. (60:20:20)**
#     - 4.2.1: Splitting the FULL DATASET into Training and Test sets. (80:20)
#     - 4.2.2: Splitting the TRAINING Dataset further into Training and CV sets. (Not required  as we are using GridSearch)
# - **4.3 Details of our Training, CV and Test datasets.**
# 
# ## 5. UPSAMPLING THE MINORITY CLASS WITH REPLACEMENT STRATEGY (Not required  as we are using GridSearch)
# 
# ## 6. PREPARING DATA FOR MODELS
# - **6.1: VECTORIZING CATEGORICAL DATA**
#     - 6.1.1: Vectorizing Categorical data: Clean Subject Categories.
#     - 6.1.2: Vectorizing Categorical data: Clean Subject Sub-Categories.
#     - 6.1.3: Vectorizing Categorical data: School State.
#     - 6.1.4: Vectorizing Categorical data: Teacher Prefix.
#     - 6.1.5: Vectorizing Categorical data: Project Grade
#         
# - **6.2: VECTORIZING NUMERICAL DATA**
#     - 6.2.1: Standarizing Numerical data: Price
#     - 6.2.2: Standarizing Numerical data: Teacher's Previous Projects
#         
# - **6.3: VECTORIZING TEXT DATA**
#     - **6.3.1: BOW**
#         - 6.3.1.1: BOW: Essays (Train, CV, Test)
#         - 6.3.1.2: BOW: Title (Train, CV, Test)
#             
#     - **6.3.2: TF-IDF**
#         - 6.3.2.1: TF-IDF: Essays (Train, CV, Test)
#         - 6.3.2.2: TF-IDF: Title (Train, CV, Test)
#                 
# ## 7. MERGING FEATURES
# - 7.1: Merging all ONE HOT features.
# - 7.2: SET 1: Merging All ONE HOT with BOW (Title and Essay) features.
# - 7.3: SET 2: Merging All ONE HOT with TF-IDF (Title and Essay) features.
#     
# ## 8. NAIVE BAYES
# - **8.1: SET 1 Applying Multinomial Naive Bayes on BOW.**
#     - 8.1.1: SET 1 Hyper parameter tuning to find the best Alpha.
#     - 8.1.2: SET 1 TESTING the performance of the model on test data, plotting ROC Curves.
#     - 8.1.3: SET 1 Confusion Matrix
#         - 8.1.3.1: SET 1 Confusion Matrix: Train
#         - 8.1.3.2: SET 1 Confusion Matrix: Test
#             
# - **8.2: SET 2 Applying Multinomial Naive Bayes on TF-IDF.**
#     - 8.2.1: SET 2 Hyper parameter tuning to find the best Alpha.
#     - 8.2.2: SET 2 TESTING the performance of the model on test data, plotting ROC Curves.
#     - 8.2.3: SET 2 Confusion Matrix
#         - 8.2.3.1: SET 2 Confusion Matrix: Train
#         - 8.2.3.2: SET 2 Confusion Matrix: Test
#     
# ## 9. Top 10 Features of Set 1
# - **9.1: Negative Class
# - **9.2: Positive Class
# 
# ## 10. Top 10 Features of Set 2
# - **10.1: Negative Class
# - **10.2: Positive Class
# 
# ## 11. CONCLUSION 

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# ## A. DATA INFORMATION 
# ### <br>About the DonorsChoose Data Set
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

# # B. OBJECTIVE
# The primary objective is to implement the k-Nearest Neighbor Algo on the DonorChoose Dataset and measure the accuracy on the Test dataset.

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

# from plotly import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter
import os
print(os.listdir("../input"))
print("DONE ---------------------------------------")


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 1. READING DATA

# In[ ]:


# project_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/train_data.csv') 
# resource_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/resources.csv')
project_data = pd.read_csv('/kaggle/input/donorschoosedataset/train_data.csv') 
resource_data = pd.read_csv('/kaggle/input/donorschoosedataset/resources.csv')
# project_data = pd.read_csv('../train_data.csv') 
# resource_data = pd.read_csv('../resources.csv')

print("Done")


# In[ ]:


project_data.shape


# In[ ]:


project_data.columns.values


# ## -> 1.1: REMOVING NaN:<br>
# **As it is clearly metioned in the dataset details that TEACHER_PREFIX has NaN values, we need to handle this at the very beginning to avoid any problems in our future analysis.**

# #### Checking total number of enteries with NaN values

# In[ ]:


prefixlist=project_data['teacher_prefix'].values
prefixlist=list(prefixlist)
cleanedPrefixList = [x for x in project_data['teacher_prefix'] if x != float('nan')] ## Cleaning the NULL Values in the list -> https://stackoverflow.com/a/50297200/4433839

len(cleanedPrefixList)
# print(len(prefixlist))


# **Observation:** 3 Rows had NaN values and they are not considered, thus the number of rows reduced from 109248 to 109245. <br>
# **Action:** We can safely delete these, as 3 is very very small and its deletion wont impact as the original dataset is very large. <br>
# Step1: Convert all the empty strings with Nan // Not required as its NaN not empty string <br> 
# Step2: Drop rows having NaN values

# In[ ]:


## Converting to Nan and Droping -> https://stackoverflow.com/a/29314880/4433839

# df[df['B'].str.strip().astype(bool)] // for deleting EMPTY STRINGS.
project_data.dropna(subset=['teacher_prefix'], inplace=True)
project_data.shape


# **Conclusion:** Now the number of rows reduced from 109248 to 109245 in project_data.

# In[ ]:


project_data['teacher_prefix'].head(10)


# In[ ]:


print("Number of data points in train data", resource_data.shape)
print(resource_data.columns.values)
resource_data.head(2)


# # 2: PRE-PROCESSING

# ## -> 2.1: Preprocessing: Project Subject Categories

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
print("done")


# ## -> 2.2: Preprocessing: Project Subject Sub Categories

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
print("done")


# ## -> 2.3: Preprocessing: Project Grade Category

# In[ ]:


# this code removes " " and "-". ie Grades 3-5 -> grage3to5
#  remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python
clean_grades=[]
for project_grade in project_data['project_grade_category'].values:
    project_grade=str(project_grade).lower().strip().replace(' ','').replace('-','to')
    
    clean_grades.append(project_grade.strip())

project_data['clean_project_grade_category']=clean_grades
project_data.drop(['project_grade_category'],axis=1,inplace=True)

my_counter = Counter()
for word in project_data['clean_project_grade_category'].values:
    my_counter.update(word.split())
    
grade_dict = dict(my_counter)
sorted_project_grade_cat_dict = dict(sorted(grade_dict.items(), key=lambda kv: kv[1]))
print("done")


# # 3. TEXT PROCESSING

# #### Merging Project Essays 1 2 3 4 into Essays

# In[ ]:


# merge two column text dataframe: 
project_data["essay"] = project_data["project_essay_1"].map(str) +                        project_data["project_essay_2"].map(str) +                         project_data["project_essay_3"].map(str) +                         project_data["project_essay_4"].map(str)


# In[ ]:


project_data.head(2)


# In[ ]:


#### Text PreProcessing Functions


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


# ## -> 3.1: Text Preprocessing: Essays

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


project_data.columns


# In[ ]:


## new column added as PreProcessed_Essays and older unProcessed essays column is deleted
project_data['preprocessed_essays'] = preprocessed_essays
project_data.drop(['essay'], axis=1, inplace=True)


# In[ ]:


project_data.columns


# In[ ]:


# after preprocesing
preprocessed_essays[20000]


# ## -> 3.2: Text Preprocessing: Title

# In[ ]:


# similarly you can preprocess the titles also
# similarly you can preprocess the titles also
# similarly you can preprocess the titles also
from tqdm import tqdm
preprocessed_titles = []
# tqdm is for printing the status bar
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


#https://stackoverflow.com/questions/26666919/add-column-in-dataframe-from-list/3849072
project_data['preprocessed_titles'] = preprocessed_titles
# project_data.drop(['project_title'], axis=1, inplace=True)


# In[ ]:


project_data.columns


# In[ ]:


project_data.shape


# #### -> Merging Price with Project Data

# In[ ]:


price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
price_data.head(2)


# In[ ]:


project_data = pd.merge(project_data, price_data, on='id', how='left')


# In[ ]:


project_data.shape


# In[ ]:


project_bkp=project_data.copy()


# In[ ]:


project_bkp.shape


# # 4. SAMPLING
# ## -> 4.1: Taking Sample from the complete dataset
# ## NOTE: A sample of 100000 Datapoints is taken due to lack computational resource.

# In[ ]:


## taking random samples of 100k datapoints
project_data = project_bkp.sample(n = 100000) 
# resource_data = pd.read_csv('../resources.csv')

project_data.shape

# y_value_counts = row1['project_is_approved'].value_counts()
y_value_counts = project_data['project_is_approved'].value_counts()
print("Number of projects thar are approved for funding:     ", y_value_counts[1]," -> ",round(y_value_counts[1]/(y_value_counts[1]+y_value_counts[0])*100,2),"%")
print("Number of projects thar are not approved for funding: ", y_value_counts[0]," -> ",round(y_value_counts[0]/(y_value_counts[1]+y_value_counts[0])*100,2),"%")


# **Observation:**
# 1. Dataset is highly **IMBALANCED**.
# 1. Approved Class (1) is the Majority class. And the Majority class portion in our sampled dataset: ~85%
# 1. Unapproved class (0) is the Minority class. And the Minority class portion in our sampled dataset: ~15%

# ## -> 4.2: Splitting the dataset into Train, CV and Test datasets. (60:20:20)

# According to Andrew Ng, in the Coursera MOOC on Introduction to Machine Learning, the general rule of thumb is to partition the data set into the ratio of ***3:1:1 (60:20:20)*** for training, validation and testing respectively.

# ###    -> -> 4.2.1: Splitting the FULL DATASET into Training and Test sets. (80:20)

# In[ ]:


# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(
#     project_data.drop('project_is_approved', axis=1),
#     project_data['project_is_approved'].values,
#     test_size=0.3,
#     random_state=42,
#     stratify=project_data[['project_is_approved']])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    project_data,
    project_data['project_is_approved'],
    test_size=0.2,
    random_state=42,
    stratify=project_data[['project_is_approved']])

print("x_train: ",x_train.shape)
print("x_test : ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test : ",y_test.shape)


# ### -> -> 4.2.2: Splitting the TRAINING Dataset further into Training and CV sets.

# In[ ]:


# x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train)

# print("x_train: ",x_train.shape)
# print("y_train: ",y_train.shape)
# print("x_cv   : ",x_cv.shape)
# print("x_cv   : ",y_cv.shape)
# print("x_test : ",x_test.shape)
# print("y_test : ",y_test.shape)


# ### -> 4.3: Details of our Training, CV and Test datasets.

# In[ ]:


# y_value_counts = row1['project_is_approved'].value_counts()
print("X_TRAIN-------------------------")
x_train_y_value_counts = x_train['project_is_approved'].value_counts()
print("Number of projects that are approved for funding    ", x_train_y_value_counts[1]," -> ",round(x_train_y_value_counts[1]/(x_train_y_value_counts[1]+x_train_y_value_counts[0])*100,2),"%")
print("Number of projects that are not approved for funding ",x_train_y_value_counts[0]," -> ",round(x_train_y_value_counts[0]/(x_train_y_value_counts[1]+x_train_y_value_counts[0])*100,2),"%")
print("\n")
# y_value_counts = row1['project_is_approved'].value_counts()
print("X_TEST--------------------------")
x_test_y_value_counts = x_test['project_is_approved'].value_counts()
print("Number of projects that are approved for funding    ", x_test_y_value_counts[1]," -> ",round(x_test_y_value_counts[1]/(x_test_y_value_counts[1]+x_test_y_value_counts[0])*100,2),"%")
print("Number of projects that are not approved for funding ",x_test_y_value_counts[0]," -> ",round(x_test_y_value_counts[0]/(x_test_y_value_counts[1]+x_test_y_value_counts[0])*100,2),"%")
print("\n")
# y_value_counts = row1['project_is_approved'].value_counts()
# print("X_CV----------------------------")
# x_cv_y_value_counts = x_cv['project_is_approved'].value_counts()
# print("Number of projects that are approved for funding    ", x_cv_y_value_counts[1]," -> ",round(x_cv_y_value_counts[1]/(x_cv_y_value_counts[1]+x_cv_y_value_counts[0])*100),2,"%")
# print("Number of projects that are not approved for funding ",x_cv_y_value_counts[0]," -> ",round(x_cv_y_value_counts[0]/(x_cv_y_value_counts[1]+x_cv_y_value_counts[0])*100),2,"%")
# print("\n")


# **Observation:**
# 1. The proportion of Majority class of 85% and Minority class of 15% is maintained in Training, CV and Testing dataset.

# **Conlusion**
# 1. **UPSAMPLING** needs to be done on the Minority class to avoid problems related to Imbalanced dataset.
# 1. Upsampling will be done by _**"Resample with replacement strategy"**_

# # 5. UPSAMPLING THE MINORITY CLASS WITH REPLACEMENT STRATEGY

# Reference: https://elitedatascience.com/imbalanced-classes

# In[ ]:


# from sklearn.utils import resample

# ## splitting x_train in their respective classes
# x_train_majority=x_train[x_train.project_is_approved==1]
# x_train_minority=x_train[x_train.project_is_approved==0]

# print("No. of points in the Training Dataset : ", x_train.shape)
# print("No. of points in the majority class 1 : ",len(x_train_majority))
# print("No. of points in the minority class 0 : ",len(x_train_minority))

# print(x_train_majority.shape)
# print(x_train_minority.shape)

# ## Resampling with replacement
# x_train_minority_upsampled=resample(
#     x_train_minority,
#     replace=True,

#     n_samples=len(x_train_majority),
#     random_state=123)

# print("Resampled Minority class details")
# print("Type:  ",type(x_train_minority_upsampled))
# print("Shape: ",x_train_minority_upsampled.shape)
# print("\n")
# ## Concatinating our Upsampled Minority class with the existing Majority class
# x_train_upsampled=pd.concat([x_train_majority,x_train_minority_upsampled])

# print("Upsampled Training data")
# print("Total number of Class labels")
# print(x_train_upsampled.project_is_approved.value_counts())
# print("\n")
# print("Old Training IMBALANCED Dataset Shape         : ", x_train.shape)
# print("New Training BALANCED Upsampled Dataset Shape : ",x_train_upsampled.shape)

# x_train_upsampled.to_csv ('x_train_upsampled_csv.csv',index=False)


# ## UPSAMPLED Balanced training dataset is now of 10,262 Datapoints.
# ## CV and Test is 8,000 data points. Therefore total dataset = 18,262 Datapoints.

# **Conclusion:**
# 1. Resampling is performed on the Training data.
# 1. Training data in now **BALANCED**.

# In[ ]:


# yy_train=x_train_upsampled['project_is_approved'].copy()


# In[ ]:


# yy_train.shape


# In[ ]:


# x_train_upsampled.shape


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


# x_cv.shape


# # 6. PREPARING DATA FOR MODELS

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
#        - quantity : numerical (optinal)
#        - teacher_number_of_previously_posted_projects : numerical
#        - price : numerical

# # -> 6.1: VECTORIZING CATEGORICAL DATA

# - https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/handling-categorical-and-numerical-features/

# ## ->-> 6.1.1: Vectorizing Categorical data: Clean Subject Categories

# In[ ]:


# we use count vectorizer to convert the values into one 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_sub = CountVectorizer(vocabulary=list(sorted_cat_dict.keys()), lowercase=False, binary=True)

vectorizer_sub.fit(x_train['clean_categories'].values)

x_train_categories_one_hot = vectorizer_sub.transform(x_train['clean_categories'].values)
# x_cv_categories_one_hot    = vectorizer_sub.transform(x_cv['clean_categories'].values)
x_test_categories_one_hot  = vectorizer_sub.transform(x_test['clean_categories'].values)


print(vectorizer_sub.get_feature_names())

print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_categories_one_hot.shape)
# print("Shape of matrix after one hot encoding -> categories: x_cv   : ",x_cv_categories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_categories_one_hot.shape)


# ## ->-> 6.1.2: Vectorizing Categorical data: Clean Subject Sub-Categories

# In[ ]:


# we use count vectorizer to convert the values into one 
vectorizer_sub_sub = CountVectorizer(vocabulary=list(sorted_sub_cat_dict.keys()), lowercase=False, binary=True)

vectorizer_sub_sub.fit(x_train['clean_subcategories'].values)

x_train_sub_categories_one_hot = vectorizer_sub_sub.transform(x_train['clean_subcategories'].values)
# x_cv_sub_categories_one_hot    = vectorizer_sub_sub.transform(x_cv['clean_subcategories'].values)
x_test_sub_categories_one_hot  = vectorizer_sub_sub.transform(x_test['clean_subcategories'].values)

print(vectorizer_sub_sub.get_feature_names())

print("Shape of matrix after one hot encoding -> sub_categories: x_train: ",x_train_sub_categories_one_hot.shape)
# print("Shape of matrix after one hot encoding -> sub_categories: x_cv   : ",x_cv_sub_categories_one_hot.shape)
print("Shape of matrix after one hot encoding -> sub_categories: x_test : ",x_test_sub_categories_one_hot.shape)


# ## ->-> 6.1.3 Vectorizing Categorical data: School State

# In[ ]:


my_counter = Counter()
for state in project_data['school_state'].values:
    my_counter.update(state.split())


# In[ ]:


school_state_cat_dict = dict(my_counter)
sorted_school_state_cat_dict = dict(sorted(school_state_cat_dict.items(), key=lambda kv: kv[1]))


# In[ ]:


from scipy import sparse ## Exporting Sparse Matrix to NPZ File -> https://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
statelist=list(project_data['school_state'].values)
vectorizer_state = CountVectorizer(vocabulary=set(statelist), lowercase=False, binary=True)

vectorizer_state.fit(x_train['school_state'])

x_train_school_state_one_hot = vectorizer_state.transform(x_train['school_state'].values)
# x_cv_school_state_one_hot    = vectorizer_state.transform(x_cv['school_state'].values)
x_test_school_state_one_hot  = vectorizer_state.transform(x_test['school_state'].values)

print(vectorizer_state.get_feature_names())

print("Shape of matrix after one hot encoding -> school_state: x_train: ",x_train_school_state_one_hot.shape)
# print("Shape of matrix after one hot encoding -> school_state: x_cv   : ",x_cv_school_state_one_hot.shape)
print("Shape of matrix after one hot encoding -> school_state: x_test : ",x_test_school_state_one_hot.shape)
# school_one_hot = vectorizer.transform(statelist)
# print("Shape of matrix after one hot encodig ",school_one_hot.shape)
# print(type(school_one_hot))
# sparse.save_npz("school_one_hot_export.npz", school_one_hot) 
# print(school_one_hot.toarray())


# ## ->-> 6.1.4 Vectorizing Categorical data: Teacher Prefix

# **Teacher Prefix has NAN values, that needs to be cleaned.
# Ref: https://stackoverflow.com/a/50297200/4433839**

# In[ ]:


# prefixlist=project_data['teacher_prefix'].values
# prefixlist=list(prefixlist)
# cleanedPrefixList = [x for x in prefixlist if x == x] ## Cleaning the NULL Values in the list -> https://stackoverflow.com/a/50297200/4433839

# ## preprocessing the prefix to remove the SPACES,- else the vectors will be just 0's. Try adding - and see
# prefix_nospace_list = []
# for i in cleanedPrefixList:
#     temp = ""
#     i = i.replace('.','') # we are placeing all the '.'(dot) with ''(empty) ex:"Mr."=>"Mr"
#     temp +=i.strip()+" "#" abc ".strip() will return "abc", remove the trailing spaces
#     prefix_nospace_list.append(temp.strip())

# cleanedPrefixList=prefix_nospace_list

# vectorizer = CountVectorizer(vocabulary=set(cleanedPrefixList), lowercase=False, binary=True)
# vectorizer.fit(cleanedPrefixList)
# print(vectorizer.get_feature_names())
# prefix_one_hot = vectorizer.transform(cleanedPrefixList)
# print("Shape of matrix after one hot encodig ",prefix_one_hot.shape)
# prefix_one_hot_ar=prefix_one_hot.todense()

# ##code to export to csv -> https://stackoverflow.com/a/54637996/4433839
# # prefixcsv=pd.DataFrame(prefix_one_hot.toarray())
# # prefixcsv.to_csv('prefix.csv', index=None,header=None)


# **Query 1.1: PreProcessing Teacher Prefix Done <br>
# Action Taken: Removed '.' from the prefixes and converted to lower case**

# In[ ]:


my_counter = Counter()
for teacher_prefix in project_data['teacher_prefix'].values:
    teacher_prefix = str(teacher_prefix).lower().replace('.','').strip()
    
    my_counter.update(teacher_prefix.split())
teacher_prefix_cat_dict = dict(my_counter)
sorted_teacher_prefix_cat_dict = dict(sorted(teacher_prefix_cat_dict.items(), key=lambda kv: kv[1]))


# In[ ]:


sorted_teacher_prefix_cat_dict.keys()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer_prefix = CountVectorizer(vocabulary=list(sorted_teacher_prefix_cat_dict.keys()), lowercase=False, binary=True)

vectorizer_prefix.fit(x_train['teacher_prefix'].values)

x_train_prefix_one_hot = vectorizer_prefix.transform(x_train['teacher_prefix'].values)
# x_cv_prefix_one_hot    = vectorizer_prefix.transform(x_cv['teacher_prefix'].values)
x_test_prefix_one_hot  = vectorizer_prefix.transform(x_test['teacher_prefix'].values)

print(vectorizer_prefix.get_feature_names())

print("Shape of matrix after one hot encoding -> prefix: x_train: ",x_train_prefix_one_hot.shape)
# print("Shape of matrix after one hot encoding -> prefix: x_cv   : ",x_cv_prefix_one_hot.shape)
print("Shape of matrix after one hot encoding -> prefix: x_test : ",x_test_prefix_one_hot.shape)


# **Observation:** 
# 1. 3 Rows had NaN values and they are not considered, thus the number of rows reduced from 109248 to 109245.

# ## ->-> 6.1.5 Vectorizing Categorical data: Project Grade

# **Query 1.2: PreProcessing Project Grade Done <br>
# Action Taken: Removed ' ' and '-' from the grades and converted to lower case**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer_grade = CountVectorizer(vocabulary=list(sorted_project_grade_cat_dict.keys()), lowercase=False, binary=True)

vectorizer_grade.fit(x_train['clean_project_grade_category'].values)

x_train_grade_category_one_hot = vectorizer_grade.transform(x_train['clean_project_grade_category'].values)
# x_cv_grade_category_one_hot    = vectorizer_grade.transform(x_cv['clean_project_grade_category'].values)
x_test_grade_category_one_hot  = vectorizer_grade.transform(x_test['clean_project_grade_category'].values)

print(vectorizer_grade.get_feature_names())

print("Shape of matrix after one hot encoding -> project_grade: x_train : ",x_train_grade_category_one_hot.shape)
# print("Shape of matrix after one hot encoding -> project_grade: x_cv    : ",x_cv_grade_category_one_hot.shape)
print("Shape of matrix after one hot encoding -> project_grade: x_test  : ",x_test_grade_category_one_hot.shape)


# In[ ]:


type(x_train_grade_category_one_hot)


# In[ ]:


x_train_grade_category_one_hot.toarray()


# # -> 6.2: VECTORIZING NUMERICAL DATA

# ## ->-> 6.2.1: Normalizing Numerical data: Price

# In[ ]:


# check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# price_standardized = standardScalar.fit(project_data['price'].values)
# this will rise the error
# ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# Reshape your data either using array.reshape(-1, 1)
# transformer = Normalizer().fit(X)
normalizer = Normalizer()

normalizer.fit(x_train['price'].values.reshape(1,-1)) # finding the mean and standard deviation of this data
# print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now normalize the data

x_train_price_normalized = normalizer.transform(x_train['price'].values.reshape(1, -1)).reshape(-1,1)
x_test_price_normalized  = normalizer.transform(x_test['price'].values.reshape(1, -1)).reshape(-1,1)
# x_cv_price_normalized    = normalizer.transform(x_cv['price'].values.reshape(1, -1)).reshape(-1,1)


print("Shape of matrix after normalization -> Teacher Prefix: x_train: ",x_train_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Teacher Prefix: x_cv   : ",x_cv_prefix_one_hot.shape)
print("Shape of matrix after normalization -> Teacher Prefix: x_test : ",x_test_prefix_one_hot.shape)


# In[ ]:


type(x_train_price_normalized)


# In[ ]:


x_train_price_normalized


# In[ ]:


# # check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# # standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# # from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer

# # price_standardized = standardScalar.fit(project_data['price'].values)
# # this will rise the error
# # ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# # Reshape your data either using array.reshape(-1, 1)
# # transformer = Normalizer().fit(X)
# normalizer = Normalizer()

# normalizer.fit(x_train_upsampled['price'].values.reshape(-1,1)) # finding the mean and standard deviation of this data
# # print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# # Now normalize the data

# x_train_price_normalized = normalizer.transform(x_train_upsampled['price'].values.reshape(-1, 1))
# x_test_price_normalized  = normalizer.transform(x_test['price'].values.reshape(-1, 1))
# x_cv_price_normalized    = normalizer.transform(x_cv['price'].values.reshape(-1, 1))


# print("Shape of matrix after normalization -> Teacher Prefix: x_train: ",x_train_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Teacher Prefix: x_cv   : ",x_cv_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Teacher Prefix: x_test : ",x_test_prefix_one_hot.shape)


# ## ->-> 6.2.2: Normalizing Numerical data: Teacher's Previous Projects

# In[ ]:


# check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# price_standardized = standardScalar.fit(project_data['price'].values)
# this will rise the error
# ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# Reshape your data either using array.reshape(-1, 1)

teacher_previous_proj_normalizer = Normalizer()
# normalizer = Normalizer()

teacher_previous_proj_normalizer.fit(x_train['teacher_number_of_previously_posted_projects'].values.reshape(1,-1)) # finding the mean and standard deviation of this data
# print(f"Mean : {teacher_previous_proj_scalar.mean_[0]}, Standard deviation : {np.sqrt(teacher_previous_proj_scalar.var_[0])}")

# Now standardize the data with above maen and variance.
x_train_teacher_previous_proj_normalized = teacher_previous_proj_normalizer.transform(x_train['teacher_number_of_previously_posted_projects'].values.reshape(1,- 1)).reshape(-1,1)
x_test_teacher_previous_proj_normalized  = teacher_previous_proj_normalizer.transform(x_test['teacher_number_of_previously_posted_projects'].values.reshape(1,- 1)).reshape(-1,1)
# x_cv_teacher_previous_proj_normalized    = teacher_previous_proj_normalizer.transform(x_cv['teacher_number_of_previously_posted_projects'].values.reshape(1,- 1)).reshape(-1,1)

print("Shape of matrix after normalization -> Teachers Previous Projects: x_train:  ",x_train_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Teachers Previous Projects: x_cv   :  ",x_cv_prefix_one_hot.shape)
print("Shape of matrix after normalization -> Teachers Previous Projects: x_test :  ",x_test_prefix_one_hot.shape)


# # -> 6.3: VECTORIZING TEXT DATA

# # ->-> 6.3.1: BOW

# ## ->->-> 6.3.1.1: BOW: Essays (Train, CV, Test)

# In[ ]:


# We are considering only the words which appeared in at least 10 documents(rows or projects).
vectorizer_essay_bow = CountVectorizer(min_df=10)

vectorizer_essay_bow.fit(x_train['preprocessed_essays'])

x_train_essays_bow = vectorizer_essay_bow.transform(x_train['preprocessed_essays'])
# x_cv_essays_bow    = vectorizer_essay_bow.transform(x_cv['preprocessed_essays'])
x_test_essays_bow  = vectorizer_essay_bow.transform(x_test['preprocessed_essays'])

print("Shape of matrix after BOW -> Essays: x_train: ",x_train_essays_bow.shape)
# print("Shape of matrix after BOW -> Essays: x_cv   : ",x_cv_essays_bow.shape)
print("Shape of matrix after BOW -> Essays: x_test : ",x_test_essays_bow.shape)


# ## ->->-> 6.3.1.2: BOW: Title (Train, CV, Test)

# In[ ]:


vectorizer_title_bow = CountVectorizer(min_df=10)

vectorizer_title_bow.fit(x_train['preprocessed_titles'])

x_train_titles_bow = vectorizer_title_bow.transform(x_train['preprocessed_titles'])
# x_cv_titles_bow    = vectorizer_title_bow.transform(x_cv['preprocessed_titles'])
x_test_titles_bow  = vectorizer_title_bow.transform(x_test['preprocessed_titles'])

print("Shape of matrix after BOW -> Title: x_train: ",x_train_titles_bow.shape)
# print("Shape of matrix after BOW -> Title: x_cv   : ",x_cv_titles_bow.shape)
print("Shape of matrix after BOW -> Title: x_test : ",x_test_titles_bow.shape)


# # ->-> 6.3.2: TF-IDF

# ## ->->-> 6.3.2.1: TF-IDF: Essays (Train, CV, Test)

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_essay_tfidf = TfidfVectorizer(min_df=10)

vectorizer_essay_tfidf.fit(x_train['preprocessed_essays'])

x_train_essays_tfidf = vectorizer_essay_tfidf.transform(x_train['preprocessed_essays'])
# x_cv_essays_tfidf    = vectorizer_essay_tfidf.transform(x_cv['preprocessed_essays'])
x_test_essays_tfidf  = vectorizer_essay_tfidf.transform(x_test['preprocessed_essays'])

print("Shape of matrix after TF-IDF -> Essay: x_train: ",x_train_essays_tfidf.shape)
# print("Shape of matrix after TF-IDF -> Essay: x_cv   : ",x_cv_essays_tfidf.shape)
print("Shape of matrix after TF-IDF -> Essay: x_test : ",x_test_essays_tfidf.shape)


# ## ->->-> 6.3.2.2: TF-IDF: Title (Train, CV, Test)

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_title_tfidf = TfidfVectorizer(min_df=10)

vectorizer_title_tfidf.fit(x_train['preprocessed_titles'])

x_train_titles_tfidf = vectorizer_title_tfidf.transform(x_train['preprocessed_titles'])
# x_cv_titles_tfidf    = vectorizer_title_tfidf.transform(x_cv['preprocessed_titles'])
x_test_titles_tfidf  = vectorizer_title_tfidf.transform(x_test['preprocessed_titles'])

print("Shape of matrix after TF-IDF -> Title: x_train: ",x_train_titles_tfidf.shape)
# print("Shape of matrix after TF-IDF -> Title: x_cv   : ",x_cv_titles_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_test : ",x_test_titles_tfidf.shape)

# Code for testing and checking the generated vectors
# v1 = vectorizer.transform([preprocessed_titles[0]]).toarray()[0]
# text_title_tfidf=pd.DataFrame(v1)
# text_title_tfidf.to_csv('text_title_tfidf.csv', index=None,header=None)


# In[ ]:


# print(STOP)


# # 7. MERGING FEATURES

# ## -> 7.1: Merging all ONE HOT features

# In[ ]:


from scipy.sparse import hstack

x_train_onehot = hstack((x_train_categories_one_hot, x_train_sub_categories_one_hot, x_train_school_state_one_hot, x_train_prefix_one_hot, x_train_grade_category_one_hot, x_train_price_normalized, x_train_teacher_previous_proj_normalized))
# x_cv_onehot    = hstack((x_cv_categories_one_hot, x_cv_sub_categories_one_hot, x_cv_school_state_one_hot, x_cv_prefix_one_hot, x_cv_grade_category_one_hot,x_cv_price_normalized, x_cv_teacher_previous_proj_normalized ))
x_test_onehot  = hstack((x_test_categories_one_hot, x_test_sub_categories_one_hot, x_test_school_state_one_hot, x_test_prefix_one_hot, x_test_grade_category_one_hot, x_test_price_normalized, x_test_teacher_previous_proj_normalized))

print("Type -> One Hot -> x_train: ",type(x_train_onehot))
print("Type -> One Hot -> x_test : ",type(x_test_onehot))
# print("Type -> One Hot -> x_cv        : ",type(x_cv_onehot))
print("\n")
print("Shape -> One Hot -> x_train: ",x_train_onehot.shape)
print("Shape -> One Hot -> x_test : ",x_test_onehot.shape)
# print("Shape -> One Hot -> x_cv         : ",x_cv_onehot.shape)


# In[ ]:


x_train_onehot.shape


# ## -> 7.2: SET 1:  Merging All ONE HOT with BOW (Title and Essay) features

# In[ ]:


x_train_onehot_bow = hstack((x_train_onehot,x_train_titles_bow,x_train_essays_bow)).tocsr()### Merging all ONE HOT features
# x_cv_onehot_bow    = hstack((x_cv_onehot, x_cv_titles_bow, x_cv_essays_bow)).tocsr()### Merging all ONE HOT features
x_test_onehot_bow  = hstack((x_test_onehot, x_test_titles_bow, x_test_essays_bow)).tocsr()### Merging all ONE HOT features
print("Type -> One Hot BOW -> x_train_cv_test: ",type(x_train_onehot_bow))
# print("Type -> One Hot BOW -> cv             : ",type(x_cv_onehot_bow))
print("Type -> One Hot BOW -> x_test         : ",type(x_test_onehot_bow))
print("\n")
print("Shape -> One Hot BOW -> x_train_cv_test: ",x_train_onehot_bow.shape)
# print("Shape -> One Hot BOW -> cv             : ",x_cv_onehot_bow.shape)
print("Shape -> One Hot BOW -> x_test         : ",x_test_onehot_bow.shape)


# ## -> 7.3: SET 2:  Merging All ONE HOT with TF-IDF (Title and Essay) features

# In[ ]:


x_train_onehot_tfidf = hstack((x_train_onehot,x_train_titles_tfidf, x_train_essays_tfidf)).tocsr()
# x_cv_onehot_tfidf    = hstack((x_cv_onehot,x_cv_titles_tfidf, x_cv_essays_tfidf)).tocsr()
x_test_onehot_tfidf  = hstack((x_test_onehot,x_test_titles_tfidf, x_test_essays_tfidf)).tocsr()
print("Type -> One Hot TFIDF -> x_train_cv_test: ",type(x_train_onehot_tfidf))
# print("Type -> One Hot TFIDF -> cv             : ",type(x_cv_onehot_tfidf))
print("Type -> One Hot TFIDF -> x_test         : ",type(x_test_onehot_tfidf))
print("\n")
print("Shape -> One Hot TFIDF -> x_train_cv_test: ",x_train_onehot_tfidf.shape)
# print("Shape -> One Hot TFIDF -> cv             : ",x_cv_onehot_tfidf.shape)
print("Shape -> One Hot TFIDF -> x_test         : ",x_test_onehot_tfidf.shape)### SET 1:  Merging ONE HOT with BOW (Title and Essay) features


# In[ ]:


# yy_train.shape


# In[ ]:


# stop


# # 8. NAIVE BAYES
# 

# # -> 8.1:<font color='red'> SET 1</font>  Applying Naive Bayes on BOW.

# ## ->-> 8.1.1: <font color='red'> SET 1</font> Hyper parameter tuning to find best 'alpha'

# In[ ]:


# def batch_predict(clf, data):
#     # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
#     # not the predicted outputs

#     y_data_pred = []
#     tr_loop = data.shape[0] - data.shape[0]%1000
#     # consider you X_tr shape is 49041, then your tr_loop will be 49041 - 49041%1000 = 49000
#     # in this for loop we will iterate unti the last 1000 multiplier
#     for i in range(0, tr_loop, 1000):
#         y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1])
#     # we will be predicting for the last data points
#     if data.shape[0]%1000 !=0:
#         y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
    
#     return y_data_pred


# In[ ]:


print("DONE TILL HERE")


# In[ ]:


# import matplotlib.pyplot as plt
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import roc_auc_score
# import math
# train_auc = []
# cv_auc = []
# log_alphas = []
# alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
# for i in tqdm(alphas):
#     nb = MultinomialNB(alpha = i)
#     nb.fit(x_train_onehot_bow, yy_train)
#     y_train_pred = batch_predict(nb, x_train_onehot_bow)
#     y_cv_pred = batch_predict(nb, x_cv_onehot_bow)
# # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# # not the predicted outputs
#     train_auc.append(roc_auc_score(yy_train,y_train_pred))
#     cv_auc.append(roc_auc_score(y_cv, y_cv_pred))
    
# for a in tqdm(alphas):
# #     b = math.log(a)
#     b = np.log10(a)
#     log_alphas.append(b)
    
# plt.figure(figsize=(10,5))
# plt.plot(log_alphas, train_auc, label='Train AUC')
# plt.plot(log_alphas, cv_auc, label='CV AUC')
# plt.scatter(log_alphas, train_auc, label='Train AUC points')
# plt.scatter(log_alphas, cv_auc, label='CV AUC points')
# plt.legend()
# plt.xlabel("log(alpha): hyperparameter")
# plt.ylabel("AUC")
# plt.title("alpha: hyperparameter v/s AUC")
# plt.grid()
# plt.show()


# In[ ]:


# log_alphas


# In[ ]:


# print(nb.classes_)


# In[ ]:


# len(train_auc)


# # GRIDSEARCHCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import math


# In[ ]:


from sklearn.model_selection import GridSearchCV
mnb_bow = MultinomialNB(class_prior=[0.5, 0.5])
parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
clf = GridSearchCV(mnb_bow, parameters, cv= 10, scoring='roc_auc',verbose=1,return_train_score=True)
# clf.fit(x_cv_onehot_bow, y_cv)
clf.fit(x_train_onehot_bow,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestAlpha_1=clf.best_params_['alpha']
bestScore_1=clf.best_score_
print("BEST ALPHA: ",clf.best_params_['alpha']," BEST SCORE: ",clf.best_score_) #clf.best_estimator_.alpha


# In[ ]:


alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
log_alphas =[]
for a in tqdm(alphas):
#     b = math.log(a)
    b = np.log10(a)
    log_alphas.append(b)
plt.figure(figsize=(10,5))
plt.plot(log_alphas, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(log_alphas,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.3,color='darkblue')
plt.plot(log_alphas, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(log_alphas,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.3,color='darkorange')
plt.scatter(log_alphas, train_auc, label='Train AUC points')
plt.scatter(log_alphas, cv_auc, label='CV AUC points')
plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("alpha: hyperparameter v/s AUC")
plt.grid()
plt.show()


# # TESTING WITH BEST HYPERPARAMETER VALUE ON SET 1

# In[ ]:


#https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
mnb_bow_testModel = MultinomialNB(alpha = bestAlpha_1,class_prior=[0.5, 0.5])
mnb_bow_testModel.fit(x_train_onehot_bow, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs
# y_train_pred = batch_predict(mnb_bow_testModel, x_train_onehot_bow)
y_train_pred=mnb_bow_testModel.predict_proba(x_train_onehot_bow)[:,1]
# y_test_pred = batch_predict(mnb_bow_testModel, x_test_onehot_bow)
y_test_pred=mnb_bow_testModel.predict_proba(x_test_onehot_bow)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

ax = plt.subplot()

auc_set1_train=auc(train_fpr, train_tpr)
auc_set1_test=auc(test_fpr, test_tpr)


ax.plot(train_fpr, train_tpr, label="Train AUC ="+str(auc(train_fpr, train_tpr)))
ax.plot(test_fpr, test_tpr, label="Test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("False Positive Rate(FPR)")
plt.ylabel("True Positive Rate(TPR)")
plt.title("AUC")
plt.grid(b=True, which='major', color='k', linestyle=':')
ax.set_facecolor("white")
plt.show()


# In[ ]:





# In[ ]:


def predict(proba, threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    predictions = []
    for i in proba:
        if i>=t:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# ### ->->-> 8.1.3.1: <font color='red'> SET 1</font> Confusion Matrix: Train

# In[ ]:


## TRAIN
print("="*100)
from sklearn.metrics import confusion_matrix
print("Train confusion matrix")
print(confusion_matrix(y_train, predict(y_train_pred, tr_thresholds, train_fpr, train_tpr)))

conf_matr_df_train = pd.DataFrame(confusion_matrix(y_train, predict(y_train_pred, tr_thresholds,train_fpr, train_tpr)), range(2),range(2))

## Heatmaps -> https://likegeeks.com/seaborn-heatmap-tutorial/
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_matr_df_train, annot=True,annot_kws={"size": 26}, fmt='g',cmap="YlGnBu")


# ### ->->-> 8.1.3.2: <font color='red'> SET 1</font> Confusion Matrix: Test

# In[ ]:


## TEST
print("="*100)
print("Test confusion matrix")
print(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)))

conf_matr_df_test = pd.DataFrame(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_matr_df_test, annot=True,annot_kws={"size": 16}, fmt='g')


# # -> 8.2:<font color='red'> SET 2</font>  Applying Naive Bayes on TFIDF.

# ## ->-> 8.1.1: <font color='red'> SET 2</font> Hyper parameter tuning to find best 'alpha' using GRIDSEARCHCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
mnb_tfidf = MultinomialNB(class_prior=[0.5, 0.5])
parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
clf = GridSearchCV(mnb_tfidf, parameters, cv= 10, scoring='roc_auc',verbose=1,return_train_score=True)
# clf.fit(x_cv_onehot_tfidf, y_cv)
clf.fit(x_train_onehot_tfidf,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestAlpha_2=clf.best_params_['alpha']
bestScore_2=clf.best_score_
print("BEST ALPHA: ",clf.best_params_['alpha']," BEST SCORE: ",clf.best_score_) #clf.best_estimator_.alpha


# In[ ]:


alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
log_alphas =[]
for a in tqdm(alphas):
#     b = math.log(a)
    b = np.log10(a)
    log_alphas.append(b)
plt.figure(figsize=(10,5))
plt.plot(log_alphas, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(log_alphas,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.3,color='darkblue')
plt.plot(log_alphas, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(log_alphas,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.3,color='darkorange')

ax = plt.subplot()
ax.scatter(log_alphas, train_auc, label='Train AUC points')
ax.scatter(log_alphas, cv_auc, label='CV AUC points')
plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("alpha: hyperparameter v/s AUC")
plt.grid(b=True, which='major', color='k', linestyle=':')
ax.set_facecolor("white")
plt.show()


# # TESTING WITH BEST HYPERPARAMETER VALUE ON SET 2

# In[ ]:


#https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
mnb_tfidf_testModel = MultinomialNB(alpha = bestAlpha_2,class_prior=[0.5, 0.5])
mnb_tfidf_testModel.fit(x_train_onehot_tfidf, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs
# y_train_pred = batch_predict(mnb_tfidf_testModel, x_train_onehot_tfidf)
# y_test_pred = batch_predict(mnb_tfidf_testModel, x_test_onehot_tfidf)
y_train_pred=mnb_tfidf_testModel.predict_proba(x_train_onehot_tfidf)[:,1]
y_test_pred=mnb_tfidf_testModel.predict_proba(x_test_onehot_tfidf)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

ax = plt.subplot()

auc_set2_train=auc(train_fpr, train_tpr)
auc_set2_test=auc(test_fpr, test_tpr)

# plt.plot(train_fpr, train_tpr, label="Train AUC ="+str(auc(train_fpr, train_tpr)))
# plt.plot(test_fpr, test_tpr, label="Test AUC ="+str(auc(test_fpr, test_tpr)))
ax.plot(train_fpr, train_tpr, label="Train AUC ="+str(auc(train_fpr, train_tpr)))
ax.plot(test_fpr, test_tpr, label="Test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()
plt.xlabel("False Positive Rate(FPR)")
plt.ylabel("True Positive Rate(TPR)")
plt.title("AUC")
plt.grid(b=True, which='major', color='k', linestyle=':')
ax.set_facecolor("white")
plt.show()


# In[ ]:


def predict(proba, threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    predictions = []
    for i in proba:
        if i>=t:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# ### ->->-> 8.1.3.1: <font color='red'> SET 2</font> Confusion Matrix: Train

# In[ ]:


## TRAIN
print("="*100)
from sklearn.metrics import confusion_matrix
print("Train confusion matrix")
print(confusion_matrix(y_train, predict(y_train_pred, tr_thresholds, train_fpr, train_tpr)))

conf_matr_df_train = pd.DataFrame(confusion_matrix(y_train, predict(y_train_pred, tr_thresholds,train_fpr, train_tpr)), range(2),range(2))

## Heatmaps -> https://likegeeks.com/seaborn-heatmap-tutorial/
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_matr_df_train, annot=True,annot_kws={"size": 26}, fmt='g',cmap="YlGnBu")


# ### ->->-> 8.1.3.2: <font color='red'> SET 2</font> Confusion Matrix: Test

# In[ ]:


## TEST
print("="*100)
print("Test confusion matrix")
print(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)))

conf_matr_df_test = pd.DataFrame(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_matr_df_test, annot=True,annot_kws={"size": 16}, fmt='g')


# # 9. TOP 10 FEATURES FOR SET 1

# ## SET 1: NEGATIVE CLASS

# In[ ]:


all_feature_names_bow=[]


# In[ ]:


## FOR SET 1 and SET 2
for i in vectorizer_sub.get_feature_names():
    all_feature_names_bow.append(i)   

for i in vectorizer_sub_sub.get_feature_names():
    all_feature_names_bow.append(i)    

for i in vectorizer_state.get_feature_names():
    all_feature_names_bow.append(i)    

for i in vectorizer_prefix.get_feature_names():
    all_feature_names_bow.append(i)   

for i in vectorizer_grade.get_feature_names():
    all_feature_names_bow.append(i)   

for i in vectorizer_title_bow.get_feature_names():
    all_feature_names_bow.append(i)    

for i in vectorizer_essay_bow.get_feature_names():
    all_feature_names_bow.append(i)   

all_feature_names_bow.append("price")

all_feature_names_bow.append("prev_proj")


# In[ ]:


print(len(all_feature_names_bow))


# In[ ]:


totalFeatureNamesBow=len(all_feature_names_bow)


# In[ ]:


x_train_onehot_bow.shape


# In[ ]:


totalFeatureNamesBow


# In[ ]:


nb_bow=MultinomialNB(alpha=0.5,class_prior=[0.5,0.5])
# nb_bow.fit(X-tr,y_train)
nb_bow.fit(x_train_onehot_bow,y_train)

# x_train_onehot
# x_cv_onehot
# x_test_onehot


# In[ ]:


bow_features_probs_neg = {}
for a in range(totalFeatureNamesBow) :
# for a in range(101) :
    bow_features_probs_neg[a] = nb_bow.feature_log_prob_[0,a]


# In[ ]:


# len(bow_features_probs)


# In[ ]:


final_bow_features_neg = pd.DataFrame({'feature_prob_estimates' : list(bow_features_probs_neg.values()),
'feature_names' : list(all_feature_names_bow)})


# In[ ]:


a = final_bow_features_neg.sort_values(by = ['feature_prob_estimates'], ascending = False)


# In[ ]:


print("TOP 10 Negative features - BOW")
a.head(10)


# ## SET 1:  POSITIVE CLASS

# In[ ]:


bow_features_probs_pos = {}
for a in range(totalFeatureNamesBow) :
# for a in range(101) :
    bow_features_probs_pos[a] = nb_bow.feature_log_prob_[1,a]

len(bow_features_probs_pos)

final_bow_features_pos = pd.DataFrame({'feature_prob_estimates' : list(bow_features_probs_pos.values()),
'feature_names' : list(all_feature_names_bow)})

b = final_bow_features_pos.sort_values(by = ['feature_prob_estimates'], ascending = False)

print("TOP 10 Positive features - BOW")
b.head(10)


# # 10. TOP 10 FEATURES FOR SET 2

# ## SET 2:  NEGATIVE CLASS

# In[ ]:


all_feature_names_tfidf=[]


# In[ ]:


## FOR TFIDF SET 2
for i in vectorizer_sub.get_feature_names():
    all_feature_names_tfidf.append(i)

for i in vectorizer_sub_sub.get_feature_names():
    all_feature_names_tfidf.append(i)

for i in vectorizer_state.get_feature_names():
    all_feature_names_tfidf.append(i)

for i in vectorizer_prefix.get_feature_names():
    all_feature_names_tfidf.append(i)

for i in vectorizer_grade.get_feature_names():
    all_feature_names_tfidf.append(i)

for i in vectorizer_essay_tfidf.get_feature_names():
    all_feature_names_tfidf.append(i)

for i in vectorizer_title_tfidf.get_feature_names():
    all_feature_names_tfidf.append(i)

all_feature_names_tfidf.append("price")
all_feature_names_tfidf.append("prev_proj")


# In[ ]:


nb_tfidf=MultinomialNB(alpha=0.5,class_prior=[0.5,0.5])
nb_tfidf.fit(x_train_onehot_tfidf,y_train)


# In[ ]:


totalFeatureNamesTfidf=len(all_feature_names_tfidf)


# In[ ]:


tfidf_features_probs_neg = {}
for a in range(totalFeatureNamesTfidf) :
# for a in range(101) :
    tfidf_features_probs_neg[a] = nb_tfidf.feature_log_prob_[0,a]


# In[ ]:


len(tfidf_features_probs_neg)


# In[ ]:


final_tfidf_features_neg = pd.DataFrame({'feature_prob_estimates' : list(tfidf_features_probs_neg.values()),
'feature_names' : list(all_feature_names_tfidf)})

c = final_tfidf_features_neg.sort_values(by = ['feature_prob_estimates'], ascending = False)

print("TOP 10 Negative features -  TFIDF")
c.head(10)


# ##  SET 2: POSITIVE CLASS

# In[ ]:


tfidf_features_probs_pos = {}
for a in range(totalFeatureNamesTfidf) :
# for a in range(101) :
    tfidf_features_probs_pos[a] = nb_tfidf.feature_log_prob_[1,a]

len(tfidf_features_probs_pos)

final_tfidf_features_pos = pd.DataFrame({'feature_prob_estimates' : list(tfidf_features_probs_pos.values()),
'feature_names' : list(all_feature_names_tfidf)})

d = final_tfidf_features_pos.sort_values(by = ['feature_prob_estimates'], ascending = False)

print("TOP 10 Positive features - TFIDF")
d.head(10)


# # 11. CONCLUSION

# In[ ]:


from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["Vectorizer", "Model", "Hyperparameter: Alpha", "Train AUC", "Test AUC"]
auc_set2_train=auc(train_fpr, train_tpr)
auc_set2_test=auc(test_fpr, test_tpr)

x.add_row(["BOW", "Multinomial Naive Bayes", bestAlpha_1, round(auc_set1_train,2),round(auc_set1_test,2)])
x.add_row(["TF-IDF", "Multinomial Naive Bayes", bestAlpha_2, round(auc_set2_train,2),round(auc_set2_test,2)])

print(x)

