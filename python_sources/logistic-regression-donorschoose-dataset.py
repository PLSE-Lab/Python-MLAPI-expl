#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression: DonorsChoose Dataset
# 
# ### A. DATA INFORMATION
# ### B. OBJECTIVE
# ## 1. READING THE DATASET
# - **1.1 Removing Nan**
# - **1.2 Adding Quantity in Dataset**
# 
# ## 2. PREPROCESSING 
# - **2.1 Preprocessing: Project Subject Categories**
# - **2.2 Preprocessing: Project Subject Sub Categories**
# - **2.3 Preprocessing: Project Grade**
# 
# ## 3. TEXT PROCESSING
# - **3.1 Text Preprocessing: Essays**
# - **3.2 Text Preprocessing: Title**
# - **3.3 Text Preprocessing: Calculating Size of Title and Essay**
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
#     - 6.2.3: Standarizing Numerical data: Title Size
#     - 6.2.4: Standarizing Numerical data: Essay Size
#     - 6.2.5: Standarizing Numerical data: Quantity
#     
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
#     - **6.3.3: AVG Word2VecF**
#         - 6.3.3.1: Avg Word2Vec: Essays (A. Train, B. CV, C. Test)
#         - 6.3.3.2: Avg Word2Vec: Title (A. Train, B. CV, C. Test)
#     - **6.3.4: TF-IDF Word2VecF**
#         - 6.3.4.1: TF-IDF Word2Vec: Essays (A. Train, B. CV, C. Test)
#         - 6.3.4.2: TF-IDF Word2Vec: Title (A. Train, B. CV, C. Test)
#                 
# ## 7. MERGING FEATURES
# - 7.1: Merging all ONE HOT features.
# - 7.2: SET 1: Merging All ONE HOT with BOW (Title and Essay) features.
# - 7.3: SET 2: Merging All ONE HOT with TF-IDF (Title and Essay) features.
# - 7.3: SET 3: Merging All ONE HOT with Avg W2v (Title and Essay) features.
# - 7.4: SET 4: Merging All ONE HOT with TF-IDF W2v (Title and Essay) features.
#     
# ## 8. LOGISTIC REGRESSION
# - **8.1: SET 1 Applying Logistic Regression on BOW.**
#     - 8.1.1: SET 1 Hyper parameter tuning to find the best "C".
#     - 8.1.2: SET 1 TESTING the performance of the model on test data, plotting ROC Curves.
#     - 8.1.3: SET 1 Confusion Matrix
#         - 8.1.3.1: SET 1 Confusion Matrix: Train
#         - 8.1.3.2: SET 1 Confusion Matrix: Test
#             
# - **8.2: SET 2 Applying Logistic Regression on TF-IDF.**
#     - 8.2.1: SET 2 Hyper parameter tuning to find the best "C".
#     - 8.2.2: SET 2 TESTING the performance of the model on test data, plotting ROC Curves.
#     - 8.2.3: SET 2 Confusion Matrix
#         - 8.2.3.1: SET 2 Confusion Matrix: Train
#         - 8.2.3.2: SET 2 Confusion Matrix: Test
#         
# - **8.3: SET 3 Applying Logistic Regression on Avg W2V.**
#     - 8.3.1: SET 3 Hyper parameter tuning to find the best "C".
#     - 8.3.2: SET 3 TESTING the performance of the model on test data, plotting ROC Curves.
#     - 8.3.3: SET 3 Confusion Matrix
#         - 8.3.3.1: SET 3 Confusion Matrix: Train
#         - 8.3.3.2: SET 3 Confusion Matrix: Test
#         
# - **8.4: SET 4 Applying Logistic Regression on TF-IDF W2V.**
#     - 8.4.1: SET 4 Hyper parameter tuning to find the best "C".
#     - 8.4.2: SET 4 TESTING the performance of the model on test data, plotting ROC Curves.
#     - 8.4.3: SET 4 Confusion Matrix
#         - 8.4.3.1: SET 4 Confusion Matrix: Train
#         - 8.4.3.2: SET 4 Confusion Matrix: Test
#         
# - **8.5: SET 5 Applying Logistic Regression on ALL (Expect Text Features)TF-IDF W2V.**
#     - **Performing Sentiment Analysis**
#     - **Merging All the non text features**
#     - 8.5.1: SET 5 Hyper parameter tuning to find the best "C".
#     - 8.5.2: SET 5 TESTING the performance of the model on test data, plotting ROC Curves.
#     - 8.5.3: SET 5 Confusion Matrix
#         - 8.5.3.1: SET 5 Confusion Matrix: Train
#         - 8.5.3.2: SET 5 Confusion Matrix: Test
# 
# 
# ## 9. CONCLUSION 

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
# import os
# print(os.listdir("../input"))
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


# In[ ]:


project_data.columns


# In[ ]:


project_data.shape


# ## Adding Quantity in the dataset

# In[ ]:


price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
project_data = pd.merge(project_data, price_data, on='id', how='left')


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


# In[ ]:


cat_dict


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


project_data.shape


# In[ ]:


## new column added as PreProcessed_Essays and older unProcessed essays column is deleted
project_data['preprocessed_essays'] = preprocessed_essays
# project_data.drop(['essay'], axis=1, inplace=True)


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


# ## -> 3.3: Calculating the size of Title and Essay

# In[ ]:


# project_data['']
for col_type, new_col in [('project_title', 'title_size'), ('essay', 'essay_size')]:
    print("Now in: ",col_type)
    col_data = project_data[col_type]
    print(col_data.head(10))
    col_size = []
    for sen in col_data:
        sen = decontracted(sen)
        col_size.append(len(sen.split()))
    project_data[new_col] = col_size
    col_size.clear()


# In[ ]:


project_data.shape


# In[ ]:


project_data.columns


# In[ ]:


project_data['title_size'].head(10)


# In[ ]:


project_data['essay_size'].head(10)


# In[ ]:


project_bkp=project_data.copy()


# In[ ]:


project_bkp.shape


# # 4. SAMPLING
# ## -> 4.1: Taking Sample from the complete dataset
# ## NOTE: A sample of 10000 Datapoints is taken due to lack computational resource.

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

# # 5. UPSAMPLING THE MINORITY CLASS WITH REPLACEMENT STRATEGY: NOT DONE 

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


# **Conclusion:**
# 1. Resampling is performed on the Training data.
# 1. Training data in now **BALANCED**.

# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


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
# vectorizer_sub = CountVectorizer(vocabulary=list(sorted_cat_dict.keys()), lowercase=False, binary=True)
vectorizer_sub = CountVectorizer( lowercase=False, binary=True)

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
# vectorizer_sub_sub = CountVectorizer(vocabulary=list(sorted_sub_cat_dict.keys()), lowercase=False, binary=True)
vectorizer_sub_sub = CountVectorizer( lowercase=False, binary=True)

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
# vectorizer_state = CountVectorizer(vocabulary=set(statelist), lowercase=False, binary=True)
vectorizer_state = CountVectorizer( lowercase=False, binary=True)

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
# vectorizer_prefix = CountVectorizer(vocabulary=list(sorted_teacher_prefix_cat_dict.keys()), lowercase=False, binary=True)
vectorizer_prefix = CountVectorizer(lowercase=False, binary=True)

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
# vectorizer_grade = CountVectorizer(vocabulary=list(sorted_project_grade_cat_dict.keys()), lowercase=False, binary=True)
vectorizer_grade = CountVectorizer(lowercase=False, binary=True)

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


# In[ ]:


x_train['price_x'].head(10)


# In[ ]:


x_train['price_y'].head(10)


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

normalizer.fit(x_train['price_x'].values.reshape(1,-1)) # finding the mean and standard deviation of this data
# print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now normalize the data

x_train_price_normalized = normalizer.transform(x_train['price_x'].values.reshape(1, -1)).reshape(-1,1)
x_test_price_normalized  = normalizer.transform(x_test['price_x'].values.reshape(1, -1)).reshape(-1,1)
# x_cv_price_normalized    = normalizer.transform(x_cv['price'].values.reshape(1, -1)).reshape(-1,1)


# print("Shape of matrix after normalization -> Teacher Prefix: x_train: ",x_train_prefix_one_hot.shape)
# # print("Shape of matrix after normalization -> Teacher Prefix: x_cv   : ",x_cv_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Teacher Prefix: x_test : ",x_test_prefix_one_hot.shape)


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

# print("Shape of matrix after normalization -> Teachers Previous Projects: x_train:  ",x_train_prefix_one_hot.shape)
# # print("Shape of matrix after normalization -> Teachers Previous Projects: x_cv   :  ",x_cv_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Teachers Previous Projects: x_test :  ",x_test_prefix_one_hot.shape)


# ## ->-> 6.2.3: Normalizing Numerical data: Title Size

# In[ ]:


x_train_teacher_previous_proj_normalized


# In[ ]:


x_train.columns


# In[ ]:


project_data['title_size'].head(10)


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

normalizer.fit(x_train['title_size'].values.reshape(1,-1)) # finding the mean and standard deviation of this data
# print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now normalize the data

x_train_title_normalized = normalizer.transform(x_train['title_size'].values.reshape(1, -1)).reshape(-1,1)
x_test_title_normalized  = normalizer.transform(x_test['title_size'].values.reshape(1, -1)).reshape(-1,1)
# x_cv_price_normalized    = normalizer.transform(x_cv['price'].values.reshape(1, -1)).reshape(-1,1)


# print("Shape of matrix after normalization -> Project Title: x_train: ",x_train_title_one_hot.shape)
# # print("Shape of matrix after normalization -> Teacher Prefix: x_cv   : ",x_cv_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Project Title: x_test : ",x_test_title_one_hot.shape)


# In[ ]:


type(x_train_title_normalized)


# In[ ]:


x_train_title_normalized


# ## ->-> 6.2.4: Normalizing Numerical data: Essay Size

# In[ ]:


project_data['essay_size']


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

normalizer.fit(x_train['essay_size'].values.reshape(1,-1)) # finding the mean and standard deviation of this data
# print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now normalize the data

x_train_essay_normalized = normalizer.transform(x_train['essay_size'].values.reshape(1, -1)).reshape(-1,1)
x_test_essay_normalized  = normalizer.transform(x_test['essay_size'].values.reshape(1, -1)).reshape(-1,1)
# x_cv_price_normalized    = normalizer.transform(x_cv['price'].values.reshape(1, -1)).reshape(-1,1)


# print("Shape of matrix after normalization -> Project Title: x_train: ",x_train_title_one_hot.shape)
# # print("Shape of matrix after normalization -> Teacher Prefix: x_cv   : ",x_cv_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Project Title: x_test : ",x_test_title_one_hot.shape)


# In[ ]:


x_train_essay_normalized


# ## ->-> 6.2.5: Normalizing Numerical data: Quantity

# In[ ]:


project_data['quantity_x']


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

normalizer.fit(x_train['quantity_x'].values.reshape(1,-1)) # finding the mean and standard deviation of this data
# print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now normalize the data

x_train_quantity_normalized = normalizer.transform(x_train['quantity_y'].values.reshape(1, -1)).reshape(-1,1)
x_test_quantity_normalized  = normalizer.transform(x_test['quantity_y'].values.reshape(1, -1)).reshape(-1,1)
# x_cv_price_normalized    = normalizer.transform(x_cv['price'].values.reshape(1, -1)).reshape(-1,1)


# print("Shape of matrix after normalization -> Project Title: x_train: ",x_train_title_one_hot.shape)
# # print("Shape of matrix after normalization -> Teacher Prefix: x_cv   : ",x_cv_prefix_one_hot.shape)
# print("Shape of matrix after normalization -> Project Title: x_test : ",x_test_title_one_hot.shape)


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


# In[ ]:


## bigram using countvectorizer example -> https://stackoverflow.com/a/24006054/4433839
# v = CountVectorizer(ngram_range=( 2,2))
# print(v.fit(["an apple a day keeps the doctor away"]).vocabulary_)

# We are considering only the words which appeared in at least 10 documents(rows or projects).
vectorizer_essay_bow_bigram = CountVectorizer(ngram_range=(2,2),min_df=10,max_features=5000)

vectorizer_essay_bow_bigram.fit(x_train['preprocessed_essays'])

x_train_essays_bow_bigram = vectorizer_essay_bow_bigram.transform(x_train['preprocessed_essays'])
# x_cv_essays_bow    = vectorizer_essay_bow.transform(x_cv['preprocessed_essays'])
x_test_essays_bow_bigram  = vectorizer_essay_bow_bigram.transform(x_test['preprocessed_essays'])

print("Shape of matrix after BOW -> Essays: x_train: ",x_train_essays_bow_bigram.shape)
# print("Shape of matrix after BOW -> Essays: x_cv   : ",x_cv_essays_bow.shape)
print("Shape of matrix after BOW -> Essays: x_test : ",x_test_essays_bow_bigram.shape)


# In[ ]:


# print(vectorizer_essay_bow_bigram.vocabulary_)
print(type(x_train_essays_bow_bigram))


# In[ ]:


print(type(vectorizer_essay_bow_bigram))


# ## ->->-> 6.3.1.2: BOW: Title (Train, CV, Test)

# In[ ]:


vectorizer_title_bow = CountVectorizer(min_df=2)

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
vectorizer_essay_tfidf_bigram = TfidfVectorizer(ngram_range=(2,2),min_df=10,max_features=5000)

vectorizer_essay_tfidf_bigram.fit(x_train['preprocessed_essays'])

x_train_essays_tfidf_bigram = vectorizer_essay_tfidf_bigram.transform(x_train['preprocessed_essays'])
# x_cv_essays_tfidf    = vectorizer_essay_tfidf.transform(x_cv['preprocessed_essays'])
x_test_essays_tfidf_bigram  = vectorizer_essay_tfidf_bigram.transform(x_test['preprocessed_essays'])

print("Shape of matrix after TF-IDF -> Essay: x_train: ",x_train_essays_tfidf_bigram.shape)
# print("Shape of matrix after TF-IDF -> Essay: x_cv   : ",x_cv_essays_tfidf.shape)
print("Shape of matrix after TF-IDF -> Essay: x_test : ",x_test_essays_tfidf_bigram.shape)


# ## ->->-> 6.3.2.2: TF-IDF: Title (Train, CV, Test)

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_title_tfidf = TfidfVectorizer(min_df=2)

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


# # -> 6.3.3:  AVG Word2Vec

# In[ ]:


'''
# Reading glove vectors in python: https://stackoverflow.com/a/38230349/4084039
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf8")
    model = {}
    for line in tqdm(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
model = loadGloveModel('glove.42B.300d.txt')

# ============================
Output:
    
Loading Glove Model
1917495it [06:32, 4879.69it/s]
Done. 1917495  words loaded!

# ============================

words = []
for i in preproced_texts:
    words.extend(i.split(' '))

for i in preproced_titles:
    words.extend(i.split(' '))
print("all the words in the coupus", len(words))
words = set(words)
print("the unique words in the coupus", len(words))

inter_words = set(model.keys()).intersection(words)
print("The number of words that are present in both glove vectors and our coupus", \
      len(inter_words),"(",np.round(len(inter_words)/len(words)*100,3),"%)")

words_courpus = {}
words_glove = set(model.keys())
for i in words:
    if i in words_glove:
        words_courpus[i] = model[i]
print("word 2 vec length", len(words_courpus))


# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/

import pickle
with open('glove_vectors', 'wb') as f:
    pickle.dump(words_courpus, f)


'''


# words_train_essays = []
# for i in preprocessed_essays_train :
#     words_train_essays.extend(i.split(' '))

# ## Find the total number of words in the Train data of Essays.
# print("all the words in the corpus", len(words_train_essays))

# ## Find the unique words in this set of words
# words_train_essay = set(words_train_essays)
# print("the unique words in the corpus", len(words_train_essay))

# ## Find the words present in both Glove Vectors as well as our corpus.
# inter_words = set(model.keys()).intersection(words_train_essay)
# print("The number of words that are present in both glove vectors and our corpus are {} which is nearly {}% ".format(len(inter_words), np.round((float(len(inter_words))/len(words_train_essay))*100)))

# words_corpus_train_essay = {}
# words_glove = set(model.keys())
# for i in words_train_essay:
#     if i in words_glove:
#         words_corpus_train_essay[i] = model[i]
# print("word 2 vec length", len(words_corpus_train_essay))


# In[ ]:


# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
# make sure you have the glove_vectors file
# with open('/content/drive/My Drive/Colab Notebooks/glove_vectors', 'rb') as f:
with open('/kaggle/input/donorschoosedataset/glove_vectors/glove_vectors', 'rb') as f:
# with open('../glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words =  set(model.keys())


# ## ->-> 6.3.3.1: Avg W2v: ESSAYS

# ###  ->->->6.3.3.1. A:  Avg W2V: Essays -> Train

# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
x_train_essays_avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
# for sentence in tqdm(x_train_upsampled['preprocessed_essays']): # for each review/sentence
for sentence in tqdm(x_train['preprocessed_essays']): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    x_train_essays_avg_w2v_vectors.append(vector)

print("Total number of vectors  : AVG-W2V -> Essay: x_train: ",len(x_train_essays_avg_w2v_vectors))
print("Length of a Single vector: AVG-W2V -> Essay: x_train: ",len(x_train_essays_avg_w2v_vectors[0]))


# ###  ->->->6.3.3.1. B:  Avg W2V: Essays -> CV

# In[ ]:


# ## NOT DONE CV

# x_cv_essays_avg_w2v_vectors = [];
# for sentence in tqdm(x_cv['preprocessed_essays']): # for each review/sentence
#     vector = np.zeros(300) # as word vectors are of zero length
#     cnt_words =0; # num of words with a valid vector in the sentence/review
#     for word in sentence.split(): # for each word in a review/sentence
#         if word in glove_words:
#             vector += model[word]
#             cnt_words += 1
#     if cnt_words != 0:
#         vector /= cnt_words
#     x_cv_essays_avg_w2v_vectors.append(vector)

# print("Total number of vectors  : AVG-W2V -> Essay: x_cv   : ",len(x_cv_essays_avg_w2v_vectors))
# print("Length of a Single vector: AVG-W2V -> Essay: x_cv   : ",len(x_cv_essays_avg_w2v_vectors[0]))


# ###  ->->->6.3.3.1. C: Avg W2V: Essays -> Test

# In[ ]:


x_test_essays_avg_w2v_vectors = [];
for sentence in tqdm(x_test['preprocessed_essays']): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    x_test_essays_avg_w2v_vectors.append(vector)

print("Total number of vectors  : AVG-W2V -> Essay: x_test : ",len(x_test_essays_avg_w2v_vectors))
print("Length of a Single vector: AVG-W2V -> Essay: x_test : ",len(x_test_essays_avg_w2v_vectors[0]))


# ## ->-> 6.3.3.2: TITLE

# ###   ->->->6.3.3.2. A: Avg W2V: Title -> Train

# In[ ]:


x_train_titles_avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
# for sentence in tqdm(x_train_upsampled['preprocessed_titles']): # for each review/sentence
for sentence in tqdm(x_train['preprocessed_titles']): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    x_train_titles_avg_w2v_vectors.append(vector)

print("Total number of vectors  : AVG-W2V -> Titles: x_train: ",len(x_train_titles_avg_w2v_vectors))
print("Length of a Single vector: AVG-W2V -> Titles: x_train: ",len(x_train_titles_avg_w2v_vectors[0]))


# ###   ->->->6.3.3.2. B: Avg W2V: Title -> CV

# In[ ]:


# ### NOT DONE HERE


# x_cv_titles_avg_w2v_vectors = [];
# for sentence in tqdm(x_cv['preprocessed_titles']): # for each review/sentence
#     vector = np.zeros(300) # as word vectors are of zero length
#     cnt_words =0; # num of words with a valid vector in the sentence/review
#     for word in sentence.split(): # for each word in a review/sentence
#         if word in glove_words:
#             vector += model[word]
#             cnt_words += 1
#     if cnt_words != 0:
#         vector /= cnt_words
#     x_cv_titles_avg_w2v_vectors.append(vector)

# print("Total number of vectors  : AVG-W2V -> Titles: x_cv   : ",len(x_cv_titles_avg_w2v_vectors))
# print("Length of a Single vector: AVG-W2V -> Titles: x_cv   : ",len(x_cv_titles_avg_w2v_vectors[0]))


# ###   ->->->6.3.3.2. C: Avg W2V: Title -> Test

# In[ ]:


x_test_titles_avg_w2v_vectors = [];
for sentence in tqdm(x_test['preprocessed_titles']): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    x_test_titles_avg_w2v_vectors.append(vector)

print("Total number of vectors  : AVG-W2V -> Titles: x_test : ",len(x_test_titles_avg_w2v_vectors))
print("Length of a Single vector: AVG-W2V -> Titles: x_test : ",len(x_test_titles_avg_w2v_vectors[0]))


# # ->6.3.4: TFIDF - W2V

# ## ->-> 6.3.4.1: ESSAYS

# ###   ->->-> 6.3.4.1. A: TF-IDF W2V: Essays -> Train

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
tfidf_model = TfidfVectorizer()
# tfidf_model.fit(x_train_upsampled['preprocessed_essays'])
tfidf_model.fit(x_train['preprocessed_essays'])
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
tfidf_words = set(tfidf_model.get_feature_names())


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
x_train_essays_tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
# for sentence in tqdm(x_train_upsampled['preprocessed_essays']): # for each review/sentence
for sentence in tqdm(x_train['preprocessed_essays']): # for each review/sentence
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
    x_train_essays_tfidf_w2v_vectors.append(vector)

print("Total number of vectors  : TFIDF-W2V -> Essays: x_train : ",len(x_train_essays_tfidf_w2v_vectors))
print("Length of a Single vector: TFIDF-W2V -> Essays: x_train : ",len(x_train_essays_tfidf_w2v_vectors[0]))


# ###   ->->-> 6.3.4.1. B: TF-IDF W2V: Essays -> CV

# In[ ]:


# ### NOT DONE HERE


# x_cv_essays_tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
# for sentence in tqdm(x_cv['preprocessed_essays']): # for each review/sentence
#     vector = np.zeros(300) # as word vectors are of zero length
#     tf_idf_weight =0; # num of words with a valid vector in the sentence/review
#     for word in sentence.split(): # for each word in a review/sentence
#         if (word in glove_words) and (word in tfidf_words):
#             vec = model[word] # getting the vector for each word
#             # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
#             tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
#             vector += (vec * tf_idf) # calculating tfidf weighted w2v
#             tf_idf_weight += tf_idf
#     if tf_idf_weight != 0:
#         vector /= tf_idf_weight
#     x_cv_essays_tfidf_w2v_vectors.append(vector)

# print("Total number of vectors  : TFIDF-W2V -> Essays: x_cv : ",len(x_cv_essays_tfidf_w2v_vectors))
# print("Length of a Single vector: TFIDF-W2V -> Essays: x_cv : ",len(x_cv_essays_tfidf_w2v_vectors[0]))


# ###   ->->-> 6.3.4.1. C: TF-IDF W2V: Essays -> Test

# In[ ]:


x_test_essays_tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['preprocessed_essays']): # for each review/sentence
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
    x_test_essays_tfidf_w2v_vectors.append(vector)

print("Total number of vectors  : TFIDF-W2V -> Essays: x_test : ",len(x_test_essays_tfidf_w2v_vectors))
print("Length of a Single vector: TFIDF-W2V -> Essays: x_test : ",len(x_test_essays_tfidf_w2v_vectors[0]))


# ## ->-> 6.3.4.2 TITLE

# ###   ->->-> 6.3.4.2. A: TF-IDF W2V: Title -> Train

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
tfidf_title_model = TfidfVectorizer()
# tfidf_title_model.fit(x_train_upsampled['preprocessed_titles'])
tfidf_title_model.fit(x_train['preprocessed_titles'])
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_title_model.get_feature_names(), list(tfidf_title_model.idf_)))
tfidf_title_words = set(tfidf_title_model.get_feature_names())


# In[ ]:


# average Word2Vec
# compute average word2vec for each title.
x_train_tfidf_w2v_title_vectors = []; # the avg-w2v for each title is stored in this list
# for sentence in x_train_upsampled['preprocessed_titles']: # for each review/sentence
for sentence in x_train['preprocessed_titles']: # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_title_weight =0; # num of words with a valid vector in the title
    for word in sentence.split(): # for each word in a title
        if (word in glove_words) and (word in tfidf_title_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_title_weight += tf_idf
    if tf_idf_title_weight != 0:
        vector /= tf_idf_title_weight
    x_train_tfidf_w2v_title_vectors.append(vector)

print("Total number of vectors  : TFIDF-W2V -> Titles: x_train : ",len(x_train_tfidf_w2v_title_vectors))
print("Length of a Single vector: TFIDF-W2V -> Titles: x_train : ",len(x_train_tfidf_w2v_title_vectors[0]))


# ###   ->->-> 6.3.4.2. B: TF-IDF W2V: Title -> CV

# In[ ]:


# ### NOT DONE HERE


# # average Word2Vec
# # compute average word2vec for each title.
# x_cv_tfidf_w2v_title_vectors = []; # the avg-w2v for each title is stored in this list
# for sentence in x_cv['preprocessed_titles']: # for each review/sentence
#     vector = np.zeros(300) # as word vectors are of zero length
#     tf_idf_title_weight =0; # num of words with a valid vector in the title
#     for word in sentence.split(): # for each word in a title
#         if (word in glove_words) and (word in tfidf_title_words):
#             vec = model[word] # getting the vector for each word
#             # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
#             tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
#             vector += (vec * tf_idf) # calculating tfidf weighted w2v
#             tf_idf_title_weight += tf_idf
#     if tf_idf_title_weight != 0:
#         vector /= tf_idf_title_weight
#     x_cv_tfidf_w2v_title_vectors.append(vector)

# print("Total number of vectors  : TFIDF-W2V -> Titles: x_cv : ",len(x_cv_tfidf_w2v_title_vectors))
# print("Length of a Single vector: TFIDF-W2V -> Titles: x_cv : ",len(x_cv_tfidf_w2v_title_vectors[0]))


# ###   ->->-> 6.3.4.2. C: TF-IDF W2V: Title -> Test

# In[ ]:


# average Word2Vec
# compute average word2vec for each title.
x_test_tfidf_w2v_title_vectors = []; # the avg-w2v for each title is stored in this list
for sentence in x_test['preprocessed_titles']: # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_title_weight =0; # num of words with a valid vector in the title
    for word in sentence.split(): # for each word in a title
        if (word in glove_words) and (word in tfidf_title_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_title_weight += tf_idf
    if tf_idf_title_weight != 0:
        vector /= tf_idf_title_weight
    x_test_tfidf_w2v_title_vectors.append(vector)

print("Total number of vectors  : TFIDF-W2V -> Titles: x_test : ",len(x_test_tfidf_w2v_title_vectors))
print("Length of a Single vector: TFIDF-W2V -> Titles: x_test : ",len(x_test_tfidf_w2v_title_vectors[0]))


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


x_train_onehot_bow = hstack((x_train_onehot,x_train_titles_bow,x_train_essays_bow_bigram)).tocsr()### Merging all ONE HOT features
# x_cv_onehot_bow    = hstack((x_cv_onehot, x_cv_titles_bow, x_cv_essays_bow)).tocsr()### Merging all ONE HOT features
x_test_onehot_bow  = hstack((x_test_onehot, x_test_titles_bow, x_test_essays_bow_bigram)).tocsr()### Merging all ONE HOT features
print("Type -> One Hot BOW -> x_train_cv_test: ",type(x_train_onehot_bow))
# print("Type -> One Hot BOW -> cv             : ",type(x_cv_onehot_bow))
print("Type -> One Hot BOW -> x_test         : ",type(x_test_onehot_bow))
print("\n")
print("Shape -> One Hot BOW -> x_train_cv_test: ",x_train_onehot_bow.shape)
# print("Shape -> One Hot BOW -> cv             : ",x_cv_onehot_bow.shape)
print("Shape -> One Hot BOW -> x_test         : ",x_test_onehot_bow.shape)


# ## -> 7.3: SET 2:  Merging All ONE HOT with TF-IDF (Title and Essay) features

# In[ ]:


x_train_onehot_tfidf = hstack((x_train_onehot,x_train_titles_tfidf, x_train_essays_tfidf_bigram)).tocsr()
# x_cv_onehot_tfidf    = hstack((x_cv_onehot,x_cv_titles_tfidf, x_cv_essays_tfidf)).tocsr()
x_test_onehot_tfidf  = hstack((x_test_onehot,x_test_titles_tfidf, x_test_essays_tfidf_bigram)).tocsr()
print("Type -> One Hot TFIDF -> x_train_cv_test: ",type(x_train_onehot_tfidf))
# print("Type -> One Hot TFIDF -> cv             : ",type(x_cv_onehot_tfidf))
print("Type -> One Hot TFIDF -> x_test         : ",type(x_test_onehot_tfidf))
print("\n")
print("Shape -> One Hot TFIDF -> x_train_cv_test: ",x_train_onehot_tfidf.shape)
# print("Shape -> One Hot TFIDF -> cv             : ",x_cv_onehot_tfidf.shape)
print("Shape -> One Hot TFIDF -> x_test         : ",x_test_onehot_tfidf.shape)### SET 1:  Merging ONE HOT with BOW (Title and Essay) features


# ## -> 7.4: SET 3:  Merging All ONE HOT with AVG W2V (Title and Essay) features

# In[ ]:


x_train_onehot_avg_w2v = hstack((x_train_onehot, x_train_titles_avg_w2v_vectors, x_train_essays_avg_w2v_vectors)).tocsr()
# x_cv_onehot_avg_w2v    = hstack((x_cv_onehot, x_cv_titles_avg_w2v_vectors, x_cv_essays_avg_w2v_vectors)).tocsr()
x_test_onehot_avg_w2v  = hstack((x_test_onehot, x_test_titles_avg_w2v_vectors, x_test_essays_avg_w2v_vectors)).tocsr()
print("Type -> One Hot AVG W2V-> x_train_cv_test: ",type(x_train_onehot_avg_w2v))
# print("Type -> One Hot AVG W2V-> cv             : ",type(x_cv_onehot_avg_w2v))
print("Type -> One Hot AVG W2V-> x_test         : ",type(x_test_onehot_avg_w2v))
print("\n")
print("Shape -> One Hot AVG W2V-> x_train_cv_test: ",x_train_onehot_avg_w2v.shape)
# print("Shape -> One Hot AVG W2V-> cv             : ",x_cv_onehot_avg_w2v.shape)
print("Shape -> One Hot AVG W2V-> x_test         : ",x_test_onehot_avg_w2v.shape)### SET 1:  Merging ONE HOT with BOW (Title and Essay) features


# ## -> 7.5: SET 4:  Merging All ONE HOT with TF-IDF W2V (Title and Essay) features

# In[ ]:


x_train_onehot_tfidf_w2v = hstack((x_train_onehot, x_train_tfidf_w2v_title_vectors, x_train_essays_tfidf_w2v_vectors)).tocsr()
# x_cv_onehot_tfidf_w2v    = hstack((x_cv_onehot, x_cv_tfidf_w2v_title_vectors, x_cv_essays_tfidf_w2v_vectors)).tocsr()
x_test_onehot_tfidf_w2v  = hstack((x_test_onehot, x_test_tfidf_w2v_title_vectors, x_test_essays_tfidf_w2v_vectors)).tocsr()
print("Type -> One Hot TFIDF W2V -> x_train_cv_test: ",type(x_train_onehot_tfidf_w2v))
# print("Type -> One Hot TFIDF W2V -> cv             : ",type(x_cv_onehot_tfidf_w2v))
print("Type -> One Hot TFIDF W2V -> x_test         : ",type(x_test_onehot_tfidf_w2v))
print("\n")
print("Shape -> One Hot TFIDF W2V -> x_train_cv_test: ",x_train_onehot_tfidf_w2v.shape)
# print("Shape -> One Hot TFIDF W2V -> cv             : ",x_cv_onehot_tfidf_w2v.shape)
print("Shape -> One Hot TFIDF W2V -> x_test         : ",x_test_onehot_tfidf_w2v.shape)### SET 1:  Merging ONE HOT with BOW (Title and Essay) features


# # 8. LOGISTIC REGRESSION
# 

# # -> 8.1:<font color='red'> SET 1</font>  Applying Logistic Regresion on BOW (Set 1).

# 
# ## ->-> 8.1.1: <font color='red'> SET 1</font> Hyper parameter tuning to find best 'C' using GRIDSEARCHCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import math


# In[ ]:


[10**x for x in range(-4,5)]


# In[ ]:


y_test.shape


# In[ ]:


from sklearn.model_selection import GridSearchCV
logreg_bow= LogisticRegression(class_weight='balanced')
# parameters = {'lambda':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
parameters = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}
clf = GridSearchCV(logreg_bow, parameters, cv= 3, scoring='roc_auc',verbose=1,return_train_score=True,n_jobs=-1)
# clf.fit(x_cv_onehot_bow, y_cv)
clf.fit(x_train_onehot_bow,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestC_1=clf.best_params_['C']
bestScore_1=clf.best_score_
print("BEST ALPHA: ",clf.best_params_['C']," BEST SCORE: ",clf.best_score_) #clf.best_estimator_.alpha


# In[ ]:


# alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
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


# 
# ## ->-> 8.1.2: <font color='red'> SET 1</font> TESTING the performance of the model on test data, plotting ROC Curves.

# In[ ]:


#https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
logreg_bow_testModel = LogisticRegression(class_weight='balanced',C=bestC_1)
logreg_bow_testModel.fit(x_train_onehot_bow, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs
# y_train_pred = batch_predict(mnb_bow_testModel, x_train_onehot_bow)
y_train_pred=logreg_bow_testModel.predict_proba(x_train_onehot_bow)[:,1]
# y_test_pred = batch_predict(mnb_bow_testModel, x_test_onehot_bow)
y_test_pred=logreg_bow_testModel.predict_proba(x_test_onehot_bow)[:,1]

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


# 
# ## ->-> 8.1.3: <font color='red'> SET 1</font> Confusion Matrix

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


# # -> 8.2:<font color='red'> SET 2</font>  Applying Logistic Regression on TFIDF.

# 
# ## ->-> 8.2.1: <font color='red'> SET 2</font> Hyper parameter tuning to find best 'C' using GRIDSEARCHCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
logreg_tfidf = LogisticRegression(class_weight='balanced')
# parameters = {'lambda':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
parameters = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}
clf = GridSearchCV(logreg_tfidf, parameters, cv= 3, scoring='roc_auc',verbose=1,return_train_score=True,n_jobs=-1)
# clf.fit(x_cv_onehot_bow, y_cv)
clf.fit(x_train_onehot_tfidf,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestC_2=clf.best_params_['C']
bestScore_2=clf.best_score_
print("BEST C: ",clf.best_params_['C']," BEST SCORE: ",clf.best_score_) #clf.best_estimator_.alpha


# In[ ]:


# alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

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


# 
# ## ->-> 8.2.2: <font color='red'> SET 2</font> TESTING the performance of the model on test data, plotting ROC Curves.

# In[ ]:


#https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
logref_tfidf_testModel = LogisticRegression(class_weight='balanced',C=bestC_2)
logref_tfidf_testModel.fit(x_train_onehot_tfidf, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs
# y_train_pred = batch_predict(mnb_tfidf_testModel, x_train_onehot_tfidf)
# y_test_pred = batch_predict(mnb_tfidf_testModel, x_test_onehot_tfidf)
y_train_pred=logref_tfidf_testModel.predict_proba(x_train_onehot_tfidf)[:,1]
y_test_pred=logref_tfidf_testModel.predict_proba(x_test_onehot_tfidf)[:,1]

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
print(type(y_test_pred))
print(type(tr_thresholds))
print(type(test_fpr))
print(type(test_tpr))
print(y_test_pred.shape)
print(tr_thresholds.shape)
print(test_fpr.shape)
print(test_tpr.shape)


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


# 
# ## ->-> 8.2.3: <font color='red'> SET 2</font> Confusion Matrix

# ### ->->-> 8.2.3.1: <font color='red'> SET 2</font> Confusion Matrix: Train

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


# # -> 8.3:<font color='red'> SET 3</font>  Applying Logistic Regression on AvgW2V.

# 
# ## ->-> 8.3.1: <font color='red'> SET 3</font> Hyper parameter tuning to find best 'C' using GRIDSEARCHCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
logreg_avgw2v = LogisticRegression(class_weight='balanced')
# parameters = {'lambda':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
parameters = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}
clf = GridSearchCV(logreg_avgw2v, parameters, cv= 3, scoring='roc_auc',verbose=1,return_train_score=True,n_jobs=-1)
# clf.fit(x_cv_onehot_bow, y_cv)
clf.fit(x_train_onehot_avg_w2v,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestC_3=clf.best_params_['C']
bestScore_3=clf.best_score_
print("BEST C: ",clf.best_params_['C']," BEST SCORE: ",clf.best_score_) #clf.best_estimator_.alpha


# In[ ]:


# alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
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


# 
# ## ->-> 8.3.2: <font color='red'> SET 3</font> TESTING the performance of the model on test data, plotting ROC Curves.

# In[ ]:


#https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
logref_avgw2v_testModel = LogisticRegression(class_weight='balanced',C=bestC_3)
logref_avgw2v_testModel.fit(x_train_onehot_avg_w2v, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs
# y_train_pred = batch_predict(mnb_tfidf_testModel, x_train_onehot_tfidf)
# y_test_pred = batch_predict(mnb_tfidf_testModel, x_test_onehot_tfidf)
y_train_pred=logref_avgw2v_testModel.predict_proba(x_train_onehot_avg_w2v)[:,1]
y_test_pred=logref_avgw2v_testModel.predict_proba(x_test_onehot_avg_w2v)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

ax = plt.subplot()

auc_set3_train=auc(train_fpr, train_tpr)
auc_set3_test=auc(test_fpr, test_tpr)

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
print(type(y_test_pred))
print(type(tr_thresholds))
print(type(test_fpr))
print(type(test_tpr))
print(y_test_pred.shape)
print(tr_thresholds.shape)
print(test_fpr.shape)
print(test_tpr.shape)


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


# 
# ## ->-> 8.3.3: <font color='red'> SET 3</font> Confusion Matrix

# ### ->->-> 8.3.3.1: <font color='red'> SET 3</font> Confusion Matrix: Train

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


# ### ->->-> 8.3.3.2: <font color='red'> SET 3</font> Confusion Matrix: Test

# In[ ]:


## TEST
print("="*100)
print("Test confusion matrix")
print(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)))

conf_matr_df_test = pd.DataFrame(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_matr_df_test, annot=True,annot_kws={"size": 16}, fmt='g')


# # -> 8.4:<font color='red'> SET 4</font>  Applying Logistic Regression on TFIDF_W2V.

# 
# ## ->-> 8.4.1: <font color='red'> SET 4</font> Hyper parameter tuning to find best 'C' using GRIDSEARCHCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
logreg_tfidf_w2v = LogisticRegression(class_weight='balanced')
# parameters = {'lambda':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
parameters = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}
clf = GridSearchCV(logreg_tfidf_w2v, parameters, cv= 3, scoring='roc_auc',verbose=1,return_train_score=True,n_jobs=-1)
# clf.fit(x_cv_onehot_bow, y_cv)
clf.fit(x_train_onehot_tfidf_w2v,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestC_4=clf.best_params_['C']
bestScore_4=clf.best_score_
print("BEST C: ",clf.best_params_['C']," BEST SCORE: ",clf.best_score_) #clf.best_estimator_.alpha


# In[ ]:


# alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
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


# 
# ## ->-> 8.4.2: <font color='red'> SET 4</font> TESTING the performance of the model on test data, plotting ROC Curves.

# In[ ]:


#https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
logref_tfidf_w2v_testModel = LogisticRegression(class_weight='balanced',C=bestC_4)
logref_tfidf_w2v_testModel.fit(x_train_onehot_tfidf_w2v, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs
# y_train_pred = batch_predict(mnb_tfidf_testModel, x_train_onehot_tfidf)
# y_test_pred = batch_predict(mnb_tfidf_testModel, x_test_onehot_tfidf)
y_train_pred=logref_tfidf_w2v_testModel.predict_proba(x_train_onehot_tfidf_w2v)[:,1]
y_test_pred=logref_tfidf_w2v_testModel.predict_proba(x_test_onehot_tfidf_w2v)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

ax = plt.subplot()

auc_set4_train=auc(train_fpr, train_tpr)
auc_set4_test=auc(test_fpr, test_tpr)

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
print(type(y_test_pred))
print(type(tr_thresholds))
print(type(test_fpr))
print(type(test_tpr))
print(y_test_pred.shape)
print(tr_thresholds.shape)
print(test_fpr.shape)
print(test_tpr.shape)


# In[ ]:


print(type(y_test_pred))
print(type(tr_thresholds))
print(type(test_fpr))
print(type(test_tpr))
print(y_test_pred.shape)
print(tr_thresholds.shape)
print(test_fpr.shape)
print(test_tpr.shape)


# In[ ]:


def predict(proba, threshould, fpr, tpr):
    print("--------------------->",threshould)
    t = threshould[np.argmax(tpr*(1-fpr))]
    print(type(proba))
    print(type(threshould))
    print(type(fpr))
    print(type(tpr))
    
    print("Shape",proba.shape)
    print("Shape",threshould.shape)
    print("Shape",fpr.shape)
    print("Shape",tpr.shape)
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    predictions = []
    
    for i in proba:
        if i>=t:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# 
# ## ->-> 8.4.3: <font color='red'> SET 4</font> Confusion Matrix

# ### ->->-> 8.4.3.1: <font color='red'> SET 4</font> Confusion Matrix: Train

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


# ### ->->-> 8.4.3.2: <font color='red'> SET 4</font> Confusion Matrix: Test

# In[ ]:


print(type(y_test_pred))
print(type(tr_thresholds))
print(type(test_fpr))
print(type(test_tpr))
print(y_test_pred.shape)
print(tr_thresholds.shape)
print(test_fpr.shape)
print(test_tpr.shape)


# In[ ]:


## TEST
print("="*100)
print("Test confusion matrix")
print(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)))

conf_matr_df_test = pd.DataFrame(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_matr_df_test, annot=True,annot_kws={"size": 16}, fmt='g')


# # -> 8.5:<font color='red'> SET 5</font>  Applying Logistic Regression on ALL EXCEPT TEXT.

# # Performing sentiment analysis

# In[ ]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

essays = x_train['essay']
essays_sentiments = []

for essay in tqdm(essays):
    res = sid.polarity_scores(essay)
    essays_sentiments.append(res['compound']) #Considering compound as a criteria.

x_train['essay_sentiment_train'] = essays_sentiments


# In[ ]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

essays_test = x_test['essay']
essays_sentiments = []

for essay in tqdm(essays_test):
    res = sid.polarity_scores(essay)
    essays_sentiments.append(res['compound']) #Considering compound as a criteria.

x_test['essay_sentiment_test'] = essays_sentiments


# In[ ]:


sentiment_test=x_test['essay_sentiment_test'].values.reshape(-1,1)
sentiment_train=x_train['essay_sentiment_train'].values.reshape(-1,1)


# ## Merging all the non text features.

# In[ ]:


from scipy.sparse import hstack

x_train_onehot_noText = hstack((x_train_school_state_one_hot, x_train_categories_one_hot, x_train_sub_categories_one_hot, x_train_grade_category_one_hot, x_train_prefix_one_hot,x_train_quantity_normalized,x_train_teacher_previous_proj_normalized, x_train_price_normalized, sentiment_train,x_train_title_normalized,x_train_essay_normalized))
# x_cv_onehot    = hstack((x_cv_categories_one_hot, x_cv_sub_categories_one_hot, x_cv_school_state_one_hot, x_cv_prefix_one_hot, x_cv_grade_category_one_hot,x_cv_price_normalized, x_cv_teacher_previous_proj_normalized ))
x_test_onehot_noText  = hstack((x_test_school_state_one_hot, x_test_categories_one_hot, x_test_sub_categories_one_hot, x_test_grade_category_one_hot, x_test_prefix_one_hot,x_test_quantity_normalized,x_test_teacher_previous_proj_normalized, x_test_price_normalized, sentiment_test,x_test_title_normalized,x_test_essay_normalized))

print("Type -> One Hot -> x_train: ",type(x_train_onehot_noText))
print("Type -> One Hot -> x_test : ",type(x_test_onehot_noText))
# print("Type -> One Hot -> x_cv        : ",type(x_cv_onehot))
print("\n")
print("Shape -> One Hot -> x_train: ",x_train_onehot_noText.shape)
print("Shape -> One Hot -> x_test : ",x_test_onehot_noText.shape)
# print("Shape -> One Hot -> x_cv         : ",x_cv_onehot.shape)


# 
# ## ->-> 8.5.1: <font color='red'> SET 5</font> Hyper parameter tuning to find best 'C' using GRIDSEARCHCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
logreg_noText = LogisticRegression(class_weight='balanced')
# parameters = {'lambda':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
parameters = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}
clf = GridSearchCV(logreg_noText, parameters, cv= 3, scoring='roc_auc',verbose=1,return_train_score=True,n_jobs=-1)
# clf.fit(x_cv_onehot_bow, y_cv)
clf.fit(x_train_onehot_noText,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestC_5=clf.best_params_['C']
bestScore_5=clf.best_score_
print("BEST C: ",clf.best_params_['C']," BEST SCORE: ",clf.best_score_) #clf.best_estimator_.alpha


# In[ ]:


# alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
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


# 
# ## ->-> 8.5.2: <font color='red'> SET 5</font> TESTING the performance of the model on test data, plotting ROC Curves.

# In[ ]:


#https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
logref_noText_testModel = LogisticRegression(class_weight='balanced',C=bestC_5)
logref_noText_testModel.fit(x_train_onehot_noText, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs
# y_train_pred = batch_predict(mnb_tfidf_testModel, x_train_onehot_tfidf)
# y_test_pred = batch_predict(mnb_tfidf_testModel, x_test_onehot_tfidf)
y_train_pred=logref_noText_testModel.predict_proba(x_train_onehot_noText)[:,1]
y_test_pred=logref_noText_testModel.predict_proba(x_test_onehot_noText)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

ax = plt.subplot()

auc_set5_train=auc(train_fpr, train_tpr)
auc_set5_test=auc(test_fpr, test_tpr)

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


# 
# ## ->-> 8.5.3: <font color='red'> SET 5</font> Confusion Matrix

# ### ->->-> 8.5.3.1: <font color='red'> SET 5</font> Confusion Matrix: Train

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


# ### ->->-> 8.5.3.2: <font color='red'> SET 5</font> Confusion Matrix: Test

# In[ ]:


## TEST
print("="*100)
print("Test confusion matrix")
print(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)))

conf_matr_df_test = pd.DataFrame(confusion_matrix(y_test, predict(y_test_pred, tr_thresholds, test_fpr, test_tpr)), range(2),range(2))
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_matr_df_test, annot=True,annot_kws={"size": 16}, fmt='g')


# # 9. CONCLUSION

# In[ ]:


from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["Vectorizer", "Model", "Hyperparameter: C", "Train AUC", "Test AUC"]
# auc_set2_train=auc(train_fpr, train_tpr)
# auc_set2_test=auc(test_fpr, test_tpr)

x.add_row(["BOW", "Logistic Regression", bestC_1, round(auc_set1_train,2),round(auc_set1_test,2)])
x.add_row(["TF-IDF", "Logistic Regression", bestC_2, round(auc_set2_train,2),round(auc_set2_test,2)])
x.add_row(["Avg W2V", "Logistic Regression", bestC_3, round(auc_set3_train,2),round(auc_set3_test,2)])
x.add_row(["TFIDF W2V", "Logistic Regression", bestC_4, round(auc_set4_train,2),round(auc_set4_test,2)])
x.add_row(["All non Text", "Logistic Regression", bestC_5, round(auc_set5_train,2),round(auc_set5_test,2)])

print(x)


# As we can observe, the Set 5 which had all non text features, has performed sufficiently well in Test AUC and the lowest in Train AUC. Since is a good indication that the model is not overfitted and has performed in a balanced way in Training and Testing. The gap between the Training AUC and Testing AUC is very minimal which is good sign of performance. Thus this is an interesting observation as can infer that we can obtain good result in Testing even without including the text data.
