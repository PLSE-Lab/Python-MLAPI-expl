#!/usr/bin/env python
# coding: utf-8

# # DonorsChoose Dataset: EDA and TNSE

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
# <li>__project_essay_4:__ "Close by sharing why your project will make a difference"</li>
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
from sklearn.manifold import TSNE
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

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
import time

from plotly import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter
# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
import scipy
print("DONE LOADING-------")


# ## 1.1 Reading Data

# In[ ]:


project_data = pd.read_csv('../input/train_data.csv')
resource_data = pd.read_csv('../input/resources.csv')


# In[ ]:


# Taking radom samples for less memory machines
# -> https://www.geeksforgeeks.org/python-pandas-dataframe-sample/
# -> https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
# project_data= project_data1.sample(n = 5000) 
# project_data= project_data1.sample(n = 5000) 


# In[ ]:


print("Number of data points in train data", project_data.shape)
print('-'*50)
print("The attributes of data :", project_data.columns.values)


# In[ ]:


print("Number of data points in train data", resource_data.shape)
print(resource_data.columns.values)
resource_data.head(2)


# # 1.2 Data Analysis

# ### As it is clearly metioned in the dataset details that TEACHER_PREFIX has NaN values, we need to handle this at the very beginning to avoid any problems in our future analysis

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


# PROVIDE CITATIONS TO YOUR CODE IF YOU TAKE IT FROM ANOTHER WEBSITE.
# https://matplotlib.org/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py


y_value_counts = project_data['project_is_approved'].value_counts()
print("Number of projects thar are approved for funding ", y_value_counts[1], ", (", (y_value_counts[1]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")
print("Number of projects thar are not approved for funding ", y_value_counts[0], ", (", (y_value_counts[0]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
recipe = ["Accepted", "Not Accepted"]

data = [y_value_counts[1], y_value_counts[0]]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                 horizontalalignment=horizontalalignment, **kw)

ax.set_title("Nmber of projects that are Accepted and not accepted")

plt.show()


# ### 1.2.1 Univariate Analysis: School State

# In[ ]:


# Pandas dataframe groupby count, mean: https://stackoverflow.com/a/19385591/4084039

temp = pd.DataFrame(project_data.groupby("school_state")["project_is_approved"].apply(np.mean)).reset_index()
# if you have data which contain only 0 and 1, then the mean = percentage (think about it)
temp.columns = ['state_code', 'num_proposals']

'''# How to plot US state heatmap: https://datascience.stackexchange.com/a/9620

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = temp['state_code'],
        z = temp['num_proposals'].astype(float),
        locationmode = 'USA-states',
        text = temp['state_code'],
        marker = dict(line = dict (color = 'rgb(255,255,255)',width = 2)),
        colorbar = dict(title = "% of pro")
    ) ]

layout = dict(
        title = 'Project Proposals % of Acceptance Rate by US States',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename='us-map-heat-map')
'''


# In[ ]:


# https://www.csi.cuny.edu/sites/default/files/pdf/administration/ops/2letterstabbrev.pdf
temp.sort_values(by=['num_proposals'], inplace=True)
print("States with lowest % approvals")
print(temp.head(5))
print('='*50)
print("States with highest % approvals")
print(temp.tail(5))


# **Observation:**
#  1. Every state has greater than 80% success rate in approval.
#  1. DE (Delaware) has the Maximimum approval rate of 89.79 %
#  1. VT (Vermont) has the Minimum approval rate of 80% then followed by DC (District of Columbia)
#     

# In[ ]:


#stacked bar plots matplotlib: https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html
def stack_plot(data, xtick, col2='project_is_approved', col3='total'):
    ind = np.arange(data.shape[0])
    
    plt.figure(figsize=(20,5))
    p1 = plt.bar(ind, data[col3].values)
    p2 = plt.bar(ind, data[col2].values)

    plt.ylabel('Projects')
    plt.title('Number of projects aproved vs rejected')
    plt.xticks(ind, list(data[xtick].values))
    plt.legend((p1[0], p2[0]), ('total', 'accepted'))
    plt.show()


# In[ ]:


def univariate_barplots(data, col1, col2='project_is_approved', top=False):
    # Count number of zeros in dataframe python: https://stackoverflow.com/a/51540521/4084039
    temp = pd.DataFrame(project_data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index()

    # Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039
    temp['total'] = pd.DataFrame(project_data.groupby(col1)[col2].agg({'total':'count'})).reset_index()['total']
    temp['Avg'] = pd.DataFrame(project_data.groupby(col1)[col2].agg({'Avg':'mean'})).reset_index()['Avg']
    
    temp.sort_values(by=['total'],inplace=True, ascending=False)
    
    if top:
        temp = temp[0:top]
    
    stack_plot(temp, xtick=col1, col2=col2, col3='total')
    print(temp.head(5))
    print("="*50)
    print(temp.tail(5))


# In[ ]:


univariate_barplots(project_data, 'school_state', 'project_is_approved', False)


# **Observation:**
#  1. Every state has greater than 80% success rate in approval.
#  1. CA (California) state has the maximum number of the project proposals ie 15387.
#  1. There is 50% reduction in the 2nd highest state for project proposals. ie TX (Texas) with 6014.
#  1. VT (Vermont) has the minimum number of project submissions.
#  1. High variation is observed in the state statistics.

# ### 1.2.2 Univariate Analysis: teacher_prefix

# In[ ]:


univariate_barplots(project_data, 'teacher_prefix', 'project_is_approved' , top=False)


# **Observation:**
#  1. Female teachers have proposed more number of projects than Male teachers.
#  1. Approval rate of Married Female teacher with prefix Mrs. is higher than Male Teachers
#  1. Teacher having Dr. prefix has proposed very less projects
#  1. Interstingly teachers with the highest qualification ie Dr. have lesser approval rate.

# ### 1.2.3 Univariate Analysis: project_grade_category

# In[ ]:


univariate_barplots(project_data, 'project_grade_category', 'project_is_approved', top=False)


# **Observation:**
# 1. Grades PreK-2 and 3-5 have large number of project proposals.
# 1. Grades 3-5 has the highest approval rate then followed by Grade PreK-2
# 1. Number of project proposals reduces 4 times from Lower most grade to highest grade. That is, number of project proposal reduces as the grades increases.

# ### 1.2.4 Univariate Analysis: project_subject_categories

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


# In[ ]:


project_data['clean_categories'] = cat_list
project_data.drop(['project_subject_categories'], axis=1, inplace=True)
project_data.head(2)


# In[ ]:


univariate_barplots(project_data, 'clean_categories', 'project_is_approved', top=20)


# **Observation:**
# 1. Literacy_Language is the Single most category having highest approval rate of 86%.
# 1. If Literacy_Language is clubed with History_Civics, the approval rate rises from 86% to 89%.
# 1. Math_Science alone has approval rate of around 82%, when clubbed with Literacy_Language, approval rate rises to 86%;  when clubbed with AppliedLearning, rises to ~84%, but when clubbed with AppliedLearning, the approval rate reduced to 81%.
# 1. Warmth and  Care_Hunger has highest approval rate of ~93%. Thus from this we can conclude that, Non-Science categories have higher approval rate than compared to Science categories including Maths.

# In[ ]:


# count of all the words in corpus python: https://stackoverflow.com/a/22898595/4084039
from collections import Counter
my_counter = Counter()
for word in project_data['clean_categories'].values:
    my_counter.update(word.split())


# In[ ]:


# dict sort by value python: https://stackoverflow.com/a/613218/4084039
cat_dict = dict(my_counter)
sorted_cat_dict = dict(sorted(cat_dict.items(), key=lambda kv: kv[1]))


ind = np.arange(len(sorted_cat_dict))
plt.figure(figsize=(20,5))
p1 = plt.bar(ind, list(sorted_cat_dict.values()))

plt.ylabel('Projects')
plt.title('% of projects aproved category wise')
plt.xticks(ind, list(sorted_cat_dict.keys()))
plt.show()


# In[ ]:


for i, j in sorted_cat_dict.items():
    print("{:20} :{:10}".format(i,j))


# **Observation:**
# 1. Literacy_Language has the highest number of project than followed by Math_Science.
# 1. Warmth and Care Hunger has the least number of projects.
# 1. Interestingly, from the previous plot, we observed that Warmth and Care_Hunger when clubbed together has the highest approval rate.
# 1. There is high variance between the 2nd and 3rd  most number of project categories counts. ie Math_Science (41419) and Health_Sports (14223)

# ### 1.2.5 Univariate Analysis: project_subject_subcategories

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


# In[ ]:


project_data['clean_subcategories'] = sub_cat_list
project_data.drop(['project_subject_subcategories'], axis=1, inplace=True)
project_data.head(2)


# In[ ]:


univariate_barplots(project_data, 'clean_subcategories', 'project_is_approved', top=50)


# **Observation:**
# 1. Projects with sub-category Literacy has the highest number of projects.
# 1. Also Literacy has the highest approval rate of 88%.
# 1. Interestingly, when Literacy is clubbed with any other sub-category, the approval rate is reduced.
# 1. Mathematics alone has lower approval rate than compared to Mathematics clubbed with any other category.
# 1. AppliedSciences College_CareerPrep has least number of project posted and also the with least approval rate.

# In[ ]:


# count of all the words in corpus python: https://stackoverflow.com/a/22898595/4084039
from collections import Counter
my_counter = Counter()
for word in project_data['clean_subcategories'].values:
    my_counter.update(word.split())


# In[ ]:


# dict sort by value python: https://stackoverflow.com/a/613218/4084039
sub_cat_dict = dict(my_counter)
sorted_sub_cat_dict = dict(sorted(sub_cat_dict.items(), key=lambda kv: kv[1]))


ind = np.arange(len(sorted_sub_cat_dict))
plt.figure(figsize=(20,5))
p1 = plt.bar(ind, list(sorted_sub_cat_dict.values()))

plt.ylabel('Projects')
plt.title('% of projects aproved state wise')
plt.xticks(ind, list(sorted_sub_cat_dict.keys()))
plt.show()


# In[ ]:


for i, j in sorted_sub_cat_dict.items():
    print("{:20} :{:10}".format(i,j))


# **Observation:**
# 1. Literacy has the highest approved projects.
# 1. Economics has the least approved projects.

# ### 1.2.6 Univariate Analysis: Text features (Title)

# In[ ]:


#How to calculate number of words in a string in DataFrame: https://stackoverflow.com/a/37483537/4084039
word_count = project_data['project_title'].str.split().apply(len).value_counts()
word_dict = dict(word_count)
word_dict = dict(sorted(word_dict.items(), key=lambda kv: kv[1]))


ind = np.arange(len(word_dict))
plt.figure(figsize=(20,5))
p1 = plt.bar(ind, list(word_dict.values()))

plt.ylabel('Numeber of projects')
plt.xlabel('Numeber words in project title')
plt.title('Words for each title of the project')
plt.xticks(ind, list(word_dict.keys()))
plt.show()


# **Observation:**
# 1. Maximum projects have 4 words, then followed by 5 and 3.
# 1. There are extremely less number of projects that have titles of 1 word and > 10 words.

# In[ ]:


approved_title_word_count = project_data[project_data['project_is_approved']==1]['project_title'].str.split().apply(len)
approved_title_word_count = approved_title_word_count.values

rejected_title_word_count = project_data[project_data['project_is_approved']==0]['project_title'].str.split().apply(len)
rejected_title_word_count = rejected_title_word_count.values


# In[ ]:


# https://glowingpython.blogspot.com/2012/09/boxplot-with-matplotlib.html
plt.boxplot([approved_title_word_count, rejected_title_word_count])
plt.xticks([1,2],('Approved Projects','Rejected Projects'))
plt.ylabel('Words in project title')
plt.grid()
plt.show()


# **Observation:**
# 1. In Approved Projects, 25th Quartile lies at 4 words. Median at 5 words and 75th Quartile at 7 words.
# 1. In Rejected Projects, 25th Quartile lies at 3 words. Median at 5 words and 75th Quartile at 6 words.
# 1. In Approved Projects, the gap between the  Median and 75th Quartile is large, where as exactly inverse scenario of a larger gap between 25th and Median is observed in Rejected Projects.
# 1. In Approved Projects, titles with more than 11 words are considered as outliers, where as in Rejected Projects, titles with more than 10 words are considered as outliers.

# In[ ]:


plt.figure(figsize=(10,3))
sns.kdeplot(approved_title_word_count,label="Approved Projects", bw=0.6)
sns.kdeplot(rejected_title_word_count,label="Not Approved Projects", bw=0.6)
plt.legend()
plt.show()


# **Observation:**
# 1. The number of Approved Projects have a slightly more words in the Title when compared to the Rejected Projects.

# ### 1.2.7 Univariate Analysis: Text features (Project Essay's)

# In[ ]:


# merge two column text dataframe: 
project_data["essay"] = project_data["project_essay_1"].map(str) +                        project_data["project_essay_2"].map(str) +                         project_data["project_essay_3"].map(str) +                         project_data["project_essay_4"].map(str)


# In[ ]:


approved_word_count = project_data[project_data['project_is_approved']==1]['essay'].str.split().apply(len)
approved_word_count = approved_word_count.values

rejected_word_count = project_data[project_data['project_is_approved']==0]['essay'].str.split().apply(len)
rejected_word_count = rejected_word_count.values


# In[ ]:


# https://glowingpython.blogspot.com/2012/09/boxplot-with-matplotlib.html
plt.boxplot([approved_word_count, rejected_word_count])
plt.title('Words for each essay of the project')
plt.xticks([1,2],('Approved Projects','Rejected Projects'))
plt.ylabel('Words in project essays')
plt.grid()
plt.show()


# 
# **Observation:**
# 1. The Median of the Approved Projects is slightly higher than the Median of the Rejected Projects, we can infer that, Approved Projects have higher number of words in the essay.
# 1. Sufficiently large essays are considered as outlier in both Approved and Rejected Projects.

# In[ ]:


plt.figure(figsize=(10,3))
sns.distplot(approved_word_count, hist=False, label="Approved Projects")
sns.distplot(rejected_word_count, hist=False, label="Not Approved Projects")
plt.title('Words for each essay of the project')
plt.xlabel('Number of words in each eassay')
plt.legend()
plt.show()


# **Observation:**
# 1. The PDF of the approved projects is denser for words around 240 to around 470. From this, we can again say that, Approved Projects have higher number of words in the essay.

# ### 1.2.8 Univariate Analysis: Cost per project

# In[ ]:


# we get the cost of the project using resource.csv file
resource_data.head(2)


# In[ ]:


# https://stackoverflow.com/questions/22407798/how-to-reset-a-dataframes-indexes-for-all-groups-in-one-step
price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
price_data.head(2)


# In[ ]:


# join two dataframes in python: 
project_data = pd.merge(project_data, price_data, on='id', how='left')


# In[ ]:


approved_price = project_data[project_data['project_is_approved']==1]['price'].values

rejected_price = project_data[project_data['project_is_approved']==0]['price'].values


# In[ ]:


# https://glowingpython.blogspot.com/2012/09/boxplot-with-matplotlib.html
plt.boxplot([approved_price, rejected_price])
plt.title('Box Plots of Cost per approved and not approved Projects')
plt.xticks([1,2],('Approved Projects','Rejected Projects'))
plt.ylabel('Price')
plt.grid()
plt.show()


# **Observation:**
# 1. The box plots for Price seems very much identical for Approved and Rejected Projects considering the 25th, 50th and 75th Quartiles.
# 1. Interestingly, the Minimum value of the both the box plots for price is very close to 0, and maximum roughly at 800.
# 1. Both the box plots for price have considered higher price as outliers.

# In[ ]:


plt.figure(figsize=(10,3))
sns.distplot(approved_price, hist=False, label="Approved Projects")
sns.distplot(rejected_price, hist=False, label="Not Approved Projects")
plt.title('Cost per approved and not approved Projects')
plt.xlabel('Cost of a project')
plt.legend()
plt.show()


# **Observation:**
# 1. The PDFs of Price are very much similar and overlapping. Thus not much can be understood.
# 1. To some extent, from the PDFs of Price, we can say Projects with higher price are generally not approved.

# In[ ]:


# http://zetcode.com/python/prettytable/
from prettytable import PrettyTable

#If you get a ModuleNotFoundError error , install prettytable using: pip3 install prettytable

x = PrettyTable()
x.field_names = ["Percentile", "Approved Projects", "Not Approved Projects"]

for i in range(0,101,5):
    x.add_row([i,np.round(np.percentile(approved_price,i), 3), np.round(np.percentile(rejected_price,i), 3)])
print(x)


# ### 1.2.9 Univariate Analysis: teacher_number_of_previously_posted_projects

# ## PDF FOR TEACHERS WITH PREVIOUS PROJECTS

# In[ ]:


plt.figure(figsize=(10,3))
sns.distplot(project_data["teacher_number_of_previously_posted_projects"], hist=False, label="Previous Projects")
plt.title('PDF For Teacher with previous projects')
plt.xlabel('Number of a previous project')
plt.legend()
plt.show()


# **Observation:**
# 1. It may not be , but at first glance, the distribution of the Teacher's number of previous project seems to look at LOG DISTRIBUTION.

# ## BAR PLOTS FOR TEACHERS WITH PREVIOUS PROJETCS

# In[ ]:


univariate_barplots(project_data, 'teacher_number_of_previously_posted_projects', 'project_is_approved', False)


# **Observation:**
# 
# Minimum Previous Project  : 0 - - - - - -Count: 30014   
# Maximum Previous Projects : 451 - - - Count: 1

# #### Lets take a close look at the pattern of the top 100

# ## HISTOGRAM FOR TEACHERS WITH PREVIOUS PROJECTS

# In[ ]:


#How to calculate number of words in a string in DataFrame: https://stackoverflow.com/a/37483537/4084039

word_count = project_data["teacher_number_of_previously_posted_projects"].value_counts()
word_count=word_count[:100]
ind = np.arange(len(word_count))
plt.figure(figsize=(20,5))
p1 = plt.bar(ind, list(word_count))

plt.ylabel('Count Of Teachers')
plt.xlabel('Number of previous projects')
plt.title('Words for each title of the project')

plt.show()


# **Observation:**
#  1. The count of teachers with more previous project reduces sharply.
#     

# To better understand the Top 5 result from the above plot, we calculate more information about the top 10 results as below:

# ## MAXIMUM COUNT ANALYSIS

# In[ ]:


##nikhil
total=project_data.shape[0]
counts=project_data["teacher_number_of_previously_posted_projects"].value_counts()
zeros=counts[0]
zeroProjectApprovedCount=0
for i in range(0,total):
    if project_data["teacher_number_of_previously_posted_projects"][i]==0 and  project_data["project_is_approved"][i]==1:
        zeroProjectApprovedCount+=1
print("Teacher with 0 previous projects:",zeros,"out of:",total," ie. ",round(zeros/total*100,2),"%")
print("Accepted: ", zeroProjectApprovedCount," ie. ",round(zeroProjectApprovedCount/zeros*100,2),"%")
print("Rejected: ", zeros-zeroProjectApprovedCount," ie. ",round((zeros-zeroProjectApprovedCount)/zeros*100,2),"%")
print("-"*90)


# **Observation:**
# Roughly 28% of the teachers have applied for the 1st time. Interestingly, 82% of those are accepted considering that they dont have any previous applications.

# ## TOP 10 ANALYSIS

# In[ ]:


##nikhil
counts=project_data["teacher_number_of_previously_posted_projects"].value_counts()
for j in range(0,11):

    zeros=counts[j]
    zeroProjectApprovedCount=0
    for i in range(0,total):
        if project_data["teacher_number_of_previously_posted_projects"][i]==j and  project_data["project_is_approved"][i]==1:
            zeroProjectApprovedCount+=1
    print("Teacher with ",j," previous projects:",zeros,"out of:",total," ie. ",round(zeros/total*100,2),"%")
    print("Accepted: ", zeroProjectApprovedCount," ie. ",round(zeroProjectApprovedCount/zeros*100,2),"%")
    print("Rejected: ", zeros-zeroProjectApprovedCount," ie. ",round((zeros-zeroProjectApprovedCount)/zeros*100,2),"%")
    print("-"*90)


# ## BOXPLOT FOR TEACHERS WITH PREVIOUS PROJECTS

# In[ ]:


## BOX PLOT
plt.boxplot(project_data["teacher_number_of_previously_posted_projects"])
plt.title('Box Plots of Number of previous projects of teachers')

plt.ylabel('Number of Projects')
plt.grid()
plt.show()


# **Observation:**
# 1. We can observe from the box plot of previous projects, that the Median extremely close to 0
# 2. The 75th Percentile of the data is roughly near 20.
# 3. A large section of the data is beyond the 75th Percentile, hence considered as outlier by the plot.

# ## HISTOGRAM WITH PDF

# In[ ]:


## HISTOGRAM WITH PDF
sns.FacetGrid(project_data,hue="project_is_approved",height=7)    .map(sns.distplot,"teacher_number_of_previously_posted_projects")    .add_legend();

plt.show();


# **Observation**:
# 1. The PDF of the Approved -> teacher_number_of_previously_posted_projects, is slightly forward than compared to the PDF of Rejected, thus this gives a slight inference that, projects whose Teacher have previous projects will have a slighly approval rate higher than those who didnt.
# 1. Teachers who had roughly 12 previous project were highly approved. ie mean seems to be around 12.
# 
# 

# ## PDF - CDF

# In[ ]:


## PDF - CDF

plt.figure(figsize=(20,6))
plt.subplot(131) ##(1=no. of rows, 3= no. of columns, 1=1st figure,2,3,4 boxes)
counts,bin_edges=np.histogram(project_data["teacher_number_of_previously_posted_projects"],bins=15,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,linewidth=3.0)
plt.plot(bin_edges[1:],cdf,linewidth=3.0)
plt.ylabel("COUNT")
plt.xlabel('Previous Projects')
plt.title('PDF-CDF of Teachers with their total previously posted projects')
plt.legend(['PDF-Prev. Projects', 'CDF-Prev. Projects'], loc = 5,prop={'size': 16})


# **Observation**:
# 1. From the CDF, we can say that teacher had Maximum project between 0-100.
# 2. After roughly 100 project counts, the CDF/PDF seems stable
#     

# ### 1.2.10 Univariate Analysis: project_resource_summary

# Please do this on your own based on the data analysis that was done in the above cells
# 
# Check if the `presence of the numerical digits` in the `project_resource_summary` effects the acceptance of the project or not. If you observe that `presence of the numerical digits` is helpful in the classification, please include it for further process or you can ignore it.

# ## BAR GRAPH FOR WORDS IN SUMMARY

# In[ ]:


#How to calculate number of words in a string in DataFrame: https://stackoverflow.com/a/37483537/4084039
proj_res_summ_word_count = project_data['project_resource_summary'].str.split().apply(len).value_counts()
proj_res_summ_word_dict = dict(proj_res_summ_word_count)
proj_res_summ_word_dict = dict(sorted(proj_res_summ_word_dict.items(), key=lambda kv: kv[1]))


ind = np.arange(len(proj_res_summ_word_dict))
plt.figure(figsize=(20,7))
p1 = plt.bar(ind, list(proj_res_summ_word_dict.values()))

plt.ylabel('Number of projects')
plt.xlabel('Number words in project resource summary')
plt.title('Words for each project resource summary')
plt.xticks(ind, list(proj_res_summ_word_dict.keys()))
plt.show()


# **Observation**
# 1. A Maximum of around 11,000 summaries is composed of 11 words.
# 2. 2000 and above summaries is comprised of words ranging between 11 - 31.

# ## PDF FOR SUMMARY

# In[ ]:


approved_proj_resource_summary_word_count = project_data[project_data['project_is_approved']==1]['project_resource_summary'].str.split().apply(len)
approved_proj_resource_summary_word_count = approved_proj_resource_summary_word_count.values

rejected_proj_resource_summary_word_count = project_data[project_data['project_is_approved']==0]['project_resource_summary'].str.split().apply(len)
rejected_proj_resource_summary_word_count = rejected_proj_resource_summary_word_count.values


# In[ ]:


plt.figure(figsize=(20,7))

sns.distplot(approved_proj_resource_summary_word_count, hist=False, label="Approved Projects")
sns.distplot(rejected_proj_resource_summary_word_count, hist=False, label="Not Approved Projects")
plt.title('Words for each project resource summary')
plt.xlabel('Number of words in each project resource summary')
plt.legend()
plt.show()

# # ALTERNATE CODE FOR KDE
# plt.figure(figsize=(10,3))
# sns.kdeplot(approved_proj_resource_summary_word_count,label="Approved Projects", bw=0.6)
# sns.kdeplot(rejected_proj_resource_summary_word_count,label="Not Approved Projects", bw=0.6)
# plt.legend()
# plt.show()


# **Observation**
# 1. Interestingly we can observe that, if the summary word count is NOMINALLY large then Approval rate is higher. However beyond nominal count, ie extremely large summaries have lower chance of approval.
# 2. Maximum project with summary having words between around 11 to 20 are approved.
# 3. Number of Summaries with word counts in the range of around 20-30 is less than compared to 30-40
# 4. Projects with summary having words less than roughly 11 and more than 40 are rejected.

# ## BOX PLOT FOR SUMMARY

# In[ ]:




## BOX PLOT
plt.boxplot([approved_proj_resource_summary_word_count, rejected_proj_resource_summary_word_count])
plt.title('Words for each essay of the project')
plt.xticks([1,2],('Approved Projects','Rejected Projects'))
plt.ylabel('Words in project essays')
plt.grid()
plt.show()


# **Observation**
# 1. Extremely large sized summaries which can be seen as outliers are approved. Contrary to what we observed in the above PDF Plot Point no. 1
# 2. The 25th, Median and 75th quartiles are almost similar in both the plots.
# 3. The IQR range cover the summaries with the word count roughly between 11 to 25.

# #### TEXT PROCESSING FUNCTIONS

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

# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]


# In[ ]:


## FUNCTION TO CHECK NUMERIC VALUE IN A SENTENCE
#https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number/31861306

import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


# In[ ]:


# Combining all the above statements --> functions reused as provided in the iPython notebook from AppliedAI

from tqdm import tqdm
preprocessed_summary = []
preprocessed_summary_with_num=[]
# tqdm is for printing the status bar
for sentance in tqdm(project_data['project_resource_summary'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_summary.append(sent.lower().strip())
    if hasNumbers(sent):
        preprocessed_summary_with_num.append(1)
    else:
        preprocessed_summary_with_num.append(0)
    


# ### Checking whether the presence of the numerical digits in the project_resource_summary effects the acceptance of the project or not.

# In[ ]:


##counting number of summary with numbers
summcountwithnum=0
for i in range(0,len(preprocessed_summary_with_num)):
    if preprocessed_summary_with_num[i]==1:
        summcountwithnum+=1
print("Total number of resource summaries with NUMERIC value in it: ",summcountwithnum)


# In[ ]:


## Counting Total Approved application with and without NUMERIC value in resource summary data
sumnumapp=0
for i in range(0,total):
    if preprocessed_summary_with_num[i]==1 and project_data['project_is_approved'][i]==1:
        sumnumapp+=1
print("Total Approved application with NUMERIC value in resource summary:",sumnumapp)
projWithoutNumberSummary=total-summcountwithnum
print("Total number of resource summaries WITHOUT NUMERIC value in it: ",projWithoutNumberSummary)


# In[ ]:


## Printing a detailed report
total= total=project_data.shape[0]#109248
totalApp=y_value_counts[1] ## 92706
totalRejected=y_value_counts[0] ## 16542

print("\n\n----------DETAILED OBSERVATION----------\n")

print("PROJECT SUMMARY WITH VS WITHOUR NUMBERIC DIGITS %")
print("Total || With Number || Without Number || Percentage")
print(total,"    ",summcountwithnum,"        ",total-summcountwithnum,"         ",round(summcountwithnum/total*100,2),"%")
print("-"*50)
print()
print("PROJECT DETAILS")
print("Total || Approved || Rejected || Approval Percentage")
print(total,"    ",totalApp,"    ",totalRejected,"      ",round(totalApp/total*100,2),"%")
print("-"*50)
print()

print("PROJECT SUMMARY WITH NUMERIC VALUE DETAILS")
print("Total || Approved || Rejected || Approval Percentage")
print(summcountwithnum,"    ",sumnumapp,"    ",summcountwithnum-sumnumapp,"      ",round(sumnumapp/summcountwithnum*100,2),"%")
print("-"*50)
print()

print("PROJECT SUMMARY WITHOUT NUMERIC VALUE DETAILS")
print("Total || Approved || Rejected || Approval Percentage")
print(total-summcountwithnum,"    ",totalApp-sumnumapp,"    ",totalRejected-(summcountwithnum-sumnumapp),"      ",round((totalApp-sumnumapp)/(total-summcountwithnum)*100,2),"%")


# **Observation**
# 1. Only a small portion of 14% have numeric values in the summary.
# 1. Around 89% of the summaries having numeric values are approved than compared to the 84% approval rate of the summaries without the numeric values.
# 1. It is therefore recommended to have a numeric value in summary to increase the chance of approval.

# ## 1.3 Text preprocessing

# ### 1.3.1 Essay Text

# In[ ]:


project_data.head(2)


# In[ ]:


# printing some random essays.
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


sent = decontracted(project_data['essay'].values[2000])
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


# Combining all the above statemennts 
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
    sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_essays.append(sent.lower().strip())


# In[ ]:


# after preprocesing
preprocessed_essays[2000]


# ### 1.3.2 Project title Text

# # DonorsChoose: Project Title Text Preprocessing

# In[ ]:


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


print("BEFORE -->  ",project_data['project_title'][7],"     NOW --> ",preprocessed_titles [7])


# ## 1. 4 Preparing data for models

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
#        - project_resource_summary: text data
#        
#        - quantity : numerical
#        - teacher_number_of_previously_posted_projects : numerical
#        - price : numerical

# ### 1.4.1 Vectorizing Categorical data

# - https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/handling-categorical-and-numerical-features/

# In[ ]:


# we use count vectorizer to convert the values into one hot encoded features
# You can use it as follows:
# Create an instance of the CountVectorizer class.
# Call the fit() function in order to learn a vocabulary from one or more documents.
# Call the transform() function on one or more documents as needed to encode each as a vector.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=list(sorted_cat_dict.keys()), lowercase=False, binary=True)
vectorizer.fit(project_data['clean_categories'].values)
print(vectorizer.get_feature_names())


categories_one_hot = vectorizer.transform(project_data['clean_categories'].values)
print("Shape of matrix after one hot encodig ",categories_one_hot.shape)
categories_one_hot.toarray()


# In[ ]:


# we use count vectorizer to convert the values into one hot encoded features
vectorizer = CountVectorizer(vocabulary=list(sorted_sub_cat_dict.keys()), lowercase=False, binary=True)
vectorizer.fit(project_data['clean_subcategories'].values)
print(vectorizer.get_feature_names())


sub_categories_one_hot = vectorizer.transform(project_data['clean_subcategories'].values)
print("Shape of matrix after one hot encodig ",sub_categories_one_hot.shape)
print(sub_categories_one_hot.toarray())


# ### STATE

# In[ ]:


from scipy import sparse ## Exporting Sparse Matrix to NPZ File -> https://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
statelist=list(project_data['school_state'].values)
vectorizer = CountVectorizer(vocabulary=set(statelist), lowercase=False, binary=True)
vectorizer.fit(statelist)
print(vectorizer.get_feature_names())


school_one_hot = vectorizer.transform(statelist)
print("Shape of matrix after one hot encodig ",school_one_hot.shape)
print(type(school_one_hot))
sparse.save_npz("school_one_hot_export.npz", school_one_hot) 
print(school_one_hot.toarray())


# ### PREFIX

# **Teacher Prefix has NAN values, that needs to be cleaned.
# Ref: https://stackoverflow.com/a/50297200/4433839**

# In[ ]:


prefixlist=project_data['teacher_prefix'].values
prefixlist=list(prefixlist)
cleanedPrefixList = [x for x in prefixlist if x == x] ## Cleaning the NULL Values in the list -> https://stackoverflow.com/a/50297200/4433839

## preprocessing the prefix to remove the SPACES,- else the vectors will be just 0's. Try adding - and see
prefix_nospace_list = []
for i in cleanedPrefixList:
    temp = ""
    i = i.replace('.','') # we are placeing all the '.'(dot) with ''(empty) ex:"Mr."=>"Mr"
    temp +=i.strip()+" "#" abc ".strip() will return "abc", remove the trailing spaces
    prefix_nospace_list.append(temp.strip())

cleanedPrefixList=prefix_nospace_list

vectorizer = CountVectorizer(vocabulary=set(cleanedPrefixList), lowercase=False, binary=True)
vectorizer.fit(cleanedPrefixList)
print(vectorizer.get_feature_names())
prefix_one_hot = vectorizer.transform(cleanedPrefixList)
print("Shape of matrix after one hot encodig ",prefix_one_hot.shape)
prefix_one_hot_ar=prefix_one_hot.todense()

##code to export to csv -> https://stackoverflow.com/a/54637996/4433839
# prefixcsv=pd.DataFrame(prefix_one_hot.toarray())
# prefixcsv.to_csv('prefix.csv', index=None,header=None)


# In[ ]:


print(type(prefix_one_hot_ar))


# **Observation:** 
# 1. 3 Rows had NaN values and they are not considered, thus the number of rows reduced from 109248 to 109245.

# ### GRADE

# In[ ]:


gradelist=project_data['project_grade_category'].values
gradelist=list(gradelist)

## preprocessing the grades to remove the SPACES,- else the vectors will be just 0's. Try adding - and see
grade_nospace_list = []
for i in gradelist:
    temp = ""
    i = i.replace(' ','_') # we are placeing all the ' '(space) with ''(empty) ex:"Grades 3-5"=>"Grades_3-5"
    i = i.replace('-','_')
    temp +=i.strip()+" "#" abc ".strip() will return "abc", remove the trailing spaces
    grade_nospace_list.append(temp.strip())

vectorizer = CountVectorizer(vocabulary=set(grade_nospace_list), lowercase=False, binary=True)
vectorizer.fit(grade_nospace_list)
print(vectorizer.get_feature_names())
grade_one_hot = vectorizer.transform(grade_nospace_list)
print("Shape of matrix after one hot encodig ",grade_one_hot.shape)
print(type(grade_one_hot))
grade_one_hot.toarray()

##code to export to csv -> https://stackoverflow.com/a/54637996/4433839
# gradecsv=pd.DataFrame(grade_one_hot.toarray())
# gradecsv.to_csv('grades.csv', index=None,header=None)


# In[ ]:


tt=grade_one_hot.todense()
print(type(tt))


# In[ ]:


grade_one_hot


# ### 1.4.2 Vectorizing Text data

# #### 1.4.2.1 Bag of words: ESSAYS

# In[ ]:


# We are considering only the words which appeared in at least 10 documents(rows or projects).
vectorizer = CountVectorizer(min_df=10)
text_bow = vectorizer.fit_transform(preprocessed_essays)
# text_bow = vectorizer.fit(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_bow.shape)

# TO CHECK A VECTOR
# https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e
# v0 = vectorizer.transform([preprocessed_essays[0]]).toarray()[0] 
# print(v0)

# # write one of the vector to CSV to confirm wether we are obtaining meaning full results or just 0's.
# text_bow=pd.DataFrame(v0)
# text_bow.to_csv('text_bow.csv', index=None,header=None)

# count number of non zero values in the vector
# count=0
# for i in range(0,len(v0)):
#     if v0[i]>0:
#         count=count+1
# print(count)
# v0.shape


# <h4> 1.4.2.2 Bag of Words: PROJECT TITLES</h4>

# In[ ]:


# you can vectorize the title also 
# before you vectorize the title make sure you preprocess it
vectorizer = CountVectorizer(min_df=10)
text_title_bow = vectorizer.fit_transform(preprocessed_titles)
print("Shape of matrix after one hot encodig ",text_title_bow.shape)


# TO CHECK A VECTOR
# https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e

# v0 = vectorizer.transform([preprocessed_titles[0]]).toarray()[0]

# write one of the vector to CSV to confirm wether we are obtaining meaning full results or just 0's.

# text_title_bow=pd.DataFrame(v0)
# text_title_bow.to_csv('text_title_bow.csv', index=None,header=None)

# count number of non zero values in the vector
# count=0
# for i in range(0,len(v0)):
#     if v0[i]>0:
#         count=count+1
# print(count)
# print(v0.shape)


# In[ ]:


text_title_bow_ar=text_title_bow.todense()
print(type(text_title_bow_ar))
text_title_bow_ar


# #### 1.4.2.3 TFIDF vectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
text_tfidf = vectorizer.fit_transform(preprocessed_essays)
print("Shape of matrix after one hot encodig ",text_tfidf.shape)


# #### 1.4.2.4 TFIDF Vectorizer on `project_title`

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
text_titles_tfidf = vectorizer.fit_transform(preprocessed_titles)
print("Shape of matrix after one hot encodig ",text_titles_tfidf.shape)

# Code for testing and checking the generated vectors
# v1 = vectorizer.transform([preprocessed_titles[0]]).toarray()[0]
# text_title_tfidf=pd.DataFrame(v1)
# text_title_tfidf.to_csv('text_title_tfidf.csv', index=None,header=None)


# #### 1.4.2.5 Using Pretrained Models: Avg W2V

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


# In[ ]:


# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
# make sure you have the glove_vectors file
with open('../input/glove_vectors/glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words =  set(model.keys())
# print(kajsdf)


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

print("Length of avg_w2v_vectors: ",len(avg_w2v_vectors))
print("Length of avg_w2v_vectors[0]: ",len(avg_w2v_vectors[0]))


# #### 1.4.2.6 Using Pretrained Models: AVG W2V on `project_title`

# In[ ]:


# Similarly you can vectorize for title also
# average Word2Vec
# compute average word2vec for each review.
avg_w2v_title_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(preprocessed_titles): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_title_vectors.append(vector)

print("length of avg_w2v_title_vectors: ",len(avg_w2v_title_vectors))
print("length avg_w2v_title_vectors[0]:  ",len(avg_w2v_title_vectors[0]))
# print(avg_w2v_title_vectors[0]) ## Checking the generated vector


# #### 1.4.2.7 Using Pretrained Models: TFIDF weighted W2V

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
tfidf_model = TfidfVectorizer()
tfidf_model.fit(preprocessed_essays)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
tfidf_words = set(tfidf_model.get_feature_names())


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

print("length of tfidf_w2v_vectors: ",len(tfidf_w2v_vectors))
print("length of tfidf_w2v_vectors[0]: ",len(tfidf_w2v_vectors[0]))


# #### 1.4.2.9 Using Pretrained Models: TFIDF weighted W2V on `project_title`

# In[ ]:


# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
tfidf_title_model = TfidfVectorizer()
tfidf_title_model.fit(preprocessed_titles)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_title_model.get_feature_names(), list(tfidf_title_model.idf_)))
tfidf_title_words = set(tfidf_title_model.get_feature_names())


# In[ ]:


# average Word2Vec
# compute average word2vec for each title.
tfidf_w2v_title_vectors = []; # the avg-w2v for each title is stored in this list
for sentence in preprocessed_titles: # for each review/sentence
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
    tfidf_w2v_title_vectors.append(vector)

print("Length of tfidf_w2v_title_vectors: ",len(tfidf_w2v_title_vectors))
print("Length of  tfidf_w2v_title_vectors[0]: ",len(tfidf_w2v_title_vectors[0]))
# print(tfidf_w2v_title_vectors[0])


# ### 1.4.3 Vectorizing Numerical features

# ### 1.4.3.1 Standardizing Price

# In[ ]:


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


price_standardized


# ### 1.4.3.2 Standardizing Teacher's Previous Projects

# In[ ]:


# check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# price_standardized = standardScalar.fit(project_data['price'].values)
# this will rise the error
# ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# Reshape your data either using array.reshape(-1, 1)

teacher_previous_proj_scalar = StandardScaler()
teacher_previous_proj_scalar.fit(project_data['teacher_number_of_previously_posted_projects'].values.reshape(-1,1)) # finding the mean and standard deviation of this data
print(f"Mean : {teacher_previous_proj_scalar.mean_[0]}, Standard deviation : {np.sqrt(teacher_previous_proj_scalar.var_[0])}")

# Now standardize the data with above maen and variance.
teacher_previous_proj_standardized = teacher_previous_proj_scalar.transform(project_data['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1))


# In[ ]:



teacher_previous_proj_standardized.shape


# ### 1.4.4 Merging all the above features

# - we need to merge all the numerical vectors i.e catogorical, text, numerical vectors

# #### Checking the shape for final merging.

# In[ ]:


avg_w2v_vectors=np.asarray(avg_w2v_vectors)
avg_w2v_title_vectors=np.asarray(avg_w2v_title_vectors)
# tfidf_w2v_vectors=np.asarray(tfidf_w2v_vectors)
tfidf_w2v_title_vectors=np.asarray(tfidf_w2v_title_vectors)

print("SCHOOL STATE: -> ",school_one_hot.shape)
print("categories_one_hot -> ",categories_one_hot.shape)
print("sub_categories_one_hot -> ",sub_categories_one_hot.shape)
print("TEACHER PREFIX -> ", prefix_one_hot.shape)
print("PROJECT GRADE -> ",grade_one_hot.shape)

print("price_standardized -> ",price_standardized.shape)
print("TEACHER PREVIOUS POSTED PROJECT -> ",teacher_previous_proj_standardized.shape)

print("text_title_bow -> ",text_title_bow.shape)
print("text_titles_tfidf -> ",text_titles_tfidf.shape)
print("avg_w2v_title_vectors -> ",avg_w2v_title_vectors.shape)
print("tfidf_w2v_title_vectors -> ",tfidf_w2v_title_vectors.shape)


# # SAVING AND LOADING MATRIX FOR EFFICIENT RUNNING

# In[ ]:


## save csr to npz python -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html


# ### SAVING VECTORS

# In[ ]:


from scipy.sparse import hstack
print("H-STACKING the required features")
CAT_NUM_BOW_vec_stack=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, text_title_bow))
CAT_NUM_TFIDF_vec_stack=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, text_titles_tfidf))
CAT_NUM_Avg_W2V_vec_stack=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, avg_w2v_title_vectors))
CAT_NUM_TFIDF_W2V_vec_stack=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, tfidf_w2v_title_vectors))
ALL_vec_stack=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, text_title_bow, text_titles_tfidf, avg_w2v_title_vectors, tfidf_w2v_title_vectors, price_standardized, teacher_previous_proj_standardized))
print("H-STACKING Completed----------------------------------\n")

print("TYPE: CAT_NUM_BOW_vec_stack: ",type(CAT_NUM_BOW_vec_stack))
print("TYPE: CAT_NUM_TFIDF_vec_stack: ",type(CAT_NUM_TFIDF_vec_stack))
print("TYPE: CAT_NUM_Avg_W2V_vec_stack: ",type(CAT_NUM_Avg_W2V_vec_stack))
print("TYPE: CAT_NUM_TFIDF_W2V_vec_stack: ",type(CAT_NUM_TFIDF_W2V_vec_stack))
print("TYPE: ALL_vec_stack: ",type(ALL_vec_stack))

print("SHAPE: CAT_NUM_BOW_vec_stack: ",CAT_NUM_BOW_vec_stack.shape)
print("SHAPE: CAT_NUM_TFIDF_vec_stack: ",CAT_NUM_TFIDF_vec_stack.shape)
print("SHAPE: CAT_NUM_Avg_W2V_vec_stack: ",CAT_NUM_Avg_W2V_vec_stack.shape)
print("SHAPE: CAT_NUM_TFIDF_W2V_vec_stack: ",CAT_NUM_TFIDF_W2V_vec_stack.shape)
print("SHAPE: ALL_vec_stack: ",ALL_vec_stack.shape)

print("============================================================\n\n")
print("Converting to CSR")
CAT_NUM_BOW_vec_stack_tocsr=CAT_NUM_BOW_vec_stack.tocsr()
CAT_NUM_TFIDF_vec_stack_tocsr=CAT_NUM_TFIDF_vec_stack.tocsr()
CAT_NUM_Avg_W2V_vec_stack_tocsr=CAT_NUM_Avg_W2V_vec_stack.tocsr()
CAT_NUM_TFIDF_W2V_vec_stack_tocsr=CAT_NUM_TFIDF_W2V_vec_stack.tocsr()
ALL_vec_stack_tocsr=ALL_vec_stack.tocsr()
print("Converted to CSR --------------------------------------\n")


print("TYPE: CAT_NUM_BOW_vec_stack_tocsr: ",type(CAT_NUM_BOW_vec_stack_tocsr))
print("TYPE: CAT_NUM_TFIDF_vec_stack_tocsr: ",type(CAT_NUM_TFIDF_vec_stack_tocsr))
print("TYPE: CAT_NUM_Avg_W2V_vec_stack_tocsr: ",type(CAT_NUM_Avg_W2V_vec_stack_tocsr))
print("TYPE: CAT_NUM_TFIDF_W2V_vec_stack_tocsr: ",type(CAT_NUM_TFIDF_W2V_vec_stack_tocsr))
print("TYPE: ALL_vec_stack_tocsr: ",type(CAT_NUM_TFIDF_W2V_vec_stack_tocsr))


print("SHAPE: CAT_NUM_BOW_vec_stack_tocsr: ",CAT_NUM_BOW_vec_stack_tocsr.shape)
print("SHAPE: CAT_NUM_TFIDF_vec_stack_tocsr: ",CAT_NUM_TFIDF_vec_stack_tocsr.shape)
print("SHAPE: CAT_NUM_Avg_W2V_vec_stack_tocsr: ",CAT_NUM_Avg_W2V_vec_stack_tocsr.shape)
print("SHAPE: CAT_NUM_TFIDF_W2V_vec_stack_tocsr: ",CAT_NUM_TFIDF_W2V_vec_stack_tocsr.shape)
print("SHAPE: ALL_vec_stack_tocsr: ",ALL_vec_stack_tocsr.shape)

print("============================================================\n\n")
print("Saving to NPZ\n")
scipy.sparse.save_npz('CAT_NUM_BOW_vec_stack_csr_sparse_matrix.npz',CAT_NUM_BOW_vec_stack_tocsr)
scipy.sparse.save_npz('CAT_NUM_TFIDF_vec_stack_csr_sparse_matrix.npz',CAT_NUM_TFIDF_vec_stack_tocsr)
scipy.sparse.save_npz('CAT_NUM_Avg_W2V_vec_stack_csr_sparse_matrix.npz',CAT_NUM_Avg_W2V_vec_stack_tocsr)
scipy.sparse.save_npz('CAT_NUM_TFIDF_W2V_vec_stack_csr_sparse_matrix.npz',CAT_NUM_TFIDF_W2V_vec_stack_tocsr)
scipy.sparse.save_npz('ALL_vec_stack_csr_sparse_matrix.npz',ALL_vec_stack_tocsr)
print("Saved to NPZ--------------------------------------\n\n")
print("============================================================\n\n")


# **Please ignore the above Scipy Error, its resolved by importing it. Since it is taking a long time to run, I have avoided to re-run it completly.**

# ### LOADING VECTORS

# In[ ]:


print("LOADING THE NPZ")
CAT_NUM_BOW_vec_tocsr=scipy.sparse.load_npz('CAT_NUM_BOW_vec_stack_csr_sparse_matrix.npz')
CAT_NUM_TFIDF_vec_tocsr=scipy.sparse.load_npz('CAT_NUM_TFIDF_vec_stack_csr_sparse_matrix.npz')
CAT_NUM_Avg_W2V_vec_tocsr=scipy.sparse.load_npz('CAT_NUM_Avg_W2V_vec_stack_csr_sparse_matrix.npz')
CAT_NUM_TFIDF_W2V_vec_tocsr=scipy.sparse.load_npz('CAT_NUM_TFIDF_W2V_vec_stack_csr_sparse_matrix.npz')
ALL_vec_tocsr=scipy.sparse.load_npz('ALL_vec_stack_csr_sparse_matrix.npz')

print("LOADING Completed ----------------------------------\n")

print("TYPE: CAT_NUM_BOW_vec_tocsr: ",type(CAT_NUM_BOW_vec_tocsr))
print("TYPE: CAT_NUM_TFIDF_vec_tocsr: ",type(CAT_NUM_TFIDF_vec_tocsr))
print("TYPE: CAT_NUM_Avg_W2V_vec_tocsr: ",type(CAT_NUM_Avg_W2V_vec_tocsr))
print("TYPE: CAT_NUM_TFIDF_W2V_vec_tocsr: ",type(CAT_NUM_TFIDF_W2V_vec_tocsr))
print("TYPE: ALL_vec_tocsr: ",type(ALL_vec_tocsr))


print("SHAPE: CAT_NUM_BOW_vec_tocsr: ",CAT_NUM_BOW_vec_tocsr.shape)
print("SHAPE: CAT_NUM_TFIDF_vec_tocsr: ",CAT_NUM_TFIDF_vec_tocsr.shape)
print("SHAPE: CAT_NUM_Avg_W2V_vec_tocsr: ",CAT_NUM_Avg_W2V_vec_tocsr.shape)
print("SHAPE: CAT_NUM_TFIDF_W2V_vec_tocsr: ",CAT_NUM_TFIDF_W2V_vec_tocsr.shape)
print("SHAPE: ALL_vec_tocsr: ",ALL_vec_tocsr.shape)
print("=========================END===================================\n\n")


# <h1><font color='red'>Assignment 2: Apply TSNE<font></h1>

#  <font color=#F4274F>If you are using any code snippet from the internet, you have to provide the reference/citations, as we did in the above cells. Otherwise, it will be treated as plagiarism without citations.</font>

# <ol> 
#     <li> In the above cells we have plotted and analyzed many features. Please observe the plots and write the observations in markdown cells below every plot.</li>
#     <li> EDA: Please complete the analysis of the feature: teacher_number_of_previously_posted_projects</li>
#     <li>
#         <ul>Build the data matrix using these features 
#             <li>school_state : categorical data (one hot encoding)</li>
#             <li>clean_categories : categorical data (one hot encoding)</li>
#             <li>clean_subcategories : categorical data (one hot encoding)</li>
#             <li>teacher_prefix : categorical data (one hot encoding)</li>
#             <li>project_grade_category : categorical data (one hot encoding)</li>
#             <li>project_title : text data (BOW, TFIDF, AVG W2V, TFIDF W2V)</li>
#             <li>price : numerical</li>
#             <li>teacher_number_of_previously_posted_projects : numerical</li>
#          </ul>
#     </li>
#     <li> Now, plot FOUR t-SNE plots with each of these feature sets.
#         <ol>
#             <li>categorical, numerical features + project_title(BOW)</li>
#             <li>categorical, numerical features + project_title(TFIDF)</li>
#             <li>categorical, numerical features + project_title(AVG W2V)</li>
#             <li>categorical, numerical features + project_title(TFIDF W2V)</li>
#         </ol>
#     </li>
#     <li> Concatenate all the features and Apply TNSE on the final data matrix </li>
#     <li> <font color='blue'>Note 1: The TSNE accepts only dense matrices</font></li>
#     <li> <font color='blue'>Note 2: Consider only 5k to 6k data points to avoid memory issues. If you run into memory error issues, reduce the number of data points but clearly state the number of datat-poins you are using</font></li>
# </ol>

# ## Sample size is varied to observe the behaviour of tSNE on different size of data. Below are the configuration details
# 
# 1. BOW = 6000
# 1. TF-IDF = 10,000
# 1. Avg W2V = 15,000
# 1. TF-IDF W2V= 20,000
# 1. ALL ABOVE = 7000
# 
# A max of 6000 datapoints were supported in an i5/8gb system, however I tried the approach of saving the vectors in NPZ format and then loading it anytime without need to load the glove vectors and run the vector generation code, this allowed me to save RAM space. This further helped me in loading datapoints upto 20,000 in a normal machine.

# <h2> 2.1 TSNE with `BOW` encoding of `project_title` feature </h2>

# tSNE with `BOW` encoding of `project_title` feature with the dataset size of 1200, 3200, 4200, 6000 for comparison purpose.

# ![../input/allBows.PNG](attachment:allBows.PNG)

# **Observation:**
# 1. The points are scattered in dataset of size 1200
# 1. With 1200 data points, hardly any clusters form are visible. Just a large cloud is visible.
# 1. As the datapoints are increased, we can see the lighter clusters are now turning into solid cluster, but any pattern is not observed.

# ### SAMPLE SIZE = 6000

# In[ ]:


project_data = pd.read_csv('../input/train_data.csv')


# In[ ]:


sampleSize=6000


# In[ ]:


## CAT_NUM_BOW Dataset
# from sklearn.manifold import TSNE

print("SAMPLE SIZE: ",sampleSize,"\n\n")
start_time = time.time()
# # CAT_NUM_BOW_vec=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, text_title_bow))
# print("TYPE: CAT_NUM_BOW_vec: ",type(CAT_NUM_BOW_vec))
# print("SHAPE: CAT_NUM_BOW_vec: ",CAT_NUM_BOW_vec.shape)
# #<class 'scipy.sparse.coo.coo_matrix'>
# print("\n\n")

# CAT_NUM_BOW_vec_tocsr=CAT_NUM_BOW_vec.tocsr()
print("CSR VECTORS ARE LOADED FROM DISK FOR SAVING MEMORY BY NOT REQUIRING ABOVE VECTOR GENERATION CODE TO BE RUN AGAIN AND ALSO SAVES RAM BY NOT LOADING THE GLOVE VECTORS IN MEMORY")
CAT_NUM_BOW_vec_n_samples = CAT_NUM_BOW_vec_tocsr[0:sampleSize,:]

print("TYPE: CAT_NUM_BOW_vec_n_samples: ",type(CAT_NUM_BOW_vec_n_samples))
# <class 'scipy.sparse.csr.csr_matrix'>

CAT_NUM_BOW_vec_n_samples_toarray=CAT_NUM_BOW_vec_n_samples.toarray()
print("TYPE: CAT_NUM_BOW_vec_n_samples_toarray: ",type(CAT_NUM_BOW_vec_n_samples_toarray))
print("SHAPE: CAT_NUM_BOW_vec_n_samples_toarray: ",CAT_NUM_BOW_vec_n_samples_toarray.shape)
# <class 'numpy.ndarray'>
print("\n\n")

tsne_model = TSNE(n_components = 2, perplexity = 100.0, random_state = 0)
tsne_data = tsne_model.fit_transform(CAT_NUM_BOW_vec_n_samples_toarray)


print("TYPE: tsne_data: ",type(tsne_data))
print("SHAPE: tsne_data: ",tsne_data.shape)
print("\n\n")

CAT_NUM_BOW_Labels=project_data["project_is_approved"]
CAT_NUM_BOW_Labels_n_samples = CAT_NUM_BOW_Labels[0: sampleSize]
print("LEN: CAT_NUM_BOW_Labels_n_samples: ",len(CAT_NUM_BOW_Labels_n_samples))
print("\n\n")

tsne_data = np.vstack((tsne_data.T, CAT_NUM_BOW_Labels_n_samples)).T
# tsne_df_b = pd.DataFrame(tsne_data_b, columns = ("1st_Dim","2nd_Dim","Labels"))
tsne_data_df = pd.DataFrame(tsne_data, columns = ["Dimension_x","Dimension_y","Score"])


print("TYPE: tsne_data-AfterStacking: ",type(tsne_data))
print("TYPE: tsne_data_df: ",type(tsne_data_df))
print("SHAPE: tsne_data_df: ",tsne_data_df.shape)
print("\n\n")
# colors = {0:'red', 1:'blue', 2:'green'}
# plt.scatter(tsne_data_df['Dimension_x'], tsne_data_df['Dimension_y'], c=tsne_data_df['Score'].apply(lambda x: colors[x]))
# plt.show()

import matplotlib.pyplot as plt
sns.FacetGrid(tsne_data_df, hue = "Score", size = 8).map(plt.scatter, "Dimension_x", "Dimension_y").add_legend()
plt.suptitle("TSNE WITH BOW ENCODING OF PROJECT TITLE FEATURE ")
plt.show()
print("PROCESSING TOOK--- %s seconds ---" % (time.time() - start_time))


# **Observations:**
# 1. Unstructured random smaller and larger clusters are visible.
# 1. Data points are very much overlapping.
# 1. Unable to draw any concrete conlusion.

# <h2> 2.2 TSNE with `TFIDF` encoding of `project_title` feature </h2>

# ### SAMPLE SIZE = 10,000

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label

# sampleSize=10000
print("SAMPLE SIZE: ",sampleSize,"\n\n")
start_time = time.time()
# CAT_NUM_TFIDF_vec=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, text_titles_tfidf))
# print("TYPE: CAT_NUM_TFIDF_vec: ",type(CAT_NUM_TFIDF_vec))
# print("SHAPE: CAT_NUM_TFIDF_vec: ",CAT_NUM_TFIDF_vec.shape)
# #<class 'scipy.sparse.coo.coo_matrix'>

# CAT_NUM_TFIDF_vec_tocsr=CAT_NUM_TFIDF_vec.tocsr()
print("CSR VECTORS ARE LOADED FROM DISK FOR SAVING MEMORY BY NOT REQUIRING ABOVE VECTOR GENERATION CODE TO BE RUN AGAIN AND ALSO SAVES RAM BY NOT LOADING THE GLOVE VECTORS IN MEMORY")
CAT_NUM_TFIDF_vec_n_samples = CAT_NUM_TFIDF_vec_tocsr[0:sampleSize,:]

print("TYPE: CAT_NUM_TFIDF_vec_n_samples: ",type(CAT_NUM_TFIDF_vec_n_samples))
# <class 'scipy.sparse.csr.csr_matrix'>

CAT_NUM_TFIDF_vec_n_samples_toarray=CAT_NUM_TFIDF_vec_n_samples.toarray()
print("TYPE: CAT_NUM_TFIDF_vec_n_samples_toarray: ",type(CAT_NUM_TFIDF_vec_n_samples_toarray))
print("SHAPE: CAT_NUM_TFIDF_vec_n_samples_toarray: ",CAT_NUM_TFIDF_vec_n_samples_toarray.shape)
# <class 'numpy.ndarray'>


tsne_model = TSNE(n_components = 2, perplexity = 100.0, random_state = 0)
tsne_data = tsne_model.fit_transform(CAT_NUM_TFIDF_vec_n_samples_toarray)


print("TYPE: tsne_data: ",type(tsne_data))
print("SHAPE: tsne_data: ",tsne_data.shape)


CAT_NUM_TFIDF_Labels=project_data["project_is_approved"]
CAT_NUM_TFIDF_Labels_n_samples = CAT_NUM_TFIDF_Labels[0: sampleSize]
print("LEN: CAT_NUM_TFIDF_Labels_n_samples: ",len(CAT_NUM_TFIDF_Labels_n_samples))


tsne_data = np.vstack((tsne_data.T, CAT_NUM_TFIDF_Labels_n_samples)).T
# tsne_df_b = pd.DataFrame(tsne_data_b, columns = ("1st_Dim","2nd_Dim","Labels"))
tsne_data_df = pd.DataFrame(tsne_data, columns = ["Dimension_x","Dimension_y","Score"])


print("TYPE: tsne_data-AfterStacking: ",type(tsne_data))
print("TYPE: tsne_data_df: ",type(tsne_data_df))
print("SHAPE: tsne_data_df: ",tsne_data_df.shape)

# colors = {0:'red', 1:'blue', 2:'green'}
# plt.scatter(tsne_data_df['Dimension_x'], tsne_data_df['Dimension_y'], c=tsne_data_df['Score'].apply(lambda x: colors[x]))
# plt.show()

import matplotlib.pyplot as plt
sns.FacetGrid(tsne_data_df, hue = "Score", size = 8).map(plt.scatter, "Dimension_x", "Dimension_y").add_legend()
plt.suptitle("TSNE WITH TF-IDF ENCODING OF PROJECT TITLE FEATURE ")
plt.show()
print("PROCESSING TOOK--- %s seconds ---" % (time.time() - start_time))


# sns.FacetGrid(tsne_df_b, hue = "Score", size = 10).map(plt.scatter, "Dimension_x", "Dimension_y").add_legend()
# # nd().fig.suptitle("TSNE WITH BOW ENCODING OF PROJECT TITLE FEATURE ")
# plt.show()
    


# **Observations:**
# 1. Unstructured random smaller and larger clusters are visible but with high overlapping of points.
# 1. Unable to draw any concrete conlusion.

# <h2> 2.3 TSNE with `AVG W2V` encoding of `project_title` feature </h2>

# ### SAMPLE SIZE = 15,000

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label

sampleSize=15000
print("SAMPLE SIZE: ",sampleSize,"\n\n")
start_time = time.time()
# CAT_NUM_Avg_W2V_vec=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, avg_w2v_title_vectors))
# print("TYPE: CAT_NUM_Avg_W2V_vec: ",type(CAT_NUM_Avg_W2V_vec))
# print("SHAPE: CAT_NUM_Avg_W2V_vec: ",CAT_NUM_Avg_W2V_vec.shape)
# #<class 'scipy.sparse.coo.coo_matrix'>

# CAT_NUM_Avg_W2V_vec_tocsr=CAT_NUM_Avg_W2V_vec.tocsr()
print("CSR VECTORS ARE LOADED FROM DISK FOR SAVING MEMORY BY NOT REQUIRING ABOVE VECTOR GENERATION CODE TO BE RUN AGAIN AND ALSO SAVES RAM BY NOT LOADING THE GLOVE VECTORS IN MEMORY")
CAT_NUM_Avg_W2V_vec_n_samples = CAT_NUM_Avg_W2V_vec_tocsr[0:sampleSize,:]

print("TYPE: CAT_NUM_Avg_W2V_vec_n_samples: ",type(CAT_NUM_Avg_W2V_vec_n_samples))
# <class 'scipy.sparse.csr.csr_matrix'>

CAT_NUM_Avg_W2V_vec_n_samples_toarray=CAT_NUM_Avg_W2V_vec_n_samples.toarray()
print("TYPE: CAT_NUM_Avg_W2V_vec_n_samples_toarray: ",type(CAT_NUM_Avg_W2V_vec_n_samples_toarray))
print("SHAPE: CAT_NUM_Avg_W2V_vec_n_samples_toarray: ",CAT_NUM_Avg_W2V_vec_n_samples_toarray.shape)
# <class 'numpy.ndarray'>


tsne_model = TSNE(n_components = 2, perplexity = 100.0, random_state = 0)
tsne_data = tsne_model.fit_transform(CAT_NUM_Avg_W2V_vec_n_samples_toarray)


print("TYPE: tsne_data: ",type(tsne_data))
print("SHAPE: tsne_data: ",tsne_data.shape)


CAT_NUM_Avg_W2V_Labels=project_data["project_is_approved"]
CAT_NUM_Avg_W2V_Labels_n_samples = CAT_NUM_Avg_W2V_Labels[0: sampleSize]
print("LEN: CAT_NUM_Avg_W2V_Labels_n_samples: ",len(CAT_NUM_Avg_W2V_Labels_n_samples))


tsne_data = np.vstack((tsne_data.T, CAT_NUM_Avg_W2V_Labels_n_samples)).T
# tsne_df_b = pd.DataFrame(tsne_data_b, columns = ("1st_Dim","2nd_Dim","Labels"))
tsne_data_df = pd.DataFrame(tsne_data, columns = ["Dimension_x","Dimension_y","Score"])


print("TYPE: tsne_data-AfterStacking: ",type(tsne_data))
print("TYPE: tsne_data_df: ",type(tsne_data_df))
print("SHAPE: tsne_data_df: ",tsne_data_df.shape)

# colors = {0:'red', 1:'blue', 2:'green'}
# plt.scatter(tsne_data_df['Dimension_x'], tsne_data_df['Dimension_y'], c=tsne_data_df['Score'].apply(lambda x: colors[x]))
# plt.show()

import matplotlib.pyplot as plt
sns.FacetGrid(tsne_data_df, hue = "Score", size = 8).map(plt.scatter, "Dimension_x", "Dimension_y").add_legend()
plt.suptitle("TSNE WITH AVERAGE WORD2VEC ENCODING OF PROJECT TITLE FEATURE ")
plt.show()
print("PROCESSING TOOK--- %s seconds ---" % (time.time() - start_time))
    


# **Observations:**
# 1. Unstructured large cluster is observed but with high overlapping of points.
# 1. Unable to draw any concrete conlusion.

# <h2> 2.4 TSNE with `TFIDF Weighted W2V` encoding of `project_title` feature </h2>

# ### SAMPLE SIZE = 20,000

# In[ ]:


# please write all the code with proper documentation, and proper titles for each subsection
# when you plot any graph make sure you use 
    # a. Title, that describes your plot, this will be very helpful to the reader
    # b. Legends if needed
    # c. X-axis label
    # d. Y-axis label
    
sampleSize=20000
print("SAMPLE SIZE: ",sampleSize,"\n\n")
start_time = time.time()
# CAT_NUM_TFIDF_W2V_vec=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, tfidf_w2v_title_vectors))
# print("TYPE: CAT_NUM_TFIDF_W2V_vec: ",type(CAT_NUM_TFIDF_W2V_vec))
# print("SHAPE: CAT_NUM_TFIDF_W2V_vec: ",CAT_NUM_TFIDF_W2V_vec.shape)
# #<class 'scipy.sparse.coo.coo_matrix'>

# CAT_NUM_TFIDF_W2V_vec_tocsr=CAT_NUM_TFIDF_W2V_vec.tocsr()
print("CSR VECTORS ARE LOADED FROM DISK FOR SAVING MEMORY BY NOT REQUIRING ABOVE VECTOR GENERATION CODE TO BE RUN AGAIN AND ALSO SAVES RAM BY NOT LOADING THE GLOVE VECTORS IN MEMORY")
CAT_NUM_TFIDF_W2V_vec_n_samples = CAT_NUM_TFIDF_W2V_vec_tocsr[0:sampleSize,:]

print("TYPE: CAT_NUM_TFIDF_W2V_vec_n_samples: ",type(CAT_NUM_TFIDF_W2V_vec_n_samples))
# <class 'scipy.sparse.csr.csr_matrix'>

CAT_NUM_TFIDF_W2V_vec_n_samples_toarray=CAT_NUM_TFIDF_W2V_vec_n_samples.toarray()
print("TYPE: CAT_NUM_TFIDF_W2V_vec_n_samples_toarray: ",type(CAT_NUM_TFIDF_W2V_vec_n_samples_toarray))
print("SHAPE: CAT_NUM_TFIDF_W2V_vec_n_samples_toarray: ",CAT_NUM_TFIDF_W2V_vec_n_samples_toarray.shape)
# <class 'numpy.ndarray'>


tsne_model = TSNE(n_components = 2, perplexity = 100.0, random_state = 0)
tsne_data = tsne_model.fit_transform(CAT_NUM_TFIDF_W2V_vec_n_samples_toarray)


print("TYPE: tsne_data: ",type(tsne_data))
print("SHAPE: tsne_data: ",tsne_data.shape)


CAT_NUM_TFIDF_W2V_Labels=project_data["project_is_approved"]
CAT_NUM_TFIDF_W2V_Labels_n_samples = CAT_NUM_TFIDF_W2V_Labels[0: sampleSize]
print("LEN: CAT_NUM_TFIDF_W2V_Labels_n_samples: ",len(CAT_NUM_TFIDF_W2V_Labels_n_samples))


tsne_data = np.vstack((tsne_data.T, CAT_NUM_TFIDF_W2V_Labels_n_samples)).T
# tsne_df_b = pd.DataFrame(tsne_data_b, columns = ("1st_Dim","2nd_Dim","Labels"))
tsne_data_df = pd.DataFrame(tsne_data, columns = ["Dimension_x","Dimension_y","Score"])


print("TYPE: tsne_data-AfterStacking: ",type(tsne_data))
print("TYPE: tsne_data_df: ",type(tsne_data_df))
print("SHAPE: tsne_data_df: ",tsne_data_df.shape)

# colors = {0:'red', 1:'blue', 2:'green'}
# plt.scatter(tsne_data_df['Dimension_x'], tsne_data_df['Dimension_y'], c=tsne_data_df['Score'].apply(lambda x: colors[x]))
# plt.show()

import matplotlib.pyplot as plt
sns.FacetGrid(tsne_data_df, hue = "Score", size = 8).map(plt.scatter, "Dimension_x", "Dimension_y").add_legend()
plt.suptitle("TSNE WITH TF-IDF WEIGHTED WORD2VEC ENCODING OF PROJECT TITLE FEATURE ")
plt.show()
print("PROCESSING TOOK--- %s seconds ---" % (time.time() - start_time))

    


# **Observations:**
# 1. Unstructured large cluster is observed but with high overlapping of points.
# 1. Unable to draw any concrete conlusion.

# <h2> 2.4 TSNE with `BOW, TF-IDF, Avg W2C and TFIDF Weighted W2V` encoding of `project_title` feature </h2>

# ### SAMPLE SIZE = 7000

# In[ ]:


## ALL
sampleSize=7000
print("SAMPLE SIZE: ",sampleSize,"\n\n")
start_time = time.time()
# # ALL_vec=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, price_standardized, teacher_previous_proj_standardized, tfidf_w2v_title_vectors))
# ALL_vec=hstack((school_one_hot, categories_one_hot, sub_categories_one_hot, prefix_one_hot, grade_one_hot, text_title_bow, text_titles_tfidf, avg_w2v_title_vectors, tfidf_w2v_title_vectors, price_standardized, teacher_previous_proj_standardized))
# print("TYPE: ALL_vec: ",type(ALL_vec))
# print("SHAPE: ALL_vec: ",ALL_vec.shape)
# #<class 'scipy.sparse.coo.coo_matrix'>
# ALL_vec_tocsr=ALL_vec.tocsr()

print("CSR VECTORS ARE LOADED FROM DISK FOR SAVING MEMORY BY NOT REQUIRING ABOVE VECTOR GENERATION CODE TO BE RUN AGAIN AND ALSO SAVES RAM BY NOT LOADING THE GLOVE VECTORS IN MEMORY")
ALL_vec_n_samples = ALL_vec_tocsr[0:sampleSize,:]

print("TYPE: ALL_vec_n_samples: ",type(ALL_vec_n_samples))
# <class 'scipy.sparse.csr.csr_matrix'>

ALL_vec_n_samples_toarray=ALL_vec_n_samples.toarray()
print("TYPE: ALL_vec_n_samples_toarray: ",type(ALL_vec_n_samples_toarray))
print("SHAPE: ALL_vec_n_samples_toarray: ",ALL_vec_n_samples_toarray.shape)
# <class 'numpy.ndarray'>


tsne_model = TSNE(n_components = 2, perplexity = 100.0, random_state = 0)
tsne_data = tsne_model.fit_transform(ALL_vec_n_samples_toarray)


print("TYPE: tsne_data: ",type(tsne_data))
print("SHAPE: tsne_data: ",tsne_data.shape)


ALL_Labels=project_data["project_is_approved"]
ALL_Labels_n_samples = ALL_Labels[0: sampleSize]
print("LEN: ALL_Labels_n_samples: ",len(ALL_Labels_n_samples))


tsne_data = np.vstack((tsne_data.T, ALL_Labels_n_samples)).T
# tsne_df_b = pd.DataFrame(tsne_data_b, columns = ("1st_Dim","2nd_Dim","Labels"))
tsne_data_df = pd.DataFrame(tsne_data, columns = ["Dimension_x","Dimension_y","Score"])


print("TYPE: tsne_data-AfterStacking: ",type(tsne_data))
print("TYPE: tsne_data_df: ",type(tsne_data_df))
print("SHAPE: tsne_data_df: ",tsne_data_df.shape)

# colors = {0:'red', 1:'blue', 2:'green'}
# plt.scatter(tsne_data_df['Dimension_x'], tsne_data_df['Dimension_y'], c=tsne_data_df['Score'].apply(lambda x: colors[x]))
# plt.show()

import matplotlib.pyplot as plt
sns.FacetGrid(tsne_data_df, hue = "Score", size = 8).map(plt.scatter, "Dimension_x", "Dimension_y").add_legend()
plt.suptitle("TSNE WITH BOW, TF-IDF, Avg W2C and TFIDF Weighted W2V ENCODING OF PROJECT TITLE FEATURE ")
plt.show()
print("PROCESSING TOOK--- %s seconds ---" % (time.time() - start_time))


# **Observations:**
# 1. Unstructured large cluster is observed but with high overlapping of points.
# 1. Unable to draw any concrete conlusion.

# <h2> 2.5 Summary </h2>

# 1. 3 cells in the Teacher_Prefix column had NaN values and those rows are not considered, thus the number of rows reduced from 109248 to 109245.
# 1. Number of projects thar are approved for funding  92703 , ( 84.85788823287108 %)
# 1. Number of projects thar are not approved for funding  16542 , ( 15.14211176712893 %)
# 1. Every state has greater than 80% success rate in approval.
# 1. DE (Delaware) has the Maximimum approval rate of 89.79 %
# 1. VT (Vermont) has the Minimum approval rate of 80% then followed by DC (District of Columbia).
# 1. Every state has greater than 80% success rate in approval.
# 1. CA (California) state has the maximum number of the project proposals ie 15387.
# 1. There is 50% reduction in the 2nd highest state for project proposals. ie TX (Texas) with 6014.
# 1. VT (Vermont) has the minimum number of project submissions.
# 1. High variation is observed in the state statistics.
# 1. Female teachers have proposed more number of projects than Male teachers.
# 1. Approval rate of Married Female teacher with prefix Mrs. is higher than Male Teachers
# 1. Teacher having Dr. prefix has proposed very less projects
# 1. Interstingly teachers with the highest qualification ie Dr. have lesser approval rate.
# 1. Grades PreK-2 and 3-5 have large number of project proposals.
# 1. Grades 3-5 has the highest approval rate then followed by Grade PreK-2.
# 1. Number of project proposals reduces 4 times from Lower most grade to highest grade. That is, number of project proposal reduces as the grades increases.
# 1. Literacy_Language is the Single most category having highest approval rate of 86%.
# 1. If Literacy_Language is clubed with History_Civics, the approval rate rises from 86% to 89%.
# 1. Math_Science alone has approval rate of around 82%, when clubbed with Literacy_Language, approval rate rises to 86%; when clubbed with AppliedLearning, rises to ~84%, but when clubbed with AppliedLearning, the approval rate reduced to 81%.
# 1. Warmth and Care_Hunger has highest approval rate of ~93%. Thus from this we can conclude that, Non-Science categories have higher approval rate than compared to Science categories including Maths.
# 1. Literacy_Language has the highest number of project than followed by Math_Science.
# 1. Warmth and Care Hunger has the least number of projects.
# 1. Interestingly, from the previous plot, we observed that Warmth and Care_Hunger when clubbed together has the highest approval rate.
# 1. There is high variance between the 2nd and 3rd most number of project categories counts. ie Math_Science (41419) and Health_Sports (14223)
# 1. Projects with sub-category Literacy has the highest number of projects.
# 1. Also Literacy has the highest approval rate of 88%.
# 1. Interestingly, when Literacy is clubbed with any other sub-category, the approval rate is reduced.
# 1. Mathematics alone has lower approval rate than compared to Mathematics clubbed with any other category.
# 1. AppliedSciences College_CareerPrep has least number of project posted and also the with least approval rate.
# 1. Literacy has the highest approved projects.
# 1. Economics has the least approved projects.
# 1. Maximum projects have 4 words, then followed by 5 and 3.
# 1. There are extremely less number of projects that have titles of 1 word and > 10 words.
# 1. In Approved Projects, 25th Quartile lies at 4 words. Median at 5 words and 75th Quartile at 7 words.
# 1. In Rejected Projects, 25th Quartile lies at 3 words. Median at 5 words and 75th Quartile at 6 words.
# 1. In Approved Projects, the gap between the Median and 75th Quartile is large, where as exactly inverse scenario of a larger gap between 25th and Median is observed in Rejected Projects.
# 1. In Approved Projects, titles with more than 11 words are considered as outliers, where as in Rejected Projects, titles with more than 10 words are considered as outliers.
# 1. The number of Approved Projects have a slightly more words in the Title when compared to the Rejected Projects.
# 1. The Median of the Approved Projects is slightly higher than the Median of the Rejected Projects, we can infer that, Approved Projects have higher number of words in the essay.
# 1. Sufficiently large essays are considered as outlier in both Approved and Rejected Projects.
# 1. The PDF of the approved projects is denser for words around 240 to around 470. From this, we can again say that, Approved Projects have higher number of words in the essay.
# 1. The box plots for Price seems very much identical for Approved and Rejected Projects considering the 25th, 50th and 75th Quartiles.
# 1. Interestingly, the Minimum value of the both the box plots for price is very close to 0, and maximum roughly at 800.
# 1. Both the box plots for price have considered higher price as outliers.
# 1. The PDFs of Price are very much similar and overlapping. Thus not much can be understood.
# 1. To some extent, from the PDFs of Price, we can say Projects with higher price are generally not approved.
# 1. It may not be , but at first glance, the distribution of the Teacher's number of previous project seems to look at LOG DISTRIBUTION.
# 1. Minimum Previous Project : 0 - - - - - -Count: 30014
# 1. Maximum Previous Projects : 451 - - - Count: 1
# 1. Roughly 28% of the teachers have applied for the 1st time. Interestingly, 82% of those are accepted considering that they dont have any previous applications.
# 1. We can observe from the box plot of previous projects, that the Median extremely close to 0
# 1. The 75th Percentile of the data is roughly near 20.
# 1. A large section of the data is beyond the 75th Percentile, hence considered as outlier by the plot.
# 1. The PDF of the Approved -> teacher_number_of_previously_posted_projects, is slightly forward than compared to the PDF of Rejected, thus this gives a slight inference that, projects whose Teacher have previous projects will have a slighly approval rate higher than those who didnt.
# 1. Teachers who had roughly 12 previous project were highly approved. ie mean seems to be around 12.
# 1. From the CDF, we can say that teacher had Maximum project between 0-100.
# 1. After roughly 100 project counts, the CDF/PDF seems stable
# 1. A Maximum of around 11,000 summaries is composed of 11 words.
# 1. 2000 and above summaries is comprised of words ranging between 11 - 31.
# 1. Interestingly we can observe that, if the summary word count is NOMINALLY large then Approval rate is higher. However beyond nominal count, ie extremely large summaries have lower chance of approval.
# 1. Maximum project with summary having words between around 11 to 20 are approved.
# 1. Number of Summaries with word counts in the range of around 20-30 is less than compared to 30-40
# 1. Projects with summary having words less than roughly 11 and more than 40 are rejected.
# 1. Extremely large sized summaries which can be seen as outliers are approved. Contrary to what we observed in the above PDF Plot Point no. 1
# 1. The 25th, Median and 75th quartiles are almost similar in both the plots.
# 1. The IQR range cover the summaries with the word count roughly between 11 to 25.
# 1. Only a small portion of 14% have numeric values in the summary.
# 1. Around 89% of the summaries having numeric values are approved than compared to the 84% approval rate of the summaries without the numeric values.
# 1. It is therefore recommended to have a numeric value in summary to increase the chance of approval.
# 1. Visualisation of TSNE with Bag of Words, TF-IDF, Avg Word2Vec, TF-IDF Weighted Word2Vec does not seem to provide an meaningfull observation. Clusters formed didnt provide us any useful insights. High overlapping of datapoints were observed in all the 5 visualization. Hence any other method needs to implemented for understanding this data.
