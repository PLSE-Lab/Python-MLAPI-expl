#!/usr/bin/env python
# coding: utf-8

# ## Imports and Data

# In[ ]:


import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import os,re

from IPython.core.display import display, HTML


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# A couple days ago I have written the [notebooks](https://www.kaggle.com/piterfm/kaggle-ml-ds-survey-2019-who-are-kagglers) with the distribution of answers for all Simple and Multiple Choice Questions. 
# 
# Simultaneously I found out that **28** columns with pattern `TEXT` in column name contain wrong data for analysis. But this data available in another table and contain additionally typed by hand question answering.
# 
# I am trying to show some insight from these responses.
# 
# Some responses are so funny and unique ;)
# 
# Let's go!

# In[ ]:


df = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv', skiprows=(1,1))
df.head()


# In[ ]:


df_full = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', skiprows=(1,1))
df_full.head()


# In[ ]:


questions = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
questions


# In[ ]:


df_col = pd.DataFrame(df.notna().sum()).reset_index()
df_col['new_index'] = df_col['index'].str.split('_').str[0].str[1:].astype('int')
df_col = df_col.sort_values('new_index')
df_col = df_col.set_index('new_index')


# You can see the distribution of the responses to the Selected Choice Questions below. The tail has tiny bars. There are two possible reasons for that:
# 1. Respondents were so tired till the end of the Survey and miss answers.
# 2. The last questions were understandable and all responses are contented with defined categories.

# In[ ]:


plt.figure(figsize=(20, 10))

ax = sns.barplot(x="index", y=0, data=df_col, color='blue')
plt.title('Number of responses for Selected Choice Question', fontsize=22)
plt.xlabel('')
plt.ylabel('')
plt.grid(axis='x', linestyle='-.')
sns.despine()

for patch, value in zip(ax.patches, df_col[0]):
    ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(),
            value,
            ha='center', va='bottom',
            fontsize=12)
plt.xticks(x="index", rotation='vertical',size=14)
plt.yticks(size=14)

plt.show()


# In[ ]:


question_dict = dict(zip(questions.iloc[0].index, questions.iloc[0].values))
question_dict = {k:v[:-18] for k, v in question_dict.items() if 'Selected Choice' in v}
que = ''.join([f'<li>{k+": "+v}</li>' for k, v in question_dict.items()])
display(HTML(f'<h3 style="color:green">List of Selected Choice Question</h3><ol>{que}</ol>'))


# ## Hacker among us! Be carefully!

# During the analysis, I noticed at least one xss-injection in responses. I decided to check all responses. Unfortunately, this attack is alone. He tried it twice for different questions. 
# 
# Curiously, this guy has 0 index in table. **He is 22-24 yers, male and from France**.
# 
# I just show full information about the person who tried this injection. [This is a payload to test for Cross-site Scripting (XSS).](https://abels.xss.ht/)
# 

# In[ ]:


def get_text_data(data, column_name):
    '''
    Return Series
    '''
    data_list = data[data[column_name].notna()][column_name].str.lower().str.strip().str.split(',| and | or |&').tolist()
    return pd.Series(sorted([item.strip() for nested in data_list for item in nested if len(item.strip())>1]))


# In[ ]:


[word for col in df.columns for word in get_text_data(df, col) if 'xss' in word]


# In[ ]:


person_info = [str(i).strip() for i in df_full.iloc[0][df_full.iloc[0].notna()] if i!=-1]
person_info_join = ''.join([f'<li>{i}</li>' for i in person_info])
display(HTML(f'<h3 style="color:green">Haker on Board;)</h3><ol>{person_info_join}</ol>'))


# In[ ]:


def get_search_info(data, col_name, search_word, title):
    q_list = [i for i in get_text_data(data, col_name).unique() if search_word in i]
    q_list_join = ''.join([f'<li>{i}</li>' for i in q_list])
    display(HTML(f'<h3 style="color:green">{title}</h3><ol>{q_list_join}</ol>'))


# In[ ]:


def barplot_top(data, xlabel, ylabel, fs, title=''):
    
    plt.figure(figsize=(10, 10))

    ax = sns.barplot(data, data.index, color='green')
    plt.title('{}\n'.format(title), fontsize=22)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    plt.grid(axis='x', linestyle='-.')
    sns.despine()

    for patch, value in zip(ax.patches, data):
        ax.text(patch.get_width() + 1, patch.get_y() + (patch.get_bbox().y1-patch.get_y())/2,
                value,
                ha="left", va='center',
                fontsize=18)

    new_ytickslabel = []
    for i in ax.get_yticklabels():
        new_ytickslabel.append( ''.join([l + '\n' * (n % 30 == 29) for n, l in enumerate(list(i.get_text()))]) )                 if len(i.get_text())>30 else new_ytickslabel.append(' '*(30-len(i.get_text()))+i.get_text())
    ax.set_yticklabels(new_ytickslabel)
    ax.tick_params(axis='both', which='major', labelsize=18)


# ## Q2: What is your gender?

# In[ ]:


def get_proposed_responses(data, column_name):
    '''
    Return all proposed responses for question
    '''
    column_name = column_name.split('_')[0]
    q_list = list(df_full[column_name].value_counts().index)
    q_list_join = ''.join([f'<li>{i}</li>' for i in q_list])
    display(HTML(f'<h3 style="color:green">Polls Responses suggested by the Authors for {column_name}:</h3>                   <ol>{q_list_join}</ol>'))


# In[ ]:


q2 = 'Q2_OTHER_TEXT'
get_proposed_responses(df_full, q2)


# The first question with text responses has only **49** responses except proposed. 
# 
# Most of them are unique and fanny. I just have chosen some of them:
# * attack helicopter
# * supermacho
# * puppy
# * unicorn
# * alien
# 
# I hidden all responses. If you are interested in, please, push the Button `output` below-right.

# In[ ]:


search_word = ''
list_title = 'Gender Variety'
get_search_info(df, q2, search_word=search_word, title=list_title)


# ## Q5: Select the title most similar to your current role (or most recent title if retired)
# 

# In[ ]:


q5 = 'Q5_OTHER_TEXT'
get_proposed_responses(df_full, q5)


# **1333** responses except **11** proposed and **779** of them are unique. The most of them I can generalize to some of the proposed category, but we are unique and can type exactly position in the company or something like that. 
# 
# For exaple let chose all title `Data Scientist` (see list below). There are 19 different responses. I guess almost every title related to Data Scientist.
# 
# You can see the distribution of the top 10 titles from typed responses on the picture below.

# In[ ]:


search_word = 'data science'
list_title = 'Variety of Title "Data Scientist"'
get_search_info(df, q5, search_word=search_word, title=list_title)


# In[ ]:


xlabel='# of Respondents'
ylabel=''
fntsz=20
title='TOP 10'


# In[ ]:


question5 = get_text_data(df, q5).value_counts()[:10]
title_picture = '{} title most similar to your current role'.format(title)
barplot_top(data=question5, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q9: Select any activities that make up an important part of your role at work

# In[ ]:


columns_multiple = [col for col in list(df_full.columns) if re.search('Part_\d{1,2}$', col)]
multiple_columns_list = [ [col]+col.split('_') for col in columns_multiple ]
ds_multiple = pd.DataFrame(multiple_columns_list).groupby([1])[0].apply(list)


# In[ ]:


def get_proposed_responses_multiple_questions(question, ds_multiple, data):
    question = question.split('_')[0]
    columns_list = ds_multiple[question]
    data = data[columns_list]
    data_list = [ data[col].value_counts().to_dict() for col in data.columns ]
    data_dict = { k:v for values in data_list for k, v in values.items() }
    q_list = list(data_dict.keys())
    q_list_join = ''.join([f'<li>{i}</li>' for i in q_list])
    display(HTML(f'<h3 style="color:green">Polls Responses suggested by the Authors for {question}:</h3>                   <ol>{q_list_join}</ol>'))


# In[ ]:


q9 = 'Q9_OTHER_TEXT'
get_proposed_responses_multiple_questions(q9, ds_multiple, df_full)


# **138** responses except proposed and we can create **192** unique features from it. The most of it is not repeated. 
# 
# I would like to show some of the activities related to **data science**. It looks like a good list to expand your resume.

# In[ ]:


search_word = 'data'
list_title = 'Which type of activities Kagglers does with DATA?'
get_search_info(df, q9, search_word=search_word, title=list_title)


# ## Q12: Who/what are your favorite media sources that report on data science topics?

# In[ ]:


q12 = 'Q12_OTHER_TEXT'
get_proposed_responses_multiple_questions(q12, ds_multiple, df_full)


# **784** responses except proposed. The most of them related to *social networks * and special *education platforms*. 
# 
# Some of the most frequent answers (linkedin, coursera, udemy) related to the next question Q13.
# 
# Some persons mentioned sites. I would like to show the list of sites. Obviously all of them relate to data science in general.
# 
# You can see the distribution of the top 10 favorite media sources from **Kagglers** on the picture below.

# In[ ]:


search_word = 'http'
list_title = 'Sites recommended Kegglers as favorite media Sources:'
get_search_info(df, q12, search_word=search_word, title=list_title)


# In[ ]:


title_picture = '{} favorite Media Sources recommended by Kagglers'.format(title)
question12 = get_text_data(df, q12).value_counts()[:10]
barplot_top(data=question12, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q13: On which platforms have you begun or completed data science courses?

# In[ ]:


q13 = 'Q13_OTHER_TEXT'
get_proposed_responses_multiple_questions(q13, ds_multiple, df_full)


# **1220** responses except proposed. The most of them are other platforms like **Pluralsight** or **Stepik**. 
# 
# Some persons mentioned sites with Data Science courses. I would like to show the list of them.
# 
# You can see the distribution of the top 10 platforms from **Kagglers** on the picture below.

# In[ ]:


search_word = 'http'
list_title = 'Sites recommended Kegglers as platforms with Data Science courses:'
get_search_info(df, q13, search_word=search_word, title=list_title)


# In[ ]:


title_picture = '{} platforms recommended by Kagglers'.format(title)
question13 = get_text_data(df, q13).value_counts()[:10]
barplot_top(data=question13, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q14: What is the primary tool that you use at work or school to analyze data?

# In[ ]:


q14 = 'Q14_OTHER_TEXT'
get_proposed_responses(df_full, q14)


# **1179** responses except proposed. The most of them I can generalize to some of the proposed category. At the same time, the distribution of responses is so diverse. A lot of respondents wrote programming languages or libraries. Most popular answer is **python**.
#  
# For example, the word `jupyter` was mentioned in **38 unique** responses. At the same time, `Jupyter Lab` is present in the defined category.
# 
# You can see the distribution of the top 10 primary tool from **Kagglers** on the picture below.

# In[ ]:


title_picture = '{} Primary Tools by Kagglers'.format(title)
question14 = get_text_data(df, q14).value_counts()[:10]
barplot_top(data=question14, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q16: Which of the following integrated development environments (IDE's) do you use on a regular basis?
# 

# In[ ]:


q16 = 'Q16_OTHER_TEXT'
get_proposed_responses_multiple_questions(q16, ds_multiple, df_full)


# **636** responses except proposed. The most of them I can generalize to some of the proposed category. The most popular answer is **eclipse**.
#  
# You can see the distribution of the top 10 IDE's from **Kagglers** on the picture below.

# In[ ]:


title_picture = "{} IDE's".format(title)
question16 = get_text_data(df, q16).value_counts()[:10]
barplot_top(data=question16, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q17: Which of the following hosted notebook products do you use on a regular basis?

# In[ ]:


q17 = 'Q17_OTHER_TEXT'
get_proposed_responses_multiple_questions(q17, ds_multiple, df_full)


# **314** responses except proposed. The most popular answer is **databricks**. 62 persons wrote this response. At that same time, only 76 persons chosen **Code Ocean**. I guess now a new question category is known.
#  
# You can see the distribution of the top 10 hosted notebook products from **Kagglers** on the picture below.

# In[ ]:


title_picture = "{} Notebook Hosters".format(title)
question17 = get_text_data(df, q17).value_counts()[:10]
barplot_top(data=question17, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q18: What programming languages do you use on a regular basis?

# In[ ]:


q18 = 'Q18_OTHER_TEXT'
get_proposed_responses_multiple_questions(q18, ds_multiple, df_full)


# **1096** responses except proposed. The most popular languages not included in defined categories are **c#** and **scala**. 
# 
# And at least we have **25** respondents still working with **FORTRAN**. **Applause**!
# 
# You can see the distribution of the top 10 languages from **Kagglers** on the picture below.

# In[ ]:


title_picture = "{} Programming Languages".format(title)
question18 = get_text_data(df, q18).value_counts()[:10]
barplot_top(data=question18, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q19: What programming language would you recommend an aspiring data scientist to learn first?

# In[ ]:


q19 = 'Q19_OTHER_TEXT'
get_proposed_responses(df_full, q19)


# Only **122** responses except proposed. The most popular programming language would you recommend to learn first and not included in defined categories is **julia**. Some people recommended python nevertheless it was included in categories.
# 
# **Julia** was recommended more than **TypeScript** at least.
# 
# You can see the distribution of the top 10 languages from **Kagglers** on the picture below.

# In[ ]:


q19 = 'Q19_OTHER_TEXT'
title_picture = "{} recommended Programming Languages".format(title)
question19 = get_text_data(df, q19).value_counts()[:10]
barplot_top(data=question19, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q20: What data visualization libraries or tools do you use on a regular basis?

# In[ ]:


q20 = 'Q20_OTHER_TEXT'
get_proposed_responses_multiple_questions(q20, ds_multiple, df_full)


# **400** responses except proposed. The most of them I can generalize to some of the proposed category. Most popular answer is **tableu**.
#  
# You can see the distribution of the top 10 data visualization libraries from **Kagglers** on the picture below.

# In[ ]:


q20 = 'Q20_OTHER_TEXT'
title_picture = "{} Data Visualization Libraries".format(title)
question20 = get_text_data(df, q20).value_counts()[:10]
barplot_top(data=question20, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Q21: Which types of specialized hardware do you use on a regular basis?

# In[ ]:


q21 = 'Q21_OTHER_TEXT'
get_proposed_responses_multiple_questions(q21, ds_multiple, df_full)


# **65** responses except proposed. The most popular and relevant answer is **fpga**.

# ## Q24: Which of the following ML algorithms do you use on a regular basis?

# In[ ]:


q24 = 'Q24_OTHER_TEXT'
get_proposed_responses_multiple_questions(q24, ds_multiple, df_full)


# **339** responses except proposed.
# 
# The TOP 3 ML algorithm from **Kagglers**:
# 1. **svm**
# 2. **knn** 
# 3. **k-means**

# ## Q25: Which categories of ML tools do you use on a regular basis?

# In[ ]:


q25 = 'Q25_OTHER_TEXT'
get_proposed_responses_multiple_questions(q25, ds_multiple, df_full)


# **140** responses except proposed.
# 
# The most popular response from **Kagglers** is **DataRobot**.

# ## Q26: Which categories of computer vision methods do you use on a regular basis?

# In[ ]:


q26 = 'Q26_OTHER_TEXT'
get_proposed_responses_multiple_questions(q26, ds_multiple, df_full)


# Only **37** responses except proposed. I did not find interesting information inside them.

# ## Q27: Which of the following natural language processing (NLP) methods do you use on a regular basis? 

# In[ ]:


q27 = 'Q27_OTHER_TEXT'
get_proposed_responses_multiple_questions(q27, ds_multiple, df_full)


# Only **33** responses except proposed. I did not find interesting information inside them.

# ## Q28: Which of the following machine learning frameworks do you use on a regular basis? 

# In[ ]:


q28 = 'Q28_OTHER_TEXT'
get_proposed_responses_multiple_questions(q28, ds_multiple, df_full)


# **289** responses except proposed.
# 
# The TOP 3 ML Frameworks from **Kagglers**:
# 1. **catboost**
# 2. **h2o** 
# 3. **matlab**

# ## Q29: Which of the following cloud computing platforms do you use on a regular basis? 

# In[ ]:


q29 = 'Q29_OTHER_TEXT'
get_proposed_responses_multiple_questions(q29, ds_multiple, df_full)


# **133** responses except proposed.
# 
# The most popular response from **Kagglers** is **Digital Ocean**

# ## Q30: Which specific cloud computing products do you use on a regular basis? 

# In[ ]:


q30 = 'Q30_OTHER_TEXT'
get_proposed_responses_multiple_questions(q30, ds_multiple, df_full)


# **183** responses except proposed.
# 
# The most popular response from **Kagglers** is **AWS SageMaker**. AWS is mentioned in most of the responses.

# In[ ]:


q30 = 'Q30_OTHER_TEXT'
search_word = 'aws'
list_title = 'Specific AWS Cloud Computing Products used Kegglers:'
get_search_info(df, q30, search_word=search_word, title=list_title)


# ## Q31: Which specific big data / analytics products do you use on a regular basis?

# In[ ]:


q31 = 'Q31_OTHER_TEXT'
get_proposed_responses_multiple_questions(q31, ds_multiple, df_full)


# **188** responses except proposed.
# 
# The most popular response from **Kagglers** is **Snowflake**.

# ## Q32: Which of the following machine learning products do you use on a regular basis?

# In[ ]:


q32 = 'Q32_OTHER_TEXT'
get_proposed_responses_multiple_questions(q32, ds_multiple, df_full)


# **172** responses except proposed.
# 
# The TOP 3 ML Products from **Kagglers**:
# 1. **DataRobot**
# 2. **Kmine** 
# 3. **IBM Watsons Studio**

# ## Q33: Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?

# In[ ]:


q33 = 'Q33_OTHER_TEXT'
get_proposed_responses_multiple_questions(q33, ds_multiple, df_full)


# **89** responses except proposed.
# 
# The TOP 3 Automated ML Tool from **Kagglers**:
# 1. **H2O Automl** 
# 2. **IBM Autoai** 
# 3. **Prevision.io**

# ## Q34: Which of the following relational database products do you use on a regular basis?

# In[ ]:


q34 = 'Q34_OTHER_TEXT'
get_proposed_responses_multiple_questions(q34, ds_multiple, df_full)


# **255** responses except proposed. Most popular answer is **Snowflake**.
#  
# You can see the distribution of the top 5 Data Relational Databases**** from **Kagglers** on the picture below.

# In[ ]:


q34 = 'Q34_OTHER_TEXT'
title = 'TOP 5'
title_picture = "{} Data Relational DataBases".format(title)
question28 = get_text_data(df, q34).value_counts()[:5]
barplot_top(data=question28, xlabel=xlabel, ylabel=ylabel, fs=fntsz, title=title_picture)


# ## Conclusions

# After a deep analysis of responses for the different question I can conclude:
# 
# 1. **Analysis of text responses lets find XSS attacker.**
# 2. **A lot of question with text responses tires respondents.**
# 3. **Good questions didn't need additional responses.**
# 
# My Top6 suggestions:
# 
# 1. Change the order of Q12 and 13. A lot of answers about media resources contain answers about platforms.
# 2. Add responce `Lecturer\Teacher` to Q5. 
# 3. At least `IntelliJ IDEA` and `Eclipse` deserve attention for Q16.
# 4. Add `Julia` instead of `TypeScript` for Q19.
# 5. Remove text responses for Q21.
# 6. Include `svm`, `knn` and `k-means` to responses for Q24.
# 
# You can find my other notebook with the distribution of answers for all Simple and Multiple Choice Questions without Text Responce [here](https://www.kaggle.com/piterfm/kaggle-ml-ds-survey-2019-who-are-kagglers).
# 
# Thanks for attention!
# 

# In[ ]:




