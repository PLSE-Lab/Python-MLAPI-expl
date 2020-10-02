#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


questions_only = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
mul_choices = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
other_text = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
schema = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')


# In[ ]:


mul_choices = mul_choices.fillna('')
mul_choices = mul_choices.replace({'-1': ''}, regex=True)


# In[ ]:


pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows',5000)


# In[ ]:


questions_only.head()


# In[ ]:


mul_choices.head()


# In[ ]:


Q5_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q5',col)]]
Q9_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q9',col)]]
Q12_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q12',col)]]
Q13_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q13',col)]]
Q14_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q14',col)]]
Q16_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q16',col)]]
Q17_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q17',col)]]
Q18_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q18',col)]]
Q19_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q19',col)]]
Q20_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q20',col)]]
Q21_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q21',col)]]
Q24_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q24',col)]]
Q25_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q25',col)]]
Q26_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q26',col)]]
Q27_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q27',col)]]
Q28_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q28',col)]]
Q29_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q29',col)]]
Q30_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q30',col)]]
Q31_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q31',col)]]
Q32_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q32',col)]]
Q33_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q33',col)]]
Q34_data =  mul_choices[[col for col in mul_choices.columns if re.match(r'^Q34',col)]]
#Q9_data = Q9_data.fillna('')


# In[ ]:


mul_choices['Q5_combined'] = Q5_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q9_combined'] = Q9_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q12_combined'] = Q12_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q13_combined'] = Q13_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q14_combined'] = Q14_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q16_combined'] = Q16_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q17_combined'] = Q17_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q18_combined'] = Q18_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q19_combined'] = Q19_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q20_combined'] = Q20_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q21_combined'] = Q21_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q24_combined'] = Q24_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q25_combined'] = Q25_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q26_combined'] = Q26_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q27_combined'] = Q27_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q28_combined'] = Q28_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q29_combined'] = Q29_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q30_combined'] = Q30_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q31_combined'] = Q31_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q32_combined'] = Q32_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q33_combined'] = Q33_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
mul_choices['Q34_combined'] = Q34_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 


# In[ ]:


cols = ['Q1','Q2','Q3','Q4','Q5_combined','Q6','Q7','Q8','Q9_combined','Q10','Q11','Q12_combined','Q13_combined','Q14_combined','Q15','Q16_combined','Q17_combined','Q18_combined','Q19_combined','Q20_combined','Q21_combined','Q22','Q23','Q24_combined','Q25_combined','Q26_combined','Q27_combined','Q28_combined','Q29_combined','Q30_combined','Q31_combined','Q32_combined','Q33_combined','Q34_combined']


# In[ ]:


mul_wo_question = mul_choices.loc[1:,cols].copy()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q34_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q34'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q33_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q33'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q32_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q32'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q31_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q31'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q30_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q30'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q29_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q29'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q28_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q28'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q27_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q27'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q26_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q26'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q25_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q25'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q24_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q24'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q21_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q21'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q20_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q20'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q19_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q19'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q18_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q18'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q17_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q17'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q16_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q16'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q14_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q14'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q13_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q13'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q5_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(mul_choices['Q5'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q12_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q12'][0])
plt.axis("off")
plt.show()


# In[ ]:


all_text = " ".join(rev.strip() for rev in mul_wo_question.Q9_combined)
wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white").generate(all_text)
plt.figure(figsize = (12,6))
plt.imshow(wordcloud, interpolation='nearest',aspect = 'auto')
plt.title(questions_only['Q9'][0])
plt.axis("off")
plt.show()


# In[ ]:


def age_categorization(x):
    if (x == '18-21'):
        return 'young'
    if (x == '22-24') | (x == '25-29')|(x == '35-39')| (x =='40-44'):
        return 'mid_age'
    return 'above_mid_age'
mul_wo_question['Q1'] = mul_wo_question.Q1.apply(age_categorization)


# In[ ]:


mul_wo_question.Q4.value_counts().nlargest(10).plot(kind='bar', title='education', figsize=(15, 5),color='black')


# In[ ]:


mul_wo_question.Q3.value_counts().nlargest(10).plot(kind='bar', title='Countrys', figsize=(15, 5),color='g')


# In[ ]:


mul_wo_question.Q1.value_counts().plot(kind='bar', title='Count (Q1)', figsize=(15, 5),color = 'grey')


# In[ ]:


mul_wo_question.Q2.value_counts().plot(kind='bar', title='Count (Q2)', figsize=(15, 5))


# In[ ]:


mul_wo_question.groupby(['Q1', 'Q2']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(15, 5))


# Most of them are from **Masters** background

# In[ ]:


ax = mul_wo_question.groupby(['Q5', 'Q4']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(17, 5))
ax.set_xlabel('Designation')


# All the companys prefering Master's degree holders

# In[ ]:


ax = mul_wo_question.groupby(['Q6', 'Q4']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("company Size")


# In[ ]:


ax = mul_wo_question.groupby(['Q7', 'Q6']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("Number of data Science roles")


# In[ ]:


ax = mul_wo_question.groupby(['Q5', 'Q22']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7),title = "Types of jobs with ML experience")
ax.set_xlabel("Jobs")

