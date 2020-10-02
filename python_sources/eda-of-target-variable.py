#!/usr/bin/env python
# coding: utf-8

# This is my first public kernetl. Please upvote if you find it useful.
# Thanks to https://www.kaggle.com/ragnar123/exploratory-data-analysis-and-baseline for providing the starter code.

# In[ ]:


import numpy as np, pandas as pd, os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns, matplotlib.pyplot as plt, matplotlib.patches as patches, plotly.offline as py, plotly.graph_objs as go, plotly.express as px, lightgbm as lgb, plotly.figure_factory as ff, gc, json
from plotly import tools, subplots
py.init_notebook_mode(connected = True)
pd.set_option('max_columns', 1000)
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from keras.preprocessing import text, sequence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


# In[ ]:


path = '/kaggle/input/tensorflow2-question-answering/'
train_path = 'simplified-nq-train.jsonl'
test_path = 'simplified-nq-test.jsonl'
sample_submission_path = 'sample_submission.csv'

def read_data(path, sample = True, chunksize = 10000):
    if sample == True:
        df = []
        with open(path, 'rt') as reader:
            for i in range(chunksize):
                df.append(json.loads(reader.readline()))
        df = pd.DataFrame(df)
        print('Our sampled dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
    else:
        df = pd.read_json(path, orient = 'records', lines = True)
        print('Our dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
        gc.collect()
    return df

train = read_data(path+train_path, sample = True)
test = read_data(path+test_path, sample = False)
train.head()


# In[ ]:


test.head()


# In[ ]:


sample_submission = pd.read_csv(path + sample_submission_path)
print('Our sample submission has {} rows'.format(sample_submission.shape[0]))
sample_submission.head()


# In[ ]:


index=0
question_text_0 = train.loc[index, 'question_text']
print('The question is : ', question_text_0)
document_text_0 = train.loc[index, 'document_text'].split()
print('Length of wiki article is : ', len(document_text_0))
long_answer_candidates_0 = train.loc[index, 'long_answer_candidates']
print('Count of long answer candidates is :', len(long_answer_candidates_0))
annotations_0 = train['annotations'][index][0]
print('Ground truth is : ', annotations_0)

if annotations_0['short_answers']!=[]:
    print('The short answer is : ', " ".join(document_text_0[annotations_0['short_answers'][0]['start_token']:annotations_0['short_answers'][0]['end_token']]))
else:
    print('Short answer doesn\'t exist')
if annotations_0['long_answer']['end_token']>0:
    print('The long answer is : ', " ".join(document_text_0[long_answer_candidates_0[annotations_0['long_answer']['candidate_index']]['start_token']:long_answer_candidates_0[annotations_0['long_answer']['candidate_index']]['end_token']]))
else:
    print('Long answer doesn\'t exist')


# In[ ]:


yes_no_answer = []
for i in range(len(train)):
    yes_no_answer.append(train['annotations'][i][0]['yes_no_answer'])
yes_no_answer = pd.DataFrame({'yes_no_answer': yes_no_answer})

yes_no_answer['yes_no_answer'].value_counts()


# In[ ]:


def extract_target_variable(df):
        short_answer = []
        for i in range(len(df)):
            short = df['annotations'][i][0]['short_answers']
            if short == []:
                yes_no = df['annotations'][i][0]['yes_no_answer']
                if yes_no == 'NO' or yes_no == 'YES':
                    short_answer.append([yes_no, -1, -1. -1])
                else:
                    short_answer.append(['EMPTY', -1, -1, -1])
            else:
                short = short[0]
                st = short['start_token']
                et = short['end_token']
                short_answer.append([f'{st}'+':'+f'{et}', st, et, et-st])
        short_answer = pd.DataFrame(short_answer, columns=['short_answer', 'short_start', 'short_end', 'short_length'])
        
        long_answer= []
        for i in range(len(df)):
            long = df['annotations'][i][0]['long_answer']
            if long['start_token'] == -1:
                long_answer.append(['EMPTY',-1,-1, -1])
            else:
                st = long['start_token']
                et = long['end_token']
                long_answer.append([f'{st}'+':'+f'{et}',st,et, et-st])
        long_answer = pd.DataFrame(long_answer, columns=['long_answer', 'long_start', 'long_end', 'long_length'])
        answer = pd.concat([long_answer, short_answer], axis=1)
        return answer

answer = extract_target_variable(train)
answer.head(10)


# In[ ]:


answer['diff_start'] = answer['short_start'] - answer['long_start']
answer['diff_end'] = answer['long_end'] - answer['short_end']


# In[ ]:


answer.head(10)


# In[ ]:


n, bins, patches = plt.hist(x=answer[answer['long_length']>-1]['long_length'], bins=1000)
plt.grid(axis='y')
plt.xlabel('Length of Long Answer')
plt.ylabel('Frequency')
plt.xscale('log')
#Majority of long answers have length between 30 and 200


# In[ ]:


n, bins, patches = plt.hist(x=answer[answer['short_length']>-1]['short_length'], bins=125)
plt.grid(axis='y')
plt.xlabel('Length of Short Answer')
plt.ylabel('Frequency')
plt.xscale('log')
#Majority of short answers have length less than 20


# In[ ]:


n, bins, patches = plt.hist(x=answer[(answer['short_length']>-1)&(answer['long_length']>-1)]['diff_start'], bins=1000)
plt.grid(axis='y')
plt.xlabel('Difference in start of short and long answer')
plt.ylabel('Frequency')
plt.xscale('log')
#Majority of different in start of short and long answers is less than 20


# In[ ]:


n, bins, patches = plt.hist(x=answer[(answer['short_length']>-1)&(answer['long_length']>-1)]['diff_end'], bins=1000)
plt.grid(axis='y')
plt.xlabel('Difference in end of short and long answer')
plt.ylabel('Frequency')
plt.xscale('log')
#Majority of difference in start of short and long answers is less than 100


# In[ ]:


answer[answer['short_answer']=='EMPTY'].shape[0]/answer.shape[0] #Percent of cases where short answer is empty


# In[ ]:


answer[answer['long_answer']=='EMPTY'].shape[0]/answer.shape[0] #Percent of cases where long answer is empty


# In[ ]:


answer[(answer['short_answer']!='EMPTY') & (answer['long_answer']=='EMPTY')].shape[0]/answer.shape[0] #Percent of cases where long answer is empty but short answer is not


# In[ ]:


answer[(answer['short_answer']=='EMPTY') & (answer['long_answer']!='EMPTY')].shape[0]/answer.shape[0] #Percent of cases where short answer is empty but long answer is not


# In[ ]:


plt.scatter(answer[(answer['short_length']>-1)&(answer['long_length']>-1)]['long_length'], answer[(answer['short_length']>-1)&(answer['long_length']>-1)]['short_length'], s=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Long Length')
plt.ylabel('Short Length')
plt.show()
print(np.corrcoef(answer[(answer['short_length']>-1)&(answer['long_length']>-1)]['long_length'], answer[(answer['short_length']>-1)&(answer['long_length']>-1)]['short_length'])[0][1])
#No correlation between length of long answer and short answer

