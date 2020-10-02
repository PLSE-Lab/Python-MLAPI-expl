#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ### Import plotting libraries ###

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
sns.set_style("white")


# ### Import modeling libraries ###

# In[ ]:


import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


# In[ ]:


import re
import nltk
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))


# In[ ]:


train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')


# In[ ]:


metrics = []


# In[ ]:


print('There are {} records in train'.format(train_set.shape[0]))
print('There are {} records in train'.format(test_set.shape[0]))


# In[ ]:


target = 'is_duplicate'
ID = 'id'


# ### Find nulls in questions ###

# In[ ]:


def find_nulls(df, column):
    res = df.ix[pd.isnull(df[column])]
    percentages = []
    percentages.append(len(res[column]))
    percentages.append(df.shape[0] - res.shape[0])
    percentages = pd.DataFrame(percentages, columns=[column], index=['Nulls', 'Non nulls'])
    return percentages


# In[ ]:


def plot_bar_chart(data, column):
    N = data.shape[0]
    ind = np.arange(N)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,3))
    rects = ax.bar(ind, data[column])
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(data.index)
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
    plt.show()
    plt.close()


# ### Find nulls in question1 ###

# In[ ]:


res = find_nulls(train_set, 'question1')
res = pd.concat([res, find_nulls(train_set, 'question2')], axis=1)
res = res.rename(columns={
    'question1': 'train_q1',
    'question2': 'train_q2'
})
res = pd.concat([res, find_nulls(test_set, 'question1')], axis=1)
res = pd.concat([res, find_nulls(test_set, 'question2')], axis=1)
res = res.rename(columns={
    'question1': 'test_q1',
    'question2': 'test_q2'
})
res


# ### Fill the NA's ###

# In[ ]:


train_set['question1'] = train_set['question1'].fillna('')
train_set['question2'] = train_set['question2'].fillna('')

test_set['question1'] = test_set['question1'].fillna('')
test_set['question2'] = test_set['question2'].fillna('')


# In[ ]:


word_counts = {}

def find_word_counts(word_counts, tokenlist):
    for token in tokenlist:
        if token in word_counts:
            word_counts[token] = word_counts[token] + 1
        else:
            word_counts[token] = 1
    return word_counts


# ### Use regex to find quoted words that can be converted to complete words  ###

# In[ ]:


train_set['q1_quotes'] = train_set['question1'].str.extract('(\w+\'\w+)\s.*')
train_set['q2_quotes'] = train_set['question2'].str.extract('(\w+\'\w+)\s.*')


# ### Pie chart to represent distributions of  sentence with quotes against unquotes ###

# In[ ]:


quote_counts = []
quote_counts.append(train_set.ix[pd.notnull(train_set['q1_quotes'])].shape[0])
quote_counts.append(train_set.ix[pd.isnull(train_set['q1_quotes'])].shape[0])
quote_counts = pd.DataFrame(quote_counts, columns=['counts'], 
                            index=['Quotes', 'Unquotes'])

colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
 
plt.pie(quote_counts['counts'], labels=quote_counts.index, colors=colors, 
        explode=(1, 0), autopct='%.1f%%',)
plt.axis('equal')
plt.show()
plt.close()


# In[ ]:


quote_counts = []
quote_counts.append(train_set.ix[pd.notnull(train_set['q2_quotes'])].shape[0])
quote_counts.append(train_set.ix[pd.isnull(train_set['q2_quotes'])].shape[0])
quote_counts = pd.DataFrame(quote_counts, columns=['counts'], 
                            index=['Quotes', 'Unquotes'])

colors = ['lightblue', 'coral']
explode = (0.1, 0)
 
plt.pie(quote_counts['counts'], labels=quote_counts.index, colors=colors, 
        explode=(1, 0), autopct='%.1f%%',)
plt.axis('equal')
plt.show()
plt.close()


# In[ ]:


re.match('^(\w+\'\w+)\s.*', "What's causing someone to be jealous?").group(1)


# In[ ]:


duplicates = train_set.ix[train_set['is_duplicate'] == 1]
non_duplicates = train_set.ix[train_set['is_duplicate'] == 0]


# In[ ]:


def find_matching_words(x):
    wq1 = str(x['question1']).lower().split(' ')
    wq2 = str(x['question2']).lower().split(' ')
    matches = set(wq1).intersection(set(wq2))
    return len(matches)


# In[ ]:


re.match('.*(\?\s).*', "Is Kristen Stewart a bad actress?  Why or why not?").group(1)


# In[ ]:


def find_bi_grams(question):
    question = question.replace
    tokens = question.lower().split(' ')
    pos_tags = nltk.pos_tag(tokens)
    bigrams = []
    
    print(len(pos_tags))
    for i in range(len(pos_tags)):
        if i == len(pos_tags) - 1:
            continue
        bigrams.append((pos_tags[i][0], pos_tags[i+1][0]))
        #if pos_tags[i][1] == 'JJ' and pos_tags[i+1][1] == 'NN':
        #    bigrams.append((pos_tags[i][0], pos_tags[i+1][0]))
    print('Done')
    return bigrams


# In[ ]:


train_set['has_questionmark'] = train_set['question1'].str.extract('.*(\?\s).*')


# In[ ]:


train_set.ix[pd.notnull(train_set['has_questionmark'])]['question1'].iloc[1]


# In[ ]:


train_set.shape


# In[ ]:


403717 / 404290


# In[ ]:


train_set['q1_bigrams'] = train_set['question1'].map(find_bi_grams)


# In[ ]:


copy_frame = train_set.copy(deep=True)


# In[ ]:


def replace_q1_puncts(sentence):
    sentence = sentence.replace('What\'s', 'What is')
    sentence = sentence.replace('What\'re', 'What are')
    sentence = sentence.replace('Who\'s', 'Who is')
    sentence = sentence.replace('who\'re', 'who are')
    sentence = sentence.replace('How\'s', 'How is')
    sentence = sentence.replace('don\'t', 'do not')
    sentence = sentence.replace('Don\'t', 'Do not')
    sentence = sentence.replace('can\'t', 'can not')
    sentence = sentence.replace('doesn\'t', 'does not')
    sentence = sentence.replace('does\'t', 'does not')
    sentence = sentence.replace('didn\'t', 'did not')
    sentence = sentence.replace('isn\'t', 'is not')
    sentence = sentence.replace('Isn\'t', 'Is not')
    sentence = sentence.replace('won\'t', 'will not')
    sentence = sentence.replace('haven\'t', 'have not')
    sentence = sentence.replace('aren\'t', 'are not')
    sentence = sentence.replace('hasn\'t', 'has not')
    sentence = sentence.replace('shouldn\'t', 'should not')
    sentence = sentence.replace('Shouldn\'t', 'Should not')
    sentence = sentence.replace('wouldn\'t', 'would not')
    sentence = sentence.replace('wasn\'t', 'was not')
    sentence = sentence.replace('couldn\'t', 'could not')
    sentence = sentence.replace('It\'s', 'It is')
    sentence = sentence.replace('that\'s', 'that is')
    sentence = sentence.replace('I\'m', 'I am')
    sentence = sentence.replace('I\'ve', 'I have')
    sentence = sentence.replace('I\'ll', 'I will')
    sentence = sentence.replace('you\'ve', 'you have')
    sentence = sentence.replace('you\'re', 'you are')
    sentence = sentence.replace('there\'s', 'there is')
    sentence = sentence.replace('they\'re', 'they are')
    return sentence


# In[ ]:


def replace_q2_puncts(sentence):
    sentence = sentence.replace('What\'s', 'What is')
    sentence = sentence.replace('don\'t', 'do not')
    sentence = sentence.replace('Don\'t', 'Do not')
    sentence = sentence.replace('can\'t', 'can not')
    sentence = sentence.replace('I\'m', 'I am')
    sentence = sentence.replace('doesn\'t', 'does not')
    sentence = sentence.replace('you\'ve', 'you have')
    sentence = sentence.replace('didn\'t', 'did not')
    sentence = sentence.replace('I\'ve', 'I have')
    sentence = sentence.replace('isn\'t', 'is not')
    sentence = sentence.replace('you\'re', 'you are')
    sentence = sentence.replace('won\'t', 'will not')
    sentence = sentence.replace('they\'re', 'they are')
    sentence = sentence.replace('haven\'t', 'have not')
    sentence = sentence.replace('aren\'t', 'are not')
    sentence = sentence.replace('hasn\'t', 'has not')
    sentence = sentence.replace('shouldn\'t', 'should not')
    sentence = sentence.replace('wouldn\'t', 'would not')
    sentence = sentence.replace('wasn\'t', 'was not')
    sentence = sentence.replace('couldn\'t', 'could not')
    
    
    sentence = sentence.replace('Doesn\'t', 'Does not')
    sentence = sentence.replace('Wouldn\'t', 'Would not')
    sentence = sentence.replace('weren\'t', 'were not')
    sentence = sentence.replace('where\'s', 'where is')
    return sentence


# In[ ]:


copy_frame['question1'] = copy_frame['question1'].map(replace_q1_puncts)


# In[ ]:


copy_frame['question2'] = copy_frame['question2'].map(replace_q2_puncts)


# In[ ]:


copy_frame['q1_quotes'] = copy_frame['question1'].str.extract('(\w+\'\w+)\s.*')
copy_frame['q2_quotes'] = copy_frame['question2'].str.extract('(\w+\'\w+)\s.*')


# In[ ]:


copy_frame['q1_length'] = copy_frame['question1'].apply(lambda x: len(x.split(' ')))
copy_frame['q2_length'] = copy_frame['question2'].apply(lambda x: len(x.split(' ')))
copy_frame['words_shared'] = copy_frame.apply(lambda x: find_matching_words(x), axis=1)


# In[ ]:


def find_words_not_in_stops(x):
    q1_not_stops = set(x['question1'].split(' ')).difference(stops)
    q2_not_stops = set(x['question2'].split(' ')).difference(stops)
    commons = q1_not_stops.intersection(q2_not_stops)
    return commons


# In[ ]:


copy_frame['words_shared_without_stops'] = copy_frame.apply(lambda x:find_words_not_in_stops(x), axis=1)


# In[ ]:


copy_frame['words_shared_without_stops_len'] = copy_frame['words_shared_without_stops'].apply(len)


# In[ ]:


quote_counts = []
quote_counts.append(copy_frame.ix[pd.notnull(copy_frame['q1_quotes'])].shape[0])
quote_counts.append(copy_frame.ix[pd.isnull(copy_frame['q1_quotes'])].shape[0])
quote_counts = pd.DataFrame(quote_counts, columns=['counts'], 
                            index=['Quotes', 'Unquotes'])

colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
 
plt.pie(quote_counts['counts'], labels=quote_counts.index, colors=colors, 
        explode=(1, 0), autopct='%.1f%%',)
plt.axis('equal')
plt.show()
plt.close()


# In[ ]:


quote_counts = []
quote_counts.append(copy_frame.ix[pd.notnull(copy_frame['q2_quotes'])].shape[0])
quote_counts.append(copy_frame.ix[pd.isnull(copy_frame['q2_quotes'])].shape[0])
quote_counts = pd.DataFrame(quote_counts, columns=['counts'], 
                            index=['Quotes', 'Unquotes'])

colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
 
plt.pie(quote_counts['counts'], labels=quote_counts.index, colors=colors, 
        explode=(1, 0), autopct='%.1f%%',)
plt.axis('equal')
plt.show()
plt.close()


# In[ ]:


train_features = ['q1_length', 'q2_length', 'words_shared']
X = copy_frame[train_features]
y = copy_frame[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)
log_loss_score = log_loss(y_test, y_proba)
metrics.append(log_loss_score)


# In[ ]:


pd.DataFrame(metrics, columns=['logloss'])


# In[ ]:


def preprocess():
    test_set['question1'] = test_set['question1'].map(replace_q1_puncts)
    test_set['question1'] = test_set['question1'].map(replace_q2_puncts)
    test_set['q1_length'] = test_set['question1'].apply(lambda x: len(x.split(' ')))
    test_set['q2_length'] = test_set['question2'].apply(lambda x: len(x.split(' ')))
    test_set['words_shared'] = test_set.apply(lambda x: find_matching_words(x), axis=1)


# In[ ]:


#preprocess()


# In[ ]:


def generate_predictions():
    test_ids = test_set['test_id']
    predictions = clf.predict_proba(test_set[train_features])

    submission = pd.DataFrame(test_ids)

    prediction_set = []
    for i in range(len(predictions)):
        prediction_set.append(predictions[i][1])
    
    prediction_set = pd.DataFrame(prediction_set, columns=[target])
    submission = pd.concat([submission, prediction_set], axis=1)
    return submission


# In[ ]:


#submission = generate_predictions()


# In[ ]:


#print(set(pd.isnull(submission[target])))
#submission.to_csv("submission_quotes_replaced.csv", index=False)

