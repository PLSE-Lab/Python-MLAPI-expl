#!/usr/bin/env python
# coding: utf-8

# # Roberta error analysis
# It is often the case in deep learning that we do abit of both random search and gradient descent. I realise alot of people including me do alot of random search but rarely do output analysis to figure out how to improve the model(aka gradient descent). The analysis here are done using [Abhishek Thakur's](https://www.kaggle.com/abhishek) v2 folds.

# # Experimental procedure
# Model : Roberta  
# Training procedure: secret but very similar to [Abhishek Thakur's](https://www.kaggle.com/abhishek) original kernel  
# The outputs are the 5 out of fold prediction. Note that I used the best validation fold as the saved model so it will overfit this training data abit.  
# This can be seen as the expectation on the training set when your model deals with new unseen examples.  

# # Import stuff

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import re
CSV_PATH = '../input/tweet5foldoutputs'


# # Reading and combining csv

# In[ ]:


df_ls = [] 
for i in range(5):
    df = pd.read_csv(f'{CSV_PATH}/fold_{i}.csv')
    df_ls.append(df)
df = pd.concat(df_ls)
df = df.drop(['Unnamed: 0'], axis=1)
df.head()


# # Calculating Jaccard scores

# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def calculate_jac(df):
    ls = []
    for row in df.iterrows():
        row = row[1]
        a = row.selected_text
        b = row.selected_text_out
        jac = jaccard(a,b)
        ls.append(jac)
    return ls


# In[ ]:


jaccard_scores = calculate_jac(df)
df['jaccard'] = jaccard_scores


# ## Jaccard2 -> Alternative calculation more resilient to noise??
# We will name this as **jaccard2**  
# Instead of splitting just on spaces we split both on spaces AND punctuation  
# Then we calculate the jaccard scores

# In[ ]:


df['selected_text_2'] = df['selected_text'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]",x))
df['selected_text_2_out'] = df['selected_text_out'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]",x))


# In[ ]:


def jaccard2(a,b):
    ls_a = []
    ls_b = []
    for word in a:
        ls_a.append(word.lower())
    for word in b:
        ls_b.append(word.lower())
    a = set(ls_a) 
    b = set(ls_b)
    c = a.intersection(b)
    denom = (len(a) + len(b) - len(c))
    if denom == 0:
        return 1
    return float(len(c)) / denom
def calculate_jac2(df):
    ls = []
    for row in df.iterrows():
        row = row[1]
        a = row.selected_text_2
        b = row.selected_text_2_out
        jac = jaccard2(a,b)
        ls.append(jac)
    return ls


# In[ ]:


df['jaccard2'] = calculate_jac2(df)


# # Analysis
# ## Final Jaccard Scores

# In[ ]:


print(df['jaccard'].mean())


# ## Final Jaccard2 Scores

# In[ ]:


print(df['jaccard2'].mean())


# ## Jaccard per fold
# Notice some difference even within each fold

# In[ ]:


for i in range(5):
    df_i = df[df.kfold == i]
    jac = df_i['jaccard'].mean()
    print(f'For out of fold {i}, jaccard is {jac}')


# In[ ]:


for i in range(5):
    df_i = df[df.kfold == i]
    jac = df_i['jaccard2'].mean()
    print(f'For out of fold {i}, jaccard2 is {jac}')


# # See why the model fails  
# Since this is a conditional probability of predicting selected text given sentiment we should analyse by each sentiment

# In[ ]:


df_positive = df[df['sentiment']=='positive']
df_negative = df[df['sentiment']=='negative']
df_neutral = df[df['sentiment']=='neutral']


# ## Jaccard scores for each sentiment

# In[ ]:


print('Jaccard for positive',df_positive['jaccard'].mean())
print('Jaccard for negative',df_negative['jaccard'].mean())
print('Jaccard for neutral',df_neutral['jaccard'].mean())


# In[ ]:


print('Jaccard2 for positive',df_positive['jaccard2'].mean())
print('Jaccard2 for negative',df_negative['jaccard2'].mean())
print('Jaccard2 for neutral',df_neutral['jaccard2'].mean())


# # Bad examples of each sentiment

# In[ ]:


def print_bad_examples(df,num=10):
    df = df.sort_values(by=['jaccard'])
    for i in range(num):
        row = df.iloc[i]
        print('text:             ', row.text.strip())
        print('selected text:    ', row.selected_text.strip())
        print('my selected text: ',row.selected_text_out.strip())
        print('-'*50)


# In[ ]:


def print_bad_examples2(df,num=10):
    df = df.sort_values(by=['jaccard2'])
    for i in range(num):
        row = df.iloc[i]
        print('text:             ', row.text.strip())
        print('selected text:    ', row.selected_text.strip())
        print('my selected text: ',row.selected_text_out.strip())
        print('-'*50)


# ## Bad examples for positive sentiments
# Seems to me here that even for bad examples the model performs fairly well and most of this can be attributed to human labelling issues.  
# 'Best Model' that wins the competition might not be objectively useful as it might learn the bias of the human labellers.  
# For example `good` and `nice` are both acceptable to me in the first/worst example!  

# In[ ]:


print('Jaccard Worst Examples positive sentiment \n')
print_bad_examples(df_positive)


# In[ ]:


print('Jaccard2 Worst Examples positive sentiment \n')
print_bad_examples2(df_positive)


# ## Bad examples for negative sentiments
# Similar conclusion to positive. But perhaps when there are two or more sentiments in the model, the model fails to pick the one that is more intense?  
# For example in `****` vs `pissed` and `stupid` vs `i hate`

# In[ ]:


print('Jaccard Worst Examples negative sentiment \n')
print_bad_examples(df_negative)


# In[ ]:


print('Jaccard2 Worst Examples negative sentiment \n')
print_bad_examples2(df_negative)


# # Bad examples for neutral sentiments
# Here I just returned the selected text as it is. Seems to me here that most of the errors are punctuations.  
# A way to easily get higher score is to objectively truncate the punctuations such as to maximise jaccard scores.  
# Seems like some of the examples are mislabbeled sentiments. The last 5 rows are to me at least mislabelled.

# In[ ]:


print('Jaccard Worst Examples neutral sentiment \n')
print_bad_examples(df_neutral)


# In[ ]:


print('Jaccard2 Worst Examples neutral sentiment \n')
print_bad_examples2(df_neutral)


# # Lets plot the distribution of my model

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
axes[0].hist(df.jaccard.values)
axes[0].set_title('Histogram of all jaccard scores')
axes[1].hist(df.jaccard2.values)
axes[1].set_title('Histogram of all jaccard2 scores')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
axes[0].hist(df_neutral.jaccard.values)
axes[0].set_title('Histogram of all neutral jaccard scores')
axes[1].hist(df.jaccard2.values)
axes[1].set_title('Histogram of all neutral jaccard2 scores')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
axes[0].hist(df_positive.jaccard.values)
axes[0].set_title('Histogram of all positive jaccard scores')
axes[1].hist(df.jaccard2.values)
axes[1].set_title('Histogram of all positive jaccard2 scores')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
axes[0].hist(df_negative.jaccard.values)
axes[0].set_title('Histogram of all negative jaccard scores')
axes[1].hist(df.jaccard2.values)
axes[1].set_title('Histogram of all negative jaccard2 scores')


# # Conclusion
# ## For positive/negative sentiments
# it seems like the model might miss out the context and only pick the word with the sentiment.  
# Also when there are more than 1 sentiment in a sentence sometimes the model picks the less intense word/span.  
# ## For neutral sentiments
# Punctuations truncations or extention. Pick the optimal truncation/extension strategy such that it maximises your jaccard scores.  
# Some mislabelled examples in training set when it is suppose to be positive or negative spotting this might result in improve of scores!
# 
# ## Last words
# Solving the above issues will take considerable effort and some needs to be hardcoded :(. Hardcoding in any solution in my opinion is bad as it is not generaliable to other problems. The best solution would be to solve the above problems without hardcoding any rules.
