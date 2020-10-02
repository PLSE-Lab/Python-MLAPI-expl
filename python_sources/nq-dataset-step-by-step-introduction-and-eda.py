#!/usr/bin/env python
# coding: utf-8

# *Understanding well our data is half the work.* 

# <span style="font-size:larger;">Contents</span>:
# * Data preparation
# * Data fields
# * Questions w/o answers
# * Good and bad questions
# * Possible answer combinations
# * Data length

# Note: 
# 
# <br>Although I'll be using for convenience the data provided by the [TensorFlow 2.0 Question Answering Kaggle competition](https://www.kaggle.com/c/tensorflow2-question-answering), this work was produced as an independent introduction for QA research. As I'm personally introducing myself to QA, I want to share with you my findings and approach when studying the NQ Dataset, in hopes to bring some useful insights.
# 
# <br>Sources can be found linked thoroughout the text, but you can find them all also here below:
# <br>
# 
# Papers:
# * [Natural Questions: a Benchmark for Question Answering Research](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf)
# * [A BERT Baseline for the Natural Questions](https://arxiv.org/pdf/1901.08634.pdf)
# 
# NQ sites:
# * [Natural Questions main page](https://ai.google.com/research/NaturalQuestions)
# * [Natural Questions github page](https://github.com/google-research-datasets/natural-questions)
# * [Natural Questions DataBrowser](https://ai.google.com/research/NaturalQuestions/databrowser)

# # Data preparation

# In[ ]:


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import gc
import ujson
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# As advanced above, we will use for our purposes the simplified NQ train and test datasets in newline-delimited JSON format provided by the TensorFlow 2.0 QA Kaggle competition.

# In[ ]:


path = '/kaggle/input/tensorflow2-question-answering/'
train_path = path + 'simplified-nq-train.jsonl'
test_path = path + 'simplified-nq-test.jsonl'


# First, we can make sure of the number of examples contained in each set by calling the following function to each. 
# 
# Simplified NQ provides the extracted text of the Wikipedia pages, excluding the HTML. The size is still huge, so running the following three cells may be unnecessarily memory consuming.

# In[ ]:


def count_examples(path):
    with open(path) as file:
        count = 0
        for line in file:
            count += 1
        print(f"number of examples: {count}")


# In[ ]:


train_data = count_examples(train_path)
train_data


# As expected, we get 307,373 training examples.

# In[ ]:


test_data = count_examples(test_path)
test_data


# In[ ]:


del train_data, test_data
del count_examples
gc.collect()


# As for the test set, Kaggle provides with 346 examples for testing. This number differs from the original NQ test set (7,842 examples).
# 
# 
# Now, let's convert the jsonl files into dataframes for data exploration. Because Panda's df are too memory expensive, we will be exploring over little less than a third of the train dataset. If you have problems running the following function over 100,000, 50,000 should work perfectly fine and shouldn't represent an important difference in results.
# 
# Examples come already in random order, so we don't have to worry about shuffling before getting our sample set.

# In[ ]:


train_size = 100000
test_size = 346

def json_to_df(path, size):
    df = []
    with open(path) as f:
        for line in f:
            data = ujson.loads(line)
            if path in train_path:
                del data['document_url']
            if len(df) > size-1:
                break
            df.append(data)
            
    df = pd.DataFrame(df)
    gc.collect()
    return df


# In[ ]:


train_df = json_to_df(train_path, train_size)
train_df.head()


# In[ ]:


train_df.shape


# We have removed "document_url" from the data fields as we won't be going over it (contains the url to the Wikipedia page in question).

# In[ ]:


test_df = json_to_df(test_path, test_size)
test_df.head()


# In[ ]:


test_df.shape


# # Data fields

# Let's now look at these data fiels in more detail by extracting each one for the first example in our train set.

# In[ ]:


question_text_0 = train_df.loc[0]['question_text']
question_text_0


# Within the NQ question identification task, the [contributors determined whether a given question is good or bad](https://research.google/pubs/pub47761/). That is, if a question seeks a fact or, on the other hand, is ambiguous, incomprehensible or opinion-seeking. We will soon explore more about this.

# In[ ]:


document_text_0 = train_df.loc[0]['document_text'].split()
' '.join(document_text_0)


# Each example comes with its annotations, in which we can identify the following:
# 
# - Short answers: is there a short answer for this question? Where? And could it be answered with "yes" or "no"?
# - Long answers: is there a suitable long answer for this question? Where?

# In[ ]:


annotations_0 = train_df['annotations'][0][0]
annotations_0


# Quick note about "short_answers" before we move forward: in 90% of cases short answers are contained in one single span of text. However, for questions like *who made it to stage 3 in american ninja warrior season 9*, a list of answers is needed. Definitely read carefully [Annotations and Data Statistics in NQ github page](https://github.com/google-research-datasets/natural-questions#Annotations) as we continue with the analysis!
# 
# <br>Another final detail about annotations, "candidate_index" selects the index for the best choice in the list of candidates of the "long_answer_candidate" field. Let's move onto that one.

# In[ ]:


long_answer_candidate_0 = train_df.loc[0]['long_answer_candidates']
long_answer_candidate_0


# Apart from the annotations, we also get a data field for [long answer candidates](https://github.com/google-research-datasets/natural-questions#long-answer-candidates). These are basically the "information boxes" from a Wikipedia page that may or not answer the given question. The annotator decided if there is an "information box" that could serve as long answer for the question. These boxes (candidates, ranges...) can be paragraphs, lists, list items, tables or table rows ([worth considering html tags for model implementation!](https://arxiv.org/abs/1901.08634)).
# 
# The shortest is preferred and all the information must be contained in one single of those ranges. If both are possible, the short and long answers to a given question come from the same paragraph.
# 
# As for the boolean flag "top_level", it's included to note if a candidate is contained (nested) inside a larger range. Therefore, top_level=True means that a candidate has another shorter one nested below, while top_level=False signifies there's no nested candidate below that one.

# In[ ]:


yes_no_answer = []
for i in range(len(train_df)):
    yes_no_answer.append(train_df['annotations'][i][0]['yes_no_answer'])
yes_no_answer = pd.DataFrame({'yes_no_answer': yes_no_answer})

yes_no_answer['yes_no_answer'].value_counts(normalize=True) * 100


# As we have already seen, there can be a yes/no answer for a question, and at the same time a normal short answer. Here we observe that in the vast majority of cases, there's no yes/no answer possible.

# # Questions w/o answers

# Taking a quick look at the Natural Questions Train Set Examples [DataBrowser](https://ai.google.com/research/NaturalQuestions/databrowser) provided by Google AI, we can observe that there are actually many questions without answers, either long, short, or even both. 
# 
# Before taking a look at the nature of those questions, and if that may be related to this lack of answers, let's see the approximate percentage of short and long answers not provided.

# In[ ]:


# Get no short answers, no long answers and both.
def get_no_answers(short_or_long, size=train_size):
    num_no_answers = 0
    if short_or_long == 'short_answers':
        for i in range(size):
            if train_df['annotations'][i][0]['short_answers'] == []:
                num_no_answers += 1
        print(f'{(num_no_answers / size) * 100:.2f}% of total questions have no short answer.')
        
    elif short_or_long == 'long_answer':
        for i in range(size):
            if train_df['annotations'][i][0]['long_answer']['start_token'] == -1:
                num_no_answers += 1
        print(f'{(num_no_answers / size) * 100:.2f}% of total questions have no long answer.')  
        
    else:
        example_ids = []
        for i in range(size):
            if train_df['annotations'][i][0]['short_answers'] == [] and train_df['annotations'][i][0]['long_answer']['start_token'] == -1:
                example_ids.append(train_df['example_id'][i])
                num_no_answers += 1
        print(f'{(num_no_answers / size) * 100:.2f}% of total questions have no answer.', '\n')
        print('-' * 40, '\n')
        
        for ex in example_ids[:100]:
            print(train_df.loc[train_df['example_id'] == ex, ['question_text'][0]])


# In[ ]:


get_no_answers('short_answers')


# First we check short answers by recovering the number of empty lists in the annotation's field. That's more than half of the questions without a short answer! Let's see for long answers.
# 
# But, how are they represented as "empty"?

# In[ ]:


# We take a small sample of 20 examples
for i in range(20):
    print(train_df['annotations'][i][0]['long_answer'])


# Now we check for our whole train set sample how many long answers in their annotations have the -1 value for their tokens.

# In[ ]:


get_no_answers('long_answer')


# Using an example id we have retrieved separately, let's confirm everything is fine.

# In[ ]:


train_df[train_df['example_id'] == 3411244446249504947]


# In[ ]:


train_df['annotations'][5][0]['short_answers']


# In[ ]:


train_df['annotations'][5][0]['long_answer']


# For this particular example, both short and long answer fields are empty. Let's check more about these.

# # Good and bad questions

# Some questions don't have any answers as annotators considered them "bad" questions. 
# <br>Below we see how many of these "bad" questions there are and, at the same time, we take a sample to read some of them.

# In[ ]:


# For this case, the function returns percentage of total questions with no answers at all
# and a sample of 100 of those questions.
get_no_answers('long_and_short_answers')


# From this very small sample, it seems that:
# 
# - Some questions are ambiguous (13, 14)
# 
# - Some are not even questions, but sound like imperative requests! (160, 168)
# 
# - Some queries cannot be answered by Wikipedia (131, 136, 138)
# 
# - Some questions could be answered by Wikipedia (4, 36)
# 
# And for sure many more other details.

# [In their paper](https://research.google/pubs/pub47761/), Kwiatkowski et al. highlight the introduction of questions starting with "who", "when", or "where". Just to finish with our small exploration on question_text, let's take a quick look at the approximate percentage of questions starting with wh-words.

# In[ ]:


wh_words = {'who': 0, 'what': 0, 'when': 0, 'where': 0, 'how': 0, 'which': 0, 'why': 0, 'whose': 0}
etc = []
wh_total = 0

question_series = train_df['question_text']
for i in range(train_size):
    first_word = question_series[i].split()
    if first_word[0] not in wh_words:
        etc.append(first_word[0])
    elif first_word[0] == 'who':
        wh_words['who'] += 1
        wh_total += 1
    elif first_word[0] == 'what':
        wh_words['what'] += 1
        wh_total += 1
    elif first_word[0] == 'when':
        wh_words['when'] += 1
        wh_total += 1
    elif first_word[0] == 'where':
        wh_words['where'] += 1
        wh_total += 1
    elif first_word[0] == 'how':   
        wh_words['how'] += 1
        wh_total += 1
    elif first_word[0] == 'which':
        wh_words['which'] += 1
        wh_total += 1
    elif first_word[0] == 'why':
        wh_words['why'] += 1
        wh_total += 1
    else:    
        wh_words['whose'] += 1
        wh_total += 1


# In[ ]:


wh_words


# The percentages are easy to guess.

# In[ ]:


wh_words_list = ['who', 'what', 'when', 'where', 'how', 'which', 'why', 'whose']
for q in wh_words_list:
    percent = (wh_words[q] / train_size) * 100
    print(f'{q}: {percent:.2f}%')


# In[ ]:


print(f'Wh-question words represent approx. {(wh_total / train_size) * 100:.2f}% of the total start words in our questions.')


# However, we would need a deeper analysis in order to get meaningful results:
# 
# - Among our questions we find two types of entries: queries and proper questions. 
# <br>We would need to explore both in more detail.
# 
# - Wh-question words often come up with *'s* either like in *what's* or *whats*.
# <br>We would need to tokenize the text accordingly for this exploration. Would be expected to find a larger percentage of wh-words.

# # Possible answer combinations

# Coming back to the answers, another important observation is what combinations of answers are possible for our questions. In other words, is it possible to have a long answer, but no short answer, and vice versa? 

# In[ ]:


# Get possible answers per example
def get_possible_answers(short_or_long, size=train_size):
    num_answers = 0
    if short_or_long == 'short_answers':
        for i in range(size):
            if train_df['annotations'][i][0]['short_answers'] != []:
                num_answers += 1
        print(f'{(num_answers / size) * 100:.2f}% of total questions have a short answer.')
        
    elif short_or_long == 'long_answer':
        for i in range(size):
            if train_df['annotations'][i][0]['long_answer']['start_token'] != -1:
                num_answers += 1
        print(f'{(num_answers / size) * 100:.2f}% of total questions have long answer.')  
    
    elif short_or_long == 'short_no_long':
        for i in range(size):
            if train_df['annotations'][i][0]['short_answers'] != [] and train_df['annotations'][i][0]['long_answer']['start_token'] == -1:
                num_answers += 1
        print(f'{(num_answers / size) * 100:.2f}% of total questions have a short but no long answer.')
    
    elif short_or_long == 'long_no_short':
        for i in range(size):
            if train_df['annotations'][i][0]['short_answers'] == [] and train_df['annotations'][i][0]['long_answer']['start_token'] != -1:
                num_answers += 1
        print(f'{(num_answers / size) * 100:.2f}% of total questions have a long but no short answer.')
   
    else:
        for i in range(size):
            if train_df['annotations'][i][0]['short_answers'] != [] and train_df['annotations'][i][0]['long_answer']['start_token'] != -1:
                num_answers += 1
        print(f'{(num_answers / size) * 100:.2f}% of total questions have a short and long answer.')


# If we do simple math, the coming results are pretty straightforward:

# In[ ]:


get_possible_answers('short_answers')


# In[ ]:


get_possible_answers('long_answer')


# In[ ]:


get_possible_answers('short_and_long')


# Now, something it was probably already intuitive at this point:

# In[ ]:


get_possible_answers('short_no_long')


# In[ ]:


get_possible_answers('long_no_short')


# It is possible to have a long answer alone, but not only a short answer.

# In addition, if we further explore the combination long + yes/no answer, for our 100,000 train sample, we would get that 1.21% of total questions have both a long and a yes/no answer.

# To sum up, in the table below are all the covered combinations with the obtained percentage of occurence for our 100,000 examples.
# <br>
# <br>
# 
# | SHORT | LONG | YES/NO | NONE| |
# |---|---|---|---|---|
# |all|-|-|-|34.77%|
# |-|all|-|-|49.44%|
# |-|-|all|-|1.26%|
# |-|-|-|all|50.56%|
# |yes|yes|-|-|34.77%|
# |yes|no|-|-|0.00%|
# |no|yes|-|-|14.67%|
# |-|yes|yes|-|1.21%|

# # Data length

# Finally, with the preparation of our model in mind, let's see the maximum length for question_text, document_text, short_answers and long_answer.

# In[ ]:


def max_length(datatype, size=train_size):
    if datatype == 'question_text':
        max_question = train_df['question_text'][0]
        for i in range(size):
            if len(max_question) < len(train_df['question_text'][i]):
                max_question = train_df['question_text'][i]
        
        print(f'Longest question: {max_question}', '\n')
        print(f'Total characters: {len(max_question)}', '\n')
        print(f'Total words: {len(max_question.split())}')
    
    elif datatype == 'document_text':
        max_document = train_df['document_text'][0]
        for i in range(size):
            if len(max_document) < len(train_df['document_text'][i]):
                max_document = train_df['document_text'][i]
        
        print(f'Total characters: {len(max_document)}', '\n')
        print(f'Total words: {len(max_document.split())}')
    
    elif datatype == 'short_answers':
        max_result = 0
        for i in range(size):
            if train_df['annotations'][i][0]['short_answers'] == []:
                continue
            max_values = list(train_df['annotations'][i][0]['short_answers'][0].values())
            if (max_values[1] - max_values[0]) > max_result:
                max_result = max_values[1] - max_values[0]
        print(f'Total characters: {max_result}')
    
    elif datatype == 'long_answer':
        max_result = 0
        for i in range(size):
            if train_df['annotations'][i][0]['long_answer']['start_token'] == -1:
                continue
            max_values = list(train_df['annotations'][i][0]['long_answer'].values())
            if (max_values[2] - max_values[0]) > max_result:
                max_result = max_values[2] - max_values[0]
        print(f'Total characters: {max_result}') 


# In[ ]:


max_length('question_text')


# In[ ]:


max_length('document_text')


# In[ ]:


max_length('short_answers')


# In[ ]:


max_length('long_answer')


# As important as it is doing our own EDA, there are many great resources for this dataset that can considerably facilitate our work.
# 
# <br> Do not hesitate to share below your thoughts and ideas.
# 
# <br>Thank you for reading!
