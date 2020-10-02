#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


gap_development = pd.read_csv('../input/gapdataset/gap-development.tsv', delimiter='\t')
gap_test = pd.read_csv('../input/gapdataset/gap-test.tsv', delimiter='\t')
gap_validation = pd.read_csv('../input/gapdataset/gap-validation.tsv', delimiter='\t')


# In[ ]:


gap_development.head()


# In[ ]:


gap_test.head()


# In[ ]:


gap_validation.head()


# In[ ]:


train = pd.concat((gap_test, gap_validation)).reset_index(drop=True)
test = gap_development

train['A-coref'] = train['A-coref'].astype(int)
train['B-coref'] = train['B-coref'].astype(int)
train['NEITHER'] = 1.0 - (train['A-coref'] + train['B-coref'])

test['A-coref'] = test['A-coref'].astype(int)
test['B-coref'] = test['B-coref'].astype(int)
test['NEITHER'] = 1.0 - (test['A-coref'] + test['B-coref'])

train.head()


# Use Bert embeding to acquire correct similarity between A and pronoun, B and pronoun.

# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py ')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/extract_features.py ')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py')


# In[ ]:


get_ipython().system('wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
get_ipython().system(' 7z x uncased_L-12_H-768_A-12.zip')


# In[ ]:


import modeling
import extract_features
import tokenization


# In[ ]:


def compute_offset_no_spaces(text, offset):
    count = 0
    for pos in range(offset):
        if text[pos] != " ": count +=1
    return count

def count_chars_no_special(text):
    count = 0
    special_char_list = ["#"]
    for pos in range(len(text)):
        if text[pos] not in special_char_list: count +=1
    return count

def count_length_no_special(text):
    count = 0
    special_char_list = ["#", " "]
    for pos in range(len(text)):
        if text[pos] not in special_char_list: count +=1
    return count


# In[ ]:


def bert_embeddings(df):
    text = df["Text"]
    text.to_csv("input.txt", index = False, header = False)
    
    # run BERT model
    os.system("python3 extract_features.py       --input_file=input.txt       --output_file=output.jsonl       --vocab_file=uncased_L-12_H-768_A-12/vocab.txt       --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json       --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt       --layers=-1       --max_seq_length=256       --batch_size=8")
    
    bert_output = pd.read_json("output.jsonl", lines = True)

    os.system("rm output.jsonl")
    os.system("rm input.txt")

    index = df.index
    columns = ["emb_A", "emb_B", "emb_P", "label"]
    emb = pd.DataFrame(index = index, columns = columns)
    emb.index.name = "ID"
    
    for i in range(len(df)): # For each line in the data file
        # get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
        P = df.loc[i,"Pronoun"].lower()
        A = df.loc[i,"A"].lower()
        B = df.loc[i,"B"].lower()

        # For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
        P_offset = compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"Pronoun-offset"])
        A_offset = compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"A-offset"])
        B_offset = compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"B-offset"])
        # Figure out the length of A, B, not counting spaces or special characters
        A_length = count_length_no_special(A)
        B_length = count_length_no_special(B)

        # Initialize embeddings with zeros
        emb_A = np.zeros(768)
        emb_B = np.zeros(768)
        emb_P = np.zeros(768)

        # Initialize counts
        count_chars = 0
        cnt_A, cnt_B, cnt_P = 0, 0, 0

        features = pd.DataFrame(bert_output.loc[i,"features"])
        
        for j in range(2,len(features)):  # Iterate over the BERT tokens for the current line; we skip over the first 2 tokens, which don't correspond to words
            token = features.loc[j,"token"]

            # See if the character count until the current token matches the offset of any of the 3 target words
            if count_chars  == P_offset: 
                # print(token)
                emb_P += np.array(features.loc[j,"layers"][0]['values'])
                cnt_P += 1
            if count_chars in range(A_offset, A_offset + A_length): 
                # print(token)
                emb_A += np.array(features.loc[j,"layers"][0]['values'])
                cnt_A +=1
            if count_chars in range(B_offset, B_offset + B_length): 
                # print(token)
                emb_B += np.array(features.loc[j,"layers"][0]['values'])
                cnt_B +=1
            # Update the character count
            count_chars += count_length_no_special(token)
        # Taking the average between tokens in the span of A or B, so divide the current value by the count	
        emb_A /= cnt_A
        emb_B /= cnt_B
        
        label = "Neither"
        if (df.loc[i,"A-coref"] == 1):
            label = "A"
        if (df.loc[i,"B-coref"] == 1):
            label = "B"

        # Put everything together in emb
        emb.iloc[i] = [emb_A, emb_B, emb_P, label]
        
    return emb


# In[ ]:


train_emb = bert_embeddings(train)


# In[ ]:


test_emb =  bert_embeddings(test)


# In[ ]:


from scipy import spatial 
def add_similarity_columns(df, df_emb):
    df['sim_A_P'] = 0
    df['sim_B_P'] = 0

    for i in range(0, len(df)):
        sim_A_P = 1 - spatial.distance.cosine(df_emb.loc[i, 'emb_A'], df_emb.loc[i, 'emb_P'])
        if not np.isnan(sim_A_P):
            df.loc[i, 'sim_A_P'] = sim_A_P
        
        sim_B_P = 1 - spatial.distance.cosine(df_emb.loc[i, 'emb_B'], df_emb.loc[i, 'emb_P'])
        if not np.isnan(sim_B_P):
            df.loc[i, 'sim_B_P'] = sim_B_P


# In[ ]:


add_similarity_columns(train, train_emb)
add_similarity_columns(test, test_emb)


# Extend feature matrix

# In[ ]:


def add_additional_features(df):
    df['Pronoun-offset2'] = df['Pronoun-offset'] + df['Pronoun'].map(len)
    df['A-offset2'] = df['A-offset'] + df['A'].map(len)
    df['B-offset2'] = df['B-offset'] + df['B'].map(len)
    df['A-dist'] = (df['Pronoun-offset'] - df['A-offset']).abs()
    df['B-dist'] = (df['Pronoun-offset'] - df['B-offset']).abs()
    df['section_min'] = df[['Pronoun-offset', 'A-offset', 'B-offset']].min(axis=1)
    df['section_max'] = df[['Pronoun-offset2', 'A-offset2', 'B-offset2']].max(axis=1)
    
add_additional_features(train)
add_additional_features(test)
train.head()


# In[ ]:


def name_replace(s, r1, r2):
    s = str(s).replace(r1,r2)
    for r3 in r1.split(' '):
        s = str(s).replace(r3,r2)
    return s

train['Text'] = train.apply(lambda r: name_replace(r['Text'], r['A'], 'subjectone'), axis=1)
train['Text'] = train.apply(lambda r: name_replace(r['Text'], r['B'], 'subjecttwo'), axis=1)

test['Text'] = test.apply(lambda r: name_replace(r['Text'], r['A'], 'subjectone'), axis=1)
test['Text'] = test.apply(lambda r: name_replace(r['Text'], r['B'], 'subjecttwo'), axis=1)


# In[ ]:


import spacy 
nlp = spacy.load('en_core_web_sm')


# In[ ]:


tags = {}
for text in train['Text']:
    doc = nlp(str(text))
    for token in doc:
        if token.text == 'subjectone' or token.text == 'subjecttwo':
            if token.dep_ in tags:
                tags[token.dep_] += 1
            else:
                tags[token.dep_] = 1
                


# Find top 5 the most often occurency tags

# In[ ]:


sorted_tags = sorted(tags.items(), key=lambda kv: kv[1])
sorted_tags[-5:]


# In[ ]:


def fill_nlp_tag_empty_cols(df, tag):
    df[f"A-{tag}"] = None
    df[f"B-{tag}"] = None
    
def fill_nlp_empty_cols(df, tags):
    for tag in tags:
        fill_nlp_tag_empty_cols(df, tag)

def fill_word_offset_empty_cols(df):
    df['Pronoun-word-offset'] = None
    df['A-word-offset'] = None
    df['B-word-offset'] = None
    df['A-word-dist'] = None
    df['B-word-dist'] = None
    
def fill_similarity(df):
    df['sim_A_P'] = 0.0
    df['sim_B_P'] = 0.0

    
def get_nlp_tag_feature(doc, tag):
    tokens = pd.DataFrame([[token.text, token.dep_] for token in doc], columns=['text', 'dep'])
    A_tag = len(tokens[((tokens['text']=='subjectone') & (tokens['dep']==tag))])
    B_tag = len(tokens[((tokens['text']=='subjecttwo') & (tokens['dep']==tag))])
    
    return A_tag, B_tag

def word_offset(doc, w):
    count = 0
    for token in doc:
        if token.text == w:
            break
        if not token.is_punct and token.text != '`':
            count += 1
    return count


def add_nlp_features(df, tags):
    size = len(df)
    fill_nlp_empty_cols(df, tags)
    fill_word_offset_empty_cols(df)
    
    for i in range(0, size):
        text = df.loc[i, 'Text']
        doc = nlp(str(text))
        
        #add tag features
        for tag in tags:
            df.loc[i, f"A-{tag}"], df.loc[i, f"B-{tag}"] = get_nlp_tag_feature(doc, tag)
            
        
        #add word offset features
        df.loc[i, 'Pronoun-word-offset'] = word_offset(doc, df.loc[i, 'Pronoun'])
        df.loc[i, 'A-word-offset'] = word_offset(doc, 'subjectone')
        df.loc[i, 'B-word-offset'] = word_offset(doc, 'subjecttwo')
        
        df.loc[i, 'A-word-dist'] = np.abs(df.loc[i, 'Pronoun-word-offset'] - df.loc[i, 'A-word-offset'])
        df.loc[i, 'B-word-dist'] = np.abs(df.loc[i, 'Pronoun-word-offset'] - df.loc[i, 'B-word-offset'])
        
        #add similarity 
#         df.loc[i, 'sim_A_P'], df.loc[i, 'sim_B_P'] = get_similarity(doc, df.loc[i, 'Pronoun'])


# In[ ]:


add_nlp_features(train, ['poss', 'nsubj', 'pobj', 'dobj', 'conj'])
add_nlp_features(test, ['poss', 'nsubj', 'pobj', 'dobj', 'conj'])
train


# In[ ]:


feature_col = [
               'Pronoun-offset', 
               'Pronoun-offset2', 
               'A-offset', 
               'A-offset2', 
               'A-dist', 
               'B-offset', 
               'B-offset2',
               'B-dist', 
               'section_min',  
               'section_max',
               'A-poss', 
               'B-poss', 
               'A-nsubj',
               'B-nsubj',
               'A-pobj',
               'B-pobj',
               'A-dobj',
               'B-dobj',
               'A-conj',
               'B-conj',
               'sim_A_P',
               'sim_B_P',
               'A-word-offset',
               'B-word-offset',
               'Pronoun-word-offset',
               'A-word-dist',
               'B-word-dist'
              ]
pred_col = ['A-coref', 'B-coref', 'NEITHER']


# In[ ]:


X_train = train[feature_col].values
Y_train = train[pred_col].values

X_test = test[feature_col].values
Y_test = test[pred_col].values


# In[ ]:


# x_train, x_test, y_train, y_test = model_selection.train_test_split(train[feature_col].fillna(-1), train[pred_col], test_size=0.2, random_state=1)


# In[ ]:


model = multiclass.OneVsRestClassifier(ensemble.RandomForestClassifier(max_depth = 7, n_estimators=2000, random_state=33))
model.fit(X_train, Y_train)


# In[ ]:


model.predict_proba(X_test)


# In[ ]:


print('log_loss: ', metrics.log_loss(Y_test, model.predict_proba(X_test)))


# In[ ]:


# gap_development['A-coref'] = train['A-coref'].astype(int)
# gap_development['B-coref'] = train['B-coref'].astype(int)
# gap_development['NEITHER'] = 1.0 - (train['A-coref'] + train['B-coref'])
# gap_development.head()


# In[ ]:


# add_additional_features(gap_development)
# gap_development_x = gap_development[feature_col]
# gap_development_y = gap_development[pred_col]
# gap_development_pred = model.predict_proba(gap_development_x)
# print('log_loss: ', metrics.log_loss(gap_development_y, gap_development_pred))


# In[ ]:


# gap_development_y.head()


# In[ ]:


# gap_development_pred


# Submission test

# In[ ]:


test_sub1 = pd.read_csv('../input/gendered-pronoun-resolution/test_stage_1.tsv', delimiter='\t')
test_sub1_bert = bert_embeddings(test_sub1)
add_similarity_columns(test_sub1, test_sub1_bert)

#add necessary  features
add_additional_features(test_sub1)

test_sub1['Text'] = test_sub1.apply(lambda r: name_replace(r['Text'], r['A'], 'subjectone'), axis=1)
test_sub1['Text'] = test_sub1.apply(lambda r: name_replace(r['Text'], r['B'], 'subjecttwo'), axis=1)

add_nlp_features(test_sub2, ['poss', 'nsubj', 'pobj', 'dobj', 'conj'])

results = model.predict_proba(test_sub1[feature_col])
test_sub1.rename(columns={'A': 'A_Noun', 'B': 'B_Noun'})
test_sub1['A'] = results[:,0].astype(np.float)
test_sub1['B'] = results[:,1].astype(np.float)
test_sub1['NEITHER'] = results[:,2].astype(np.float)
test_sub1[['ID', 'A', 'B', 'NEITHER']].to_csv('submission1.csv', index=False)
test_sub1.head()


# In[ ]:


# test_sub2 = pd.read_csv('../input/gendered-pronoun-resolution/test_stage_2.tsv', delimiter='\t')
# test_sub2_bert = bert_embeddings(test_sub2)
# add_similarity_columns(test_sub2, test_sub2_bert)

# #add necessary  features
# add_additional_features(test_sub2)

# test_sub2['Text'] = test_sub2.apply(lambda r: name_replace(r['Text'], r['A'], 'subjectone'), axis=1)
# test_sub2['Text'] = test_sub2.apply(lambda r: name_replace(r['Text'], r['B'], 'subjecttwo'), axis=1)

# add_nlp_features(test_sub2, ['poss', 'nsubj', 'pobj', 'dobj', 'conj'])

# results = model.predict_proba(test_sub2[feature_col])
# test_sub2.rename(columns={'A': 'A_Noun', 'B': 'B_Noun'})
# test_sub2['A'] = results[:,0].astype(np.float)
# test_sub2['B'] = results[:,1].astype(np.float)
# test_sub2['NEITHER'] = results[:,2].astype(np.float)
# test_sub2[['ID', 'A', 'B', 'NEITHER']].to_csv('submission2.csv', index=False)
# test_sub2.head()


# In[ ]:


test_sub2

