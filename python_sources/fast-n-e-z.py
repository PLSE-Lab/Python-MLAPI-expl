#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import csv
import json
import string
import keras
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from math import floor
import spacy

get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

import time
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import regex as re


# In[ ]:


nlp = spacy.load('en_core_web_sm')


# In[ ]:


def word_locate(sentence, location): 
    count_words = 0
    count_chars = 2 #2 is to count for the two spaces in the beginning
    for word in sentence.split():
        count_words += 1
        if location == count_chars:
            return word, count_words
        count_chars += len(word)
        count_chars += 1 #for space


# In[ ]:


def split_sent_by_comma(sentence):
    #splitting sentence if structure is (subj+obj+verb, subj+obj+verb), that is subj immediately following comma
    doc = nlp(sentence)
    prev_token_dep = ""
    prev_token_text = ""
    list_of_sub_sent = []
    curr_sent = ""
    for token in doc:
        if prev_token_dep == "punct" and token.dep_ == "nsubj":
            list_of_sub_sent.append(curr_sent)
            curr_sent = ""
            prev_token_text = ""
        curr_sent += prev_token_text
        if token.dep_ != "punct":
            curr_sent += " " #there is space between words, but not before comma
        prev_token_dep = token.dep_
        prev_token_text = token.text
    list_of_sub_sent.append(curr_sent)
    return list_of_sub_sent


# In[ ]:


def find_name_between_paran(sentence):
    capture = ""
    trigger_on = 0
    for char in sentence:
        if char == ")":
            trigger_on = 0
        if trigger_on == 1:
            capture += char
        if char == "(":
            trigger_on = 1
    return capture


# In[ ]:


def find_name_between_commas(sentence):
    list_of_names = []
    for sent in sentence.split(","):
        if len(sent.split()) == 1:
            if sent.split()[0][0] in ["A","B","C","D","E","F","G","H","I","J","K","L","M","O","P","Q","R","S","T","U","V","W","X","Y","Z"]:
                list_of_names.append(sent.split()[0])
    if len(list_of_names) != 1:
        list_of_names.append("none")
    return list_of_names[0]


# In[ ]:


analyze = "Despite this, it failed to chart in any other country. Bonnie Tyler (then known as Gaynor Hopkins) spent seven years performing in local pubs and clubs around South Wales between 1969 and 1976, first as part of Bobbie Wayne & the Dixies, and then with her own band Imagination."


# In[ ]:


print(find_name_between_paran(analyze))


# In[ ]:


def which_name_first(sentence, name1, name2):
    name1_check = 0
    for word_punct in sentence.split():
        for word_comma in word_punct.split(";"):
            for word in word_comma.split(","):
                if word == name2 and name1_check == 0:
                    return name2
                if word == name1:
                    name1_check = 1
    return name1


# In[ ]:


def curr_prev_sentence(sentence, loc):
    current_sentence = ""
    prev_sentence = ""
    trunc_curr_sentence = ""
    remainder_curr = ""
    detect = 0
    count = 0
    for char in sentence:
        count += 1
        current_sentence += char
        remainder_curr += char
        if ((char == "." or char == ";") and detect == 0 and sentence[count] != ","): #the last arguement to prevent ., as in sent #4
            prev_sentence = current_sentence 
            current_sentence = ""
        if char == "." and detect == 1:
            return current_sentence, prev_sentence, trunc_curr_sentence, remainder_curr
        if count == loc:
            detect = 1
            trunc_curr_sentence = current_sentence
            remainder_curr = ""
    return current_sentence, prev_sentence, trunc_curr_sentence, remainder_curr


# In[ ]:


def remove_last_word(sentence):
    new_sent = sentence.split()
    new_sent = new_sent[:-1]
    return " ".join(new_sent)


# In[ ]:


def check_if_name(word):
    if word[0] in ["A","B","C","D","E","F","G","H","I","J","K","L","M","O","P","Q","R","S","T","U","V","W","X","Y","Z"]:
        return True
    else:
        return False


# In[ ]:


def remove_first_word(sentence):
    new_sent = sentence.split()
    new_sent = new_sent[1:]
    return " ".join(new_sent)


# In[ ]:


def find_name_words(sentence):
    name = "none"
    for word in sentence.split():
        if check_if_name(word):
            return word
    return name


# In[ ]:


def find_subject(sentence): #finds last subject
    doc = nlp(sentence)
    subject = "none"
    prev_tok_dep = ""
    for token in doc:
        if (token.dep_ == "nsubj" or token.dep_ == "nsubjpass") and prev_tok_dep != "xcomp"         and token.text != "she" and token.text != "he" and token.text != "She" and token.text != "He" and token.text != "who":
            subject = token.text
        prev_tok_dep = token.dep_
    return subject


# In[ ]:


def find_first_subject(sentence): #finds first subject
    doc = nlp(sentence)
    prev_tok_dep = ""
    for token in doc:
        if (token.dep_ == "nsubj" or token.dep_ == "nsubjpass") and prev_tok_dep != "xcomp"             and token.text != "she" and token.text != "he" and token.text != "She" and token.text != "He" and token.text != "who":
            return token.text
        prev_tok_dep = token.dep_
    return "none"


# In[ ]:


def find_second_subject(sentence): #finds second subject
    doc = nlp(sentence)
    subject = "none"
    prev_tok_dep = ""
    count = 0
    for token in doc:
        if (token.dep_ == "nsubj" or token.dep_ == "nsubjpass") and prev_tok_dep != "xcomp"             and token.text != "she" and token.text != "he" and token.text != "She" and token.text != "He" and token.text != "who":
            count += 1
            if count == 2:
                subject = token.text
        prev_tok_dep = token.dep_
    return subject


# In[ ]:


def find_dobj(sentence): #find first dobj
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "dobj" and (token.text != "him" and token.text != "her"):
            return token.text
    return "none"


# In[ ]:


def find_last_dobj(sentence): #finds last dobj
    doc = nlp(sentence)
    dobj = "none"
    for token in doc:
        if token.dep_ == "dobj" and (token.text != "him" and token.text != "her"):
            dobj = token.text
    return dobj


# In[ ]:


def find_last_possessing_noun(sentence): #find last poss
    doc = nlp(sentence)
    poss = "none"
    for token in doc:
        if token.dep_ == "poss" and (token.head.pos_ == "PROPN" or token.head.pos_ == "NOUN"):
            poss = token.text
    return poss


# In[ ]:


def find_possessing_noun(sentence): #find first poss
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "poss" and (token.head.pos_ == "PROPN" or token.head.pos_ == "NOUN"):
            return token.text
    return "none"


# In[ ]:


def find_appos(sentence): #returns first appos ; sometimes Spacy mislabels nsubj as appos
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "appos" and token.head.pos_ == "PROPN":
            return token.text


# In[ ]:


with open('../input/gap-training-data/gap-development.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    test_ids= []
    text_list = []
    pronoun_list = []
    pronoun_offset_list = []
    correct_name_list = []
    previous_sents_list = []
    current_sents_list = []
    sent_num = 0
    word_btwn_comma_list = []
    word_btwn_paran_list = []
    trun_dobj_list = []
    trun_subj_list = []
    prev_dobj_list = []
    prev_subj_list = []
    prev_last_subj_list = []
    prev_first_subj_list = []
    prev_second_subj_list = []
    prev_last_dobj_list = []
    curr1st_subj_list = []
    curr2nd_subj_list = []
    curr1st_dobj_list = []
    curr2nd_dobj_list = []
    curr1st_appos_list = []
    
    for row in reader:
        test_ids.append(row['ID'])
        text = row['Text']
        proffset = int(row['Pronoun-offset']) 
        sent_num += 1
        current, prev, trunc_curr, remainder = curr_prev_sentence(text, proffset)
        previous_sents_list.append(prev)
        current_sents_list.append(current)
        trunc_curr = remove_last_word(trunc_curr)
        prev_subj = find_subject(prev)
        prev_first_subj = find_first_subject(prev)
        prev_first_subj_list.append(prev_first_subj)
        prev_second_subj = find_second_subject(prev)
        prev_second_subj_list.append(prev_second_subj)
        trunc_curr_subj = find_subject(trunc_curr)
        trunc_curr_dobj = find_dobj(trunc_curr)
        prev_dobj = find_dobj(prev)
        prev_name_btn_comma = find_name_words(find_name_between_commas(prev))
        curr_name_btn_paran = find_name_words(find_name_between_paran(current))
        curr_1st_subj = find_first_subject(current)
        curr_2nd_subj = find_second_subject(current)
        curr_1st_dobj = find_dobj(current)
        if not check_if_name(curr_1st_dobj):
            curr_1st_dobj = find_possessing_noun(current)
        curr_2nd_dobj = find_last_dobj(current)
        if not check_if_name(curr_2nd_dobj):
            curr_2nd_dobj = find_last_possessing_noun(current)
        curr_1st_appos = find_appos(current)
            
        if prev_name_btn_comma != "none" and prev_name_btn_comma != which_name_first(prev,prev_name_btn_comma,prev_subj) :
            word_btwn_comma_list.append(prev_name_btn_comma)
        else:
            word_btwn_comma_list.append("none")
        
        trun_dobj_list.append(trunc_curr_dobj)
        trun_subj_list.append(trunc_curr_subj)
        prev_dobj_list.append(prev_dobj)
        prev_subj_list.append(prev_subj)
        
        
        if row['A-coref'] == 'TRUE':
            correct_name_list.append(row['A'])
        elif row['B-coref'] == 'TRUE':
            correct_name_list.append(row['B'])
        else:
            correct_name_list.append('Neither')

        prev_last = split_sent_by_comma(row['Text'])[-1]
        
        prev_last_subj = find_subject(prev_last)
        prev_last_dobj = find_dobj(prev_last)
        
        prev_last_subj_list.append(prev_last_subj)
        prev_last_dobj_list.append(prev_last_dobj)
        curr1st_subj_list.append(curr_1st_subj)
        curr2nd_subj_list.append(curr_2nd_subj)
        curr1st_dobj_list.append(curr_1st_dobj)
        curr2nd_dobj_list.append(curr_2nd_dobj)
        curr1st_appos_list.append(curr_1st_appos)
        word_btwn_paran_list.append(curr_name_btn_paran)
        
        text_list.append(" %%%%%%%%%%%%%%%%%%%%%% ")
        text_list.append(sent_num)
        text_list.append(" ********************** ")
        text_list.append(text)
        pronoun_list.append(row['Pronoun'])
        pronoun_offset_list.append(proffset)


# In[ ]:


results_df = pd.DataFrame({"correct":correct_name_list})
results_df['prv_obj'] = prev_dobj_list
results_df['pr_ls_sbj'] = prev_subj_list
results_df['pr_1_sbj'] = prev_first_subj_list
results_df['pr_2_sbj'] = prev_second_subj_list
results_df['c1st_sj'] = curr1st_subj_list
results_df['c2nd_sj'] = curr2nd_subj_list
results_df['c1st_ob'] = curr1st_dobj_list
results_df['c2nd_ob'] = curr2nd_dobj_list
results_df['c1st_ap'] = curr1st_appos_list
results_df['w_bt_pa'] = word_btwn_paran_list
results_df['pronoun'] = pronoun_list
results_df['offset'] = pronoun_offset_list


# In[ ]:


results_df.head(250)


# In[ ]:


print(text_list[229])


# In[ ]:


analyze = "Despite this, it failed to chart in any other country. Bonnie Tyler (then known as Gaynor Hopkins) spent seven years performing in local pubs and clubs around South Wales between 1969 and 1976, first as part of Bobbie Wayne & the Dixies, and then with her own band Imagination."


# In[ ]:


current, prev, trunc_curr, remainder = curr_prev_sentence(analyze, 246)


# In[ ]:


print(find_first_subject(current))


# In[ ]:


print(find_last_dobj(analyze))


# In[ ]:


doc = nlp(analyze)
for token in doc:
    print(token.text, token.dep_, token.head.pos_)


# In[ ]:


print(text_list[900:950])


# In[ ]:


out_df = pd.DataFrame({"ID":test_ids})


# In[ ]:


out_df['A'] = results_A
out_df['B'] = results_B
out_df['NEITHER'] = results_N


# In[ ]:


out_df.to_csv("submission.csv", index=False)

