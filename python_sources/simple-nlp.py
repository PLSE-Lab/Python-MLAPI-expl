#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#First version to have successful rand_forest is V42
#v64 first to group out rand forest based on pronoun type
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
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

import time
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import regex as re


# In[ ]:


import nltk 
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize 


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


def name_btwn_paran(sentence):
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


def which_name_first(sentence, name1, name2): #If name1 is first, return True
    name1_check = 0
    for word_punct in sentence.split():
        for word_comma in word_punct.split(";"):
            for word in word_comma.split(","):
                if word == name2 and name1_check == 0:
                    return False
                if word == name1:
                    name1_check = 1
    return True


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


def check_if_capital(word):
    if word[0] in ["A","B","C","D","E","F","G","H","I","J","K","L","M","O","P","Q","R","S","T","U","V","W","X","Y","Z"]:
        return True
    else:
        return False


# In[ ]:


def list_of_name_words(tokenized):
    names_list = []
    for word_tuple in nltk.pos_tag(tokenized):
        if word_tuple[1] == "NNP":
            names_list.append(word_tuple[0])
    return names_list


# In[ ]:


def check_if_name(tokenized,word):
    text = tokenized
    for word_tuple in nltk.pos_tag(text):
        if word_tuple[0] == word:
            if word_tuple[1] == "NNP":
                return True
            else:
                return False


# In[ ]:


def find_name_words(sentence):
    name = "none"
    for word in sentence.split():
        if check_if_capital(word):
            return word
    return name


# In[ ]:


def remove_first_word(sentence):
    new_sent = sentence.split()
    new_sent = new_sent[1:]
    return " ".join(new_sent)


# In[ ]:


def find_nth_subj(doc, n): #finds subject number n
    subject = "none"
    count = 0
    for token in doc:
        if (token.dep_ == "nsubj" or token.dep_ == "nsubjpass"):
            count += 1
            if count == n:
                subject = token.text
    return subject


# In[ ]:


def find_nth_dobj(doc, n): #finds direct object number n
    dobj = "none"
    count = 0
    for token in doc:
        if (token.dep_ == "dobj"):
            count += 1
            if count == n:
                dobj = token.text
    return dobj


# In[ ]:


def find_nth_poss(doc, n): #finds possessing noun number n
    poss = "none"
    count = 0
    for token in doc:
        if (token.dep_ == "poss"):
            count += 1
            if count == n:
                poss = token.text
    return poss


# In[ ]:


def find_nth_appos(doc, n): #finds appos number n; sometimes Spacy mislabels nsubj as appos
    appos = "none"
    count = 0
    for token in doc:
        if (token.dep_ == "appos"):
            count += 1
            if count == n:
                appos = token.text
    return appos


# In[ ]:


def check_if_poss_her(doc, pronoun): #tells whether it is her as in his or her as in him
    #assumes only one her in the whole sentence (inaccurate?)
    for token in doc:
        if token.text == pronoun:
            if token.dep_ == "poss":
                return True
            else:
                return False


# In[ ]:


with open('../input/gap-other-training-data/gap-training.tsv') as tsvfile:
#with open('../input/gap-training-data/gap-development.tsv') as tsvfile:
    
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    
    train_ids= []
    text_list = []
    pronoun_list = []
    pronoun_offset_list = []
    correct_name_list = []
    dict_of_all_list = []
    
    sent_num = 0
    
    p_f_s = [] #prev first subject
    p_l_s = [] #prev last subject
    p_f_o = [] #prev first object
    p_l_o = [] #prev last object
    tc_f_s = [] #trunc curr first subject
    tc_f_o = [] #trunc curr first obj
    tc_f_a = [] #trunc curr first aposs
    tc_l_s = [] #trunc curr last subject
    tc_l_o = [] #trunc curr last obj
    tc_l_p = [] #trunc curr last poss
    p_f_wp = [] #prev first word between paranthesis
    tc_l_wp = [] #curr word between paranthesis
    tc_l_nw = [] #last name word other than a subj in trunc curr
    r_f_s = [] #remainder first subj
    r_f_o = [] #remainder first object
    r_f_a = [] #remainder first appos
    p_f_a = [] #prev first appos
    c_f_a = [] #curr first appos
    poss_her = [] #possessive her true or false
    pro_typ = [] #pronoun type
    
    p_f_s_clf = [] #prev first subject Random Forest labels
    p_l_s_clf = [] #prev last subject Random Forest labels
    p_f_o_clf = [] #prev first object Random Forest labels
    p_l_o_clf = [] #prev last object Random Forest labels
    tc_f_s_clf = [] #trunc curr first subject Random Forest labels
    tc_f_o_clf = [] #trunc curr first obj Random Forest labels
    tc_f_a_clf = [] #trunc curr first aposs Random Forest labels
    tc_l_s_clf = [] #trunc curr last subject Random Forest labels
    tc_l_o_clf = [] #trunc curr last obj Random Forest labels
    tc_l_p_clf = [] #trunc curr last poss Random Forest labels
    p_f_wp_clf = [] #prev first wb para Random Forest labels
    tc_l_wp_clf = [] #curr wb para Random Forest labels
    tc_l_nw_clf = [] #last name word in trunc curr Random Forest labels
    r_f_s_clf = [] #remainder first subj Random Forest labels
    r_f_o_clf = [] #remainder first object Random Forest labels
    r_f_a_clf = [] #remainder first appos Random Forest labels
    p_f_a_clf = [] #prev first appos Random Forest labels
    c_f_a_clf = [] #curr first appos Random Forest labels
    pro_typ_clf = [] #pronoun type for Rand Forest
    train_idx = [] #training data indices. We do not want those with "neither" 
    
    for row in reader:
        
        train_ids.append(row['ID'])
        text = row['Text']
        text_list.append(text)
        dict_of_all = {}
        
        proffset = int(row['Pronoun-offset']) 
        pronoun_offset_list.append(proffset)
        
        pronoun = row['Pronoun']
        pronoun_list.append(pronoun)
        
        if row['A-coref'] == 'TRUE':
            correct_name_list.append(row['A'])
            train_idx.append(sent_num)
        elif row['B-coref'] == 'TRUE':
            correct_name_list.append(row['B'])
            train_idx.append(sent_num)
        else:
            correct_name_list.append('Neither')
        
        sent_num += 1
        
        curr, prev, trunc_curr, remainder = curr_prev_sentence(text, proffset)
        curr_doc = nlp(curr)
        prev_doc = nlp(prev) 
        curr_tok = word_tokenize(curr)
        prev_tok = word_tokenize(prev)
        trunc_curr_tok = word_tokenize(trunc_curr)
        
        #get first subj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_s = "none"
        for n in [1,2,3,4,5]: #number of n is from common sense
            dummy_p_f_s = find_nth_subj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_s) and get_p_f_s == "none":
                get_p_f_s = dummy_p_f_s
        
        ####For sentence no. 5, spacy and nltk both failed to identify Collins as a propn.
        ### therefore, we will add a new line here making sure we have a name.
        
        if get_p_f_s == "none":
            if check_if_capital(find_nth_subj(prev_doc,1)):
                get_p_f_s = find_nth_subj(prev_doc,1)

        p_f_s.append(get_p_f_s)
        
        ### pfs Random forest classifier label special line:
        if get_p_f_s in correct_name_list[-1]                or correct_name_list[-1] in get_p_f_s: #last input of correct name
            p_f_s_clf.append(1)
        else:
            p_f_s_clf.append(0)
        
        #get last  subj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_l_s = "none"
        for n in [1,2,3,4,5]:
            dummy_p_l_s = find_nth_subj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_l_s):
                get_p_l_s = dummy_p_l_s
        
        p_l_s.append(get_p_l_s)
        
        ### pls Random forest classifier label special line:
        if get_p_l_s in correct_name_list[-1]                or correct_name_list[-1] in get_p_l_s: #last input of correct name
            p_l_s_clf.append(1)
        else:
            p_l_s_clf.append(0)
        
        #get first  obj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_o = "none"
        for n in [1,2,3,4,5]: 
            dummy_p_f_o = find_nth_dobj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_o) and get_p_f_o == "none":
                get_p_f_o = dummy_p_f_o
        
        p_f_o.append(get_p_f_o)
        
        ### pfo Random forest classifier label special line:
        if get_p_f_o in correct_name_list[-1]                or correct_name_list[-1] in get_p_f_o: #last input of correct name
            p_f_o_clf.append(1)
        else:
            p_f_o_clf.append(0)
            
        #get last  dobj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_l_o = "none"
        for n in [1,2,3,4,5]: 
            dummy_p_l_o = find_nth_dobj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_l_o):
                get_p_l_o = dummy_p_l_o
        
        p_l_o.append(get_p_l_o)
        
        ### plo Random forest classifier label special line:
        if get_p_l_o in correct_name_list[-1]                or correct_name_list[-1] in get_p_l_o: #last input of correct name
            p_l_o_clf.append(1)
        else:
            p_l_o_clf.append(0)
        
        #get last  subj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_s = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_s = find_nth_subj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_l_s)                    and (dummy_tc_l_s in trunc_curr): #this is slightly inaccurate but oh well
                get_tc_l_s = dummy_tc_l_s 
            
        tc_l_s.append(get_tc_l_s)
        
        ### tcls Random forest classifier label special line:
        if get_tc_l_s in correct_name_list[-1]                or correct_name_list[-1] in get_tc_l_s: #last input of correct name
            tc_l_s_clf.append(1)
        else:
            tc_l_s_clf.append(0)
            
        #get last  dobj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_o = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_o = find_nth_dobj(curr_doc,n)
            if (dummy_tc_l_o in trunc_curr)                                        and check_if_name(curr_tok,dummy_tc_l_o): 
                get_tc_l_o = dummy_tc_l_o 
            
        tc_l_o.append(get_tc_l_o)
        
        ### tclo Random forest classifier label special line:
        if get_tc_l_o in correct_name_list[-1]                or correct_name_list[-1] in get_tc_l_o: #last input of correct name
            tc_l_o_clf.append(1)
        else:
            tc_l_o_clf.append(0)
        
        #get last  poss in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_p = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_p = find_nth_poss(curr_doc,n)
            if (dummy_tc_l_p in trunc_curr)                                        and check_if_name(curr_tok,dummy_tc_l_p): 
                get_tc_l_p = dummy_tc_l_p 
            
        tc_l_p.append(get_tc_l_p)
        
        ### tclp Random forest classifier label special line:
        if get_tc_l_p in correct_name_list[-1]                or correct_name_list[-1] in get_tc_l_p: #last input of correct name
            tc_l_p_clf.append(1)
        else:
            tc_l_p_clf.append(0)
        
        #get first subj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_s = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_s = find_nth_subj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_s) and get_tc_f_s == "none":
                get_tc_f_s = dummy_tc_f_s 
            
        tc_f_s.append(get_tc_f_s)
        
        ### tcfs Random forest classifier label special line:
        if get_tc_f_s in correct_name_list[-1]                or correct_name_list[-1] in get_tc_f_s: #last input of correct name
            tc_f_s_clf.append(1)
        else:
            tc_f_s_clf.append(0)
            
        #get first dobj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_o = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_o = find_nth_dobj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_o) and get_tc_f_o == "none": 
                get_tc_f_o = dummy_tc_f_o 
            
        tc_f_o.append(get_tc_f_o)
        
        ### tcfo Random forest classifier label special line:
        if get_tc_f_o in correct_name_list[-1]                or correct_name_list[-1] in get_tc_f_o: #last input of correct name
            tc_f_o_clf.append(1)
        else:
            tc_f_o_clf.append(0)
    
        #get last  non-subj name word  in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_nw = "none"
        candidate = "none"
        tc_name_words = list_of_name_words(trunc_curr_tok) 
        if len(tc_name_words) > 0:
            candidate = tc_name_words[-1]
        if candidate in get_tc_f_s or candidate in get_tc_l_s:
            if len(tc_name_words) > 1:
                candidate = tc_name_words[-1]
        if check_if_name(curr_tok,candidate):
            get_tc_l_nw = candidate
        
        tc_l_nw.append(get_tc_l_nw)
        
        ### tclnw Random forest classifier label special line:
        if get_tc_l_nw in correct_name_list[-1]                or correct_name_list[-1] in get_tc_l_nw: #last input of correct name
            tc_l_nw_clf.append(1)
        else:
            tc_l_nw_clf.append(0)
        
        #get first aposs in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_a = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_a = find_nth_appos(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_a) and get_tc_f_a == "none": 
                get_tc_f_a = dummy_tc_f_a 
            
        tc_f_a.append(get_tc_f_a)
        
        ### tcfa Random forest classifier label special line:
        if get_tc_f_a in correct_name_list[-1]                or correct_name_list[-1] in get_tc_f_a: #last input of correct name
            tc_f_a_clf.append(1)
        else:
            tc_f_a_clf.append(0)
    
        #get word btwn paranthesis in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_wp = find_name_words(name_btwn_paran(prev))
        
        if check_if_name(prev_tok,get_p_f_wp): #Add only proper nouns into list
            p_f_wp.append(get_p_f_wp)
        else:
            p_f_wp.append("none")
        
        ### pfwp Random forest classifier label special line:
        if get_p_f_wp in correct_name_list[-1]                or correct_name_list[-1] in get_p_f_wp: #last input of correct name
            p_f_wp_clf.append(1)
        else:
            p_f_wp_clf.append(0)
            
        #get word btwn paranthesis in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_wp = find_name_words(name_btwn_paran(curr))  
        
        if check_if_name(curr_tok,get_tc_l_wp): #Add only proper nouns into list
            tc_l_wp.append(get_tc_l_wp)
        else:
            tc_l_wp.append("none")
            
        ### tclwp Random forest classifier label special line:
        if get_tc_l_wp in correct_name_list[-1]                or correct_name_list[-1] in get_tc_l_wp: #last input of correct name
            tc_l_wp_clf.append(1)
        else:
            tc_l_wp_clf.append(0)
            
        #get last subj in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_s = "none"
        for n in [1,2,3,4,5,6,7,8]: #in the final version, each of the name subjects will be accunted for
            dummy_r_f_s = find_nth_subj(curr_doc,n)
            if dummy_r_f_s in remainder and check_if_name(curr_tok,dummy_r_f_s):
                get_r_f_s = dummy_r_f_s 
            
        r_f_s.append(get_r_f_s)
        
        ### rfs Random forest classifier label special line:
        if get_r_f_s in correct_name_list[-1]                or correct_name_list[-1] in get_r_f_s: #last input of correct name
            r_f_s_clf.append(1)
        else:
            r_f_s_clf.append(0)
            
        #get last dobj in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_o = "none"
        for n in [1,2,3,4,5,6,7,8]: #in the final version, each of the name objects will be accunted for
            dummy_r_f_o = find_nth_dobj(curr_doc,n)
            if dummy_r_f_o in remainder and check_if_name(curr_tok,dummy_r_f_o):
                get_r_f_o = dummy_r_f_o 
            
        r_f_o.append(get_r_f_o)
        
        ### rfo Random forest classifier label special line:
        if get_r_f_o in correct_name_list[-1]                or correct_name_list[-1] in get_r_f_o: #last input of correct name
            r_f_o_clf.append(1)
        else:
            r_f_o_clf.append(0)
            
        #get last appos in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_a = "none"
        for n in [1,2,3,4]:
            dummy_r_f_a = find_nth_appos(curr_doc,n)
            if dummy_r_f_a in remainder and check_if_name(curr_tok,dummy_r_f_a): 
                get_r_f_a = dummy_r_f_a 
            
        r_f_a.append(get_r_f_a)
        
        ### rfa Random forest classifier label special line:
        if get_r_f_a in correct_name_list[-1]                or correct_name_list[-1] in get_r_f_a: #last input of correct name
            r_f_a_clf.append(1)
        else:
            r_f_a_clf.append(0)
        
        #get first appos in current @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_c_f_a = "none"
        for n in [1,2,3,4]:
            dummy_c_f_a = find_nth_appos(curr_doc,n)
            if check_if_name(curr_tok,dummy_c_f_a) and get_c_f_a == "none": 
                get_c_f_a = dummy_c_f_a 
            
        c_f_a.append(get_c_f_a)
        
        ### cfa Random forest classifier label special line:
        if get_c_f_a in correct_name_list[-1]                or correct_name_list[-1] in get_c_f_a: #last input of correct name
            c_f_a_clf.append(1)
        else:
            c_f_a_clf.append(0)
        
        #get first appos in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_a = "none"
        for n in [1,2,3,4]:
            dummy_p_f_a = find_nth_appos(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_a) and get_p_f_a == "none": 
                get_p_f_a = dummy_p_f_a 
            
        p_f_a.append(get_p_f_a)
        
        ### pfa Random forest classifier label special line:
        if get_p_f_a in correct_name_list[-1]                or correct_name_list[-1] in get_p_f_a: #last input of correct name
            p_f_a_clf.append(1)
        else:
            p_f_a_clf.append(0)
    
        #check_if_poss_her
        get_poss_her = check_if_poss_her(curr_doc, pronoun)
        poss_her.append(get_poss_her)
    
        #rand_forest classifier for pronoun type:
        if pronoun == "he" or pronoun == "she": 
            pro_typ.append(1)
        elif pronoun == "He" or pronoun == "She": 
            pro_typ.append(2)
        elif pronoun == "his" or (pronoun == "her" and get_poss_her): 
            pro_typ.append(3)
        elif pronoun == "him" or (pronoun == "her" and not get_poss_her): 
            pro_typ.append(4)
        elif pronoun == "His" or (pronoun == "Her" and get_poss_her): 
            pro_typ.append(5)
        else:
            pro_typ.append(6)
        
        dict_of_all["p_f_s"] = get_p_f_s
        dict_of_all["p_l_s"] = get_p_l_s
        dict_of_all["p_f_o"] = get_p_f_o
        dict_of_all["p_l_o"] = get_p_l_o
        dict_of_all["tc_f_s"] = get_tc_f_s
        dict_of_all["tc_f_o"] = get_tc_f_o
        dict_of_all["tc_f_a"] = get_tc_f_a
        dict_of_all["tc_l_s"] = get_tc_l_s
        dict_of_all["tc_l_o"] = get_tc_l_o
        dict_of_all["tc_l_p"] = get_tc_l_p
        dict_of_all["p_f_wp"] = get_p_f_wp
        dict_of_all["tc_l_wp"] = get_tc_l_wp
        dict_of_all["tc_l_nw"] = get_tc_l_nw
        dict_of_all["r_f_s"] = get_r_f_s
        dict_of_all["r_f_o"] = get_r_f_o
        dict_of_all["r_f_a"] = get_r_f_a
        dict_of_all["p_f_a"] = get_p_f_a
        dict_of_all["c_f_a"] = get_c_f_a
        dict_of_all["poss_her"] = poss_her
        
        dict_of_all_list.append(dict_of_all)


# In[ ]:


results_df = pd.DataFrame({"correct":correct_name_list})

results_df['pr_fsub'] = p_f_s
results_df['pr_lsub'] = p_l_s
results_df['pr_fobj'] = p_f_o
results_df['pr_lobj'] = p_l_o
results_df['tc_lsub'] = tc_l_s
results_df['tc_lobj'] = tc_l_o
results_df['tc_fsub'] = tc_f_s
results_df['tc_fobj'] = tc_f_o
results_df['tc_fapo'] = tc_f_a
results_df['tc_lnw'] = tc_l_nw
results_df['tc_lp'] = tc_l_p
results_df['re_sub'] = r_f_s
results_df['re_obj'] = r_f_o
results_df['re_app'] = r_f_a
results_df['pr_para'] = p_f_wp
results_df['tc_para'] = tc_l_wp
results_df['pr_app'] = p_f_a
results_df['cr_app'] = c_f_a
results_df['pronoun'] = pronoun_list
results_df['offset'] = pronoun_offset_list


# In[ ]:


count = 0
for idx in range(20):
    print(text_list[idx])
    print("*********************")
    print(count)
    print("@@@")
    count += 1


# In[ ]:


#THE NEXT FEW CELLS ARE DEDICATED TO A PLAYGROUND FOR MANUAL LOGIC. RANDFOREST IS AFTER THAT. 


# In[ ]:


#### THIS PART IS INDEPENDENT FROM THE RANDOM FOREST SECTION.
def coref_logic(dict_of_all, pronoun):
    guess = "none"
    case_group = "none"
    
    #CASE-A: Pronoun he or she or He or She
    if pronoun == "he" or pronoun == "she" or pronoun == "He" or pronoun == "She": 
        
        #CASE-1
        if dict_of_all["tc_f_s"] == "none" and dict_of_all["p_f_s"] != "none":
            guess = dict_of_all["p_f_s"] #pls will be = pfs if no other subj
            case_group = "A1"
        #CASE-2
        elif dict_of_all["tc_f_s"] != "none":
            guess = dict_of_all["tc_f_s"] #tcls will be = tcfs if no other subj
            case_group = "A2"
        #CASE-3
        elif dict_of_all["tc_f_s"] == "none" and dict_of_all["p_f_s"] == "none":
            if dict_of_all["tc_f_o"] == "none" and dict_of_all["p_f_o"] != "none":
                guess = dict_of_all["p_f_o"]
                case_group = "A3"
        #CASE-4
        elif dict_of_all["tc_f_s"] == "none" and dict_of_all["p_f_s"] == "none":
            if dict_of_all["tc_f_o"] != "none" and dict_of_all["p_f_o"] == "none":
                guess = dict_of_all["tc_f_o"]
                case_group = "A4"
    #CASE B: Pronoun his or her (possessive her):
    if pronoun == "his" or (pronoun == "her" and dict_of_all["poss_her"]): 
        
        #CASE-1 #assuming the pronoun is also in the paranthesis
        if dict_of_all["tc_l_p"] != "none": 
            guess = dict_of_all["tc_l_p"]
            case_group = "B1"
        #CASE-2 #assuming the pronoun is also in the paranthesis
        elif dict_of_all["tc_l_wp"] != "none": 
            guess = dict_of_all["tc_l_wp"]
            case_group = "B2"
        #CASE-3
        elif dict_of_all["r_f_a"] != "none": 
            guess = dict_of_all["r_f_a"]
            case_group = "B3"
        #CASE-4
        elif dict_of_all["r_f_s"] != "none": 
            guess = dict_of_all["r_f_s"]
            case_group = "B4"
        #CASE-5
        elif dict_of_all["tc_f_s"] == "none" and dict_of_all["p_f_s"] == "none"                    and dict_of_all["r_f_s"] == "none" and dict_of_all["tc_f_o"] != "none": 
            guess = dict_of_all["tc_f_o"]
            case_group = "B5"
        #CASE-6
        elif dict_of_all["tc_f_s"] != "none" and dict_of_all["p_f_s"] != "none"                    and dict_of_all["r_f_s"] == "none" and dict_of_all["tc_f_o"] == "none": 
            guess = dict_of_all["tc_f_s"]
            case_group = "B6"    
        #CASE-7
        elif dict_of_all["tc_f_s"] == "none" and dict_of_all["p_f_s"] != "none"                    and dict_of_all["r_f_s"] == "none" and dict_of_all["tc_f_o"] == "none": 
            guess = dict_of_all["p_f_s"]
            case_group = "B7"
            
    #CASE C: Pronoun his or her (object her):
    if pronoun == "him" or (pronoun == "her" and not dict_of_all["poss_her"]): 
        
        #CASE-1
        if dict_of_all["p_f_s"] == "none" and dict_of_all["tc_f_o"] == "none"                        and dict_of_all["tc_f_s"] != "none": 
            guess = dict_of_all["tc_l_nw"]
            case_group = "C1"
        
        # to be continued..    
        
    return guess, case_group


# In[ ]:


guesses = []
case_groups = []
for dict_of_all, pronoun in zip(dict_of_all_list, pronoun_list):
    guesses.append(coref_logic(dict_of_all, pronoun)[0])
    case_groups.append(coref_logic(dict_of_all, pronoun)[1])


# In[ ]:


guesses_df = pd.DataFrame({"correct":correct_name_list})
guesses_df['guesses'] = guesses
guesses_df['case_group'] = case_groups


# In[ ]:


guesses_df.head(50)


# #RAND FOREST TRAINING DATA
# tr_p_f_s = [p_f_s_clf[idx] for idx in train_idx] 
# tr_p_l_s = [p_l_s_clf[idx] for idx in train_idx]
# tr_p_f_o = [p_f_o_clf[idx] for idx in train_idx]
# tr_p_l_o = [p_l_o_clf[idx] for idx in train_idx]
# tr_tc_f_s = [tc_f_s_clf[idx] for idx in train_idx]
# tr_tc_f_o = [tc_f_o_clf[idx] for idx in train_idx]
# tr_tc_f_a = [tc_f_a_clf[idx] for idx in train_idx]
# tr_tc_l_s = [tc_l_s_clf[idx] for idx in train_idx]
# tr_tc_l_o = [tc_l_o_clf[idx] for idx in train_idx]
# tr_tc_l_p = [tc_l_p_clf[idx] for idx in train_idx]
# tr_p_f_wp = [p_f_wp_clf[idx] for idx in train_idx]
# tr_tc_l_wp = [tc_l_wp_clf[idx] for idx in train_idx]
# tr_tc_l_nw = [tc_l_nw_clf[idx] for idx in train_idx]
# tr_r_f_s = [r_f_s_clf[idx] for idx in train_idx]
# tr_r_f_o = [r_f_o_clf[idx] for idx in train_idx]
# tr_r_f_a = [r_f_a_clf[idx] for idx in train_idx]
# tr_p_f_a = [p_f_a_clf[idx] for idx in train_idx]
# tr_c_f_a = [c_f_a_clf[idx] for idx in train_idx]

# In[ ]:


data_matrix = []
data_matrix1 = []
data_matrix2 = []
data_matrix3 = []
data_matrix4 = []
data_matrix5 = []
data_matrix6 = []

tr_pro_t1_idxs = []
tr_pro_t2_idxs = []
tr_pro_t3_idxs = []
tr_pro_t4_idxs = []
tr_pro_t5_idxs = []
tr_pro_t6_idxs = []

for idx in range(len(p_f_s)):
    
    data_vector = []
    
    if p_f_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)    
    if p_l_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)
    if p_f_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)
    if p_l_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)
    if tc_f_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)
    if tc_f_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_f_a[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_p[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if p_f_wp[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_wp[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_nw[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if r_f_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if r_f_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if r_f_a[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if p_f_a[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if c_f_a[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if poss_her[idx] == True:
        data_vector.append(1)
    else:
        data_vector.append(0)
    #pronoun type, already numerical
    if pro_typ[idx] == 1:
        tr_pro_t1_idxs.append(idx)
        data_matrix1.append(data_vector)
    if pro_typ[idx] == 2:
        tr_pro_t2_idxs.append(idx)
        data_matrix2.append(data_vector)
    if pro_typ[idx] == 3:
        tr_pro_t3_idxs.append(idx)
        data_matrix3.append(data_vector)
    if pro_typ[idx] == 4:
        tr_pro_t4_idxs.append(idx)
        data_matrix4.append(data_vector)
    if pro_typ[idx] == 5:
        tr_pro_t5_idxs.append(idx)
        data_matrix5.append(data_vector)
    if pro_typ[idx] == 6:
        tr_pro_t6_idxs.append(idx)
        data_matrix6.append(data_vector)
        
    data_matrix.append(data_vector)
    
clf1_df = pd.DataFrame(data_matrix1, columns = dict_of_all_list[0].keys())
clf2_df = pd.DataFrame(data_matrix2, columns = dict_of_all_list[0].keys())
clf3_df = pd.DataFrame(data_matrix3, columns = dict_of_all_list[0].keys())
clf4_df = pd.DataFrame(data_matrix4, columns = dict_of_all_list[0].keys())
clf5_df = pd.DataFrame(data_matrix5, columns = dict_of_all_list[0].keys())
clf6_df = pd.DataFrame(data_matrix6, columns = dict_of_all_list[0].keys())


# In[ ]:


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators=150, n_jobs=2, random_state=0)


# In[ ]:


#Now make features for the test dataset. This requires doing alllllll those all over again.

# IT IS JUST COPY-PASTE FROM THE TRAINING PROCEDURE (EXCEPT THE LABELS, WHICH WE DON'T NEED.)

with open('../input/gendered-pronoun-resolution/test_stage_1.tsv') as tsvfile:
    
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    
    test_ids= []
    text_list = []
    pronoun_list = []
    pronoun_offset_list = []
    dict_of_all_list = []
    
    sent_num = 1
    
    p_f_s = [] #prev first subject
    p_l_s = [] #prev last subject
    p_f_o = [] #prev first object
    p_l_o = [] #prev last object
    tc_f_s = [] #trunc curr first subject
    tc_f_o = [] #trunc curr first obj
    tc_f_a = [] #trunc curr first aposs
    tc_l_s = [] #trunc curr last subject
    tc_l_o = [] #trunc curr last obj
    tc_l_p = [] #trunc curr last poss
    p_f_wp = [] #prev first word between paranthesis
    tc_l_wp = [] #curr word between paranthesis
    tc_l_nw = [] #last name word other than a subj in trunc curr
    r_f_s = [] #remainder first subj
    r_f_o = [] #remainder first object
    r_f_a = [] #remainder first appos
    p_f_a = [] #prev first appos
    c_f_a = [] #curr first appos
    poss_her = [] #possessive her true or false
    pro_typ = [] #pronoun type
      
    for row in reader:
        
        train_ids.append(row['ID'])
        text = row['Text']
        sent_num += 1
        text_list.append(text)
        dict_of_all = {}
        
        proffset = int(row['Pronoun-offset']) 
        pronoun_offset_list.append(proffset)
        
        pronoun = row['Pronoun']
        pronoun_list.append(pronoun)
              
        curr, prev, trunc_curr, remainder = curr_prev_sentence(text, proffset)
        curr_doc = nlp(curr)
        prev_doc = nlp(prev) 
        curr_tok = word_tokenize(curr)
        prev_tok = word_tokenize(prev)
        trunc_curr_tok = word_tokenize(trunc_curr)
        
        #get first subj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_s = "none"
        for n in [1,2,3,4,5]: #number of n is from common sense
            dummy_p_f_s = find_nth_subj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_s) and get_p_f_s == "none":
                get_p_f_s = dummy_p_f_s
        
        ####For sentence no. 5, spacy and nltk both failed to identify Collins as a propn.
        ### therefore, we will add a new line here making sure we have a name.
        
        if get_p_f_s == "none":
            if check_if_capital(find_nth_subj(prev_doc,1)):
                get_p_f_s = find_nth_subj(prev_doc,1)

        p_f_s.append(get_p_f_s)
        
        #get last  subj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_l_s = "none"
        for n in [1,2,3,4,5]:
            dummy_p_l_s = find_nth_subj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_l_s):
                get_p_l_s = dummy_p_l_s
        
        p_l_s.append(get_p_l_s)
                
        #get first  obj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_o = "none"
        for n in [1,2,3,4,5]: 
            dummy_p_f_o = find_nth_dobj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_o) and get_p_f_o == "none":
                get_p_f_o = dummy_p_f_o
        
        p_f_o.append(get_p_f_o)
                    
        #get last  dobj in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_l_o = "none"
        for n in [1,2,3,4,5]: 
            dummy_p_l_o = find_nth_dobj(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_l_o):
                get_p_l_o = dummy_p_l_o
        
        p_l_o.append(get_p_l_o)
                
        #get last  subj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_s = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_s = find_nth_subj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_l_s)                    and (dummy_tc_l_s in trunc_curr): #this is slightly inaccurate but oh well
                get_tc_l_s = dummy_tc_l_s 
            
        tc_l_s.append(get_tc_l_s)
                    
        #get last  dobj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_o = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_o = find_nth_dobj(curr_doc,n)
            if (dummy_tc_l_o in trunc_curr)                                        and check_if_name(curr_tok,dummy_tc_l_o): 
                get_tc_l_o = dummy_tc_l_o 
            
        tc_l_o.append(get_tc_l_o)
                
        #get last  poss in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_p = "none"
        for n in [1,2,3,4]:
            dummy_tc_l_p = find_nth_poss(curr_doc,n)
            if (dummy_tc_l_p in trunc_curr)                                        and check_if_name(curr_tok,dummy_tc_l_p): 
                get_tc_l_p = dummy_tc_l_p 
            
        tc_l_p.append(get_tc_l_p)
                
        #get first subj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_s = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_s = find_nth_subj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_s) and get_tc_f_s == "none":
                get_tc_f_s = dummy_tc_f_s 
            
        tc_f_s.append(get_tc_f_s)
                    
        #get first dobj in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_o = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_o = find_nth_dobj(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_o) and get_tc_f_o == "none": 
                get_tc_f_o = dummy_tc_f_o 
            
        tc_f_o.append(get_tc_f_o)
            
        #get last  non-subj name word  in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_nw = "none"
        candidate = "none"
        tc_name_words = list_of_name_words(trunc_curr_tok) 
        if len(tc_name_words) > 0:
            candidate = tc_name_words[-1]
        if candidate in get_tc_f_s or candidate in get_tc_l_s:
            if len(tc_name_words) > 1:
                candidate = tc_name_words[-1]
        if check_if_name(curr_tok,candidate):
            get_tc_l_nw = candidate
        
        tc_l_nw.append(get_tc_l_nw)
                
        #get first aposs in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_f_a = "none"
        for n in [1,2,3,4]:
            dummy_tc_f_a = find_nth_appos(curr_doc,n)
            if check_if_name(curr_tok,dummy_tc_f_a) and get_tc_f_a == "none": 
                get_tc_f_a = dummy_tc_f_a 
            
        tc_f_a.append(get_tc_f_a)
            
        #get word btwn paranthesis in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_wp = find_name_words(name_btwn_paran(prev))
        
        if check_if_name(prev_tok,get_p_f_wp): #Add only proper nouns into list
            p_f_wp.append(get_p_f_wp)
        else:
            p_f_wp.append("none")
                    
        #get word btwn paranthesis in trunc curr @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_tc_l_wp = find_name_words(name_btwn_paran(curr))  
        
        if check_if_name(curr_tok,get_tc_l_wp): #Add only proper nouns into list
            tc_l_wp.append(get_tc_l_wp)
        else:
            tc_l_wp.append("none")
                        
        #get last subj in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_s = "none"
        for n in [1,2,3,4,5,6,7,8]: #in the final version, each of the name subjects will be accunted for
            dummy_r_f_s = find_nth_subj(curr_doc,n)
            if dummy_r_f_s in remainder and check_if_name(curr_tok,dummy_r_f_s):
                get_r_f_s = dummy_r_f_s 
            
        r_f_s.append(get_r_f_s)
                    
        #get last dobj in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_o = "none"
        for n in [1,2,3,4,5,6,7,8]: #in the final version, each of the name objects will be accunted for
            dummy_r_f_o = find_nth_dobj(curr_doc,n)
            if dummy_r_f_o in remainder and check_if_name(curr_tok,dummy_r_f_o):
                get_r_f_o = dummy_r_f_o 
            
        r_f_o.append(get_r_f_o)
                    
        #get last appos in remainder @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_r_f_a = "none"
        for n in [1,2,3,4]:
            dummy_r_f_a = find_nth_appos(curr_doc,n)
            if dummy_r_f_a in remainder and check_if_name(curr_tok,dummy_r_f_a): 
                get_r_f_a = dummy_r_f_a 
            
        r_f_a.append(get_r_f_a)
               
        #get first appos in current @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_c_f_a = "none"
        for n in [1,2,3,4]:
            dummy_c_f_a = find_nth_appos(curr_doc,n)
            if check_if_name(curr_tok,dummy_c_f_a) and get_c_f_a == "none": 
                get_c_f_a = dummy_c_f_a 
            
        c_f_a.append(get_c_f_a)
                
        #get first appos in prev @@@@@@@@@@@@@@@@@@@@@@@@@@@
        get_p_f_a = "none"
        for n in [1,2,3,4]:
            dummy_p_f_a = find_nth_appos(prev_doc,n)
            if check_if_name(prev_tok,dummy_p_f_a) and get_p_f_a == "none": 
                get_p_f_a = dummy_p_f_a 
            
        p_f_a.append(get_p_f_a)
            
        #check_if_poss_her
        poss_her.append(check_if_poss_her(curr_doc, pronoun))
    
        #rand_forest classifier for pronoun type:
        if pronoun == "he" or pronoun == "she": 
            pro_typ.append(1)
        elif pronoun == "He" or pronoun == "She": 
            pro_typ.append(2)
        elif pronoun == "his" or (pronoun == "her" and get_poss_her): 
            pro_typ.append(3)
        elif pronoun == "him" or (pronoun == "her" and not get_poss_her): 
            pro_typ.append(4)
        elif pronoun == "His" or (pronoun == "Her" and get_poss_her): 
            pro_typ.append(5)
        else:
            pro_typ.append(6)
    
        dict_of_all["p_f_s"] = get_p_f_s
        dict_of_all["p_l_s"] = get_p_l_s
        dict_of_all["p_f_o"] = get_p_f_o
        dict_of_all["p_l_o"] = get_p_l_o
        dict_of_all["tc_f_s"] = get_tc_f_s
        dict_of_all["tc_f_o"] = get_tc_f_o
        dict_of_all["tc_f_a"] = get_tc_f_a
        dict_of_all["tc_l_s"] = get_tc_l_s
        dict_of_all["tc_l_o"] = get_tc_l_o
        dict_of_all["tc_l_p"] = get_tc_l_p
        dict_of_all["p_f_wp"] = get_p_f_wp
        dict_of_all["tc_l_wp"] = get_tc_l_wp
        dict_of_all["tc_l_nw"] = get_tc_l_nw
        dict_of_all["r_f_s"] = get_r_f_s
        dict_of_all["r_f_o"] = get_r_f_o
        dict_of_all["r_f_a"] = get_r_f_a
        dict_of_all["p_f_a"] = get_p_f_a
        dict_of_all["c_f_a"] = get_c_f_a
        dict_of_all["poss_her"] = poss_her
        
        dict_of_all_list.append(dict_of_all)


# In[ ]:


#AND NOW COPY-PASTE ALL THAT FEATURE EXTRACTION PROCEDURE FROM THE TRAINING CELLS:

data_matrix = []
data_matrix1 = []
data_matrix2 = []
data_matrix3 = []
data_matrix4 = []
data_matrix5 = []
data_matrix6 = []

te_pro_t1_idxs = []
te_pro_t2_idxs = []
te_pro_t3_idxs = []
te_pro_t4_idxs = []
te_pro_t5_idxs = []
te_pro_t6_idxs = []

for idx in range(len(p_f_s)):
    
    data_vector = []
    
    if p_f_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)    
    if p_l_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)
    if p_f_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)
    if p_l_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)
    if tc_f_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)
    if tc_f_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_f_a[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_p[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if p_f_wp[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_wp[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if tc_l_nw[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if r_f_s[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if r_f_o[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if r_f_a[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if p_f_a[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if c_f_a[idx] == "none":
        data_vector.append(0)
    else:
        data_vector.append(1)   
    if poss_her[idx] == True:
        data_vector.append(1)
    else:
        data_vector.append(0)
    #pronoun type, already numerical
    if pro_typ[idx] == 1:
        te_pro_t1_idxs.append(idx)
        data_matrix1.append(data_vector)
    if pro_typ[idx] == 2:
        te_pro_t2_idxs.append(idx)
        data_matrix2.append(data_vector)
    if pro_typ[idx] == 3:
        te_pro_t3_idxs.append(idx)
        data_matrix3.append(data_vector)
    if pro_typ[idx] == 4:
        te_pro_t4_idxs.append(idx)
        data_matrix4.append(data_vector)
    if pro_typ[idx] == 5:
        te_pro_t5_idxs.append(idx)
        data_matrix5.append(data_vector)
    if pro_typ[idx] == 6:
        te_pro_t6_idxs.append(idx)
        data_matrix6.append(data_vector)
    
    data_matrix.append(data_vector)
    
test1_df = pd.DataFrame(data_matrix1, columns = dict_of_all_list[0].keys())
test2_df = pd.DataFrame(data_matrix2, columns = dict_of_all_list[0].keys())
test3_df = pd.DataFrame(data_matrix3, columns = dict_of_all_list[0].keys())
test4_df = pd.DataFrame(data_matrix4, columns = dict_of_all_list[0].keys())
test5_df = pd.DataFrame(data_matrix5, columns = dict_of_all_list[0].keys())
test6_df = pd.DataFrame(data_matrix6, columns = dict_of_all_list[0].keys())


# In[ ]:


def train_n_pred(train_df, test_df, clf, train_y_list, train_y_idx):
    features = train_df.columns[:-1]
    train_y = [train_y_list[i] for i in train_y_idx]
    train_y = np.asarray(train_y)
    clf.fit(train_df[features], train_y)
    return clf.predict(test_df[features])


# In[ ]:


# CASE I : PRONOUN TYPE 1:
pred_C1_p_f_s = train_n_pred(clf1_df, test1_df, clf, p_f_s_clf, tr_pro_t1_idxs)
pred_C1_p_l_s = train_n_pred(clf1_df, test1_df, clf, p_l_s_clf, tr_pro_t1_idxs)
pred_C1_p_f_o = train_n_pred(clf1_df, test1_df, clf, p_f_o_clf, tr_pro_t1_idxs)
pred_C1_p_l_o = train_n_pred(clf1_df, test1_df, clf, p_l_o_clf, tr_pro_t1_idxs)
pred_C1_tc_f_s = train_n_pred(clf1_df, test1_df, clf, tc_f_s_clf, tr_pro_t1_idxs)
pred_C1_tc_f_o = train_n_pred(clf1_df, test1_df, clf, tc_f_o_clf, tr_pro_t1_idxs)
pred_C1_tc_f_a = train_n_pred(clf1_df, test1_df, clf, tc_f_a_clf, tr_pro_t1_idxs)
pred_C1_tc_l_s = train_n_pred(clf1_df, test1_df, clf, tc_l_s_clf, tr_pro_t1_idxs)
pred_C1_tc_l_o = train_n_pred(clf1_df, test1_df, clf, tc_l_o_clf, tr_pro_t1_idxs)
pred_C1_tc_l_p = train_n_pred(clf1_df, test1_df, clf, tc_l_p_clf, tr_pro_t1_idxs)
pred_C1_p_f_wp = train_n_pred(clf1_df, test1_df, clf, p_f_wp_clf, tr_pro_t1_idxs)
pred_C1_tc_l_wp = train_n_pred(clf1_df, test1_df, clf, tc_l_wp_clf, tr_pro_t1_idxs)
pred_C1_tc_l_nw = train_n_pred(clf1_df, test1_df, clf, tc_l_nw_clf, tr_pro_t1_idxs)
pred_C1_r_f_s = train_n_pred(clf1_df, test1_df, clf, r_f_s_clf, tr_pro_t1_idxs)
pred_C1_r_f_o = train_n_pred(clf1_df, test1_df, clf, r_f_o_clf, tr_pro_t1_idxs)
pred_C1_r_f_a = train_n_pred(clf1_df, test1_df, clf, r_f_a_clf, tr_pro_t1_idxs)
pred_C1_p_f_a = train_n_pred(clf1_df, test1_df, clf, p_f_a_clf, tr_pro_t1_idxs)
pred_C1_c_f_a = train_n_pred(clf1_df, test1_df, clf, c_f_a_clf, tr_pro_t1_idxs)

# CASE II : PRONOUN TYPE 2:
pred_C2_p_f_s = train_n_pred(clf2_df, test2_df, clf, p_f_s_clf, tr_pro_t2_idxs)
pred_C2_p_l_s = train_n_pred(clf2_df, test2_df, clf, p_l_s_clf, tr_pro_t2_idxs)
pred_C2_p_f_o = train_n_pred(clf2_df, test2_df, clf, p_f_o_clf, tr_pro_t2_idxs)
pred_C2_p_l_o = train_n_pred(clf2_df, test2_df, clf, p_l_o_clf, tr_pro_t2_idxs)
pred_C2_tc_f_s = train_n_pred(clf2_df, test2_df, clf, tc_f_s_clf, tr_pro_t2_idxs)
pred_C2_tc_f_o = train_n_pred(clf2_df, test2_df, clf, tc_f_o_clf, tr_pro_t2_idxs)
pred_C2_tc_f_a = train_n_pred(clf2_df, test2_df, clf, tc_f_a_clf, tr_pro_t2_idxs)
pred_C2_tc_l_s = train_n_pred(clf2_df, test2_df, clf, tc_l_s_clf, tr_pro_t2_idxs)
pred_C2_tc_l_o = train_n_pred(clf2_df, test2_df, clf, tc_l_o_clf, tr_pro_t2_idxs)
pred_C2_tc_l_p = train_n_pred(clf2_df, test2_df, clf, tc_l_p_clf, tr_pro_t2_idxs)
pred_C2_p_f_wp = train_n_pred(clf2_df, test2_df, clf, p_f_wp_clf, tr_pro_t2_idxs)
pred_C2_tc_l_wp = train_n_pred(clf2_df, test2_df, clf, tc_l_wp_clf, tr_pro_t2_idxs)
pred_C2_tc_l_nw = train_n_pred(clf2_df, test2_df, clf, tc_l_nw_clf, tr_pro_t2_idxs)
pred_C2_r_f_s = train_n_pred(clf2_df, test2_df, clf, r_f_s_clf, tr_pro_t2_idxs)
pred_C2_r_f_o = train_n_pred(clf2_df, test2_df, clf, r_f_o_clf, tr_pro_t2_idxs)
pred_C2_r_f_a = train_n_pred(clf2_df, test2_df, clf, r_f_a_clf, tr_pro_t2_idxs)
pred_C2_p_f_a = train_n_pred(clf2_df, test2_df, clf, p_f_a_clf, tr_pro_t2_idxs)
pred_C2_c_f_a = train_n_pred(clf2_df, test2_df, clf, c_f_a_clf, tr_pro_t2_idxs)

# CASE III : PRONOUN TYPE 3:
pred_C3_p_f_s = train_n_pred(clf3_df, test3_df, clf, p_f_s_clf, tr_pro_t3_idxs)
pred_C3_p_l_s = train_n_pred(clf3_df, test3_df, clf, p_l_s_clf, tr_pro_t3_idxs)
pred_C3_p_f_o = train_n_pred(clf3_df, test3_df, clf, p_f_o_clf, tr_pro_t3_idxs)
pred_C3_p_l_o = train_n_pred(clf3_df, test3_df, clf, p_l_o_clf, tr_pro_t3_idxs)
pred_C3_tc_f_s = train_n_pred(clf3_df, test3_df, clf, tc_f_s_clf, tr_pro_t3_idxs)
pred_C3_tc_f_o = train_n_pred(clf3_df, test3_df, clf, tc_f_o_clf, tr_pro_t3_idxs)
pred_C3_tc_f_a = train_n_pred(clf3_df, test3_df, clf, tc_f_a_clf, tr_pro_t3_idxs)
pred_C3_tc_l_s = train_n_pred(clf3_df, test3_df, clf, tc_l_s_clf, tr_pro_t3_idxs)
pred_C3_tc_l_o = train_n_pred(clf3_df, test3_df, clf, tc_l_o_clf, tr_pro_t3_idxs)
pred_C3_tc_l_p = train_n_pred(clf3_df, test3_df, clf, tc_l_p_clf, tr_pro_t3_idxs)
pred_C3_p_f_wp = train_n_pred(clf3_df, test3_df, clf, p_f_wp_clf, tr_pro_t3_idxs)
pred_C3_tc_l_wp = train_n_pred(clf3_df, test3_df, clf, tc_l_wp_clf, tr_pro_t3_idxs)
pred_C3_tc_l_nw = train_n_pred(clf3_df, test3_df, clf, tc_l_nw_clf, tr_pro_t3_idxs)
pred_C3_r_f_s = train_n_pred(clf3_df, test3_df, clf, r_f_s_clf, tr_pro_t3_idxs)
pred_C3_r_f_o = train_n_pred(clf3_df, test3_df, clf, r_f_o_clf, tr_pro_t3_idxs)
pred_C3_r_f_a = train_n_pred(clf3_df, test3_df, clf, r_f_a_clf, tr_pro_t3_idxs)
pred_C3_p_f_a = train_n_pred(clf3_df, test3_df, clf, p_f_a_clf, tr_pro_t3_idxs)
pred_C3_c_f_a = train_n_pred(clf3_df, test3_df, clf, c_f_a_clf, tr_pro_t3_idxs)

# CASE IV : PRONOUN TYPE 4:
pred_C4_p_f_s = train_n_pred(clf4_df, test4_df, clf, p_f_s_clf, tr_pro_t4_idxs)
pred_C4_p_l_s = train_n_pred(clf4_df, test4_df, clf, p_l_s_clf, tr_pro_t4_idxs)
pred_C4_p_f_o = train_n_pred(clf4_df, test4_df, clf, p_f_o_clf, tr_pro_t4_idxs)
pred_C4_p_l_o = train_n_pred(clf4_df, test4_df, clf, p_l_o_clf, tr_pro_t4_idxs)
pred_C4_tc_f_s = train_n_pred(clf4_df, test4_df, clf, tc_f_s_clf, tr_pro_t4_idxs)
pred_C4_tc_f_o = train_n_pred(clf4_df, test4_df, clf, tc_f_o_clf, tr_pro_t4_idxs)
pred_C4_tc_f_a = train_n_pred(clf4_df, test4_df, clf, tc_f_a_clf, tr_pro_t4_idxs)
pred_C4_tc_l_s = train_n_pred(clf4_df, test4_df, clf, tc_l_s_clf, tr_pro_t4_idxs)
pred_C4_tc_l_o = train_n_pred(clf4_df, test4_df, clf, tc_l_o_clf, tr_pro_t4_idxs)
pred_C4_tc_l_p = train_n_pred(clf4_df, test4_df, clf, tc_l_p_clf, tr_pro_t4_idxs)
pred_C4_p_f_wp = train_n_pred(clf4_df, test4_df, clf, p_f_wp_clf, tr_pro_t4_idxs)
pred_C4_tc_l_wp = train_n_pred(clf4_df, test4_df, clf, tc_l_wp_clf, tr_pro_t4_idxs)
pred_C4_tc_l_nw = train_n_pred(clf4_df, test4_df, clf, tc_l_nw_clf, tr_pro_t4_idxs)
pred_C4_r_f_s = train_n_pred(clf4_df, test4_df, clf, r_f_s_clf, tr_pro_t4_idxs)
pred_C4_r_f_o = train_n_pred(clf4_df, test4_df, clf, r_f_o_clf, tr_pro_t4_idxs)
pred_C4_r_f_a = train_n_pred(clf4_df, test4_df, clf, r_f_a_clf, tr_pro_t4_idxs)
pred_C4_p_f_a = train_n_pred(clf4_df, test4_df, clf, p_f_a_clf, tr_pro_t4_idxs)
pred_C4_c_f_a = train_n_pred(clf4_df, test4_df, clf, c_f_a_clf, tr_pro_t4_idxs)

# CASE V : PRONOUN TYPE V:
pred_C5_p_f_s = train_n_pred(clf5_df, test5_df, clf, p_f_s_clf, tr_pro_t5_idxs)
pred_C5_p_l_s = train_n_pred(clf5_df, test5_df, clf, p_l_s_clf, tr_pro_t5_idxs)
pred_C5_p_f_o = train_n_pred(clf5_df, test5_df, clf, p_f_o_clf, tr_pro_t5_idxs)
pred_C5_p_l_o = train_n_pred(clf5_df, test5_df, clf, p_l_o_clf, tr_pro_t5_idxs)
pred_C5_tc_f_s = train_n_pred(clf5_df, test5_df, clf, tc_f_s_clf, tr_pro_t5_idxs)
pred_C5_tc_f_o = train_n_pred(clf5_df, test5_df, clf, tc_f_o_clf, tr_pro_t5_idxs)
pred_C5_tc_f_a = train_n_pred(clf5_df, test5_df, clf, tc_f_a_clf, tr_pro_t5_idxs)
pred_C5_tc_l_s = train_n_pred(clf5_df, test5_df, clf, tc_l_s_clf, tr_pro_t5_idxs)
pred_C5_tc_l_o = train_n_pred(clf5_df, test5_df, clf, tc_l_o_clf, tr_pro_t5_idxs)
pred_C5_tc_l_p = train_n_pred(clf5_df, test5_df, clf, tc_l_p_clf, tr_pro_t5_idxs)
pred_C5_p_f_wp = train_n_pred(clf5_df, test5_df, clf, p_f_wp_clf, tr_pro_t5_idxs)
pred_C5_tc_l_wp = train_n_pred(clf5_df, test5_df, clf, tc_l_wp_clf, tr_pro_t5_idxs)
pred_C5_tc_l_nw = train_n_pred(clf5_df, test5_df, clf, tc_l_nw_clf, tr_pro_t5_idxs)
pred_C5_r_f_s = train_n_pred(clf5_df, test5_df, clf, r_f_s_clf, tr_pro_t5_idxs)
pred_C5_r_f_o = train_n_pred(clf5_df, test5_df, clf, r_f_o_clf, tr_pro_t5_idxs)
pred_C5_r_f_a = train_n_pred(clf5_df, test5_df, clf, r_f_a_clf, tr_pro_t5_idxs)
pred_C5_p_f_a = train_n_pred(clf5_df, test5_df, clf, p_f_a_clf, tr_pro_t5_idxs)
pred_C5_c_f_a = train_n_pred(clf5_df, test5_df, clf, c_f_a_clf, tr_pro_t5_idxs)


# In[ ]:


#Now convert predictions to list of names: (The lists were over-written with test data.)
list_of_pred_names = []
for idx in range(len(p_f_s)):
    current_predictions = []
    
    if idx in te_pro_t1_idxs:
        pred_p_f_s = pred_C1_p_f_s[te_pro_t1_idxs.index(idx)]  
        pred_p_l_s = pred_C1_p_l_s[te_pro_t1_idxs.index(idx)]
        pred_p_f_o = pred_C1_p_f_o[te_pro_t1_idxs.index(idx)]
        pred_p_l_o = pred_C1_p_l_o[te_pro_t1_idxs.index(idx)]
        pred_tc_f_s = pred_C1_tc_f_s[te_pro_t1_idxs.index(idx)]
        pred_tc_f_o = pred_C1_tc_f_o[te_pro_t1_idxs.index(idx)]
        pred_tc_f_a = pred_C1_tc_f_a[te_pro_t1_idxs.index(idx)]
        pred_tc_l_s = pred_C1_tc_l_s[te_pro_t1_idxs.index(idx)]
        pred_tc_l_o = pred_C1_tc_l_o[te_pro_t1_idxs.index(idx)]
        pred_tc_l_p = pred_C1_tc_l_p[te_pro_t1_idxs.index(idx)]
        pred_p_f_wp = pred_C1_p_f_wp[te_pro_t1_idxs.index(idx)]
        pred_tc_l_wp = pred_C1_tc_l_wp[te_pro_t1_idxs.index(idx)]
        pred_tc_l_nw = pred_C1_tc_l_nw[te_pro_t1_idxs.index(idx)]
        pred_r_f_s = pred_C1_r_f_s[te_pro_t1_idxs.index(idx)]
        pred_r_f_o = pred_C1_r_f_o[te_pro_t1_idxs.index(idx)]
        pred_r_f_a = pred_C1_r_f_a[te_pro_t1_idxs.index(idx)]
        pred_p_f_a = pred_C1_p_f_a[te_pro_t1_idxs.index(idx)]
        pred_c_f_a = pred_C1_c_f_a[te_pro_t1_idxs.index(idx)]
    if idx in te_pro_t2_idxs:
        pred_p_f_s = pred_C2_p_f_s[te_pro_t2_idxs.index(idx)]
        pred_p_l_s = pred_C2_p_l_s[te_pro_t2_idxs.index(idx)]
        pred_p_f_o = pred_C2_p_f_o[te_pro_t2_idxs.index(idx)]
        pred_p_l_o = pred_C2_p_l_o[te_pro_t2_idxs.index(idx)]
        pred_tc_f_s = pred_C2_tc_f_s[te_pro_t2_idxs.index(idx)]
        pred_tc_f_o = pred_C2_tc_f_o[te_pro_t2_idxs.index(idx)]
        pred_tc_f_a = pred_C2_tc_f_a[te_pro_t2_idxs.index(idx)]
        pred_tc_l_s = pred_C2_tc_l_s[te_pro_t2_idxs.index(idx)]
        pred_tc_l_o = pred_C2_tc_l_o[te_pro_t2_idxs.index(idx)]
        pred_tc_l_p = pred_C2_tc_l_p[te_pro_t2_idxs.index(idx)]
        pred_p_f_wp = pred_C2_p_f_wp[te_pro_t2_idxs.index(idx)]
        pred_tc_l_wp = pred_C2_tc_l_wp[te_pro_t2_idxs.index(idx)]
        pred_tc_l_nw = pred_C2_tc_l_nw[te_pro_t2_idxs.index(idx)]
        pred_r_f_s = pred_C2_r_f_s[te_pro_t2_idxs.index(idx)]
        pred_r_f_o = pred_C2_r_f_o[te_pro_t2_idxs.index(idx)]
        pred_r_f_a = pred_C2_r_f_a[te_pro_t2_idxs.index(idx)]
        pred_p_f_a = pred_C2_p_f_a[te_pro_t2_idxs.index(idx)]
        pred_c_f_a = pred_C2_c_f_a[te_pro_t2_idxs.index(idx)]
    if idx in te_pro_t3_idxs:
        pred_p_f_s = pred_C3_p_f_s[te_pro_t3_idxs.index(idx)]
        pred_p_l_s = pred_C3_p_l_s[te_pro_t3_idxs.index(idx)]
        pred_p_f_o = pred_C3_p_f_o[te_pro_t3_idxs.index(idx)]
        pred_p_l_o = pred_C3_p_l_o[te_pro_t3_idxs.index(idx)]
        pred_tc_f_s = pred_C3_tc_f_s[te_pro_t3_idxs.index(idx)]
        pred_tc_f_o = pred_C3_tc_f_o[te_pro_t3_idxs.index(idx)]
        pred_tc_f_a = pred_C3_tc_f_a[te_pro_t3_idxs.index(idx)]
        pred_tc_l_s = pred_C3_tc_l_s[te_pro_t3_idxs.index(idx)]
        pred_tc_l_o = pred_C3_tc_l_o[te_pro_t3_idxs.index(idx)]
        pred_tc_l_p = pred_C3_tc_l_p[te_pro_t3_idxs.index(idx)]
        pred_p_f_wp = pred_C3_p_f_wp[te_pro_t3_idxs.index(idx)]
        pred_tc_l_wp = pred_C3_tc_l_wp[te_pro_t3_idxs.index(idx)]
        pred_tc_l_nw = pred_C3_tc_l_nw[te_pro_t3_idxs.index(idx)]
        pred_r_f_s = pred_C3_r_f_s[te_pro_t3_idxs.index(idx)]
        pred_r_f_o = pred_C3_r_f_o[te_pro_t3_idxs.index(idx)]
        pred_r_f_a = pred_C3_r_f_a[te_pro_t3_idxs.index(idx)]
        pred_p_f_a = pred_C3_p_f_a[te_pro_t3_idxs.index(idx)]
        pred_c_f_a = pred_C3_c_f_a[te_pro_t3_idxs.index(idx)]
    if idx in te_pro_t4_idxs:
        pred_p_f_s = pred_C4_p_f_s[te_pro_t4_idxs.index(idx)]
        pred_p_l_s = pred_C4_p_l_s[te_pro_t4_idxs.index(idx)]
        pred_p_f_o = pred_C4_p_f_o[te_pro_t4_idxs.index(idx)]
        pred_p_l_o = pred_C4_p_l_o[te_pro_t4_idxs.index(idx)]
        pred_tc_f_s = pred_C4_tc_f_s[te_pro_t4_idxs.index(idx)]
        pred_tc_f_o = pred_C4_tc_f_o[te_pro_t4_idxs.index(idx)]
        pred_tc_f_a = pred_C4_tc_f_a[te_pro_t4_idxs.index(idx)]
        pred_tc_l_s = pred_C4_tc_l_s[te_pro_t4_idxs.index(idx)]
        pred_tc_l_o = pred_C4_tc_l_o[te_pro_t4_idxs.index(idx)]
        pred_tc_l_p = pred_C4_tc_l_p[te_pro_t4_idxs.index(idx)]
        pred_p_f_wp = pred_C4_p_f_wp[te_pro_t4_idxs.index(idx)]
        pred_tc_l_wp = pred_C4_tc_l_wp[te_pro_t4_idxs.index(idx)]
        pred_tc_l_nw = pred_C4_tc_l_nw[te_pro_t4_idxs.index(idx)]
        pred_r_f_s = pred_C4_r_f_s[te_pro_t4_idxs.index(idx)]
        pred_r_f_o = pred_C4_r_f_o[te_pro_t4_idxs.index(idx)]
        pred_r_f_a = pred_C4_r_f_a[te_pro_t4_idxs.index(idx)]
        pred_p_f_a = pred_C4_p_f_a[te_pro_t4_idxs.index(idx)]
        pred_c_f_a = pred_C4_c_f_a[te_pro_t4_idxs.index(idx)]
    if idx in te_pro_t5_idxs:
        pred_p_f_s = pred_C5_p_f_s[te_pro_t5_idxs.index(idx)]
        pred_p_l_s = pred_C5_p_l_s[te_pro_t5_idxs.index(idx)]
        pred_p_f_o = pred_C5_p_f_o[te_pro_t5_idxs.index(idx)]
        pred_p_l_o = pred_C5_p_l_o[te_pro_t5_idxs.index(idx)]
        pred_tc_f_s = pred_C5_tc_f_s[te_pro_t5_idxs.index(idx)]
        pred_tc_f_o = pred_C5_tc_f_o[te_pro_t5_idxs.index(idx)]
        pred_tc_f_a = pred_C5_tc_f_a[te_pro_t5_idxs.index(idx)]
        pred_tc_l_s = pred_C5_tc_l_s[te_pro_t5_idxs.index(idx)]
        pred_tc_l_o = pred_C5_tc_l_o[te_pro_t5_idxs.index(idx)]
        pred_tc_l_p = pred_C5_tc_l_p[te_pro_t5_idxs.index(idx)]
        pred_p_f_wp = pred_C5_p_f_wp[te_pro_t5_idxs.index(idx)]
        pred_tc_l_wp = pred_C5_tc_l_wp[te_pro_t5_idxs.index(idx)]
        pred_tc_l_nw = pred_C5_tc_l_nw[te_pro_t5_idxs.index(idx)]
        pred_r_f_s = pred_C5_r_f_s[te_pro_t5_idxs.index(idx)]
        pred_r_f_o = pred_C5_r_f_o[te_pro_t5_idxs.index(idx)]
        pred_r_f_a = pred_C5_r_f_a[te_pro_t5_idxs.index(idx)]
        pred_p_f_a = pred_C5_p_f_a[te_pro_t5_idxs.index(idx)]
        pred_c_f_a = pred_C5_c_f_a[te_pro_t5_idxs.index(idx)]
    if idx in te_pro_t6_idxs:
        pred_p_f_s = pred_C6_p_f_s[te_pro_t6_idxs.index(idx)]
        pred_p_l_s = pred_C6_p_l_s[te_pro_t6_idxs.index(idx)]
        pred_p_f_o = pred_C6_p_f_o[te_pro_t6_idxs.index(idx)]
        pred_p_l_o = pred_C6_p_l_o[te_pro_t6_idxs.index(idx)]
        pred_tc_f_s = pred_C6_tc_f_s[te_pro_t6_idxs.index(idx)]
        pred_tc_f_o = pred_C6_tc_f_o[te_pro_t6_idxs.index(idx)]
        pred_tc_f_a = pred_C6_tc_f_a[te_pro_t6_idxs.index(idx)]
        pred_tc_l_s = pred_C6_tc_l_s[te_pro_t6_idxs.index(idx)]
        pred_tc_l_o = pred_C6_tc_l_o[te_pro_t6_idxs.index(idx)]
        pred_tc_l_p = pred_C6_tc_l_p[te_pro_t6_idxs.index(idx)]
        pred_p_f_wp = pred_C6_p_f_wp[te_pro_t6_idxs.index(idx)]
        pred_tc_l_wp = pred_C6_tc_l_wp[te_pro_t6_idxs.index(idx)]
        pred_tc_l_nw = pred_C6_tc_l_nw[te_pro_t6_idxs.index(idx)]
        pred_r_f_s = pred_C6_r_f_s[te_pro_t6_idxs.index(idx)]
        pred_r_f_o = pred_C6_r_f_o[te_pro_t6_idxs.index(idx)]
        pred_r_f_a = pred_C6_r_f_a[te_pro_t6_idxs.index(idx)]
        pred_p_f_a = pred_C6_p_f_a[te_pro_t6_idxs.index(idx)]
        pred_c_f_a = pred_C6_c_f_a[te_pro_t6_idxs.index(idx)]
    
    if pred_p_f_s == 1:
        current_predictions.append(p_f_s[idx])
    if pred_p_l_s == 1:
        current_predictions.append(p_l_s[idx])
    if pred_p_f_o == 1:
        current_predictions.append(p_f_o[idx])
    if pred_p_l_o == 1:
        current_predictions.append(p_l_o[idx])
    if pred_tc_f_s == 1:
        current_predictions.append(tc_f_s[idx])
    if pred_tc_f_o == 1:
        current_predictions.append(tc_f_o[idx])
    if pred_tc_f_a == 1:
        current_predictions.append(tc_f_a[idx])
    if pred_tc_l_s == 1:
        current_predictions.append(tc_l_s[idx])
    if pred_tc_l_o == 1:
        current_predictions.append(tc_l_o[idx])
    if pred_tc_l_p == 1:
        current_predictions.append(tc_l_p[idx])
    if pred_p_f_wp == 1:
        current_predictions.append(p_f_wp[idx])
    if pred_tc_l_wp == 1:
        current_predictions.append(tc_l_wp[idx])
    if pred_tc_l_nw == 1:
        current_predictions.append(tc_l_nw[idx])
    if pred_r_f_s == 1:
        current_predictions.append(r_f_s[idx])
    if pred_r_f_o == 1:
        current_predictions.append(r_f_o[idx])
    if pred_r_f_a == 1:
        current_predictions.append(r_f_a[idx])
    if pred_p_f_a == 1:
        current_predictions.append(p_f_a[idx])
    if pred_c_f_a == 1:
        current_predictions.append(c_f_a[idx])
        
    list_of_pred_names.append(current_predictions)


# In[ ]:


#Compare Random Forest preds with A and B: 
with open('../input/gendered-pronoun-resolution/test_stage_1.tsv') as tsvfile:
    
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    count_idx = 0
    test_ids = []
    results_A = []
    results_B = []
    results_N = []
    
    for row in reader:
        
        result_A = 0.33
        result_B = 0.33
        result_N = 0.33
        
        test_ids.append(row['ID'])
        
        num_A = 0
        num_B = 0
            
        for name in list_of_pred_names[count_idx]:    
            if name in row['A']:
                num_A += 1 
            elif name in row['B']:
                num_B += 1
                
        if num_A >= 1:
            result_A = 0.56 + num_A*0.2 #numbers are arbitrary
        if num_B >= 1:
            result_B = 0.56 + num_B*0.2 #numbers are arbitrary
        if num_A > num_B:
            result_A += 0.15 
            result_B -= 0.15
        if num_A == 0 and num_B == 0:
            result_N = 0.86
            result_A = 0.1
            result_B = 0.1
        
        results_A.append(result_A)
        results_B.append(result_B)
        results_N.append(result_N)
        
        count_idx += 1


# In[ ]:


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y, pred_y))


# In[ ]:


his_her_list = []
count = 0
for pronoun in pronoun_list:
    if pronoun == "his" or pronoun == "her":
        his_her_list.append(count)
    count += 1


# In[ ]:


### rules for "her" or "his": # represents sent num (each sent diff rule):
#1 if no subj or obj in trunc curr, then not the prev subj but the prev obj
#2


# In[ ]:


analyze = "Slant Magazine's Sal Cinquemani viewed the album as formulaic and ``competently, often frustratingly more of the same from an artist who still seems capable of much more.'' Greg Kot of the Chicago Tribune perceived ``formula production and hack songwriting'', but complimented Pink's personality and its ``handful'' of worthy tracks. In his list for The Barnes & Noble Review, Robert Christgau named The Truth About Love the fourth best album of 2012."


# In[ ]:


curr, prev, trunc_curr, remainder = curr_prev_sentence(analyze, pronoun_offset_list[5])
analyze_para_lst = find_name_words(name_btwn_paran(trunc_curr))
tok = word_tokenize(curr)


# In[ ]:


doc = nlp(curr)


# In[ ]:


list_of_name_words(tok)


# print(text_list[0])
# print("0")
# print(text_list[1])
# print("1")
# print(text_list[2])
# print("2")
# print(text_list[3])
# print("3")
# print(text_list[5])
# print("5")
# print(text_list[7])
# print("7")
# print(text_list[14])
# print("14")
# print(text_list[16])
# print("16")
# print(text_list[18])
# print("18")
# print(text_list[19])
# print("19")
# print(text_list[23])
# print("23")
# print(text_list[24])
# print("24")
# print(text_list[25])
# print("25")
# print(text_list[26])
# print("26")
# print(text_list[27])
# print("27")
# print(text_list[28])
# print("28")
# print(text_list[29])
# print("29")
# print(text_list[220])
# print("220")
# print(text_list[221])
# print("221")
# print(text_list[224])
# print("224")
# print(text_list[226])
# print("226")
# print(text_list[228])
# print("228")
# print(text_list[229])
# print("229")
# print(text_list[230])
# print("230")
# print(text_list[231])
# print("231")
# print(text_list[232])
# print("232")
# print(text_list[233])
# print("233")
# print(text_list[234])
# print("234")
# print(text_list[237])
# print("237")
# print(text_list[240])
# print("240")
# print(text_list[245])
# print("245")
# print(text_list[246])
# print("246")
# print(text_list[247])
# print("247")
# print(text_list[249])
# print("249")

# In[ ]:


curr_tok = word_tokenize(curr)
for n in [1,2,3,4]:
    dummy_r_f_a = find_nth_appos(doc,n)
    #if dummy_r_f_a in remainder and check_if_name(curr_tok,dummy_r_f_a): 
    print(dummy_r_f_a) 


# In[ ]:


out_df = pd.DataFrame({"ID":test_ids})


# In[ ]:


out_df['A'] = results_A
out_df['B'] = results_B
out_df['NEITHER'] = results_N


# In[ ]:


out_df.to_csv("submission.csv", index=False)

