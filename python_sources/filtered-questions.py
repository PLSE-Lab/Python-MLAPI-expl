#!/usr/bin/env python
# coding: utf-8

# # Installs

# In[ ]:


pip install grammarbot


# ## Check GPU

# In[ ]:


get_ipython().system('nvidia-smi')


# # Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import pandas as pd

# import the client library
from grammarbot import GrammarBotClient

import spacy

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Setup

# In[ ]:


client = GrammarBotClient()

nlp = spacy.load("en_core_web_sm") 


# ### Setup for iterating over every json document

# In[ ]:


CLEAN_CSV_FILE = "/kaggle/input/stanford-plato-corpus/clean.csv"


# In[ ]:


df = pd.read_csv(CLEAN_CSV_FILE, index_col=0)

BAD_COL = "Unnamed: 0.1"
if BAD_COL in df.columns:
    df = df.drop(BAD_COL, axis=1)

BAD_COL = "Unnamed: 0.1.1"
if BAD_COL in df.columns:
    df = df.drop(BAD_COL, axis=1)

BAD_COL = "Unnamed: 0.1.1.1"
if BAD_COL in df.columns:
    df = df.drop(BAD_COL, axis=1)

df.head(1)


# In[ ]:


# @LogDecorator()
# def _json_file_to_dataframe(json_filename):
#     with open(json_filename,'r') as json_file:
#         json_dict = json.loads(json_file.read())
    
#     data = []
#     for k, v in json_dict.items():
#         data.append({"json_filename":json_filename, "question":k, "summarized_answer":v})
#     return pd.DataFrame(data)

# JSON_DIR = '/kaggle/input/stanford-plato-corpus/Questions from 20 docs/Questions from 20 docs/'

# df = pd.concat(
#     [
#         _json_file_to_dataframe(jf) 
#         for jf in (os.path.join(JSON_DIR, jf) 
#          for jf in filter(
#              lambda f: f.endswith(".json"), 
#              os.listdir(JSON_DIR)
#          ))
#     ]
# )


# In[ ]:


df.shape


# In[ ]:


df.head()


# # Heuristics / Functions

# ## Heuristic 1 -- grammar check is good

# In[ ]:



def _get_rule_1_data(question):
    """
    Given a question
    Return a grammar dictionary & and extra bit of data
    """
    rj = client.check(question).raw_json
    extra = [m.get("rule", dict()).get("category", dict()).get("id", "") for m in rj["matches"]]
    return (rj, extra)


def _rule_1(question_data, qd_extra):
    """
    Given a question_data && qd_extra
    Return True/False depending on the rule application
        True means keep this question
        False means forget-about-it
    """
    if question_data["matches"] == []:
        # no grammar problems found
        return True
    elif "TYPOGRAPHY" in qd_extra:
        # TYPOGRAPHY often means some whitespace issues for our questions
        # that's okay, don't worry about it. 
        return True
    else:
        return False


def rule_1(question):
    """
    Given question
    Return a tuple (question_data, boolean_result_of_rule)
    """
    qd, qd_extra = _get_rule_1_data(question)
    res = _rule_1(qd, qd_extra)
    return (qd, qd_extra, res)


# ## Heuristic 2 -- wh-words

# In[ ]:



def _get_rule_2_data(question):
    """
    Given a question
    Return a list of the found wh-words & an extra bit of data
    """
    doc = nlp(question)
    wh_words = []
    for token in doc:
        if token.tag_ in ["WDT","WP", "WP$", "WRB"]:
            wh_words.append({"token": token, "tag": token.tag_ })
    return wh_words, None


def _rule_2(question_data):
    """
    Given a question_data
    Return True/False depending on the rule application
        True means keep this question
        False means forget-about-it
    """
    if len(question_data) <= 1:
        return True
    else:
        return False


def rule_2(question):
    """
    Given question
    Return a tuple (question_data, boolean_result_of_rule)
    """
    qd, _ = _get_rule_2_data(question)
    res = _rule_2(qd)
    return (qd, None, res)


# ## Heuristic 3 -- NER ??

# In[ ]:



def _get_rule_3_data(question):
    """
    Given a question
    Return a list of the found wh-words & an extra bit of data
    """
    doc = nlp(question)
    ent_data = []
    for ent in doc.ents:
        ent_data.append(
            {
                "text": ent.text,
                "label": ent.label_,
            }
        )
    return ent_data, None


def _rule_3(question_data):
    """
    Given a question_data
    Return True/False depending on the rule application
        True means keep this question
        False means forget-about-it
    """
    if len(question_data) >= 1:
        # if there exists at least 1 entity, then maybe prioritize such a question?
        return True
    else:
        return False


def rule_3(question):
    """
    Given question
    Return a tuple (question_data, boolean_result_of_rule)
    """
    qd, _ = _get_rule_3_data(question)
    res = _rule_3(qd)
    return (qd, None, res)


# # Apply The Heuristics

# In[ ]:


# ques  = list()

# rule1_data = list()
# rule1_extra = list()
# rule1_bools = list()

# rule2_data = list()
# rule2_extra = list()
# rule2_bools = list()

# rule3_data = list()
# rule3_extra = list()
# rule3_bools = list()

i = 0
def _with_printouts(fn):
    def _fn(*args, **kwargs):
        global i
        i += 1
        tabs10 = "\t"*10
        tabs100 = "\t"*100
        print(f"{i}{tabs10}{fn.__name__}{tabs10}{args}{tabs10}{kwargs}{tabs100}", end="\r")
        return fn(*args, **kwargs)
    return _fn


fun = _with_printouts(rule_1)
rule_data_frame = df["question"].apply(fun).apply(pd.Series)
rule_data_frame.columns = ["data_rule_1", "extra_rule_1", "bool_rule_1"]
df = df.join(rule_data_frame, rsuffix="_rule_1")

fun = _with_printouts(rule_2)
rule_data_frame = df["question"].apply(fun).apply(pd.Series)
rule_data_frame.columns = ["data_rule_2", "extra_rule_2", "bool_rule_2"]
df = df.join(rule_data_frame, rsuffix="_rule_2")

fun = _with_printouts(rule_3)
rule_data_frame = df["question"].apply(fun).apply(pd.Series)
rule_data_frame.columns = ["data_rule_3", "extra_rule_3", "bool_rule_3"]
df = df.join(rule_data_frame, rsuffix="_rule_3")


# for question in json_dict.keys():
#     ques.append(question)

#     r1d, r1e, r1b = rule_1(question)
#     rule1_data.append(r1d)
#     rule1_extra.append(r1e)
#     rule1_bools.append(r1b)

#     r2d, r2e, r2b = rule_2(question)
#     rule2_data.append(r2d)
#     rule2_extra.append(r2e)
#     rule2_bools.append(r2b)

#     r3d, r3e, r3b = rule_3(question)
#     rule3_data.append(r3d)
#     rule3_extra.append(r3e)
#     rule3_bools.append(r3b)


# In[ ]:


df


# In[ ]:


mask1 = df["bool_rule_1"] & df["bool_rule_2"] 
mask2 = df["bool_rule_1"] & df["bool_rule_3"] 
mask3 = df["bool_rule_2"] & df["bool_rule_3"] 

filtered_df = df[ 
    # if any two bool rules agree with each other, keep it, else filter out
    mask1 | mask2 | mask3 
]


# In[ ]:


df["extra_rule_1"].astype(str).unique()

# df[df["rule1_extra"].astype(str).str.contains("TYPO")]["question"].apply(print)
# df[df["rule1_extra"].astype(str).str.contains("TYPOGRAPHY")]["question"].apply(print)
# df[df["rule1_extra"].astype(str).str.contains("GRAMMAR")]["question"].apply(print)


# In[ ]:


filtered_df


# # Final Rule -- each article must have <= 25 questions

# In[ ]:


filtered_max_25_per_article_df = filtered_df.groupby("title").head(25)


# In[ ]:


filtered_max_25_per_article_df.groupby("title").describe()


# # Save the annotated CSV

# In[ ]:


df.to_csv("annotated.csv")


# # Save the filtered CSV

# In[ ]:


filtered_df.to_csv("filtered.csv")


# # Save the filtered_max_25_per_article_df CSV

# In[ ]:


filtered_max_25_per_article_df.to_csv("filtered_max_25_per_article.csv")


# In[ ]:




