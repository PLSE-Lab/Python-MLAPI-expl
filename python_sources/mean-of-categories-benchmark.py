#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


train = pd.read_csv("../input/google-quest-challenge/train.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")


# In[ ]:


target_cols = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']


# In[ ]:


category_means_map = train.groupby("category")[target_cols].mean().T.to_dict()


# In[ ]:


preds = train["category"].map(category_means_map).apply(pd.Series)


# In[ ]:


from scipy.stats import spearmanr


# In[ ]:


overall_score = 0
for col in target_cols:
    overall_score += spearmanr(preds[col], train[col]).correlation/len(target_cols)
    print(col, spearmanr(preds[col], train[col]).correlation)


# In[ ]:


overall_score


# In[ ]:


test_preds = test["category"].map(category_means_map).apply(pd.Series)


# In[ ]:


sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")


# In[ ]:


for col in target_cols:
    sub[col] = test_preds[col]


# In[ ]:


sub.to_csv("submission.csv", index = False)


# In[ ]:




