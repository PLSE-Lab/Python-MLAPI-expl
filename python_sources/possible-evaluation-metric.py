#!/usr/bin/env python
# coding: utf-8

# ## Evaluation Metric
# 
# There has been some discussion in a number of thread forums. This is not a definitive kernel as I am just as unclear as many participants. However, it does seem to achieve a similar score on the `tiny-dev` set to what the same model achieves on the public test set on the leaderboard. So it could be something close to the metric being used for the LB.

# In[ ]:


import json
import pandas as pd


# The `tiny-dev` set fron NQ is quite useful and can be downloaded from google storage as follows:

# In[ ]:


get_ipython().system('gsutil cp -r gs://bert-nq/tiny-dev .')
get_ipython().system('gunzip tiny-dev/*')
get_ipython().system('ls tiny-dev -hl')


# Please excuse the naming confusion, but theres also a `tinydev` dataset which is a submission csv based on my model's predictions from `nq-dev-sample.jsonl` downloaded above. To evaluate your own model, just make a private dataset with your own csv predictions from this json.

# In[ ]:


predictions = pd.read_csv('../input/tinydev/ken_predictions.csv', na_filter=False).set_index('example_id')


# In[ ]:


def long_annotations(example):
    longs = [('%s:%s' % (l['start_token'],l['end_token']))
                for l in [a['long_answer'] for a in example['annotations']]
                if not l['candidate_index'] == -1
            ]
    return longs #list of long annotations


# In[ ]:


def short_annotations(example):
    shorts = [('%s:%s' % (s['start_token'],s['end_token']))
              for s in 
              # sum(list_of_lists, []) is not very efficient gives an easy flat map for short lists
              sum([a['short_answers'] for a in example['annotations']], [])
             ]
    return shorts #list of short annotations


# In[ ]:


def yes_nos(example):
    return [
        yesno for yesno in [a['yes_no_answer'] for a in example['annotations']]
        if not yesno == 'NONE'
    ]


# In[ ]:


# This is the critical method where I guess at the competition metric.
class Score():
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
    def F1(self):
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)
    def increment(self, prediction, annotations, yes_nos):
        if prediction in yes_nos:
            print(prediction, yes_nos)
            self.TP += 1
        elif len(prediction) > 0:
            if prediction in annotations:
                self.TP += 1
            else:
                self.FP += 1
        elif len(annotations) == 0:
            self.TN += 1
        else:
            self.FN +=1
    def scores(self):
        return 'TP = {}   FP = {}   FN = {}   TN = {}   F1 = {:.2f}'.format(
            self.TP, self.FP, self.FN, self.TN, self.F1())


# In[ ]:


long_score = Score()
short_score = Score()
total_score = Score()
for example in map(json.loads, open('tiny-dev/nq-dev-sample.jsonl', 'r')):
    long_pred = predictions.loc[str(example['example_id']) + '_long', 'PredictionString']
    long_score.increment(long_pred, long_annotations(example), [])
    total_score.increment(long_pred, long_annotations(example), [])
    short_pred = predictions.loc[str(example['example_id']) + '_short', 'PredictionString']
    short_score.increment(short_pred, short_annotations(example), yes_nos(example))
    total_score.increment(short_pred, short_annotations(example), [])


# In[ ]:


print(short_score.scores())


# In[ ]:


print(long_score.scores())


# In[ ]:


print(total_score.scores() + ' (LB score)')

