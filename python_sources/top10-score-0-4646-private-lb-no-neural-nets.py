#!/usr/bin/env python
# coding: utf-8

# One thing our team has learnt is never underestimating the power of conventional models, like Logistics Regression. We believe simple models should always be attempted before fancy neural nets and deeplearning. 
# 
# Here, we demonstrate how to use simply CountVectorizer + Logistic Regression to achieve near top 10 score.

# In[ ]:


import os,json
from time import time
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[ ]:


data_path = '../input'
submission_path = '.'
def load_df(file):
    print('loading file {} >>>'.format(file))
    df = pd.read_csv(os.path.join(data_path,file))
    print('file dimension:', df.shape)
#     display(df.head())
    return df

def load_json(file):
    with open(os.path.join(data_path,file)) as json_file:  
        attr_map = json.load(json_file)
    return attr_map


# helper functions

# In[ ]:


# update name to map to submission format
def expand_attr(df, attrs):
    r = []
    for col in attrs:
        sub_df = df[df[col].notna()]
        tmp = sub_df['itemid'].astype(str).apply(lambda s: s+'_'+col)
        try:
            sub_df[col] = sub_df[col].astype(int).astype(str)
        except:
            sub_df[col] = sub_df[col].astype(str)
        tmp = pd.concat([tmp,sub_df[col]], axis=1)
        tmp.columns = ['id','tagging']
        r.append(tmp)
#         display(tmp.head(2))
    return pd.concat(r)


def create_submit(submit_df, pred_df):
    return pd.concat([submit_df,pred_df])


# Baseline + Top 2 prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', 'submit_df = pd.DataFrame([],columns=[\'id\',\'tagging\'])\n\nfor cat in [\'beauty\',\'mobile\',\'fashion\']:\n    print(\'#\'*30,\'Category:\',cat,\'#\'*30)\n    train = load_df(cat+\'_data_info_train_competition.csv\')\n    test = load_df(cat+\'_data_info_val_competition.csv\')    \n    attr_info = load_json(cat+\'_profile_train.json\')\n    \n    for col in attr_info.keys():\n        print(\'\\t processing attribute:\',col)\n        pipeline = Pipeline([\n            (\'vect\', CountVectorizer(min_df=1,ngram_range=(1,3))),\n            (\'clf\', LogisticRegression()),\n        ])\n        \n        parameters = {\n            \'vect__ngram_range\': [(1,3),(1,5)],\n        }\n\n        # fit first model\n        train_wo_na = train[~train.isna()[col]]\n        pipeline.fit(train_wo_na[\'title\'], train_wo_na[col])\n        \n        # grid search for best estimator\n        t0 = time()\n        grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2,scoring=\'accuracy\')\n        grid_search.fit(train_wo_na[\'title\'], train_wo_na[col])\n        \n        print("done in %0.3fs" % (time() - t0))\n        print(grid_search.cv_results_[\'mean_test_score\'])\n\n        print("Best score: %0.5f" % grid_search.best_score_)\n        print("Best parameters set:")\n        best_parameters = grid_search.best_estimator_.get_params()\n        for param_name in sorted(parameters.keys()):\n            print("\\t%s: %r" % (param_name, best_parameters[param_name]))\n        \n        # predict on test dataset, select top 2, instead of just one.\n        estimator = grid_search.best_estimator_\n        probs = estimator.predict_proba(test[\'title\'])\n        best_2 = pd.DataFrame(np.argsort(probs, axis=1)[:,-2:],columns=[\'top2\',\'top1\'])\n        test[col] = best_2.apply(lambda row: \' \'.join(estimator.classes_[[row[\'top1\'],row[\'top2\']]].astype(int).astype(str)) ,axis=1)\n        \n        # save models\n#         joblib.dump(pipeline, os.path.join(checkpoints_path,\'model_{}_{}.ckpt\'.format(cat,col)))\n        \n    display(test.head(2))\n    test.fillna(0,inplace=True)\n    pred_df = expand_attr(test, attr_info); print(pred_df.shape)\n    submit_df = create_submit(submit_df, pred_df); print(submit_df.shape)\n    ')


# In[ ]:


submit_df.to_csv(os.path.join(submission_path,'baseline_top2.csv'),index=False)


# This solution only used the most common **CountVectorizer**, **LogisticRegression** with simple Gridsearch. Default parameters are taken except for `ngram_range`. 
# 
# The result can be further improved with model stacking and more sophisticated feature creation. 
