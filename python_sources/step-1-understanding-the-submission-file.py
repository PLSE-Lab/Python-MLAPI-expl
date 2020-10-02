#!/usr/bin/env python
# coding: utf-8

# # Predict Future Sales
# ## Step 1: Understanding the Submission File
# 
# This notebook explains the steps I took to understand the sample submission file and to prepare and submit a first batch of 'predictions' to the competitive-data-science-predict-future-sales competition.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from pathlib import Path
p=Path("/kaggle/input/competitive-data-science-predict-future-sales/")


# First, what files do we have available?

# In[ ]:


get_ipython().system('ls -lh /kaggle/input/competitive-data-science-predict-future-sales/')


# I know from the description that this competition is about predicting the number of sales for a given month across a number of products (items) across a number of stores. 
# 
# I also know that to get a score on the leaderboard I have to submit a `csv` file containing my predicted sales and that this file should match the format of `sample_submission.csv`. 
# 
# So first off I'll try to understand this file and then make a successful submission to the competition. By doing so I hope to be clearer about what it is I have to submit (and so what it is I have to predict). I'm not planning on looking at any of the historical sales data at this stage.

# In[ ]:


sample_submission = pd.read_csv(p/"sample_submission.csv")
sample_submission.head()


# Interesting. There is no mention of shop or item explicitly in the `sample_submission.csv` file. 
# 
# Instead we have an `ID` column. This `ID` corresponds to a particular shop/item combination for which we require a prediction. The lookup for which shop and item corresponds to each ID is given in the file `test.csv`. 

# In[ ]:


test = pd.read_csv(p/"test.csv")
display(test.head())

s1="equal" if sample_submission.ID.nunique() == sample_submission.ID.nunique() else "not equal"
s2="match" if set(sample_submission.ID)==set(sample_submission.ID) else "do not match"

print(f"Number of IDs in sample ({sample_submission.ID.count():,}) is {s1} to test ({test.ID.count():,})")
print(f"Values of unique IDs in sample and test {s2}")


# So the above shows me that the `sample_submission.csv` and the `test.csv` files contain the same set of IDs.
# 
# Therefore I will be producing predictions for each shop/item combination in the `test.csv` file.
# 
# I fancy a little dig around in this test file to check if there's anything unusual about what combinations of shops and items we have to produce predictions for.
# 

# In[ ]:


print(test.columns)
print(f"unique shop_ids: {test.shop_id.nunique():,}")
print(f"unique item_ids: {test.item_id.nunique():,}")
print(f"unique IDs: {test.ID.nunique():,} ({test.shop_id.nunique()} * {test.item_id.nunique():,} = {test.shop_id.nunique()*test.item_id.nunique():,})")
print(f"\n#shop_id")
print(f"range of shop_id = [{test.shop_id.min()},{test.shop_id.max()}] (N={test.shop_id.max()-test.shop_id.min()+1})")
missing_shop_id = sorted(set(range(test.shop_id.min(),test.shop_id.max()+1))-set(test.shop_id))
print(f"shop_id missing from contiguous: {missing_shop_id} (N={len(missing_shop_id)})")
print(f"\n#item_id")
print(f"range of item_id = [{test.item_id.min()},{test.item_id.max()}] (N={test.item_id.max()-test.item_id.min()+1:,})")
missing_item_id = sorted(set(range(test.item_id.min(),test.item_id.max()+1))-set(test.item_id))
print(f"item_id missing from contiguous: (N={len(missing_item_id):,})")


# `shop_id` and `item_id` seem like they could be contiguous ranges of integers but the 'missing' `id`s in the test file suggest that it is a subset of all possible shops and items that I have to produce a prediction for. Although we won't know for sure until understood more about how the id's are generated. 
# 
# Assuming for now that the `shop_id` and `item_id` are from continuous ranges, is there any pattern to the values that have been included? Looking at the missing `shop_id` above I don't really see a pattern to the missing `shop_id`. 
# 
# Eyeballing 17k `item_id` is unlikely to yeild much of an insight. So I created a very simple visualisation to look for patterns in the `item_id` sampling. In the following plot yellow dots show which `item_id` are present over a range of the possible `item_id` assuming they were from a contiguous range. I've just reshaped the data into a grid to give a compact overview. Adjusting the size of the grid didn't seem to yield any patterns but this is of course not an exhaustive analysis!
# 

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(8,8))
W=150
Y=(22168//W)+1
x=np.zeros((len(range(0,W*Y)),));
x[test.item_id.unique()]=1
a=plt.imshow(np.reshape(x,(Y,W)))


# So there are missing `item_id` from across the range. Some contiguous groups present but my impression is that there is no particular pattern to the missing/present `item_id`. At this stage I don't think it's worth digging any deeper into this as we don't really have any hard information about how the ids were generated. Mainly I was just checking for any big contiguous blocks of missing/present ids.
# 
# Also I think it is worth checking to see if I will be making predictions for the same items in each shop or different items in different shops?
# 

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(12,4))
_=test.groupby('shop_id').item_id.count().plot.bar(ax=ax, fontsize=16)
_=ax.set_ylabel('count of item_id', fontsize=16)


# Looks like same 5,100 items in each of the 42 shops! 
# 
# So I take away that we are producing 214,200 predictions in total which represent monthly sales for a particular subset of 5,100 items across 42 shops. These are the shop and item combinations present in the test.csv file, each of which has a unique id.

# ### Making a submission
# 
# Time to test making a submission. Downloading the `sample_submission.csv` file to my local machine and submitting it to the competition via the kaggle api:
# 
#     $> kaggle competitions submit -f sample_submission.csv -m "sample submission" -competition "competitive-data-science-predict-future-sales"
#     
# leads to a score of `1.23646` 
# 
# But really I'd like to set up a bit of machinery for generating my own submission based on the predictions of a model. It's overkill at this stage but helps me think in terms of models, predictions and submissions and the process for generating them reliably.
# 
# As suggested in the course a first prediction could be a constant value for the number of sales for all shop/items. 
# 
# The Class below represents this constant value model and include some convience functions for generating a predictions file for all the shop/items in the test.csv file in the correct format, which can then be downloaded and submitted.

# In[ ]:


from IPython.display import HTML
import base64
import datetime
import json

class ConstantModel:
    def __init__(self, C=1.0, name=None, test_df=None):
        """ 
        Initialise the Model Class.
        test_df: a dataframe containing the test examples
        """
        self.test_df = test_df 
        self.C = C
        self.summary = {}
    
    def predict_sales(self, X_df, return_inputs=True):
        """
        Produce a prediction for given inputs:
        X: the inputs as a dataframe, one example per row
        """
        # we actually ignore any inputs for the constant prediction
        X_df['item_cnt_month'] = self.C
        if return_inputs:
            return X_df
        else:
            return X_df.Y
    
    def _create_download_link(self,df, title = "Download ", filename = "data.csv", include_index=False): 
        """
        Thanks to Racheal Tatman (https://www.kaggle.com/rtatman) for this snippet to create a download link in the notebook.
        """
        csv = df.to_csv(index=include_index)
        b64 = base64.b64encode(csv.encode())
        payload = b64.decode()
        html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(payload=payload,title=title+filename,filename=filename)
        return HTML(html)

    def create_submission(self, print_summary=True):
        fname=f"submission_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        preds=self.predict_sales(self.test_df)[['ID','item_cnt_month']]
        self.summary['predictions'] = {
                                        'count':preds['item_cnt_month'].count(),
                                        'range':{'min':preds.min(), 'max':preds.max()},
                                        'mean':preds['item_cnt_month'].mean(),
                                        'stdev':preds['item_cnt_month'].std(),
                                        'median':preds['item_cnt_month'].median()
                                        }

        if print_summary:
            print(json.dumps(json.loads(pd.DataFrame(self.summary).to_json()),indent=4))
            
        return self._create_download_link(preds,filename=fname)
        
    


# In[ ]:


test_df=pd.read_csv(p/'test.csv')
mymodel = ConstantModel(C=1, test_df=test_df)
mymodel.create_submission()


# On submitting a file that predicts a constant `1` sale for every shop/item I get a leaderboard score of `1.41241`. This was worse than the score for the `sample_submission` file, which made a constant prediction of `0.5`. 
# 
# I decided to try a lower constant value of 0.1:

# In[ ]:


test_df=pd.read_csv(p/'test.csv')
mymodel = ConstantModel(C=0.1, test_df=test_df)
mymodel.create_submission()


# This lead to an improved score of `1.23125` - the best so far.
# 
# This tiny bit of leaderboard has already given us some valuable information. The better performing constant predictions were < 1 - i.e. fractional. A fractional number of sales does not make sense for the number of sales of any individual item. But it shows that the average of the sales counts across all items/shops in the test set is probably less than 1. It's worth remebering that the competition rules state the 'score' we have been receiving is the root mean squared error (RMSE). Therefore our constant predictions are getting us towards the true mean of the `item_cnt_month` across the test set. In the real world a fractional mean for sales probably means that a number of items have integer numbers of sales while some items have no sales at all. 
# 
# ### Summary
# 
# In this notebook I have gone through my very first steps in the competition: understanding what has to be submitted, the evaluation metric and set up some basic machinery to make submissions more repeatable.
# 
# In the next notebook I will look at the suggested next step which is to make predictions based on time-lagged values of sales.

# In[ ]:




