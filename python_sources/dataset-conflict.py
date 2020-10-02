#!/usr/bin/env python
# coding: utf-8

# ## Problem - Original Post
# It seems that this dataset has different examples in the `sample_submission.csv` as compared to the test set `simplified-nq-test.jsonl`. This very short kernel illustrates that difference.

# ## Update
# Thanks to comments by @mourad below and @kashnitsky in [this thread](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/123781https://www.kaggle.com/c/tensorflow2-question-answering/discussion/123781), I have realised this is not a data conflict but an error in my code. My code uses `pandas.read_json` without specifying data type, so the `example_id` is read as an integer type with insufficient accuracy for the 20-digit id.
# 
# I have added a solution (specifying 'Object' data type) in the last cell of this notebook, in case anyone else gets stuck on a similar problem.

# In[ ]:


import pandas as pd


# In[ ]:


sample = pd.read_csv('/kaggle/input/tensorflow2-question-answering/sample_submission.csv').set_index('example_id')
sample.sort_index().head()


# In[ ]:


test = pd.read_json('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl', orient = 'records', lines = True).set_index('example_id')
test.index = test.index.astype(str)
test.sort_index().head()


# ## Solution
# Specify dtype as 'Object' for `example_id`

# In[ ]:


test2 = pd.read_json('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl',
                    dtype = {'example_id': 'Object'},
                    lines = True).set_index('example_id')
test2.sort_index().head()


# In[ ]:




