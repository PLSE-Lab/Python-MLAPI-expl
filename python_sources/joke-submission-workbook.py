#!/usr/bin/env python
# coding: utf-8

# # Joke Submissions for the GA Customer Revenue Prediction Competition
# Because life is too short, this workbook: <br>
# 1. Generates a submission file that predicts **no return customers** (because the proportion of return to new customers in a given time period is **TINY**, and this will likely be a hard benchmark to beat come Feb!)
# 2. Generates a submission file that calculates the **right answer** through aggregating the revenue per customer in the test file - in essence, calculating the "hidden" answer (because it feels great being at the top of a leaderboard, even when you know your method cannot generalize :p)

# In[ ]:


import pandas as pd
import numpy as np
import ast # for dict parsing down the track


# In[ ]:


test = pd.read_csv('../input/test_v2.csv',dtype={'fullVisitorId': 'str'})


# In[ ]:


submission_sample = pd.read_csv('../input/sample_submission_v2.csv',dtype={'fullVisitorId': 'str'})
submission_sample.shape


# In[ ]:


submission_sample.head()


# In[ ]:


test.shape


# ## 1. Submission File Predicting No Return Customers

# In[ ]:


test_prep = test.copy()
test_prep['PredictedLogRevenue'] = 0
submission_no_return = test_prep.groupby('fullVisitorId',as_index=False).PredictedLogRevenue.sum()
submission_no_return.head()


# In[ ]:


submission_no_return.shape


# In[ ]:


submission_no_return.to_csv('submission_no_return.csv',index=False)


# ## 2. Submission File Aggregating Right Answer

# In[ ]:


test_prep_2 = test.copy()
test_prep_2['Revenue'] = test_prep_2['totals'].apply(lambda x: ast.literal_eval(x).get('transactionRevenue',np.nan))


# In[ ]:


test_prep_2['Revenue'] =pd.to_numeric(test_prep_2['Revenue'],errors='coerce')


# In[ ]:


submission_right_answer = test_prep_2.groupby('fullVisitorId',as_index=False).Revenue.sum()
submission_right_answer['PredictedLogRevenue'] = submission_right_answer.Revenue.apply(lambda x: np.log(x+1))
submission_right_answer.drop(columns='Revenue',inplace=True)
submission_right_answer.head()


# In[ ]:


submission_right_answer.shape


# In[ ]:


submission_right_answer.to_csv('submission_right_answer.csv',index=False)


# ## 3. How do these perform?
# No return customers? Scores **RMSE of 2.11** <br>
# ![Predictably bad](https://imgur.com/YSxDQbh.png)
# Right answer? Scores **RMSE of 0.00** <br>
# ![So good](https://imgur.com/roUWAdp.png)
# 

# # End Joke
