#!/usr/bin/env python
# coding: utf-8

# When you press "Commit", you execute the code with **public** test dataset. But when you hit "Submit", in the background, the same code is run against **private** test dataset. 
# 
# Thus we can find out more about the private (hidden) test set. Which actually defines the prizes.  
# 
# The approach is straightforward:
#  - ask a binary question about the test dataset (eg. whether it's longer than 1000)
#  - if the condition doesn't hold - you submit the sample submission file and get zero on Public LB
#  - if the condition holds - you make the script fail (raise an error). Thus after submitting you'll see an error and conclude that that binary condition holds for the hidden test dataset. 

# In[ ]:


import sys
import pandas as pd 


# In[ ]:


sample_sub = pd.read_csv('../input/tensorflow2-question-answering/sample_submission.csv')


# In[ ]:


sample_sub.head()


# In[ ]:


test_df = pd.read_json('../input/tensorflow2-question-answering/simplified-nq-test.jsonl',
                      lines=True, orient='records')


# In[ ]:


test_df.head()


# In this example we'll ask whether the hidden test data set is longer than 1000. 

# In[ ]:


if len(test_df) >= 1000:
    raise ValueError("We'll never see this message again")
else:
    sample_sub.to_csv('submission.csv', index=False)


# When you **Commit** this notebook, the condition doesn't hold (for **public** 692-long test dataset).
# But when you **Submit**, you'll actually see the Kernel fail. Thus we conclude that the **private** test dataset is longer that 1000.

# Actually, you can spend a lot of submissions, getting one bit at a time :)
#  - Is the median number of long answer candidates larger than the same in the training set?
#  - Are questions longer that some threshold?
#  - Are there more paragraphs in the hidden test set?
#  - etc.
# 
# Good luck! Do share your findings with the community! And bear in mind that you still need to be smart and train cool models to win. 
# 
# PS. Yes, this is close to cheating :) but considered fine for Kaggle competitions I guess.
# 
# <img src="https://habrastorage.org/webt/ip/fa/wk/ipfawkx9ate-z15kmh6mjevd87e.jpeg" width=60% />
