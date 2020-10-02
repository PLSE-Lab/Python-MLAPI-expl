#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


ets_forecasts = pd.read_csv('../input/ensemble-grocery-01/sub_ets_log.csv', index_col=0)
test = pd.read_csv('../input/favorita-grocery-sales-forecasting/test.csv', 
                   parse_dates=['date'], index_col=0)


# In[ ]:


ets_forecasts.head()


# In[ ]:


test.head()


# In[ ]:


results = test[['date']].join(ets_forecasts).sort_values('unit_sales',ascending=False)


# In[ ]:


results.head(20)


# Look at the first row above.  It predicts that some store sold nearly 2 billion units of some item on August 31.  Without bothering to look up which store or which item, I'm pretty confident taking the "under" on that one.
# 
# I would guess that at least the first 10 of the entries above, if not all 20, and possibly a lot more down the list that aren't shown, are going to come in with quite large errors.  But the submission file from which I took these entries earned a very respectable public leaderboard score of 0.556.  As of this writing, if you sort by best score, [the kernel that produced it](https://www.kaggle.com/dongxu027/time-series-ets-starter-lb-0-556) comes in 9th among public kernels,  (And in a sense it really comes in 3rd, becuase all the ones above it represent variations on, and/or ensembles of, just two basic approaches.)
# 
# Notice that all the dates above are closer to the end of the test period, and recall that the test data are split on time.  (Moreover, tarobxl [has demonstrated](https://www.kaggle.com/tarobxl/how-the-test-set-is-split-lb-0-532) exactly how it is split, and none of the dates above are part of the public test data.)
# 
# There are problems with using standard time series methods to project more than a few periods into the future.  Sometimes those forecasts can be useful, but in general they need to be approached with a healthy skepticism.  It can be quite dangerous if you're getting feedback only from near-future results but being evaluated only on far-future results.

# In[ ]:




