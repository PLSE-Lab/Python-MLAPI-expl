#!/usr/bin/env python
# coding: utf-8

# Hi, by this notebook I would like to make an extremely useful tutorial, "How to Appear on Leaderboard"

# In[ ]:


import pandas as pd
submission = pd.DataFrame()
submission['product_title'] = ['Relief deer iphone6s phone shell Apple 7plus 4.7 silicone protective cover cartoon',
                              'Summer new small black and white wave pattern knit short-sleeved']
submission['category'] = ['Mobile Accessories',
                         "Women's Apparel"]
submission.to_csv('sub.csv', index = False)


# > Short and concise, hope this helpful
