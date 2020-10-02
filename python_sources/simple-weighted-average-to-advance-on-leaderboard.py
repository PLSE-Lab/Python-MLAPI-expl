#!/usr/bin/env python
# coding: utf-8

# # Kaggle inclass competition to predict DOTA2 match results
# https://www.kaggle.com/c/mlcourse-dota2-win-prediction

# In[ ]:


import pandas as pd 


# # Take the public solution with LGBM, public score 0.83919
# https://www.kaggle.com/artgor/dota-eda-fe-and-models

# In[ ]:


import os
print(os.listdir("../input/dota2-solutions"))
submission_best = pd.read_csv("../input/dota2-solutions/submission_best.csv") # I have just renamed csv file
submission_best.head()


# # Take the another public solution, different features + LGBM, public score 0.83564
# https://www.kaggle.com/clair14/gold-is-the-reason-teams-and-bayes-for-lightgbm

# In[ ]:


submission_gold = pd.read_csv("../input/dota2-solutions/submission_gold.csv") # I have just renamed csv file
submission_gold.head()


# # Take the 3rd best public solution, different features + Neural net , public score 0.83054
# https://www.kaggle.com/shokhan/neural-network-to-predict-dota-2-winner

# In[ ]:


submission_nn = pd.read_csv("../input/dota2-solutions/submission_2019-04-03_19-04-57.csv") # I have just renamed csv file
submission_nn.head()


# # For all 3 solutions I renamed csv and make a simple weighted average of final predictions with 4:3:3 proportion.
# That give 0.84432 on public leaderboard and also advancement on private, basically for free!
# You did not even need to run the kernel, just download the csv.

# In[ ]:


submission = pd.DataFrame()
submission['match_id_hash'] = submission_best['match_id_hash']
submission['radiant_win_prob'] =  0.4*submission_best['radiant_win_prob']+                                   0.3*submission_gold['radiant_win_prob']+                                   0.3*submission_nn['radiant_win_prob']

submission.to_csv('submission.csv',index=False)
submission.head()


# # That's it. Enjoy and please vote up.

# In[ ]:




