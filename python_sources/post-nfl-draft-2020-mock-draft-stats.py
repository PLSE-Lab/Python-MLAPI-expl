#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
mdstats = pd.read_excel('/kaggle/input/mock-draft-post-draft-stats/MockDraftPostDraftStats.xlsx')
cm = sns.diverging_palette(125, 20, n=7, as_cmap=True)
mdstats_colored = mdstats.style.background_gradient(cmap=cm)


# # Post NFL Draft 2020 Mock Draft Stats
# 
# A day before the 2020 NFL Draft, I scrapped 47 Mock Drafts for the 1st Round only. The dataset of those drafts can be found here: [Mock Drafts 2020](https://www.kaggle.com/sherkt1/mock-drafts-2020). I then provided descriptive statistics of mock draft position taken for every player that appeared in any of those mock drafts which can be found here: [stats](https://www.kaggle.com/sherkt1/mock-draft-post-draft-stats). This kernel condenses the stats from these datasets and provides a quick analysis. I might continue to update if I have the time.

# # Explanation
# 
# * Count: Number of Appearances in mock drafts
# * Mean, StdDev, Max, Min: of mock draft draft positions
# * Actual: actual pick position
# * 1st Round Mocked: percentage of mock drafts that had that player mocked in the first round
# * Within 1 Std. Deviation: if actual pick was within one standard deviation of the mean
# * Difference Between Mean and Actual: of mean pick postion and actual pick position
# * Damon Arnette didn't appear in any mock draft so I set his mock draft mean pick as 33 which is outside of the first round.

# In[ ]:


mdstats_colored


# ##### Quick Analysis
# 
# * The first 13 picks were consistent with what most mocks had in terms of player mix - only two players pick position fell outside of one standard deviation (Andrew Thomas and Tristan Wirfs).
# * Of the first 13 picks Tua Tagovailoa and Justin Herbert were the most controversial - both had the greatest standard deviation. There are many explanations for this.
# * Damon Arnette was the biggest reach (+14) followed by Andrew Thomas (7.02), Jalen Reagor(6.4), and Austin Jackson (6.16).
# * Tristan Wirfs (-5.85) fell the most followed by Ceedee Lamb (-4.53) and Patrick Queen (-3.34). I am not listing Noah Igbinoghene here because he only appeared in one mock draft and I am hesitant to list Jeff Gladney because he only appeared 20 times. Technically Jeff Gladney fell the most (-6.5) outside of Noah.
# * Damon Arnette (0), Noah Igbinoghene (1), Clyde Edwards-Helaire (3), and Jordyn Brooks (4) were the biggest surprises - they appeared in the fewest mock drafts.
# 
