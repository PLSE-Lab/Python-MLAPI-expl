#!/usr/bin/env python
# coding: utf-8

# Question: Do men or women have more inflated market value perception? 
# Do the number of matches men or women get correlate logically to their self-ratings in various personality charactieristics?
# The assumption is that the higher a person is in the five traits (fun, attractiveness, sincerity, intelligence, and ambition), the more matches they should recieve. 
# Secondly, how do men or women with similar self-perceptions of those personality traits compare in terms of the amount of matches recieved? This is an indicator of how various traits are valued in a gender. 

# Part 1: Preparing the data
# Load data of how people self-assessed themselves (personality3_1) and match. 
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn
seaborn.set(style='ticks')
df = pd.DataFrame.from_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1",index_col=False)
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
actual_matches = []
actual_matches = [(df.iloc[i:i+10-1].sum(axis=0)["match"])for i in range(0, len(df), 10)]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
df = df.iloc[::10, :] 
df["match"] = actual_matches

low_memory = False
_genders= [0, 1]
df = pd.DataFrame({
    'Self-labeled level of sincerity': df["sinc3_1"], 
    'Number of Matches': df["match"],
    'Gender': df["gender"]
})

ax2 = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
ax2.map(pyplot.scatter, 'Self-labeled level of sincerity','Number of Matches').add_legend()
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn
seaborn.set(style='ticks')
df = pd.DataFrame.from_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1",index_col=False)
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
actual_matches = []
actual_matches = [(df.iloc[i:i+10-1].sum(axis=0)["match"])for i in range(0, len(df), 10)]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
df = df.iloc[::10, :] 
df["match"] = actual_matches

low_memory = False
_genders= [0, 1]
df = pd.DataFrame({
    'Self-labeled level of sincerity': df["sinc3_1"], 
    'Number of Matches': df["match"],
    'Gender': df["gender"]
})

ax2 = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
ax2.map(pyplot.scatter, 'Self-labeled level of sincerity','Number of Matches').add_legend()
ax2.fig.suptitle("Market Value: Sincerity")





# In[ ]:



df = pd.DataFrame.from_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1",index_col=False)
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
actual_matches = []
actual_matches = [(df.iloc[i:i+10-1].sum(axis=0)["match"])for i in range(0, len(df), 10)]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
df = df.iloc[::10, :] 
df["match"] = actual_matches

low_memory = False
_genders= [0, 1]
df = pd.DataFrame({
    'Self-percieved level of attractiveness': df["attr3_1"], 
    'Number of Matches': df["match"],
    
    'Gender': df["gender"]
})

ax2 = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
ax2.map(pyplot.scatter, 'Self-percieved level of attractiveness','Number of Matches').add_legend()
ax2.fig.suptitle("Market Value: Attractiveness')


# In[ ]:


df = pd.DataFrame.from_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1",index_col=False)
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
actual_matches = []
actual_matches = [(df.iloc[i:i+10-1].sum(axis=0)["match"])for i in range(0, len(df), 10)]
df = df[keep_cols]
#df['match']  = actual_matches
df = df.iloc[::10, :] 
df["match"] = actual_matches

_genders= [0, 1]
df = pd.DataFrame({
    'Self-percieved level of intelligence': df["intel3_1"], 
    'Number of Matches': df["match"],
    'Gender': df["gender"]
})

ax2 = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
ax2.map(pyplot.scatter, 'Self-percieved level of intelligence','Number of Matches').add_legend()
ax2.fig.suptitle("Market Value: Intelligence")


# In[ ]:



df = pd.DataFrame.from_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1",index_col=False)
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
actual_matches = []
actual_matches = [(df.iloc[i:i+10-1].sum(axis=0)["match"])for i in range(0, len(df), 10)]
df = df[keep_cols]
#df['match']  = actual_matches
df = df.iloc[::10, :] 
df["match"] = actual_matches

_genders= [0, 1]
df = pd.DataFrame({
    'Self-percieved level of fun': df["fun3_1"], 
    'Number of Matches': df["match"],
    'Gender': df["gender"]
})

ax2 = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
ax2.map(pyplot.scatter, 'Self-percieved level of fun','Number of Matches').add_legend()
ax2.fig.suptitle("Market Value: Fun")


# In[ ]:



df = pd.DataFrame.from_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1",index_col=False)
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
actual_matches = []
actual_matches = [(df.iloc[i:i+10-1].sum(axis=0)["match"])for i in range(0, len(df), 10)]
df = df[keep_cols]
#df['match']  = actual_matches
df = df.iloc[::10, :] 
df["match"] = actual_matches

_genders= [0, 1]
df = pd.DataFrame({
    'Self-percieved level of ambition': df["amb3_1"], 
    'Number of Matches': df["match"],
    'Gender': df["gender"]
})

ax2 = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
ax2.map(pyplot.scatter, 'Self-percieved level of ambition','Number of Matches').add_legend()
ax2.fig.suptitle("Market Value: Ambition")


# Conclusion: Thus, in all departments, the trend generally is as expected for both genders. Except for sincerity, the higher both genders rated themselves in personality traits, the more matches they actually got. For attractiveness, there was a bell curve-like trend, with the highest matches occurring at a self-labeling of 7. For intelligence, for both genders, there is strong correlation between intelligence and number of matches. In ambition, for men, ambition seemed not to matter as men of all values of self-labeling had a fairly consistent number of matches. For women, however, ambition is a key factor, as seen by the semi-linear relationship between intelligence and matches. For fun, there is a semi-linear correlation for both genders. In all of these graphs, women have higher matches then men (given similar self-labelings of personality). Thus, this could mean that either women label themselves as lower then they actually are (disinflated market value), or men label themselves as higher (inflated market value). It could also be the fact that women are more selective than men, making matches for men harder than for women. 
# Constraints: The group selected for this dataset are very niche, so this experiment is only reflective of young professionals. 

# So let's explore just a little bit more, to see if women truly rank themselves lower than men (and yet get more matches then men). We find the 95% confidence interval for the self-labeling for each of the personality traits. 

# In[ ]:


import scipy  
import numpy as np, scipy.stats as st
import math
from numpy import sqrt, abs, round
from scipy.stats import norm
df = pd.DataFrame.from_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1",index_col=False)
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
print("Women on Intelligence")
df_w = df[df["gender"] == 0]["intel3_1"]
print(df_w.mean())
print("Men on Intelligence")
df_m = df[df["gender"] == 1]["intel3_1"]
print(df_m.mean())
print("Women on Ambition")
df_w = df[df["gender"] == 0]["amb3_1"]
print(df_w.mean())

print("Men on Ambition")
df_m = df[df["gender"] == 1]["amb3_1"]
print(df_m.mean())
df_m = df[df["gender"]==1]


# Thus it is not hte case that there is inflation or deflation to blame for this disparity in the number of matches between men or women, but the fact that women are generally pickier than men when choosing mates. 

# In[ ]:




