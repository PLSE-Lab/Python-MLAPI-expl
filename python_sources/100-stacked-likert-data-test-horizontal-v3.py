#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.figure(figsize=(5,25), dpi=(300))

#read the data
df=pd.read_csv("../input/sample-likert-recruit.csv")

#change the order so question with most agree is at the top
df = df.sort_values(by=['l_sa'])
 
#populate the variables from the csv
questions = df.question
strongdisagree = df.l_sd
disagree = df.l_d
neutral = df.l_n
agree = df.l_a
strongagree = df.l_sa

ind = [x for x, _ in enumerate(questions)]

#calculate the percentages for the 100% stacked bars
total = strongdisagree+disagree+neutral+agree+strongagree
proportion_strongdisagree = np.true_divide(strongdisagree, total) * 100
proportion_disagree = np.true_divide(disagree, total) * 100
proportion_neutral = np.true_divide(neutral, total) * 100
proportion_agree = np.true_divide(agree, total) * 100
proportion_strongagree = np.true_divide(strongagree, total) * 100

plt.subplots_adjust(right=4)

#plot the bars
plt.barh(ind, proportion_strongagree, label='SA', color='#1b617b', left=proportion_strongdisagree+proportion_disagree+proportion_neutral+proportion_agree)
plt.barh(ind, proportion_agree, label='A', color='#879caf', left=proportion_strongdisagree+proportion_disagree+proportion_neutral)
plt.barh(ind, proportion_neutral, label='N', color='#e7e7e7', left=proportion_strongdisagree+proportion_disagree)
plt.barh(ind, proportion_disagree, label='D', color='#e28e8e', left=proportion_strongdisagree)
plt.barh(ind, proportion_strongdisagree, label='SD', color='#c71d1d') 

#set the axes
plt.yticks(ind, questions)
#plt.ylabel("Questions")
#plt.xlabel("Responses")
#plt.title("Survey Responses")
plt.xlim=1.0

#fine tune the labels
ax=plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.grid(color='black', linestyle='-', axis="x", linewidth=1)
ax.set_facecolor('white')
plt.tick_params(labelsize=24)

plt.show()

cols=['l_sa','l_a','l_n','l_d','l_sd']
df[cols] = df[cols].div(df[cols].sum(axis=1), axis=0).multiply(100)

df[cols] = df[cols].round(2)
#df = df.iloc[:, :-1]
#df = df.sort_values(by=['l_sa'], ascending=False)

df


# In[ ]:




