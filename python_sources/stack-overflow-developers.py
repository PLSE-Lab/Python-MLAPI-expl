#!/usr/bin/env python
# coding: utf-8

# # Trends and Beliefs among Stack Overflow Users

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Members of the Stack Overflow community vs Moderators

# In[ ]:


df = pd.read_csv("../input/survey_results_public.csv")
#limit to columns of interest
keep = ['StackOverflowFoundAnswer', 'StackOverflowCopiedCode', 'StackOverflowMetaChat',
        'StackOverflowAnswer', 'StackOverflowModeration', 'StackOverflowCommunity', 
        'StackOverflowHelpful', 'StackOverflowBetter']

dfN = pd.DataFrame(df[keep])
#convert frequency answers for first four columns to numeric values
dfN.replace({'At least once each day': 4, 'At least once each week': 3, 
             'Several times': 2, 'Once or twice': 1, 'Haven\'t done at all': 0}, 
                                        inplace = True)
dfN.pivot_table(values = 'StackOverflowCopiedCode', 
                index = 'StackOverflowCommunity', columns= 'StackOverflowModeration', 
                aggfunc = np.mean)


# # THE COPY-AND-PASTERS

# ### The people who copy/ paste Stack Overflow code most frequently strongly feel like members of the Stack Overflow Community, and strongly believe that the moderation is unfair. Those who copy/paste code the least feel the most disconnected from the stack overflow community, and also believe that the moderation is unfair.  

# In[ ]:


dfN.pivot_table(values = 'StackOverflowFoundAnswer', 
                index = 'StackOverflowCommunity', columns= 'StackOverflowModeration', 
                aggfunc = np.mean)


# # THE ANSWER-FINDERS

# ### This trend is also retained to a lesser degree among those who find answers on Stack Overflow

# In[ ]:


dfN.pivot_table(values = 'StackOverflowMetaChat', 
                index = 'StackOverflowCommunity', columns= 'StackOverflowModeration', 
                aggfunc = np.mean)


# # THE METACHATTERS

# ### We see from the table above the MetaChat is most used by those who strongly feel that they belong to the Stack Overflow community AND strongly believe that the moderation is unfair. Not surprisingly, those who don't feel as if they belong to the Stack Overflow community rarely use the MetaChat.

# In[ ]:


dfN.pivot_table(values = 'StackOverflowAnswer', 
                index = 'StackOverflowCommunity', columns= 'StackOverflowModeration', 
                aggfunc = np.mean)


# # THE ANSWER-POSTERS

# ### It appears that those who post answers on Stack Overflow the most frequently strongly feel that they belong to the community, however they are divided about how they view the moderation (either strongly agree or strongly disagree that it is unfair, with strongly agree leading by a nose).

# # Is belonging to the Stack Overflow community fulfilling?

# In[ ]:


dfN.pivot_table(values = 'StackOverflowFoundAnswer', 
                index = 'StackOverflowCommunity', columns= 'StackOverflowHelpful', 
                aggfunc = np.mean)


# ### Those groups who find their answers on stack overflow find them helpful, for the most part (see table above.  Yet, one hilarious finding stands out: the members who agree that they belong the Stack Overflow community, yet strongly disagree that the answers they find there are helpful actually find the MOST answers on Stack Overflow.  This is why it's important to always ask the same questions in many different ways on surveys :)

# In[ ]:


dfN.pivot_table(values = 'StackOverflowAnswer', 
                index = 'StackOverflowCommunity', columns= 'StackOverflowBetter', 
                aggfunc = np.mean)


# ### Those who post anwers on Stack Overflow largely feel that they belong to the community.  Interestingly, a large portion of answer-posters and community-belongers do NOT feel that Stack Overflow makes the world a better place.

# In[ ]:




