#!/usr/bin/env python
# coding: utf-8

# ## kernel 4: Feature Exploration and Word Clouds
# #### Can we find any useful features in the violation description field?
# 
# 
# 
# As a first step, we're going to import some useful tools and load the data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
from wordcloud import WordCloud as wc
import matplotlib.pyplot as plt
import os


# First we will pull in the violations data

# In[ ]:


violations = pd.read_csv('../input/violations.csv')


# Let's take a look at this dataset. We will focus our exploration on the violation_description field

# In[ ]:


violations.head()


# First we concatonate all of the violation descriptions into one long string. 

# > > 

# In[ ]:


viol_desc = violations.violation_description
viol_str = viol_desc.str.cat(sep = ' ')


# We then tokenize this string to extract each word in the violations. Tokenization takes the long string and breaks it on every space into multiple words. We additionally will remove stop words. Stop words are common words in the language you are using that are unlikely to help in your feature engineering. Some examples are "the", "a", "and".

# In[ ]:


stop = set(stopwords.words('english'))
list_of_words = [i.lower() for i in wordpunct_tokenize(viol_str) if i.lower() not in stop and i.isalpha()]
list_of_words[:15]


# To explore the data more we will first see what are the most common words in our corpus.

# In[ ]:


wordfreqdist = nltk.FreqDist(list_of_words)
mostcommon = wordfreqdist.most_common(30)
print(mostcommon)


# We next plot a histogram of these words to get a visual representation of the frequency.

# In[ ]:


plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()


# Now let's have some fun with the visualization! 
# 
# We will show these words as a word cloud that emphasizes the most common words in our datset.

# In[ ]:


wc1 = wc().generate(' '.join(list_of_words))
 
plt.imshow(wc1)
plt.axis("off")
plt.show()


# Cool visualization but how can this give us insight into which inspections lead to passes and failures? 
# 
# To determine this we will need to compare word clouds from the passed and the failed inspections.
# 
# Let's first pull in the inspections dataset and join it to the descriptions dataset.

# In[ ]:


inspections = pd.read_csv('../input/inspections_train.csv')


# In[ ]:


inspections.head()


# In[ ]:


combinedDF = pd.merge(violations,inspections[['camis','passed']], right_on = 'camis', left_on = 'camis')


# In[ ]:


combinedDF.head()


# Now that we have a combined DataFrame let's build out our tokens for passed and failed descriptions separately.

# In[ ]:


passedDF = combinedDF[combinedDF['passed'] == 1]
failedDF = combinedDF[combinedDF['passed'] == 0]


# In[ ]:


viol_desc_passed = passedDF.violation_description
viol_desc_failed = failedDF.violation_description


# In[ ]:


viol_str_passed = viol_desc_passed.str.cat(sep = ' ')
viol_str_failed = viol_desc_failed.str.cat(sep = ' ')


list_of_words_passed = [i.lower() for i in wordpunct_tokenize(viol_str_passed) if i.lower() not in stop and i.isalpha()]
list_of_words_failed = [i.lower() for i in wordpunct_tokenize(viol_str_failed) if i.lower() not in stop and i.isalpha()]


# Now let's see the most common words in the passed corpus

# In[ ]:


wordfreqdistpassed = nltk.FreqDist(list_of_words_passed)
mostcommonpassed = wordfreqdistpassed.most_common(30)
print(mostcommonpassed)


# In[ ]:


plt.barh(range(len(mostcommonpassed)),[val[1] for val in mostcommonpassed], align='center')
plt.yticks(range(len(mostcommonpassed)), [val[0] for val in mostcommonpassed])
plt.show()


# Let's now do the same for the failed corpus.

# In[ ]:


wordfreqdistfailed = nltk.FreqDist(list_of_words_failed)
mostcommonfailed = wordfreqdistfailed.most_common(30)
print(mostcommonfailed)


# In[ ]:


plt.barh(range(len(mostcommonfailed)),[val[1] for val in mostcommonfailed], align='center')
plt.yticks(range(len(mostcommonfailed)), [val[0] for val in mostcommonfailed])
plt.show()


# Now we generate wordclouds for both of these to see if we can detect differences.

# In[ ]:


wcpassed = wc().generate(' '.join(list_of_words_passed))
 
plt.imshow(wcpassed)
plt.axis("off")
plt.show()


# In[ ]:


wcfailed = wc().generate(' '.join(list_of_words_failed))
 
plt.imshow(wcfailed)
plt.axis("off")
plt.show()
 


# Looking at the charts and wordclouds we do see some words stand out as occuring in failed more often than passed inspections. Flies and vermin are two of the most common and seem like they should lead to failures!
# 
# How can we make these differences more obvious?
# 
# One approach is to create a wordcloud based on the relative frequency of words in the failed vs. passed inspections. A naive approach to this is to first convert the frequency of each word in the failed (or passed) set to a % of total words. As an example if vermin appears 100 out of 1,000 words we convert the frequency to 10%. We can then take the difference in % occurence between the failed and the passed sets of words. If vermin appears in 10% of failed words and 2% of passed words then we set the relative occurence to 8%. We can then build a wordcloud using this relative frequency.
# 
# Can you think of any issues with calculating relative frequencies in this manner? How can you improve upon this method?

# First we normalize the frequency for the failed words.

# In[ ]:


failedwordlen = len(list_of_words_failed)
worddictfailed = dict(wordfreqdistfailed)
worddictfailednormalized = {k: float(v) / failedwordlen for k, v in worddictfailed.items()}
worddictfailednormalized


# Then we normalize the frequency for the passed words.

# In[ ]:


passedwordlen = len(list_of_words_passed)
worddictpassed = dict(wordfreqdistpassed)
worddictpassednormalized = {k: float(v) / passedwordlen for k, v in worddictpassed.items()}
worddictpassednormalized


# Now let's create our relative frequency dictionary.

# In[ ]:



worddictrelative = {k: worddictfailednormalized[k] - worddictpassednormalized[k] 
                    for k in worddictfailednormalized if k in worddictpassednormalized}

worddictrelative


# Let's take a look at the word cloud.

# In[ ]:


wcrel = wc().generate_from_frequencies(worddictrelative)

plt.imshow(wcrel)
plt.axis("off")
plt.show()
 


# Now we see some words popping out that make sense for failures: Flies, Vermin, Mice.
# 
# It seems like the violation description field could be useful, but how can we leverage these findings for our final prediction? Are there any issues with looking at the violation descriptions as independent words? Might there be a better way to do this?
# 
# 
