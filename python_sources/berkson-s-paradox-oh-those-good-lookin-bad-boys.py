#!/usr/bin/env python
# coding: utf-8

# # Berkson's Paradox
# 
# Have you ever had a friend complain that all the best looking folks on the dating scene _always_ turn out to be absolute jerks? Well it could be that facial symmetry traits are carried on the disagreeability chromosome or perhaps the more aesthetically gifted are conditioned by society to take advantage of their looks. _Maybe!_ But it's much more likely this is an example of [Berkson's paradox](https://en.wikipedia.org/wiki/Berkson%27s_paradox) or perhaps more aptly Berkson's fallacy.
# 
# 
# ### Berkson swipes right
# Berkson's paradoxes are counterintuitive and spurious correlations that occur between traits caused by sampling bias. Consider the relationship between attractiveness and personality in your friend's dating experience. It's easy to imagine that your bud may only swipe right when they see someone who is either very good looking or who has really warm and friendly profile text. To put it another way, perhaps your friend is subconsciously filtering on a minimum hotness rating of, say, 6 for either personality or looks. Potential dates with neither are immediately passed over and are never met in person. This would happen without even noticing that biased sampling is going on (especially on swipy-swipy apps!) and the net result is an apparent but false relationship between attractiveness and personality!

# In[ ]:


import numpy as np
import pandas as pd

from numpy.random import multivariate_normal, seed

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

seed(42)
df = pd.DataFrame( np.round(multivariate_normal([5,5],[[6,0],[0,6]],100),1, ),
                 columns = ['attractiveness', 'personality'])
df = df.applymap(lambda x: 0 if x < 1 else x)
df = df.applymap(lambda x: 10 if x > 10 else x)

fig, axarr = plt.subplots(1, 2, figsize=(16, 4))
plt.subplot(1,2,1)
_ = sns.regplot(x = "attractiveness", y = "personality", data=df)
plt.title('All Potential Dates')
plt.xlim([-0.5,10.5])
plt.ylim([-0.5,10.5])
plt.subplot(1,2,2)
sns.regplot(x = "attractiveness",
                y = "personality", 
                data=df[np.logical_or(df.attractiveness > 6, df.personality > 6)])
plt.title('At least a 6')
plt.xlim([-0.5,10.5])
plt.ylim([-0.5,10.5])
_ = plt.fill_between(x=[-0.5,6], y1=[6,6], y2=[-0.5,-0.5], hatch='/', edgecolor="r", facecolor="none")


# ### Holywood ruins books
# 
# YouTuber [Numberphile](https://www.youtube.com/watch?v=FUD8h9JpEVQ) offers another fun example. Why, she wondered, does Hollywood always ruin good books. It's just so easy to think of absolutely incredible books that were made into truly awful films. Hitchhiker's Guide to the Galaxy. The Golden Compass. Oh heavens, Alice in Wonderland (why Johnnie, why?). It _could be_ that Hollywood producers are out there ruining books out of spite, having not received enough love as children, but it's far more likely that this is also an example of Berkson's Paradox. Books that are sufficiently bad are never made into movies at all. And movies that are testing terribly get scrapped or aren't widely released. The result is an entire section of the sample space that you haven't seen and (mercifully) never will! 

# In[ ]:


seed(101)
df = pd.DataFrame( np.round(multivariate_normal([5,5],[[6,0],[0,6]],100),1, ),
                 columns = ['book', 'movie'])
df = df.applymap(lambda x: 0 if x < 1 else x)
df = df.applymap(lambda x: 10 if x > 10 else x)

fig, axarr = plt.subplots(1, 2, figsize=(16, 4))
plt.subplot(1,2,1)
_ = sns.regplot(x = "book", y = "movie", data=df, color='indigo')
plt.title('All Movies')
plt.xlim([-0.5,10.5])
plt.ylim([-0.5,10.5])
plt.subplot(1,2,2)
sns.regplot(x = "book",
            y = "movie", 
            data=df[df.book + df.movie > 8],
            color='indigo')
plt.title('Constant Minimum Quality')
plt.xlim([-0.5,10.5])
plt.ylim([-0.5,10.5])
_ = plt.fill_between(x=[-0.5,8], y1=[8,0], y2=[-0.5,-0.5], hatch='/', edgecolor="r", facecolor="none")


# ### Don't get your gall bladder removed to cure your diabetes
# 
# While these examples are a bit silly, this effect has been observed in real published studies. Berkson gave his name to the fallacy when wrote about the prevelant "impression" amongst medical practitioners that there existed a relationship between cholecystitis (gall bladder inflammation) and diabetes in the [International Journal of Epidemiology](https://academic.oup.com/ije/article/43/2/511/680126). According to Berkson, "in certain medical circles, the gall bladder was being removed as a treatment for diabetes". To drive home his point Berkson used as his control group patients presenting at the clinic with strong refractive errors in need of glasses, "a diagnosis which cannot reasonably be thought to be correlated with cholecystitis". Sure enough, the prevalence of cholecystitis was far higher amongst diabetics than those turning up to get their eyes checked.
# 
# Why is it that we would see this relationship occur in the hospital but not in the general population? The reason is that the ratio of multiple diagnoses to single diagnoses in the hospital will always be greater than in the general population, since someone with both cholecystitis and diabetes has two conditions that may cause them to need treatment.

# ##### Hospital Population

# In[ ]:


df = pd.DataFrame({'cholecystitis': [28, 68],
                    'not cholecystitis': [548, 2606]},
                   index=['diabetes', 'needs glasses'])
df['prevalence %'] = round(100*df['cholecystitis'] / (df['cholecystitis'] + df['not cholecystitis']),1)
df


# ##### General Population

# In[ ]:


df = pd.DataFrame({'cholecystitis': [3000 ,  29700],
                    'not cholecystitis': [97000 , 960300 ]},
                   index=['diabetes', 'needs glasses'])
df['prevalence %'] = round(100*df['cholecystitis'] / (df['cholecystitis'] + df['not cholecystitis']),1)
df


# ### Don't let Berkson's fallacy get you too!
# 
# There you have it folks, Berkson's Paradox in a nutshell! Don't forget to look out for sampling bias in your own studies or you too may succumb to this crinkly little fallacy. Remember that any time you are conducting observational research (e.g. without random allocation) you are at risk. As a data scientist you may not even be in control of data collection and experimental design, and so it is doubly important that you are very clear as to how the data was collected and how samples were selected. When in doubt ask lots of questions!  
# 
# Know of any other interesting examples of Berkson's Paradox turning up in real data? I'd love to hear about them!
# 
# ### fin

# In[ ]:




