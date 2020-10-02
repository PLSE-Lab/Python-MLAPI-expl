#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../input/Sentiment.csv')


# In[ ]:


dt_df = df[df['candidate'] == 'Donald Trump']
sentiment_by_candidate = dt_df.groupby('sentiment').size()
print(sentiment_by_candidate)


# In[ ]:


trump_sent = np.array(sentiment_by_candidate)
total = sum(trump_sent[0:])
print(total)


# In[ ]:


p = np.asarray((trump_sent/total) * 100)
percent = p.round()
print(percent)


# In[ ]:


from matplotlib import style
plt.style.use('fivethirtyeight')
labels = 'Negative', 'Neutral', 'Positive'
explode = (0, 0, 0.05)
colors = ['lightgreen', 'lightskyblue', 'lightcoral']
plt.pie(percent, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',shadow=True, startangle=100)
plt.axis('equal')
plt.title('Donald Trump tweet Sentiment')
plt.show()

