#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os


# In[ ]:


data=pd.read_csv(os.path.join('/kaggle/input/tweet-sentiment-extraction', 'test.csv'))


# In[ ]:


data.head()


# In[ ]:


data=data.fillna(' ')


# In[ ]:


data['tokens'] = data['text'].apply(lambda x: x.split())
data


# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[ ]:


data['tokens'] = data['tokens'].apply(lambda x: [i for i in x if i not in stop])


# In[ ]:


import nltk
data['pos_tags']= data['tokens'].apply(lambda x: nltk.tag.pos_tag([i.lower() for i in x]))
data.head()


# In[ ]:


data['cleaned_tags'] = data['pos_tags'].apply(lambda x: [word for word,tag in x if tag != 'NNP' and tag != 'NNPS'])


# In[ ]:


data['selected_text'] = data['cleaned_tags'].apply(lambda x: ' '.join(x))


# In[ ]:


data[['textID','selected_text']].to_csv('/kaggle/working/submission.csv',index=False)


# In[ ]:





# In[ ]:




