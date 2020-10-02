#!/usr/bin/env python
# coding: utf-8

# ## 2-D PCA analysis on BERT-generated word embedding vector

# ### In this Kernel, I picked up 5 words from News Headline Dataset for sarcasm and obtain BERT-generated word embedding vector. Then, run Principle Component Analysis in 2-D and plot for visualization and realization purpose.

# ### I have already obtained Word Embedding Vector using BERT Base and add to workspace

# In[ ]:


import pandas as pd
df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)
df.head()


# In[ ]:


file2 = open('../input/bert-embeddingvector/output.txt', "r")
temp = file2.read()
file2.close()


# In[ ]:


import ast 

temp2 = temp.split("\n")
ls1 = []
for t in range(0,100):
    temp3 = ast.literal_eval(temp2[t])
    ls1.append(temp3)


# #### Words picked up are: teacher, artist, author, holidays, circus
# We can look up the sentences from whcih I picked up these words.

# In[ ]:


artist = ls1[20]['features'][3]['layers'][0]['values']
print(df['headline'][20])


# In[ ]:


authors = ls1[63]['features'][-2]['layers'][0]['values'] 
circus =  ls1[23]['features'][-2]['layers'][0]['values']
teachers = ls1[80]['features'][3]['layers'][0]['values']
holiday = ls1[44]['features'][-3]['layers'][0]['values']


# ### Run PCA analysis on these 5 words (artist, authors, circus, teachers, and holiday)

# In[ ]:


d = {'authors':authors, 'artist':artist, 'circus':circus,'teachers':teachers,'holiday':holiday}
x = pd.DataFrame(data=d)
x = x.transpose()


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


# In[ ]:


adjustDf = pd.concat([principalDf,pd.DataFrame(['authors','artist','circus','teachers','holiday'])],axis=1)
adjustDf.columns = ['x', 'y', 'group']

print(adjustDf)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


# In[ ]:


# basic plot
p1=sns.regplot(data=adjustDf, x="x", y="y", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':400})

# add annotations one by one with a loop
for line in range(0,adjustDf.shape[0]):
     p1.text(adjustDf.x[line]+0.2, adjustDf.y[line], adjustDf.group[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
plt.title("PCA analysis on word embedding vector from BERT")
plt.show()


# #### Conclusion:
# - artist, authors and teachers have very similar 2-D vector. I assume this is because they are contextually similar words in human language.
# - Among all the words pair, artist and authors are the smallest distance pair. These interpretation seems helpful for any NLP algorithm.
# - Using pre-trained BERT without fine-tuning with your dataset to get word embedding vector would bring us further improvement for solving NLP tasks. 
