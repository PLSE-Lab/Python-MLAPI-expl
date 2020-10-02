#!/usr/bin/env python
# coding: utf-8

# Hello all, im a nebie here and i'll  try to explain you what i've learned so far in an understable way.
# 
# Please upvote at the end if you like my kernel and encourage me.
# 
# ![](https://78.media.tumblr.com/0a56b418334765ec595a0982fe25aac3/tumblr_ouloa3CUT41wq17fxo3_400.gif)

# In[26]:


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


# In[27]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[28]:


train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# In[29]:


train_text_df.shape,test_text_df.shape


# In[30]:


train_variants_df.shape,test_variants_df.shape


# In[31]:


train_variants_df.head(3)


# In[32]:


train_text_df.head(3)


# In[33]:


gene_group = train_variants_df.groupby("Gene")['Gene'].count()
minimal_occ_genes = gene_group.sort_values(ascending=True)[:10]
print("Genes with maximal occurences\n", gene_group.sort_values(ascending=False)[:10])
print("\nGenes with minimal occurences\n", minimal_occ_genes)


# In[34]:


test_variants_df.head(3)


# In[35]:


test_text_df.head(3)


# In[36]:


train_text_df.Text[0]


# In[37]:


train_variants_df.Class.unique()


# In[38]:


plt.figure(figsize=(15,5))
sns.countplot(train_variants_df.Class,data = train_variants_df)


# In[39]:


print(len(train_variants_df.Gene.unique()))


# In[40]:


train_df = pd.merge(train_text_df,train_variants_df,on = 'ID')
print(train_df.shape)
train_df.head(3)


# In[41]:


test_df = pd.merge(test_text_df,test_variants_df,on = 'ID')
print(test_df.shape)
test_df.head(3)


# This is multi class classification problem and number of classes are total 9. we have to predicat the classes probabalitie  for particular Id. Now we'll see how the submission file should be.

# In[42]:


submission_file = pd.read_csv("../input/submissionFile")
submission_file.head()


# In[43]:


train_df.isnull().sum()


# In[44]:


train_df.dropna(inplace=True)


# In[45]:


from sklearn.model_selection import train_test_split

train ,test = train_test_split(train_df,test_size=0.2) 
np.random.seed(0)
train.head()


# In[46]:


X_train = train['Text'].values
X_test = test['Text'].values
y_train = train['Class'].values
y_test = test['Class'].values


# In[47]:


train.isnull().sum()


# In[48]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb


# In[49]:


svc = svm.LinearSVC()
rfc = RandomForestClassifier()
etrc = ExtraTreesClassifier()
xgbc = xgb.XGBClassifier()
lgbc = lgb.LGBMClassifier()
clf = [svc,rfc]
# ,etrc,xgbc,lgbc


# In[50]:



for i in clf:
        text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True,stop_words='english',encoding='utf-8')),('tfidf', TfidfTransformer()),('clf', i)])
        text_clf = text_clf.fit(X_train,y_train)
        y_test_predicted = text_clf.predict(X_test)
        acc = np.mean(y_test_predicted == y_test)
        print('accuracy of :',str(i),'is: ',acc )


# Note here i consider only train['Text'] values only into consideration while training but not other features like Gene.
# 
# * Here i considered word vectors (tfidf) in determining class and same used for training the model.so if i supply the same words in test so it will determine the particular class instead of considereing Gene and other features which is wrong. Though we got 60% here, this is not generalized model. what are your thoughts let me in the comments.

# Further we'll apply some NLP concepts here and get the most out of the huge text data we have. To be honest to solve this problem one should have domain knowledge in this particulat field. Even though we dont have that biology realted knowledge we'll try to implement some NLP techniques like wordembendings to get the important information.

# More to come soon so  watch this space.
# 
# **Please encourage me by upvoting**
# 
# Thank you.
# 
# ![](https://media.giphy.com/media/cyoN6pC6kek2A/giphy.gif)
