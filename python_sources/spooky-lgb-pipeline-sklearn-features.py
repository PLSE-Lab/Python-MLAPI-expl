#!/usr/bin/env python
# coding: utf-8

# This is just a brief dive, some visual fun, and then a quick model. 

# In[15]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from nltk.corpus import stopwords
from collections import Counter

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import cv2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[16]:


trainDf = pd.read_csv('../input/spooky-author-identification/train.csv', index_col = 0)
trainDf.head()


# In[19]:


def textClean(text):
    text=text.lower().split()
    stops = {'so', 'his', 't', 'y', 'ours', 'herself', 
         'your', 'all', 'some', 'they', 'i', 'of', 'didn', 
         'them', 'when', 'will', 'that', 'its', 'because', 
         'while', 'those', 'my', 'don', 'again', 'her', 'if',
         'further', 'now', 'does', 'against', 'won', 'same', 
         'a', 'during', 'who', 'here', 'have', 'in', 'being', 
         'it', 'other', 'once', 'itself', 'hers', 'after', 're',
         'just', 'their', 'himself', 'theirs', 'whom', 'then', 'd', 
         'out', 'm', 'mustn', 'where', 'below', 'about', 'isn',
         'shouldn', 'wouldn', 'these', 'me', 'to', 'doesn', 'into',
         'the', 'until', 'she', 'am', 'under', 'how', 'yourself',
         'couldn', 'ma', 'up', 'than', 'from', 'themselves', 'yourselves',
         'off', 'above', 'yours', 'having', 'mightn', 'needn', 'on', 
         'too', 'there', 'an', 'and', 'down', 'ourselves', 'each',
         'hadn', 'ain', 'such', 've', 'did', 'be', 'or', 'aren', 'he', 
         'should', 'for', 'both', 'doing', 'this', 'through', 'do', 'had',
         'own', 'but', 'were', 'over', 'not', 'are', 'few', 'by', 
         'been', 'most', 'no', 'as', 'was', 'what', 's', 'is', 'you', 
         'shan', 'between', 'wasn', 'has', 'more', 'him', 'nor',
         'can', 'why', 'any', 'at', 'myself', 'very', 'with', 'we', 
         'which', 'hasn', 'weren', 'haven', 'our', 'll', 'only',
         'o', 'before'}
#     stops = set(stopwords.words("English"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


# In[20]:


texts = []
for t in trainDf.text:
    texts.append(textClean(t))


# In[21]:


texts[0][:100]


# In[22]:


tops = Counter(str(texts).split()).most_common()[:30]
labs, vals = zip(*tops)
idx = np.arange(len(labs))
wid=0.6
fig, ax=plt.subplots(1,1,figsize=(14,8))
ax=plt.bar(idx, vals, wid, color='orange')
ax=plt.xticks(idx - wid/8, labs, rotation=30, size=14)
plt.title('Top Thirty Counts of Most-Common Words Among Text')


# Once __UPON__ a midnight dreary... This is great. The words dont seem all that specific though.

# In[23]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x="author", data=trainDf, 
#                    color='purple',
                   palette="Greys")
plt.ylabel('Frequency'); plt.xlabel('Author')
plt.title('Freq. of Authors')
plt.show()


# A little more __POE__ than __Shelley__ and __Lovecraft__.

# In[26]:


trainDf['text'].to_csv('wc_text.txt')
img = cv2.imread("..")
spookyColors = 255.0-cv2.imread("../input/pumpkin-pic/hp.jpg")


# In[27]:


wc_text = open('wc_text.txt').read()
wordcloud = WordCloud(background_color="black",
                      width=1200,height=800, mask=spookyColors).generate(wc_text)
image_colors = ImageColorGenerator(spookyColors)


# In[28]:


fig, ax=plt.subplots(1,1,figsize=(16,16))
ax.grid(False);
ax.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");


# [](http://)LOL....sweeeet. Its a pumpkin.

# ### And now for some modeling...

# In[30]:


trainDf = pd.read_csv('../input/spooky-author-identification/train.csv', index_col = 0)
trainDf.head()


# In[31]:


lbl = preprocessing.LabelEncoder()
y = lbl.fit_transform(trainDf.author)


# In[33]:


testDf = pd.read_csv('../input/spooky-author-identification/test.csv',index_col=0)
pid = testDf.index


# In[34]:


texts = []
for t in trainDf.text:
    texts.append(textClean(t))
testText = []
for t in testDf.text:
    testText.append(textClean(t))


# In[35]:


testText[0][:100]


# In[36]:


trainDf['newText'] = texts
trainDf.drop(['author','text'],axis=1,inplace=True)
trainDf.head()


# In[37]:


testDf['newText'] = testText
testDf.drop(['text'],axis=1,inplace=True)
testDf.head()


# In[38]:


ax = sns.countplot(x=y, color='black')
## Quick Confirmation


# ### Build some features...

# In[39]:


cvec = CountVectorizer(analyzer=u'char', ngram_range=(1, 8), max_features=1000,
                       strip_accents='unicode', stop_words='english',
                       token_pattern=r'\w+')

tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, 
                        strip_accents='unicode',
                        lowercase =True, analyzer='word', token_pattern=r'\w+',
                        use_idf=True, smooth_idf=True, sublinear_tf=True, 
                        stop_words = 'english')

nmfNC = 50
nmf = NMF(n_components=nmfNC, random_state=42,
          alpha=.1, l1_ratio=.5)
ldaNT = 50
lda = LatentDirichletAllocation(n_topics=ldaNT, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=42)

textNC = 150
tsvdText = TruncatedSVD(n_components=textNC, n_iter=25, random_state=42)
tsvdCount = TruncatedSVD(n_components=textNC, n_iter=25, random_state=42)


# In[40]:


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        print('fit...')
        return self
    def transform(self, x):
        print('transform...')
        return x[self.key].apply(str)
    
class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        print('fit...')
        return self
    def transform(self, x):
        print('transform and drop...')
        x = x.drop(['newText'],axis=1).values
        return x
    
print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
#            n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pip1', pipeline.Pipeline([('newText', cust_txt_col('newText')),('counts', cvec),('tsvdCountText', tsvdCount)])),       
            ('pip2', pipeline.Pipeline([('nmf_Text', cust_txt_col('newText')),('tfidf_Text', tfidf),('nmfText', nmf)])),
            ('pip3', pipeline.Pipeline([('lda_Text', cust_txt_col('newText')),('tfidf_Text', tfidf),('ldaText', lda)])),
            ('pip4', pipeline.Pipeline([('newText', cust_txt_col('newText')),('tfidf_Text', tfidf),('tsvdText', tsvdText)]))
        ])
    )])


# In[41]:


for c in trainDf.columns:
    if c == 'newText':
        trainDf[c+'_len'] = trainDf[c].map(lambda x: len(str(x)))
        trainDf[c+'_words'] = trainDf[c].map(lambda x: len(str(x).split(' ')))        
        
for c in testDf.columns:
    if c == 'newText':
        testDf[c+'_len'] = testDf[c].map(lambda x: len(str(x)))
        testDf[c+'_words'] = testDf[c].map(lambda x: len(str(x).split(' ')))     


# In[42]:


trainDf.head()


# In[43]:


train = fp.fit_transform(trainDf); print(train.shape)
test = fp.transform(testDf); print(test.shape)


# This LGB model is taking quite a long time to run in this notebook and Im not sure why.

# In[49]:


params = {'learning_rate':0.05
         ,'max_depth':4
         ,'objective':'multiclass'
         ,'num_class':3
         ,'metric':{'multi_logloss'}
#          ,'num_iterations':256
         ,'num_leaves':128
         ,'min_data_in_leaf':128
         ,'bagging_fraction':0.85 
         ,'feature_fraction':0.85 
         ,'lambda_l1':1.0}


# Edited this for just 1 fold because it was taking so long. 

# In[51]:


fold=1
preds=0
for i in range(fold):
    xt,xv,yt,yv = train_test_split(train, y, test_size=0.15, random_state=i*314)
    dtx = lgb.Dataset(xt, label=yt)
    dtv = lgb.Dataset(xv, label=yv)
    model = lgb.train(params, train_set=dtx, valid_sets=dtv, valid_names=['val'],
                      num_boost_round=1000,
                      early_stopping_rounds=100,
                      verbose_eval=False)
    pred = model.predict(test)
    preds += pred
preds /= fold    


# In[52]:


ax = lgb.plot_importance(model, max_num_features=50, figsize=(12,8), **{'color':'orange'})
ax.grid(False)
ax.set_facecolor('black')


# In[53]:


import datetime
now = datetime.datetime.now()
submission = pd.DataFrame(preds, columns=['EAP','HPL','MWS'])
submission['ID'] = pid
submission.to_csv('lgbPipeline_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv', index=False)


# In[ ]:




