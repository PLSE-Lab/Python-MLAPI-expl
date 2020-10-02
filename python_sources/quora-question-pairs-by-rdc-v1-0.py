# encoding = 'UTF-8'

#load packages
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from wordcloud import WordCloud

#load data
train =pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_qs =pd.Series(train['question1'].tolist()+train['question2'].tolist()).astype(str)
test_qs =pd.Series(test['question1'].tolist()+test['question2'].tolist()).astype(str)

combine_qs = pd.Series(train_qs.tolist()+test_qs.tolist())

#EDA

##question len distinct

train_len = train_qs.apply(len)
test_len =test_qs.apply(len)

plt.hist(train_len,bins=200,range=[0,200],color='red',normed=True,label='train',alpha=0.5)
plt.hist(test_len,bins=200,range=[0,200],color='green',normed=True,label='test',alpha=0.5)
plt.legend()

#wordcloud
cloud = WordCloud(width=1440,height=1080).generate("".join(train_qs.astype(str)))
plt.imshow(cloud)