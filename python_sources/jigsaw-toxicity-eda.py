#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Jigsaw Toxicity - EDA</font></center></h1>
# 
# <br>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data for analysis</a>  
# - <a href='#3'>Data exploration</a>   
#     - <a href='#31'>Target feature</a>   
#     - <a href='#32'>Identity attributes</a>   
#     - <a href='#33'>Time Series Analysis</a>   
#     - <a href='#34'>Comment length analysis</a>   
#     - <a href='#35'>Word clouds</a>  
# - <a href='#4'>References</a>   

# # <a id='1'>Introduction</a>  
# 
# ## Competition objective
# 
# The competition objective is to build models that detect toxicity and reduce unwanted bias. 
# 
# ## Background
# 
# At the end of 2017 the Civil Comments platform shut down and chose make their ~2m public comments from their platform available in a lasting open archive so that researchers could understand and improve civility in online conversations for years to come. Jigsaw sponsored this effort and extended annotation of this data by human raters for various toxic conversational attributes.

# # <a id='2'>Prepare the data for analysis</a>  
# 
# ## Load packages

# In[ ]:


import gc
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from wordcloud import WordCloud, STOPWORDS
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

pyLDAvis.enable_notebook()
np.random.seed(2018)
warnings.filterwarnings('ignore')


# ## Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'JIGSAW_PATH = "../input/"\ntrain = pd.read_csv(os.path.join(JIGSAW_PATH,\'train.csv\'), index_col=\'id\')\ntest = pd.read_csv(os.path.join(JIGSAW_PATH,\'test.csv\'), index_col=\'id\')')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print("Train and test shape: {} {}".format(train.shape, test.shape))


# In[ ]:


train.info()


# ## Style
# 
# We insert a small *style* cell to try to correct the style applied by the topic modelling visualization using pyLDAVis. 

# In[ ]:


get_ipython().run_cell_magic('html', '', '<style>\n.container { width:1000px !important; }\n</style>')


# # <a id='3'>Data exploration</a>  
# 
# The comments are stored in `train` and `test` in `comment_text` column.  
# Additionally, in `train` we have identity attributes, representing the identities that are mentioned in the comment.
# * **race or ethnicity**: asian, black, jewish, latino, other_race_or_ethnicity, white  
# * **gender**: female, male, transgender, other_gender  
# * **sexual orientation**: bisexual, heterosexual, homosexual_gay_or_lesbian, other_sexual_orientation  
# * **religion**: atheist,buddhist,  christian, hindu, muslim, other_religion  
# * **disability**: intellectual_or_learning_disability, other_disability, physical_disability, psychiatric_or_mental_illness  
# 
# Toxicity subtype attributes:
# * severe_toxicity
# * obscene
# * threat
# * insult
# * identity_attack
# * sexual_explicit
# 
# We also have few article/comment identification information:
# * created_date  
# * publication_id   
# * parent_id  
# * article_id 
# 
# Several user feedback information associated with the comments are provided:
# * rating  
# * funny  
# * wow  
# * sad  
# * likes  
# * disagree  
# * sexual_explicit  
# 
# In the datasets are also 2 fields relative to annotations:
# * identity_annotator_count  
# * toxicity_annotator_count
# 
# 
# 

# ##  <a id='31'>Target feature</a>
# 
# Let's check the distribution of `target` value in the train set.

# In[ ]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,4))
sns.distplot(train.target, kde=False, bins=40).set_title("Histogram Plot of target", fontsize=15)


# In[ ]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,4))
sns.kdeplot(train.target).set_title("Kernel Density Estimate(kde) Plot of target", fontsize=15)


# The above plots tell us that most of the comments in the train data set are non-toxic.
# Also, kde is plotted to get the real shape of the data not a line plot because otherwise, we wouldn't get a smooth plot since a lot of outliers and inbetweeners would be introduced.

# Let us also plots the **kernel density estimates of toxicity(target) subtype attributes** as below.

# In[ ]:


fig, axarr = plt.subplots(nrows=3,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of toxicity subtype attributes in train data', fontsize=15)
sns.kdeplot(train['severe_toxicity'], ax=axarr[0][0])
sns.kdeplot(train['obscene'], ax=axarr[0][1])
sns.kdeplot(train['threat'], ax=axarr[1][0])
sns.kdeplot(train['insult'], ax=axarr[1][1])
sns.kdeplot(train['identity_attack'], ax=axarr[2][0])
sns.kdeplot(train['sexual_explicit'], ax=axarr[2][1])
sns.despine()


# So, the data distriution shapes of the target feature and its sub-types are similar 

# Evaluation will be done like so:
# + target >= 0.5 ==> toxic comment
# + target < 0.5 ==> non-toxic comment
# 
# Let us plot a bar chart of the count of comments labelled as toxic vs non-toxic

# In[ ]:


train['target_binarized'] = train['target'].apply(lambda x : 'Toxic' if  x >= 0.5 else 'Non-Toxic')
fig, axarr = plt.subplots(1,1,figsize=(12, 4))
train['target_binarized'].value_counts().plot.bar(fontsize=10).set_title("Toxic vs Non-Toxic Comments Count", 
                                                                         fontsize=15)
sns.despine(bottom=True,  left=True)


# In[ ]:


#train = train.drop(columns='target_binarized')


# In[ ]:


f = (
    train.loc[:, ['target', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']]
        .applymap(lambda v: float(v))
        .dropna()
)


# In[ ]:


f.head()


# In[ ]:


f_corr=f.corr()


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12,6))
sns.heatmap(f_corr, annot=True)


# One thing that really stands out is the high degree of correlation between 'target' and its sub-type 'insult'.

# ## <a id='32'>Identity attributes</a>
# 
# Let's check now the distribution of these attribute values.

# In[ ]:


fig, axarr = plt.subplots(nrows=3,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of race and ethnicity features values in the train set', fontsize=15)
sns.kdeplot(train['asian'], ax=axarr[0][0], color='mediumvioletred')
sns.kdeplot(train['black'], ax=axarr[0][1], color='mediumvioletred')
sns.kdeplot(train['jewish'], ax=axarr[1][0], color='mediumvioletred')
sns.kdeplot(train['latino'], ax=axarr[1][1], color='mediumvioletred')
sns.kdeplot(train['other_race_or_ethnicity'], ax=axarr[2][0], color='mediumvioletred')
sns.kdeplot(train['white'], ax=axarr[2][1], color='mediumvioletred')
sns.despine()


# **Distribution of ethnicity with respect to target_binarized : ** 

# In[ ]:


g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "asian", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "black", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "jewish", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "latino", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "other_race_or_ethnicity", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "white", color='mediumvioletred')


# In[ ]:


fig, axarr = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
fig.suptitle('Distribution of toxicity target for every race/ethnicity sample that has been annotated with value of 1', 
             fontsize=14)
sns.violinplot(train[train['asian'] == np.max(train.asian)]['target'], ax=axarr[0]).set_title("asian")
sns.violinplot(train[train['black'] == np.max(train.black)]['target'], ax=axarr[1]).set_title("black")
sns.despine()
fig, axarr = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
sns.violinplot(train[train['jewish'] == np.max(train.jewish)]['target'], ax=axarr[0]).set_title("jewish")
sns.violinplot(train[train['latino'] == np.max(train.latino)]['target'], ax=axarr[1]).set_title("latino")
sns.despine()
fig, axarr = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
sns.violinplot(train[train['other_race_or_ethnicity'] == np.max(train.other_race_or_ethnicity)]['target'], ax=axarr[0]).set_title("other_race_or_ethnicity")
sns.violinplot(train[train['white'] == np.max(train.white)]['target'], ax=axarr[1]).set_title("white")
sns.despine()


# In[ ]:


np.min(train.target)


# In[ ]:


fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of gender in the train set', fontsize=15)
sns.kdeplot(train['female'], ax=axarr[0][0], color='violet')
sns.kdeplot(train['male'], ax=axarr[0][1], color='violet')
sns.kdeplot(train['transgender'], ax=axarr[1][0], color='violet')
sns.kdeplot(train['other_gender'], ax=axarr[1][1], color='violet')
sns.despine()


# **Distribution of gender with respect to target_binarized : ** 

# In[ ]:


g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "female", color="violet")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "male", color="violet")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "transgender", color="violet")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "other_gender", color="violet")


# In[ ]:


fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of sexual orientation in the train set', fontsize=15)
sns.kdeplot(train['bisexual'], ax=axarr[0][0], color='red')
sns.kdeplot(train['heterosexual'], ax=axarr[0][1], color='red')
sns.kdeplot(train['homosexual_gay_or_lesbian'], ax=axarr[1][0], color='red')
sns.kdeplot(train['other_sexual_orientation'], ax=axarr[1][1], color='red')
sns.despine()


# **Distribution of sexual orientation with respect to target_binarized : ** 

# In[ ]:


g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "bisexual", color="red")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "heterosexual", color="red")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "homosexual_gay_or_lesbian", color="red")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "other_sexual_orientation", color="red")


# In[ ]:


fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of disability in the train set', fontsize=15)
sns.kdeplot(train['intellectual_or_learning_disability'], ax=axarr[0][0], color='green')
sns.kdeplot(train['physical_disability'], ax=axarr[0][1], color='green')
sns.kdeplot(train['psychiatric_or_mental_illness'], ax=axarr[1][0], color='green')
sns.kdeplot(train['other_disability'], ax=axarr[1][1], color='green')
sns.despine()


# **Distribution of disability with respect to target_binarized :**

# In[ ]:


g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "intellectual_or_learning_disability", color="green")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "physical_disability", color="green")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "psychiatric_or_mental_illness", color="green")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "other_disability", color="green")


# ##  <a id='33'>Time Series Analysis</a>
# 
# Let's do some basic time series analysis on the train data.

# In[ ]:


train['created_date_time'] = pd.to_datetime(train['created_date']).values.astype('datetime64[M]')
#datetime64[Y] ==> Month and Date is always 1
#datetime64[M] ==> Date is always 1
#datetime64[D] ==> Year, Month and Data not neglected


# In[ ]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
train['created_date_time'].value_counts().sort_values().plot.line(fontsize=10).set_title("Number of comments vs Year-Month", 
                                                                                           fontsize=15)
sns.despine(bottom=True,  left=True)


# In[ ]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
train['created_date_time'].value_counts().resample('Y').sum().plot.line(fontsize=10).set_title("Number of comments vs Year", 
                                                                                           fontsize=15)
sns.despine(bottom=True,  left=True)


# So, year-wise, the trend is that the number of comments have increased.

# In[ ]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
lag_plot(train['target']).set_title("Lag Plot", fontsize=15)


# This suggests that the relationship between 'target' and time is weak or infact no relationship. So, let's not go ahead with an autocorrelation plot which would quantify the strength and type of relationship between observation and their lags.

# ##  <a id='34'>Comment Length Analysis</a>

# In[ ]:


train['comment_text_length'] = train['comment_text'].apply(lambda x : len(x))
fig, axarr = plt.subplots(1,1,figsize=(12, 6))
sns.kdeplot(train['comment_text_length']).set_title("Distribution of comment_text_length", fontsize=15)


# So, the maximum number of comments have a length of about ~70 characters

# **Distribution of comment_text_length with respect to target_binarized : ** 

# In[ ]:


g = sns.FacetGrid(train, col="target_binarized", size=4, aspect=1.5)
g.map(sns.kdeplot, "comment_text_length", color='red')


# Very long Toxic comments are lesser in number in comparison with Non-Toxic comments. Also Toxic comments with #characters > ~1100 don't exist.

# **Scatter plots**

# Considering not all, but only 100000 samples of train data for a scatter plot using ggplot library.

# In[ ]:


(
    ggplot(train.sample(100000))
        + geom_point()
        + aes(color='comment_text_length')
        + aes('comment_text_length', 'target')
)


# In[ ]:


sns.jointplot(x='comment_text_length', y='target', data=train, kind='hex', gridsize=20, size=8)


# Highest overlap between comment_text_length = ~90 and target = 0

# ##  <a id='35'>Word clouds</a>

# In[ ]:


stopwords = set(STOPWORDS)

def plot_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[ ]:


plot_wordcloud(train['comment_text'], title = 'Frequently used words in train data')


# In[ ]:


plot_wordcloud(train[train['target'] == np.max(train.target)]['comment_text'], title = 'Frequent Words : Toxicity target value = 1 #Toxic')


# In[ ]:


plot_wordcloud(train[train['target'] == np.min(train.target)]['comment_text'], title = 'Frequent Words : Toxicity target value = 0 #Non-Toxic')


# In[ ]:


plot_wordcloud(train[(train['female'] >0)&(train['target']>0.8)]['comment_text'], title = 'Frequent Words : toxicity target > 0.8 and Female')


# In[ ]:


plot_wordcloud(train[(train['male'] >0)&(train['target']>0.8)]['comment_text'], title = 'Frequent Words : toxicity target > 0.8 and Male')


# In[ ]:


plot_wordcloud(train[(train['insult'] >0.8)&(train['target']>0.8)]['comment_text'], title = 'Frequent Words : toxicity target > 0.8 and insult > 0.8')


# In[ ]:


plot_wordcloud(train[(train['threat'] >0.8)&(train['target']>0.8)]['comment_text'], title = 'Frequent Words : toxicity target > 0.8 and threat > 0.8')


# ##  <a id='4'>References</a>

# + [1] https://www.kaggle.com/gpreda/jigsaw-eda
# + [2] https://www.kaggle.com/chewzy/eda-toxicity-of-identities-updated-29-4
# + [3] https://www.kaggle.com/artgor/toxicity-eda-logreg-and-nn-interpretation

# **Please upvote if you find this kernel useful and comment below for any suggestions.**
