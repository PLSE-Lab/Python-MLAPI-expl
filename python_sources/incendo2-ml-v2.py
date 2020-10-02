#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import string


# In[ ]:


train = pd.read_csv('../input/train_dataset.csv')
test = pd.read_csv('../input/test_dataset.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.dtypes


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# Let's Proceed with some EDA

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train['Essayset'].value_counts(normalize=True).plot.bar()


# A fairly uniform distribution is present

# In[ ]:


train['max_score'].value_counts().plot.bar()


# Let's try to impute the missing values for each of the columns

# One good way to impute the missing value of EassaySet column is to take the advantage of the fact that essay set {1,2,5,6} has the max value of 10 and rest have the max value of 2.
# 
# So if the max score for the row is 3, I'll be imputing the set with the highest mode in {1,2,5,6} else with the mode of remaning value

# In[ ]:


from collections import defaultdict

def count_value(df):
        dic = defaultdict(int)
        dic1 = defaultdict(int)
        df = df.dropna()
        for val in df['Essayset']:
            if val in [1.0,2.0,5.0,6.0]:
                dic[val]+=1
            else:
                dic1[val]+=1
        return dic,dic1


# In[ ]:


X = train.copy()
X_test = test.copy()


# In[ ]:


count_value(X)


# In[ ]:


X.loc[(X['Essayset'].isna() == True) & (X['max_score'] == 3),['Essayset']] = 6.0
X.loc[(X['Essayset'].isna() == True) & (X['max_score'] == 2),['Essayset']] = 8.0


# So if the value of Eassayset is missing and the max marks for the row is 3 we will be imputing the value of 6.0 and if the max marks is 2 we will impute 8.0

# In[ ]:


import seaborn as sns


# In[ ]:


plt.figure(1,figsize=(16,16))


plt.subplot(321)
sns.distplot(X.loc[X['score_3'].isna()!=True,['score_3']])
plt.subplot(322)
sns.boxplot(y=X.loc[X['score_3'].isna()!=True,['score_3']])

plt.subplot(323)
sns.distplot(X.loc[X['score_4'].isna()!=True,['score_4']])
plt.subplot(324)
sns.boxplot(y=X.loc[X['score_4'].isna()!=True,['score_4']])

plt.subplot(325)
sns.distplot(X.loc[X['score_5'].isna()!=True,['score_5']])
plt.subplot(326)
sns.boxplot(y=X.loc[X['score_5'].isna()!=True,['score_5']])

plt.show()


# We will be imputying the missing values of scores by taking the mean score for each category of max_marks

# In[ ]:


mean_3_3 = np.mean(train.loc[train['max_score']==3,'score_3'])
mean_4_3 = np.mean(train.loc[train['max_score']==3,'score_4'])
mean_5_3 = np.mean(train.loc[train['max_score']==3,'score_5'])

mean_3_2 = np.mean(train.loc[train['max_score']==2,'score_3'])
mean_4_2 = np.mean(train.loc[train['max_score']==2,'score_4'])
mean_5_2 = np.mean(train.loc[train['max_score']==2,'score_5'])


# In[ ]:


X.loc[(X['score_3'].isna()==True) & (X['max_score']==3),'score_3'] = mean_3_3
X.loc[(X['score_4'].isna()==True) & (X['max_score']==3),'score_4'] = mean_4_3
X.loc[(X['score_5'].isna()==True) & (X['max_score']==3),'score_5'] = mean_5_3

X.loc[(X['score_3'].isna()==True) & (X['max_score']==2),'score_3'] = mean_3_2
X.loc[(X['score_4'].isna()==True) & (X['max_score']==2),'score_4'] = mean_4_2
X.loc[(X['score_5'].isna()==True) & (X['max_score']==2),'score_5'] = mean_5_2


# In[ ]:


X.isna().sum()


# Computing the average of score_1...score_5

# In[ ]:


X['score'] = X.loc[:,['score_1','score_2','score_3','score_4','score_5']].mean(axis=1)
X = X.drop(labels = ['score_1','score_2','score_3','score_4','score_5'],axis =1)


# In[ ]:


X['score'] = X['score'].round().astype('category')
X.head()


# In[ ]:


X['score'].value_counts(normalize=True).plot.bar()


# In[ ]:


from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go


# In[ ]:


count = X['score'].value_counts()
labels = count.index
value = np.array((count/count.sum())*100)

plot = go.Pie(labels=labels,values = value)
layout = go.Layout(title='Target Value Distribution')
fig = go.Figure(data=[plot],layout=layout)
py.iplot(fig,filename='Target Distribution')


# Distribution of different scores is quite skwed

# Let's see the word cloud of essay text

# In[ ]:


from wordcloud import STOPWORDS,WordCloud

def wcloud(text,title=None,figure_size=(24.0,16.0)):
    stopwords = set(STOPWORDS)
    stopwords = stopwords.union({'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'})
    
    wordcloud = WordCloud(stopwords=stopwords,random_state = 42,width=800, 
                    height=400,).generate(str(text))
    
    plt.figure(figsize=figure_size)
    plt.title(title,fontdict={'size': 40,})
    plt.imshow(wordcloud)


# In[ ]:


wcloud(X[X['score']==3]['EssayText'],'3 Marks: Essay_Text_Cloud')


# In[ ]:


wcloud(X[X['score']==2]['EssayText'],'2 Marks: Essay_Text_Cloud')


# In[ ]:


wcloud(X[X['score']==1]['EssayText'],'1 Mark: Essay_Text_Cloud')


# In[ ]:


wcloud(X[X['score']==0]['EssayText'],'0 Mark: Essay_Text_Cloud')


# From the above word clouds we can see that the highest scoring essay texts includes words such as Osmosis,Diffusion,etc and these words are coherent with the context of essay, on other hand low scoring essays includes such as white black, etc which seems to be incoherent with the context of essay

# It's time to create some meta features and check out how they are distributed in the dataset:
# 
# * Number of words in the text
# * Number of unique words in the text
# * Number of characters in the text
# * Number of stopwords
# * Number of punctuations
# * Number of upper case words
# * Number of title case words
# * Average length of the words

# In[ ]:


X['words'] = X['EssayText'].apply(lambda x: len(x.split()))
X_test['words'] = X_test['EssayText'].apply(lambda x: len(x.split()))

X['unique'] = X['EssayText'].apply(lambda x: len(set(x.split())))
X_test['unique'] = X_test['EssayText'].apply(lambda x: len(set(x.split())))

X['char'] = X['EssayText'].apply(lambda x: len(x))
X_test['char'] = X_test['EssayText'].apply(lambda x: len(x))

X['stop'] = X['EssayText'].apply(lambda x: len([word for word in str(x).lower().split() if word in set(STOPWORDS)]))
X_test['stop'] = X_test['EssayText'].apply(lambda x: len([word for word in str(x).lower().split() if word in set(STOPWORDS)]))

X['punct'] = X['EssayText'].apply(lambda x: len([punct for punct in str(x) if punct in string.punctuation]))
X_test['punct'] = X_test['EssayText'].apply(lambda x: len([punct for punct in str(x) if punct in string.punctuation]))

X['upper'] = X['EssayText'].apply(lambda x: len([word for word in x.split() if word.isupper()]))
X_test['upper'] = X_test['EssayText'].apply(lambda x: len([word for word in x.split() if word.isupper()]))

X['title'] = X['EssayText'].apply(lambda x: len([word for word in x.split() if word.istitle()]))
X_test['title'] = X_test['EssayText'].apply(lambda x: len([word for word in x.split() if word.istitle()]))

X['avg_word'] = X['EssayText'].apply(lambda x: (np.sum([len(word) for word in x.split()]))/len(x.split()))
X_test['avg_word'] = X_test['EssayText'].apply(lambda x: (np.sum([len(word) for word in x.split()]))/len(x.split()))


# In[ ]:


X.head()


# In[ ]:


X.iloc[15010].EssayText


# We can see that there are responses having greater than 50 words as opposed to the problem statement

# In[ ]:


# Truncate some extreme values for better visuals ##
X['words'].loc[X['words']>100] = 100 #truncation for better visuals
X['punct'].loc[X['punct']>10] = 10 #truncation for better visuals
X['char'].loc[X['char']>450] = 450 #truncation for better visuals

f, axes = plt.subplots(3, 1, figsize=(10,20))
sns.boxplot(x='score', y='words', data=X, ax=axes[0])
axes[0].set_xlabel('Score', fontsize=12)
axes[0].set_title("Number of words in each class", fontsize=15)

sns.boxplot(x='score', y='punct', data=X, ax=axes[1])
axes[1].set_xlabel('Score', fontsize=12)
axes[1].set_title("Number of characters in each class", fontsize=15)

sns.boxplot(x='score', y='char', data=X, ax=axes[2])
axes[2].set_xlabel('Score', fontsize=12)
axes[2].set_title("Number of punctuations in each class", fontsize=15)
plt.show()


# From the box plots we can see that students scoring more marks have more number of words and punctuations in their answer,these features might be useful for our model.

# In[ ]:


# X['words'].loc[X['words']>100] = 100 #truncation for better visuals
# X['punct'].loc[X['punct']>10] = 10 #truncation for better visuals
# X['char'].loc[X['char']>450] = 450 #truncation for better visuals

f, axes = plt.subplots(5, 1, figsize=(10,30))
sns.boxplot(x='score', y='unique', data=X, ax=axes[0])
axes[0].set_xlabel('Score', fontsize=12)
axes[0].set_title("Number of unique words in each class", fontsize=15)

sns.boxplot(x='score', y='stop', data=X, ax=axes[1])
axes[1].set_xlabel('Score', fontsize=12)
axes[1].set_title("Number of stop words in each class", fontsize=15)

sns.boxplot(x='score', y='upper', data=X, ax=axes[2])
axes[2].set_xlabel('Score', fontsize=12)
axes[2].set_title("Number of Upper Case in each class", fontsize=15)


sns.boxplot(x='score', y='title', data=X, ax=axes[3])
axes[3].set_xlabel('Score', fontsize=12)
axes[3].set_title("Number of Title Case in each class", fontsize=15)

sns.boxplot(x='score', y='avg_word', data=X, ax=axes[4])
axes[4].set_xlabel('Score', fontsize=12)
axes[4].set_title("Number of average in each class", fontsize=15)

plt.show()


# Probably the best way to fill the missing values of clarity and cohernt columns would be on the basis of marks

# In[ ]:


X.clarity.value_counts()


# In[ ]:


X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 0) & (X['max_score']==3.0)] = 'worst'
X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 1) & (X['max_score']==3.0)] = 'average'
X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 2) & (X['max_score']==3.0)] = 'above_average'
X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 3) & (X['max_score']==3.0)] = 'excellent'

X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 0) & (X['max_score']==2.0)] = 'worst'
X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 1) & (X['max_score']==2.0)] = 'average'
X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 2) & (X['max_score']==2.0)] = 'excellent'

X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 0) & (X['max_score']==3.0)] = 'worst'
X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 1) & (X['max_score']==3.0)] = 'average'
X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 2) & (X['max_score']==3.0)] = 'above_average'
X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 3) & (X['max_score']==3.0)] = 'excellent'

X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 0) & (X['max_score']==2.0)] = 'worst'
X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 1) & (X['max_score']==2.0)] = 'average'
X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 2) & (X['max_score']==2.0)] = 'excellent'


# In[ ]:


X.isna().sum()


# As clarity and coherncy are ordinal values we will be encoding them to numerical one

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le_clarity = LabelEncoder()
le_coherent = LabelEncoder()

X['clarity'] = le_clarity.fit_transform(X['clarity'])
X_test['clarity'] = le_clarity.transform(X_test['clarity'])

X['coherent'] = le_coherent.fit_transform(X['coherent'])
X_test['coherent'] = le_coherent.transform(X_test['coherent'])


# In[ ]:


X = X.drop(labels = ['ID','min_score','max_score','EssayText','avg_word'],axis=1)
X_test = X_test.drop(labels = ['ID','min_score','max_score','EssayText','avg_word'],axis=1)


# In[ ]:


X.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,classification_report, log_loss,f1_score
from sklearn.svm import LinearSVC


# In[ ]:


kf = KFold(n_splits=5,shuffle=True,random_state=42)
eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']
cv_scores = []
pred_val = np.zeros([X.shape[0]])
for train_index, val_index in kf.split(X):
    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values
    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values
    classifier = LogisticRegression(penalty= 'l1',class_weight='balanced', C = 1.0,
                                    multi_class = 'auto',solver='liblinear',random_state=42,max_iter=200)
    classifier.fit(X_train,y_train)
    pred_prob = classifier.predict_proba(X_val)
    pred = classifier.predict(X_val)
    pred_val[val_index] = pred
    print(accuracy_score(y_val,pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


kf = KFold(n_splits=5,shuffle=True,random_state=42)
eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']
cv_scores = []
pred_val = np.zeros([X.shape[0]])
for train_index, val_index in kf.split(X):
    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values
    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values
    
    rfc = RandomForestClassifier(n_estimators = 70,random_state=42,n_jobs=-1,criterion='entropy',
                                min_samples_leaf=20,min_samples_split=10)
    rfc.fit(X_train,y_train)
    
    pred_prob = rfc.predict_proba(X_val)
    pred = rfc.predict(X_val)
    pred_val[val_index] = pred
    print(accuracy_score(y_val,pred))


# In[ ]:


kf = KFold(n_splits=5,shuffle=True,random_state=42)
eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']
cv_scores = []
pred_val = np.zeros([X.shape[0]])
for train_index, val_index in kf.split(X):
    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values
    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values
    
    abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42,
                                                                   criterion='entropy',
                                                                   min_samples_leaf=20,
                                                                   min_samples_split=10))
    abc.fit(X_train,y_train)
    
    pred_prob = abc.predict_proba(X_val)
    pred = abc.predict(X_val)
    pred_val[val_index] = pred
    print(accuracy_score(y_val,pred))


# In[ ]:


kf = KFold(n_splits=5,shuffle=True,random_state=42)
eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']
cv_scores = []
pred_val = np.zeros([X.shape[0]])
for train_index, val_index in kf.split(X):
    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values
    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values
    
    etc = ExtraTreesClassifier(n_estimators = 30,random_state=42,n_jobs=-1,criterion='entropy',
                               min_samples_leaf=10,min_samples_split=10)
    etc.fit(X_train,y_train)
    
    pred_prob = etc.predict_proba(X_val)
    pred = etc.predict(X_val)
    pred_val[val_index] = pred
    print(accuracy_score(y_val,pred))


# In[ ]:


kf = KFold(n_splits=5,shuffle=True,random_state=42)
eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']
cv_scores = []
pred_val = np.zeros([X.shape[0]])
for train_index, val_index in kf.split(X):
    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values
    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values
    
    knb = KNeighborsClassifier(n_neighbors=15, weights='distance',n_jobs=-1)
    knb.fit(X_train,y_train)
    
    pred_prob = knb.predict_proba(X_val)
    pred = knb.predict(X_val)
    pred_val[val_index] = pred
    print(accuracy_score(y_val,pred))


# In[ ]:


kf = KFold(n_splits=5,shuffle=True,random_state=42)
eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']
cv_scores = []
pred_val = np.zeros([X.shape[0]])
for train_index, val_index in kf.split(X):
    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values
    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values
    
    mnb = MultinomialNB()
    mnb.fit(X_train,y_train)
    
    pred_prob = mnb.predict_proba(X_val)
    pred = mnb.predict(X_val)
    pred_val[val_index] = pred
    print(accuracy_score(y_val,pred))


# We can see that for our dataset Random Forest Classifier Model is giving the best results, we will be using that model to produce final result

# In[ ]:


pred_sub = rfc.predict(X_test)
X_test['essay_score'] = pred_sub
X_test.head()
sub = test.copy()
sub['essay_score'] = pred_sub
sub = sub.drop(labels=['min_score','max_score','clarity','coherent','EssayText'],axis=1)
sub.columns = ['id','essay_set', 'essay_score']
sub.head()


# In[ ]:


sub.to_csv(path_or_buf = 'submission.csv',index=False)

