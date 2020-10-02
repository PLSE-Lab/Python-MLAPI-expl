#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Credits to HyungsukKang's kernel on stratified K fold on XGBoost
#https://www.kaggle.com/sudosudoohio/stratified-kfold-xgboost-eda-tutorial-0-281


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score


plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"));

# Any results you write to the current directory are saved as output.


# ****Yelp:****
# 
# Yelp is an American multinational corporation headquartered in San Francisco, California. It develops, hosts and markets Yelp.com and the Yelp mobile app, which publish crowd-sourced reviews about local businesses, as well as the online reservation service Yelp Reservations. The company also trains small businesses in how to respond to reviews, hosts social events for reviewers, and provides data about businesses, including health inspection scores.
# 
# 
# *
# source: Wikipedia*
# 
# ![image.png](attachment:image.png)
# 
# 
# 
# 
# 
# 
# Initally we will begin with some exploratory data analysis.
# 
# Based on what we observe, we will formulate a business question and come up with a model to solve the business problem followed by model validation.
# 
# This is referred to as QMV approach
# 
# Q - Formulating a question
# 
# M - Building a model
# 
# V - Validation of model
# 
# We iterate the process until we end up some satisfactory results.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# **1.1  Businesses on Yelp **

# In[ ]:


yelp_business = pd.read_csv('../input/yelp_business.csv')


# In[ ]:


yelp_business.head()


# Now let's check for the missing values
# 
# There are many missing values in neighborhood and postal code fields.

# **1.2 Missing values in our dataset:**

# In[ ]:


yelp_business.isnull().sum()


# The following heatmap can be used to visualize the missing values effectively

# In[ ]:


plt.figure(figsize=(12,10))
f = sns.heatmap(yelp_business.isnull(),yticklabels=False, cbar=False, cmap = 'viridis')


# let's check how many of the businesses are open using a count plot on seaborn

# **1.3 Distribution of variables**

# In[ ]:


plt.figure(figsize=(6,6))
sns.countplot(x='is_open',data=yelp_business);


# In[ ]:


#let's look at the number of unique values in the stars variable.

yelp_business['stars'].nunique()


# let's use a plot to visualize the frequency of the ratings

# In[ ]:


sns.countplot(x='stars',data=yelp_business);


# We see that most of the businesses got a rating of either 3 or above

# Let's visualize the distribution of number of reviews that a business has, we have applied log since the distribution is extremely skewed

# In[ ]:


sns.distplot(yelp_business['review_count'].apply(np.log1p));


# In[ ]:


yelp_business_attributes = pd.read_csv('../input/yelp_business_attributes.csv')


# Our intuition is that the number of businesses vary a lot from one state to another.
# 
# We can use a tree map to visualize some of them since  a tree map might not be able to fit all the states

# In[ ]:



by_state = yelp_business.groupby('state')


# In[ ]:


import squarify    # pip install squarify (algorithm for treemap)
plt.figure(figsize=(12,12))

a = by_state['business_id'].count()

a.sort_values(ascending=False,inplace=True)

squarify.plot(sizes= a[0:15].values, label= a[0:15].index, alpha=0.9)

plt.axis('off')
plt.tight_layout()


# From the tree chat, we see that Arizona and Nevada has the most number of businesses

# **1.4 Leading Business Categories:**
# 
# Let's look at the leading business categories in out dataset. 
# A business is linked to multiple categories in our dataset, so we have to do a bit of preprocessing.
# 

# In[ ]:




business_cats=';'.join(yelp_business['categories'])
cats=pd.DataFrame(business_cats.split(';'),columns=['category'])
cats_ser = cats.category.value_counts()


cats_df = pd.DataFrame(cats_ser)
cats_df.reset_index(inplace=True)

plt.figure(figsize=(12,10))
f = sns.barplot( y= 'index',x = 'category' , data = cats_df.iloc[0:20])
f.set_ylabel('Category')
f.set_xlabel('Number of businesses');


# **1.5 Top Business Names:**
# 
# The top business names in our dataset can be visualized with a word cloud.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))

wordcloud = WordCloud(background_color='white',
                          width=1200,
                      stopwords = STOPWORDS,
                          height=1000
                         ).generate(str(yelp_business['name']))


plt.imshow(wordcloud)
plt.axis('off');


# It's time to hypothesize something and build a model to answer the question in hand.
# 
# One of the feature that we observed in our exploratory data analysis is 'is_open' which means whether a business is open or closed. Since the possible values for this variable is 0 or 1, This is a typical classification problem.
# 
# From the count plot above, we can infer that the proportion of the variable 'is_open' in our dataset is imbalanced. The businesses that are open are close to 84% and the businesses are closed are 16%. We will see the effects of class imbalance in a few minutes.
# 
# It's time to collect the features that are useful in answering the question.
# 
# Most of the features in our dataset are categorical.
# We have too many values in name, neighborhood, address, city and states vairables. One hot encoding increases the dimensionality drastically for these features.
# 
# Intuitively I have selected latitude, longitude, stars and review_count intially to see to what extent we could answer the question
# 
# I want to build a basic logistic regression model and create new features and move on to using XGBoost model.

# cols1 is a list of our explanatory variables we want to use in our models.
# The dependent variable we are trying to predict is whether a business is open.

# In[ ]:


cols1 = ['latitude',
 'longitude',
 'stars',
 'review_count']


# In[ ]:


X = yelp_business[cols1]
y = yelp_business['is_open']


# We have one missing value for latitude and longitude in our dataset. 
# I am going to impute this with a 0 to prevent a long list of scikit learn errors.

# In[ ]:


X.fillna(0.0,inplace=True)


# We don't have separate test and train datasets.
# 
# So lets split our dataset to train and test sets. I want to keep aside 0.3 times of the data we have as the test data.
# 
# Class imbalance arises due to the fact that model is trained predominantly on the label of majority class and very little on the minority class.
# 
# One of the negative effects of class imbalance is that - Since the proportion of majority class in our dataset is 84%, the model will predict everything to be of the majority class and we will end up with an accuracy of 84%. But it doesnt really mean anything and useless. 
# 
# 
# One way to get around this is to use SMOTE.
# SMOTE stands for - Synthetic Minority Over-sampling Technique.
# 
# As per wikipedia "Oversampling and undersampling in data analysis are techniques used to adjust the class distribution of a data set (i.e. the ratio between the different classes/categories represented).
# 
# Oversampling and undersampling are opposite and roughly equivalent techniques. They both involve using a bias to select more samples from one class than from another.
# 
# The usual reason for oversampling is to correct for a bias in the original dataset. One scenario where it is useful is when training a classifier using labelled training data from a biased source, since labelled training data is valuable but often comes from un-representative sources.
# 
# For example, suppose we have a sample of 1000 people of which 66.7% are male. We know the general population is 50% female, and we may wish to adjust our dataset to represent this. Simple oversampling will select each female example twice, and this copying will produce a balanced dataset of 1333 samples with 50% female. Simple undersampling will drop some of the male samples at random to give a balanced dataset of 667 samples, again with 50% female."
# 
# The over sampling is done by creating synthetic observations using K-Means.
# 
# We will use confusion matrix in conjunction with accuracy for the sake of interpretation. Nevertheless, we will focus on improving the accuracy of the minority class prediction.

# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.3, random_state = 42)


# We are going to smote just the training data because creating synthetic observations using the test dataset violates the rules of cross validation.

# In[ ]:


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(train_X, train_y)


# In[ ]:


X_res = pd.DataFrame(X_res)
y_res = pd.DataFrame(y_res)
test_X = pd.DataFrame(test_X)
test_y = pd.DataFrame(test_y)


# In[ ]:


L = [0.0001,0.001,0.01,0.1,1,10]
accuracy = {}
for i in L:
    LR = LogisticRegression(C=i)
    LR.fit(X_res,y_res)
    pred_y = LR.predict(test_X)
    
    accuracy[i] = 100*accuracy_score(test_y,pred_y)


# In[ ]:


accuracy


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot( list(accuracy.keys()), list(accuracy.values()), '--')
plt.xticks(list(accuracy.keys()))
plt.xlim(0,1)
plt.xlabel("C Values")
plt.ylabel("Accuracy")
("")


# Based on the above visualization, I have chosen a C value of 0.001
# A lower C value means more regularization

# In[ ]:


LR = LogisticRegression(C=0.001)
LR.fit(X_res,y_res)
pred_y = LR.predict(test_X)


# In[ ]:


confusion_matrix(test_y,pred_y)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(test_y,pred_y)


# An accuracy score of 56.2 isn't really good enough, But we have used very shallow features.
# 
# We need to include more features to our model.

# We have the review data that's written by users for businesses.
# 
# This might be a good place to collect some features.
# 
# The stars included in the review might be useful to gauge as to how good a business is doing.
# 
# We have grouped the businesses by the business_id and extracted features such as mean, median and count of the reviews a business received.

# In[ ]:


review = pd.read_csv('../input/yelp_review.csv')
#checkin = pd.read_csv('../input/yelp_checkin.csv')

review_busines = review.groupby(by='business_id')

review_businesid = pd.DataFrame()
review_businesid['Mean'] = review_busines['stars'].mean()
review_businesid['Median'] = review_busines['stars'].median()
review_businesid['NumberOfReviews'] = review_busines['stars'].count()


# If a business is performing well, we can notice more customers visiting the business and consequently more checkins.
# 
# We have the checkin data as well. We have extracted the sum of checkins at a business_id level and merged it review features.

# In[ ]:


checkin = pd.read_csv("../input/yelp_checkin.csv")

checkin_bus = checkin.groupby(by='business_id')

checkin_busid = pd.DataFrame()
checkin_busid['TotalCheckins'] = checkin_bus['checkins'].sum()

checkin_busid.reset_index(inplace=True)
review_businesid.reset_index(inplace=True)


review_businesid = pd.merge(left=review_businesid,right=checkin_busid,on='business_id', how='left')


# The last step is merging this data frame to our original dataframe yelp_business

# In[ ]:


yelp_business = pd.merge(left=yelp_business,right=review_businesid,on='business_id',how='left')


# In[ ]:


yelp_business.columns


# We have decided to drop lat and long because it was overfitting to an extent on our data.

# In[ ]:


cols1 = ['stars',
 'review_count',
        'Mean', 'Median', 'NumberOfReviews','TotalCheckins']


# In[ ]:


X = yelp_business[cols1]
y = yelp_business['is_open']

X.fillna(0.0,inplace=True)


# Train test split again

# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.3, random_state = 42)


# SMOTE

# In[ ]:


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(train_X, train_y)


# In[ ]:


X_res = pd.DataFrame(X_res,columns = train_X.columns.tolist())
y_res = pd.DataFrame(y_res)
test_X = pd.DataFrame(test_X)
test_y = pd.DataFrame(test_y)


# In[ ]:


L = [0.0001,0.001,0.01,0.1,1,10]

accuracy = {}
for i in L:
    LR = LogisticRegression(C=i)
    LR.fit(X_res,y_res)
    pred_y = LR.predict(test_X)
    
    accuracy[i] = 100*accuracy_score(test_y,pred_y)


# In[ ]:


accuracy


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot( list(accuracy.keys()), list(accuracy.values()), '--')
plt.xticks(list(accuracy.keys()))
plt.xlim(-0.1,1)
plt.xlabel("C Values")
plt.ylabel("Accuracy");


# 

# We chose a C value of 0.0001 based on the above plot.

# In[ ]:


LR = LogisticRegression(C=0.0001)
LR.fit(X_res,y_res)
pred_y = LR.predict(test_X)


# In[ ]:


accuracy_score(test_y,pred_y)


# In[ ]:


confusion_matrix(test_y,pred_y)


# The accuracy improved by more than 2% to 58.5%,  But this isn't good enough.
# 
# 

# Let's try an XGBoost model on this dataset and see how it performs.

# In[ ]:


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=400):
    param = {}
    param['objective'] = 'multi:softmax'
    param['num_class'] = 2
    param['eta'] = 0.12
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'merror'
    param['min_child_weight'] = 1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return model,pred_test_y


# In[ ]:


test_y_pred= pd.DataFrame()
test_y_pred['id'] = test_y.index
test_y_pred['is_open'] = np.zeros_like(test_y.index)


# In[ ]:


X_res = X_res.values
y_res = y_res.values
test_X = test_X.values


# In[ ]:


kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)

for i, (train_index, test_index) in enumerate(skf.split(X_res, y_res)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X_res[train_index], X_res[test_index]
    y_train, y_valid = y_res[train_index], y_res[test_index]
    
    model1,y = runXGB( X_train,y_train,X_valid,y_valid)
    test_pred = model1.predict(xgb.DMatrix(test_X))
    test_y_pred['is_open'] += test_pred/kfold


# In[ ]:


test_y_pred['is_open'] = np.round(test_y_pred['is_open']).apply(int)


# In[ ]:


accuracy_score(test_y,test_y_pred['is_open'])


# In[ ]:


confusion_matrix(test_y,test_y_pred['is_open'])


# WOW!
# 
# The accuracy jumped from 58.5% to 75.15% using an XGBoost model with Stratied Kfold for cross validation.
# 
# I will add in few more details when i get time and finetune.
# 
# Thanks folks!
# 
# Feedback is highly appreciated, upvote if you have liked it.

# In[ ]:




