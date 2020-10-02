#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
pd.set_option('chained_assignment',None)

# Any results you write to the current directory are saved as output.


# # About the Data
# This dataset consists of a nearly 3000 Amazon customer reviews (input text), star ratings, date of review, variant and feedback of various amazon Alexa products like Alexa Echo, Echo dots, Alexa Firesticks etc. for learning how to train Machine for sentiment analysis.
# 
# ![image.png](attachment:image.png)

# In[ ]:


data = pd.read_csv("../input/amazon_alexa.tsv", sep='\t')
print("Shape of Dataframe is {}".format(data.shape))


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# ## EDA & Visualizations

# In[ ]:


test = data.copy(deep=True)
test.loc[test['feedback'] == 1, 'feedback'] = 'Positive'
test.loc[test['feedback'] == 0, 'feedback'] = 'Negative'


# In[ ]:


plt.figure(figsize=(12, 7))
sns.scatterplot(x="rating", y="rating", hue="feedback",data=test)
plt.title("Relation between Rating and Overall Feedback");


# > * For feedback to be positive (1), rating >= 3

# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(24, 10))

data.feedback.value_counts().plot.barh(ax=axs[0])
axs[0].set_title(("Class Distribution - Feedback {1 (positive) & 0 (negative)}"));

data.rating.value_counts().plot.barh(ax=axs[1])
axs[1].set_title("Class Distribution - Ratings");


# > * We have a highly positive skewed distribution here in both cases, and these products have been pretty well received by the customers !
# > * Gentle Reminder : Make sure to stratify the data, to avoid class imbalance for review classification models

# In[ ]:


data.variation.value_counts().plot.barh(figsize=(12, 7))
plt.title("Class Distribution - Variation");


# > * We have distinct bins here, showing pattern of customer preference for various models, with Black Dot as the most popular one

# In[ ]:


data.groupby('variation').mean()[['rating']].plot.barh(figsize=(12, 7))
plt.title("Variation wise Mean Ratings");


# > * No obvious patterns here, all the variations of the product have been equally well received

# In[ ]:


data['review_length'] = data.verified_reviews.str.len()


# In[ ]:


pd.DataFrame(data.review_length.describe()).T


# > * Summary Stats reveal the overall skew in review length distribution
# > * Histogram below will confirm our findings 

# In[ ]:


data['review_length'].plot.hist(bins=200, figsize=(16, 7))
plt.title("Histogram of Review Lengths");


# In[ ]:


data.groupby('rating').mean()[['review_length']].plot.barh(figsize=(12, 7))
plt.title("Mean Length of Reviews - Grouped by Ratings");


# > * Rating 2 : Customers tend to describe the flaws  in detail and it's natural to be vocal about something you didn't find good. 
# > * Rating 5 : I guess we'll have broadly two kinds of reviews here;  people who actually describe the positives in about 100 words or so, and the ones who comment "Awesome", "Loved it !" etc. 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')


# In[ ]:


cv.fit_transform(data.verified_reviews);


# In[ ]:


vector = cv.fit_transform(data.verified_reviews)


# In[ ]:


sum_words = vector.sum(axis=0)


# In[ ]:


words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)


# In[ ]:


freq_df = pd.DataFrame(words_freq, columns=['word', 'freq'])


# In[ ]:


freq_df.head(15).plot(x='word', y='freq', kind='barh', figsize=(20, 12))
plt.title("Most Frequently Occuring Words - Top 15");


# In[ ]:


from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white',width=800, height=500).generate_from_frequencies(dict(words_freq))
plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize=22);


# ## Classifier - Predict Feedback

# ### Features : 
# >> * Vectorized Review words
# >> * Review Lengths
# >> * Product Variation
# 
# ### Model Config
# >> * RandomForest Classifier - Grid Search with 5 Cross Validation

# In[ ]:


features = pd.DataFrame(vector.toarray(), columns=list(sorted(cv.vocabulary_)))


# In[ ]:


features = features.join(data[['review_length', 'variation']], rsuffix='_base')
features = pd.get_dummies(features)


# In[ ]:


target = data[['feedback']].astype(int)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)


# In[ ]:


model = RandomForestClassifier()


# In[ ]:


params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv_object = StratifiedKFold(n_splits=5)

grid = GridSearchCV(estimator=model, param_grid=params, cv=cv_object, verbose=0, return_train_score=True)
grid.fit(x_train, y_train.values.ravel())


# In[ ]:


pd.crosstab(y_train['feedback'], grid.predict(x_train), rownames=['True'], colnames=['Predicted'], margins=True)


# In[ ]:


print("Best Parameter Combination : {}".format(grid.best_params_))


# In[ ]:


print("Mean Cross Validation Accuracy - Train Set : {}".format(grid.cv_results_['mean_train_score'].mean()*100))
print("="*70)
print("Mean Cross Validation Accuracy - Validation Set : {}".format(grid.cv_results_['mean_test_score'].mean()*100))


# In[ ]:


feature_imp_df = pd.DataFrame([grid.best_estimator_.feature_importances_], columns=list(x_train.columns)).T
feature_imp_df.columns = ['imp']
feature_imp_df.sort_values('imp', ascending=False, inplace=True)


# In[ ]:


feature_imp_df.head(15).plot.barh(figsize=(16, 9))
plt.title("15 Most Important Features");


# > * Review Length seems to be driving the predictions with the highest feature importance
# > * We can also see significant words like 'terrible', 'stopped', 'love' etc in the top 15. Overall pretty decent feature selection.

# In[ ]:


y_test['pred'] = grid.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print("Accuracy Score for Test Set : {}".format(accuracy_score(y_test.feedback, y_test.pred)*100))


# In[ ]:


pd.crosstab(y_test['feedback'], grid.predict(x_test), rownames=['True'], colnames=['Predicted'], margins=True)

