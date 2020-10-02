#!/usr/bin/env python
# coding: utf-8

# # Daily News for Stock Market Prediction
# 
# Determine if the news affect the stock market.

# ## A. Data Exploration/Preparation

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Open file and inspect first five rows.
df_A = pd.read_csv('../input/Combined_News_DJIA.csv')
df_A.head()


# The features Top1 to Top25 represent the top-ranked news headlines as voted by users in the Reddit World News Channel. The date feature is the date the news were reported, and the label feature represents stock market movement on that day, with 1 being positive or neutral and 0, negative.

# In[ ]:


# Create features that concatenates the columns and returns the text length.
df_A['combined_text'] = df_A.iloc[:,2:27].apply(lambda x: ''.join((x).astype(str)), axis=1)
df_A['text_length'] = df_A['combined_text'].apply(len)


# In[ ]:


# Create a feature that determines text sentiment.
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df_A['sentiment_score'] = df_A['combined_text'].apply(lambda y: sia.polarity_scores(y)['compound'])
df_A['sentiment'] = df_A['sentiment_score'].apply(lambda z: 'positive' if z >= 0 else 'negative')


# In[ ]:


# Show the updated dataframe.
df_A.head()


# In[ ]:


# Plot market movement vs. news sentiment.
n_groups = 2
mark_cat = (df_A['Label'].value_counts()[1], df_A['Label'].value_counts()[0])
news_cat = (df_A['sentiment'].value_counts()['positive'], df_A['sentiment'].value_counts()['negative'])
 
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, mark_cat, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Market')
 
rects2 = plt.bar(index + bar_width, news_cat, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Sentiment')
 
plt.xlabel('Category')
plt.ylabel('Count/Days')
plt.title('Market Movement vs News Sentiment Distribution')
plt.xticks(index + 1/2*bar_width, ('Positive', 'Negative'))
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Show proportion.
print('Positive market days: ', (df_A['Label'] == 1).mean())
print('Positive news sentiment: ', (df_A['sentiment'] == 'positive').mean())
print('Negative market days: ', (df_A['Label'] == 0).mean())
print('Negative news sentiment: ', (df_A['sentiment'] == 'negative').mean())


# In[ ]:


# Run a t-test on the label column.
from scipy import stats
stats.ttest_ind(df_A['Label']==1, df_A['Label']==0)


# In[ ]:


# Correlation between the label and sentiment score features.
print('Correlation: ', df_A['Label'].corr(df_A['sentiment_score']))


# In[ ]:


# Plot text length distribution vs. market movement.
g = sns.FacetGrid(data=df_A, col='Label')
g.map(plt.hist, 'text_length', bins=25)
plt.show()
sns.boxplot(x='Label', y='text_length', data=df_A)
plt.show()


# In[ ]:


# Run a wordcloud on the positive market days.
pos_txt = []
for row in range(0, len(df_A[df_A['Label'] == 1])):
    pos_txt.append(' '.join(str(x) for x in df_A[df_A['Label'] == 1].iloc[row,2:27]))
    
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vect_A = CountVectorizer(stop_words='english')
pos_trans = vect_A.fit_transform(pos_txt)
pos_count = pos_trans.toarray().sum(axis=0) 

col_names = vect_A.get_feature_names()
pos_dict = dict(zip(col_names, pos_count))

from wordcloud import WordCloud,STOPWORDS
pos_wc = WordCloud(background_color='white', width=3000, height=2500).generate_from_frequencies(pos_dict)
plt.figure(1,figsize=(8,8))
plt.imshow(pos_wc)
plt.axis('off')
plt.show()


# In[ ]:


# Show the top 10 words and their counts and percentages. 
import operator
pos_sort = sorted(pos_dict.items(), reverse=True, key=operator.itemgetter(1))
df_pos = pd.DataFrame(data=pos_sort, columns=['token', 'count'])
df_pos['percentage'] = df_pos['count']/df_pos['count'].sum()*100
df_pos.head(10)


# In[ ]:


# Run a wordcloud on the negative market days.
neg_txt = []
for row in range(0, len(df_A[df_A['Label'] == 0])):
    neg_txt.append(' '.join(str(x) for x in df_A[df_A['Label'] == 0].iloc[row,2:27]))
    
neg_trans = vect_A.fit_transform(neg_txt)
neg_count = neg_trans.toarray().sum(axis=0) 

neg_dict = dict(zip(col_names, neg_count))

neg_wc = WordCloud(background_color='black', width=3000, height=2500).generate_from_frequencies(neg_dict)
plt.figure(1,figsize=(8,8))
plt.imshow(neg_wc)
plt.axis('off')
plt.show()


# In[ ]:


# Show the top 10 words and their counts and percentages.
neg_sort = sorted(neg_dict.items(), reverse=True, key=operator.itemgetter(1))
df_neg = pd.DataFrame(data=neg_sort, columns=['token', 'count'])
df_neg['percentage'] = df_neg['count']/df_neg['count'].sum()*100
df_neg.head(10)


# Train/test split before the CountVectorizer to properly simulate the real world where future data contains words the model has not seen before. If the dataset is vectorized before the train/test split, the document-term matrix would contain every single feature in the training and test sets.

# In[ ]:


# Set variables.
X = df_A.drop('Label', axis=1) 
y = df_A['Label']

# Split data into train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=123)


# In[ ]:


# Combine the train columns' text into a string.
train_txt = []
for row in range(0, len(X_train.index)):
    train_txt.append(' '.join(str(x) for x in X_train.iloc[row,1:26]))


# Create a feature vector that will perform the classification task. Use ski-kit learn's CountVectorizer(), which takes the bag of words approach, to create a matrix of numbers to represent the text. Each news will be separated into words (i.e., tokens) and the number of times each token occurs in the news will be counted. Passing text to CountVectorizer's default settings will convert words to lowercase, remove punctuation, and exclude duplicates and words with less than two letters. 

# In[ ]:


# Fit the train text into the vectorizer.
vect_A.fit(train_txt)

# Print the tokens and their corresponding column indices.
print(vect_A.vocabulary_)


# In[ ]:


# Transform the CountVectorizer object to create a document term matrix populated with token counts.
dtm_A_train = vect_A.transform(train_txt)
print(dtm_A_train)


# The resulting matrix is in compressed sparse row format (i.e., sparse matrix), which means only the location and value of non-zero values is saved. For example, the first object (0, 1) indicates the 2nd feature (i.e., token/word) has one instance in the first text.

# In[ ]:


# Show the equivalent dataframe (i.e., dense matrix version).
pd.DataFrame(dtm_A_train.toarray(), columns=vect_A.get_feature_names())


# In this particular dataframe, each row represents a date and each column a token or word, which indicates the default settings in CountVectorizer identified almost 30,000 unique tokens. The numbers represent the token's count on that particular date. 
# 
# Token counts are discrete features and are best suited with classification models. The model is trained by creating probability classes based on the tokens' instances in both market directions. As an example:
# 
# |Token      | 1   | 0  | Total |
# |-----------|-----|----|-------|
# | gov't     | 50  | 50 | 100   |
# | functions | 75  | 25 | 100   |
# | well      | 90  | 10 | 100   |
# | total     | 215 | 85 | 300   |
# 
# Positive class = 215/300 = 0.717 <br>
# Negative class = 85/300 = 0.283
# 
# Text that contains "gov't functions well" can be broken down as:
# 
# Positive class = (0.717)(50/215)(75/215)(90/215) = 0.024 <br>
# Negative class = (0.283)(50/85)(25/85)(10/85) = 0.006
# 
# The higher positive class rating indicates the text can be classified in the positive category.

# In[ ]:


# Concatenate the test columns and convert to a document term matrix.
test_txt = []
for row in range(0, len(X_test.index)):
    test_txt.append(' '.join(str(x) for x in X_test.iloc[row,1:26]))

dtm_A_test = vect_A.transform(test_txt)


# In[ ]:


# Confirm train and test set sizes before modeling.
print('Train features: ', dtm_A_train.shape)
print('Train target: ', y_train.shape)
print('Test features: ', dtm_A_test.shape)
print('Test target: ', y_test.shape)


# ## B. Models

# Run and compare the accuracy of different classification models.

# ### 1. Naive Bayes

# In[ ]:


# Fit the model.
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(dtm_A_train, y_train)

# Make predictions.
pred_B1 = mnb.predict(dtm_A_test)

# Classification report.
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_B1))


# ### 2. Logistic Regression

# In[ ]:


# Fit the model.
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(dtm_A_train, y_train)

# Make predictions.
pred_B2 = logr.predict(dtm_A_test)

# Classification report.
print(classification_report(y_test, pred_B2))


# ### 3. Random Forest

# In[ ]:


# Fit the model.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(dtm_A_train, y_train)

# Make predictions.
pred_B3 = rfc.predict(dtm_A_test)

# Classification report.
print(classification_report(y_test, pred_B3))


# ### 4. Gradient Boost

# In[ ]:


# Fit the model.
from sklearn import ensemble
gbc = ensemble.GradientBoostingClassifier(n_estimators=500, max_depth=2, loss='deviance')
gbc.fit(dtm_A_train, y_train)

# Make predictions.
pred_B4 = gbc.predict(dtm_A_test)

# Classification report.
print(classification_report(y_test, pred_B4))


# ### 5. KNN

# In[ ]:


# Fit the model.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
knn.fit(dtm_A_train, y_train)

# Make predictions.
pred_B5 = knn.predict(dtm_A_test)

# Classification report.
print(classification_report(y_test, pred_B5))


# ## C. Modified Models

# Run modifications on the models to determine if accuracy can be improved. To save on computation costs, only the top performing model(s) from the preceding section will be modified.

# ### 1. TfidfVectorizer
# 
# TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer, which transforms a count matrix to a normalized tf or tf-idf representation. Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting scheme in information retrieval that has also found good use in document classification. The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

# In[ ]:


# Set parameters.
vect_C1 = TfidfVectorizer()

# Fit the TfidfVectorizer and transform the train set.
dtm_C1_train = vect_C1.fit_transform(train_txt)
print('Shape: ', dtm_C1_train.shape, '\n')

# Fit the gradient boost model.
mnb.fit(dtm_C1_train, y_train)

# Transform the test set.
dtm_C1_test = vect_C1.transform(test_txt)

# Make predictions.
pred_C1 = mnb.predict(dtm_C1_test)

# Classification report.
print(classification_report(y_test, pred_C1))


# ### 2. SelectKBest
# 
# Use an automated feature selection method that narrows down the features from the dense matrix version of the original CountVectorizer. The resulting features, which will be the most correlated to the target, will then be fed into the model.

# In[ ]:


# Convert the train and test document term matrices to their dense matrix versions.
df_C2_train = pd.DataFrame(dtm_A_train.toarray(), columns=vect_A.get_feature_names())
df_C2_test = pd.DataFrame(dtm_A_test.toarray(), columns=vect_A.get_feature_names())

# Run Pipeline to initiate SelectKBest with the train set, which is carried over to the test set.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
mnb_pipe = Pipeline([('reducer', SelectKBest(chi2, k=int(1/16*(df_C2_train.shape[1])))), ('clf', mnb)])

# Fit, predict and print classification report.
mnb_pipe.fit(df_C2_train, y_train)
pred_C2 = mnb_pipe.predict(df_C2_test)
print(classification_report(y_test, pred_C2))


# ### 3. Grid Search
# 
# GridSearchCV is a hyper-parameter fine tuning technique that exhaustively considers and searches for the optimum parameter combination, which are passed as the model's arguments. 

# #### a. Naive Bayes

# In[ ]:


# Set the range or values of the parameters the GridSearchCV will iterate over.
from sklearn.model_selection import GridSearchCV
param_mnb = {'alpha': [0.1, 1.0],
             'fit_prior': ['True', 'False']}
gs_mnb = GridSearchCV(mnb, param_mnb, cv= 5)
gs_mnb.fit(dtm_A_train, y_train)
print(gs_mnb.best_params_)


# The optimum combination is the model's default settings, therefore, no further improvements can be made.

# #### b. TfidfVectorizer

# In[ ]:


# Set parameter range.
pipe_tfidf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', mnb)])
param_tfidf = {'tfidf__min_df': [0.01, 0.05],
               'tfidf__max_df': [0.95, 0.99],
               'tfidf__ngram_range': [(1, 1), (1, 2), (2, 1), (2, 2)]}
gs_tfidf = GridSearchCV(pipe_tfidf, param_tfidf, cv=5)
gs_tfidf.fit(train_txt, y_train)
print (gs_tfidf.best_params_)


# In[ ]:


# Run the TfidfVectorizer with the new parameters.
vect_C3 = TfidfVectorizer(min_df=0.05, max_df=0.95, ngram_range = (2, 2))
dtm_C3_train = vect_C3.fit_transform(train_txt)
print('Shape: ', dtm_C3_train.shape, '\n')
mnb.fit(dtm_C3_train, y_train)
dtm_C3_test = vect_C3.transform(test_txt)
pred_C3 = mnb.predict(dtm_C3_test)
print(classification_report(y_test, pred_C3))


# The updated parameters reduced the tokens from almost 30,000 to over 280, a 99% decrease, but the model's accuracy barely improved.

# ## D. Conclusion

# Observations:<br>
# Models performed no better than random guessing.<br>
# Modifications didn't do much to improve accuracy.<br>
# Practical implication: TfidfVectorizer was 100% accurate on positive market movement.
# 
# For improvement:<br>
# Conduct a more exhaustive grid search.<br>
# Design additional features that are specifically suited to the problem.<br>
# Use a more exhaustive stopwords list.<br>
# Normalize the corpus.<br>
# 
# For further study:<br>
# Investigate a date + n approach, where n is the number of days or periods.<br>
# Use multiple sources of news.

# In[ ]:




