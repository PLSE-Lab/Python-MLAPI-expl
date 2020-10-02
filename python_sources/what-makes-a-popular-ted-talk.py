#!/usr/bin/env python
# coding: utf-8

# # What Makes a Popular TED Talk?
# 
# The project intends to predict whether a TED talk will be popular or not.
# 
# (Recent update: Model 3 - Averaging random forest model and Naive Bayes model.)

# In[ ]:


# Loading data
import numpy as np
import pandas as pd
import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

ted = pd.read_csv("../input/ted_main.csv")
transcripts = pd.read_csv('../input/transcripts.csv')
print (ted.shape, transcripts.shape)


# In[ ]:


ted.head()


# Among 2550 talks in the dataset, some are in fact not TED or TEDx events (for example, there is a video filmed in 1972, even before TED is established). They will be removed in this study

# In[ ]:


# Categorize events into TED and TEDx; exclude those that are non-TED events
ted = ted[ted['event'].str[0:3]=='TED'].reset_index()
ted.loc[:,'event_cat'] = ted['event'].apply(lambda x: 'TEDx' if x[0:4]=='TEDx' else 'TED')

print ("No. of talks remain: ", len(ted))


# Here, we change the Unix timstamp to human readable date format. Then we extract month and day of week from film date and published date.

# In[ ]:


ted['film_date'] = ted['film_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
ted['published_date'] = ted['published_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
ted['film_month'] = ted['film_date'].apply(lambda x: x.month)
ted['pub_month'] = ted['published_date'].apply(lambda x: x.month)
ted['film_weekday'] = ted['film_date'].apply(lambda x: x.weekday()) # Monday: 0, Sunday: 6
ted['pub_weekday'] = ted['published_date'].apply(lambda x: x.weekday())
ted[['film_date','published_date']].head()


# TED users can give ratings to each talk. There are 14 possible ratings and they will be categorized
# as positive, negative and neutral:
# -  Positive: 'Beautiful', 'Courageous', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', 'Persuasive'
# -  Negative: 'Confusing', 'Longwinded', 'Obnoxious', 'Unconvincing'
# - Neutral: 'OK'
# 
# Here, we define a "popular" TED talk by its ratio of positive to negative ratings (which we call it "popularity ratio" here). If the popularity ratio is above 5, it is defined as "Popular", otherwise it is "Not Popular". Transformation is made to avoid "divided by zero" error. The following code is adopted from [this kernel](https://www.kaggle.com/allpower/are-co-presented-talks-more-successful) to convert 'ratings' column (a JSON object) into columns of each rating

# In[ ]:


ted['ratings']=ted['ratings'].str.replace("'",'"')
ted=ted.merge(ted.ratings.apply(lambda x: pd.Series(pd.read_json(x)['count'].values,index=pd.read_json(x)['name'])), 
            left_index=True, right_index=True)


# In[ ]:


Positive = ['Beautiful', 'Courageous', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', 'Persuasive']
Negative = ['Confusing', 'Longwinded', 'Obnoxious', 'Unconvincing']
ted['positive']=ted.loc[:,Positive].sum(axis=1)+1
ted['negative']=ted.loc[:,Negative].sum(axis=1)+1
ted['pop_ratio']=ted['positive']/ted['negative']
ted.loc[:,'Popular'] = ted['pop_ratio'].apply (lambda x: 1 if x >5 else 0)

print ("No. of Not Popular talks: ", len(ted[ted['Popular']==0]))
# print ("Ratio of Popular talks: {:.4f}".format(len(ted[ted['Popular']==1])/ float(len(ted))))
overall_mean_popular = np.mean(ted.Popular)
print ("Ratio of Popular talks: {:.4f}".format(overall_mean_popular))


# # Initial Exploratory Data Analysis
# 
# We first plot relationships between several numerical variables.

# In[ ]:


nums = ['comments', 'duration', 'languages', 'num_speaker', 'views']
sns.pairplot(ted, vars=nums, hue='Popular', hue_order = [1,0], diag_kind='kde', size=3);


# Talks with large number of views (10 million or above) are all popular.
# 
# Then we look at corrleation between different ratings:

# In[ ]:


ratings = ['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Longwinded', 'Confusing', 'Informative', 'Fascinating', 'Unconvincing', 
           'Persuasive', 'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring', 'Popular']
plt.figure(figsize=(10,8))
sns.heatmap(ted[ratings].corr(), annot=True, cmap='RdBu');


# Are talks regarded as "Longwinded" have anything to do with time? We plot the ratio of "Longwinded" rating versus talk duration:

# In[ ]:


plt.figure(figsize=(9,6))
lw = ted.Longwinded / (ted.positive + ted.negative)
plt.scatter(ted.duration, lw, s=7)
plt.show()


# Some talks have over 20% rating of "Longwinded". Let's see who they are:

# In[ ]:


lw_talks_id = lw[lw>0.2].index
ted.loc[lw_talks_id,['title','main_speaker','speaker_occupation','event','duration','Longwinded','positive','negative','Popular']]


# Among the 6 talks, two are spoken by Frank Gehry!

# We plot the ratios of popular talks by month, net of overall mean:

# In[ ]:


fm = ted.groupby('film_month')['Popular'].mean().round(4) - overall_mean_popular
pm = ted.groupby('pub_month')['Popular'].mean().round(4) - overall_mean_popular
by_month = pd.concat([fm, pm], axis=1)
by_month.columns = ['Filmed','Published']
by_month.plot(kind='bar', figsize=(9,6))
plt.title('Ratio of Popular Talks by Month (Net of Overall Mean)', fontsize=14)
plt.xticks(rotation=0)
plt.show()


# Talks filmmed in January got the highest ratio being popular while those on February and August are the lowest.

# Ratio of popular talks by day of week:

# In[ ]:


fw = ted.groupby('film_weekday')['Popular'].mean().round(4) - overall_mean_popular
pw = ted.groupby('pub_weekday')['Popular'].mean().round(4) - overall_mean_popular
by_weekday = pd.concat([fw, pw], axis=1)
by_weekday.columns = ['Filmed', 'Published']
by_weekday.index = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']
by_weekday.plot(kind='bar', figsize=(9,6))
plt.title('Ratio of Popular Talks by Day of Week', fontsize=14)
plt.xticks(rotation=0)
plt.show()


# Talks filmmed on Saturday has a notably lower ratio of being popular, while those published on Sunday have a higher proportion of being popular.
# 
# Let's look at relationshp between number of speakers and popularity:

# In[ ]:


x = ted.groupby('num_speaker').mean()['Popular'].rename('pct_popular')
pd.concat([x, ted.num_speaker.value_counts().rename('talks_count')], axis=1)


# Only a tiny number of talks are delivered by 3 or more speakers. Their ratio of popular talks are not that indicative.

# We then use count vectorizer on tags. Only those occur 20 times or more will be kept:

# In[ ]:


count_vector = CountVectorizer(stop_words='english',min_df=20/len(ted)) # Only keep those with 20 or more occurrences
tag_array = count_vector.fit_transform(ted.tags).toarray()
tag_matrix = pd.DataFrame(tag_array, columns = count_vector.get_feature_names())
all_tags = tag_matrix.columns
tag_matrix = pd.concat([tag_matrix, ted.Popular], axis=1)
by_tag = dict()
for col in all_tags:
    by_tag[col]=tag_matrix.groupby(col)['Popular'].mean()[1] - overall_mean_popular
tag_rank = pd.DataFrame.from_dict(by_tag, orient='index')
tag_rank.columns = ['pop_rate_diff']

plt.figure(figsize=(16,7))
plt.subplot(121)
bar_2 = tag_rank.sort_values(by='pop_rate_diff', ascending=False)[:15]
sns.barplot(x=bar_2.pop_rate_diff, y=bar_2.index, color='blue')
plt.title('15 Most Popular Tags')
plt.xlabel('Ratio of Popular Talk (Net of Mean)', fontsize=14)
plt.yticks(fontsize=12)
plt.subplot(122)
bar_1 = tag_rank.sort_values(by='pop_rate_diff')[:15]
sns.barplot(x=bar_1.pop_rate_diff, y=bar_1.index, color='red')
plt.title('15 Most Unpopular Tags')
plt.xlabel('Ratio of Popular Talk (Net of Mean)', fontsize=14)
plt.yticks(fontsize=12)
plt.show()


# Findings:
# - Talks with tags 'physiology', 'empathy', 'success', 'sense', and 'water' are all "Popular";
# - 'cars' is the tag with lowest proportion of "Popular" talks, followed by 'industrial', 'security', and 'religion'

# Then we do count vectorizer on 'speaker_occupation'. Before that, some data cleaning is needed.

# In[ ]:


ted.loc[:,'occ'] = ted.speaker_occupation.copy()
ted.occ = ted.occ.fillna('Unknown')
ted.occ = ted.occ.str.replace('singer/songwriter', 'singer, songwriter')
ted.occ = ted.occ.str.replace('singer-songwriter', 'singer, songwriter')
count_vector2 = CountVectorizer(stop_words='english', min_df=20/len(ted))
occ_array = count_vector2.fit_transform(ted.occ).toarray()
occ_matrix = pd.DataFrame(occ_array, columns = count_vector2.get_feature_names())
all_occ = occ_matrix.columns
occ_matrix = pd.concat([occ_matrix, ted.Popular], axis=1)
by_occ = dict()
for col in all_occ:
    by_occ[col]=occ_matrix.groupby(col)['Popular'].mean()[1] - overall_mean_popular
occ_rank = pd.DataFrame.from_dict(by_occ, orient='index')
occ_rank.columns = ['pop_rate_diff']

plt.figure(figsize=(16,7))
plt.subplot(121)
bar_2 = occ_rank.sort_values(by='pop_rate_diff', ascending=False)[:10]
sns.barplot(x=bar_2.pop_rate_diff, y=bar_2.index, color='blue')
plt.title('10 Most Popular Occupation Keywords', fontsize=14)
plt.xlabel('Ratio of Popular Talk (Net of Mean)')
plt.yticks(fontsize=12)
plt.subplot(122)
bar_1 = occ_rank.sort_values(by='pop_rate_diff')[:10]
sns.barplot(x=bar_1.pop_rate_diff, y=bar_1.index, color='red')
plt.title('10 Most Unpopular Occupation Keywords', fontsize=14)
plt.xlabel('Ratio of Popular Talk (Net of Mean)')
plt.yticks(fontsize=12)
plt.show()


# Findings:
# - Most popular occupation keywords are "neuroscientist", "health", and "poet";
# - Lease popular occupation keywords are "theorist", "economist", and "futurist"

# We then visualize Popular and Unpopular talks by word clouds:

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist, word_tokenize
stopwords = set(STOPWORDS)

plt.figure(figsize=(15,12))

plt.subplot(121)
word_pos = FreqDist(w for w in word_tokenize(' '.join(ted.loc[ted.Popular==1, 'title']).lower()) if (w not in stopwords) & (w.isalpha()))
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_pos)
plt.imshow(wordcloud)
plt.title("Wordcloud for Title - Popular", fontsize=16)
plt.axis("off")

plt.subplot(122)
word_neg = FreqDist(w for w in word_tokenize(' '.join(ted.loc[ted.Popular==0, 'title']).lower()) if (w not in stopwords) & (w.isalpha()))
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_neg)
plt.imshow(wordcloud)
plt.title("Wordcloud for Title - Unpopular", fontsize=16)
plt.axis("off")
plt.show()


# "Design" and "music" are notably big in unpopular talks' word cloud.
# 
# Then we do the same for description. A new set of stopwords is used. Lemmatization is used as well:

# In[ ]:


from nltk.stem import WordNetLemmatizer
import string

wnl = WordNetLemmatizer()

extrasw = set(['say', 'says', 'talk', 'us', 'world', 'make'])
stopwords2 = stopwords.union(extrasw)

pos_str = ' '.join(ted.loc[ted.Popular==1, 'description']).lower().translate(str.maketrans('','',string.punctuation))
neg_str = ' '.join(ted.loc[ted.Popular==0, 'description']).lower().translate(str.maketrans('','',string.punctuation))

plt.figure(figsize=(15,12))

plt.subplot(121)
word_pos = FreqDist(wnl.lemmatize(wnl.lemmatize(w), pos='v') for w in word_tokenize(pos_str) if w not in stopwords2)
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_pos)
plt.imshow(wordcloud)
plt.title("Wordcloud for Description - Popular", fontsize=16)
plt.axis("off")

plt.subplot(122)
word_neg = FreqDist(wnl.lemmatize(wnl.lemmatize(w), pos='v') for w in word_tokenize(neg_str) if w not in stopwords2)
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_neg)
plt.imshow(wordcloud)
plt.title("Wordcloud for Description - Unpopular", fontsize=16)
plt.axis("off")
plt.show()


# Findings:
# - "Share" is a much bigger keyword in Popular talks than in Unpopular ones
# - On the other hand, "show" and "new" are much bigger in Unpopular word clouds than in Popular ones

# # Building a Prediction Model
# 
# We will build a prediction model to prediction which talks are popular, starting with tags, speaker occupations, title and description. As we are having a problem of highly unbalanced class, we only predict a talk to be 'Popular' if the probability of having a 'Popular' label is above 65%.

# ### Model 1: Tags and speaker occupation, random forest model

# In[ ]:


y = ted.Popular
x = pd.concat([occ_matrix.drop('Popular', axis=1), tag_matrix.drop('Popular', axis=1)], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=36)


# In[ ]:


# Write function on training and testing
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from time import time

def train_predict(clf, parameters):
        
    clf.fit(x_train, y_train)
    scorer = make_scorer(fbeta_score, beta=1)

    # 5-fold cross validation
    start = time()

    grid_obj = GridSearchCV(clf, parameters, cv=5, scoring=scorer)
    grid_fit = grid_obj.fit(x_train, y_train)
    best_clf = grid_fit.best_estimator_
    best_prob_train = best_clf.predict_proba(x_train)
    best_prob = best_clf.predict_proba(x_test)
    best_pred_train = (best_prob_train[:,1]>0.65)*1
    best_pred = (best_prob[:,1]>0.65)*1

    end = time()

    run_time = end - start

    # Report results
    print (clf.__class__.__name__ + ":")
    print ("Accuracy score on training data (optimized by grid-search CV): {:.4f}".format(best_clf.score(x_train, y_train)))
    print ("Accuracy score on testing data (optimized by grid-search CV): {:.4f}".format(best_clf.score(x_test, y_test)))
    print ("F1-score on training data (optimized by grid-search CV): {:.4f}".format(fbeta_score(y_train, best_pred_train, beta = 1)))
    print ("F1-score on testing data (optimized by grid-search CV): {:.4f}".format(fbeta_score(y_test, best_pred, beta = 1)))
    print ("Parameters: ", grid_fit.best_params_)
    # print (confusion_matrix(y_test, best_predictions))
    print ("Total runtime: {:.4f} seconds".format(run_time))
    return best_prob_train, best_prob, best_pred_train, best_pred


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 108)
parameters = {'n_estimators': range(10,21,2), 'max_features': ['auto', 'log2'], 'min_samples_split': range(3, 7)}
rf_prob_train, rf_prob, rf_pred_train, rf_pred = train_predict(clf, parameters)
# train_predict(clf, parameters)


# In[ ]:


pd.DataFrame(confusion_matrix(y_train, rf_pred_train))


# In[ ]:


pd.DataFrame(confusion_matrix(y_test, rf_pred))


# ### Model 2: Title and description, Naive Bayes

# In[ ]:


from scipy.sparse import hstack

y = ted.Popular

cv_t = CountVectorizer(stop_words='english', max_features=10000, lowercase=True) # Title
cv_d = CountVectorizer(stop_words='english', max_features=1000, lowercase=True) # Description
x_t = cv_t.fit_transform(ted.title)
x_d = cv_d.fit_transform(ted.description)
x_all = hstack([x_t, x_d])

x_train, x_test, y_train, y_test = train_test_split(x_all, y, test_size=0.25, random_state=36)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

clf2 = MultinomialNB()
parameters={}
nb_prob_train, nb_prob, nb_pred_train, nb_pred = train_predict(clf2, parameters)


# In[ ]:


pd.DataFrame(confusion_matrix(y_train, nb_pred_train))


# In[ ]:


pd.DataFrame(confusion_matrix(y_test, nb_pred))


# ### Model 3: Averaging Model 1 and 2

# In[ ]:


m3_prob_train = (rf_prob_train + nb_prob_train)/2
m3_prob = (rf_prob + nb_prob)/2
m3_pred_train = (m3_prob_train[:,1]>0.65)*1
m3_pred = (m3_prob[:,1]>0.65)*1


# In[ ]:


from sklearn.metrics import accuracy_score
print ("Model 3:")
print ("Accuracy score on training data (optimized by grid-search CV): {:.4f}".format(accuracy_score(y_train, m3_pred_train)))
print ("Accuracy score on testing data (optimized by grid-search CV): {:.4f}".format(accuracy_score(y_test, m3_pred)))
print ("F1-score on training data (optimized by grid-search CV): {:.4f}".format(fbeta_score(y_train, m3_pred_train, beta = 1)))
print ("F1-score on testing data (optimized by grid-search CV): {:.4f}".format(fbeta_score(y_test, m3_pred, beta = 1)))


# In[ ]:


pd.DataFrame(confusion_matrix(y_train, m3_pred_train))


# In[ ]:


pd.DataFrame(confusion_matrix(y_test, m3_pred))


# Stay tuned for more analysis!
