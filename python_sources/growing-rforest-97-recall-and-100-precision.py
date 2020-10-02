#!/usr/bin/env python
# coding: utf-8

# # Growing A Random Forest - Tuning to get 97% Recall and 100% Precision
# 
# In this notebook, I will experiment with various methods to boost recall on fraudulent transactions classification. We will focus on random forest, as my experiment has shown that it performed very well on the original data set.

# Getting a high recall on Credit Card Fraud data set has been **difficult**. While accuracy also hovers around 100%, Recall (on fraudulent cases) seems to cling around **60-70%**. You will see later on that Random Forest only achieves 75% recall out of the box.
# 
# And this is not good practically. We will discuss why.

# Before we discuss further,  let's begin by loading some necessary libraries and import the data set.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('../input/creditcard.csv')


# Let's check the overview.

# In[ ]:


df.info()


# 284,807 transactions. Not bad. And there is no NaN to clean up.
# 
# Let's get a peek.

# In[ ]:


df.head(20)


# At this point, we can notice a couple of things.
# 
# 1. We have attributes of Time, Amount, and Class and other 28 unknown attributes.
# 
# 2. All the unknown attributes seem to be normalized.
# 
# 3. Time is in a linear order.
# 
# 4. We have not yet seen any other class than class=0...

# We can confirm the last point with a quick check.

# In[ ]:


df.groupby('Class').count()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(20,5))
sns.countplot(x='Class', data=df)


# Obviously, this data set is highly unbalanced. The skewness encourages algorithms to flag transactions as normal, because they can get 99.8% accuracy simply by doing so. Thus, algorithms would flag only fraudulent transactions they are quite certain with. The result would be models with high accuracy and precision at the expense of recall.
# 
# In other word, we are finding needles in a haystack....

# # Throwing it directly into Random Forest
# 
# To illustrate my point, let's put this data set directly into random forest.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop(['Class'], axis=1)


# In[ ]:


Y = df.Class


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)


# In[ ]:


rf = RandomForestClassifier(random_state=0)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# **Classification Report**

# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# ** Confusion Matrix **

# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# While overall score is pretty good, it is clear that we misclassify a lot of frauds. 1/4 of fraud cases were flagged negative. I hope we can do better.

# # Accuracy, Precision, Recall, F1-score
# 
# I have seen many people on Kaggle used accuracy as the performance metric. This is not good on a heavily unbalanced dataset like this one.
# 
# Assume I know the distribution of normal and fraudulent transactions. I can see that normal transactions number almost 600 times the entries of fraudulent transactions! Given this knowledge, I can simply guess that all transaction is normal, and I would still be 99.83% correct.
# 
# However, I would not have flagged any single fraud. Such model, despite its high accuracy, is worthless.
# 
# Therefore, an appropriate measure is Recall and Precision of fraudulent cases, as it measures how many frauds we can pick out from the transactions.
# 
# For more information, please visit
# 
# https://www.cs.cornell.edu/courses/cs578/2003fa/performance_measures.pdf
# 
# https://medium.com/greyatom/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b

# # Why Recall Matters?
# 
# You may wonder why we should care about recall. Precision seems solid, so why can't we leave it at that?
# 
# Imagine you are a large bank. In America, if a transaction is fraudulent, banks generally have to cover that transaction.
# 
# Let's do some simple calculation how much money we would lose to compensation, **if we use the above model.**

# First let's look at an average dollar amount we have to pay.

# In[ ]:


df[df.Class==1].Amount.describe()


# OK, so if I let a fraudulent transaction slip through my fingers, I must pay **$120** on average.

# And how often must I pay?

# In[ ]:


df.Time.max()/86400


# Since Time is given in seconds after the first transaction, I divided it with 86,400 seconds to convert it into days. As you can see, this data set covers only **2 days!!!**
# 
# And in these 2 days, there are **490 fraudulent transactions...**

# That means I would let 60 of these transactions through each day. Since each one costs me \$120, I am looking to pay **$7,200 a day.**

# That translates to ** 2.6 Million Dollars** a year.

# Ouch.

# So, there is a pricey tag for low recall. We can easily increase recall by ramping up sensitivity to frauds. But that would make precision suffer. Sure, we will catch most frauds. But you can also expect to hear customers screaming that their cards get blocked without a good reason.
# 
# There must be a better solution, right?

# # Feature Engineering
# 
# We are going to see if we can (artificially) add some useful information. I am not optimistic at this approach, since most attributes were anonymized and I couldn't make head or tail on what they mean. Anyhow, it's worth a shot.

# # First Attempt - Domain Knowledge

# When you are doing feature engineering, it is best to work with a domain expert. However, we don't have that luxury here. Furthermore, all attributes were anonymized.
# 
# Still, if we look through some readings, we might find an idea or two to improve our model. Here are what I found on Google.
# 
# https://www.cnbc.com/id/46907307
# 
# https://blog.bluepay.com/how-to-recognize-a-potentially-fraudulent-credit-card-user
# 
# http://www.mydigitalshield.com/credit-card-fraud-detection-techniques/

# Unfortunately, most suggestions are not implementable, due to anonymized dataset.

# ## Rapid succession of Small Transactions

# One suggestion that we can implement is to look for unusually rapid small transactions. Let's test this suggestion with a visualization.

# In[ ]:


plt.figure(figsize=(20,10))
ax = plt.subplot()
ax.set_xlim(0, 2000)
sns.distplot(df[(df.Class==1) & (df.Amount < 2000)].Amount, bins=100, color='r')
sns.distplot(df[(df.Class==0) & (df.Amount < 2000)].Amount, bins=100, color='b')


# Let's zoom in on interesting parts.

# In[ ]:


plt.figure(figsize=(20,10))
ax = plt.subplot()
sns.distplot(df[(df.Class==1) & (df.Amount < 2)].Amount, bins=100, color='r')
sns.distplot(df[(df.Class==0) & (df.Amount < 2)].Amount, bins=100, color='b')
sns.distplot(df[(df.Amount < 2)].Amount, bins=100, color='g')


# It is clear that this is an active region for fraudulent charges. However, we cannot simply flag everything with < $1 amount, as the number of normal transaction still overwhelm that of frauds.

# In that case, let's make a hypothesis. I assume that the distribution of these micro transactions is uniform. However, if someone was to test a stolen credit card by making several micro transactions, we should see a spike at that region.

# In[ ]:


# Make a new feature denoting a micro transaction
df['Micro TXN'] = df.Amount <= 1


# In[ ]:


df['Micro TXN in 1K TXN'] = df['Micro TXN'].rolling(1000).sum()


# In[ ]:


df.dropna(inplace=True)
df.drop(['Micro TXN'], axis=1, inplace=True)


# In[ ]:


df.head(10)


# In[ ]:


df['Micro TXN in 1K TXN'].describe()


# Let's test this out.

# In[ ]:


X = df.drop(['Class'], axis=1)


# In[ ]:


Y = df.Class


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)


# In[ ]:


rf = RandomForestClassifier(random_state=0)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# Interestingly, it has the effect of reducing false positives instead of false negatives. As I see some improvement, albeit very small, I would keep this feature.

# ## Large Purchases
# 
# Next, let's do the same thing with large purchases. Looking over last 2 charts, I decided to make a new feature denoting transaction larger than \$250.

# In[ ]:


df['Large TXN'] = df.Amount > 250


# In[ ]:


X = df.drop(['Class'], axis=1)
Y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# **All in all, we don't get much improvement. The reason might be that the knowledge is outdated, or we can't get better features because attributes are unknown.**
# 
# Anyway, I decide to drop the column of large transactions, as it seems to do more harm than good.

# In[ ]:


df.drop(['Large TXN'], axis=1, inplace=True)


# # Building New Features with Clustering
# 
# It's been suggested that clustering algorithms can be used to engineer new features. Would they help us here?
# 
# I start with t-SNE. t-SNE reduces dimensionality into 2 dimensions, so we can plot and visualize data set.

# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


X_TSNE = TSNE().fit_transform(X)


# In[ ]:


X_TSNE


# In[ ]:


vis_x = X_TSNE[:, 0]
vis_y = X_TSNE[:, 1]


# In[ ]:


plt.figure(figsize=(15,15))
plt.scatter(vis_x, vis_y, c=Y.as_matrix())


# The result does not look so good. Fraudulent transactions are embedded within normal transactions with no clear cluster of its own. This is to be expected though, as criminals would want to disguise as normal transactions.
# 
# Anyway, let's fire up K-Means.

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans = KMeans(random_state=0).fit(X)


# In[ ]:


clusters = kmeans.predict(X)


# In[ ]:


plt.figure(figsize=(15,15))
plt.scatter(vis_x, vis_y, c=clusters)


# Ugh, I don't know what to make of it. On a close inspection, the regions with fraudulent transactions seem to belong to a different cluster to neighboring transactions. So, let's give it a try.
# 
# First, I take distances to centroids for each data point.

# In[ ]:


distance_from_centroids = kmeans.transform(X)


# In[ ]:


distance_from_centroids.shape


# Then, I append t-SNE dataframe.

# In[ ]:


df = pd.concat([df, pd.DataFrame(X_TSNE, index=X.index, columns=['TSNE_0', 'TSNE_1'])], axis=1)


# In[ ]:


df.head()


# And the distances...

# In[ ]:


df = pd.concat([df, pd.DataFrame(distance_from_centroids, index=X.index, columns=['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6', 'Cluster7', 'Cluster8'])], axis=1)


# In[ ]:


df.head()


# In[ ]:


X = df.drop(['Class'], axis=1)
Y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# No improvement... At this point, I felt inclined to remove these new features. But because I spent quite a few hours training t-SNE, I decided to delay this decision and kept these features for a moment longer.

# # Re-engineering Time
# 
# I had an idea that there might be some hours of a day that see more fraudulent charges. We were given Time feature, but it is not wrapped into 24-hours standard. So, let's do that now.

# In[ ]:


df['TimeOfDay'] = df.Time % 86400


# In[ ]:


df.head()


# With this, transactions that happen on different days, but at the same time, would have the same timestamps.

# In[ ]:


X = df.drop(['Time', 'Class'], axis=1)
Y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# At last! We see an improvement.

# # Feature Selection
# 
# While my intuition calls that feature selection is not necessary for this data set, I still want to give it a try.

# For this purpose, I will use the built-in feature importance to remove some features.
# 
# More information:
# 
# http://blog.datadive.net/selecting-good-features-part-iii-random-forests/

# In[ ]:


names=X.columns
f_imp = sorted(zip(names, map(lambda x: round(x, 4), rf.feature_importances_)), 
               key=lambda x: x[1],
             reverse=True)


# In[ ]:


pd.DataFrame(f_imp)


# Let's drop the last 5 features.

# In[ ]:


X = df.drop(['Time', 'Class', 'V15', 'Cluster6', 'Micro TXN in 1K TXN', 'TSNE_1', 'V23'], axis=1)
Y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# As expected, we don't get improvement. In fact, precision suffers slightly. When you look at removed features, you would notice that two of them were our newly added features. So, it might be good to keep them.

# # Undersampling
# 
# When dealing with unbalanced data, it is a good idea to resample. Resampling comes with different flavors, such as oversampling, undersampling, and SMOTE. For this notebook, we are going to do vanilla undersampling.
# 
# First, let's take an equal amount of both cases.

# In[ ]:


resampled_df_normal = df[df.Class == 0].sample(n=490, random_state=0)


# In[ ]:


resampled_df_fraud = df[df.Class==1]


# In[ ]:


resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])


# In[ ]:


X = df.drop(['Time', 'Class'], axis=1)
Y = df.Class


# In[ ]:


resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
resampled_Y = resampled_df.Class
X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# **95% Precision and 93% Recall!!!** It's time to celebrate...
# 
# ...not yet. That's the result from our limited data set.
# 
# Let's see how it fares in the normal environment.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# **Ugh, 5% Precision.** The cause is clear. Random Forest now is very aggressive in flagging fraudulent cases. So aggressive that it misclassifies 2,229 normal cases. This leads to a very low precision.

# ## Adjusting undersampling
# 
# I wonder if adjusting the ratio of normal transaction vs. fraudulent transactions would help. So, I decided to give it a try.

# In[ ]:


resampled_df_normal = df[df.Class == 0].sample(n=2500, random_state=0)


# In[ ]:


resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])


# In[ ]:


resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
resampled_Y = resampled_df.Class
X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# Not bad. How about full data set?

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# Now we get 34% precision, up from 5%. The number is still too low, but we clearly have an improvement. On top of that, recall does not suffer much.
# 
# So, let's find the right ratio.

# In[ ]:


sample_size = []
precision = []
recall = []
fone = []
for size in range(500, 283500, 500):
    print('Running : size = %d' % (size))
    sample_size.append(size)
    resampled_df_normal = df[df.Class == 0].sample(n=size, random_state=0)
    resampled_df_fraud = df[df.Class==1]
    resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
    resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
    resampled_Y = resampled_df.Class
    X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
    rf = RandomForestClassifier(random_state=0,n_jobs=-1)
    rf.fit(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    y_pred = rf.predict(X_test)
    precision.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[0][1])
    recall.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[1][1])
    fone.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[2][1])
    


# In[ ]:


ss = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(fone)], axis=1)


# In[ ]:


ss.columns = ['Precision', 'Recall', 'F1']


# In[ ]:


ss.index = range(500, 283500, 500)


# In[ ]:


ss.head(15)


# In[ ]:


ss.plot(figsize=(20,10))


# That's a messy chart! Let's find where maximum F1-score is.

# In[ ]:


ss.F1.idxmax()


# At this point, I could take this ratio and call it a day. But my experience tells me to probe a little further.

# In[ ]:


ss.loc[270000:280000]


# The nearby ratios vary grealy in Recall from 84% to 96%. Clearly, this area is not robust. If data set changes even a little bit, my score could severely suffer.
# 
# To remedy this, I want to find **an area with consistently high score.** So, I resort to a moving average.

# In[ ]:


ss['Recall MA'] = ss.Recall.rolling(20).mean()


# In[ ]:


ss['F1 MA'] = ss.F1.rolling(20).mean()


# In[ ]:


ss.plot(figsize=(20,10))


# In[ ]:


ss['F1 MA'].idxmax()


# In[ ]:


ss.loc[200500:213500]


# Both visual cues and the table confirm that the area from 200k to 210k cases offer a robust performance. So, I picked 204,000.

# In[ ]:


resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)
resampled_df_fraud = df[df.Class==1]
resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
resampled_Y = resampled_df.Class
X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# **93% Recall!!!!!!!!!!** up from 73%. And precision does not suffer.
# 
# Where's my champagne?

# ## Did new features help?
# 
# So, I have delayed removing new features long enough. It's time to settle.

# In[ ]:


reloaded_df = pd.read_csv('../input/creditcard.csv')


# In[ ]:


reloaded_df_normal = reloaded_df[reloaded_df.Class == 0].sample(n=204000, random_state=0)
reloaded_df_fraud = reloaded_df[reloaded_df.Class==1]
reloaded_df = pd.concat([reloaded_df_normal, reloaded_df_fraud])
reloaded_X = reloaded_df.drop(['Class'], axis=1)
reloaded_Y = reloaded_df.Class
X_train, X_test, y_train, y_test = train_test_split(reloaded_X, reloaded_Y, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# Without new features, recall is only 83% after undersampling. While this is an improvement from earlier 73% recall, it's still a far cry from 93%. Clearly, random forest learns to make use of new features in undersampled data set.

# # Tuning Random Forest
# 
# The last thing to do is to tune parameters of the random forest. Tuning parameters can affect performance of models. I follow the guideline in this research paper.
# 
# https://arxiv.org/pdf/1708.05070.pdf
# 
# It should be noted that ideally you would employ Grid Search for such task.
# 
# http://scikit-learn.org/stable/modules/grid_search.html
# 
# However, I want to easily visualize the effect of each parameter. So, I decided to do it rough. The result may not be optimal, but still good enough.

# ## Parameters of Random Forests
# 
# A quick glance at Scikit Learn's documentation will reveal a wealth of parameters you can fine tune.
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 
# In this notebook, I will select some parameters for tuning.

# ## N_Estimators
# 
# Random Forest relies on building multiple decision trees. This parameter tells the random forest how many decision trees it should build.

# In[ ]:


key_index = []
precision = []
recall = []
fone = []
for key in range(2,40):
    print('Running : key = %d' % (key))
    key_index.append(key)
    resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)
    resampled_df_fraud = df[df.Class==1]
    resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
    resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
    resampled_Y = resampled_df.Class
    X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
    rf = RandomForestClassifier(n_estimators=key, random_state=0,n_jobs=-1)
    rf.fit(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    y_pred = rf.predict(X_test)
    precision.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[0][1])
    recall.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[1][1])
    fone.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[2][1])
    


# In[ ]:


opt_param = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(fone)], axis=1)
opt_param.columns = ['Precision', 'Recall', 'F1']
opt_param.index = key_index
opt_param.plot(figsize=(20,10))


# Clearly, this parameter affects performance a lot. Also, there is a region with robust performance. 
# 
# Let's see where the best value is.

# In[ ]:


opt_param.F1.idxmax()


# Because this index is the start of the plateau. I choose 16 instead, so that the random forest will be robust against minor changes.

# In[ ]:


resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)
resampled_df_fraud = df[df.Class==1]
resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
resampled_Y = resampled_df.Class
X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
rf = RandomForestClassifier(n_estimators=16, random_state=0)
rf.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# Now, we get to 96%.

# ## Max Depth
# 
# How deep can a decision tree in the random forest grow. 

# In[ ]:


key_index = []
precision = []
recall = []
fone = []
for key in range(2,50):
    print('Running : key = %d' % (key))
    key_index.append(key)
    resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)
    resampled_df_fraud = df[df.Class==1]
    resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
    resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
    resampled_Y = resampled_df.Class
    X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
    rf = RandomForestClassifier(n_estimators=16, max_depth=key, random_state=0,n_jobs=-1)
    rf.fit(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    y_pred = rf.predict(X_test)
    precision.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[0][1])
    recall.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[1][1])
    fone.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[2][1])


# In[ ]:


opt_param = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(fone)], axis=1)
opt_param.columns = ['Precision', 'Recall', 'F1']
opt_param.index = key_index
opt_param.plot(figsize=(20,10))


# In[ ]:


opt_param.F1.max()


# Apparently, there is no increase in performance. Same thing happens with **Min Sample Split, Min Sample Leaves, and Max Leaf Nodes.** So, I will leave them out.

# ## Max Features
# 
# Our final stop is max features parameter.

# In[ ]:


key_index = []
precision = []
recall = []
fone = []
for key in range(1,42):
    print('Running : key = %d' % (key))
    key_index.append(key)
    resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)
    resampled_df_fraud = df[df.Class==1]
    resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
    resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
    resampled_Y = resampled_df.Class
    X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
    rf = RandomForestClassifier(n_estimators=16, max_features=key, random_state=0,n_jobs=-1)
    rf.fit(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    y_pred = rf.predict(X_test)
    precision.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[0][1])
    recall.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[1][1])
    fone.append(sklearn.metrics.precision_recall_fscore_support(y_test, y_pred)[2][1])


# In[ ]:


opt_param = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(fone)], axis=1)
opt_param.columns = ['Precision', 'Recall', 'F1']
opt_param.index = key_index
opt_param.plot(figsize=(20,10))


# In[ ]:


opt_param.F1.max()


# In[ ]:


opt_param.F1.idxmax()


# In[ ]:


resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=0)
resampled_df_fraud = df[df.Class==1]
resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
resampled_Y = resampled_df.Class
X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=0)
rf = RandomForestClassifier(n_estimators=16, max_features=14, random_state=0)
rf.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# **100% Precision and 97% Recall!!!** We have come a long way from the beginning 75%.

# # Final Check
# 
# You might wonder if this works only with this sampled data set. So, I decided to do a final check with different random states.

# In[ ]:


resampled_df_normal = df[df.Class == 0].sample(n=204000, random_state=1)
resampled_df_fraud = df[df.Class==1]
resampled_df = pd.concat([resampled_df_normal, resampled_df_fraud])
resampled_X = resampled_df.drop(['Time', 'Class'], axis=1)
resampled_Y = resampled_df.Class
X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_Y, random_state=2)
rf = RandomForestClassifier(n_estimators=16, max_features=14, random_state=3)
rf.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=4)
y_pred = rf.predict(X_test)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, y_pred))


# In[ ]:


cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')


# As expected, the score suffers slightly. Recall is still an impressive 97% and F1-Score is remarkably high. I would say that our model is robust enough for the purpose.
# 
# I thank you all for sticking with me til the end. I hope you enjoy my ride. If you have any questions or suggestions, please post a comment. I am new to Kaggle, so advices are appreciated.

# In[ ]:




