#!/usr/bin/env python
# coding: utf-8

# <center> <h1> Math 154: Group Data Project </h1> </center>
# <center> <h2> Give Me Some Credit Kaggle Competition </h2> </center>
# <center> <i>  Seena Huang, Cecelia Sanborn, Harry Bendekgey, Simon Posada Fishman </i> </center>

# In[ ]:


# Main tools we used:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Models:
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Python utilities:
import time
import os


# # Data Exploration
# 
# First let's take a look at what our predictors are:

# In[ ]:


df = pd.read_csv('../input/cs-training.csv')
list(df)


# We also want to know how balanced the data set is, so we check what portion defaulted:

# In[ ]:


df.SeriousDlqin2yrs.mean()


# "If you're over 55 and need a mortgage, the important thing to know if that lenders can't deny you a loan based on your age. But age can factor into your mortgage equation." ([Source](https://www.chicagotribune.com/news/ct-xpm-2005-11-20-0511200488-story.html))
# 
# This should be our first concern: making sure we're in legally okay territory. In order to solve this, we need to confirm that in our data set age has a negative influence on probability of defaulting. Then we can assume that the models we build reflect that trend, and won't deny someone a loan simply because they are old. 

# In[ ]:


age_hist = df['age'].hist(by=df['SeriousDlqin2yrs'], bins=20, layout=(2,1))
age_hist[0].set_xlim((0,100))
age_hist[0].set_title('SeriousDlqin2yrs = 0')
age_hist[1].set_xlim((0,100))
age_hist[1].set_title('SeriousDlqin2yrs = 1')


# Nice! Looking at the distribution of people that defaulted vs those who did not shows us that, generally, younger people were more responsible for defaulting. So we're good to go, legally. 
# 
# Let's take a look at our data. We can start with Debt Ratio, the ratio of debt to assets. 

# In[ ]:


df.DebtRatio.describe()


# It is very concerning that someone owes 330,000 times what they own. Maybe that person is a single outlier? Let's take a look:

# In[ ]:


df.DebtRatio.quantile([.975])


# Apparently 2.5% of the dataset owes more than 3,500 times what they own. We need to investigate further to see if these are outliers or not. We can see:

# In[ ]:


df[df['DebtRatio'] > 3489.025][['SeriousDlqin2yrs','MonthlyIncome']].describe()


# We see two particularly concerning things here. The first is that of 4,000 records with DebtRatio > 3,500, only 185 of them have a value for monthly income. Further, the people who do have monthly income seem to either have a monthly income of either 1 or 0. We see that: 

# In[ ]:


df[(df['DebtRatio'] > 3489.025) & (df['SeriousDlqin2yrs'] == df['MonthlyIncome'])].shape[0]


# ...of those 185 entries, 164 of them have the same value for 2 year default rate and monthly income, indicating that there is a data-entry error. 
# 
# The second problem is that despite owing thousands of times what they own, these people aren't defaulting any more than the general population. We can conclude that these entries must be data-entry errors, so we will remove them from our model.
# 
# Now let's take a look at NumberOfTimes90DaysLate.

# In[ ]:


df.groupby('NumberOfTimes90DaysLate').NumberOfTimes90DaysLate.count()


# It is interesting that no one is between 17 and 96 times late, but hundreds of people are 98 times late. We can take a look at these few hundred records:

# In[ ]:


df[df['NumberOfTimes90DaysLate'] > 95][['SeriousDlqin2yrs','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate']].describe()


# Somehow, all these (300) people were 30-59 days late 96/98 times, 60-89 days late 96/98 times, and 90+ days late 96/98 times. This is really concerning.  However, the data might not be garbage, because (as expected) these people are defaulting at a massive rate (55%) compared to the population (6%). So we don't want to throw away this data. We can Winsorize by replacing all the 96/98s with 20s to make them not extreme outliers and see if that improves the models. We don't expect that to improve the random forests, which are robust to outliers, but it might improve the SVMs.

# Finally, let's take a look at revolving utilization of unsecured lines. It represents the ratio of money owed to credit limit, so it shouldn't get much above 1. First let's look at values close to 1 and see at what rate they're defaulting. We start with 0.9 to 4.0:

# In[ ]:


df[(df['RevolvingUtilizationOfUnsecuredLines'] > .9) & (df['RevolvingUtilizationOfUnsecuredLines'] <= 4)].SeriousDlqin2yrs.describe()


# These 20,000 people are defaulting at a rate of almost 1-in-4. This is important! But what about people with more RUUL? Let's consider 4 to 10:

# In[ ]:


df[(df['RevolvingUtilizationOfUnsecuredLines'] > 4) & (df['RevolvingUtilizationOfUnsecuredLines'] <= 10)].SeriousDlqin2yrs.describe()


# There are only 23 records in this region, but they're still defaulting at a high rate. What about if we go even higher, and consider those with RUUL > 10?

# In[ ]:


df[df['RevolvingUtilizationOfUnsecuredLines'] > 10].describe()


# These 241 people are not defaulting any more than anyone else, despite some of them owing hundreds of thousands of times their credit limits. These seem to be inconsistent with the rest of the data, so we can remove them from our model.
# 
# Finally, let's take a look at using regression to fill in the missing values for MonthlyIncome:

# In[ ]:


not_missing = df.dropna()
target = 'MonthlyIncome'
predictors = [c for c in list(not_missing) if c not in [target, 'Unnamed: 0','SeriousDlqin2yrs']]
X_data = not_missing[predictors]
y_data = not_missing[target]
regr = LinearRegression().fit(X_data, y_data)


# In[ ]:


regr.score(X_data, y_data)


# This is a terrible $R^2$. It seems that our best bet is to just replace the missing values with the median (instead of the mean, because there are some extreme outliers in MonthlyIncome, who are making millions of dollars a month). The other column with missing values is NumberOfDependents. >50% of the non-missing values have 0 dependents, and if someone leaves the field blank it is likely to be due to not having any dependents, we'll replace these values with 0.

# ## Datasets

# Based on all of these insights, we built some datasets that will potentially improve our model performance:

# In[ ]:


# Median Fill, Outliers Removed
removed_debt_outliers = df.drop(df[df['DebtRatio'] > 3489.025].index)
removed_debt_outliers = removed_debt_outliers.fillna(removed_debt_outliers.median())


# In[ ]:


# Removed utilization outliers
dfus = removed_debt_outliers.drop(removed_debt_outliers[removed_debt_outliers['RevolvingUtilizationOfUnsecuredLines'] > 10].index)


# In[ ]:


# Removed 98s
dfn98 = dfus.copy()
dfn98.loc[dfn98['NumberOfTime30-59DaysPastDueNotWorse'] > 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 18
dfn98.loc[dfn98['NumberOfTime60-89DaysPastDueNotWorse'] > 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 18
dfn98.loc[dfn98['NumberOfTimes90DaysLate'] > 90, 'NumberOfTimes90DaysLate'] = 18


# We will revisit these in the next section.

# # Model Testing

# To be able to rapidly test our models and datasets and see how they compare against each other, we developed a tester program that automatically runs k-fold cross-validation on scikit models accross different datasets. 

# In[ ]:


# A utility class to test all of our models on different datasets
class Tester():
    def __init__(self, target):
        self.target = target
        self.datasets = {}
        self.models = {}
        self.cache = {} # we added a simple cache to speed things up

    def addDataset(self, name, df):
        self.datasets[name] = df.copy()

    def addModel(self, name, model):
        self.models[name] = model
        
    def clearModels(self):
        self.models = {}

    def clearCache(self):
        self.cache = {}
    
    def testModelWithDataset(self, m_name, df_name, sample_len, cv):
        if (m_name, df_name, sample_len, cv) in self.cache:
            return self.cache[(m_name, df_name, sample_len, cv)]

        clf = self.models[m_name]
        
        if not sample_len: 
            sample = self.datasets[df_name]
        else: sample = self.datasets[df_name].sample(sample_len)
            
        X = sample.drop([self.target], axis=1)
        Y = sample[self.target]

        s = cross_validate(clf, X, Y, scoring=['roc_auc'], cv=cv, n_jobs=-1)
        self.cache[(m_name, df_name, sample_len, cv)] = s

        return s

    def runTests(self, sample_len=80000, cv=4):
        # Tests the added models on all the added datasets
        scores = {}
        for m_name in self.models:
            for df_name in self.datasets:
                # print('Testing %s' % str((m_name, df_name)), end='')
                start = time.time()

                score = self.testModelWithDataset(m_name, df_name, sample_len, cv)
                scores[(m_name, df_name)] = score
                
                end = time.time()
                
                # print(' -- %0.2fs ' % (end - start))

        print('--- Top 10 Results ---')
        for score in sorted(scores.items(), key=lambda x: -1 * x[1]['test_roc_auc'].mean())[:10]:
            auc = score[1]['test_roc_auc']
            print("%s --> AUC: %0.4f (+/- %0.4f)" % (str(score[0]), auc.mean(), auc.std()))

            
# We will use a tester object across all models
tester = Tester('SeriousDlqin2yrs')

# You can add datasets like this:
tester.addDataset('Drop Missing', df.dropna())

# And models like this:
rfc = RandomForestClassifier(n_estimators=15, max_depth = 6, random_state=0)
tester.addModel('Simple Random Forest', rfc)
tester.addModel('Simple SVM', svm.LinearSVC())

# You can then use it to run the tests
tester.runTests()


# Now that we have this tester up and running, let's go back to the datasets we created in our data exploration section. We can easily add them to the tester program to see how our models perform on them.

# In[ ]:


tester.addDataset('Median Fill', df.fillna(df.median()))
tester.addDataset('Median Fill, Outliers Removed', removed_debt_outliers)
tester.addDataset('Removed 98s', dfn98)
tester.addDataset('Removed utilization outliers', dfus)

tester.runTests()


# We noticed that the modified datasets have a big influence on the SVM AUC scores, but they don't affect the Random Forests as much. However, there's a noticeable gain in performance between simply dropping the missing values and the modified datasets accross all models. 

# ## Random Forests

# Random Forests seemed to be the best bet, so we decided to move forward with them and figure out the best tuning parameters. Running two for-loops allowed us to find the best value for max_depth and number of estimators, with our best model being the random forest with a depth of 9 and 16 estimators. When we use the data set that removes the RUUL outliers, it gives us an AUC of .8662. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

for i in range(5,10):
    for j in range(10,20):
        rfc = RandomForestClassifier(n_estimators=j,max_depth = i, random_state=0)
        tester.addModel('Random Forest '+'d: '+str(i)+' est: '+str(j)  ,rfc)

tester.runTests()


# ## K-Nearest Neighbors

# We were also curious to see how a simpler learner would perform, so we tested a simple K-Nearest Neighbors model. Unsurprisingly, the results were not that great.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

tester.clearModels()

for i in range(5, 10):
    neigh = KNeighborsClassifier(n_neighbors=i)
    tester.addModel('KNN k=%d' % i, neigh)
    
tester.runTests(50000)


# ## Neural Network

# On the other end of the spectrum, we were also curious about trying out really fancy models, so we created and ran a neural network on the data. While K-Nearest-Neighbors was the weakest model we could think to run, a neural network was the strongest thing we could think to do. We expected its performance to be quite high, but discovered exactly the opposite. When initially training the network, we were excited to see it classifying 93% of examples correctly until we realized that it was guessing the same value every time. Essentially, the neural network was not learning a real decision boundary but was instead capitalizing on the imbalance on our data to achieve a high accuracy score. In order to combat this, we attempted to weight our classes differently, but unfortunately were unsuccessful. If we had wanted to, we could have pursued undersampling in order to balance the sample. However, we decided it would be best to not throw away large portions of our data. Because of this in combination with the fact that we do not fully understand the inner workings of a neural network, we decided to use a model whose theory we did understand and that enabled us to take advantage of all of the data.
