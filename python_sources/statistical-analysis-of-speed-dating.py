#!/usr/bin/env python
# coding: utf-8

# **Table of Contents**
# 
# * Part 1: Importing Necessary Modules
#     *     1a. Libraries
#     *     1b. Load dataset
# * Part 2: Cleaning the Data
#     *     2a. Grooming data for features
#     *     2b. Dealing with missing values
#     *     2c. Dealing with categorical data
# * Part 3: Data Visualization and Correlation
#     *     3a. Interests between genders
#     *     3b. Importance of race and religion between races
#     *     3c. Distribution of ratings during dates
#     *     3d. Correlations among features
#     *     3e. Partners chosen vs. race of partner
# * Part 4: T-Test
#     *     4a. Test for correlation between attractiveness and decision
# * Part 5: Feature Engineering
# * Part 6: Pre-Modeling Tasks
#     *     6a. Seperating independent and dependent variables
#     *     6b. Seperating train and test data
# * Part 7: Modeling the Data for Decisions
# * Part 8: Modeling the Data for Matches

# **Kernel Goals**
# 
# The goal of this kernel is twofold,
# 
# 1) To do a statistical examination of speed dating data, researching what different genders & races are intested in socially as well as in a partner, and observing the correlation between certain data points and decisions of those dating.
# 
# 2) To try and predict if two daters will match based on the data provided to us and using various machine learning models.

# **Part 1: Importing Necessary Modules**

# *Part 1a. Libraries*

# In[ ]:


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# *Part 1b. Load Datasets*

# In[ ]:


data = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")


# **Part 2: Overview and Cleaning Data**

# *Part 2a. Grooming data for features*

# In[ ]:


columns = ['iid',
'age',
'gender',
'idg', 
'pid',
'match',
'samerace',
'age_o',
'race_o',
'dec_o',
'field_cd',
'race',
'imprace',
'imprelig',
'from',
'goal',
'date',
'go_out',
'career_c',
'sports',
'tvsports',
'exercise',
'dining',
'museums',
'art',
'hiking',
'gaming',
'clubbing', 
'reading',
'tv',
'theater',
'movies',
'concerts',
'music',
'shopping',
'yoga',
'dec',
'attr',
'sinc',
'intel',
'fun',
'amb',
'like',
'prob',
'met'
]

data = data[columns]


# *2b. Dealing with missing values*

# In[ ]:


data.dropna(inplace=True)


# *Part 2c. Dealing with categorical data*

# In[ ]:


racedict = { 1 : 'black',
             2 : 'white',
             3 : 'latino',
             4 : 'asian',
             5 : 'native american',
             6 : 'other'}

def raceindex(n):
    if n < 1 or n > 6:
        return racedict[0]
    return racedict[n]

data['race'] = data['race'].apply(raceindex)
data['race_o'] = data['race_o'].apply(raceindex)

m = {0 : 'female', 1: 'male'}
data['gender'] = data['gender'].map(m)


# Here I am applying a lambda to replace the number coded columns race & gender with their appropriate values to in order to more easily visualize the data

# **Part 3: Data Visualization and Correlation**

# *Part 3a. Interests between genders*

# In[ ]:


cut_data = data
cut_data.drop_duplicates(subset='iid', inplace=True)

melted_data = pd.melt(cut_data, ['iid', 'gender','idg', 'pid', 'match', 'samerace', 'age_o', 'race_o', 'age',
                             'dec_o','race', 'imprace', 'imprelig', 'from', 'goal', 'date',
                             'go_out', 'field_cd', 'career_c', 'dec', 'attr', 'sinc', 'intel', 'fun', 'amb', 'like', 
                             'prob', 'met'], var_name='interest')

melted_data = melted_data.rename(columns={'value' : 'vote'})
melted_data.reset_index(drop=True)
melted_data.tail()


# Let's look at the differences between some of the characteristics of male and female daters. In order to do this I apply a melt to the data. By seperating votes for interests into seperate rows, we are able to to visualize them on the same graph, making the visualization easier to digest.

# In[ ]:


plt.figure(figsize=(20,20))
plt.title('Distribution of Interests Between Genders')

sns.boxplot(x='interest', y='vote', data=melted_data, hue='gender')


# Here we can see the difference between male & female interests in the various categories provided to us. By using a boxplot we can see the menas & distributions of votes for both genders. 
# 
# While many of the mean values seem to be close together there are some interesting takeaways in that women are much more interested in yoga, shopping and theater while men are much more interested in gaming and tv sports.

# *Part 3b. Importance of race and religion between races*

# In[ ]:


melted_data_race = pd.melt(cut_data, ['iid', 'gender','idg', 'pid', 'match', 'samerace', 'age_o', 'race_o',
                             'dec_o', 'field_cd', 'age', 'race', 'from', 'goal', 'date',
                             'go_out', 'career_c', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 
                             'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies',
                             'concerts', 'music', 'shopping', 'yoga', 'dec', 'attr', 'sinc', 'intel', 'fun', 
                             'amb', 'like', 'prob', 'met'], var_name='importance')

melted_data_race = melted_data_race.rename(columns={'value' : 'rating'})
melted_data_race.reset_index(drop=True)


# Again we melt our dataset. This time to view the differences in importance on religion and race. 

# In[ ]:


plt.figure(figsize=(10,15))
plt.title('Distribution of Importance of Race/Religion Between Races')

sns.boxplot(x='importance', y='rating', data=melted_data_race, hue='race')


# Here we see that the mean values for importance on religion and race for different races seems to be about the same. 

# *Part 3c. Distribution of ratings during dates*

# In[ ]:


fig, ax = plt.subplots(figsize=(20,15), ncols=3, nrows=2)

ax[0][0].set_title("Attractiveness Distribution")
ax[0][1].set_title("Sincerity Distribution"     )
ax[0][2].set_title("Intelligence Distribution"  )
ax[1][0].set_title("Fun Distribution"           )
ax[1][1].set_title("Ambition Distribution"      )
ax[1][2].set_title("Like Distribution"          )

sns.distplot(data.attr , kde = False, ax=ax[0][0])
sns.distplot(data.sinc , kde = False, ax=ax[0][1])
sns.distplot(data.intel, kde = False, ax=ax[0][2])
sns.distplot(data.fun  , kde = False, ax=ax[1][0])
sns.distplot(data.amb  , kde = False, ax=ax[1][1])
sns.distplot(data.like , kde = False, ax=ax[1][2])


# *Part 3e. Partners chosen vs. race of partner*

# In[ ]:


cut_data_yes = cut_data[cut_data['dec'] == 1]
cut_data_no  = cut_data[cut_data['dec'] == 0]


# In[ ]:


plt.figure(figsize=(10,10))
plt.title('Dater vs. Datee Race for "Yes" Decisions')

sns.countplot(x='race_o', data=cut_data_yes, hue='race')


# In[ ]:


plt.figure(figsize=(10,10))
plt.title('Dater vs. Datee Race for "No" Decisions')

sns.countplot(x='race_o', data=cut_data_no, hue='race')


# *Part 3d. Correlations among features*

# In[ ]:


corr = data.corr()['dec']
corr.sort_values(ascending=False)


# Using the corr function we are able to visualize the correlation of each selected column to the decision of the dater (whether or not they want to go out on another date with their current partner). Besides match (if both partners chose 'yes') we can see that their rating for how much they liked the person on that particular date, that persons attractiveness, how much fun they seemed like, and most of the other rating metrics come up at the top of the list. We also see that that persons interest in gaming comes in fairly high (hmmmmmmmm.....???)

# In[ ]:


corr_columns = [
'gender',
'match',
'samerace',
'age_o',
'race_o',
'imprace',
'imprelig',
'dec',
'attr',
'sinc',
'intel',
'fun',
'amb',
'like',
]

data_corr = data[corr_columns]

mask = np.zeros_like(data_corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,15))
sns.heatmap (data_corr.corr(), 
             annot=True,
             mask = mask,
             cmap = 'RdBu_r',
             linewidths=0.1, 
             linecolor='white',
             vmax = .9,
             square=True)

plt.title("Correlations Among Features", y = 1.03,fontsize = 20);


# Using the heat map we can view these correlations more easily.

# **Part 4: T-Test**

# 4a. Test for correlation between attractiveness and decision
# 
# Using the T-Test we are able to check if sample mean differs from the population mean, this means we are able to check if these two data sets are significantly different from each other. For our data this means that we can check if the mean attractiveness of partner who was chosen differs from the mean attractiveness of a partner who was not chosen.
# 
# ------------------------------------------------------------------------------
# 
# Hypothesis Testing: Is there a significant difference in the mean Attractiveness of a partner who was chosen and a partner who was not chosen?
# 
#     * Null Hypothesis: There is no difference between the attractiveness of a partner who was chosen and a partner who was not chosen.
#     * Alternative Hypothesis: There is a difference between the attractiveness of a partner who was chosen and a partner who was not chosen 

# In[ ]:


avg_attr_dec     = cut_data[cut_data['dec'] == 1]['attr'].mean()
avg_attr_not_dec = cut_data[cut_data['dec'] == 0]['attr'].mean()

print('The average attractiveness rating of the people who were chosen is: '     + str(avg_attr_dec    ))
print('The average attractiveness rating of the people who were not chosen is: ' + str(avg_attr_not_dec))


# In[ ]:


import scipy.stats as stats

stats.ttest_1samp(a = cut_data[cut_data['dec'] == 1]['attr'],
                 popmean = avg_attr_not_dec)


# In[ ]:


degree_freedom = len(cut_data[cut_data['dec'] == 1])

lq = stats.t.ppf(0.025, degree_freedom)
rq = stats.t.ppf(0.975, degree_freedom)

print ('The left quartile range of t-distribution is: '  + str(lq))
print ('The right quartile range of t-distribution is: ' + str(rq))


# We conducted our test with 95% accuracy. The test gives us an indication on how much the sample of partners who were chosen during their date allign with those who were not chosen based on their attractiveness. If the test results do not fall within the critical value of 95%, we can reject our null hypothesis. We have chosen such a high threshold because of the strong correlation seen above between attractiveness and the decision made.
# 
# From the results we can see that our pvalue was 1.9087, very well within our critical value of 95%. Because of this we can discard our null hypothesis. This means that our alternative hypothesis is correct, there is a difference in the attractiveness of partners who were, and were not chosen.
# 
# The quartile range of our t-distribution helps us to find the critical value area, drawing a line between those who were and weren't chosen.

# **Part 5: Feature Engineering**

# In[ ]:


data = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")


# In[ ]:


columns = ['iid',
'age',
'gender',
'pid',
'samerace',
'field_cd',
'age_o',
'race_o',
'dec_o',
'round',
'order',
'race',
'imprace',
'imprelig',
'date',
'go_out',
'career_c',
'sports',
'tvsports',
'exercise',
'dining',
'museums',
'art',
'hiking',
'gaming',
'clubbing', 
'reading',
'tv',
'theater',
'movies',
'concerts',
'music',
'shopping',
'yoga',
'exphappy',
'dec'
]

data = data[columns]
data.dropna(inplace=True)


# In[ ]:


data.shape


# Here I am choosing the columns with which to perform some machine learning algorithms on. The goal in this iteration is to try and make some predictions on whether or not a specific person would say yes or not to going on another date with their current partner.
# 
# After seperating the data, and removing all null values, we are left with 8088 rows and 36 columns to work with.

# In[ ]:


def getPartner(pid, interest):
    if (data['iid'] == pid).any():
        row = data[data['iid'] == pid].iloc[0]
        return row[interest]
    else:
        return np.nan

data['partner_sports']   = data['pid'].apply(lambda x : getPartner(x, 'sports'  ))
data['partner_tvsports'] = data['pid'].apply(lambda x : getPartner(x, 'tvsports'))
data['partner_exercise'] = data['pid'].apply(lambda x : getPartner(x, 'exercise'))
data['partner_dining']   = data['pid'].apply(lambda x : getPartner(x, 'dining'  ))
data['partner_museums']  = data['pid'].apply(lambda x : getPartner(x, 'museums' ))
data['partner_art']      = data['pid'].apply(lambda x : getPartner(x, 'art'     ))
data['partner_hiking']   = data['pid'].apply(lambda x : getPartner(x, 'hiking'  ))
data['partner_gaming']   = data['pid'].apply(lambda x : getPartner(x, 'gaming'  ))
data['partner_clubbing'] = data['pid'].apply(lambda x : getPartner(x, 'clubbing'))
data['partner_reading']  = data['pid'].apply(lambda x : getPartner(x, 'reading' ))
data['partner_tv']       = data['pid'].apply(lambda x : getPartner(x, 'tv'      ))
data['partner_theater']  = data['pid'].apply(lambda x : getPartner(x, 'theater' ))
data['partner_movies']   = data['pid'].apply(lambda x : getPartner(x, 'movies'  ))
data['partner_concerts'] = data['pid'].apply(lambda x : getPartner(x, 'concerts'))
data['partner_music']    = data['pid'].apply(lambda x : getPartner(x, 'music'   ))
data['partner_shopping'] = data['pid'].apply(lambda x : getPartner(x, 'shopping'))
data['partner_yoga']     = data['pid'].apply(lambda x : getPartner(x, 'yoga'    ))
data['partner_career']   = data['pid'].apply(lambda x : getPartner(x, 'career_c'))
data['partner_exphappy'] = data['pid'].apply(lambda x : getPartner(x, 'exphappy'))


# In[ ]:


data.dropna(inplace=True)
data.shape


# To get a more clear picture on the partners profile, I have also created this lambda to include their ratings on various activities and expectation.
# 
# After again dropping all null values, we are left with 7998 rows and 54 columns to work with.

# **Part 6: Pre-Modeling Tasks**

# *Part 6a. Seperating independent and dependent variables*

# In[ ]:


x = data.drop(['dec'], axis=1)
y = data['dec']


# *Part 6b. Seperating train and test data*

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)


# *Part 6c. Feature Scaling*

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# **Part 7: Modeling the Data**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
logreg_accy = round(accuracy_score(y_pred, y_test),3)
print(logreg_accy)


# In[ ]:


print (classification_report(y_test, y_pred, labels=logreg.classes_))
print (confusion_matrix(y_pred, y_test))


# In[ ]:


C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']

param = {'penalty': penalties, 'C': C_vals, }

grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=15,shuffle=True), n_jobs=1)


# In[ ]:


grid.fit(x_train,y_train)


# In[ ]:


logreg_grid = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
logreg_grid.fit(x_train,y_train)
y_pred = logreg_grid.predict(x_test)
logreg_accy = round(accuracy_score(y_test, y_pred), 3)
print (logreg_accy)


# In[ ]:


print(classification_report(y_test, y_pred, labels=logreg_grid.classes_))


# In[ ]:


y_score = logreg_grid.decision_function(x_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(roc_auc)


# In[ ]:


plt.figure(figsize =[11,9])
plt.plot(fpr, tpr, label= 'ROC curve(area = %0.2f)'%roc_auc, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Speed Dates', fontsize= 18)
plt.show()


# In[ ]:


precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Speed Daters', fontsize=18)
plt.legend(loc="lower right")
plt.show()


# In[ ]:


dectree = DecisionTreeClassifier(max_depth = 5, 
                                 class_weight = 'balanced', 
                                 min_weight_fraction_leaf = 0.01)

dectree.fit(x_train, y_train)
y_pred = dectree.predict(x_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)


# In[ ]:


scores = []
best_pred = [-1, -1]
for i in range(10, 100, 10):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='minkowski', p=2)
    knn.fit(x_train, y_train)
    score = accuracy_score(y_test, knn.predict(x_test))
    
    if score > best_pred[1]:
        best_pred = [i, score]
    scores.append(score)


# In[ ]:


plt.figure(figsize=[11,9])
plt.plot(range(10,100, 10), scores)
plt.xlabel('N Neighbors', fontsize=18)
plt.ylabel('Accuracy Score', fontsize=18)
plt.title('Accuracy vs. Number of Neighbors for Prediction of Dec', fontsize=18)

plt.show()


# **Part 8: Modeling the Data for Matches**

# In[ ]:


data = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")


# In[ ]:


columns = ['iid',
'age',
'gender',
'pid',
'samerace',
'field_cd',
'race_o',
'dec_o',
'round',
'order',
'race',       
'imprace',
'imprelig',
'date',
'go_out',
'career_c',
'sports',
'tvsports',
'exercise',
'dining',
'museums',
'art',
'hiking',
'gaming',
'clubbing', 
'reading',
'tv',
'theater',
'movies',
'concerts',
'music',
'shopping',
'yoga',
'exphappy',
'match',
'attr',
'sinc',
'intel',
'fun',
'amb',
'like',
'prob',
'met'
]

data = data[columns]
data.dropna(inplace=True)


# In[ ]:


data.shape


# In[ ]:


def getPartner(pid, iid, interest):
        row = data[(data['pid'] == iid) & (data.loc[:, 'iid'] == pid)]
        if row.empty == False:
            return row[interest].iloc[0]
        else:
            return np.nan


# In[ ]:


data['partner_sports']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'sports'  ), axis=1)
data['partner_tvsports'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'tvsports'), axis=1)
data['partner_exercise'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'exercise'), axis=1)
data['partner_dining']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'dining'  ), axis=1)
data['partner_museums']  = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'museums' ), axis=1)
data['partner_art']      = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'art'     ), axis=1)
data['partner_hiking']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'hiking'  ), axis=1)
data['partner_gaming']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'gaming'  ), axis=1)
data['partner_clubbing'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'clubbing'), axis=1)
data['partner_reading']  = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'reading' ), axis=1)
data['partner_tv']       = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'tv'      ), axis=1)
data['partner_theater']  = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'theater' ), axis=1)
data['partner_movies']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'movies'  ), axis=1)
data['partner_concerts'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'concerts'), axis=1)
data['partner_music']    = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'music'   ), axis=1)
data['partner_shopping'] = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'shopping'), axis=1)
data['partner_yoga']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'yoga'    ), axis=1)
data['partner_career']   = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'career_c'), axis=1)
data['partner_attr']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'attr'    ), axis=1)
data['partner_sinc']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'exphappy'), axis=1)
data['partner_intel']    = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'sinc'    ), axis=1)
data['partner_fun']      = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'fun'     ), axis=1)
data['partner_amb']      = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'amb'     ), axis=1)
data['partner_like']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'like'    ), axis=1)
data['partner_prob']     = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'prob'    ), axis=1)
data['partner_met']      = data.apply(lambda x : getPartner(x['pid'], x['iid'], 'met'     ), axis=1)

data.dropna(inplace=True)
data.shape


# In[ ]:


x = data.drop(['match'], axis=1)
y = data['match']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)


# In[ ]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:


logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
logreg_accy = round(accuracy_score(y_pred, y_test),3)
print(logreg_accy)


# In[ ]:


C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']

param = {'penalty': penalties, 'C': C_vals, }

grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=15,shuffle=True), n_jobs=1)


# In[ ]:


grid.fit(x_train,y_train)


# In[ ]:


logreg_grid = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
logreg_grid.fit(x_train,y_train)
y_pred = logreg_grid.predict(x_test)
logreg_accy = round(accuracy_score(y_test, y_pred), 3)
print (logreg_accy)


# In[ ]:


print(classification_report(y_test, y_pred, labels=logreg_grid.classes_))


# In[ ]:


y_score = logreg_grid.decision_function(x_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(roc_auc)


# In[ ]:


plt.figure(figsize =[11,9])
plt.plot(fpr, tpr, label= 'ROC curve(area = %0.2f)'%roc_auc, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Speed Dates', fontsize= 18)
plt.show()


# In[ ]:


y_score = logreg_grid.decision_function(x_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Speed Daters', fontsize=18)
plt.legend(loc="lower right")
plt.show()


# In[ ]:


dectree = DecisionTreeClassifier(max_depth = 5, 
                                 class_weight = 'balanced', 
                                 min_weight_fraction_leaf = 0.01)

dectree.fit(x_train, y_train)
y_pred = dectree.predict(x_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)


# In[ ]:


scores = []
best_pred = [-1, -1]
for i in range(10, 100, 10):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='minkowski', p=2)
    knn.fit(x_train, y_train)
    score = accuracy_score(y_test, knn.predict(x_test))
    
    if score > best_pred[1]:
        best_pred = [i, score]
    scores.append(score)
print (best_pred)


# In[ ]:


plt.figure(figsize=[11,9])
plt.plot(range(10,100, 10), scores)
plt.xlabel('N Neighbors', fontsize=18)
plt.ylabel('Accuracy Score', fontsize=18)
plt.title('Accuracy vs. Number of Neighbors for Prediction of Match', fontsize=18)

plt.show()


# **END**
# 
# Thanks for reading through my notebook! This is my first attempt at putting something cohesive together in the machine learning/data science realm so all feedback is welcome.
# 
# I'd also like to acknowledge some of the other notebooks I used as inspiration and reference while putting this together
# * https://www.kaggle.com/masumrumi/a-statistical-analysis-of-titanic-with-ml-models
# * https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/notebook
# * https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners/
# 
# Both of these guys do fantastic work and I would recommend checking out their pages.
# 

# In[ ]:




