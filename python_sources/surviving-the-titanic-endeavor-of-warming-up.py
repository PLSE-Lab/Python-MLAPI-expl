#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import re
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import cross_validation, metrics
import xgboost as xgb
from xgboost import XGBClassifier

## Import the training data and look at it
train_df = pd.read_csv('../input/train.csv', sep = ',', index_col = 0)
print(train_df.describe())
print('\nnumber of null values for each category\n', train_df.isnull().sum())
print(train_df.head())


# ## 1) Initial feature impressions
# - It looks like Age has NaNs but the other numerical features (Pclass, SibSp, Parch, and Fare) don't. Since a large majority of passengers have a defined age (714/891) it is probably useful to impute the NaN ages instead of removing these passengers.
# - Fare seems to have a good range, which might be able to be used to indicate wealth. This could possibly influence survival decisions (maybe wealthy people held more sway in the decision as they have historically in most situations).

# ### Pclass
# - There are only three values here: 1 is like first class and 3 is like economy (if we are comparing to today's airline classes).
# - Presumably 1 corresponds to higher ticket prices than the other two classes.
# - This can be used to infer survival potential if we assume that wealthier people had a better say in their survival.
# 
# Let's see if class relates to fare and how it relates to survival...

# In[ ]:


sns.countplot(train_df['Pclass'])
print('Most passengers were third class')


# In[ ]:


sns.violinplot(x = 'Fare', y = 'Pclass', data = train_df, orient = 'h', width = 1)
print('Median fare for first class:', train_df['Fare'].loc[train_df['Pclass'] == 1].median())
print('Median fare for second class:', train_df['Fare'].loc[train_df['Pclass'] == 2].median())
print('Median fare for third class:', train_df['Fare'].loc[train_df['Pclass'] == 3].median())
print('\nThere is a clear association between Pclass and Fare, suggesting that this could be used to infer wealth.')


# In[ ]:


print('First class survival rate:', 100*136./(136+80))
print('Second class survival rate:', 100*87./(87+97))
print('Third class survival rate:', 100*119./(119+372))
print(train_df[['Pclass', 'Fare', 'Survived']].groupby(['Pclass', 'Survived']).count())
print('\nFirst class passengers clearly survived better than those in second or third class (63% vs 47% or 24%)')


# ### Pclass (cont)
# - Pclass relates to Fare and can be used to infer wealth. First class passengers had a much higher survival rate than others, suggesting this may be a useful feature in the prediction.

# ### Name
# - Name doesn't seem like it could help much unless I used titles, which could display wealth.

# In[ ]:


## Make text lowercase
train_df['Name'] = train_df['Name'].apply(lambda x: x.lower())
## Extract title from the name
train_df['title'] = train_df['Name'].apply(lambda x: re.split('\.', re.split(', ', x)[1])[0])
## See how survival relates to title
train_df[['title', 'Survived', 'Name']].groupby(['title', 'Survived']).count()
print('I am defining titled as anything but Mr, Mrs, Ms, Miss, Master (designated boys back then), or Mlle')
## make a simple function to return 0 if the title passed is in a common list, otherwise return 1.
def title_finder(title):
    simple_titles = ['mr', 'mrs', 'ms', 'miss', 'mlle', 'master']
    if title in simple_titles:
        return 0
    else:
        return 1
train_df['has_title'] = train_df['title'].apply(title_finder)


# In[ ]:


print('Untitled survival rate:', 100*333./(333+534))
print('Titled class survival rate:', 100*9./(15+9))
print( train_df[['has_title', 'Name', 'Survived']].groupby(['has_title', 'Survived']).count())
print('\nTitled and untitled passengers fare similarly, but by eye it appears that there may be a significant gender gap in this metric')


# In[ ]:


print('Male untitled survival rate:', 100*104./(104+453))
print('Female untitled class survival rate:', 100*229./(229+81))
print('Male titled class survival rate:', 100*5./(15+5))
print('Female titled survival rate:', 100*4./(4+0))
print(train_df[['has_title', 'Sex', 'Name', 'Survived']].groupby(['has_title', 'Sex', 'Survived']).count())
print("Fisher's exact test p-value for males", st.fisher_exact([[104, 5],[453, 15]])[1])
print("Fisher's exact test p-value for females", st.fisher_exact([[229, 4],[81, 0]])[1])
## The low sample size for titled men and women make a [2x2]x[2x2] chi squared contingency table assessment unreliable
# print "Chi squared contingency table of survival by gender and title status:", st.chi2_contingency([[[229, 4],[81, 0]], [[104, 5],[453, 15]]])[1]
print('\nTitled passengers appear to do better by eye when gender is taken into account, but this is not supported by conservative statistics')


# ### Name (cont)
# - Name doesn't seem very useful for predictions unless titles are taken into account.
# - Whether or not someone has a title is not likely to be predictive, but what their title is might matter.

# ### Sex
# - Sex seems to matter by eye.

# In[ ]:


## Lowercase sex
train_df['Sex'] = train_df['Sex'].apply(lambda x: x.lower())
## Show gender differences here
print(train_df[['Sex', 'Name', 'Survived']].groupby(['Sex', 'Survived']).count())
print("There is an obvious difference in the survival rate by gender that is supported by statistics.")
print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[233, 109],[81, 468]])[1])


# ### Age
# - Age may be an important factor in determining survival due to the "women and children first" mentality.
# - Some ages are unknown, but these can be imputed. Given that the "Master" title corresponds to boys despite the lack of an equivalent distingishing title for girls, we could split the passengers by gender and impute their ages as the median age unless they hold the title of "Master" for which we would impute those ages as the median age of all "Master" passengers.
# - We might be able to glean more information about the age of the female passengers from the "Name," "SibSp," and "Parch" fields. Maybe investigate this later.

# In[ ]:


## Look at how age relates to survival
train_df['Age'].loc[train_df['Survived'] == 0].hist(bins = 100)
train_df['Age'].loc[train_df['Survived'] == 1].hist(bins = 100)


# In[ ]:


## Impute age
master_med_age = train_df['Age'].loc[train_df['title'] == 'Master'].median()
male_med_age = train_df['Age'].loc[train_df['Sex'] == 'male'].median()
female_med_age = train_df['Age'].loc[train_df['Sex'] == 'female'].median()

def median_age_assigner(row):
    ## Assigns null ages based on gender and title. Checks for the "Master" title
    ### to identify boys and returns the median age for boys for passengers with 
    ### this title but without an age. Passengers without this title are checked
    ### for gender and if their age is null and gender is female they are assigned
    ### the median age for all females. 
    row_age = row['Age']
    row_sex = row['Sex']
    row_title = row['title']
    if row.isnull().loc['Age']:
        if row_title == 'Master':
            return master_med_age
        elif row_sex == 'male':
            return male_med_age
        elif row_sex == 'female':
            return female_med_age
    else:
        return row_age
        
train_df['Age'] = train_df.apply(median_age_assigner, axis = 1)


# In[ ]:


## Look at how age relates to survival after imputation
train_df['Age'].loc[train_df['Survived'] == 0].hist(bins = 100)
train_df['Age'].loc[train_df['Survived'] == 1].hist(bins = 100)


# ### Age (cont)
# - Let's look at children and how they fare.

# In[ ]:


train_df['Age'].loc[train_df['Survived'] == 0].loc[train_df['Age'] <= 21].hist(bins = 21)
train_df['Age'].loc[train_df['Survived'] == 1].loc[train_df['Age'] <= 21].hist(bins = 21)
print("It appears that survival drops off around age 18, so we'll use this as the age cutoff for children and make a new feature 'is_child'")


# In[ ]:


train_df['is_child'] = 0
train_df['is_child'].loc[train_df['Age'] < 18] = 1
print(train_df[['is_child', 'Name', 'Survived']].groupby(['is_child', 'Survived']).count())
print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[279, 63],[495, 54]])[1])
print("It appears that children/minors have a higher survival rate")


# ### SibSp
# - SibSp seems like it has potential to be predictive of survival, but I don't think it will be as good as Parch.
# 
# Let's see how the two relate...

# In[ ]:


print(train_df[['SibSp', 'Name', 'Survived']].groupby(['SibSp', 'Survived']).count())
train_df['has_sibsp'] = 0
train_df['has_sibsp'].loc[train_df['SibSp'] > 0] = 1
print(train_df[['has_sibsp', 'Name', 'Survived']].groupby(['has_sibsp', 'Survived']).count())


# ### SibSp (cont)
# - It appears that either married couples or people with only one or two siblings are better off than the rest when it comes to survival, whereas people with many siblings are worse off.
# - This could mean that individuals with larger families on board were less likely to survive, suggesting a family size feature may be useful.
# - We should be able to combine SibSp with Parch to get family size.

# ### Parch
# - Parch may be useful if combined with SibSp, but let's see how well it does on its own...

# In[ ]:


print(train_df[['Parch', 'Name', 'Survived']].groupby(['Parch', 'Survived']).count())
train_df['has_parch'] = 0
train_df['has_parch'].loc[train_df['Parch'] > 0] = 1
print(train_df[['has_parch', 'Name', 'Survived']].groupby(['has_parch', 'Survived']).count())


# ### Parch (cont)
# - Again, it appears that individuals in small families fare better than the average, while those in large families do not.
# - We should definitely combine SibSp and Parch to make family size feature.

# In[ ]:


train_df['family_size'] = train_df['SibSp'] + train_df['Parch'] + 1
print(train_df[['family_size', 'Name', 'Survived']].groupby(['family_size', 'Survived']).count())
print("Single people didn't do too well, and neither did members of large families (5 or more members). Small families (2-4 members), however, fared better than average.")


# ### Ticket
# - I'm not sure if any information derived from the ticket number would be particularly useful other than maybe predicting wealth possibly based on some type of qualifier in the ticket number (like maybe A/5 for passenger 1).
# - Let's find the qualifiers for tickets and see if tickets with qualifiers do better or if certain qualifiers do better.

# In[ ]:


## Lowercase Ticket
train_df['Ticket'] = train_df['Ticket'].apply(lambda x: x.lower())
## Extract qualifier from the ticket
def get_ticket_qualifier(ticket):
    separated_ticket = re.split(' ', ticket)
    if len(separated_ticket) > 1:
        return separated_ticket[0]
    else:
        return 'none'
train_df['ticket_qualifier'] = train_df['Ticket'].apply(get_ticket_qualifier)
print(train_df['ticket_qualifier'].unique().tolist())


# ### Ticket (cont)
# - It appears that some ticket qualifiers are probably the same qualifier as others but they are either capitalized or differ by whether or not they have a period or slash at a certain place.
# - Making them all lowercase and removing any non-alphanumerics would consolidate.

# In[ ]:


## Extract qualifier from the ticket
def get_ticket_qualifier(ticket):
    separated_ticket = re.split(' ', ticket)
    if len(separated_ticket) > 1:
        temp_qual = re.sub('[^A-Za-z0-9]+', '', separated_ticket[0])
        return temp_qual
    else:
        return 'none'
train_df['ticket_qualifier'] = train_df['Ticket'].apply(get_ticket_qualifier)
train_df['has_ticket_qualifier'] = 0
train_df['has_ticket_qualifier'].loc[train_df['ticket_qualifier'] != 'none'] = 1
print(train_df['ticket_qualifier'].unique().tolist())


# In[ ]:


print(train_df[['has_ticket_qualifier', 'Name', 'Survived']].groupby(['has_ticket_qualifier', 'Survived']).count())
print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[255, 87],[410, 139]])[1])
print(train_df[['ticket_qualifier', 'Name', 'Survived']].groupby(['ticket_qualifier', 'Survived']).count())
def does_ticket_start_with_p(ticket_text):
    if str(ticket_text)[0].lower() == 'p':
        return 1
    else:
        return 0
train_df['ticket_starts_with_p'] = train_df['Ticket'].apply(does_ticket_start_with_p)
print(train_df[['ticket_starts_with_p', 'Name', 'Survived']].groupby(['ticket_starts_with_p', 'Survived']).count())
print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[300, 42],[526, 23]])[1])


# ### Ticket (cont)
# - Having a ticket qualifier did not seem to matter for survival overall.
# - Certain ticket qualifiers seem to do better than the average, though, and these include ones that start with the letter "p" which might stand for something like "personal" or "private" cabin.
# - This would probably designate wealth, which seems to be predictive based on Pclass.

# ### Fare
# - Fare seems like it should be a decent predictor of survival if wealth is an indicator of survival.

# In[ ]:


sns.violinplot(x = 'Fare', y = 'Survived', data = train_df, orient = 'h')
print(st.ttest_ind(train_df['Fare'].loc[train_df['Survived'] == 0], train_df['Fare'].loc[train_df['Survived'] == 1]))
print("People who paid higher fares were more likely to survive.")


# ### Cabin

# In[ ]:


## Lowercase cabin
train_df['Cabin'] = train_df['Cabin'].dropna().apply(lambda x: x.lower())
print(train_df['Cabin'].unique().tolist())
print("\nnumber of nulls in cabin", train_df['Cabin'].isnull().sum())
plt.figure(figsize = [9,4])
sns.countplot(train_df['Cabin'].dropna())


# ### Cabin (cont)
# - There are a lot of NaNs in cabin (687/891), so this is probably not a super predictive feature. Whether or not a cabin is assigned might important, so let's see.

# In[ ]:


## Add had_cabin_assigment to train_df
train_df['had_cabin_assignment'] = 0
train_df['had_cabin_assignment'].loc[train_df['Cabin'].isnull() != True] = 1
## Fill NaNs with None
train_df['Cabin'].fillna('None', inplace = True)
print(train_df[['had_cabin_assignment', 'Survived', 'Name']].groupby(['had_cabin_assignment', 'Survived']).count())


# ### Cabin (cont)
# - Hmm, 67% of passengers who had a designated cabin survived (136/204) as opposed to only 30% of those without a designated cabin (206/687). This suggests that whether or not a passenger had a designated cabin could be a useful feature to add to the dataset.
# - This could be for a variety of reasons:
#     - Maybe this is a coincidence, but that seems unlikely given the mean 38% survival rate and the large discrepancy between the two groups in question.
#     - Maybe there was a general admission status that stayed in a hostel-like section as well as a private room status that was assigned a cabin.
#     - Maybe only certain cabins were available for request/purchase and the passengers who did not request/purchase them were randomly assigned to the other cabins. In this case it would benefit order to keep track of the purchased cabins and who purchased them.
# - Let's see how cabin assignment relates to fare

# In[ ]:


sns.violinplot(x = 'Fare', y = 'had_cabin_assignment', data = train_df, orient = 'h')


# ### Cabin (cont)
# - Passengers with an assigned cabin clearly paid more for their tickets.
# - This can be interpretted as evidence for one of the aforementioned admission grouping scenarios and reveals a dependence between cabin assigment and fare. Using both of these thus might not provide much improvement over one or the other, but we'll see.

# ### Embarked
# - I don't really think the embarkation point will make much of a difference, but let's see.

# In[ ]:


print(train_df['Embarked'].unique().tolist())
print("\nnumber of nulls in embarked", train_df['Embarked'].isnull().sum())
sns.countplot(train_df['Embarked'].dropna())


# ### Embarked (cont)
# - Only 2 passengers have NaN for this field, so we can impute their embarkation point as S since a large majority of people embarked from S.

# In[ ]:


## Impute nans to S
train_df['Embarked'].fillna('S', inplace = True)
## Make Embarked lowercase
train_df['Embarked'] = train_df['Embarked'].apply(lambda x: x.lower())
print(train_df[['Embarked', 'Name', 'Survived']].groupby(['Embarked', 'Survived']).count())
print("Chi squared contingency table p-value with two-sided null hypothesis", st.chi2_contingency([[93, 30, 217],[75, 47, 427]])[1])


# ### Embarked (cont)
# - I was wrong. The embarkation point definitely made a difference, and the only reason I can think of why this may be is once again due to wealth.
# 
# Let's see...

# In[ ]:


sns.violinplot(x = 'Fare', y = 'Embarked', data = train_df, orient = 'h')
print(st.ttest_ind(train_df['Fare'].loc[train_df['Embarked'] == 'C'], train_df['Fare'].loc[train_df['Embarked'] != 'C']))
print("Passengers who embarked from 'C' clearly paid more for their trip, which could be an indicator of wealth.")


# ## Summary of feature impressions
# - Wealth seems to have been influential in survival. We have a few different features that predict wealth: Pclass, Fare, Cabin, etc.
# - Family size seems to matter, with singles and individuals of families with 5 or more people faring poorly while individuals of families with 2-4 members fared well.
# - Age matters, with children/minors faring better than adults.
# - Gender matters, with females faring better than males.

# ## 2) Let's start figuring out which features are predictive
# - First we need to do all the same stuff we did to the training data to the test data.
# - I like to make a specific function for this to keep everything organized.

# In[ ]:


def data_processor(df):
    ## Set embarked to categorical integers
    df['Embarked'].fillna('S', inplace = True)
    df['emb_s'] = 0
    df['emb_s'].loc[df['Embarked'] == 'S'] = 1
    df['emb_c'] = 0
    df['emb_c'].loc[df['Embarked'] == 'C'] = 1
    df['emb_q'] = 0
    df['emb_q'].loc[df['Embarked'] == 'Q'] = 1
    ## Set new feature "had_cabin"
    df['had_cabin'] = 0
    df['had_cabin'].loc[df['Cabin'].isnull() == False] = 1
    ## Get title
    df['Name'] = df['Name'].apply(lambda x: x.lower())
    ### Extract title from the name
    df['title'] = df['Name'].apply(lambda x: re.split('\.', re.split(', ', x)[1])[0])
    ### make a simple function to return 0 if the title passed is in a common list, otherwise return 1.
    def title_finder(title):
        simple_titles = ['mr', 'mrs', 'ms', 'miss', 'mlle', 'master']
        if title in simple_titles:
            return 0
        else:
            return 1
    df['had_title'] = df['title'].apply(title_finder)
    ## Impute age
    master_med_age = df['Age'].loc[df['title'] == 'Master'].median()
    male_med_age = df['Age'].loc[df['Sex'] == 'male'].median()
    female_med_age = df['Age'].loc[df['Sex'] == 'female'].median()

    def median_age_assigner(row):
        ## Assigns null ages based on gender and title. Checks for the "Master" title
        ### to identify boys and returns the median age for boys for passengers with 
        ### this title but without an age. Passengers without this title are checked
        ### for gender and if their age is null and gender is female they are assigned
        ### the median age for all females. 
        row_age = row['Age']
        row_sex = row['Sex']
        row_title = row['title']
        if row.isnull().loc['Age']:
            if row_title == 'Master':
                return master_med_age
            elif row_sex == 'male':
                return male_med_age
            elif row_sex == 'female':
                return female_med_age
        else:
            return row_age

    df['Age'] = df.apply(median_age_assigner, axis = 1).astype(int)
#     df['Age'].fillna(df['Age'].median(), inplace = True)
    ## Set new feature "child"
    df['child'] = 0
    df['child'].loc[df['Age'] < 18] = 1
    ## Set sex to categorical integers
    df['Sex'].loc[df['Sex'] != 'male'] = 0
    df['Sex'].loc[df['Sex'] == 'male'] = 1
    df['Sex'] = df['Sex'].astype(int)
    ## Family size
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    ## Ticket qualifier
    df['Ticket'] = df['Ticket'].apply(lambda x: x.lower())
    ## Extract qualifier from the ticket
    def get_ticket_qualifier(ticket):
        separated_ticket = re.split(' ', ticket)
        if len(separated_ticket) > 1:
            temp_qual = re.sub('[^A-Za-z0-9]+', '', separated_ticket[0])
            return temp_qual
        else:
            return 'none'
    df['ticket_qualifier'] = df['Ticket'].apply(get_ticket_qualifier)
    df['had_ticket_qualifier'] = 0
    df['had_ticket_qualifier'].loc[df['ticket_qualifier'] != 'none'] = 1
    def does_ticket_start_with_p(ticket_text):
        if str(ticket_text)[0].lower() == 'p':
            return 1
        else:
            return 0
    df['ticket_starts_with_p'] = df['Ticket'].apply(does_ticket_start_with_p)
    ## Fare
    pc1_fare_med = df['Fare'].loc[df['Pclass'] == 1].median()
    pc2_fare_med = df['Fare'].loc[df['Pclass'] == 2].median()
    pc3_fare_med = df['Fare'].loc[df['Pclass'] == 3].median()
    def fare_imputer(row):
        if row.isnull().sum() > 0:
            pc = row['Pclass']
            if pc == 1:
                return pc1_fare_med
            elif pc == 2:
                return pc2_fare_med
            else:
                return pc3_fare_med
        else:
            return row['Fare']
    df['Fare'] = df.apply(fare_imputer, axis = 1)
    ## Set fare above 75 to 75
#     df['Fare']
#     df['Fare'].loc[df['Fare'] > 75] = 75
#     df['Fare'] = df['Fare'].apply(round).astype(int)
    ## Pclass
    df['first_class'] = 0
    df['first_class'].loc[df['Pclass'] == 1] = 1
    df['second_class'] = 0
    df['second_class'].loc[df['Pclass'] == 2] = 1
    df['third_class'] = 0
    df['third_class'].loc[df['Pclass'] == 3] = 1
    return df


# In[ ]:


train_proc = data_processor(train_df)


# In[ ]:


train_proc.head()


# In[ ]:


train_x = train_proc.drop(['Pclass', 'Name', 'title', 'Ticket', 'ticket_qualifier', 'Cabin', 'Embarked', 'Survived'], axis = 1)
train_y = train_proc['Survived']


# In[ ]:


train_x.head()


# In[ ]:


lr = LogisticRegression(random_state = 0)
kf = KFold(n_splits=3, random_state = 0)
parameters = {'C': [i/10. for i in range(1, 50)]}

lr_gscv = GridSearchCV(lr, parameters, cv = kf, verbose = 1)
lr_gscv.fit(train_x, train_y)

print('Using all the features with a logistic regression scored', lr_gscv.score(train_x, train_y))
print('')
print(sns.barplot(y = train_x.columns, x = np.abs(lr_gscv.best_estimator_.coef_[0]), orient = 'h'))
print('')
print( "So it looks like Sex, has_sibsp, Pclass, Sibsp, and is_child are the top 5 features that the logistic regression likes.")
print('')
print( "It also appears that the features that are binary or only contain a small amount of categories are more predictive, which makes sense given the way the classifier works.")


# In[ ]:


kf = KFold(n_splits=3, random_state = 0)

parameters = {'C': [0.7], 
              'kernel': ['linear']}
svc = SVC(decision_function_shape = 'ovr', random_state = 0)
svc_gscv = GridSearchCV(svc, parameters, cv = kf, verbose = 2)
svc_gscv.fit(train_x, train_y)
print(svc_gscv.best_params_)
print('')
print('Using all the features with a svc scored', np.mean(svc_gscv.best_score_))
print(sns.barplot(y = train_x.columns, x = np.abs(svc_gscv.best_estimator_.coef_[0]), orient = 'h'))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
kf = KFold(n_splits=3, random_state = 0)
gnb = GaussianNB()
gnb.fit(train_x, train_y)
print('Using all the features with a GNB classifier scored', gnb.score(train_x, train_y))


# In[ ]:


parameters = {
    'n_estimators': [100, 125, 150], 
    'max_features': [5, 10, 15],
    'max_depth': (2, 5, 10, 15),
    'min_samples_split': [2, 3, 5, 10], 
    'min_samples_leaf': [1, 2, 3, 5, 10]
}

kf = KFold(n_splits = 3, random_state = 0)

rfc = RandomForestClassifier(random_state = 0)
rfc_gscv = GridSearchCV(rfc, parameters, cv = kf, verbose = 1)
rfc_gscv.fit(train_x, train_y)

print(rfc_gscv.best_params_)
print("")
print('Using all the features with a random forest classifier scored', np.mean(rfc_gscv.best_score_))
print(sns.barplot(y = train_x.columns, x = np.abs(rfc_gscv.best_estimator_.feature_importances_), orient = 'h'))


# In[ ]:


## Set the grid search parameters for the xgb classifier to pass to GridSearchCV

parameters = {
    'n_estimators': [110],
    'max_depth':range(3, 10, 2),
    'min_child_weight':range(1, 6, 2),
    #'gamma': [i/10.0 for i in range(0, 6)],
    #'subsample': [i/10.0 for i in range(5, 11)],
    #'colsample_bytree': [i/10.0 for i in range(5, 11)],
    #'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],
    #'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],
    #'learning_rate': [i/100. for i in range(1, 15)]
}

training_data = train_x.join(train_y)

predictors = training_data.drop('Survived', axis = 1).columns
outcome = 'Survived'

## Instantiate the xgb classifier
xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 
                     gamma=0, subsample=0.8, colsample_bytree=0.8, 
                     objective= 'binary:logistic', 
                     nthread=-1, seed=0, silent = False)
## Set up the k folds function to pass to GridSearchCV
kf = KFold(n_splits = 3, random_state = 0)
## Instantiate the grid search across the parameters and with cross validation on 3 folds
xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)
## Do the grid search and cv
xgbc_gscv.fit(training_data[predictors], training_data[outcome])

print(xgbc_gscv.best_params_)
print(xgbc_gscv.best_score_)
feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)
print(feat_imp.plot(kind='bar', title='Feature Importances'))


# In[ ]:


## Set the grid search parameters for the xgb classifier to pass to GridSearchCV

parameters = {
    'n_estimators': [110],
    'max_depth': [5],
    'min_child_weight': [5],
    'gamma': [i/10.0 for i in range(0, 6)],
    #'subsample': [i/10.0 for i in range(5, 11)],
    #'colsample_bytree': [i/10.0 for i in range(5, 11)],
    #'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],
    #'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],
    #'learning_rate': [i/100. for i in range(1, 15)]
}

training_data = train_x.join(train_y)

predictors = training_data.drop('Survived', axis = 1).columns
outcome = 'Survived'

## Instantiate the xgb classifier
xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 
                     gamma=0, subsample=0.8, colsample_bytree=0.8, 
                     objective= 'binary:logistic', 
                     nthread=-1, seed=0, silent = False)
## Set up the k folds function to pass to GridSearchCV
kf = KFold(n_splits = 3, random_state = 0)
## Instantiate the grid search across the parameters and with cross validation on 3 folds
xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)
## Do the grid search and cv
xgbc_gscv.fit(training_data[predictors], training_data[outcome])

print(xgbc_gscv.best_params_)
print(xgbc_gscv.best_score_)
feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)
print(feat_imp.plot(kind='bar', title='Feature Importances'))


# In[ ]:


## Set the grid search parameters for the xgb classifier to pass to GridSearchCV

parameters = {
    'n_estimators': [110],
    'max_depth': [5],
    'min_child_weight': [5],
    'gamma': [0],
    'subsample': [i/10.0 for i in range(5, 11)],
    'colsample_bytree': [i/10.0 for i in range(5, 11)],
    #'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],
    #'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],
    #'learning_rate': [i/100. for i in range(1, 15)]
}

training_data = train_x.drop(['Name', 'Ticket'], axis = 1).join(train_y)

predictors = training_data.drop('Survived', axis = 1).columns
outcome = 'Survived'

## Instantiate the xgb classifier
xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 
                     gamma=0, subsample=0.8, colsample_bytree=0.8, 
                     objective= 'binary:logistic', 
                     nthread=-1, seed=0, silent = False)
## Set up the k folds function to pass to GridSearchCV
kf = KFold(n_splits = 3, random_state = 0)
## Instantiate the grid search across the parameters and with cross validation on 3 folds
xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)
## Do the grid search and cv
xgbc_gscv.fit(training_data[predictors], training_data[outcome])

print(xgbc_gscv.best_params_)
print(xgbc_gscv.best_score_)
feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)
print(feat_imp.plot(kind='bar', title='Feature Importances'))


# In[ ]:


## Set the grid search parameters for the xgb classifier to pass to GridSearchCV

parameters = {
    'n_estimators': [110],
    'max_depth': [5],
    'min_child_weight': [5],
    'gamma': [0],
    'subsample': [1],
    'colsample_bytree': [0.7],
    'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],
    'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],
    #'learning_rate': [i/100. for i in range(1, 15)]
}

training_data = train_x.drop(['Name', 'Ticket'], axis = 1).join(train_y)

predictors = training_data.drop('Survived', axis = 1).columns
outcome = 'Survived'

## Instantiate the xgb classifier
xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 
                     gamma=0, subsample=0.8, colsample_bytree=0.8, 
                     objective= 'binary:logistic', 
                     nthread=-1, seed=0, silent = False)
## Set up the k folds function to pass to GridSearchCV
kf = KFold(n_splits = 3, random_state = 0)
## Instantiate the grid search across the parameters and with cross validation on 3 folds
xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)
## Do the grid search and cv
xgbc_gscv.fit(training_data[predictors], training_data[outcome])

print(xgbc_gscv.best_params_)
print(xgbc_gscv.best_score_)
feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)
print(feat_imp.plot(kind='bar', title='Feature Importances'))


# In[ ]:


## Set the grid search parameters for the xgb classifier to pass to GridSearchCV

parameters = {
    'n_estimators': [110],
    'max_depth': [5],
    'min_child_weight': [5],
    'gamma': [0],
    'subsample': [1],
    'colsample_bytree': [0.7],
    'reg_alpha': [0.1],
    'reg_lambda': [0],
    'learning_rate': [i/100. for i in range(1, 15)]
}

training_data = train_x.drop(['Name', 'Ticket'], axis = 1).join(train_y)

predictors = training_data.drop('Survived', axis = 1).columns
outcome = 'Survived'

## Instantiate the xgb classifier
xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 
                     gamma=0, subsample=0.8, colsample_bytree=0.8, 
                     objective= 'binary:logistic', 
                     nthread=-1, seed=0, silent = False)
## Set up the k folds function to pass to GridSearchCV
kf = KFold(n_splits = 3, random_state = 0)
## Instantiate the grid search across the parameters and with cross validation on 3 folds
xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)
## Do the grid search and cv
xgbc_gscv.fit(training_data[predictors], training_data[outcome])

print(xgbc_gscv.best_params_)
print(xgbc_gscv.best_score_)
feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)
print(feat_imp.plot(kind='bar', title='Feature Importances'))


# In[ ]:


## Set the grid search parameters for the xgb classifier to pass to GridSearchCV

parameters = {
    'n_estimators': [110],
    'max_depth': [5],
    'min_child_weight': [5],
    'gamma': [0],
    'subsample': [1],
    'colsample_bytree': [0.7],
    'reg_alpha': [0.1],
    'reg_lambda': [0],
    'learning_rate': [0.12]
}

training_data = train_x.join(train_y)

predictors = training_data.drop('Survived', axis = 1).columns
outcome = 'Survived'

## Instantiate the xgb classifier
xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 
                     gamma=0, subsample=0.8, colsample_bytree=0.8, 
                     objective= 'binary:logistic', 
                     nthread=-1, seed=0, silent = False)
## Set up the k folds function to pass to GridSearchCV
kf = KFold(n_splits = 3, random_state = 0)
## Instantiate the grid search across the parameters and with cross validation on 3 folds
xgbc_gscv = GridSearchCV(xgbc, parameters, cv = kf, verbose = 2)
## Do the grid search and cv
xgbc_gscv.fit(training_data[predictors], training_data[outcome])

print(xgbc_gscv.best_params_)
print(xgbc_gscv.best_score_)
feat_imp = pd.Series(xgbc_gscv.best_estimator_.booster().get_fscore()).sort_values(ascending=False)
print(feat_imp.plot(kind='bar', title='Feature Importances'))


# ### Top 5 features for each classifier type:
# - Logistic regression: Sex, has_sibsp, Pclass, SibSp, is_child
# - Random forest classifier: Sex, Age, Fare, Pclass, Cabin
# - XGB classifier: Fare, Age, Cabin, title, Embarked

# ### Logistic Regression
# - Top 5 features Sex, has_sibsp, Pclass, SibSp, is_child

# In[ ]:


log_reg1 = LogisticRegressionCV(cv = 3, random_state = 0)
log_reg2 = LogisticRegressionCV(cv = 3, random_state = 0)

log_reg1.fit(train_x.drop(['Name', 'Ticket'], axis = 1), train_y)
log_reg2.fit(train_x[['Sex', 'has_sibsp', 'Pclass', 'SibSp', 'is_child']], train_y)

print('Using all the features with a logistic regression scored', log_reg1.score(train_x.drop(['Name', 'Ticket'], axis = 1), train_y))
print('Using the top 5 features with a logistic regression scored', log_reg2.score(train_x[['Sex', 'has_sibsp', 'Pclass', 'SibSp', 'is_child']], train_y))


# ### XGB Classifier
# - Top 5 features Fare, Age, Cabin, title, Embarked

# In[ ]:


## Set the grid search parameters for the xgb classifier to pass to GridSearchCV
parameters = {
    'n_estimators': [110],
    'max_depth': [5],
    'min_child_weight': [5],
    'gamma': [0],
    'subsample': [1],
    'colsample_bytree': [0.7],
    'reg_alpha': [0.1],
    'reg_lambda': [0],
    'learning_rate': [0.12]
}

## Instantiate the xgb classifier
xgbc = XGBClassifier(learning_rate =0.1, n_estimators=110, max_depth=5, min_child_weight=1, 
                     gamma=0, subsample=0.8, colsample_bytree=0.8, 
                     objective= 'binary:logistic', 
                     nthread=-1, seed=0, silent = False)
## Set up the k folds function to pass to GridSearchCV
kf = KFold(n_splits = 3, random_state = 0)
## Instantiate the grid search across the parameters and with cross validation on 3 folds
xgbc_gscv1 = GridSearchCV(xgbc, parameters, cv = kf, verbose = 1)
xgbc_gscv2 = GridSearchCV(xgbc, parameters, cv = kf, verbose = 1)
## Do the grid search and cv
xgbc_gscv1.fit(train_x.drop(['Name', 'Ticket'], axis = 1), train_y)
xgbc_gscv2.fit(train_x[['Fare', 'Age', 'Cabin', 'title', 'Embarked']], train_y)

print('XGB classifier with all features scored', xgbc_gscv1.best_score_)
print('XGB classifier with top 5 features scored', xgbc_gscv2.best_score_)


# ### Test the correlation between model predictions to see which ones to use for a voting classifier

# In[ ]:


from sklearn.model_selection import train_test_split

## First, I want to make subsets of the training data to use as input for prediction models
train_ins = []
test_ins = []
train_outs = []
test_outs = []

for tts_ind in range(0,10):
    train_in, test_in, train_out, test_out = train_test_split(train_x,
                                                              train_y, 
                                                              test_size = 0.33,
                                                              random_state = tts_ind)
    train_ins.append(train_in)
    test_ins.append(test_in)
    train_outs.append(train_out)
    test_outs.append(test_out)
    
#print("First 5 ids of split 1: train -", train_ins[0].head().index.tolist(), "; test -", train_outs[0].head().index.tolist())
#print("First 5 ids of split 2: train -", train_ins[1].head().index.tolist(), "; test -", train_outs[1].head().index.tolist())


# In[ ]:


## Next, I need to make predictions from multiple models using each subset of data

### make correlation vectors
lr_svc_corrs = []
lr_xgb_corrs = []
svc_xgb_corrs = []

### set up the models
lr_pred_corr = LogisticRegression(C = 0.7, random_state = 0)
svc_pred_corr = SVC(C = 0.7, kernel = 'linear', decision_function_shape = 'ovr', 
                    random_state = 0)
xgb_pred_corr = XGBClassifier(learning_rate =0.12, n_estimators=110, max_depth=5, 
                              min_child_weight=5, gamma=0, subsample=1,
                              reg_alpha = 0.1, reg_lambda = 0,
                              colsample_bytree=0.7, objective= 'binary:logistic',
                              seed=0, silent = False)

for pred_ind in range(0,10):
    ## Predict the logistic regression outcomes of the test set
    lr_pred_corr.fit(train_ins[pred_ind], train_outs[pred_ind])
    lr_preds_temp = lr_pred_corr.predict(test_ins[pred_ind])
    ## Predict the svc classifier outcomes of the test set
    svc_pred_corr.fit(train_ins[pred_ind], train_outs[pred_ind])
    svc_preds_temp = svc_pred_corr.predict(test_ins[pred_ind])
    ## Predict the xgb classifier outcomes of the test set
    xgb_pred_corr.fit(train_ins[pred_ind], train_outs[pred_ind])
    xgb_preds_temp = xgb_pred_corr.predict(test_ins[pred_ind])
    ## Append correlations to correlation vectors
    lr_svc_corrs.append(st.pearsonr(lr_preds_temp, svc_preds_temp)[0])
    lr_xgb_corrs.append(st.pearsonr(lr_preds_temp, xgb_preds_temp)[0])
    svc_xgb_corrs.append(st.pearsonr(svc_preds_temp, xgb_preds_temp)[0])
    
print('lr vs svc:', np.mean(lr_svc_corrs), lr_svc_corrs)
print('lr vs xgb:', np.mean(lr_xgb_corrs), lr_xgb_corrs)
print('svc vs xgb:', np.mean(svc_xgb_corrs), svc_xgb_corrs)


# In[ ]:


## Compare predictions
lr_stack = LogisticRegression(C = 0.7, random_state = 0)
svc_stack = SVC(C = 0.7, decision_function_shape = 'ovr', random_state = 0)
rfc_stack = RandomForestClassifier(random_state = 0, max_features = 15, 
                                   min_samples_split = 10, max_depth = 15, 
                                   n_estimators = 150, min_samples_leaf = 2)
xgb_stack = XGBClassifier(learning_rate =0.12, n_estimators=110, max_depth=5, 
                           min_child_weight=5, gamma=0, subsample=1,
                           reg_alpha = 0.1, reg_lambda = 0,
                           colsample_bytree=0.7, objective= 'binary:logistic',
                           seed=0, silent = False)
from sklearn.naive_bayes import GaussianNB
gnb_stack = GaussianNB()

def make_preds(train_in, train_out, test_in):
    lr_stack.fit(train_in, train_out)
    gnb_stack.fit(train_in, train_out)
    svc_stack.fit(train_in, train_out)
    rfc_stack.fit(train_in, train_out)
    xgb_stack.fit(train_in, train_out)
    lr_stack_preds = lr_stack.predict(test_in)
    svc_stack_preds = svc_stack.predict(test_in)
    gnb_stack_preds = gnb_stack.predict(test_in)
    rfc_stack_preds = rfc_stack.predict(test_in)
    xgb_stack_preds = xgb_stack.predict(test_in)
    stack_preds = pd.DataFrame({'lr_preds': lr_stack_preds,
                                #'gnb_preds': gnb_stack_preds,
                                'svc_preds': svc_stack_preds, 
                                'rfc_preds': rfc_stack_preds,
                                'xgb_preds':xgb_stack_preds})
    return stack_preds

preds_0 = make_preds(train_ins[0], train_outs[0], test_ins[0])
stacking_classifier = LogisticRegression(random_state = 0)
stacking_classifier.fit(preds_0, test_outs[0])
xgb_stack.fit(train_ins[0], train_outs[0])
for pred_ind in range(1,10):
    preds_1 = make_preds(train_ins[pred_ind], train_outs[pred_ind], test_ins[pred_ind])
    print('Score', 
          pred_ind, 
          'stack -', 
          round(stacking_classifier.score(preds_1, 
                                          test_outs[pred_ind]), 3), 
          '; XGBC -', 
          round(xgb_stack.score(test_ins[pred_ind],
                                test_outs[pred_ind]), 3),
          '; RFC -', 
          round(rfc_stack.score(test_ins[pred_ind], 
                                test_outs[pred_ind]), 3), 
          '; SVC -', 
          round(svc_stack.score(test_ins[pred_ind], 
                                test_outs[pred_ind]), 3),
          #'; GNB -', 
          #round(gnb_stack.score(test_ins[pred_ind], 
          #                      test_outs[pred_ind]), 3),
          '; LR -', 
          round(lr_stack.score(test_ins[pred_ind], 
                               test_outs[pred_ind]), 3))
    
print(sns.barplot(y = preds_0.columns, x = np.abs(stacking_classifier.coef_[0]), orient = 'h'))


# ### The stacking classifier worked best most of the time in pairwise comparisons on subsets of the original data.
#  - It was better than the XGB classifier alone for 5 of 8 subsets (1 tie).
#  - It was better than the RF classifier alone for 5 of 8 subsets (1 tie) .
#  - It was better than the SV classifier alone for 9 of 9 subsets .
#  - It was better than the GNB classifier alone for 9 of 9 subsets .
#  - It was better than the LR classifier alone for 9 of 9 subsets .
# ### It gave the highest score of all models 3 times out of 9 (with 1 tie) when using a mixture of XGB, SVC, RFC, and LRC. This ties XGB, so it's difficult to say if it's an improvement over just XGB.
# 
# Let's make a submission using our various models.

# In[ ]:




