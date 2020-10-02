#!/usr/bin/env python
# coding: utf-8

# # Summary

# In this kernel, classification based on simple decision rule is proposed. Using in total 2 features 'Title' and 'FamSize' = 1 + 'SibSp' + 'ParCh', the public score of 0.78947 is achieved. Adding 'Embarked' as a third feature on top allows to push the public score to 0.80382. Additionally, modeling using support vector machine (SVM) is considered. By comparing the predictions on the test set between the models relying on simple decision rules and SVM, insights into decision rules generated by SVM are obtained.

# The kernel is organized as follows. First, feature engineering is described. Then, exploratory data analysis (EDA) is performed. Building on top of EDA, models based on simple decision rules are derived. In the end, modeling using SVM is described.

# # Table of Contents
# * [Standard Routines](#standard)
# * [Feature Engineering](#feature)
#     * [Title, Family Size, Group Size, is Alone, has Cabin](#feature-1)
#     * [with Siblings, with Spouse, with Children, with Parents](#feature-2)
# * [Exploratory Data Analysis](#EDA)
#     * [Class 1](#EDA-1)
#     * [Class 2](#EDA-2)
#     * [Class 3](#EDA-3)
# * [Models Based on Simple Rules](#models)
#     * [Model 1](#model-1)
#     * [Model 2](#model-2)
#     * [Model 3](#model-3)
# * [SVM Modeling](#SVM)
#     * [Data Preparation](#SVM-1)
#     * [Training](#SVM-2)
#     * [Relationship to Model 3](#SVM-3)
# * [Conclusions](#conclusions)

# # Standard Routines <a class="anchor" id="standard"></a>

# For the sake of readability, we summarize all imports below.

# In[ ]:


import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from fancyimpute import KNN
from fancyimpute import IterativeImputer
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')


# We start with loading the train and test sets and building the joint train+test set.

# In[ ]:


# Load data
train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)

# Merge train and test sets
test.insert(1,'Survived',np.nan)
all = pd.concat([train, test])


# During the preparation of this kernel, few errors in the features 'SibSp' and 'ParCh' have been identified. We correct them below, although this is not going to change the accuracy of modeling and is done solely for the sake of more precise EDA.

# In[ ]:


# Perform corrections
corr_dict = {248: pd.Series([0,1], index=['SibSp', 'Parch'],),
             313: pd.Series([1,0], index=['SibSp', 'Parch'],),
             418: pd.Series([0,0], index=['SibSp', 'Parch'],),
             756: pd.Series([0,1], index=['SibSp', 'Parch'],),
             1041: pd.Series([1,0], index=['SibSp', 'Parch'],),
             1130: pd.Series([0,0], index=['SibSp', 'Parch'],),
             1170: pd.Series([2,0], index=['SibSp', 'Parch'],),
             1254: pd.Series([1,0], index=['SibSp', 'Parch'],),
             1274: pd.Series([1,0], index=['SibSp', 'Parch'],),
             539: pd.Series([1,0], index=['SibSp', 'Parch'],)
             }

all[['SibSp','Parch']] = all.apply(lambda s: corr_dict[s['PassengerId']]
    if s['PassengerId'] in [248,313,418,756,1041,1130,1170,1254,1274,539] else s[['SibSp','Parch']], axis = 1)


# # Feature Engineering <a class="anchor" id="feature"></a>

# For the models based on simple rules, only few features necessary. However, for the derivation of the models, a larger set of features is considered.

# ## Title, Family Size, Group Size, is Alone, has Cabin <a class="anchor" id="feature-1"></a>

# Following the standard practice, we create the 'Title' feature.

# In[ ]:


# Add Title
all['Title'] =  all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# Due to rare occurrence of some titles, we replace them by the more frequent ones.

# In[ ]:


# Replace rare titles
all.loc[all['Title'].isin(['Ms','Mlle']), 'Title'] = 'Miss'
all.loc[all['Title'].isin(['Mme','Lady','Dona','Countess']), 'Title'] = 'Mrs'
all.loc[all['Title'].isin(['Col','Major','Sir','Rev','Capt','Don','Jonkheer']), 'Title'] = 'Mr'
all.loc[(all['Title'] == 'Dr') & (all['Sex'] == 'male'),'Title'] = 'Mr'
all.loc[(all['Title'] == 'Dr') & (all['Sex'] == 'female'),'Title'] = 'Mrs'


# We add the feature representing the size of the family. The 'isAlone' feature identifies whether the person is traveling alone.

# In[ ]:


# Add Family Size and is-Alone
all['FamSize'] = all.apply(lambda s: 1+s['SibSp']+s['Parch'], axis = 1)
all['isAlone'] = all.apply(lambda s: 1 if s['FamSize'] == 1 else 0, axis = 1)


# Additionally, we create the feature representing the number of people sharing the same ticket. Also, we extract information about the availability of the cabin information.

# In[ ]:


# Add Group Size
ticket_counts = all['Ticket'].value_counts()
all['GrSize'] = all.apply(lambda s: ticket_counts.loc[s['Ticket']], axis=1)

# Add has-Cabin
all['Cabin'].fillna('U',inplace=True)
all['hasCabin'] = all.apply(lambda s: 0 if s['Cabin'] == 'U' else 1,axis = 1)


# ## with Siblings, with Spouse, with Children, with Parents <a class="anchor" id="feature-2"></a>

# Here, we create additional binary features.
# 
# * 'wSib': whether person is traveling ONLY with siblings
# * 'wSp': whether person is traveling ONLY with spouse
# * 'wCh': whether person is traveling with children
# * 'wPar': whether person is traveling with parents

# In[ ]:


# Add Family Name
all['Fname'] =  all.Name.str.extract('^(.+?),', expand=False)

# Search for passengers with siblings
Pas_wSib = []
all_x_0 = all[(all['SibSp'] > 0) & (all['Parch'] == 0)]
name_counts_SibSp = all_x_0['Fname'].value_counts()
for label, value in name_counts_SibSp.items():
    entries = all_x_0[all_x_0['Fname'] == label]
    if (entries.shape[0] > 1 and (not (entries['Title'] == 'Mrs').any())) or        (entries.shape[0] == 1 and entries['Title'].values[0] == 'Mrs'):
            Pas_wSib.extend(entries['PassengerId'].values.tolist())
    else:
        Pas_wSib.extend(             entries[(entries['Title'] == 'Miss')|(entries['GrSize'] == 1)]['PassengerId'].values.tolist())

# Search for Mrs-es with parents
Mrs_wPar = []
all_x_y = all[all['Parch'] > 0]
name_counts_Parch = all_x_y['Fname'].value_counts()
for label, value in name_counts_Parch.items():
    entries = all_x_y[all_x_y['Fname'] == label]
    if entries.shape[0] == 1:
        if entries['Title'].values[0] == 'Mrs' and entries['Age'].values[0] <= 30:
            Mrs_wPar.extend(entries['PassengerId'].values.tolist())

def get_features(row):

    features = pd.Series(0, index = ['wSib','wSp','wCh','wPar'])

    if row['PassengerId'] in Pas_wSib:
        features['wSib'] = 1
    else:
        if (row['SibSp'] != 0) & (row['Parch'] == 0):
            features['wSp'] = 1
        else:
            if  ( (row['Title']=='Mrs')&(not row['PassengerId'] in Mrs_wPar) )|                 ( (row['Title']=='Mr')&(not row['PassengerId'] == 680)&
                                        ( ((row['Pclass']==1)&(row['Age']>=30))|
                                          ((row['Pclass']==2)&(row['Age']>=25))|
                                          ((row['Pclass']==3)&(row['Age']>=20)) ) ):
                features['wCh'] = 1
            else:
                features['wPar'] = 1

    return features

all[['wSib','wSp','wCh','wPar']] = all.apply(lambda s: get_features(s) if s['isAlone'] == 0 else [0,0,0,0], axis = 1)


# The features 'wSib', 'wSp', 'wPar', 'wCh' and 'isAlone' are mutually exclusive, where 'wSib' + 'wSp' + 'wCh' + 'wPar'  + 'isAlone' = 1 holds. Obviously, 'wCh' == 1 identifies parents. 'wPar' == 1, in turn, can help to identify children, although it also identifies adults travelling with parents. To distinguish children from adults travelling with parents, we will use the 'Title' feature: children are said to be those with title 'Master' or title 'Miss' and 'wPar' == 1.

# Before proceeding further, let us clean our dataframe a bit. We remove 'Fname', 'Name', 'Cabin', 'Ticket', 'Fare', 'SibSp' and 'Parch' features as they are not used for the rest of the kernel.

# In[ ]:


all = all.drop(['Fname','Name','Cabin','Ticket','Fare','SibSp','Parch'], axis = 1)


# # Exploratory Data Analysis <a class="anchor" id="EDA"></a>

# We perform EDA in order to derive the simple rules for the models. In the following, survival statistics for each passenger class are considered.

# ## Class 1 <a class="anchor" id="EDA-1"></a>

# In[ ]:


all[all['Pclass'] == 1].groupby(['Title','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])


# 'Mean' defines the survival rate: the percentage of the survived passengers in the train set, where 'NaN' corresponds to the case when all samples of a particular group are in the test set. 'count' and 'size' stand for the size of the train and joint train+test sets for particular groups, i.e. 'size' - 'count' is the size of the test set. The findings here: adult females and children ('Title' != 'Mr') mostly survive. The fate of the remaining adult males is different: the survival rate is somewhere between 0.23 (parents) and 0.46 (traveling only with spouses).

# As already mentioned in other kernels, having groups of people with high uncertainity in the survival (survival rate close to 0.5) is going to limit the prediction accuracy of the classifier. Hence, an effort has to be put to resolve such uncertainties.  Let us check, whether 'hasCabin' feature can help us to do so for the adult males.

# In[ ]:


all[(all['Pclass'] == 1)&(all['Title'] == 'Mr') ].groupby(['hasCabin','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])


# The 'hasCabin' feature has some predictive power as it differentiates the survival rate for the ones traveling alone: 0.39 and 0.22 for those with and without cabin information, respectively. Unfortunately, as both survival rates are less than 0.5, this information alone is not helpful for building decision rules. Let us now move to class 2.

# ## Class 2 <a class="anchor" id="EDA-2"></a>

# In[ ]:


all[all['Pclass'] == 2].groupby(['Title','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])


# Similarly to class 1, children all survive. The survival rate of adult females is close to 1, with the exception of 0.77 for ones traveling only with spouses. The fate of adult males with the survival rates 0-0.1 is certain: they are more likely to perish. Notice, the survival for those traveling alone and only with siblings is similar, i.e. these groups can be merged (which we will later do when applying SVM).

# Let us recapture first, what we got so far. For classes 1 and 2, all adult females and children ('Title' != 'Mr') are more likely to survive and the remaining adult males are more likely to perish. Now, we consider class 3.

# ## Class 3 <a class="anchor" id="EDA-3"></a>

# In[ ]:


all[all['Pclass'] == 3].groupby(['Title','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])


# In class 3, the situation is quite different. First, children have fairly low survival rates between 0.34-0.37. Adult females have the rates greater than 0.5, with the exception of 0.42 for ones traveling with children (the survival rate of 0 for those with wSib == 1 is statistically insignificant). Adult males have the lowest survival rates of 0-0.15, surprisingly however, this is slightly greater than that for class 2.

# Similarly to classes 1 and 2, the fate of adult males is certain: they are more likely to perish. The fate of children and adult females, however, is much less certain. Let us try to resolve this uncertainty by using the 'FamSize' feature.

# In[ ]:


all[(all['Pclass'] == 3)&(all['Title'] != 'Mr')].groupby(['Title','FamSize'])['Survived'].agg(['count','size','mean'])


# According to the statistic, the ones having 'FamSize' > 4 have significantly lower chances to survive than the remaining ones. Let us clean the statistic by introducing two family size bins: less than or equal to / greater than 4.

# In[ ]:


# Make FamSize bins
all['FamSizeBin'] = pd.cut(all['FamSize'], bins = [0,4,11], labels = False)
all = all.drop(['FamSize'], axis = 1)


# In[ ]:


all[(all['Pclass'] == 3)&(all['Title'] != 'Mr')].groupby(['Title','FamSizeBin','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])


# As one can see, the family size feature helps to clarify the fate of children and adult females. The survival rate of the ones with families greater than 4 is remarkably low: 0.05-0.13. For all the rest, the forecast is now rather positive: children have the survival rates of 0.62 (female) and 1 (male), and all adult females have the rates above 0.5.

# ## Models Based on Simple Rules  <a class="anchor" id="models"></a>

# Summarizing the conducted study, we can build our models. We start with the simplest one relying on in total 2 features.

# ## Model 1 <a class="anchor" id="model-1"></a>

# <b>Model 1.</b>
# <cite> All adult males are deemed to perish as well as the ones in class 3 with families greater than 4. The rest all survive. </cite>

# In[ ]:


def get_survived_1(row):
    if row['Pclass'] in [1,2]:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSizeBin'] == 1:
            survived = 0
        else:
            survived = 1

    return survived


# Let us apply our model to the train and test sets and see the accuracy.

# In[ ]:


# Form train and test sets
X_train = all.iloc[:891,:]
X_test = all.iloc[891:,:]
y_train = all.iloc[:891,:]['Survived']

# Make predictions (train)
y_train_hat = X_train.apply(lambda s: get_survived_1(s), axis = 1)

# Make predictions (test)
predictions = pd.DataFrame( {'PassengerId': test['PassengerId'], 'Survived': 0} )
predictions['Survived'] = X_test.apply(lambda s: get_survived_1(s), axis = 1)
predictions.to_csv('submission-1.csv', index=False)

# Train score
score = metrics.accuracy_score(y_train_hat, y_train)
print('Train Accuracy: {}'.format(score))


# Checking the test set predictions on Kaggle yields the accuracy of 0.78947, which is already a decent score, improving upon the simple gender-based model and achieving the same accuracy as the kernels using complex learning algorithms (e.g. Random Forest). Our proposed model, however, relies on simple decision rules, hence it is less likely to overfit on the public part of the test .

# ## Model 2 <a class="anchor" id="model-2"></a>

# Now, let try to improve Model 1 by adding the 'Embarked' feature on top. As we have seen, females in class 3 with family sizes less than 5 have still high uncertainty in the survival. Let us print their survival statistic for different embarkment ports.

# In[ ]:


all[(all['Pclass'] == 3)&(all['Title'] != 'Mr')&(all['FamSizeBin'] == 0)].groupby(['Title','Embarked'])['Survived'].agg(['count','size','mean'])


# In general, 'Embarked' feature has weak predictive power with the exception for those with 'Title' == 'Miss': with survival rate 0.44, the Misses embarked in Southampton (S) have now more chances to perish. This judgment seems to be statistically significant, as it is based on 39 samples. Based on this new information, we create the new model.

# <b>Model 2.</b>
# <cite> All adult males are deemed to perish as well as the ones in class 3 with families greater than 4. Also, Misses in class 3 embarked in S perish. The rest all survive. </cite>

# In[ ]:


def get_survived_2(row):
    if row['Pclass'] in [1,2]:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSizeBin'] == 1 or (row['Title'] == 'Miss' and row['Embarked'] == 'S'):
            survived = 0
        else:
            survived = 1

    return survived


# Let us update the predictions on the train and test sets to see how well Model 2 performs.

# In[ ]:


# Make predictions (train)
y_train_hat = X_train.apply(lambda s: get_survived_2(s), axis = 1)

# Make predictions (test)
predictions['Survived'] = X_test.apply(lambda s: get_survived_2(s), axis = 1)
predictions.to_csv('submission-2.csv', index=False)

# Train score
score = metrics.accuracy_score(y_train_hat, y_train)
print('Train Accuracy: {}'.format(score))


# Checking the test set predictions on Kaggle yields now the accuracy of 0.80382, which at the time of writing this kernel is within the top 12 percent results on the public leaderboard. Need to be said, that we are possibly dealing here with a slight overfit on the public part of the test set. We elaborate more on this in the following.

# ## Model 3 <a class="anchor" id="model-3"></a>

# As we have seen, male children (Master)  in class 3 with family sizes less than 5 have all survived. Normally, we would expect this also to be true for female children, irrespective of embarkment port. Let us print the survival statistic to confirm our hypothesis.

# In[ ]:


all[(all['Pclass'] == 3)&(all['Title'] == 'Miss')&(all['FamSizeBin'] == 0)].groupby(['Title','wPar','Embarked'])['Survived'].agg(['count','size','mean'])


# As we can see, the female children embarked in S are more likely to survive, although, we are dealing here with a small number of samples. Let us correct the model and see whether the accuracy is going to increase.

# <b>Model 3.</b>
# <cite> All adult males are deemed to perish as well as the ones in class 3 with families greater than 4. Also, Misses in class 3, non-chlidren and embarked in S perish. The rest all survive. </cite>

# In[ ]:


def get_survived_3(row):
    if row['Pclass'] in [1,2]:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSizeBin'] == 1 or         (row['Title'] == 'Miss' and row['Embarked'] == 'S' and row['wPar'] == 0):
            survived = 0
        else:
            survived = 1

    return survived


# In[ ]:


# Make predictions (train)
y_train_hat = X_train.apply(lambda s: get_survived_3(s), axis = 1)

# Make predictions (test)
predictions['Survived'] = X_test.apply(lambda s: get_survived_3(s), axis = 1)
predictions.to_csv('submission-3.csv', index=False)

# Train score
score = metrics.accuracy_score(y_train_hat, y_train)
print('Train Accuracy: {}'.format(score))


# As we can see, the training accuracy increases, while the public test score decreases to 0.79904, which 1 correct classification less than Model 2. Model 3 however creates the test set predictions which are almost identical (up to 1 classification entry) to those which will be obtained with SVM later on. That been said, Models 2 and 3 have almost identical accuracy and to decide between the two, one need to test the accuracy on the private part of test set. We leave this out to the reader.

# ## SVM Modeling <a class="anchor" id="SVM"></a>

# In this part of the kernel, SVM modeling is considered. We employ the same set of features which was used for EDA: 'Title', 'Age', 'FamSizeBin', 'Embarked', ('isAlone' + 'wSib'), 'wSp', 'wCh', 'wPar' and 'hasCabin'. 

# ## Data Preparation <a class="anchor" id="SVM-1"></a>

# We start with preprocessing the features. First, we apply dummy encoding. To reduce the number of features, we map K categories onto (K - 1)-sized space, where K is a natural number.

# In[ ]:


# Select and convert categorical features into numerical ones (1)
all['Sex'] = all['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
all['Embarked'].fillna(all['Embarked'].value_counts().index[0], inplace=True)
all_dummies =  pd.get_dummies(all, columns = ['Title','Pclass','Embarked'],                                 prefix=['Title','Pclass','Embarked'], drop_first = True)
all_dummies = all_dummies.drop(['PassengerId','Survived'], axis = 1)


# In our kernel, we use KNN age imputation.

# In[ ]:


# KNN imputation
all_dummies_i = pd.DataFrame(data=KNN(k=3, verbose = False).fit_transform(all_dummies).astype(int),
                            columns=all_dummies.columns, index=all_dummies.index)


# We group 'isAlone' and 'wSib' together. Also we get rid of the 'Sex' (redundant) and 'GrSize' (not used) features.

# In[ ]:


# Convert categorical features into numerical ones (2)
all_dummies_i['isAlwSib'] = all_dummies_i.apply(lambda s: 1 if (s['isAlone'] == 1)|(s['wSib'] == 1) else 0 ,axis = 1)
all_dummies_i = all_dummies_i.drop(['isAlone','wSib','Sex','GrSize'], axis = 1)


# In order to apply SVM correctly, 'Age' feature has to be normalized. First, we re-build train and test sets.

# In[ ]:


# Form train and test sets
X_train = all_dummies_i.iloc[:891,:]
X_test = all_dummies_i.iloc[891:,:]


# Now we calculate the scaling based on the train set and apply it to both train and test sets.

# In[ ]:


# Perform scaling
scaler = StandardScaler()
scaler.fit(X_train[['Age']])
X_train['Age'] = scaler.transform(X_train[['Age']])
X_test['Age'] = scaler.transform(X_test[['Age']])


# ## Training <a class="anchor" id="SVM-2"></a>

# Now, we are ready to apply SVM. First, let us define the cross-validation strategy. We form for 80/20 percent train/test splits, in total 10 times. 

# In[ ]:


# Cross-validation parameters
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)


# To find the parameter of SVM (regularization factor 'C'), we use an exhaustive grid search.

# In[ ]:


# Grid search parameters
svm_grid = {'C': [10,11,12,13,14,15,16,17,18,19,20], 'gamma': ['auto']}
svm_search = GridSearchCV(estimator = SVC(), param_grid = svm_grid, cv = cv, refit=True, n_jobs=1)


# We run our grid search and print the cross-validation statistics: mean and standard deviation.

# In[ ]:


# Apply grid search
svm_search.fit(X_train, train['Survived'])
svm_best = svm_search.best_estimator_
print("Cross-validation accuracy: {}, standard deviation: {}, with parameters {}"
       .format(svm_search.best_score_, svm_search.cv_results_['std_test_score'][svm_search.best_index_],
               svm_search.best_params_))


# Now we make the predictions on the train and test sets and print the train set accuracy.

# In[ ]:


y_train_hat = svm_best.predict(X_train)
print('Train Accuracy: {}'
        .format(metrics.accuracy_score(y_train_hat, y_train)))

predictions['Survived'] = svm_best.predict(X_test)
predictions.to_csv('submission-svm.csv', index=False)


# As we can see, the train accuracy is slightly greater than that we achieved with Model 3 (0.8418). Uploading the predictions to Kaggle yields the accuracy of 0.80382, which is as good as the best accuracy we got with Model 2.

# ## Relationship to Model 3 <a class="anchor" id="SVM-3"></a>

# As we already mentioned, Model 3 and SVM yield almost identical predictions on the test. The difference between the two is in a single entry which is a young Miss in class 3, non-child and embarked in S: SVM decides her rather to survive. Thus, the SVM decision rule can be interpreted as a corrected Model 3, where now all females in class 3 younger than 18 survive.

# <b>SVM rule.</b>
# <cite> All adult males are deemed to perish as well as the ones in class 3 with families greater than 4. Also, 18 and older Misses in class 3 embarked in S perish. The rest all survive. </cite>

# In[ ]:


def get_survived_svm_rule(row):
    if row['Pclass'] in [1,2]:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSizeBin'] == 1 or         (row['Title'] == 'Miss' and row['Embarked'] == 'S' and row['Age'] >= 18):
            survived = 0
        else:
            survived = 1

    return survived


# The function above summarizes the SVM decision rule, feel free to test it by forking the script. Note, that Model 3 and SVM differ only in single prediction which is for the public part of the test set and for the private part of the test set the predictions are identical. Additional insight here: from the large set of features which were used to build the SVM classifier, in total only 4 features 'Title', 'Age', 'FamSizeBin' and 'Embarked' were important for the test set accuracy.

# # Conclusions <a class="anchor" id="conclusions"></a>

# In this kernel, 3 models based on simple decision rules and SVM modeling were proposed. It has been shown, that the public score of 0.78947 can be achieved by using Model 1 based on in total only 2 features: 'Title' and 'FamSize'. Models 2 and 3 employ 'Embarked' feature in addition and yield the accuracy of 0.80382 and 0.79904, respectively, where Model 3 is potentially more robust. SVM achieves 0.80382 and produces the predictions on the test set almost identical to that of Model 3, where to increase the accuracy, it exploits the 'Age' feature.