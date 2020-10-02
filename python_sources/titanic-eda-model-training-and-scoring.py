#!/usr/bin/env python
# coding: utf-8

# Titanic is such a trategy and mystery. There are lots of theories and anecdotes about why "the ship that will never sink" actually sunk as well as who are the survivors VS non-survivors. It would be interesting to see how data analysis and machine learning algorithm can give us more insights on the characteristics and likelihood of survival of Titanic shipwreck. 
# 
# In this analysis I would like to focus on the following topics:
# * Data processing and feature engineering
# * Model training and selection
# * Parameter/hyperparameter tuning for selected model

# ## 0.Load data and basic feature engineering
# As the first step, let's load titanic train and test data and take a look at the data records.

# In[ ]:


dfTrain = pd.read_csv('/kaggle/input/titanic/train.csv')
dfTest = pd.read_csv('/kaggle/input/titanic/test.csv')
print(dfTrain.shape)
print(dfTest.shape)


# In[ ]:


print('There are {t} passengers in train data, of which {s} made it and {d} did not'.format(t = dfTrain.PassengerId.nunique(), s = dfTrain[dfTrain['Survived'] == 1].PassengerId.nunique(), d = dfTrain[dfTrain['Survived'] == 0].PassengerId.nunique()))


# In[ ]:


dfTrain.head()


# In[ ]:


dfTest.head()


# ### 0.0 Describe train data and process missing values/outliers

# In[ ]:


print(dfTrain.columns)
dfTrain.describe()


# Some initial observations:
# * From the statistics above there doesn't seem to be obvious outliers
# * Pclass is taken as a numeric variable from the original dataset, but apparently it should be categorical. Let's turn it into a categorical variable

# In[ ]:


# Turn Pclass into a cateogrical variable
dfTrain['Pclass'] = dfTrain.Pclass.astype('category')


# Next let's check for missing values and come up with strategies to fill them in.

# In[ ]:


# Show missing value counts by columns
dfTrain.isnull().sum()


# There are three variables with missing values: Age, Cabin and Embarked. Here are my thoughts on strategies to handle missing values:
# 1. Age: since Pclass and Sex do not have missing values, we can calculate average age group by Pclass and Sex and fill in missing values accordingly
# 2. Cabin: this variable may indicate where the passenger's cabin room is in the ship, and may have an impact on survival rate. But since people could move around in the ship this variable may not contribute too much useful information, especially given the # of missing values. But let's take a look and see what we can find
# 3. Embarked: this should be related to Fare. We can take a look at Fare range for each existing Embarked variable and decide which Embarked value should be given the Fare

# In[ ]:


dfTrain.groupby(['Sex', 'Pclass']).agg({'Age': ['mean', 'median', 'min', 'max']}).reset_index()


# In[ ]:


# assign passengers with missing age the average age of gender and class
avgAge = dfTrain.groupby(['Sex', 'Pclass']).Age.mean().reset_index()
dfTrain = dfTrain.merge(avgAge, 'left', on = ['Sex', 'Pclass'], suffixes = ('_orig', '_avg'))
dfTrain.head(5)


# In[ ]:


# Fill in average age if original age value is NaN
def fill_in_avg_age(row):
    if np.isnan(row['Age_orig']):
        return row['Age_avg']
    else:
        return row['Age_orig']

dfTrain['Age'] = dfTrain.apply(lambda row: fill_in_avg_age(row), axis = 1) 


# In[ ]:


# Double check if things are filled in correctly
dfTrain[dfTrain.Age_orig.isnull()].head()


# Next let's examine Cabin. It seems that Cabin assignment may be related to Pclass so we can also take a look at missing values by Pclass.

# In[ ]:


print(dfTrain.Cabin.unique())
print('There are {n} different unique Cabin values'.format(n = dfTrain.Cabin.nunique()))
dfTrain[dfTrain.Cabin.isnull()].groupby('Pclass').PassengerId.nunique()


# There are 147 different values of Cabin, and they are in the format of 'X00'. Also the upper class tickets have much less missing values for Cabin, indicating that upper class passengers have an assigned room on the ship while lower class passengers may not (just standby?). We can extract the first letter of Cabin variable and find out.

# In[ ]:


# Fill unknown Cabin with 'U'
dfTrain.loc[dfTrain.Cabin.isnull(), 'Cabin'] = 'U'

# extract first letter of Cabin to indicate location on the ship
dfTrain['CabinLoc'] = dfTrain['Cabin'].str[0]

# assign an indicator to show if the passenger has Cabin assignment or not
def assign_cabin_ind(row):
    if row['Cabin'] == 'U':
        return 0
    else:
        return 1

dfTrain['CabinInd'] = dfTrain.apply(lambda x: assign_cabin_ind(x), axis = 1).astype('category')


# In[ ]:


print(dfTrain.groupby('CabinLoc').PassengerId.nunique().reset_index())
print(dfTrain.groupby(['Pclass', 'CabinInd']).PassengerId.nunique().reset_index())
print(dfTrain.groupby(['Pclass', 'CabinInd']).agg({'PassengerId': 'count', 'Fare': 'mean'}).reset_index())


# In[ ]:


dfTrain['CabinInd'] = dfTrain['CabinInd'].astype('category')


# Next let's find out missing Embarked from fare range. I'm going to examine the range of Fare for each embarkment port and use 15th and 85th percentile as lower and higher ends. For passengers missing the port, if their Fare price falls within the range for a port, the port will be assigned to them. The order of assignment will be C, S, Q.

# In[ ]:


# assign embark based on people's fare
dfTrain.groupby(['Embarked']).agg({'Fare': ['min', 'mean', 'max']}).reset_index()


# In[ ]:


dfTrain['CLow'] = dfTrain[dfTrain['Embarked'] == 'C'].Fare.quantile(0.15)
dfTrain['CHigh'] = dfTrain[dfTrain['Embarked'] == 'C'].Fare.quantile(0.85)
dfTrain['QLow'] = dfTrain[dfTrain['Embarked'] == 'Q'].Fare.quantile(0.15)
dfTrain['QHigh'] = dfTrain[dfTrain['Embarked'] == 'Q'].Fare.quantile(0.85)
dfTrain['SLow'] = dfTrain[dfTrain['Embarked'] == 'S'].Fare.quantile(0.15)
dfTrain['SHigh'] = dfTrain[dfTrain['Embarked'] == 'S'].Fare.quantile(0.85)


# In[ ]:


dfTrain.loc[dfTrain.Embarked.isnull(), 'Embarked'] = 'U'


# In[ ]:


def assign_missing_embarked(row):
    if row['Embarked'] != 'U':
        return row['Embarked']
    else:
        if row['Fare'] <= row['CHigh'] and row['Fare'] >= row['CLow']:
            return 'C'
        elif row['Fare'] <= row['SHigh'] and row['Fare'] >= row['SLow']:
            return 'S'
        elif row['Fare'] <= row['QHigh'] and row['Fare'] >= row['QLow']:
            return 'Q'
        else:
            return 'U'

dfTrain['Embarked_clean'] = dfTrain.apply(lambda x: assign_missing_embarked(x), axis = 1)


# In[ ]:


dfTrain.loc[dfTrain['Embarked'] == 'U', ['Embarked', 'Embarked_clean']]


# Now that we have examined and cleaned up the data, let's take a look of columns and keep/rename of those we want to keep.

# In[ ]:


dfTrain = dfTrain.drop(columns = ['Age_orig', 'Embarked', 'Age_avg', 'Cabin', 'CLow', 'CHigh', 'QLow', 'QHigh', 'SLow', 'SHigh'])
dfTrain = dfTrain.rename(columns = {'Embarked_clean': 'Embarked'})
print(dfTrain.columns)


# In[ ]:


print(dfTrain.isnull().sum())
print(dfTrain.shape)


# ### 0.1 More featuring engineering
# There are two variables that may give out more information about the passegners: Name and Ticket.
# Let's look at ticket first. Ideally the ticket information may tell you important information such as where the passenger boarded/unboarded and the starting port and destination port, etc. But unfortunately this time I don't have enough knowledge to know the coding of ticket numbers for this data set so won't be able to find out pattern (although I did notice a 'Paris' in one ticket number) so I'll pass this one for now. 
# 
# As for Name variable, I noticed I can extract title from the names, and for married women their own names and husbands' names, so maybe it is possible to know more detailed background of passengers (like if they are noble, military officers, religion/academic professionals or civilians).

# In[ ]:


# code to extract titles from names
def extract_title(row):
    return row['Name'].split(',')[1].split('.')[0].strip()

dfTrain['title'] = dfTrain.apply(lambda x: extract_title(x), axis = 1)


# In[ ]:


dfTrain.groupby('title').PassengerId.nunique()


# In[ ]:


# categorize titles into Military, Religion, Noble and Civilian
def categorize_titles(row):
    if row['title'] in ['Capt', 'Col', 'Major']:
        return 'Military'
    elif row['title'] in ['Rev', 'Dr']:
        return 'Religion'
    elif row['title'] in ['Don', 'Dona', 'Jonkheer', 'Lady', 'Master', 'Sir', 'the Countess']:
        return 'Noble'
    else:
        return 'Civilian'

dfTrain['TitleCate'] = dfTrain.apply(lambda x: categorize_titles(x), axis = 1)


# In[ ]:


dfTrain.groupby('TitleCate').PassengerId.count()


# From the Name it is also possible to extract names of husbands from married female passengers (Mrs's) and check male passengers in a pool of married gentlemen to indicate if the husband is onboard with the wife. 

# In[ ]:


import re
def extract_names(row):
    if row['title'] in ['Mrs', 'the Countess']:
        s = row['Name'].split(',')[1]
        return re.sub('^.*\((.*?)\)[^\(]*$', '\g<1>', s)
    else:
        return row['Name'].split(',')[1].split('.')[1].strip() + ' ' + row['Name'].split(',')[0]
        

dfTrain['RealName'] = dfTrain.apply(lambda x: extract_names(x), axis = 1)


# In[ ]:


dfTrain.head()


# In[ ]:


# for Mrs's, extract their husbands name and create a list of husband names
def extract_husband_name(row):
    if row['title'] == 'Mrs':
        return row['Name'].split(',')[1].split('.')[1].split('(')[0].strip() + ' ' + row['Name'].split(',')[0].strip()
    else:
        return 'Unknown'
    
dfTrain['HusbandName'] = dfTrain.apply(lambda x: extract_husband_name(x), axis = 1)


# In[ ]:


husband_list = dfTrain[dfTrain['HusbandName'] != 'Unknown'].HusbandName.tolist()
name_list = dfTrain['RealName'].tolist()


# In[ ]:


dfTrain.head()


# In[ ]:



def assign_couple_onboard_ind(row):
    if row['title'] == 'Mrs':
        if row['HusbandName'] in name_list:
            return 1
        else:
            return 0
    else:
        if row['RealName'] in husband_list:
            return 1
        else:
            return 0

dfTrain['CoupleOnboardInd'] = dfTrain.apply(lambda x: assign_couple_onboard_ind(x), axis = 1)


# In[ ]:


dfTrain['CoupleOnboardInd'] = dfTrain['CoupleOnboardInd'].astype('category')


# In[ ]:


dfTrain.groupby(['CoupleOnboardInd']).PassengerId.count()


# In[ ]:


dfTrain.head()


# In summary, after more feature engineering, I can add three more variables to the original data set: CabinLoc, CabinInd, title, TitleCate and CoupleOnboardInd. I want to drop name related variables and Ticket because I don't think they could help with further EDA and modeling/scoring. The final data set looks like below:

# In[ ]:


dfTrain = dfTrain.drop(columns = ['Name', 'RealName', 'HusbandName', 'Ticket'])


# SibSp and Parch can determine # travel companions and family size

# In[ ]:


dfTrain['TravelCompanionSize'] = dfTrain['SibSp'] + dfTrain['Parch']
def travel_companions(row):
    if row['TravelCompanionSize'] == 0:
        return 'Single Traveler'
    elif row['TravelCompanionSize'] <= 4:
        return 'Small Travel Group'
    else:
        return 'Big Travel Group'

def family_size(row):
    if row['Parch'] == 0:
        return 'Not with family'
    elif row['Parch'] <= 4:
        return 'Small family'
    else:
        return 'Big family'

dfTrain['TravelType'] = dfTrain.apply(lambda x: travel_companions(x), axis = 1)
dfTrain['FamilySize'] = dfTrain.apply(lambda x: family_size(x), axis = 1)


# In[ ]:


dfTrain.head()


# ## 1. More EDA and visualization

# Now that I have the model-ready data set, I would like to create some visualizations to further look at details of the data. There are two topics I'm interested:
# * General profiles of passengers
# * Characteristics of survivors vs non-survivors

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


f, axes = plt.subplots(3, 2, figsize = (15, 20))
sns.countplot(x = 'Pclass', hue = 'Sex', data = dfTrain, ax = axes[0, 0])
sns.boxplot(x = "Pclass", y = "Age", hue = "Sex", data = dfTrain, ax = axes[0, 1])
sns.stripplot(x = 'Pclass', y = 'SibSp', jitter = False, data = dfTrain, ax = axes[1, 0])
sns.stripplot(x = 'Pclass', y = 'Parch', jitter = False, data = dfTrain, ax = axes[1, 1])
sns.boxplot(x = 'Pclass', y = 'Fare', data = dfTrain, ax = axes[2, 0])
sns.boxplot(x = 'Embarked', y = 'Fare', data = dfTrain, ax = axes[2, 1])
f.show()


# In general, there are more male and lower class passengers. Upper class passengers tend to be older and have smaller family size/less travel companions.

# In[ ]:


f, axes = plt.subplots(2, 4, figsize = (30, 15))
sns.countplot(x = 'Sex', hue = 'Survived', data = dfTrain, ax = axes[0, 0])
sns.countplot(x = 'Pclass', hue = 'Survived', data = dfTrain, ax = axes[0, 1])
sns.boxplot(x = "Survived", y = "Age", hue = "Sex", data = dfTrain, ax = axes[0, 2])
sns.countplot(y = 'CabinLoc', hue = 'Survived', order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], data = dfTrain, ax = axes[0, 3])
sns.countplot(x = 'TravelType', hue = 'Survived', data = dfTrain, ax = axes[1, 0])
sns.countplot(x = 'FamilySize', hue = 'Survived', data = dfTrain, ax = axes[1, 1])
sns.countplot(x = 'CoupleOnboardInd', hue = 'Survived', data = dfTrain, ax = axes[1, 2])
sns.countplot(x = 'TitleCate', hue = 'Survived', data = dfTrain, ax = axes[1, 3])
f.show()


# The survivors may have the following characteristics:
# 1. Upper class/noble female passengers between 20 - 40
# 2. On deck B-F
# 3. Travel with smaller group or smaller family size
# 4. Their spouse/partner is on deck as well
# 
# Let's fit some models and see if model results align with what we observed

# ## 2.Train models

# ### 2.0 Process test data
# First I want to decide what variables to include and categorize them. Then I want to write up a function to process data and do it for test data set as well.

# In[ ]:


dfTest.describe()


# In[ ]:


dfTest.isnull().sum()


# In[ ]:


# data processing function to feature engineer test data
def feature_engineering(dat):
   df = dat.copy()
   
   # fill missing Age
   avgAge = df.groupby(['Sex', 'Pclass']).Age.mean().reset_index()
   df = df.merge(avgAge, 'left', on = ['Sex', 'Pclass'], suffixes = ('_orig', '_avg'))
   df['Age'] = df.apply(lambda row: fill_in_avg_age(row), axis = 1) 
   
   # fill missing Fare
   avgFare = df.groupby('Embarked').Fare.mean().reset_index()
   df = df.merge(avgFare, 'left', on = 'Embarked', suffixes = ('_orig', '_avg'))
   def fill_fare(row):
       if np.isnan(row['Fare_orig']):
           return row['Fare_avg']
       else:
           return row['Fare_orig']
   df['Fare'] = df.apply(lambda x: fill_fare(x), axis = 1)
   
   # fill missing Cabin
   df.loc[df.Cabin.isnull(), 'Cabin'] = 'U'
   df['CabinLoc'] = df['Cabin'].str[0]
   df['CabinInd'] = df.apply(lambda x: assign_cabin_ind(x), axis = 1).astype('category')
   
   # add feature engineered columns
   df['title'] = df.apply(lambda x : extract_title(x), axis = 1)
   df['TitleCate'] = df.apply(lambda x: categorize_titles(x), axis = 1)
   df['RealName'] = df.apply(lambda x: extract_names(x), axis = 1)
   df['HusbandName'] = df.apply(lambda x: extract_husband_name(x), axis = 1)
   husband_list = df[df['HusbandName'] != 'Unknown'].HusbandName.tolist()
   name_list = df['RealName'].tolist()
   df['CoupleOnboardInd'] = df.apply(lambda x: assign_couple_onboard_ind(x), axis = 1)
   
   # final clean-up
   df['Pclass'] = df['Pclass'].astype('category')
   df['CabinInd'] = df['CabinInd'].astype('category')
   df['CoupleOnboardInd'] = df['CoupleOnboardInd'].astype('category')
   df = df.drop(columns = ['Age_orig', 'Age_avg', 'Fare_orig', 'Fare_avg', 'Name', 'RealName', 'HusbandName', 'Ticket'])
   
   return df
   


# In[ ]:


dfTestClean = feature_engineering(dfTest)
dfTestClean.head()


# ### 2.1 Train baseline models and pick the best model for further tuning
# Next let's split the train data for training and validation and batch-train a bunch of models

# In[ ]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import time as t


# In[ ]:


colID = ['PassengerId']
colLabel = ['Survived']
colNum = ['Age', 'SibSp', 'Parch', 'Fare']
colCat = ['Sex', 'Pclass', 'CabinLoc', 'CabinInd', 'Embarked', 'title', 'TitleCate', 'CoupleOnboardInd']
y = dfTrain['Survived'].astype('category')
X = dfTrain[colNum + colCat]


# In[ ]:


XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size = 0.15, random_state = 777, stratify = y)


# In[ ]:


# Center and scale numeric variables and one hot coding for categorical variables
# train encoders on training data and apply it on validation and test data
scaler = StandardScaler().fit(XTrain[colNum])
encoder = OneHotEncoder(handle_unknown = 'ignore').fit(XTrain[colCat])
def apply_scaler_encoder(dat):
    
    df = dat.copy()
    print('Shape of original data')
    print(df.shape)
    dfScaled = scaler.transform(df[colNum])
    dfEncoded = encoder.transform(df[colCat]).toarray()
    dfFinal = np.concatenate([dfScaled, dfEncoded], axis = 1)
    print('Shape of processed data')
    print(dfFinal.shape)
    
    return dfFinal


# In[ ]:


XTrainFinal = apply_scaler_encoder(XTrain)
XValidFinal = apply_scaler_encoder(XValid)


# Then write up a function to batch train and evaluate several models. Pick the best performing one for parameter/hyperparameter tuning later.

# In[ ]:


modelsToFit = {
    'Logistic Regression': LogisticRegression(random_state = 777),
    'SVM': SVC(random_state = 777, probability = True),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state = 777),
    'Random Forest': RandomForestClassifier(random_state = 777),
    'AdaBoost': AdaBoostClassifier(random_state = 777),
    'GBT': GradientBoostingClassifier(random_state = 777),
    'XGB': XGBClassifier(random_state = 777)
}



def batch_fit_models(xT, yT, xV, yV, models):

    # initiate a dictionary to record model results
    resultCols = [
        'Model', 'Train Time', 
        'Train Accuracy', 'Validation Accuracy',
        'Train Precision', 'Validation Precision',
        'Train Recall', 'Validation Recall',
        'Train f1', 'Validation f1',
        'Train AUC', 'Validation AUC'
    ]

    result = dict([(key, []) for key in resultCols])
    
    # batch train models
    for model_name, model in models.items():
        
        result['Model'].append(model_name)
        
        # train model and record time laps
        trainStart = t.process_time()
        fit = model.fit(xT, yT)
        trainEnd = t.process_time()
        
        # back fit the model on train data
        predLabelTrain = fit.predict(xT)
        predScoreTrain = fit.predict_proba(xT)[:,1]
        
        # fit the model on validation data
        predLabel = fit.predict(xV)
        predScore = fit.predict_proba(xV)[:,1]
        
        # create data for result dict
        result['Train Time'].append(trainEnd - trainStart)
        result['Train Accuracy'].append(accuracy_score(yT, predLabelTrain))
        result['Validation Accuracy'].append(accuracy_score(yV, predLabel))
        result['Train Precision'].append(precision_score(yT, predLabelTrain))
        result['Validation Precision'].append(precision_score(yV, predLabel))
        result['Train Recall'].append(recall_score(yT, predLabelTrain))
        result['Validation Recall'].append(recall_score(yV, predLabel))
        result['Train f1'].append(f1_score(yT, predLabelTrain))
        result['Validation f1'].append(f1_score(yV, predLabel))
        result['Train AUC'].append(roc_auc_score(yT, predScoreTrain))
        result['Validation AUC'].append(roc_auc_score(yV, predScore))
        
    # turn result dict into a df
    dfResult = pd.DataFrame.from_dict(result)
    
    return dfResult


# In[ ]:


batch_fit_models(XTrainFinal, yTrain, XValidFinal, yValid, modelsToFit).sort_values(by = 'Validation AUC', ascending = False)


# Here is the list of best models in terms of ...
# * Accuracy: SVM
# * Precision: SVM
# * Recall: AdaBoost
# * f1 score: GBT
# * AUC: SVM
# 
# Since my model will be scored on accuracy, I'll use accuracy as my metric as well and pick SVM to do parameter tuning.

# In[ ]:


svmFit = modelsToFit['SVM'].fit(XTrainFinal, yTrain)


# In[ ]:


svmFit.get_params()


# In[ ]:


svmPredLabel = svmFit.predict(XValidFinal)
svmPredScore = svmFit.predict(XValidFinal)


# In[ ]:


print(classification_report(yValid, svmPredLabel))


# In[ ]:


confusion_matrix(yValid, svmPredLabel)


# ### 2.2 Parameter/Hyperparameter tuning for SVM

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


paramGrid = {
    'kernel': ('linear', 'rbf', 'poly'),
    'gamma': ('auto', 'scale'),
    'C': [0.1, 0.5, 1],
    'degree': [3, 5]
}


svcTune = GridSearchCV(
    estimator = modelsToFit['SVM'],
    param_grid = paramGrid,
    scoring = 'accuracy'
)

svcCVResults = svcTune.fit(XTrainFinal, yTrain)


# In[ ]:


print('Best model parameters')
print(svcCVResults.best_params_)
print('Best model score')
print(svcCVResults.best_score_)


# The best SVC model has the following parameters: C = 0.5, gamma is 1 / (n_features * X.var()), and kernel is rbf

# In[ ]:


bestSVC = SVC(
    random_state = 777,
    probability = True,
    C = 0.5,
    degree = 3,
    gamma = 'scale',
    kernel = 'rbf'
)


# In[ ]:


print(classification_report(yValid, bestSVC.fit(XTrainFinal, yTrain).predict(XValidFinal)))


# ## 3. Fit the model to test data and submit predictions!

# In[ ]:


testID = dfTestClean[colID]
XTest = dfTestClean[colNum + colCat]
XTestFinal = apply_scaler_encoder(XTest)


# In[ ]:


testPred = bestSVC.predict(XTestFinal)


# In[ ]:


submission = pd.concat([testID, pd.DataFrame(testPred)], axis = 1)
submission = submission.rename(columns = {0: 'Survived'})
submission.to_csv('titanic_submission_20200601.csv', index = False)


# In[ ]:


submission

