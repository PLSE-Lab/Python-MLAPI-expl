#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import warnings  
warnings.filterwarnings('ignore')
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


# # <font color=blue>Exploratory Data Analysis</font>

# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Save the PassengerId for later construction of the submission file
PassengerId = test['PassengerId']


# #### Look at the number of rows of data returned from each data set. Our goal is to clean and transform our data (where necessary) <br>while maintaining these same numbers of rows (i.e. we should not be deleting rows). 

# In[3]:


(train.shape[0], test.shape[0])


# #### Now let's take a look at a sample of the data:

# In[4]:


train.head(3)


# ## Description of Data Columns:
#  - <font color=red>Survived</font> - Survival (0 = No; 1 = Yes)
#  - <font color=red>Pclass</font> - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
#  - <font color=red>Name</font>
#  - <font color=red>Sex</font>
#  - <font color=red>Age</font>
#  - <font color=red>SibSp</font> - Number of Siblings/Spouses Aboard
#  - <font color=red>Parch</font> - Number of Parents/Children Aboard
#  - <font color=red>Ticket</font> - Ticket Number
#  - <font color=red>Fare</font> - Passenger Fare
#  - <font color=red>Cabin</font> - Cabin Number
#  - <font color=red>Embarked</font> - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# ## Handle Missing Data:

# ### We will now analyze each data set and calculate the percentage of the data that is present for each column: 

# In[5]:


train.count() / len(train)


# In[6]:


test.count() / len(test)


# ### Given that we have a description of the data columns and an understanding of the data quantity, we can make executive decisions about which columns to keep and about how to fill in gaps in the data:  
# -  The <font color=red>PassengerId</font> column can be removed from both train and test sets as it is not relevant to the analysis. 
# -  In both sets, only about 22% of the data exists for the <font color=red>Cabin</font> column, so we will drop that column from both. 
# -  ~80% of the <font color=red>Age</font> column exists for both sets, so we will impute the missing ages by replacing the NaN values with the gender mean. 
# -  Values will be imputed for the missing <font color=red>Embarked</font> data in Train and <font color=red>Fare</font> data in test.

# #### First, define a method--'drop_incomplete_cols'--which removes a column if less than a certain percentage of the data is present:

# In[7]:


def drop_incomplete_cols(df):
    s = df.count() / len(df)
    threshold = .4
    for col_name in s.index:
        if s[col_name] < threshold:
            df.drop(col_name, axis=1, inplace=True)


# #### Next, use the above function in a loop that also drops any additional unneeded columns, and imputes missing <font color=red>Age</font>, <font color=red>Fare</font>, and <font color=red>Embarked data</font>:

# In[8]:


all_data = [train, test]

for df in all_data:
    df.drop(['PassengerId'], axis=1, inplace=True)
    #impute missing Age with the mean across Sex
    df['Age'].fillna(df.groupby(['Sex'])['Age'].transform(np.mean), inplace=True)
    #impute missing Fare with the mean across Pclass and Embarked
    df['Fare'].fillna(df.groupby(['Pclass', 'Embarked'])['Fare'].transform(np.mean), inplace=True)
    #impute missing Embarked with mode of data set
    df['Embarked'].fillna(df['Embarked'].mode().iloc[0], inplace=True)
    drop_incomplete_cols(df)


# #### Finally, check the completeness of the data in both sets:

# In[9]:


train.count()/len(train)


# In[10]:


test.count()/len(test)


# #### All missing data has been handled without any loss of training or test instances:

# In[11]:


(train.shape[0], test.shape[0])


# ## Check correlation of features

# #### It is helpful to look at a correlation matrix to see whether two features are highly correlated and thus redundant. Here, we are returning the top relevant rows of a sorted, unstacked correlation matrix of the train data:

# In[12]:


train.corr().abs().unstack().sort_values(ascending=False)[len(train.corr().columns):len(train.corr().columns) + 10]


# #### The highest level of correlation is between <font color=red>Fare</font> and <font color=red>Pclass</font>, but not high enough to warrant removal of either at this point.

# # <font color=blue>Data Visualization</font>

# ### Here, we will take a look at how survival rates were distributed across <font color=red>Age</font> and <font color=red>Sex</font>:

# In[13]:


fig, axes = plt.subplots(1, 2, figsize=(30, 8))
women = train[train['Sex'] == 'female']
men = train[train['Sex'] == 'male']
ax = sns.distplot(women[women['Survived']==1].Age, bins=65, label='survived', ax=axes[0], kde=False)
ax = sns.distplot(women[women['Survived']==0].Age, bins=65, label='not survived', ax=axes[0], kde=False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age, bins=85, label=                   'survived', ax=axes[1], kde=False)
ax = sns.distplot(men[men['Survived']==0].Age, bins=85, label=                   'not survived', ax=axes[1], kde=False)
ax.legend()
ax.set_title('Male')


# #### Above left, we see that the most common passenger age for women is about 28, and about 2/3 of that group survived. Above right,the most common age range for men is about 31, and only about 1/6 of that group survived. In fact, there is a low survival rate for males across the board. The opposite is true for females.

# # <font color=blue>Feature Engineering</font>

# ### Here, we will seek to generate new features from existing features in order to create more meaningful and/or compact representations of the data.

# #### For example, The <font color=red>SibSp</font> and <font color=red>Parch</font> columns do not seem particularly helpful from an interpretation standpoint, since we cannot distinguish between siblings and spouses in the former case and parents and children in the latter (although this is debateable). We will combine these fields into a column called <font color=red>num_relatives</font>, and add a column, <font color=red>solo</font> which flags whether the passenger had relatives on board.

# In[14]:


all_data = [train, test]
for df in all_data:
    df['num_relatives'] = df['SibSp'] + df['Parch']
    df.loc[df['num_relatives'] > 0, 'solo'] = 0
    df.loc[df['num_relatives'] == 0, 'solo'] = 1
    df['solo'] = df['solo'].astype(int)
    #drop 'SibSp' and 'Parch' columns:
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# #### We can now graph the survival rate for passengers across various numbers of relatives:

# In[15]:


axes = sns.catplot('num_relatives', 'Survived',
                  data=train, kind='point', aspect=2)


# #### The above graph shows that across males and females, the mean survival rate only breaks 50% when a passenger has 1-3 relatives. Next, we will split out these results by gender:

# In[16]:


fig, axes = plt.subplots(1, 2, figsize=(30, 8))
women = train[train['Sex'] == 'female']
men = train[train['Sex'] == 'male']
ax = sns.distplot(women[women['Survived']==1].num_relatives, label='survived', ax=axes[0], kde=False)
ax = sns.distplot(women[women['Survived']==0].num_relatives, label='not survived', ax=axes[0], kde=False)
ax.legend()
ax.set_title('Female')
ax.set_xticks(range(0,10))
ax = sns.distplot(men[men['Survived']==1].num_relatives, label=                   'survived', ax=axes[1], kde=False)
ax = sns.distplot(men[men['Survived']==0].num_relatives, label=                   'not survived', ax=axes[1], kde=False)
ax.legend()
ax.set_xticks(range(0,10))
ax.set_title('Male')


# #### The graphs above show that while survival rates for both men and women were low for passengers with 4 relatives or more, the survival rate for women with 3 or fewer relatives dramatically increases while the survival rate for similar men remains low.

# #### There seems to be a social class factor associated with survival rates, so we will use a regular expression to extract the honorifics from passenger's names into a <font color=red>title</font> column. We will then use <font color=red>title</font> with <font color=red>Pclass</font> to engineer a <font color=red>social_status</font> column:

# In[17]:


def add_title_col(df):
    pattern = r'(Mr\.|Mrs|Ms|Miss|Master|Dr\.|Don\.|Dona\.|Rev\.|Sir\.|Lady|Mme|Mlle|Major|Col\.|Capt\.|Countess|Jonkheer)'
    title = df.Name.str.extract(pattern).fillna('NONE')
    title.columns = ['title']
    return pd.concat((df, title), axis=1)


# In[18]:


def add_social_status_col(df):
    classes = ['peerage', 'upper', 'officer', 'clergy', 'middle', 'lower']
    peerage = ['Don.', 'Dona.', 'Sir.', 'Lady', 'Mme', 'Mlle', 'Countess', 'Jonkheer']
    officer = ['Col.', 'Major', 'Capt.']
    clergy = ['Rev.']
    basic_honorific = ['Mr.', 'Mrs', 'Ms', 'Miss', 'Master', 'Dr.']
    
    df.loc[df['title'].isin(peerage), 'social_status'] = 'peerage'
    df.loc[(df['title'].isin(basic_honorific) & (df['Pclass'] == 1)), 'social_status'] = 'upper'
    df.loc[df['title'].isin(officer), 'social_status'] = 'officer'
    df.loc[df['title'].isin(clergy), 'social_status'] = 'clergy'
    df.loc[(df['title'].isin(basic_honorific) & (df['Pclass'] == 2)), 'social_status'] = 'middle'
    df.loc[(df['title'].isin(basic_honorific) & (df['Pclass'] == 3)), 'social_status'] = 'lower'
    
    #test:
    if len(df[~df['social_status'].isin(classes)]) == 0:
        print('All passengers have been assigned a social status')
    else:
        print('social status assignment was NOT successful.')
        
    return df


# In[19]:


train = add_title_col(train)
train = add_social_status_col(train)

test = add_title_col(test)
test = add_social_status_col(test)


# In[20]:


train.head()


# #### Here, we will create a feature that is based on the first digit of the numerical portion of each ticket. In the interests of intellectual honesty, this bit of feature engineering is also known as <font color=red>"taking a flyer"</font>:

# In[21]:


def create_ticket_digit_col(df):
    if 'first_digit' not in df.columns:
        pattern = r'(\d{1})\d+$'
        first_digit = df.Ticket.str.extract(pattern).fillna('0')
        first_digit.columns = ['first_digit']
        df = pd.concat([df, first_digit], axis=1)  
        return df
    else:
        return df


# In[22]:


train = create_ticket_digit_col(train)
test = create_ticket_digit_col(test)


# In[23]:


train.head()


# ##### Print out the survival rate for each ticket prefix.

# In[24]:


def view_ticket_survival_stats(df):
    pre = []
    m = []
    c = []
    fd_list = df['first_digit'].value_counts().index.sort_values()
    for i in fd_list:
        pre.append(i)
        m.append(df[df['first_digit'] == i].loc[:, 'Survived'].mean())
        c.append(df[df['first_digit'] == i].loc[:, 'Survived'].count())

    prefix_survival_pct = pd.DataFrame({'prefix': pre, 'count': c, 'survival_rate': m})
    return prefix_survival_pct.sort_values(by='survival_rate', ascending=False)


# In[25]:


view_ticket_survival_stats(train)


# ##### The most common ticket prefixes  (1, 2, & 3) show significant drop-off in survival rates: 60%, 41%, and 26%, respectively. Other ticket prefixes match the survival rate of 3, or worse. The exception is prefix 9, which has a 100% survival rate. This is less significant however, since we only count 3 examples. We will keep the ticket prefix as a predictor.

# # <font color=blue>Data Cleaning and Preparation for Machine Learning Algorithms</font>

# In[26]:


train.head()


# ##### We will now remove columns we don't need, and label encode categorical columns. We will also scale the remaining numerical columns.

# In[27]:


def safe_column_remove(df, columns):
    for col in columns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df


# In[28]:


#save the labels before removing the column:
train_labels = train['Survived']

cols_to_remove = ['Survived', 'Name', 'Ticket', 'title']
train = safe_column_remove(train, cols_to_remove)
test = safe_column_remove(test, cols_to_remove)


# In[29]:


def titanic_scaler_encoder(df, isTreeInput=True):
    if isTreeInput==True:
        ss_cols  = ['Age', 'Fare', 'num_relatives']
        encoded_cols = ['Pclass', 'Sex', 'Embarked', 'social_status', 'first_digit']
        unchd_cols = ['solo']
        
        scaler = StandardScaler()  
        scaled_data  = scaler.fit_transform(df[ss_cols])  
        label_encoded_data = df[encoded_cols].apply(LabelEncoder().fit_transform)
        
        return np.concatenate([scaled_data, label_encoded_data, df[unchd_cols]], axis=1)     
    else:
        ss_cols  = ['Age', 'Fare', 'num_relatives']
        unchd_cols = ['solo', 'Pclass', 'Sex', 'Embarked', 'social_status', 'first_digit']
    
        scaler = StandardScaler()  
        scaled_data  = scaler.fit_transform(df[ss_cols])
        
        return np.concatenate([scaled_data, df[unchd_cols]], axis=1)


# In[30]:


train_prepared = titanic_scaler_encoder(train)
test_prepared = titanic_scaler_encoder(test)


# In[31]:


train_prepared


# In[32]:


test_prepared


# # <font color=blue>Grid Search for Logistic Regression</font>

# In[33]:


param_grid = [{'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 20), 'solver' : ['liblinear']}]
lg_clf = GridSearchCV(linear_model.LogisticRegression(), param_grid, cv=5, scoring='roc_auc')
lg_clf.fit(train_prepared, train_labels)
lg_predictions = lg_clf.best_estimator_.predict(test_prepared)


# In[34]:


lg_clf.best_params_, lg_clf.best_score_


# # <font color=blue>Create Submission File</blue>

# In[35]:


submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': lg_predictions})
submission.Survived.astype(int)
submission.head(20)


# In[36]:


submission.to_csv('titanic_submission_5_1.csv', float_format='%.f', index=False)

