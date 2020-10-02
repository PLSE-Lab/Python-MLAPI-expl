#!/usr/bin/env python
# coding: utf-8

# **Costa Rican Household Poverty Level Prediction**
# **Can you identify which households have the highest need for social welfare assistance?**
# 
# We start with looking at what are the files available.We have:
# *  train.csv
# * test.csv 
# * sample_submission.csv

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


# Lets read the training and test data both as pandas dataframe

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


train.shape,test.shape


# In[ ]:


train.head()


# **Exploratory Data Analysis**
# 
# Find what data types exist in training data

# In[ ]:


train.dtypes.value_counts()


# First lets look at the five object data types that exist

# In[ ]:


train.select_dtypes(include=['object']).head()


# We have id-which is the ID per row and idhogar which is id per household more over we have
# dependency ,edjefe and edjefa.Now lets look at the three variables one by one

# **dependency**-Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64

# We should replace all non numeric values in dependency/edjefe/edjefa as:
# 
# if no=0
# if yes=median
# 
# 
# 

# Lets combine the train and test set so that we could clean up the variables

# In[ ]:


test['Target']=0

train['is_train']=1

test['is_train']=0
#del train2

train2=train.copy()
train2=train2.append(test,ignore_index=True)


# In[ ]:


train2['is_train'].value_counts()


# **How many missing do we have across all the fields?**

# In[ ]:


train2.columns[train2.isnull().any()]


# In[ ]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


# Missing values statistics
missing_values = missing_values_table(train2)
missing_values.head(20)


# For  rez_esc,v18q1 and v2a1 we have greater than 70% missing.We can drop there three.For meanedu and SQBmeaned we replace them my the means of their respective columns

# In[ ]:


del train2['rez_esc']
del train2['v18q1']
del train2['v2a1']


# In[ ]:



train2['meaneduc'].fillna((train2['meaneduc'].mean()), inplace=True)
train2['SQBmeaned'].fillna((train2['SQBmeaned'].mean()), inplace=True)


# Lets replace yes/no from the categorical(object) variables we identified previously (dependency ,edjefe and edjefa)

# In[ ]:


dependency=pd.DataFrame()
dependency['dependency']=train2['dependency'].loc[train2['dependency']!=('yes')]
dependency['dependency']=dependency['dependency'].loc[dependency['dependency']!=('no')]


# In[ ]:


dependency['dependency'].astype('float64').mean()


# In[ ]:


train2.loc[train2['dependency']=='yes','dependency'] = 1.59
train2.loc[train2['dependency']=='no','dependency'] = 0


# In[ ]:


train2['dependency']=train2['dependency'].astype('float64')


# In[ ]:


dependency['edjefe']=train2['edjefe'].loc[train2['edjefe']!=('yes')]
dependency['edjefe']=dependency['edjefe'].loc[dependency['edjefe']!=('no')]
dependency['edjefe'].astype('float64').mean()


# In[ ]:


train2.loc[train2['edjefe']=='yes','edjefe'] = 8.54
train2.loc[train2['edjefe']=='no','edjefe'] = 0


# In[ ]:


train2['edjefe']=train2['edjefe'].astype('float64')


# In[ ]:


dependency['edjefa']=train2['edjefa'].loc[train2['edjefa']!=('yes')]
dependency['edjefa']=dependency['edjefa'].loc[dependency['edjefa']!=('no')]
dependency['edjefa'].astype('float64').mean()


# In[ ]:


train2.loc[train2['edjefa']=='yes','edjefa'] = 8.47
train2.loc[train2['edjefa']=='no','edjefa'] = 0


# In[ ]:


train2['edjefa']=train2['edjefa'].astype('float64')


# In[ ]:


train2.select_dtypes(include=['object']).head()


# In[ ]:


train2.dtypes.value_counts()


# Now Lets seperate train and test so that we go forward with EDA

# In[ ]:


train=train2.loc[train2['is_train']==1]
test=train2.loc[train2['is_train']==0]


# In[ ]:


train.shape,test.shape


# **Lets start with EDA of the variables**

# **We start with Target**
# 
# 1 = extreme poverty 
# 2 = moderate poverty 
# 3 = vulnerable households 
# 4 = non vulnerable households

# In[ ]:


train['Target'].astype('int64').plot.hist()


# Lets try this plot in a better way so that we are able to find the percentage split

# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
train['Target'].value_counts(normalize=True).plot(ax=ax, kind='bar')


# **`What we observe is ~60% of households have been classified as non vulnerable while ~5% are close to extreme poverty**

# We should try and look and some of the most important factors which can help us identify the vulnerabilty of households.Going through ~140 variables may not be the most optimized use of our time .Lets use the amazingly used and abused  random forest for our advantage

# Before we do that Lets once quicky Glance at the dataset that we have to spot any anomalies

# In[ ]:


train.describe()


# We could not find anything offbeat per se,but even if we are missing something we can always return back here.Let's see what the Random forest trees have to say about this

# In[ ]:


from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import  f1_score

def f1_macro(y_true, y_pred): return f1_score(y_true, y_pred, average='macro')


# In[ ]:


def print_score(m):
    res = [f1_macro(m.predict(X_train), y_train), f1_macro(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


from sklearn.model_selection import train_test_split
df=train.copy()
y=df['Target']
del df['Target']
del df['Id']
del df['idhogar']
X_train, X_valid, y_train, y_valid = train_test_split(
 df, y, test_size=0.33, random_state=42)


# In[ ]:


m =RandomForestClassifier(random_state=2,n_jobs=-1,criterion="gini" )


# In[ ]:



get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# Lets find out which variables does random forest feel are important 

# In[ ]:


feature_importances = pd.DataFrame(m.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


#Which are the top 10 factors influencing the vulnerability of a family
feature_importances.head(10)


# Lets look at the variables which random forest has deemed to be important

# **1. mean education**-average years of education for adults (18+)
# This kind of makes sense education of adults in household governs their employability  which in turn governs their income
# 
# Let us have a look at how mean education is distributed with respect to the poverty levels

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)


# In[ ]:


grid = sns.FacetGrid(train, col='Target', margin_titles=True)
grid.map(plt.hist,'meaneduc',normed=True, bins=np.linspace(0, 40, 15));


# We observe that that for non vulnerable(Target=4) there are higher proportion of people in higher mean education bracket(i.e greater than 10) while that is lower for other outcomes

# **2.SQBmeaned**-square of the mean years of education of adults (>=18) in the household
# 
# This variable is also related to education but only for adults(>=18 years)

# In[ ]:


grid = sns.FacetGrid(train, col='Target', margin_titles=True)
grid.map(plt.hist,'SQBmeaned',normed=True, bins=np.linspace(0, 40, 15));


# **3.dependency**-Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# 
# This should indicate that does having more individuals in your family who are not bread earners but dependent impact the poverty levels.

# In[ ]:


grid = sns.FacetGrid(train, col='Target', margin_titles=True)
grid.map(plt.hist,'dependency',normed=True, bins=np.linspace(0, 40, 15));


# **Feature Engineering:**Lets try and create some logical ratios

# In[ ]:


train2=train.copy()

#First lets try to get a mobiles used per person which should be highly predictive of financial well being
train2['Tot_persons']=train2['overcrowding']*train2['bedrooms']
train2['mob_perperson']=train2['qmobilephone']/train2['Tot_persons'] #This has a higher correlation than individual variables it uses
#Can we merge both the Education and overcrowding together?

train['Edu_crwd_ratio']=train['meaneduc']/train['overcrowding']


# It does not look that profound but non vulnerable families seem to have lower number of dependents 

# **One good way to quantify these relationships will be a correlation matrix**

# In[ ]:



data = train2[['Target','Edu_crwd_ratio','Tot_persons','mob_perperson', 'dependency', 'SQBmeaned', 'meaneduc','qmobilephone','overcrowding','SQBhogar_nin','edjefe','escolari','SQBovercrowding']]
data_corrs = data.corr()
data_corrs


# In[ ]:


plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# **What are the major factors that emerge by looking at top 10 important variables?**

# **Factor-1**-Education,There are many variables here which indicate the importance of education  in impacting the poverty vulnerability of households.There is ~34% correlation between mean education and the poverty level of a household

# **Factor-2**-Family size(or number of members in a household who are dependent and are not capable of earning bread for themselves).This looks intuitive as if a family has more members to be taken care compared to members who are earning and providing lower is the possibility of them becoming prosperous.

# In[ ]:


plt.figure(figsize = (10, 12))

# iterate through the sources
for i, source in enumerate(['dependency', 'SQBmeaned', 'meaneduc']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(train.loc[train['Target'] == 1, source], label = 'target == 1')
    # plot loans that were not repaid
    sns.kdeplot(train.loc[train['Target'] == 2, source], label = 'target == 2')
    
    sns.kdeplot(train.loc[train['Target'] == 3, source], label = 'target == 3')
    
    sns.kdeplot(train.loc[train['Target'] == 4, source], label = 'target == 4')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)


# **Lets try and submit the predictions of the Random forest we created**

# In[ ]:


test.head()


# In[ ]:


del test['Target']
test_df=test.copy()
del test_df['Id']
del test_df['idhogar']
Target=m.predict(test_df)


# In[ ]:


test.tail()


# In[ ]:


pd.options.mode.chained_assignment = None
test['Target'] = Target


# In[ ]:


test[['Id', 'Target']].to_csv('submission.csv', index= False)


# In[ ]:




