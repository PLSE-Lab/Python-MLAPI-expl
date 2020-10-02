#!/usr/bin/env python
# coding: utf-8

# # ReadMe
# This Notebook is broken into the following sections
# 1. Loading Data
# 2. Wrangling Data
# 3. Feature Engineering
# 4. Encoding with Pipelines
# 5. Modeling
# 6. Confusion Matrix 
# 7. Hypothetical Consultant Project from the Government
# 8. Kaggle Submission 

# In[ ]:


import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Load Data

# In[ ]:


#path = '/Users/ridleyleisy/Documents/lambda/unit_two/DS-Unit-2-Classification-1/ds4-predictive-modeling-challenge/'


# In[ ]:


train = pd.read_csv('../input/train_features.csv')
test = pd.read_csv('../input/test_features.csv')
labels = pd.read_csv('../input/train_labels.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# ## Wrangle Data

# ### Dropping Cols
# 1. num_private: only one unique observation
# 2. recorded_by: only one unique observation

# In[ ]:


def drop_cols_rows(df):
    '''
    Function takes in a pandas Dataframe and drops columns: num_private and recorded_by
    Returns: df
    '''
    df.drop('num_private',axis=1,inplace=True)
    df.drop('recorded_by',axis=1,inplace=True)
    return df


# In[ ]:


#applying changes to train and test data
train = drop_cols_rows(train)
test = drop_cols_rows(test)


# ### Replace longitude
# Replacing missing longitude data with mean of longitude

# In[ ]:


#test whether you want to keep longitude
long_mean = train.loc[train['longitude'] !=0]['longitude'].mean()
train['longitude'].replace(0,long_mean,inplace=True)
test['longitude'].replace(0,long_mean,inplace=True)


# ### Fill in Missing Construction Year Data
# Since we have missing construction year data, we will create a random forest to predict the missing values

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import seaborn as sns


# #### Random Forest

# In[ ]:


# replacing 0 construction year with NaN values
train['construction_year'] = train['construction_year'].replace(0,np.nan)
test['construction_year'] = test['construction_year'].replace(0,np.nan)


# In[ ]:


def transform_construction(df):
    '''
    Function that takes in pandas Dataframe and returns predicted values for construction year
    Returns: np.array of predicted values
    '''
    df = df.select_dtypes(include=np.number)
    X = df.loc[~df['construction_year'].isna()]
    
    # can only use these featuers since they differ 
    features = ['amount_tsh', 'gps_height', 'longitude', 'latitude',
       'region_code', 'district_code', 'population']
    target = 'construction_year'
    
    X_train = X[features]
    y_train = X[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train)
    m = RandomForestRegressor(n_estimators=20,max_depth=25)
    m.fit(X_train, y_train)
    
    vals = m.predict(df.loc[df['construction_year'].isna()][features])
    
    return  vals


# In[ ]:


# applying random forest to np.nan values
train.loc[train['construction_year'].isna(),'construction_year'] = transform_construction(train)
test.loc[test['construction_year'].isna(),'construction_year'] = transform_construction(test)
# rounding construction year so it aligns with existing data
train['construction_year'] = round(train['construction_year'])
test['construction_year'] = round(test['construction_year'])


# ## Feature Engineering
# Features we're adding
# 1. time_since_construction = date_recorded - construction year
# 2. Ratios:
#     <br>a. tsh_by_longitude
#     <br>b. tsh_by_latitude
#     <br>c. tsh_by_height

# ### Time Since Construction

# In[ ]:


def add_construction_diff(df):
    '''
    Returns pandas Dataframe with time_since_construction year column added
    '''
    # convert series to datetime objects
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['construction_year'] = pd.to_datetime(df['construction_year'].astype(int),format="%Y")
    
    # difference of date recorded and construction year
    df['time_since_construction'] = (df['date_recorded'] - df['construction_year']).dt.days
    
    # removing time_since_construction data that's less than 0 
    df.loc[df['time_since_construction'] < 0,'time_since_construction'] = 0    
    df['construction_year'] = df['construction_year'].dt.year
    return df


# In[ ]:


test = add_construction_diff(test)
train = add_construction_diff(train)


# In[ ]:


sns.distplot(train['construction_year'])


# ### Ratios

# In[ ]:


def add_ratios(df):
    '''
    Returns pandas Dataframe that includes tsh ratios
    '''
    df['tsh_by_longitude'] = df['amount_tsh'] / df['longitude']
    df['tsh_by_latitude'] = df['amount_tsh'] / abs(df['latitude'])
    df['tsh_by_height'] =  df['amount_tsh'] / df['gps_height']
    df['tsh_by_height'] = df['tsh_by_height'].replace(np.inf,0).replace(np.nan,0)
    return df


# In[ ]:


test = add_ratios(test)
train = add_ratios(train)


# #### 3d Visualization

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(train['longitude'], train['latitude'], train['gps_height'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()


# ### Reduce label size to fit new features

# In[ ]:


labels = labels.merge(train,on='id')[['id','status_group']]


# ## Encoding and Scaling Categorical Data

# In[ ]:


import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[ ]:


# grabbing unique counts for categorical variables
unique = train.describe(exclude=np.number).T.sort_values(by='unique')


# In[ ]:


# appending categorical features with less than 130 values to list
cat_features = list(unique.loc[unique['unique'] < 130].index)


# In[ ]:


# numerical features to list
numeric_features = list(train.describe().columns[1:])


# In[ ]:


encode_features = cat_features
features = numeric_features + encode_features


# ### Pipeline

# In[ ]:


def transform_data_hot(features:list):
    '''
    Function that processes training data so it's ready for an ml model
    Args: list of features to process
    
    Returns:
    1 - X_train encoded 
    2 - X_val encoded
    3 - y_train
    4 - y_val
    5 - X_train_sub
    6 - X_val_sub
    7 - Encoded column names
    8 - Test dataset encoded
    
    '''
    # using one hot encoding for categorical features
    encoder = ce.OneHotEncoder(use_cat_names=True)
    scaler = RobustScaler()
    
    X_train = train[features]
    y_train = labels['status_group']
    
    # train test splitting the data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=0.80, test_size=0.20, 
        stratify=y_train, random_state=42)

    X_train_sub = X_train[features]
    X_val_sub = X_val[features]
    
    #creating pipeline that includes encoder and scaler
    pipeline1 = Pipeline([('encoder',encoder),('scaler',scaler)])
    # apply pipeline to train and val datasets
    X_train_sub_encoded = pipeline1.fit_transform(X_train_sub)
    X_val_sub_encoded = pipeline1.transform(X_val_sub)
    
    # test encoded is encoded and scaled dataframe for us to run our model on test dataset
    test_encoded = pipeline1.transform(test[features])
    
    # X_encode_cols is dataframe 
    X_encode_cols = encoder.transform(X_train_sub).columns
    
    return X_train_sub_encoded, X_val_sub_encoded, y_train, y_val, X_train_sub, X_val_sub, X_encode_cols, test_encoded


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# ### Grabbing data for modeling

# In[ ]:


X_train_sub_encoded, X_val_sub_encoded, y_train, y_val, X_train_sub, X_val_sub, X_encode_cols, test_encoded = transform_data_hot(features)


# ### Model

# In[ ]:


# creating random forest model 
m = RandomForestClassifier(n_estimators=300,max_depth=28,max_features='auto',n_jobs=-1)


# In[ ]:


# fitting our training data to the model
m.fit(X_train_sub_encoded,y_train)


# In[ ]:


print(f'Our Random Forest Score for Training is {m.score(X_train_sub_encoded,y_train)}')
print(f'Our Random Forest Score for Validation is {m.score(X_val_sub_encoded,y_val)}')


# In[ ]:


val_preds = m.predict(X_val_sub_encoded)


# In[ ]:


train_preds = m.predict(X_train_sub_encoded)


# In[ ]:


# grabbing predictions for our test dataset 
preds = m.predict(test_encoded)


# ### Feature Importance

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# setting feature importance to a pandas Dataframe
feat_impt = pd.DataFrame(m.feature_importances_,X_encode_cols).sort_values(by=0)


# In[ ]:


# Cumulative summing feature importance
cumsum = np.cumsum(feat_impt[::-1])
# Selecting all features that sum up to 95%
sub_sum = cumsum.loc[cumsum[0] < .95]


# In[ ]:


# Distplot on feature importance 
sns.distplot(feat_impt.iloc[len(sub_sum.index):-1])
plt.text(x=.084,y=100,s='longitude',rotation=90,fontsize=12)
plt.text(x=.041,y=120,s='gps_height',rotation=90,fontsize=12)
plt.text(x=.068,y=230,s='time_since_construction',rotation=90,fontsize=12);


# In[ ]:


# plotting cumulative importance on a line graph
fig, ax1 = plt.subplots(nrows=1,ncols=1)
fig.set_figheight(8)
fig.set_figwidth(20)

sub_sum.plot(ax=ax1)
ax1.set_xticklabels(np.arange(0,150,20))
ax1.set_title('Cumulative Importance by Number of Features',fontsize=16)
ax1.set_xlabel('Number of Features',fontsize=14)
ax1.set_ylabel('Cumlative Importance',fontsize=14)
ax1.get_legend().remove()


# ## Confusion Matrix...Confusion? Don't be

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report


# In[ ]:


confusion_matrix(val_preds,y_val)


# In[ ]:


def plot_confusion_matrix(y_true, y_pred):
    labels = unique_labels(y_true)
    columns = [f'Predicted {label}' for label in labels]
    index = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true, y_pred), 
                         columns=columns, index=index)
    return sns.heatmap(table, annot=True, fmt='d', cmap='viridis')
    
plot_confusion_matrix(val_preds,y_val);


# In[ ]:


print(classification_report(val_preds,y_val))


# ## You are now a consultant for the Tanzanian Government
# 
# You are instructed by the government to assess total costs to fix their damaged water wells. The government informs you that
# 1. On average, it costs \\$300 to repair a well and \\$500 to fix a non functional well
# 2. On average, it takes 7 days to fix and 5 days to repair a well
# 3. The government can allocate 1 worker for the job. Yup only one!
# 4. It takes 2 days for the worker to relocate to a new well (this includes showing up to a well that was predicted as broken but is actually in working order
# 
# 
# The government prioritizes the following wells 
# 1. Non functional wells with populations greater than 1,000
# 2. Needs repair wells with populations greater than 1,000
# 
# Questions to Answer
# 1. How long will this project take?
# 2. What are the expected costs of this project?
# 
# **Dataset to use -- Validation** <br>
# **The historical confusion matrix for their previous model is given. Note, it is different than the matrix above**

# In[ ]:


#hard coded but numbers can be accessed from the confusion matrix above
non_fun_precision = 0.78 
repair_precision = 0.33


# In[ ]:


# let's merge our predicted data with the original dataset to get population stats


# In[ ]:


X_val_sub['preds'] = val_preds
df = X_val_sub[['population','preds']]


# In[ ]:


# let's split the dataframe into non functional wells and wells that need repair for populations above 1000
non_func = df.loc[(df['population'] > 1000) & (df['preds'] == 'non functional')].reset_index()
repairs = df.loc[(df['population'] > 1000) & (df['preds'] == 'functional needs repair')].reset_index()


# Our model predicts that out of the non functional wells, we expect the well to actually be non functional 78% of the time. Since we aren't perfect, we expect that 11% are in need of repair and the other 11% are functional

# In[ ]:


# let's add a actual column where the worker shows up and sees the condition of the well
#allocating our error in prediction from non_fun_precision
actual_func = round(len(non_func) * (1-non_fun_precision) / 2) 
actual_repair = round(len(non_func) * (1-non_fun_precision) / 2)
actual_non_func = len(non_func) - actual_func - actual_repair


# In[ ]:


actual_func + actual_non_func + actual_repair


# In[ ]:


non_func['actual'] = 0
non_func['days'] = 0
non_func['cost'] = 0


# In[ ]:


non_func.loc[0:actual_func,'actual'] = 'functioning'
# allocating 2 days to relocate to another well
non_func.loc[0:actual_func,'days'] = 2
non_func.loc[0:actual_func,'cost'] = 0


# In[ ]:


non_func.loc[actual_func:actual_func+actual_repair-1,'actual'] = 'repair'
#allocating 2 days of relocating and 5 days of repairing
non_func.loc[actual_func:actual_func+actual_repair-1,'days'] = 7
non_func.loc[actual_func:actual_func+actual_repair-1,'cost'] = 300


# In[ ]:


non_func['actual'] = non_func['actual'].replace(0,'non_functional')
# allocating 2 days of relocating and 7 days to fix
non_func.loc[non_func['actual'] == 'non_functional','days'] = 9
non_func.loc[non_func['actual'] == 'non_functional','cost'] = 500


# In[ ]:


non_func.head()


# Our model predicts that out of the need repair wells, we expect the well to actually be in a state of repair 33% of the time. Since we aren't perfect, we expect that the other 67% are in non functional state

# In[ ]:


# let's add a actual column where the worker shows up and sees the condition of the well
actual_repair = round(len(repairs) * .33)
actual_non_func = len(repairs) -  actual_repair


# In[ ]:


actual_repair + actual_non_func


# In[ ]:


repairs['actual'] = 0
repairs['days'] = 0
repairs['cost'] = 0


# In[ ]:


repairs.loc[0:actual_repair,'actual'] = 'repair'
#allocating 2 days of relocating and 5 days of repairing
repairs.loc[0:actual_repair,'days'] = 7
repairs.loc[0:actual_repair,'cost'] = 300


# In[ ]:


repairs['actual'] = repairs['actual'].replace(0,'non_functional')
# allocating 2 days of relocating and 7 days to fix
repairs.loc[repairs['actual'] == 'non_functional','days'] = 9
repairs.loc[repairs['actual'] == 'non_functional','cost'] = 500


# In[ ]:


repairs


# ### Reporting

# In[ ]:


f"It will take {repairs.sum()['days'] + non_func.sum()['days']} days to complete"


# In[ ]:


f"It will cost {repairs.sum()['cost'] + non_func.sum()['cost']} dollars"


# ### How many days did we waste on scouting functional wells?

# In[ ]:


f"We lost {non_func.loc[non_func['actual'] == 'functioning']['days'].sum()} days scouting out functioning wells"


# ## Predicting for Kaggle

# In[ ]:


# submission = pd.DataFrame(test['id'])
# submission['status_group'] = preds
# submission.to_csv('submission-01.csv',index=False)


# In[ ]:


# import pandas as pd

# # Filenames of your submissions you want to ensemble
# files = ['submission-01.csv', 'submission-02.csv', 'submission-03.csv']

# submissions = (pd.read_csv(file)[['status_group']] for file in files)
# ensemble = pd.concat(submissions, axis='columns')
# majority_vote = ensemble.mode(axis='columns')[0]

# sample_submission = pd.read_csv('sample_submission.csv')
# submission = sample_submission.copy()
# submission['status_group'] = majority_vote
# submission.to_csv('my-ultimate-ensemble-submission.csv', index=False)


# In[ ]:





# In[ ]:




