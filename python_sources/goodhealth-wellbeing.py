#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://www.kaggle.com/cdc/national-health-and-nutrition-examination-survey/home

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier


print("Input Directory:")
os.listdir("../input")


# In[ ]:


"""
Diseases of the rich -> Heart disease
Linked to high cholesteral
Use demog and habbtis to see who's at risk of high cholesteral

https://www.webmd.com/heart-disease/guide/heart-disease-lower-cholesterol-risk#1
https://www.medicalnewstoday.com/articles/315900.php
https://www.who.int/nutrition/topics/2_background/en/
"""


# In[ ]:


"""
Just going to use Demographic, questionnaire (habbits) & cholestoral level to help us answer the question:

COULD WE IDENTIFY PEOPLE AT RISK OF HIGH CHOLESTEROL LEVELS FROM A SHORT BEHAVIOURAL SURVEY
"""

demog = pd.read_csv("../input/national-health-and-nutrition-examination-survey/demographic.csv")
ques = pd.read_csv("../input/national-health-and-nutrition-examination-survey/questionnaire.csv")
labs = pd.read_csv("../input/national-health-and-nutrition-examination-survey/labs.csv")


# In[ ]:


# check to see if there are shared column names in our two tables, before combining
demog_cols = demog.columns
ques_cols = ques.columns
print("The demog and question datasets share the {} column".format(set(demog_cols).intersection(ques_cols)))
print("So will join them on this column")

# check shaped of these columne
print("\nShape of Demographics dataset: {}".format(demog.shape))
print("Shape of Questionnaire dataset: {}".format(ques.shape))

# Join the two datasets on the unique number for respondants
dataset = demog.join(ques.set_index('SEQN'), on='SEQN')
print("\nShape of COMBINED dataset: {}**".format(dataset.shape))
print("\n** one less column since SEQN is shared")


# In[ ]:


# Get cholesteral levels info, first see how many answers we have
labs['LBXTC'].describe()


# In[ ]:


cholesterol = labs[['SEQN','LBXTC']]
dataset = dataset.join(cholesterol.set_index('SEQN'), on='SEQN')

print("Finally, add the cholesterol level (target) column.")
print("New Shape: {}".format(dataset.shape))

# only keep the rows where we have cholesteral info
dataset = dataset[dataset['LBXTC'].isna() == False]
print("/nFiltering for records where we have Cholesterol info gives final shape: {}".format(dataset.shape))


# In[ ]:


## define function to categorise the cholesteral level
def cholesterol_level(mg):
    """
    1 - OK
    2 - Borderline High (at risk)
    3 - High
    """
    if mg < 200.0:
        return 1
    elif mg < 240:
        return 2
    else:
        return 3


# In[ ]:


# categorise cholesterol
dataset['Target'] = dataset['LBXTC'].apply(lambda x: cholesterol_level(x))

print("Group cholesterol levels in to 3 categories:")
print("1 - OK")
print("\n2 - Borderline High (at risk)")
print("\n3 - High")

print("\nWith the following totals:")
print(dataset['Target'].value_counts())


# * 999 features is quite a lot to start with
# * lets check for missing values to see if we should drop some of these
# 
# ## Demographic feature selection

# In[ ]:


### first check for the Demographics
msno.matrix(dataset.iloc[:, 0:47])


# #### Get a list of these columns with missing values

# In[ ]:


#print(dataset.columns[0:47])
#dataset.iloc[:,0:47].isna().sum()
cols_missing_vals = []
tot_missing_vals = []
for col,missing in zip(dataset.columns[0:47], dataset.iloc[:,0:47].isna().sum()):
    if missing > 0:
        cols_missing_vals.append(col)
        tot_missing_vals.append(missing)

cols_missing_vals_df = pd.DataFrame({'Variable Name': cols_missing_vals,'Records Mising': tot_missing_vals})


# In[ ]:


## using data from https://wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx?Component=Demographics&CycleBeginYear=2013
## get decription of the questions asked

demog_qs = pd.read_csv("../input/goodhealth/demog_qs.csv")
print("Total number of cols in description file (should be 47): {}".format(len(demog_cols)))
print("Display descriptions with missing records where relevant")

# filter out duplicate Qs by only using those Qs with no use constraints
demog_description  = demog_qs[demog_qs['Variable Name'].isin(demog_cols)]
demog_description  = demog_description[demog_description['Use Constraints']=='None']

# View descriptions of columns with missing data
pd.set_option('display.max_colwidth', -1)
pd.merge(demog_description, cols_missing_vals_df, how='outer', on=['Variable Name'])


# #### Based on this we can:
# * Remove those columsn with many missing values
# * Remove many of the other columns as well
# * Goal is to have a survey someone could fill out online or their mobile
# * Keep:
#     * RIDAGEYR - Age
#     * 'WTINT2YR' -  Weight
#     * RIAGENDR - Gender
#     * INDFMIN2- Total FAMIL  income (need to remove 84 records?)
#     * DMDFMSIZ	Total number of people in the Family	
#     * DMDHHSIZ	Total number of people in the Household	
#     * DMDHHSZA	Number of children aged 5 years or younger in the household	
#     * DMDHHSZB	Number of children aged 6-17 years old in the household	
#     * DMDHHSZE	Number of adults aged 60 years or older in the household

# In[ ]:


##TRY TO COMBINE DMDEDUC2 & DMDEDUC3
DEMO_KEEP_COLS = ['SEQN','RIAGENDR','WTINT2YR','RIDAGEYR','INDFMIN2','DMDHHSIZ','DMDHHSZA','DMDHHSZB','DMDHHSZE']


# * Let's fix the education

# In[ ]:


msno.matrix(dataset[['DMDEDUC2','DMDEDUC3']])


# ### As suspected there missing values complement each other

# In[ ]:


dataset['YearsOfEduc'] = dataset.fillna(0)['DMDEDUC3'] + dataset.fillna(0)['DMDEDUC2']
print("Number of missing records in combined colum: {}".format(dataset['YearsOfEduc'].isna().sum()))


# ### AWESOME!

# In[ ]:


DEMO_KEEP_COLS = DEMO_KEEP_COLS + ['YearsOfEduc']


# ## QUESTIONNAIRE Feature Selection

# In[ ]:


# see if missing values in these columns
count_missing =[]
name_ques_missing =[]
for col, missing in zip(dataset.columns[48:],dataset.iloc[:,48:].isna().sum()):
    if missing > 380:
        count_missing.append(1)
        name_ques_missing.append(col)
print("{} columns with 5% or more missing records".format(len(count_missing)))
      
msno.matrix(dataset.iloc[:, 48:])


# * This is A LOT of missing data. 822 have > 1000 missing! 
# * NOT all might be relevant (could be subquestions)
# * Let's remove the ones with >5% missing - kepp in mind could be eliminating **useful** gender specific data here
# * However, let's hand pick some features.

# In[ ]:


### our list of columns in the questionnaire
### dropping those with so many missing
ques_cols_new = ques_cols.drop(name_ques_missing)


# In[ ]:


ques_qs = pd.read_csv("../input/goodhealth/questionnaire_qs.csv")
print("Total number of cols in description file (should be 47): {}".format(len(ques_cols)))
print("Display descriptions with missing records where relevant")

# filter out duplicate Qs by only using those Qs with no use constraints and where >5% of records are NaN
ques_description  = ques_qs[ques_qs['Variable Name'].isin(ques_cols.drop(name_ques_missing))]
print(ques_description.shape)
ques_description  = ques_description[(ques_description['Use Constraints']=='None') & (ques_description['Variable Name'] != 'SEQN')]
print(ques_description.shape)
len(ques_description['Variable Name'].unique())
ques_description


# ### Based on the info we've selected the below columns to start

# In[ ]:


QUES_KEEP_COLS = ['DLQ010','DLQ020','DLQ040','DLQ050','DLQ060','MCQ010','MCQ053','MCQ082','MCQ086','MCQ092','MCQ203','HIQ011','HUQ051',
                  'HUQ071','HUQ090','PAQ710','PAQ715','DIQ010','SMD460','HOD050','HOQ065','INQ060','INQ080','INQ090','INQ132','INQ140',
                  'INQ150','CBD120','CBD130','FSD032A','FSD032B','FSD032C','FSD151','FSQ165','OHQ030']


# In[ ]:


KEEP_COLS = DEMO_KEEP_COLS + QUES_KEEP_COLS + ['Target']
print("A total of {} columns to keep".format(len(KEEP_COLS)))


# In[ ]:


dataset = dataset[KEEP_COLS]
msno.matrix(dataset)


# In[ ]:


### Not to many NaNs. Lets just take them all out.
dataset = dataset.dropna(axis='index')
print("Final Dataset has {} unique records".format(len(dataset['SEQN'].unique())))
print("{} missing data points".format(dataset.isna().sum().sum()))


# ## Great. FINALLY have our dataset. Now we can explore it a bit

# In[ ]:


dataset.head()


# ###  Some more data cleaning
# *  These data are BINAY, CATEGORICAL, NUMERICAL
# *  From the data scource we can see that there are several options for "don't know" or "refused to answer"
# * We'll clean these out aswell

# In[ ]:


### CATEGORIZE THE COLUMNS
BINARY_COLS = ['RIAGENDR','DLQ010','DLQ020','DLQ040','DLQ050','DLQ060','MCQ010','MCQ053','MCQ082','MCQ086','MCQ092','MCQ203','HIQ011','HUQ071','HUQ090','INQ060','INQ080','INQ090','INQ132','INQ140','INQ150']
# INQ060 7,9
CAT_COLS = ['INDFMIN2','PAQ710','PAQ715','DIQ010','SMD460','HOQ065','FSD032A','FSD032B','FSD032C','FSD151','FSQ165'] #remove don't know/refused as 7,9,77,99,777,999
CAT_COLS_SPEC = ['HUQ051','OHQ030'] # these have 7 & 9 as vals so treat seperate
NUM_COLS = ['WTINT2YR','RIDAGEYR','DMDHHSIZ','DMDHHSZA','DMDHHSZB','DMDHHSZE','YearsOfEduc','HOD050','CBD120','CBD130']
# HOD050 remove 777,999; CBD120/CBD130 remove 777777, 999999


# In[ ]:


### Define a function to remove the rows of "don't know" and "refused"
def remove_refused_dontknow(dataframe, cols_list, removal_list):
    for col in cols_list:
        dataframe = dataframe[~dataframe[col].isin(removal_list)]
    return dataframe


# In[ ]:


print("Original number of records: {}".format(dataset.shape))

dataset = remove_refused_dontknow(dataset, BINARY_COLS, [7,9])
print("Records after BINARY processed: {}".format(dataset.shape))

dataset = remove_refused_dontknow(dataset, CAT_COLS, [7,9,77,99,777,999])
print("Records after CAT processed: {}".format(dataset.shape))

dataset = remove_refused_dontknow(dataset, CAT_COLS_SPEC, [77,99,777,999])
print("Records after CAT_SPEC processed: {}".format(dataset.shape))

dataset = remove_refused_dontknow(dataset, NUM_COLS, [777,999,777777,999999])
print("Records after NUM processed: {}".format(dataset.shape))

dataset = remove_refused_dontknow(dataset, ['YearsOfEduc'], [55,66,99])
print("Records after  Years of Education processed: {}".format(dataset.shape))


# In[ ]:


### DROP THE SEQN COLUMN
dataset = dataset.drop(columns=['SEQN'], axis=1)


# # Target Column
# * Above we created 3 options for Cholesterol - OK, borderline-high, High
# * Since the values for high are relvatively low we are going to re-categorize the problem as OK vs At Risk
# * Makes sense since if we identified someone as having Borderline-high or High we'd advise them to see doctor and get tested

# In[ ]:


sns.set(rc={'figure.figsize':(9, 5)})
sns.countplot(dataset['Target'])
plt.title("Target Variables")
plt.show()


# In[ ]:


def at_risk(score):
    if score == 1:
        return 0 # no risk
    else:
        return 1 # at risk


# In[ ]:


dataset['Target'] = dataset['Target'].apply(lambda x: at_risk(x))
print(dataset.Target.value_counts() / dataset.Target.count()) #not balanced yet, but we'll come back to it
print(dataset.Target.value_counts())
# look at response variable
sns.set(rc={'figure.figsize':(9, 5)})
sns.countplot(dataset['Target'])
plt.title("Target Variables")
plt.show()


# # EDA
# ## Correlation

# In[ ]:


def plot_corr_matrix(dataset):
    # Correlation analasys
    corrMatt = dataset.corr()
    mask = np.array(corrMatt)
    mask[np.tril_indices_from(mask)] = False
    plt.figure(figsize = (20,10))
    sns.heatmap(corrMatt, mask = mask, annot = True)


# In[ ]:


plot_corr_matrix(dataset[NUM_COLS])


# * Some sizable correlations amongst the household size figures and age
# * in retrospect was a bit silly to include all
# * will remove the HH size variables since AGE is important for sure
# 
# First lets correlation with Response variable

# In[ ]:


# see correlatiosn with the RESPONSE variable
dataset.iloc[:,:-1].corrwith(dataset.Target).plot.bar(figsize= (20,10),
                                            title = 'Correlations with Response Variable',
                                            fontsize = 15, rot = 45, grid = True) 
plt.show()


# * HH size variable do correlate, but not as strongly as age.
# * let's still take them out

# In[ ]:


NUM_COLS = [e for e in NUM_COLS if e not in ('DMDHHSIZ','DMDHHSZA','DMDHHSZB','DMDHHSZE')]
dataset = dataset.drop(columns = ['DMDHHSIZ','DMDHHSZA','DMDHHSZB','DMDHHSZE'], axis=1)


# In[ ]:


sns.pairplot(dataset, hue='Target', vars=NUM_COLS)


# ## Check out spread of the data first

# In[ ]:


## Box plots of features and response
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(18,10)
a = sns.boxplot(data=dataset[NUM_COLS[1:]], orient='v', ax=axes[0]) # numerical features except salary
b = sns.boxplot(data=dataset['WTINT2YR'], orient='v', ax=axes[1])  #salary
a.set_xticklabels(labels = NUM_COLS[1:], rotation=90)
b.set_xticklabels(labels=['Household Income'])
a.set_title('Box plot of numerical features (not salary)')
b.set_title('Box plot of salary')
plt.show()


# * Amounts spent on foodout and takeaways have some outliers, lets remove

# In[ ]:


def remove_outliers(dataframe, column, num_std_dev):
    """
    dataframe -- dataframe to trim
    column -- column in the datarame to trim
    num_std -- the number of Standard dev above which to remove (for normal disribution > 2 std is 2.2%)
    """
    mean = np.mean(dataset[column])
    std = np.std(dataset[column])
    
    dataframe = dataframe[dataframe[column] < mean + num_std_dev*std]
    return dataframe


# In[ ]:


## REMOVE OUTLIERS
dataset = remove_outliers(dataset, 'CBD120', 2)
dataset = remove_outliers(dataset, 'CBD130', 2)


# In[ ]:


## CHECK HOW IT LOOKS
sns.set(rc={'figure.figsize':(18, 10)})
ax = sns.boxplot(data=dataset[NUM_COLS[1:]], orient='v') # numerical features except salary
ax.set_xticklabels(labels = NUM_COLS[1:], rotation=90)
ax.set_title('Box plot of numerical features (not salary) - OUTLIERS REMOVED')


# ## How about Binary and Categorical columns

# In[ ]:


(dataset[BINARY_COLS].shape[1])


# In[ ]:


fig = plt.figure(figsize=(18,10))

plt.suptitle('Pie Chart Distributions - BINARY', fontsize = 20)

for i in range(1, dataset[BINARY_COLS].shape[1]+1):
    plt.subplot(5, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset[BINARY_COLS].columns.values[i-1])
    
    values = dataset[BINARY_COLS].iloc[:, i-1].value_counts(normalize = True).values
    index = dataset[BINARY_COLS].iloc[:, i-1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# * Take a look at those small values

# In[ ]:


dataset[dataset['DLQ060'] == 1].Target.value_counts()


# In[ ]:


dataset[dataset['MCQ053'] == 1].Target.value_counts()


# In[ ]:


dataset[dataset['MCQ082'] == 1].Target.value_counts()


# In[ ]:


dataset[dataset['MCQ086'] == 1].Target.value_counts()


# In[ ]:


dataset[dataset['MCQ203'] == 1].Target.value_counts()


# * Both values are represented so we can leave these features

# ### Same for CATEGORICAL features

# In[ ]:


CAT_COLS = CAT_COLS + CAT_COLS_SPEC


# In[ ]:


plt.suptitle('Pie Chart Distributions - CATEGORICAL', fontsize = 20)

for i in range(1, dataset[CAT_COLS].shape[1]+1):
    plt.subplot(5, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset[CAT_COLS].columns.values[i-1])
    
    values = dataset[CAT_COLS].iloc[:, i-1].value_counts(normalize = True).values
    index = dataset[CAT_COLS].iloc[:, i-1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:


dataset[dataset['INDFMIN2'] == 13].Target.value_counts()


# In[ ]:


dataset[dataset['PAQ710'] == 8].Target.value_counts()


# In[ ]:


dataset[dataset['DIQ010'] == 3].Target.value_counts()


# In[ ]:


dataset[dataset['SMD460'] == 3].Target.value_counts()


# In[ ]:


dataset[dataset['HUQ051'] == 7].Target.value_counts()


# In[ ]:


dataset[dataset['OHQ030'] == 7].Target.value_counts()


# * OK these are also good

# # OHE

# In[ ]:


cat_str = dataset[CAT_COLS].astype('category')
cat_ohe = pd.get_dummies(cat_str)

## since these "binary" columns are currently stored as 1 or 2
bin_str = dataset[BINARY_COLS].astype('category')
bin_ohe = pd.get_dummies(bin_str)

dataset = dataset.drop(columns=CAT_COLS, axis=1)
dataset = dataset.drop(columns=BINARY_COLS, axis=1)


# In[ ]:


print(cat_ohe.shape)
print(bin_ohe.shape)
print(dataset.shape)


# In[ ]:


## drop the extra columns to remove dependency
cat_ohe = cat_ohe.drop(columns=['INDFMIN2_1.0','PAQ710_0.0','PAQ715_0.0','DIQ010_1.0','SMD460_0.0','HOQ065_1.0','FSD032A_1.0','FSD032B_1.0',
                          'FSD032C_1.0','FSD151_1.0','FSQ165_1.0','HUQ051_0','OHQ030_1.0'], axis=1)

bin_ohe = bin_ohe.drop(columns = ['RIAGENDR_1','DLQ010_1.0','DLQ020_1.0','DLQ040_1.0','DLQ050_1.0','DLQ060_1.0','MCQ010_1.0','MCQ053_1.0',
                        'MCQ082_1.0','MCQ086_1.0','MCQ092_1.0','MCQ203_1.0','HIQ011_1','HUQ071_1','HUQ090_1.0','INQ060_1.0','INQ080_1.0',
                        'INQ090_1.0','INQ132_1.0','INQ140_1.0','INQ150_1.0'], axis=1)

print(cat_ohe.shape)
print(bin_ohe.shape)


# In[ ]:


dataset = dataset.join(cat_ohe).join(bin_ohe)


# In[ ]:


dataset.shape


# # SPLIT dataset

# In[ ]:


X = dataset.drop(columns=['Target'], axis=1)
y = dataset['Target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)


# In[ ]:


X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[ ]:


print("Size of Training set is {} records".format(X_train.shape[0]))
print("Size of Dev set is {} records".format(X_dev.shape[0]))
print("Size of Test set is {} records".format(X_test.shape[0]))


# # Scale dataset

# In[ ]:


X_Sc = StandardScaler()


# In[ ]:


## Only Scaling the Numerical Columns not Binary
X_train_bin = X_train.drop(NUM_COLS, axis = 1) 
X_dev_bin = X_dev.drop(NUM_COLS, axis = 1) 
X_test_bin = X_test.drop(NUM_COLS, axis = 1) 


X_train = X_train[NUM_COLS]
X_dev = X_dev[NUM_COLS] 
X_test = X_test[NUM_COLS]


# In[ ]:


X_train2 = pd.DataFrame(X_Sc.fit_transform(X_train))
X_dev2 = pd.DataFrame(X_Sc.transform(X_dev))
X_test2 = pd.DataFrame(X_Sc.transform(X_test))

#scaler returns numpy array and lose index and columns names which we don't want!
X_train2.columns = X_train.columns.values
X_dev2.columns = X_dev.columns.values
X_test2.columns = X_test.columns.values

X_train2.index = X_train.index.values
X_dev2.index = X_dev.index.values
X_test2.index = X_test.index.values

# combine the numerical and categorical values
X_train = pd.concat([X_train2, X_train_bin],axis=1, sort=False)
X_dev = pd.concat([X_dev2, X_dev_bin],axis=1, sort=False)
X_test = pd.concat([X_test2, X_test_bin],axis=1, sort=False)

# check shape
print(X_train.shape)
print(X_dev.shape)
print(X_test.shape)

print(y_train.shape)
print(y_dev.shape)
print(y_test.shape)


# In[ ]:


X_train.head(10)


# # **FINALLY** Lets build some models!

# In[ ]:


## First lets have make a table to store our results
results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


# ## 1.Random Forest

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=400)

y_train = pd.DataFrame(y_train)
random_forest.fit(X_train, y_train.values.ravel())


# In[ ]:


### function for plotting scores

def how_did_it_do(ml_model, X_dev, y_dev, use_model_dot_score=True, cf_matrix=True):
    ## predict from dev set
    y_pred = ml_model.predict(X_dev)
    print("performance on X_dev:")
    
    if use_model_dot_score:
        # Accuracy
        print("\nAccuracy:")
        acc = round(ml_model.score(X_dev, y_dev), 3)
        print(acc)
    else:
        print("\nAccuracy score:")
        acc = round(accuracy_score(y_dev, y_pred), 3)
        print(acc)    
    


    # of predicted +ve, how many correct
    print("Precision score:")
    prec = round(precision_score(y_dev, y_pred, average='macro'), 3)
    print(prec)


    # of all actual +ve how many did we get
    print("Recall score:")
    rec = round(recall_score(y_dev, y_pred, average='macro'), 3)
    print(rec)

    # f1 combines
    print("Global F1 score:")
    f1 = round(f1_score(y_dev, y_pred, average='macro'), 3)
    print(f1)
    
    ### plot confusion matrix if needed
    if cf_matrix:
        cm = confusion_matrix(y_dev, y_pred.round())
        df_cm = pd.DataFrame(cm, index = (0,1), columns=(0,1))
        plt.figure(figsize = (10,7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot = True, fmt='g')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
    


# In[ ]:


how_did_it_do(random_forest, X_dev, y_dev, cf_matrix=True)


# In[ ]:


model_results = pd.DataFrame([['RandomForest_1 (n=400)', 0.753, 0.663, 0.571, 0.57]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# ### predicting OK when at risk... could be due to unbalnced data set. Lets balance training set

# # Balance Training Data

# In[ ]:


from imblearn.over_sampling import SMOTE
X_train_nonBal = X_train
y_train_nonBal = y_train


# In[ ]:


train_cols = X_train.columns
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
X_train_res = pd.DataFrame(X_train_res, columns=train_cols)


# In[ ]:


X_train = X_train_res
y_train = pd.DataFrame(y_train_res, columns=["Target"])


# In[ ]:


y_train.Target.value_counts()


# ### **DATA IS BALANCED - MODELLING AGAIN!!**

# ## 2. RANDOM FOREST

# In[ ]:


random_forest_2 = RandomForestClassifier(n_estimators=400)

y_train = pd.DataFrame(y_train)
random_forest_2.fit(X_train, y_train.values.ravel())


# In[ ]:


how_did_it_do(random_forest_2, X_dev, y_dev)


# In[ ]:


model_results = pd.DataFrame([['RandomForest_2 (Balanced Data, n=400)', 0.74, 0.639, 0.595, 0.603]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# ### Improve with Grid Search

# In[ ]:


## Set the params we're gonna try
parameters = {'max_depth': [100, None],
             "max_features" : ["auto", "log2", "sqrt"],
             'n_estimators': [400, 500],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 5, 10],
             'bootstrap': [True, False],
             'criterion': ['gini','entropy']}

from sklearn.model_selection import GridSearchCV

# n_job = -1tells it to use all cores on your computer
grid_search = GridSearchCV(estimator = random_forest_2,
                           param_grid = parameters,
                           scoring = 'f1_macro',
                           verbose=1,
                           cv = 2,
                           n_jobs = 8)


# In[ ]:


"""import time
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train.values.ravel())
t1 = time.time()

print("Took %0.2f seconds" % (t1 - t0))"""


# In[ ]:


"""rf_best_accuracy = grid_search.best_score_
fr_best_paramaters = grid_search.best_params_

print(rf_best_accuracy)
fr_best_paramaters"""


# ## 3. Random Forest

# In[ ]:


random_forest_3 = RandomForestClassifier(n_estimators=400, bootstrap=False, criterion='gini', max_depth=100, max_features='log2',
                                        min_samples_leaf=1, min_samples_split=2)

y_train = pd.DataFrame(y_train)
random_forest_3.fit(X_train, y_train.values.ravel())


# In[ ]:


how_did_it_do(random_forest_3, X_dev, y_dev)


# In[ ]:


model_results = pd.DataFrame([['RandomForest_3 (Grid Search x1)', 0.738, 0.628, 0.573, 0.577]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# ## 4. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(random_state = 0)

logistic_regression.fit(X_train, y_train.values.ravel())


# In[ ]:


how_did_it_do(logistic_regression, X_dev, y_dev)


# In[ ]:


model_results = pd.DataFrame([['Logistic Regression', 0.638, 0.601, 0.628, 0.595]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# In[ ]:


### K-Fold Cross Val 
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = logistic_regression,
                            X = X_train, 
                            y = y_train.values.ravel(),
                            scoring = 'f1',
                            cv = 5)
print(accuracies)
print(accuracies.mean())


# * Analyze coefficinets
# *  RFE

# ## 5. SVM (Linear)

# In[ ]:


from sklearn.svm import SVC

svc_linear = SVC(random_state = 0, kernel = 'linear')

svc_linear.fit(X_train, y_train.values.ravel())


# In[ ]:


how_did_it_do(svc_linear, X_dev, y_dev)


# In[ ]:


model_results = pd.DataFrame([['SVM (linear)', 0.636, 0.61, 0.642, 0.6]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# ## 6. SVM (RBF)

# In[ ]:


from sklearn.svm import SVC

svc_rbf = SVC(random_state = 0, kernel = 'rbf')

svc_rbf.fit(X_train, y_train.values.ravel())


# In[ ]:


how_did_it_do(svc_rbf, X_dev, y_dev)


# In[ ]:


model_results = pd.DataFrame([['SVM (RBF)', 0.63, 0.624, 0.662, 0.604]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# In[ ]:


from sklearn import svm
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, verbose=1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


svc_param_selection(X_train, y_train.values.ravel(), 3)


# In[ ]:


svc_rbf_2 = svm.SVC(kernel = 'rbf', C=10, gamma=1, random_state=42)


# In[ ]:


svc_rbf_2.fit(X_train, y_train.values.ravel())


# In[ ]:


how_did_it_do(svc_rbf_2, X_dev, y_dev)


# In[ ]:


model_results = pd.DataFrame([['SVM (RBF 2) C=10, gamma=1', 0.743, 0.371, 0.5, 0.426]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# # 6. DNN

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[ ]:


model = Sequential([
    Dense(units = 16, input_dim = 79, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dropout(0.7),
    Dense(20, activation = 'relu'),
    Dropout(0.7),
    Dense(20, activation = 'relu'),
    Dropout(0.7),
    Dense(24, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, batch_size=32, epochs=50)


# In[ ]:


y_pred = model.predict(X_dev)
y_pred = y_pred.round()


# In[ ]:


print("\nAccuracy score:")
print(round(accuracy_score(y_dev, y_pred), 3))    

# of predicted +ve, how many correct
print("Precision score:")
print(round(precision_score(y_dev, y_pred, average='macro'), 3))


# of all actual +ve how many did we get
print("Recall score:")
print(round(recall_score(y_dev, y_pred, average='macro'), 3))

# f1 combines
print("Global F1 score:")
print(round(f1_score(y_dev, y_pred, average='macro'), 3))

### plot confusion matrix 

cm = confusion_matrix(y_dev, y_pred.round())
df_cm = pd.DataFrame(cm, index = (0,1), columns=(0,1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot = True, fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:


model_results = pd.DataFrame([['DNN', 0.637, 0.601, 0.629, 0.595]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# # 7. DNN (On original Data - no balance)

# In[ ]:


model_2 = Sequential([
    Dense(units = 16, input_dim = 79, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dropout(0.7),
    Dense(20, activation = 'relu'),
    Dropout(0.7),
    Dense(20, activation = 'relu'),
    Dropout(0.7),
    Dense(24, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

model_2.summary()


# In[ ]:


model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_2.fit(X_train_nonBal, y_train_nonBal, batch_size=32, epochs=50)


# In[ ]:


y_pred = model_2.predict(X_dev)
y_pred = y_pred.round()


# In[ ]:


print("\nAccuracy score:")
print(round(accuracy_score(y_dev, y_pred), 3))    

# of predicted +ve, how many correct
print("Precision score:")
print(round(precision_score(y_dev, y_pred, average='macro'), 3))


# of all actual +ve how many did we get
print("Recall score:")
print(round(recall_score(y_dev, y_pred, average='macro'), 3))

# f1 combines
print("Global F1 score:")
print(round(f1_score(y_dev, y_pred, average='macro'), 3))

### plot confusion matrix 

cm = confusion_matrix(y_dev, y_pred.round())
df_cm = pd.DataFrame(cm, index = (0,1), columns=(0,1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot = True, fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:


model_results = pd.DataFrame([['DNN 2 (Original Data)', 0.643, 0.371, 0.5, 0.426]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results_df = results_df.append(model_results, ignore_index=True)
results_df


# ### **Seems like the SVM RBF (1) is best for us since it minimizes false negatives i.e. catches most people at risk**
# ### Let's test on the test set. Then let's plot some trainnig graphs on the DNN for fun

# # Test set

# In[ ]:


how_did_it_do(svc_rbf, X_test, y_test)


# ## Keras training graphs (for fun)

# In[ ]:


### re-run first DNN but plot learning curve
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=50, batch_size=16, verbose=1)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




