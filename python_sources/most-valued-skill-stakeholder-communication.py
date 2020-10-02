#!/usr/bin/env python
# coding: utf-8

# In the Kaggle Survey 2018 one of the questions considered the annual income. Some Kagglers submitted the information allowing some insight in why some make more than others. With this kernel i highlight the most important features of a kaggler coinciding with high income.
# 
# The following Hypothesis can be formed from the data:
# 
# **Stakeholder communication is the most important skill among professionals**
# The participants spending most of their time with Stakeholder communication (Q34) have the best yearly income. Thus this might be the most important skill for an actual ML/DS Job
# 
# **Age is more relevant than ML/DS Experience**
#  Age and professional Experience in ML/DS are among the most important factors determining income. However Age is a stronger predicotr as you will see later. This might be a hint that most kagglers do not actually work in a ML/DS related Job, or not since a long time
# 
# **Education earned online is not as valued as formal education or working experience**
# When it comes to education (Q35) Working experience beats University education beats online education (including kaggle) as indicator for high income. 
# 
# The Text field for Q12 Part 4, providing details about Local or hosted development environments as entered by the participant is the single most important feature in the model. However the actual text entered is lost to anonymization.
# 
# Please let me know whether you agree/disagree with my interpretation of these observations.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/multipleChoiceResponses.csv')

questions = df.iloc[0,:]
answers = df.iloc[1:,:]


# In[ ]:


def tofloat(row, colname, replacement):
    try:
        row[colname] = float(row[colname])
    except:
        row[colname] = replacement
    return row

def tryfloat(row, columns):
    # try to make everything a float that does not resist with force
    for col in columns:
        try:
            row[col] = float(row[col])
        except:
            pass
    return row

def replace_by_middle(row, colname):
    # try to replace range by middle value. It gets kind of messy to deal with the different notations in these columns
    try:
        row[colname] = float(row[colname])
        return row
    except:
        pass
    if row[colname] == 'NaN':
        row[colname] = -999999
        return row
    try:
        row[colname] = float(row[colname].split('+')[0])
        return row
    except:
        pass 
    row[colname] = row[colname].replace(',000','')
    if row[colname] == 'I do not wish to disclose my approximate yearly compensation':
        row[colname] = -999999
        return row
    try:
        split = row[colname].split('-')
        row[colname] = (float(split[0])+float(split[1]))/2
    except:
        #print('Mapping ' + str(row[colname]) + ' to low value')
        row[colname] = -999999
    return row

# time tends to show up prominently, in feature importance. this does not look like a useful result
answers = answers.drop('Time from Start to Finish (seconds)', axis = 1)
tflambda = lambda x: tryfloat(x, answers.columns)
answers = answers.apply(tflambda, axis = 1)

for column in answers.columns:
    if '_OTHER_TEXT' in column:
        conversion = lambda x: tofloat(x,column,-999999)
        answers = answers.apply(conversion, axis=1)

for colname in ['Q2','Q8','Q9']:
    conversion = lambda x: replace_by_middle(x,colname)
    answers = answers.apply(conversion, axis=1)

answers = answers[answers['Q9'] > 0] # dismiss where no answer was provided


# This is the income distribution with values above 250K excluded and those who did not respond. It pretty skewed to the right.

# In[ ]:


# remove were now answer is given
answers = answers[answers['Q9'] > 0]
# The income distribution is pretty skewed, we will cut it off at the price which is still reasonably far away from the median.
# This way the results are more meaningful for the biggest pack of people living around 50K/year
answers = answers[answers['Q9'] < 250]
# hist of incomes
plt.figure()
answers['Q9'].plot.hist()


# In[ ]:


df_train = pd.get_dummies(answers)


# In[ ]:


y = df_train['Q9']
X = df_train.drop('Q9', axis = 1)
n_train = int(X.shape[0] * 0.7)
X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_valid, y_valid = X.iloc[n_train:], y.iloc[n_train:]


# To judge the importance of difference features for the income of the participant we will train a gradient boosting random forest to the data with the result of Q9, which is the yearly income, as target variable. To train a regression models the bins have been replaced by their midpoints in the data.
# 
# The resulting feature importances are shown below, but only for features with importance > 0.5%

# In[ ]:



m = lgb.LGBMRegressor(objective = 'regression_l1',
                      num_boost_round=1000,
                        learning_rate = 0.1,
                        num_leaves = 127,
                        max_depth = -1,
                        lambda_l1 = 0.0,
                        lambda_l2 = 1.0,
                        metric = 'l2',
                        seed = 42)  
m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
            early_stopping_rounds=50)


# In[ ]:


def getQuestion(row):
    row['question'] = row['Feature'].split('_')[0]
    return row

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
feature_imp = pd.DataFrame(sorted(zip(m.feature_importances_,X.columns)), columns=['Value','Feature'])
feature_imp['Value'] = feature_imp['Value']/feature_imp['Value'].sum() # normalize to 1
feature_imp = feature_imp.apply(getQuestion, axis = 1)
#feature_imp = feature_imp.drop('Feature', axis = 1)
#feature_imp = feature_imp.groupby(by='Feature', as_index=False).sum()
# group the values by Question
sorted_values = feature_imp.sort_values(by="Value", ascending=False)
# keep only the top 80% important
sorted_values['cum'] = sorted_values['Value'].cumsum()/sorted_values['Value'].sum()
sorted_values = sorted_values[sorted_values['Value'] > 0.005]

# save for later use
important_features = sorted_values['Feature'].copy()
#sorted_values = sorted_values.head(10)

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=sorted_values)
plt.title('LightGBM Question Importanct')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# Questions Q2 and Q8 are about age and professional experience. Hence their importance for salary seems intuitive. All Features which contribute less than 0,5% are not shown. Note how country, result of Q3, does not show up.
# 
# Most participants are from the US and India. This might indicate that the pay gap within each country is more significant than the pay gap between countries.
# 
# First lets have a look at the relationship of age and working experience with salary. The axis are logarithmic to deal with the skewed nature of the distribution.

# In[ ]:


print('Correlation of Age and Experience: ' + str(answers['Q2'].corr(answers['Q8'])))
# Age 
sns.jointplot(np.log(answers['Q2']), np.log(answers['Q9']), kind='kde').set_axis_labels('log Age [years]', 'log Income [$1000]', fontsize=16)
# Working Experience im ML
sns.jointplot(np.log(answers['Q8']), np.log(answers['Q9']), kind='kde').set_axis_labels('log Experience in ML/DS [years]', 'log Income [$1000]', fontsize=16)


# Feature importances and KDE Plots indicate higher importance of Age, than working experience in ML
# One explanation might be that ML is not the core qualification as workers for most kagglers.
# 
# When diving into the Question 12, 35 and 34 we will be looking at the median income in different groups rather than the mean. This is because income is crowded in the lower segments with some high rollers. Thus it is far from normal and the median will give a value nearer to  the peak of the distribution.
# 
# Next thing up is the Question 12 Part 4 which has scored as most important for income. It is a free text field asking which tools we use. Bad thing the free text answers are seperated and i do not know if there is any way to link this back to the original text. Also i am not sure what we see in the column, maybe the number of characters used in the answer?

# In[ ]:


answers['Q12_Part_4_TEXT'].describe()


# We know for sure that -1 means no response. So we compare the median income of participants who have provided an answer and those who did not.

# In[ ]:


def setranking(row):
    if row['Q12_Part_4_TEXT'] < 0:
        row['Q12_ranking'] = 'No answer'
        return row
    row['Q12_ranking'] = 'Some answer'
    return row

answersq12 = answers.copy()
#split by quantiles
answersq12['Q12_ranking'] = 'NaN'
answersq12 = answersq12.apply(setranking, axis=1)
answersq12 = pd.DataFrame(answersq12.filter(['Q12_ranking','Q9'],axis=1).groupby(by='Q12_ranking', as_index=False).median())

#sns.jointplot(answersq12['Q12_Part_4_TEXT'], answersq12['Q9'], kind='kde') # make the col strictly positive

plt.figure(figsize=(20, 10))
sns.barplot(x="Q9", y="Q12_ranking", data=answersq12.sort_values(by='Q9', ascending=False))
plt.title('Median Income in Q12 groups')
plt.tight_layout()
plt.show()
plt.savefig('q12_on_income.png')


# The Median income is 10K higher for people who have provided any reply for the kinds of tools they use. However i could not yet identify a strong relationship with the number given in the field. I suspect a lot of meaning was lost when scrambling the answers to provide anonymity.
# 
# The questions 35 deals with how much different sources of learning material contributet to your ML education. The question asked to distribute 100% toward the following sources of learning:
# 
# 1: self taught
# 
# 2: Online Courses
# 
# 3: Work
# 
# 4: University
# 
# 5: Kaggle
# 
# 6: Other
# 
# So we are going to split the participants based on where most points were given

# In[ ]:


def setranking(row):
    q35 = [row['Q35_Part_1'], row['Q35_Part_2'], row['Q35_Part_3'], row['Q35_Part_4'], row['Q35_Part_5'], row['Q35_Part_6']]
    if q35[0] == max(q35):
        row['q35_group'] = 'Self taught'
    if q35[1] == max(q35):
        row['q35_group'] = 'Online Courses'
    if q35[2] == max(q35):
        row['q35_group'] = 'Work'
    if q35[3] == max(q35):
        row['q35_group'] = 'University'
    if q35[4] == max(q35):
        row['q35_group'] = 'Kaggle'
    if q35[5] == max(q35):
        row['q35_group'] = 'Other'
    return row

answersq35 = answers.copy()
#split by quantiles
answersq35['q35_group'] = 'Not specified'
answersq35 = answersq35.apply(setranking, axis=1)
answersq35 = pd.DataFrame(answersq35.filter(['q35_group','Q9'],axis=1).groupby(by='q35_group', as_index=False).median())

#sns.jointplot(answersq12['Q12_Part_4_TEXT'], answersq12['Q9'], kind='kde') # make the col strictly positive

plt.figure(figsize=(20, 10))
sns.barplot(x="Q9", y="q35_group", data=answersq35.sort_values(by="Q9", ascending=False))
plt.title('Median Income in Q35 groups')
plt.tight_layout()
plt.show()
plt.savefig('q35_on_income.png')


# It is no surprise to see work on the top. University education beats kaggle and online courses by about by around 10K. Being able to get the actual text entered for option OTHER would be a huge plus. But as of my understanding we can not get it.
# 
# We will look at Q34 in a very similar manner. Kagglers were asked to distribute a total of 100% on the following activites, in relation to time spent in projects:
# 
# 1: Gathering data
# 
# 2: Cleaning data
# 
# 3: Visualizing data
# 
# 4: Model building
# 
# 5: Putting the Model to production
# 
# 6: Stakeholder communication
# 

# In[ ]:


def setranking(row):
    q34 = [row['Q34_Part_1'], row['Q34_Part_2'], row['Q34_Part_3'], row['Q34_Part_4'], row['Q34_Part_5'], row['Q34_Part_6']]
    if q34[0] == max(q34):
        row['q34_group'] = 'Gathering data'
    if q34[1] == max(q34):
        row['q34_group'] = 'Cleaning data'
    if q34[2] == max(q34):
        row['q34_group'] = 'Visualizing data'
    if q34[3] == max(q34):
        row['q34_group'] = 'Model building'
    if q34[4] == max(q34):
        row['q34_group'] = 'Putting Model to Production'
    if q34[5] == max(q34):
        row['q34_group'] = 'Stakeholder communication'
    return row

answersq34 = answers.copy()
#split by quantiles
answersq34['q34_group'] = 'Not specified'
answersq34 = answersq34.apply(setranking, axis=1)
answersq34 = pd.DataFrame(answersq34.filter(['q34_group','Q9'],axis=1).groupby(by='q34_group', as_index=False).median())

#sns.jointplot(answersq12['Q12_Part_4_TEXT'], answersq12['Q9'], kind='kde') # make the col strictly positive

plt.figure(figsize=(20, 10))
sns.barplot(x="Q9", y="q34_group", data=answersq34.sort_values(by="Q9", ascending=False))
plt.title('Median Income in Q34 groups')
plt.tight_layout()
plt.show()
plt.savefig('q34_on_income.png')


# The median income is highest for the group which spends most time with stakeholder communication, surpassing the next group by about 10K.
# 
# 
# This is it for now, please leave a comment on what i can improve. Hope you enjoyed
