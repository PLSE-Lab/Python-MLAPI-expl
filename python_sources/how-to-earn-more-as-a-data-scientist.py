#!/usr/bin/env python
# coding: utf-8

# **Changes from Version 1:** Explored impact of *"Time from Start to Finish (seconds)"* feature on Salary-predicting model resulting in conclusion that high cardinality features can easily overfit and this feature doesn't influence the salary.
# 
# ## How to earn more?
# 
# This is one of the most important questions for every employee. Of course, salary is not the only aspect to consider when choosing your job. Things like interesting duties, prestigious company name, good boss, friendly colleagues, etc. still matters. But that's already a psychology and not in scope of this notebook. Here we'll take a very pragmatic look at the financial side of being Data Scientist through the possibilities given us by *Kaggle ML & DS Survey* dataset.
# 
# Objective of this notebook is to explore two main questions:
# * what are the most important aspects defining how big is our salary in Data Science and related fields;
# * what things to consider if we want to raise chances of getting paid more.
# 
# We'll take a look at these questions in two scopes - globally and for a specific group of respondents, who specified Job title *"Data Scientist"*.
# 
# First let's take a look, what are salary ranges marked by Respondents.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)

def cat_to_num(x):
    try:
        return int(x.replace(',000', '').replace('+', '').replace('< 1', '0').replace('%', '-').replace(' year', '-').strip().split('-')[0])
    except:
        return -1

df = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False, header=[0,1])
column_descriptions = [col[1] for col in df.columns if col[0] != 'Q9']
df.columns = [col[0] for col in df.columns]

def salary_stats(data, title):
    dfg = data.groupby('Q9')['Q1'].count().reset_index()
    dfg.columns = ['Salary range','# of respondents']
    dfg['sort'] = dfg['Salary range'].apply(cat_to_num)
    dfg['Salary range'] = dfg['Salary range'].apply(lambda x: 'Unknown' if 'not' in x else x)
    dfg = dfg.sort_values('sort')
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x='Salary range', y='# of respondents', data=dfg)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.title(title)
    plt.tight_layout()
    plt.show()
salary_stats(df, 'Number of respondents by salary ranges (USD per year)')


# As we see, salaries can be really different and ranges from less than 10'000 to 500'000+ USD per year. Most of the respondents, who specified their salary, falls in 0-10'000 range. Probably big part of them could be students. Let's plot the same graph more specifically for only those respondents with Job title *"Data Scientist"*.

# In[ ]:


salary_stats(df[df['Q6'] == 'Data Scientist'], 'Number of respondents by salary ranges (USD per year) - Job title: Data Scientist')


# Relative size of 0-10'000 range here is smaller, but overall salaries still differs very significantly - even within the same Job title. So what are the most important reasons for such big differences and what can we as employees do to raise our chances to be on the right side of this graph? What to do to earn more? Let's find out step by step.
# 
# ## What impacts salary size?
# 
# This is the first question to be answered and there are many ways how to answer it. We could go through all the columns of survey data and check how they correlate with salary. Or we could make an educated guess using our domain expert knowledge to choose the most imporant ones. But I prefer more ML related approach. Every question can be answered by a ML model, given enough training data. So let's build a model to answer this one.
# 
# We'll build a regression model predicting salary size from all other columns available. Then we'll inspect what features model finds to be the most important ones. There are 2 things you should know about this model:
# * it is not optimized as we don't care much about how accurate it is - all we need is the list of feature imporances;
# * salary (and other range-based categorical variables) is converted to numerical by taking first part of the range (e.g. 10-20000 becomes 10).

# In[ ]:


import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

dff = df.copy() # copy version to use in ML model
for col in ['Q2','Q8','Q9','Q23','Q25','Q43','Q46']: # convert ranges to numerical
    dff[col] = dff[col].apply(cat_to_num).replace(-1, np.nan)
dff = dff[~dff['Q9'].isnull()] # leave only entries where salary is defined

cat_vars = [f for f in dff.columns if dff[f].dtype == 'object']
for col in cat_vars: # label-encode categorical variables
    lbl = LabelEncoder()
    dff[col] = lbl.fit_transform(dff[col].values.astype('str'))

params = {
    'objective':'regression',
    'metric':'rmse',
    'nthread':4,
    'learning_rate':0.08,
    'num_leaves':31,
    'colsample_bytree':0.9,
    'subsample':0.8,
    'max_depth':5,
    'verbose':-1
}
    
cols = [f for f in dff.columns if f != 'Q9']
train_df = dff[cols]
train_x = lgb.Dataset(train_df, dff['Q9'].values)
clf = lgb.train(params, train_x, 150)

# rename to more reasonable names
train_dfx = train_df[cols].rename({'Q3':'Country','Q2':'Age','Q6':'Job title','Q8':'Job experience',
                           'Q7':'Industry','Q25':'ML methods experience','Q10':'Employer uses ML',
                           'Q1':'Gender','Q23':'Time spent coding','Q42_Part_1':'Measures success by revenue',
                           'Q5':'Undergraduate majors','Q35_Part_3':'ML training at work',
                           'Q32':'Type of the data','Q24':'Experience in analysing data',
                           'Q43':'Exploring unfail bias','Q7_OTHER_TEXT':'Industry - other',
                           'Q34_OTHER_TEXT':'Most time in ML devoted to - other','Q12_Part_1_TEXT':'Tools used for data analysis',
                           'Q1_OTHER_TEXT':'Gender - other','Q35_Part_1':'Self-taught ML','Q6_OTHER_TEXT':'Job title - other',
                           'Q15_Part_2':'Experience with Amazon Web Services','Q11_Part_4':'Builds ML prototypes',
                           'Q27_Part_1':'Experience with AWS Elastic Compute Cloud (EC2)','Q33_Part_9':'Uses GitHub',
                           'Q13_Part_9':'Uses Notepad++','Q11_Part_2':'Builds ML services','Q11_Part_1':'Analyzes data',
                           'Q13_Part_11':'Uses vim','Q30_Part_24':'Uses Big Data products'},axis=1)
train_cols = train_dfx.columns

def display_importances(clf, feats, importance_type='split', title='Salary'):
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = list(feats)
    feature_importance_df["importance"] = clf.feature_importance(importance_type=importance_type)
    cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:22].index
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM features importance for predicting {}'.format(title))
    plt.tight_layout()
    plt.show()
display_importances(clf, train_cols, 'split')


# There is one more option to evaluate features importance - we'll plot also summary by SHAP [1], which shows what impact features have on the value predicted by model. The features importance plot above shows just the fact that one or another feature is important for model to make predictions, but it doesn't give us a clue, what exactly impact this feature has (wether it increases or decreases predictions). SHAP can show us that direction of impact - here also more important features are on the top of the plot.

# In[ ]:


import shap
shap.initjs()
shap_values = shap.TreeExplainer(clf).shap_values(train_dfx)
shap.summary_plot(shap_values, train_dfx)


# The tops of both plots are similar and most of the top features seems logical - it's not a surprise that salary differs a lot in different countries or that job experience helps to get bigger salary. There are also some more interesting/surprising features with noticeable impact - like experience with AWS or usage of GitHub. 
# 
# Now let's go through some of the most impacting and interesting features and see, what exactly impact do they have and is there something for us to consider to use this knowledge in our favour to raise chances of being paid better.

# ## Country
# *In which country do you currently reside?*
# 
# Country is the top influencing feature for salary size in both - LightGBM feature importance and also SHAP estimation. Let's inspect this feature by ploting average salary size by country.

# In[ ]:


def display_stats_by(data, col, col_name, sort='desc', height=10):
    dfg = data.groupby(col)['Salary'].mean().reset_index()
    if sort == 'asc':
        dfg = dfg.sort_values('Salary').head(80)
    if sort == 'desc':
        dfg = dfg.sort_values('Salary',ascending=False).head(80)
    if sort == 'col_to_num':
        dfg['sort'] = dfg[col].apply(cat_to_num)
        dfg = dfg.sort_values('sort').head(80).drop('sort', axis=1)
    dfg['Salary'] = np.round(dfg['Salary']*1000).astype(int)
    dfg.columns = [col_name,'Average salary']
    plt.figure(figsize=(12, height))
    ax = sns.barplot(y="Average salary", x=col_name, data=dfg)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.title('Average salary by ' + col_name)
    plt.tight_layout()
    plt.show()
df['Salary'] = df['Q9'].apply(cat_to_num)
display_stats_by(df, 'Q3', 'Country')


# It is obvious - salary is highly infuenced by the country you live in and difference can be really huge. So relocation to e.g. United States of America or Switzerland is one option to consider to improve chances for bigger salary.
# Let's take a look at the same plot for *"Data Scientist"* Job title.

# In[ ]:


display_stats_by(df[df['Q6'] == 'Data Scientist'], 'Q3', 'Country - Job title: Data Scientist')


# Country to live in is very important also for Data Scientists and salary differences looks to be even bigger here. 
# 
# Conclusion on best option to do for maximizing salary by this feature: **relocate to USA or Switzerland**

# ## Age
# *What is your age (# years)?*
# 
# Age is the second most important feature for salary size in both - LightGBM feature importance and also SHAP estimation. Let's inspect this feature by ploting average salary size by age as usual - globally and for *"Data Scientists"* Job title.

# In[ ]:


display_stats_by(df, 'Q2', 'Age', False, 5)
display_stats_by(df[df['Q6'] == 'Data Scientist'], 'Q2', 'Age - Job title: Data Scientist', False, 5)


# Main trend is quite logical - salary increases by age. The more experience and knowledge you gain, the more evaluated you are by employers. Increase seems to be litle bit more smooth in general than for Data Scientists. I guess there's nothing special we can do regarding this feature to optimize on salary - just have to be patient and gain experience. Salary most likely will increase gradually year by year.
# 
# Conclusion on best option to do for maximizing salary by this feature: **gain experience**

# ## Job title
# *Select the title most similar to your current role (or most recent title if retired): - Selected Choice*
# 
# Job is third feature by importance for salary size in both - feature importance and also SHAP estimation. Let's inspect this feature by ploting average salary size by job title.

# In[ ]:


display_stats_by(df, 'Q6', 'Job title', height=7)


# Not surprisingly - job title matters a lot. Students or assistants get way smaller salary if compared to e.g. managers and chief officiers. In general Data Scientist seems to be quite good job title for receiving big salary, although some other jobs shows way better earning perspectives.
# 
# Conclusion on best option to do for maximizing salary by this feature: **try to become Chief Officer or Principal Investigator**

# ## Job experience
# *How many years of experience do you have in your current role?*
# 
# Job experience is #4 feature for SHAP and #5 in LightGBM feature importances list. Let's inspect this feature by ploting average salary size by job experience:

# In[ ]:


display_stats_by(df, 'Q8', 'Job experience in current role', 'col_to_num', 5)
display_stats_by(df[df['Q6'] == 'Data Scientist'], 'Q8', 'Job experience in current role - Job title: Data Scientist', 'col_to_num', 5)


# No surprises here also - more work experience in current role gives more chances on good salary.
# 
# Conclusion on best option to do for maximizing salary by this feature: **gain work experience**

# ## ML methods experience
# *For how many years have you used machine learning methods (at work or in school)?*
# 
# ML experience is #5 feature for SHAP and #7 in feature importances list. Let's inspect this feature by ploting average salary size by ML methods experience:

# In[ ]:


display_stats_by(df, 'Q25', 'ML methods experience', 'asc')
display_stats_by(df[df['Q6'] == 'Data Scientist'], 'Q25', 'ML methods experience - Job title: Data Scientist', 'asc')


# Trend looks similar to Job experience - more years spent with using ML methods increases chances for good salary.
# 
# Conclusion on best option to do for maximizing salary by this feature: **gain experience working with ML methods**

# ## Industry
# *In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice*
# 
# Industry is also very important - this is #6 feature in LightGBM feature importances list and #7 in SHAP estimation. Let's inspect this feature by ploting average salary size by Industry.

# In[ ]:


display_stats_by(df, 'Q7', 'Industry', height=8)
display_stats_by(df[df['Q6'] == 'Data Scientist'], 'Q7', 'Industry - Job title: Data Scientist', height=8)


# In general it looks that biggest chances for good salary is for employees in *Hospitality/Entertainment/Sports* industry. It is even more noticeable for Job title *"Data Scientist"*.
# 
# Conclusion on best option to do for maximizing salary by this feature: **consider working in *Hospitality/Entertainment/Sports* industry**

# ## Time from Start to Finish (seconds)
# *Average Time Spent Taking the Survey*
# 
# This one looks really suspicious. Time of completing the Survey seems to be one of the top features for salary size - it is #4 feature in LightGBM feature list and #8 in SHAP estimation. Would that mean our salary depends on how fast we are answering Kaggle Surveys? Of course not - model is simply overfitting on this feature. One of most likely reasons for overfitting is very high cardinality of this feature relatively to others. 

# In[ ]:


print('Unique values of "Time from Start to Finish (seconds)":',df['Time from Start to Finish (seconds)'].unique().shape[0])


# This feature has 6522 unique values while all other Survey questions have very limited choices for response.
# High cardinality of this feature allows model to find some small sub-sets of those 6522 values which correlates with salary and makes model to perform splits on these sub-sets resulting in overfitting and high importance of this feature.
# 
# In order to prove this assumption let us make a quick experiment - let's fit the same model on some random target. We'll make random target by simply shifting real target values by one - in this case distribution and other characteristics of data remains the same, but target values are now randomly assigned to different data rows. Let's fit this model and display importances.

# In[ ]:


train_x2 = lgb.Dataset(train_df, dff['Q9'].shift(1).values)
clf2 = lgb.train(params, train_x2, 150)
display_importances(clf2, train_cols, 'split', title='Random noise')


# As expected, *"Time from Start to Finish (seconds)"* feature is the most important on fitting random target because of it's high cardinality, which allows model to find "useful" sub-sets even for fitting to noise. Btw, this is the reason why I also always double-check cases when some very high-cardinality feature turns out to be on the top of importance list in e.g. Kaggle competitions.
# 
# Conclusion on best option to do for maximizing salary by this feature: **beware of features with very high cardinality ;)**

# # Summary
# 
# This is the second version of notebook, exploring 7 most important features which influence size of the salary in general and more specifically for Data Scientists.
# 
# Main conclusions so far:
# * Country seems to be the most dominant factor influencing salary - best option for Data Scientist (and also in general) is to live in Switzerland or USA.
# * Work experience in general and also more specifically experience with ML is also very important.
# * Job title matters a lot for salary size. *"Data Scientist"* role is quite well paid if compared to other jobs, although there are also more profitable positions.
# * High cardinality features can easily result in overfitting, especially if other features have very low cardinality.
# 
# Most of the features we've explored so far all were logical and quite predictable. More interesting and surprising features are to be explored in next versions.
