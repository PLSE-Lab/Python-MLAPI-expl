#!/usr/bin/env python
# coding: utf-8

# # Learning Analytics - Multiple Linear Regression

# Data Source: https://www.kaggle.com/rocki37/open-university-learning-analytics-dataset
# Data description: https://analyse.kmi.open.ac.uk/open_dataset#description
# 
# In this investigation, I am looking at whether student data such as gender and age, as well as interaction in the VLE can be a predictor of assessment scores by module. 

# Database Schema:
# 

# In[ ]:


from IPython.display import Image
Image("../input/databaseschema/DataBase_schema.png")


# ## Initial Question: 
# Can we use what we know about students, and their engagement on the VLE to predict assessment outcomes?  
# 
# We will use:  
# average number of clicks per module  
# average score on assessments per module

# ## Loading the Data

# In[ ]:


import os
print(os.listdir('../input/open-university-learning-analytics-dataset'))


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


assessments = pd.read_csv('../input/open-university-learning-analytics-dataset/assessments.csv')
student_assessment = pd.read_csv('../input/open-university-learning-analytics-dataset/studentAssessment.csv')
student_info = pd.read_csv('../input/open-university-learning-analytics-dataset/studentInfo.csv')
vle_activity = pd.read_csv('../input/open-university-learning-analytics-dataset/studentVle.csv')

#we only need code_module so that we can perform a join with studenAssessment and then studentInfo tables
assessments.drop(['code_presentation','assessment_type','date','weight'], axis = 1, inplace = True)
#assessments.code_module.value_counts()


# In[ ]:


#merge individual assessments data with assessment data, so that we know which module each assessment belongs to
comb_assess = pd.merge(student_assessment,assessments,on='id_assessment')
comb_assess.drop(['is_banked','date_submitted'],axis = 1,inplace=True)
comb_assess.dtypes


# In[ ]:


#clean some of the data - we don't want score as a string, we want it as an integer. So let's remove the ?
#comb_assess.drop(comb_assess[comb_assess.score == '?'].index, inplace = True);
#comb_assess.score = comb_assess.score.astype(int);
#comb_assess.dtypes


# In[ ]:


#how much data do we have?
comb_assess.shape


# In[ ]:


#group by student and then subgroup by module, so that we retain ability to evaluate different module predictability
grouped = comb_assess.groupby(['id_student','code_module']).mean()
grouped.sort_values('id_student')
#we can't keep id_assessment, because we have just grouped by module. We have to group by module, as our vle interaction data is 
#only grouped by module
grouped.drop(['id_assessment'],axis=1,inplace = True)
grouped.head()


# In[ ]:


student_info.shape


# In[ ]:


#put it all together
student_all_info = pd.merge(student_info,grouped,on='id_student')
#just getting a feel for it - how many modules is each student enrolled in?
fig1 = student_all_info.groupby(['id_student']).code_module.count().sort_values().hist()
fig1.set_title('Number of modules by student')
fig1.set_xlabel('Number of modules')
fig1.set_ylabel('Number of students')


# Note: The plot above shows most students are enrolled in one module. 

# In[ ]:


#total number of clicks per student by module
vle_grouped = vle_activity.groupby(['id_student','code_module']).sum()
#we have to drop the columns below as we have grouped by student and subgrouped by module, so they are meaningless
vle_grouped.drop(['id_site','date'],axis=1,inplace=True)
vle_grouped.head()


# In[ ]:


student_all_info.shape


# In[ ]:


#left join as we want to keep student info where no clicks were made
df = pd.merge(student_all_info,vle_grouped,on = ['id_student','code_module'],how='left')


# In[ ]:


df.shape


# In[ ]:


#remove rows where there are null values for sum_click. There are only 201 so it won't have a huge impact
df.dropna(inplace=True)


# In[ ]:


#save the new table
df.to_csv('joinedData.csv',index=False)


# So we now have a dataframe with student info, average assessment score for that module and average number of clicks for that module. We will now inspect the data for missing data, and ensure it's clean. 

# ## We need to create the categorical variables

# In[ ]:


df.dtypes


# In[ ]:


df.code_module = pd.Categorical(df.code_module)
df.code_presentation = pd.Categorical(df.code_presentation)
df.gender = pd.Categorical(df.gender)
df.region = pd.Categorical(df.region)
df.highest_education = pd.Categorical(df.highest_education)
df.imd_band = pd.Categorical(df.imd_band)
df.age_band = pd.Categorical(df.age_band)
df.disability = pd.Categorical(df.disability)
df.final_result = pd.Categorical(df.final_result)


# In[ ]:


df.dtypes


# ## Initial exploration of the data

# In[ ]:


df.head()


# In[ ]:


import plotly.express as px
data = df
fig = px.box(data, x="code_module", y="score",title='Student average scores by Module')

fig.show()


# In[ ]:


data = df
fig = px.box(data, x="region", y="score",title='Student average scores by Region')

fig.show()


# ### What this tells us:
# All modules have a lower tail, and 6 out of 7 have zero scores included  
# Scores by region are fairly consistent  
# 
# Let's look at numbers of each students' highest achieved education level, and see if that tells us anything

# In[ ]:


highest_ed = df.highest_education.value_counts()
f, ax = plt.subplots(figsize=(18,5))
ax.bar(highest_ed.index,highest_ed)
ax.set_ylabel('number of students')
df.highest_education.cat.categories


# ### What this tells us:
# We need to be careful about generalising for the two categories Post Grad and No prior quals as they only have small numbers  
# 
# And below, we can see there isn't much variation by gender

# In[ ]:


for_bar = df.pivot_table(index = 'highest_education', columns='gender', values = 'score')
for_bar.plot(kind='bar')


# In[ ]:


df.head()


# In[ ]:


interaction_by_module = df.sum_click
fig2, ax2 = plt.subplots(figsize=(5,5));
ax2.hist(interaction_by_module,bins=50);
ax2.set_xlabel('Number of Clicks by module');
ax2.set_title('Number of clicks by module for each student');
ax2.set_ylabel('Number of occurences');


# In[ ]:


interaction_by_module = df.sum_click
fig2, ax2 = plt.subplots(figsize=(5,5));
ax2.boxplot(interaction_by_module);
ax2.set_xlabel('Number of Clicks by module');
ax2.set_title('Number of clicks by module for each student');
ax2.set_ylabel('Number of occurences');


# In[ ]:


plt.scatter((df.sum_click),(df.score))


# ### What this shows us
# Most students perform under 1000 clicks per module. There is a long tail of keen students though!  
# There isn't a clear linear relationship between number of clicks and average assessment score by module

# In[ ]:


plt.scatter((df.studied_credits),(df.score))


# ### What this shows us
# We should put the studied_credits data in bins, as it would make more sense as categorical data - it only appears as certain value.

# In[ ]:


bins = [0,50,100,150,200,250,300,350,400,450,500,550,600]
df['studied_credits'] = pd.cut(df['studied_credits'], bins=bins)


# In[ ]:


df2 = df.groupby(['gender','code_module']).score.mean()


# In[ ]:


df3 = df.groupby(['gender','code_module']).sum_click.mean()
codes =  df2.index.get_level_values(1)
codes
sns.scatterplot(df3,df3.index.get_level_values(1), hue = df2.index.get_level_values(0), legend='full');


# ### What this shows us
# Women tend to be more engaged with the VLE

# ### Question: Does this higher engagement for women translate to better scores? 

# In[ ]:


import seaborn as sns
codes =  df2.index.get_level_values(1)
codes
sns.scatterplot(df2,df2.index.get_level_values(1), hue = df2.index.get_level_values(0), legend='full');


# ### What this shows us
# There aren't huge differences in performance by gender for each module

# ## Does the total number of clicks impact the final result?

# In[ ]:


df.groupby('final_result').sum_click.mean().sort_values().plot(kind='bar',)


# So there does seem to be some pattern between final score and engagement on the vle

# Let's look at the range now

# In[ ]:


data = df
fig = px.box(data, x="final_result", y="sum_click",title='Student final score by the total clicks on the vle')

fig.show()


# So it looks like there are some keen beans skewing our conclusion a bit, but there is still some difference in clicks by result  
# 
# And how about final score and average score for that module? There should be a relationship no?

# In[ ]:


df.groupby('final_result').score.mean().sort_values().plot(kind='bar',)


# ### What this shows us:
# It looks as though the number of clicks above 700 isn't that predictive, but below 700 could be. Particularly of withdrawal and failure. There also seems to be a link between final_result and number of clicks

# ### Next question: Let's look just at those people/modules that have less than 700 clicks, and see if it's more predictive

# In[ ]:


plt.scatter((df.sum_click[df.sum_click < 700]),(df.score[df.sum_click < 700]))


# ## Overall conclusion so far
# We have seen that number there is a relationship between number of clicks and final result, and particularly that it might be possible at the lower end of engagement to predict withdrawal or failure from VLE engagement. However our independent variables by themselves don't seem to be predictive of mean assessment score.  
# 
# Let's continue anyway, to see if we get any results from a multiple linear regression

# In[ ]:


#Let's check the datatypes
df.dtypes


# In[ ]:


df.isnull().sum()
df.dropna(inplace=True)


# In[ ]:


df.describe()


# So num_of_prev_attempts is heavily positively skewed. studied_credits might have significant outliers. There is a large variation in sum_click, with std larger than mean. 

# In[ ]:


df_target = df.score
df.drop(['score'],axis=1,inplace=True)
df.drop(['id_student'],axis=1,inplace=True)


# Check for multicollinearity

# In[ ]:


df.head()
df.dtypes


# In[ ]:


#Check multicollinearity
corr = df.corr()
corr


# There is no significant multicollinearity between independent variables  
# 
# Next, let's normalise the continuous variables - num_of_prev_attempts, studied_credits,score,date, sum_clicks, plus the target variable score

# In[ ]:


df.hist()


# ### Scaling the continuous features

# In[ ]:


df.num_of_prev_attempts = (df.num_of_prev_attempts - df.num_of_prev_attempts.mean())/df.num_of_prev_attempts.std()
df.sum_click = (df.sum_click - df.sum_click.mean())/df.sum_click.std()
df_target = (df_target - df_target.mean())/df_target.std()


# In[ ]:


df.hist();


# In[ ]:


df_target.hist();


# ### How normal are these independent variables?

# In[ ]:


#test skew and kurtosis
print("Kurtosis",df.kurtosis(axis=0))
print("Skew",df.skew(axis=0))
print("Target Kurtosis",df_target.kurtosis(axis=0))
print("Target Skew",df_target.skew(axis=0))


# All variables have positive skewness and kurtosis nowhere near zero. However we can't log transform with the data we have, as there are lots of invalid values including 0s. 

# ### Perform log transformation on the variables - apart from date which has negative values

# In[ ]:


df.num_of_prev_attempts.value_counts()


# In[ ]:


df_trans = df
df_trans.head()
#df_trans['num_of_prev_attempts'] = np.log(df.num_of_prev_attempts)
#df_trans['studied_credits'] = np.log(df.studied_credits)
#df_trans['sum_click'] = np.log(df.sum_click)
#df_trans_target = np.log(df_target)


# In[ ]:


df_target.hist()
df_trans.hist()


# # Modelling

# ## One hot encoding

# In[ ]:


df_trans = pd.get_dummies(df_trans)
df_trans.shape
df_trans.dtypes
for i in df_trans.columns[2:]:
    df_trans[i] = df_trans[i].astype('category')


# In[ ]:


df_trans.dtypes
df_trans['score']=df_target


# ## Check for Linearity between each independent variable and the target variable

# In[ ]:


plt.scatter(df_trans['sum_click'],df_target)
df_trans.sum_click.isnull().sum()
df_trans.shape


# In[ ]:


#replace spaces in strings with _ for modelling purposes
df_trans.columns = df_trans.columns.str.replace(' ', '_')
df_trans.columns = df_trans.columns.str.replace('-', '_')
df_trans.columns = df_trans.columns.str.replace('%', '')
df_trans.columns = df_trans.columns.str.replace('?', '')
df_trans.columns = df_trans.columns.str.replace('<', '')
df_trans.columns = df_trans.columns.str.replace('=', '')
df_trans.columns = df_trans.columns.str.replace(']', ')')
df_trans.columns = df_trans.columns.str.replace('(', '')
df_trans.columns = df_trans.columns.str.replace(')', '')
df_trans.columns = df_trans.columns.str.replace(',', '')

df_trans.head()


# ## Let's look at linear regression by variable

# In[ ]:


import statsmodels.formula.api as smf

dfcol = ['num_of_prev_attempts','sum_click']
result = []
for count, i in enumerate(dfcol):
    formula = 'score ~' + ' ' + i
    model = smf.ols(formula, data = df_trans).fit()
    #print(model.params[0],model.params[1],model.pvalues[1])
    result.append([i, model.rsquared, model.params[0],model.params[1],model.pvalues[1]])
result


# So not great. Let's try the categorical variables

# In[ ]:


df_trans.columns


# In[ ]:


cols_module= df_trans.columns[2:9]
cols_pres= df_trans.columns[9:13]
cols_gender = df_trans.columns[13:15]
cols_region = df_trans.columns[13:28]
cols_ed = df_trans.columns[28:33]
cols_imd = df_trans.columns[33:44]
cols_age = df_trans.columns[44:47]
cols_cred = df_trans.columns[47:59]
cols_dis = df_trans.columns[59:61]
cols_result = df_trans.columns[61:65]

print(cols_result)


# In[ ]:


cols = [cols_module, cols_pres , cols_gender, cols_region,cols_ed,cols_imd,cols_age,cols_cred,cols_dis,cols_result]
for col in cols:
    sum_cols = "+".join(col)
    form = "score ~" + sum_cols
    model = smf.ols(formula= form, data= df_trans).fit()
    #model = smf.ols(formula, data = df).fit()
    print(model.summary())


# Let's drop some of the less predictive columns, and one from each category

# In[ ]:


df_final = df_trans.drop(["num_of_prev_attempts","code_module_AAA","code_presentation_2013B","gender_F","region_East_Anglian_Region","highest_education_No_Formal_quals","imd_band_0_10","studied_credits_550_600","disability_Y","final_result_Fail"], axis=1)
y = df_final[["score"]]
X = df_final.drop(["score"], axis=1)


# In[ ]:


df_final.shape


# ## Feature selection
# We will use recursive feature selection to find the best combination of features. 

# In[ ]:


import statsmodels.formula.api as smf
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression



r_list = []
adj_r_list = []
list_n = list(range(1,56,2))
for n in list_n: 
    linreg = LinearRegression()
    select_n = RFE(linreg, n_features_to_select = n)
    select_n = select_n.fit(X, np.ravel(y))
    selected_columns = X.columns[select_n.support_ ]
    linreg.fit(X[selected_columns],y)
    yhat = linreg.predict(X[selected_columns])
    SS_Residual = np.sum((y-yhat)**2)
    SS_Total = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    print(r_squared)
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    print(adjusted_r_squared)
r_list.append(r_squared)
adj_r_list.append(adjusted_r_squared)


# ## Conclusion:
# This regression didn't yield much. However we can look further into the relationship between final score and vle engagement using logarithmic regression. 
