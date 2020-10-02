#!/usr/bin/env python
# coding: utf-8

# # IBM HR Dataset: Exploratory Data Analysis
# 
# Source data available here:
# https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
# 
# This is the first of a set of these I will be working on to show off some of the analytic skills I've gained over the past 12 months. It's not a polished report and the graphs are stock Seaborn graphs but hopefully it gives you an idea of how I think about data and the sorts of things I can do.
# 
# This dataset isn't as `dirty` as I would like to show off some of my cleaning skills but alas that'll have to wait for another day. 
# 
# For this analysis I'm going to assume it's for a fictional company called TechCo! Which of course is the name of some real companies but it's too late to turn back now.
# 
# ![](http://www.techco.ab.ca/logotech.gif)
# 
# ## Part 1: Cleaning the Data
# ## Part 2: Who leaves TechCo?
# ## Part 3: Gender Equality at TechCo
# ## Part 4: Predicting attrition at TechCo

# In[37]:


# REQUIRES PYTHON 3.6

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re

from scipy import stats
from functools import reduce

# Some matplotlib options
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use("ggplot")

# General pandas options
pd.set_option('display.max_colwidth', -1)  # Show the entire column 
pd.options.display.max_columns = 100 
pd.options.display.max_rows = 10000 

# Seaborn options
sns.set_style("whitegrid")


# In[38]:


# Load the data in
hr = pd.read_csv("../input/ibm-hr-wmore-rows/IBM HR Data.csv")

# Lets see what it looks like
print(hr.shape)
before_dedup = hr.shape[0]
hr.describe(include='all')


# ## Initial Brainstorm
# 
# - Might have some interesting clusters 
# - Correlation plot would be nice
# - Can we train a model to predict attrition? Can get some feature importances.
# - What role does gender and age play in this workplace?
# - How much does income matter?
# - Are more educated people more likely to leave?
# - Which Education Field is more likely to leave?
# - Does the source of the employee matter? Which website produces the best employees?
# - Is JobLevel equivalent to performance? Can we create a feature to measure this?
# - Is there a difference in gender?
# - People who live more than 25 minutes away from work tend to be less happy than those who don't. Lets see whether this is the case.
# - What sources are best for high performing employees?
# - What factors contribute to retaining high performing employees?
# - How does satisfaction and work-life balance relate to employee retention?
# - How many new hires leave in less than a year and why?

# In[39]:


hr.head()


# ## Part 1: Cleaning the Data
# 
# - Some of the rows are misaligned at different places but it appears the last row always gets filled
# - Few missing values but appears to only be a small fraction of the entire dataset
# - Might have some duplicates

# In[40]:


# Check for missings
print(np.count_nonzero(hr.isnull().values))
print(hr.isnull().any())

# Check for duplicates
print(hr[hr.duplicated(keep=False)].shape)

# Strip whitespaces
hr = hr.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Check for conflicting types
hr.dtypes


# In[41]:


# Hate to throw away data but it's only 353 values out of over 23 thousand
hr.dropna(axis=0, inplace=True)

# Get rid of all the duplicates
hr.drop_duplicates(inplace=True)

# Lets see what it looks like now
print("Duplicates Removed: " + str(before_dedup - hr.shape[0]))
hr.describe()


# ### Holy Moly! Lots of duplicates in here.
# ![alt text](http://theinfosphere.org/images/6/6a/Bender_duplicates.png "Yay")
# From looking at the dtypes I know there must be some issues with HourlyRate, JobSatisfaction, MonthlyIncome, and PercentSalaryHike because they all should be floats 

# In[42]:


hr['JobSatisfaction'].value_counts()


# In[43]:


hr['JobSatisfaction'].unique()


# In[44]:


# Half the rows in JobSatisfaction seem to be strings. 
# It's the same for the other columns. Let's cast them to floats.
cols = ['JobSatisfaction', 'HourlyRate', 'MonthlyIncome', 'PercentSalaryHike']
hr[cols] = hr[cols].applymap(np.float64)


# In[45]:


# I know from looking in Excel that certain fields are useless so lets get rid of them
hr = hr.drop(['EmployeeCount', 'Over18', "StandardHours", "EmployeeNumber"], 1)


# In[46]:


# Lets try find some more funky rows
for col in hr:
    print(col)
    print(hr[col].unique())


# In[47]:


hr.to_csv("hr-clean.csv")


# ## YAY ITS PRETTY MUCH ALL CLEAN
# ![alt text](https://s1.qwant.com/thumbr/0x0/9/e/a72e8a67a8f50c338ab166942d1eac/b_1_q_0_p_0.jpg?u=https%3A%2F%2Fwww.yay.com%2Fstatic%2Fimg%2Fabout-01.png&q=0&b=1&p=0&a=1 "Yay")
# 
# That was too easy. It's almost as if this dataset isn't real O_o.
# 
# ## Finding correlations
# 
# Before I move any further I'd like to check over the data broadly to see if anything in the data catches my eye. Plotting a correlation heatmap, which is a measure of the strength of a relationship between two variables, can tell me quite a lot about the data in a short amount of time. 

# In[48]:


# Subset the dataset into all the numerical values
numeric_hr = hr.select_dtypes(include=[np.number])

# Compete the correlation matrix
corr = numeric_hr._get_numeric_data().corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, center=0.0,
                      vmax = 1, square=True, linewidths=.5, ax=ax)
plt.savefig('corr-heat.png')
plt.show()


# ## Interesting!
# 
# Most of the values appear to be weakly correlated with each other. But there's lots of insights here to be had.
# 
# Takeaways
# - Perhaps unsurprisingly TotalWorkingYears is highly correlated to Job Level (i.e., the longer you work the higher job level you achieve)
# - HourlyRate, DailyRate, and MonthlyRate are completely uncorrelated with each other which makes no sense. As MonthlyIncome is highly correlated to Job Level i'm inclined to keep using that over any of these. 
# - Age is correlated JobLevel and Education (i.e., the older you are the more educated and successful you are)
# - Work life Balance correlates to pretty much none of the numeric values 
# - Number of companies you've worked at is weakly correlated with the time you've spent at the company (might indicate you're likely the leave)
# - If your performance rating is high you'll get a bigger raise! 
# 
# Now lets look at some numbers:

# In[49]:


# Lets drop the rates from the numerics dataframe
numeric_hr = numeric_hr.drop(["HourlyRate","DailyRate", "MonthlyRate"], 1)


# In[50]:


hr.describe()


# Some nice summary statistics to be had here:
# 
# - Mean age of TechCo employees is 37
# - Most people get a promotion in 2-5 years
# - Average time employed at TechCo is 7 years
# - No one has a performance rating under 3
# - Lots of training times at TechCo: seems like most people get training 2-3 times a year
# 
# So everyone is performing well, getting training, and is staying at their jobs for quite a while. Who does leave then?
# 
# ## Part 2: Who leaves TechCo?
# 
# There are lots of ways to approach this, I could spend a long time digging into this thoroughly, looking at the impact of travel, the department you're in, gender, income, & work life balance. Given this is already rather long I'm only going to focus on age, the time spent at TechCo, and whether you worked overtime or not.

# In[51]:


print(hr.Attrition.value_counts())

# Easier to join all leaver together for my analyses considering there are very few terminations
hr['Attrition'] = hr['Attrition'].replace("Termination", "Voluntary Resignation")
hr['Attrition'] = hr['Attrition'].replace("Voluntary Resignation", "Former Employees")
hr['Attrition'] = hr['Attrition'].replace("Current employee", "Current Employees")

plt.figure(figsize=(12,8))
plt.title('Number of Former/Current Employees at TechCo')
sns.countplot(x="Attrition", data=hr)
hr['Attrition'].value_counts()/hr['Attrition'].count()*100


# It appears that barely anyone has left! Considering we have people who have worked at TechCo for 40 years I would expect there to be more former than current employees. Perhaps TechCo has suddenly had a rapid expansion. 
# 
# Lets see whether age is a factor.

# In[52]:


temp3 = pd.crosstab([hr.Department,hr.Gender,hr.MaritalStatus,hr.WorkLifeBalance], hr['Attrition'])
print(temp3)
income_pivot = hr.pivot_table(values=["MonthlyIncome"], index=["Gender","MaritalStatus","WorkLifeBalance"], aggfunc=[np.mean, np.std])
print(income_pivot)


# In[53]:


# Plot the distribution of age by Attrition Factor
plt.figure(figsize=(12,8))
plt.title('Age distribution of Employees at Telco by Attrition')
sns.distplot(hr.Age[hr.Attrition == 'Former Employees'], bins = np.linspace(1,70,35))
sns.distplot(hr.Age[hr.Attrition == 'Current Employees'], bins = np.linspace(1,70,35))
#sns.distplot(hr.Age[hr.Attrition == 'Termination'], bins = np.linspace(0,70,35))
plt.legend(['Former Emploees','Current Employees'])


# Looks like most people leave TechCo in their early 30's. Maybe TechCorp has trouble retaining young employees? How long do employees tend to stick around at TechCo for? When do the majority of employees leave? In their first year perhaps?

# In[54]:


# Plot the distribution of Years at Company by Attrition
plt.figure(figsize=(12,8))
plt.title('Distribution of the Number of Years Employees Spend at Telco by Attrition')
#sns.distplot(hr.YearsAtCompany, bins = np.linspace(0,40,40))
sns.distplot(hr.YearsAtCompany[hr.Attrition == 'Former Employees'], bins = np.linspace(0,40,40))
sns.distplot(hr.YearsAtCompany[hr.Attrition == 'Current Employees'], bins = np.linspace(0,40,40))
plt.legend(['Former Emploees','Current Employees'])


# Two takeaways:
# 
# - The highest attrition rate occurs in the first year of the job. Over 20% of all employees who left did so in their first year. 
# - The vast majority of the workforce has been at TechCo for under 10 years. Perhaps only the best and brightest get the privledge to continue on after 10 years.
# 
# I suspect if you're over worked you should be more likely to leave. Lets see if that's the case.

# In[55]:


# Plot out the counts of OverTime
sns.factorplot("Attrition", col="OverTime", data=hr, kind="count", col_wrap=2, size=5)
plt.subplots_adjust(top=.85)
plt.suptitle('Attrition Counts by whether an Employee worked Over Time')

# Chi squared test of independence
# H0: Overtime and Attrition are independent of each other
res_1 = hr.OverTime[hr.Attrition == 'Current Employees'].value_counts()
res_2 = hr.OverTime[hr.Attrition == 'Former Employees'].value_counts()
obs = np.array([res_1, res_2])
stats.chi2_contingency(obs)


# By eyeballing this plot I can see the employees who work over time leave at a higher rate than those who do not. But since I was unsure I did a Chi Squared Test of Independence to see whether this is true. Turns out with a p-value of 4.2^-85 (p < 0.05), I can confidently say over time is related to attrition.
# 
# Given this assoication and that of age: perhaps over worked employees are more likely to be under 30?

# In[56]:


# Plot the distribution of Years at Company by Attrition
plt.figure(figsize=(12,8))
plt.title('Age Distribution of Employees who have worked Over Time')
#sns.distplot(hr.YearsAtCompany, bins = np.linspace(0,40,40))
sns.distplot(hr.Age[hr.OverTime == 'Yes'], bins = np.linspace(0,70,35))


# It turns out TechCo doesn't discriminate on who works overtime. It appears to be distributed equally throughout all ages. I feel sorry for the 60 year olds putting in the hard yards. 
# 
# Now let's look at the hard hitting question:
# 
# ## Part 3: Gender Equality at TechCo

# In[57]:


plt.figure(figsize=(12,8))
sns.countplot(x="Gender", data=hr)
# Proportion of males
plt.title('Frequency of Gender at TechCo')
hr['Gender'].value_counts().Male/hr['Gender'].count()*100


# Clearly there's a gender imbalance (58% Male/42% Female) at TechCo. Perhaps there's a gender imbalance in attrition too?

# In[58]:


# First lets cast these string columns into categories
cats = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
for col in cats:
    hr[col] = hr[col].astype('category')

group_hr = hr.groupby(cats)

# Plot the distribution of females in this workplace
plt.figure(figsize=(12,8))
#sns.countplot(x="Gender", hue="Attrition", data=hr[hr['Attrition'].isin(['Voluntary Resignation', 'Termination'])])

attrition_counts = (hr.groupby(['Gender'])['Attrition']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('Attrition'))
plt.title('Percent Distribution of Gender by Attrition at TechCo')
sns.barplot(x="Gender", y="percentage", hue="Attrition", data=attrition_counts)

# It's the same, looks suss
print(attrition_counts)

# Nope my code is alright


# It's rather odd that the proportion of attrition rates is almost exactly the same for men and women. Let's see if women travel more?

# In[59]:


# Plot the distribution of females in this workplace
plt.figure(figsize=(12,8))
#sns.countplot(x="Gender", hue="Attrition", data=hr[hr['Attrition'].isin(['Voluntary Resignation', 'Termination'])])

attrition_counts = (hr.groupby(['Gender'])['BusinessTravel']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('BusinessTravel'))
plt.title('Percent Distribution of Gender by Business Travel Frequency')
sns.barplot(x="Gender", y="percentage", hue="BusinessTravel", data=attrition_counts) 

#sns.countplot(x="Gender", data=hr, palette="Greens_d")


# Same again. What about across departments?

# In[60]:


# Plot the distribution of females in this workplace
plt.figure(figsize=(12,8))
#sns.countplot(x="Gender", hue="Attrition", data=hr[hr['Attrition'].isin(['Voluntary Resignation', 'Termination'])])

attrition_counts = (hr.groupby(['Gender'])['Department']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('Department'))
plt.title('Distribution of Gender across Departments')
sns.barplot(x="Gender", y="percentage", hue="Department", data=attrition_counts) 


# Same thing! Freaky if it was real. It's likely gender was assigned randomly over the whole dataset. 

# In[61]:


# Plot the distribution of age by gender
plt.figure(figsize=(12,8))
sns.distplot(hr.Age[hr.Gender == 'Male'], bins = np.linspace(0,70,35))
sns.distplot(hr.Age[hr.Gender == 'Female'], bins = np.linspace(0,70,35))
plt.title('Distribution of Age by Gender')
plt.legend(['Males','Females'])


# The distribution of ages at this company is mostly equal, but 36-38 year old middle-aged males (of which there are roughly 670) outnumber the rest of the age brackets.
# 
# Do males earn more at TechCo than females?

# In[62]:


# It appears women are under-represented at this company. Let's see if they get paid less.
plt.figure(figsize=(15,10))
plt.title('Average Monthly Income by Gender')
sns.barplot(x="Gender", y="MonthlyIncome", data=hr)

# T-Test
cat1 = hr[hr['Gender']=='Male']
cat2 = hr[hr['Gender']=='Female']

print(stats.ttest_ind(cat1['MonthlyIncome'], cat2['MonthlyIncome']))


# Females have a slightly higher monthly income than males. A t-test confirms that this is the case.
# 
# Let's look at the distribution

# In[63]:


plt.figure(figsize=(15,10))
plt.title('Distribution of Monthly Income by Gender')
sns.distplot(hr.MonthlyIncome[hr.Gender == 'Male'], bins = np.linspace(0,20000,20))
sns.distplot(hr.MonthlyIncome[hr.Gender == 'Female'], bins = np.linspace(0,20000,20))
plt.legend(['Males','Females'])


# The most common monthly wage comes in the 2-3K mark. There are noticeably more males in this wage bracket than females.
# 
# Does martial status have an effect? Married people might have a more stable personal life which leads to better performance in their jobs.

# In[64]:


# What about all the single ladies?
plt.figure(figsize=(15,10))
plt.title('Average Monthly Income by Gender and Maritial Status')
sns.barplot(x="MaritalStatus", y="MonthlyIncome", hue="Gender", data=hr)


# It appears that while there are less females in the workplace, they earn more than the males, but leave at the same rates. Married people have clear advantage over single people but it might be correlated simply to age.
# 
# Let's do a rough check on the age distributions to be sure.

# In[65]:


# Age by Gender and Martial Status 
plt.figure(figsize=(15,15))
plt.title('Average Monthly Income by Gender and Maritial Status')
sns.boxplot(x="MaritalStatus", y="Age", hue="Gender", data=hr, width=.5) 


# Ha! There's no real difference in the distribution of age over these categories. Lets split up the entire dataset based on age brackets.

# In[66]:


# Trying to get a binned distribution in of Age by MonthlyIncome in Seaborn
plt.figure(figsize=(15,15))
bins=[18, 25, 35, 50, 70]
out = hr.groupby(pd.cut(hr['Age'], bins=bins, include_lowest=True)).aggregate(np.mean)
print(out.head())
#out[['Age']] = out[['Age']].applymap(str)
out['Age Bracket'] = ['18-25', '26-35', '36-50', '51-70']

# Fixed X-axis labels currently looking awful!
plt.title('Average Monthly Income by Age Bracket')
sns.barplot('Age Bracket', 'MonthlyIncome', data=out, palette="muted")
out.head()


# We can see that wage seems to increase linearly as you move through the age brackets. This could be correlated to how long you've been working at TechCo.

# In[67]:


# Trying to get a binned distribution in of Age by MonthlyIncome in Seaborn
plt.figure(figsize=(15,15))
bins=[0, 10, 20, 30, 40]
out = hr.groupby(pd.cut(hr['YearsAtCompany'], bins=bins, include_lowest=True)).aggregate(np.mean)
out[['YearsAtCompany']] = out[['YearsAtCompany']].applymap(str)
out['Years at Company Bracket'] = ['0-10', '11-20', '21-30', '31-40']

# Fixed X-axis labels currently looking awful!
plt.title('Average Monthly Income by Years Worked at TechCo')
sns.barplot('Years at Company Bracket', 'MonthlyIncome', data=out, palette="muted")
out.head()


# In[68]:


plt.figure(figsize=(15,15))
sns.lmplot("YearsAtCompany", "MonthlyIncome", data=hr, size=10) 


# We can see that there's plenty of employees at TechCo earning over 10K a month regardless of how long they've been at the company. But after the 10 year mark there's an attrition of lower paid employees raising the average through the age brackets.

# # MODELS!
# 
# ## Part 4: Predicting attrition at TechCo
# 
# ![alt text](https://static01.nyt.com/images/2016/12/18/magazine/18ai5/18mag-18ai-t_CA1-master675.jpg "the man")
# 
# Not really going to do much here. R is a lot better for modelling with such a signifcant class imbalance. Lets extract some feature importances then see where a PCA leads us.

# In[69]:


hr["Attrition"].value_counts() # Large class imbalance


# In[70]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper, gen_features, cross_val_score

# Encode the categorical variables so that scikit-learn can read them
cat_cols = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Employee Source']
feature_def = gen_features(
    columns= cat_cols,
    classes=[LabelEncoder]
)
mapper = DataFrameMapper(feature_def)
hr[cat_cols] = mapper.fit_transform(hr)
hr.head()


# In[71]:


# Build a forest to predict attrition and compute the feature importances
rf = RandomForestClassifier(class_weight="balanced", n_estimators=500) 
rf.fit(hr.drop(['Attrition'],axis=1), hr.Attrition)
importances = rf.feature_importances_
names = hr.columns
importances, names = zip(*sorted(zip(importances, names)))

# Lets plot this
plt.figure(figsize=(12,8))
plt.barh(range(len(names)), importances, align = 'center')
plt.yticks(range(len(names)), names)
plt.xlabel('Importance of features')
plt.ylabel('Features')
plt.title('Importance of each feature')
plt.show()


# In[72]:


# Make predictions using 10-K-Fold-CV

# Baseline:
print((hr.Attrition.value_counts()/(hr.shape[0]))*100)

# Accuracy
scores = cross_val_score(rf, hr.drop(['Attrition'],axis=1), hr.Attrition, cv=10, scoring='accuracy')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# ROC
scores = cross_val_score(rf, hr.drop(['Attrition'],axis=1), hr.Attrition, cv=10, scoring='roc_auc')
print(scores)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# The feature importances look pretty good on the surface. In terms of attrition it appears age is the most important feature, followed by whether you travel, then what department you're in. However, my model doesn't look like it's performing any better than random chance, given the mean accuracy is 84% and 83.6% of my sample is made up of current employees. How important can these features be if they're making a model that's predicting nothing? More proof to the axiom that all models are bad but only some are useful.
# 
# As a note, in scikit-learn, the feature importances are calculated as the "gini importance" or "mean decrease impurity" and is defined as the total decrease in node impurity (weighted by the probability of reaching that node (which is approximated by the proportion of samples reaching that node)) averaged over all trees of the ensemble. Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. 
# 
# Given there are a fair few features in the data why don't I do a PCA to find if constructing some new variables in my dataset can explain some variation in the dataset well.

# In[73]:


from sklearn.decomposition import PCA

# Normalise PCA as it can have a large effect on the result then fit
std_clf = make_pipeline(StandardScaler(), PCA())
std_clf.fit(hr.drop(['Attrition'], axis=1))
existing_2d = std_clf.transform(hr.drop(['Attrition'],axis=1))


# In[74]:


# Print out the ratio of explained variance for each principal component
pca_std = std_clf.named_steps['pca']
print(pca_std.explained_variance_ratio_.cumsum())


# In[75]:


# Convert result to dataframe, add the labels
existing_hr_2d = pd.DataFrame(existing_2d)
existing_hr_2d = pd.concat([existing_hr_2d, hr[['Attrition']]], axis = 1)
existing_hr_2d.columns = ['PC' + str(i) for i in range(1, existing_hr_2d.shape[1])] + ['Attrition']
di = {0.0: "Current Employee", 1.0: "Former Employee"}
existing_hr_2d = existing_hr_2d.replace({"Attrition":di})
#ax = existing_hr_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8))

# Plot with Seaborn
plt.figure(figsize=(16,8))
sns.lmplot("PC1", "PC2", data=existing_hr_2d, hue='Attrition', fit_reg=False, size=15)


# This PCA doesn't tell me much other than there's no clear clustering for attirition along the two directions of most variation. There's a cluster of values to the right which tells me that there are more values that are similiar to each other than dissimilar which I already knew. Given PC1 describes only 13.8% of the variation in my dataset and PC2 describes 6.2% I can say it's going to be tough for any ML algorithm to create any decent predictions.
# 
# # THANK YOU VERY MUCH
# 
# This brings us to the end of this little EDA. I appreciate the time you took to read this. If you have any feedback I am more than welcome to receive it.

# ### Retrospective
# 
# With more time:
# 
# - Try creating graphs in Bokeh or Plotly
# - Use R instead :D
# - Conduct statistical tests to see if these distributions are significantly different from each other
# - Radar plots would be nice
# - Linear Regression using categories
