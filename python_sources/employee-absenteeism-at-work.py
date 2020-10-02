#!/usr/bin/env python
# coding: utf-8

# # **Classification based predictive model for absenteeism of employees at work**

# * Employees with low performance cause a vital lose for organizations and the absenteeism consider to be one of the factors that affect performance So, understanding the causes of absenteeism may power the organization with a competitive advantages tool and open the area of research for computer and human resources fields. 
# * The aim of this analysis is to discover the factors and causes of employees absence using computerized technologies.
# * Original dataset is available at https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work
# 
# ![Employees](http://bbrc.in/bbrc/wp-content/uploads/2019/02/fig1_opt-7.jpeg)

# # 1. Import and Clean Data
# >The first step is to import and clean the data for analysis. The process of cleaning might vary based on the quality and size of the dataset.

# In[ ]:


pip install treeinterpreter


# > Allows decomposing each prediction into `bias` and `feature contribution` components as described in http://blog.datadive.net/interpreting-random-forests/

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
#from statsmodels.graphics import tsaplots
#import statsmodels.api as sm
import seaborn as sns
import numpy as np
#import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error as MSE
#from scipy.stats import spearmanr, pearsonr
from sklearn import metrics
from sklearn.metrics import classification_report
from scipy.cluster import hierarchy as hc
import scipy
from treeinterpreter import treeinterpreter as ti
from collections import Counter


# In[ ]:


plt.style.use('fivethirtyeight')


# In[ ]:


df =  pd.read_excel('../input/Absenteeism_at_work_new.xls')


# In[ ]:


df.info()


# > The above results shows that this dataset has just 740 rows with the size of 122 KB. One of the column "Work load Average/day" has forward slash and we are replacing to avoid unnecessary errors in future.

# In[ ]:


df.columns = df.columns.str.replace('/', 'per').str.strip()


# > Let us check the data quickly to understand the contents. ID is the unique employee id in the dataset. The below command returns the count of rows for each ID from the dataset.

# In[ ]:


df.groupby('ID')[['ID']].count().head()


# > The above results shows that this dataset is not a regular attendence dataset. This dataset only has absent hours of the employees for the period.
# 
# > First step in the cleaning of the dataset is to change the appropriate data types of the columns.

# In[ ]:


len(df.columns)


# In[ ]:


df.shape


# ## 1.1 Manually setting up the data types of certain variables for EDA purpose

# In[ ]:


df.head()


# In[ ]:


df['Social drinker'] = df['Social drinker'].astype('bool')
df['Social smoker'] = df['Social smoker'].astype('bool')
df['Disciplinary failure'] = df['Disciplinary failure'].astype('bool')
df['Seasons'] = df['Seasons'].astype('category')
df['Education'] = df['Education'].astype('category')
df['Day of the week'] = df['Day of the week'].astype('category')
df['Month of absence'] = df['Month of absence'].astype('category')
df['Reason for absence'] = df['Reason for absence'].astype('category')


# In[ ]:


df.info()


# > After changing the datatype of few columns you can notice the size of the dataset reduced from 122KB to 84KB.
# The next step is to check whether there is any missing values in the dataset.

# In[ ]:


df.isnull().sum()


# > Looks like there is no missing value in the dataset. Now we have to check the valid values in the dataset. There is no specific logic to this process. If there is any datetime value, need to check invalid date time values. If there are numeric values, need to check the outliers. Check the distribution of values etc.,

# In[ ]:


df['Absenteeism time in hours'].describe


# In[ ]:


df[df['Month of absence']==0]


# > Now let us check the invalid values in the target variable.

# In[ ]:


df[df['Reason for absence']==27][['Absenteeism time in hours']].mean()


# > We can see the mean value for Reason 27 and let us update Absenteeism hours as 3 for this row.

# In[ ]:


df.loc[(df['Reason for absence']==27) & (df['Absenteeism time in hours']==0),'Absenteeism time in hours']=3


# >After updating the value for the above row, now we can update Absenteeism in hours as 8 for all the rows with Disciplinary failure.

# In[ ]:


df.loc[(df['Absenteeism time in hours']==0),'Absenteeism time in hours']=8


# In[ ]:


len(df[df['Absenteeism time in hours']==8])


# >We performed some data cleaning in excel which are not specified here
# 
# >But if you need the cleaned dataset use this one =>=>=> https://www.kaggle.com/miracle9to9/absenteeism-dataset

# >After checking the zero values in Absenteeism in hours column, let us get ready for Exploratory Data Analysis. It's better to add additional columns for visualization as this dataset has only numeric values.

# In[ ]:


season_mapping = {1:'Summer', 2:'Autumn', 3:'Winter', 4:'Spring'}
df['season_name'] = df.Seasons.map(season_mapping)
df['season_name'] = df['season_name'].astype('category')
df.drop_duplicates(['Seasons', 'season_name'])[['Seasons','season_name']]


# >The above will have Season Name along with Season and the below command will have Month names.

# In[ ]:


import calendar
df['month_name'] =  df['Month of absence'].apply(lambda x: calendar.month_abbr[x])


# In[ ]:


reason_mapping = {
    0: 'Unknown',
    1: 'Certain infectious and parasitic diseases',
    2: 'Neoplasms',
    3: 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
    4: 'Endocrine, nutritional and metabolic diseases',
    5: 'Mental and behavioural disorders',
    6: 'Diseases of the nervous system',
    7: 'Diseases of the eye and adnexa',
    8: 'Diseases of the ear and mastoid process',
    9: 'Diseases of the circulatory system',
    10: 'Diseases of the respiratory system',
    11: 'Diseases of the digestive system',
    12: 'Diseases of the skin and subcutaneous tissue',
    13: 'Diseases of the musculoskeletal system and connective tissue',
    14: 'Diseases of the genitourinary system',
    15: 'Pregnancy, childbirth and the puerperium',
    16: 'Certain conditions originating in the perinatal period',
    17: 'Congenital malformations, deformations and chromosomal abnormalities',
    18: 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
    19: 'Injury, poisoning and certain other consequences of external causes',
    20: 'External causes of morbidity and mortality',
    21: 'Factors influencing health status and contact with health services',
    22: 'Patient follow-up',
    23: 'Medical consultation',
    24: 'Blood donation',
    25: 'Laboratory examination',
    26: 'Unjustified absence',
    27: 'Physiotherapy',
    28: 'Dental consultation'
}
df['reason_text'] = df['Reason for absence'].map(reason_mapping)


# > The above reason is available in the UCI data description document and the below will update the education.

# In[ ]:


education_mapping = {
    1: 'High School',
    2: 'Graduate',
    3: 'Post Graduate',
    4: 'Master & Doctor'
}
education_list = {'High School', 'Graduate', 'Post Graduate', 'Master & Doctor'}
df['Education_detail'] = df['Education'].map(education_mapping)
#df['Education_detail'] = df['Education_detail'].astype('category')
category_education = pd.api.types.CategoricalDtype(categories=education_list, ordered=True)
df['Education_detail'] = df['Education_detail'].astype(category_education)


# >After adding new columns for the reference columns, let us quickly check the first 5 rows from the dataset.

# In[ ]:


df.head()


# # 2. Exploratory Data Analysis

# ### 2.1 Agewise Employee Count

# >Let us check the employee counts by Age. Notice one minor hack to count the unique IDs. As the employee count are almost similar, we are not able to decipher much.

# In[ ]:


age_count = df.groupby(['Age']).agg({'ID': pd.Series.nunique})
ax = age_count.plot(kind='bar', figsize=(10,4), legend=False, color="navajowhite",edgecolor='darkred')
for i, v in enumerate(age_count.values):
    ax.text(i-.24, v +0.2, str(v[0]), color='firebrick')
ax.set_xlabel('Age')
ax.set_ylabel('Count of employees')
ax.set_title('Agewise count of employees')
plt.show()


# ### 2.2 Educationwise Employee Count

# > The below graph shows that High School educated employees are higher than the rest.

# In[ ]:


edu_count = df.groupby(['Education_detail']).agg({'ID': pd.Series.nunique})
ax = edu_count.plot(kind='bar', figsize=(8,5), legend=False, color="navajowhite",edgecolor='darkred')
for i, v in enumerate(edu_count.values):
    ax.text(i-.065, v + 0.8, str(v[0]), color='firebrick')
ax.set_xlabel('Education')
ax.set_ylabel('Count')
ax.set_title('Educationwise count of employees')
plt.show()


# ### 2.3 Average work load by Age

# >The work load seems to be same irrespective of the age.

# In[ ]:


age_work_sum = df.groupby('Age', as_index=False)[['Work load Averageperday']].mean()
ax = age_work_sum.plot(kind='bar', x='Age', figsize=(8,6), legend=False, color="navajowhite",edgecolor='darkred')
ax.set_ylabel('Work load average per day')
ax.set_title('Average work load per day by age')
plt.show()


# ### 2.4 Average Absenteeism hours by Age

# >Absenteeism seems to be same across Age except for one age.

# In[ ]:


age_abs = df.groupby('Age')[['Absenteeism time in hours']].mean()
ax = age_abs.plot(kind='bar', figsize=(8,6), legend=False, color="navajowhite",edgecolor='darkred')
for i, v in enumerate(age_abs.values):
    ax.text(i-.25, v + 0.2, str(np.int(np.round(v))), color='firebrick')
ax.set_ylabel('Absenteeism time in hours')
ax.set_title('Average Absenteeism time in hours by age')
plt.show()


# ### 2.5 Average Absenteeism hours by Distance to work

# >The hypothesis i have was that if the distance to work increase the absenteeism hours will increase. But the below graph nullifies my hypothesis.

# In[ ]:


dis_abs = df.groupby('Distance from Residence to Work')[['Absenteeism time in hours']].mean()
ax = dis_abs.plot(kind='bar', figsize=(8,6), legend=False, color="navajowhite",edgecolor='darkred')
for i, v in enumerate(dis_abs.values):
    ax.text(i-.25, v + 0.1, str(np.int(np.round(v))), color='firebrick')
ax.set_xlabel('Distance from Residence to Work (km)')
ax.set_ylabel('Absenteeism time in hours')
ax.set_title('Average Absenteeism time in hours by distance')
plt.show()


# ### 2.6 Average distance to work by Age

# >Another hypothesis I had was that higher age employees might stay closer to the office. It might be true till the Age 33, but the other values are not significant to compare.

# In[ ]:


age_dis = df.groupby('Age')[['Distance from Residence to Work']].mean()

ax = age_dis.plot(kind='bar', figsize=(8,6), legend=False, color="navajowhite",edgecolor='darkred')

for i, v in enumerate(age_dis.values):
    ax.text(i-.28, v + 1, str(np.int(np.round(v))), color='firebrick')
ax.set_ylabel('Distance from Residence to Work')
ax.set_title('Average Distance from Residence to Work by age')
plt.show()


# ### 2.7 Average Transportation expense by Distance

# >The transportation expense is not increasing by distance but we don't have transport mode, so this is not helping us.

# In[ ]:


dis_exp = df.groupby('Distance from Residence to Work')[['Transportation expense']].mean()
ax = dis_exp.plot(kind='bar', figsize=(10,6), legend=False, color="navajowhite",edgecolor='darkred')
for i, v in enumerate(dis_exp.values):
    ax.text(i-.45, v + 8, str(np.int(np.round(v))), color='firebrick')
ax.set_ylabel('Transportation expense')
ax.set_title('Average Transportation expense by distance to work')
plt.show()


# ### 2.8 Pet & Son counts by Age

# >The below graph shows that the employees who has Son are mostly having a pet. This is interesting.

# In[ ]:


ax = df.groupby('Age')['Son', 'Pet'].sum().plot(figsize=(8,6))
ax.set_ylabel('Count')
ax.set_title('Count of Pet & Son by Age')
plt.show()


# ### 2.9 Smoker & Drinker Stats

# >How much percentage of Social Drinkers are Social Smokers also?

# In[ ]:


# % of Social drinker those are smokers
emp_social = df.drop_duplicates(['ID', 'Social drinker', 'Social smoker'])[['ID', 'Social drinker', 'Social smoker']]
emp_social[emp_social['Social drinker']==True]['Social smoker'].mean()


# >How much percentage of Social Smokers are Social Drinkers also?

# In[ ]:


# % of Social smokers are drinkers
emp_social[emp_social['Social smoker']==True]['Social drinker'].mean()


# ### 2.10 Absenteeism by Social habits

# >Looks like 60% of Social drinkers are absent and interestingly 32% of Non-smoker & Non-drinker are also absent.

# In[ ]:


drink_sum = df[(df['Social drinker'] == True) & (df['Social smoker']==False)]['Absenteeism time in hours'].sum()
smok_sum = df[(df['Social drinker'] == False) & (df['Social smoker']==True)]['Absenteeism time in hours'].sum()
drink_smok_sum = df[(df['Social drinker'] == True) & (df['Social smoker']==True)]['Absenteeism time in hours'].sum()
abs_sum = df[(df['Social drinker'] == False) & (df['Social smoker']==False)]['Absenteeism time in hours'].sum()
absen = [drink_sum, smok_sum, drink_smok_sum, abs_sum]
pie_labels = ['drinker', 'smoker', 'drinker & smoker', 'No drinker/smoker']
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

wedges, texts, autotexts = ax.pie(absen, autopct=lambda pct: func(pct, absen), textprops=dict(color='w'))
ax.legend(wedges, pie_labels, title='Social Drinkers/Smokers', loc='right', bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=10, weight="bold")
ax.set_title('Absenteeism by Social Drinkers/Smokers')
#plt.pie(absen, labels=pie_labels)
plt.show()


# ### 2.11 Absenteeism by Reason

# >The below table shows the highest Abseenteeism hours to lowest by Reason. The data is from courier company and the employees needs to deliver the packages and the Top 2 reason for absence shows that.
# There is one hack to hide the index column.

# In[ ]:


reason_abs = df.groupby('reason_text', as_index=False)['Absenteeism time in hours'].sum()
with pd.option_context('display.max_colwidth', -1):
    display(reason_abs.sort_values('Absenteeism time in hours', ascending=False).style.hide_index())


# ### 2.12 Absenteeism hours by Seasons

# >Looks like winter has highest Absenteeism.

# In[ ]:


season_abs = df.groupby('season_name')['Absenteeism time in hours'].sum()
ax = season_abs.plot(kind='bar', figsize=(8,5), legend=False, color="navajowhite",edgecolor='darkred')
for i, v in enumerate(season_abs.values):
    ax.text(i-.12, v + 22, str(np.int(np.round(v))), color='firebrick')
ax.set_xlabel('Seasons')
ax.set_ylabel('Sum of Absenteeism hours')
ax.set_title('Sum of Absenteeism hours by Seasons')
plt.show()


# ### 2.13 Absenteeism hours by Month

# >Getting to detail from Seasons to Month, Looks like March & July has the highest number of Absenteeism hours.

# In[ ]:


month_abs = df.groupby('month_name')['Absenteeism time in hours'].sum()
ax = month_abs.plot(kind='bar', figsize=(8,6), legend=False, color="navajowhite",edgecolor='darkred')
for i, v in enumerate(month_abs.values):
    ax.text(i-0.3, v + 12, str(np.int(np.round(v))), color='firebrick')
ax.set_xlabel('Month')
ax.set_ylabel('Sum of Absenteeism hours')
ax.set_title('Sum of Absenteeism hours by Month')
plt.show()


# ### 2.14 March month Absenteeism hours by Reason

# >Let us check the March month reason.

# In[ ]:


mar_abs = df[df['Month of absence']==3].groupby('reason_text', as_index=False)['Absenteeism time in hours'].sum()
with pd.option_context('display.max_colwidth', -1):
    display(mar_abs.sort_values('Absenteeism time in hours', ascending=False).style.hide_index())


# ### 2.15 July month Absenteeism hours by Reason

# >Looks like the March month & July month reasons are not matching.

# In[ ]:


jul_abs = df[df['Month of absence']==7].groupby('reason_text', as_index=False)['Absenteeism time in hours'].sum()
with pd.option_context('display.max_colwidth', -1):
    display(jul_abs.sort_values('Absenteeism time in hours', ascending=False).style.hide_index())


# # 3. Train and Test Split

# In[ ]:


df.head()


# >After using the FeatureImportance attribute of RandomForest, We removed four columns from df_features that are Season, Social_smoker, Social Drinker, Education 
X = df_features
y = df['Absenteeism time in hours']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# >This is the challenge in the small dataset. We changed the target variable as Classification
# 
# >The target variable is Absenteeism in hours. Let us change to classification, so the multiclass variable is '=<4' as '0' and '>4' as '1'. This is just random assumption that employee absence can be half day, full day or more......

# In[ ]:


bins = [25, 35, 45, 55, np.inf]
names = [25, 35, 45, 55]
df['age_range'] = pd.cut(df['Age'], bins, labels=names)


# In[ ]:


abs_bins = [0, 4, np.inf]
abs_names = ['0', '1']
df['abs_range'] = pd.cut(df['Absenteeism time in hours'], abs_bins, labels=abs_names)


# In[ ]:


df_features = df[['Reason for absence', 'Month of absence', 'Day of the week', 'Transportation expense', 
        'Distance from Residence to Work', 'Service time', 'age_range', 'Work load Averageperday', 'Hit target',
        'Disciplinary failure', 'Son', 'Pet', 'Weight', 'Height', 'Body mass index']]


# In[ ]:


df_features.head()


# >After using the FeatureImportance atttribute of RF Classifier we removed four columns from our df_features they are season, social_smoker, social_drinker, eduaction to increase accuracy.

# # 3.3 ML Model
# >Let us split the data

# In[ ]:


X = df_features
y = df['abs_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# >Gaussian Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
y_pred_nb = gnb.predict(X_test) 
  
print(classification_report(y_test, y_pred_nb))


# >AdaBoost Classifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=20)
clf.fit(X_train, y_train)

y_pred_adb = clf.predict(X_test)

print(classification_report(y_test, y_pred_adb))


# >RandomForest Classifier

# In[ ]:


model = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
model.fit(X_train,y_train)

y_pred_rf = model.predict(X_test)

print(classification_report(y_test, y_pred_rf))


# ### ***Feature Importance (Important of features that recognised by the RF model)***
# >This is one way of interpreting the Random Forest Model and we can see the contributions from the features. This is at high level we can understand the importance of features.

# In[ ]:


ax = (pd.Series(model.feature_importances_, index=X.columns)
   .nlargest(19)
   .plot(kind='barh', figsize=(8,6), color='navajowhite',edgecolor='darkred'))
plt.show()


# # 3.4 Hyper Parameter tuning with Cross Validation

# >Hyper parameter tuning is a process is to change the parameters of the model and identify which parameter gives us the better accuracy. The cross validation in Grid & Random Search helps us to use Stratified K-fold to create validation set within trained data. The CV value 5 or above uses Stratified K-fold.

# ### **RandomSearchCV**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=100, stop = 2000, num = 20)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_sample_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf': min_sample_leaf,
               'bootstrap' : bootstrap
              }

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv=5, error_score= np.nan, 
                               verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)


# In[ ]:


# The below will provide us the best parameters from the Random Search CV.

rf_random.best_params_


# >Let us use the above model to predict & evaluate.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_random = rf_random.best_estimator_
model_random.fit(X_train, y_train)
predictions_random = model_random.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, predictions_random))


# In[ ]:


print(classification_report(y_test, predictions_random))


# In[ ]:


metrics.f1_score(y_test, predictions_random, average='micro')


# > Let us check the train & test data for each classification.

# In[ ]:


from collections import Counter


# In[ ]:


Counter(y_train)


# In[ ]:


Counter(y_test)


# >The above value shows that there are less values in the test/validation set.
# 
# >The random search gives us the range of the values and using that We can do the Grid Search to tune the model with exact parameters.

# ### **GridSearch**

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap' : [True],
    'max_depth' : [8, 10, 12, 14],
    'max_features' : ['sqrt'],
    'min_samples_leaf' : [1, 3, 4],
    'min_samples_split' : [7, 10, 12],
    'n_estimators' : [250, 275, 300, 325]
    
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv=5, n_jobs = -1, error_score=np.nan, verbose = 2)

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model_grid = grid_search.best_estimator_
model_grid.fit(X_train, y_train)
predictions_grid = model_grid.predict(X_test)


# In[ ]:


print(classification_report(y_test, predictions_grid))


# # 4. Interpreting the Model

# >The most common problem we have is to justify the predictions. Why the model has predicted in a way? otherwise it is alwo known as blackbox.
# 
# >This challenge can be solved easily in the Random Forest by the beautiful python package TreeInterpreter
# 
# >Let us take one row and try to predict & interpret.

# In[ ]:


row = X_test.values[None, 0]
row


# >Let us predict using Tree Interpreter for the above feature. The Tree Interpreter provides us the prediction, bias of the trainset & contributions to prediction for each feature. I have used the initial model and we can use any model.
# 

# In[ ]:


prediction, bias, contributions = ti.predict(model, row)


# >Let us check the prediction & bias values.

# In[ ]:


prediction[0], bias[0]


# >To check the prediction of the Tree Interpreter, let us predict for the same feature using Random Forest model. Here we are using prediction probability to compare the values.

# In[ ]:


print(model.predict_proba(row))


# >The Tree interpreter values and the model's prediction probability matches and the 4th class has the highest value. Now let us check the classess from the model.

# In[ ]:


model.classes_ 


# >As per the classes above, the prediction is <4 for the selected row. Let us check that also

# In[ ]:


print(model.predict(row))


# >Most importantly we have to see the contributions for each feature. The below list clearly shows the contributions for all the 4 classes.

# In[ ]:


for c, feature in zip(contributions[0], X.columns):
    print (feature, c)


# In[ ]:


# Let us sum the contributions for this row.

print(contributions.sum(axis=1))


# >We have the bias from the trainset, let us add the above contributions to the bias.

# In[ ]:


with np.printoptions(precision=3, suppress=True):
    print(contributions.sum(axis=1) + bias[0])


# >The above matches with earlier prediction probabilities from the model. Now we know exactly the why the model has predicted the 4th class <4.

# # 5. Conclusions
# 

# > =>From this analysis company should predominantly focus on employees who are marking high absenteeism (represented with '>4' or class'1' in an analysis) which is major cause of business loss.
# 
# > =>Organisation can consider following points for improvement:-
# > 1. Arrange health camps as health issues are observed amongst employees
# > 2. Provide paid leaves, Encash leaves.
# > 3. Rented accomodation for employees who are living far.
# > 4. Grant excess compensation for overtime and more....

# >Thank You
