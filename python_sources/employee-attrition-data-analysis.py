#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
import os
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Reading Datasets

# In[ ]:


train = pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')
test = pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')
submission_format = pd.read_csv('/kaggle/input/summeranalytics2020/Sample_submission.csv')


# ## Viewing different Datasets

# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


test.head()


# In[ ]:


test.describe()


# In[ ]:


submission_format.head()


# ## EDA and Visualizations 

# ### Department-wise and Education fields based Employees attrition 
# Here, we will observe employees with which Education background and Department have faced more firing. ie: higher attrition

# In[ ]:


education_df = train[['EducationField','Department','Attrition']]
ed_field_list = []
dept_list = []
attrition_rate_list = []
HR_count = len(education_df[education_df['Department']=='Human Resources']) 
RD_count = len(education_df[education_df['Department']=='Research & Development'])
Sales_count = len(education_df[education_df['Department']=='Sales'])
education_dict = {}
for ed_field,dept_df in education_df.groupby('EducationField'):
    dept_df = dept_df.groupby('Department').sum()
    dept_dict = dept_df['Attrition'].to_dict()
    if 'Human Resources' in dept_dict:
        dept_dict['Human Resources'] = dept_dict['Human Resources']*100/HR_count
    if 'Sales' in dept_dict:
        dept_dict['Sales'] = dept_dict['Sales']*100/Sales_count
    if 'Research & Development' in dept_dict:
        dept_dict['Research & Development'] = dept_dict['Research & Development']*100/Sales_count
    education_dict[ed_field] = dept_dict
education_df = pd.DataFrame(education_dict)
count = 1 
plt.rcParams['figure.figsize'] = (10,35)
for i in list(education_df.columns):
    plt.subplot(6,1,count)
    count+=1
    plt.title('Employees with '+i+' Education Field')
    plt.bar(list(education_df.index),list(education_df[i]),color = ['orange','red','green'])
    plt.xlabel('Department')
    plt.ylabel('Department Attrition %')


# * Among all the Employees working in HR Department, for Employees with HR Education background, attrition percent is highest (about 30%). 
# * Among all the Employees working in R&D Department, for Employees with Medical and Life Sciences background, attrition percent is higher (about 29% each) compared to other backgrounds.
# * Among all the Employees working in Sales Department, for Employees with Marketing Education background, attrition percent is highest (about 20%)
# 
# This could mean that the versatility and different skill-sets obtained due to different education background comes in handy in reducing the chances of being fired.

# ### Impact of Job Satisfaction,Job Involvement, Work Environment on the Employee Attrition   

# In[ ]:


def RatingMap(value):
    rating_dict = {1:'Low',2:'Medium',3:'High',4:'Very High'}
    return rating_dict[value]
employee_satisfaction_df = train[['JobSatisfaction','JobInvolvement','EnvironmentSatisfaction','Attrition']]
employee_retained = employee_satisfaction_df[employee_satisfaction_df['Attrition']==0]
employee_left = employee_satisfaction_df[employee_satisfaction_df['Attrition']==1]
for df in [employee_retained,employee_left]:
    for col in ['JobSatisfaction','JobInvolvement','EnvironmentSatisfaction']:
        df[col] = df[col].apply(lambda x:RatingMap(x))

#Visualizing for Employees Retained\Left
count = 1
for employee_df in [employee_retained,employee_left]:
    plt.rcParams['figure.figsize'] = (20,10)
    for col in ['JobSatisfaction','JobInvolvement','EnvironmentSatisfaction']:
        retained_dict = employee_df.groupby(col)['Attrition'].count().to_dict()
        df = pd.DataFrame({col:list(retained_dict.keys()),'Employee Count':list(retained_dict.values())})
        plt.subplot(2,3,count)
        sns.barplot(data = df, x=col,y='Employee Count')
        count+=1
        if(count==3):
            plt.title('Employees who remained in the company')
        if(count==6):
            plt.title('Employees who left the company')


# In the above graph, we can see that many good Employees(more than 250) despite having High Job Involvement, Job Satisfaction, Environment Satisfaction left the company. This clearly indicates that were forced to leave due to the company's loss of revenues. Interestingly, in the case of Employees leaving the company, equal number of them have left citing the reason as bad work environment, while in the case Employees retained less than 100 of them find it unsatisfactory. Similarly, close to 200 employees have left due to lack of Job Satisfaction, only close to 150 of them found it highly satisfactory while in the case of retained employees, less than 150 have low Job Satisfaction and more than 250 employees find it satisfactory. In the case of Job involvement,Employees who left had very low involment compared to retained Employees by a factor of three.      

# ### Department-wise Environment, Job Satisfaction, Job Involvement
# Let us look at which department has better work environment, job satisfaction and job involvement. Since these are the three reasons obtained from the above graph that significantly impacts whether the employee leaves the company or not. Also, we need to explore reason why the dept are facing higher attrition rate.

# In[ ]:


department_df = train[['Department','EnvironmentSatisfaction','JobSatisfaction','JobInvolvement']]
for i in ['EnvironmentSatisfaction','JobSatisfaction','JobInvolvement']:
    department_df[i] = department_df[i].apply(lambda x:RatingMap(x))
count = 1
plt.rcParams['figure.figsize'] = (30,20)
for i in ['EnvironmentSatisfaction','JobSatisfaction','JobInvolvement']:
    department_df_list = []
    for department,rating_df in department_df.groupby('Department'):
        department_dict = {}
        df = rating_df.groupby(i)[i].count()
        df = df*100/df.sum()
        rating_dict = df.to_dict()
        department_dict[department] = rating_dict
        feature_department_df = pd.DataFrame(department_dict)
        department_df_list.append(feature_department_df)
    feature_department_df = pd.concat(department_df_list,axis=1)
    x = list(feature_department_df.index)
    for j in ['Human Resources','Research & Development','Sales']:
        plt.subplot(3,3,count)
        plt.bar(x,feature_department_df[j],color=['#32CD32','red','orange','green'])
        plt.xlabel('Satisfaction/Involvement Level')
        plt.ylabel('Percentage')
        count+=1
        plt.title(j+' Department '+i+' Level',loc='center')


# In the above figure, we can see that these metrics are measured uniformly in terms of the Satisfaction/Job Involvement Level across every department with a bare difference of less than 5% in their respective ratings, which implies that employees have similar feelings/pulse about their work, irrespective of the department they belong to.   

# ### Employee Performance Rating vs Attrition
# We will observe whether the employees performance rating impacts their retaining in the company  

# In[ ]:


def performance(val):
    performance_dict = {1:'Low',2:'Medium',3:'High',4:'Very High'}
    return performance_dict[val]
performance_df = train[['PerformanceRating','Attrition']]
performance_df['PerformanceRating'] = performance_df['PerformanceRating'].apply(lambda x:performance(x))
employees_left = performance_df[performance_df['Attrition']==1]
employees_retained = performance_df[performance_df['Attrition']==0]
#For Employees Left
plt.subplot(2,1,1)
plt.title('Performance Rating of Employees who left the organization', fontsize=14, ha='center')
employees_left = employees_left.groupby('PerformanceRating').count()
plt.pie(employees_left,explode = (0,0.1),autopct='%1.1f%%',labels=list(employees_left.index)) 
#For Employess Retained
plt.subplot(2,1,2)
plt.title('Performance Rating of Employees retained in the organization', fontsize=14, ha='center')
employees_retained = employees_retained.groupby('PerformanceRating').count()
plt.pie(employees_retained,explode = (0,0.1),autopct='%1.1f%%',labels=list(employees_left.index))


# Interestingly, we see that Employees who left the organisation collectively had 0.7% greater than the retained employees in the very high performance category, even though employees there were higher number of Low Satisfaction cases in the Employees who left compared to Employees who were retained. Let us what else could be the reason for employees leaving vs employees being retained. 

# ### Avg.years at company, Avg.years in current role, Avg.years since last promotion vs Attrition
# Let us explore another aspect of Employee Attrition, which is Years at Current Role, Years since Promotion since above we have also found that highly skilled Employees have also left from the organization, which had higher number of low Job Satisfaction cases.

# In[ ]:


employee_year_df = train[['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','Attrition']]
employee_retained_df = employee_year_df[employee_year_df['Attrition']==0]
employee_left_df = employee_year_df[employee_year_df['Attrition']==1]
plt.rcParams['figure.figsize'] = (10,3)

Avg_years_company_retained = employee_retained_df['YearsAtCompany'].mean()
Avg_years_company_left = employee_left_df['YearsAtCompany'].mean()
plt.barh(['Retained Employees','Employees who Left'],[Avg_years_company_retained,Avg_years_company_left],label='Avg.Years at Company')

Avg_years_company_retained = employee_retained_df['YearsInCurrentRole'].mean()
Avg_years_company_left = employee_left_df['YearsInCurrentRole'].mean()
plt.barh(['Retained Employees','Employees who Left'],[Avg_years_company_retained,Avg_years_company_left],label='Avg.Years in Current Role')

Avg_years_company_retained = employee_retained_df['YearsSinceLastPromotion'].mean()
Avg_years_company_left = employee_left_df['YearsSinceLastPromotion'].mean()
plt.barh(['Retained Employees','Employees who Left'],[Avg_years_company_retained,Avg_years_company_left],label='Avg.Years since Last Promotion')

plt.xlabel('Average Years')
plt.legend()


# So we see that, Retained Employees have spent on average about 2.5 years in the current role, while the Employees leaving the organization less than a year in the current role. Also, total years spent by retained employees on average were 3 years, while employees who left spent an average of 2 years. This shows that majority of sacked employees were promoted recently into the current role, albeit high performing and in equal footing to the Retained Employees. Also, we see that they were promoted faster on average compared to their retained counterparts.     

# ### Job Role, Behaviour and Communication Skill vs Attrition 
# Here we will observe which Job Roles faced higher attrition rate, also whether Behaviour and Communication Skills impacts the rate in any way. Also, we will observe the Behaviour and Communication Skill difference across various job roles.

# In[ ]:


def behaviour(val):
    behaviour_dict = {1:'Good',2:'Bad',3:'Not Rated'}
    return behaviour_dict[val]
def communication(val):
    communication_dict = {1:'Bad',2:'Average',3:'Good',4:'Better',5:'Best'}
    return communication_dict[val]
df = train[['JobRole','Behaviour','CommunicationSkill']]
df['Behaviour'] = train['Behaviour'].apply(lambda x: behaviour(x))
df['CommunicationSkill'] = train['CommunicationSkill'].apply(lambda x: communication(x))
jobrole_count = len(set(df['JobRole']))
count = 1
plt.rcParams['figure.figsize'] = (30,30) 
for jobrole,jobrole_df in df.groupby('JobRole'):
    total_count = len(jobrole_df)
    behaviour_dict = (jobrole_df['Behaviour'].value_counts()*100/total_count).to_dict()
    comm_dict = (jobrole_df['CommunicationSkill'].value_counts()*100/total_count).to_dict()
    plt.subplot(jobrole_count,2,count)
    plt.title(jobrole+' Behaviour')
    plt.barh(list(behaviour_dict.keys()),list(behaviour_dict.values()),color='#90EE90')
    count+=1
    plt.subplot(jobrole_count,2,count)
    plt.title(jobrole+' Communication Skill')
    plt.barh(sorted(comm_dict.keys()),list(comm_dict.values()),color=['orange','red','green','lime','#90EE90'])
    count+=1


# We see that across all Job Roles, every employee behaviour is good, whereas the highest percentage of Bad Communication Skill is observed among Sales Executive, Sales Representative (about 23%). Interestingly, more than 30% of Sales Representative have average Communication Skill, third only to Human Resources, Managers. Highest percentage of best communication is observed among Human Resources (about 22.5%).

# In[ ]:


df = train[['JobRole','Attrition']]
JobRole_attrition_df = df.groupby('JobRole').sum()
total_Attrition = JobRole_attrition_df['Attrition'].sum()
JobRole_attrition_df['JobRole'] = list(JobRole_attrition_df.index) 
JobRole_attrition_df.index = range(len(JobRole_attrition_df))
JobRole_attrition_df['Attrition %'] = JobRole_attrition_df['Attrition']*100/total_Attrition
plt.rcParams['figure.figsize'] = (10,3)
sns.barplot(data = JobRole_attrition_df,x='JobRole',y='Attrition %')
plt.xticks(rotation=90)


# It appears that Attrition % is highest among Sales Executive (23%), closely followed by Laboratory (22%) Technician, followed by Research Scientist (21%). These three job roles have suffered more compared to other job roles. Lowest Attrition % is observed for the role of Research Director (0%). The general trend observed here is that bigger the role is in an organization, less are the chances of being released from the company. 

# Since we found that every employee had good behaviour irrespective of which Job Roles they belonged to. The behaviour doesn't dictate the firing or retaining of the Employees, let us check the Communication Skill.  

# In[ ]:


df = train[['CommunicationSkill','Attrition']]
df['CommunicationSkill'] = df['CommunicationSkill'].apply(lambda x:communication(x))
attrition_df = df[['CommunicationSkill','Attrition']].groupby('CommunicationSkill').sum()
comm_skill_count_df = df['CommunicationSkill'].value_counts()
attrition_df['Attrition %'] = attrition_df['Attrition']*100/comm_skill_count_df
attrition_df['CommunicationSkill'] = list(attrition_df.index)
attrition_df.index = range(len(attrition_df))
sns.barplot(data=attrition_df,x='CommunicationSkill',y='Attrition %')
plt.title('Communication Skill vs Attrition %')


# It's interesting to see that Employees with Best Communication Skill have highest Attrition % (about 56%), while Employees with Bad Communication Skill have lowest (about 36%). A significant difference of 20% ! One would have expected quite the opposite.

# ### Age, Gender vs Attrition %
# Here, we will observe which Age Group and Gender have been impacted more by the Job Loss. 

# In[ ]:


age_gen_df = train[['Age','Gender','Attrition']]
sns.distplot(age_gen_df['Age'])
plt.ylabel('Employee Relative frequency')


#  In the above figure, we see that all the employees lies in the range of 10-60 yrs, with majority of them in the age group of 30-40. We will use the range 10-60 and divide it into several age-groups at the interval of 10 to calculate attrition % among them, which makes sense since after 60yrs of age, Employees retire from the organization.

# In[ ]:


age_groups = [[10,20],[21,30],[31,40],[41,50],[51,60]]
result = []
total_attrition = age_gen_df['Attrition'].sum()
for group in age_groups:
    df = age_gen_df[(age_gen_df['Age']>=group[0]) & (age_gen_df['Age']<=group[1])]
    age_df = df[['Age','Attrition']].groupby('Age').sum()
    for gen in ['Male','Female']:
        gender_df = df[df['Gender']==gen]
        result.append([str(group[0])+'-'+str(group[1]),gender_df['Attrition'].sum()*100/total_attrition,gen])
age_group_df = pd.DataFrame(data = result,columns=['Age-group','Attrition %','Gender'])
sns.barplot(data = age_group_df,x='Age-group',y='Attrition %',hue='Gender')
plt.title('Gender-wise Age group vs Attrition %')


# So, we see that the age-group of 21-30 suffers the highest attrition (about 23% in Male, 13% in Female), which makes sense since majority of them are freshers in company starting straight after finishing from the college. They are the dispensable resources due to lack of skills, and are let go first if the cash flow of the company starts falling due to external factors. For 10-20 age group its less because very less no. of people start their corporate career at the age of 19 or 20. For higher age group, the attrition % starts falling due to high amount of exposure, experience leads to firm grip in their positioning and role in the company.  

# ### Business Travel, Marital Status vs Attrition %
# Since Lockdown measures in the current scenario, have restricted travelling, let us examine which type of employees frequently traveling or rarely traveling were affected the most. Also, we will check the marital status of employees and its correlation with Business Travel and Attrition.

# In[ ]:


marriage_df = train[['MaritalStatus','BusinessTravel']]
df_list = []
for travel_info,status_df in marriage_df.groupby('BusinessTravel'):
    new_status = status_df['MaritalStatus'].value_counts()
    new_status_df = pd.DataFrame({'MaritalStatus':list(new_status.index),'Employee Count':list(new_status)})
    new_status_df.index = range(len(new_status_df))
    new_status_df['BusinessTravel'] = [travel_info]*len(new_status_df)
    df_list.append(new_status_df)
marriage_df = pd.concat(df_list)
sns.barplot(data=marriage_df,x='BusinessTravel',y='Employee Count',hue='MaritalStatus')
plt.title('Business Travel, Marital Status of Employees')


# Here, we see that Married Employees travel rarely (about 500),which is highest in that category since they have to take care of their household, rarely making time for business travel. While, in the Travel Frequently Category, Bachelors take up highest count(about 180) since they don't have the burden of raising children, household chores unlike Married Employees. But, interestingly, Divorced Employee Count is lowest in all the three category. This can also be because of less number of records for Divorced Employees.

# In[ ]:


travel_df = train[['BusinessTravel','Attrition']]
total_attrition = travel_df['Attrition'].sum()
travel_attrition_df = travel_df[['BusinessTravel','Attrition']].groupby('BusinessTravel').sum()*100/total_attrition
plt.barh(list(travel_attrition_df.index),list(travel_attrition_df['Attrition']),color=['Green','Orange','Red'])
plt.xlabel('Attrition %')
plt.title('Traveling Frequency vs Attrition %')


# Jobs involving rare traveling have been affected the most having more than 60% Attrition Rate. While, Jobs involving Non-travel have the least Attrition Rate of less than 5%. Makes sense, because in recent times, majority of firings have been taken place for working in IT Sectors as Software Engineers in companies like Capgemini, Cognizant, Uber, Ola Cabs etc. This concludes that majority of jobs that were affected rarely involved traveling, traveling restrictions didn't impact jobs as much as loss of revenues due to the lockdown measures.   

# In[ ]:


marriage_df = train[['MaritalStatus','Attrition']]
total_attrition = marriage_df['Attrition'].sum()
marriage_attrition_df = marriage_df[['MaritalStatus','Attrition']].groupby('MaritalStatus').sum()*100/total_attrition
plt.barh(list(marriage_attrition_df.index),list(marriage_attrition_df['Attrition']),color=['Green','Orange','Blue'])
plt.xlabel('Attrition %')
plt.title('Marital Status of Employees vs Attrition %')


# It appears that Bachelor Employees had the highest attrition rate while Divorced Employees had lowest. This is quite related to Age vs Attrition graph where we saw that age group of 21-30 were affected from job loss the most. Generally, these lower age groups fall under Bachelors category, so the higher attrition rate is justified due to lack of required skillsets and experience in Coporate World. Married and Divorced Employees belong to higher age groups who have plenty of corporate and business exposure, leading to lower attrition rate. 

# ## Correlation Analysis & Feature Engineering 
# We have already done some Correlation of Features with the target variable through different Visualization technique. Let us look further into identifying all the significant features with the Outcome. But before that, we will do some Pre-Processing on the dataset. 

# In[ ]:


new_train = train.copy()
bool_series = new_train['EmployeeNumber'].duplicated()
new_train = new_train[~bool_series]


# In[ ]:


len(new_train[new_train['Attrition']==0])*100/len(new_train)


# In[ ]:


len(new_train[new_train['Attrition']==1])*100/len(new_train)


# ### Missing Value Treatment 
# Let us look whether there are any missing values in the dataset

# In[ ]:


new_train.isnull().sum()


# In[ ]:


test.isnull().sum()


# We see that there's no need of missing value treatment since there are no null values in the columns

# ### Label Encoding Categorical Features

# In[ ]:


for col in new_train.columns:
    if(isinstance(train[col][0],str)):
        new_train[col] = LabelEncoder().fit_transform(new_train[col])


# ### Dropping unnecessary columns

# Here 'Id' and 'EmployeeNumber' are not important features for prediction. Let us drop them.

# In[ ]:


new_train = new_train.drop(['Id','EmployeeNumber'],axis = 1)


# In[ ]:


corr_df = new_train.drop('Behaviour',axis=1).corr()
sns.heatmap(corr_df,annot=True)
plt.rcParams['figure.figsize'] = (30,30)


# ### Splitting the full train data into train and validation set. 

# In[ ]:


X = new_train.drop('Attrition',axis=1)
X['MonthlyIncome'] = np.cbrt(X['MonthlyIncome'])
X['TotalWorkingYears'] = np.cbrt(X['TotalWorkingYears'])
X['YearsAtCompany'] = np.cbrt(X['YearsAtCompany'])
X['YearsSinceLastPromotion'] = np.cbrt(X['YearsSinceLastPromotion'])
X['DistanceFromHome'] = np.cbrt(X['DistanceFromHome'])
Y = new_train['Attrition']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=1)


# ### Performing GridSearch 
# Commented during runtime to save computation resources

# In[ ]:


#rf = RandomForestClassifier()
#param = {'n_estimators':[50,100,200],'max_features':range(1,X.shape[1]+1),'random_state':[0]}
#rf_gridsearch = GridSearchCV(rf,param_grid = param,n_jobs=-1,scoring='roc_auc')
#rf_gridsearch.fit(X_test,Y_test)
#rf_gridsearch.best_params_
#best_params = {'max_features': 24, 'n_estimators': 100, 'random_state': 0}


# In[ ]:


#dtc = DecisionTreeClassifier()
#param = {'max_features':range(1,X.shape[1]+1),'random_state':[0]}
#dtc_gridsearch = GridSearchCV(dtc,param_grid = param,n_jobs=-1,scoring='roc_auc')
#dtc_gridsearch.fit(X_test,Y_test)
#dtc_gridsearch.best_params_
#best_params = {'max_features': 24, 'random_state': 0}


# In[ ]:


#gbc = GradientBoostingClassifier()
#param = {'n_estimators':[50,100,200],'random_state':[0],'max_features':range(1,X.shape[1]+1),
#        'learning_rate':[0.01,0.1,1]}
#gbc_gridsearch = GridSearchCV(gbc,param_grid = param,n_jobs=-1,scoring='roc_auc')
#gbc_gridsearch.fit(X_test,Y_test)
#gbc_gridsearch.best_params_
#best_param = {'learning_rate': 1, 'max_features': 5, 'n_estimators': 100, 'random_state': 0}


# In[ ]:


#svc = SVC(probability=True)
#param = {'kernel':['rbf'],'gamma':[0.001,0.01,0.1,1,10],'C':[0.001,0.01,0.1,1,10]}
#svc_gridsearch = GridSearchCV(svc,param_grid = param,n_jobs=-1,scoring='roc_auc')
#svc_gridsearch.fit(X_test,Y_test)
#svc_gridsearch.best_params_
#best_params = {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}


# In[ ]:


#log_reg = LogisticRegression()
#param = {'max_iter':[100,1000,10000],'C':[0.01,0.1,1,10]}
#log_reg_gridsearch = GridSearchCV(log_reg,param_grid = param,n_jobs=-1,scoring='roc_auc')
#log_reg_gridsearch.fit(X_test,Y_test)
#log_reg_gridsearch.best_params_
#best_params = {'C': 1, 'max_iter': 1000}


# In[ ]:


#mlp = MLPClassifier()
#param = {'random_state':[0],'activation':['logistic'],'max_iter':range(100,1100,100),
#         'solver':['lbfgs', 'sgd', 'adam'],'hidden_layer_sizes':[(100,),(1000,),(10000,)]}
#mlp_gridsearch = GridSearchCV(mlp,param_grid=param,n_jobs=-1,scoring='roc_auc')
#mlp_gridsearch.fit(X_test,Y_test)
#mlp_gridsearch.best_params_
#best_params = {'activation': 'logistic','hidden_layer_sizes': (10000,), 'max_iter': 300, 'random_state': 0, 'solver': 'adam'}


# ### Checking ROC-AUC Score for Test data based on GridSearch results

# In[ ]:


rf = RandomForestClassifier(n_estimators=100, random_state = 0,max_features = 24)
rf.fit(X_train,Y_train)
print('For Random Forest Classifier')
score = roc_auc_score(Y_train, rf.predict_proba(X_train)[:,1])
print('Train roc_auc_score:',score)
score = roc_auc_score(Y_test, rf.predict_proba(X_test)[:,1])
print("Test roc_auc_score:",score)


# In[ ]:


gbc = GradientBoostingClassifier(n_estimators=100, random_state = 0,learning_rate = 1,max_features=5)
gbc.fit(X_train,Y_train)
print('For Gradient Boost Classifier')
score = roc_auc_score(Y_train, gbc.predict_proba(X_train)[:,1])
print('Train roc_auc_score:',score)
score = roc_auc_score(Y_test, gbc.predict_proba(X_test)[:,1])
print("Test roc_auc_score:",score)


# In[ ]:


dtc = DecisionTreeClassifier(random_state = 0,max_features=24)
dtc.fit(X_train,Y_train)
print('For Decision Tree Classifier')
score = roc_auc_score(Y_train, dtc.predict_proba(X_train)[:,1])
print('Train roc_auc_score:',score)
score = roc_auc_score(Y_test, dtc.predict_proba(X_test)[:,1])
print("Test roc_auc_score:",score)


# In[ ]:


svc = SVC(probability=True,kernel='rbf',C=0.1,gamma=0.001)
svc.fit(X_train,Y_train)
print('For Support Vector Classifier')
score = roc_auc_score(Y_train, svc.predict_proba(X_train)[:,1])
print('Train roc_auc_score:',score)
score = roc_auc_score(Y_test, svc.predict_proba(X_test)[:,1])
print("Test roc_auc_score:",score)


# In[ ]:


log_reg = LogisticRegression(C = 1,max_iter=1000) 
log_reg.fit(X_train,Y_train)
print('For Logistic Regression')
score = roc_auc_score(Y_train, log_reg.predict_proba(X_train)[:,1])
print('Train roc_auc_score:',score)
score = roc_auc_score(Y_test, log_reg.predict_proba(X_test)[:,1])
print("Test roc_auc_score:",score)


# In[ ]:


mlp = MLPClassifier(random_state=0,activation='logistic',max_iter=300,hidden_layer_sizes=(10000,))
mlp.fit(X_train,Y_train)
print('For Mulit-Layer Perceptron')
score = roc_auc_score(Y_train, mlp.predict_proba(X_train)[:,1])
print('Train roc_auc_score:',score)
score = roc_auc_score(Y_test, mlp.predict_proba(X_test)[:,1])
print("Test roc_auc_score:",score)


# So, we see that our models are doing well on the validation dataset. Let us train the model on the entire train and finally submit the results.  

# In[ ]:


#Fitting Models
models = [rf,gbc,dtc,svc,log_reg,mlp]
for model in models:
    model.fit(X,Y)


# ### Performing similar operations on test data like we did on train data

# In[ ]:


new_test = test.copy()
for col in new_test.columns:
    if(isinstance(test[col][0],str)):
        new_test[col] = LabelEncoder().fit_transform(new_test[col])
new_test = new_test.drop(['Id','EmployeeNumber'],axis = 1)
X_test = new_test
X_test['MonthlyIncome'] = np.cbrt(X_test['MonthlyIncome'])
X_test['TotalWorkingYears'] = np.cbrt(X_test['TotalWorkingYears'])
X_test['YearsAtCompany'] = np.cbrt(X_test['YearsAtCompany'])
X_test['YearsSinceLastPromotion'] = np.cbrt(X_test['YearsSinceLastPromotion'])
X_test['DistanceFromHome'] = np.cbrt(X_test['DistanceFromHome'])


# ### Outputting Results

# In[ ]:


models = [rf,gbc,dtc,svc,log_reg,mlp]
modelname = ['Random Forest','GradientBoost','DecisionTree','SupportVector','Logistic_reg','MLPClassifier']
for model,name in zip(models,modelname):
    test_prob = model.predict_proba(X_test)[:,1]
    result = pd.DataFrame({'Id':list(test['Id']),'Attrition':list(test_prob)})
    result.to_csv('/kaggle/working/'+str(name)+'.csv',index=False)

