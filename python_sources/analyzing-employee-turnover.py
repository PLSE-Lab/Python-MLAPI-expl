#!/usr/bin/env python
# coding: utf-8

# # Providing value to the Business: Analyzing Employee Turnover

# # Motivation
# Attrition is a key factor for any HR organization. [Research](https://psycnet.apa.org/doiLanding?doi=10.1037%2Fa0030723) shows that increased talent retention is associated with increased productivity while lower attrition rates are associated with higher sales and financial performance. Additionally, attrition directly impacts the bottom line as [researchers](https://books.google.com/books?hl=en&lr=&id=fgJzAwAAQBAJ&oi=fnd&pg=PP1&dq=Griffeth,+R.,+%26+Hom,+P.+2001+&ots=nvMC_k8KXy&sig=pX_aPKq0de93WbEPDXhnD5gjVfo#v=onepage&q=Griffeth%2C%20R.%2C%20%26%20Hom%2C%20P.%202001&f=false) have shown that the cost to replace a single worker ranges anywhere from 93% to 200% of their annual salary. On top of this research, many companies have taken note with many looking specifically at [employee churn](https://www.wsj.com/articles/companies-step-up-efforts-to-keep-workers-from-quitting-11583058602) and how to respond in an attempt to improve the quality of their workforce.  

# # Guide
# To better understand and predict attrition for this organization we will follow these steps:
# 1. Data Cleaning
# 2. Exploratory Data Analysis
# 3. Model Building
# 4. Conclusion

# # Data Cleaning
# We will start by loading the data and then looking at it to determine what features we don't need to look at.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['figure.figsize']=(30,15)
plt.style.use('fivethirtyeight')

df = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# Let's start with the general information to gain a better understanding of what we are looking at

# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.describe()


# We can start removing the columns that don't add anything to our analysis. We can see that emeployee count is a dummy variable for each individual, employee number is an arbitrary number assigned to them, everyone is over 18 and all are at 80 standard hours. 

# In[ ]:


df = df.drop(['EmployeeCount','EmployeeNumber', 'Over18','StandardHours'], axis = 1)


# Now that we have cleaned up the data we can start looking at the data for general insights.
# # Exploratory Data Analysis
# Now let's create a df for the numerical variables. Numerical data are measurements or counts.

# In[ ]:


df_num = df[['Age',
             'DailyRate',
             'DistanceFromHome',
             'HourlyRate',
             'MonthlyIncome',
             'MonthlyRate',
             'NumCompaniesWorked',
             'PercentSalaryHike',
             'TotalWorkingYears',
             'TrainingTimesLastYear',
             'YearsAtCompany',
             'YearsInCurrentRole',
             'YearsSinceLastPromotion',
             'YearsWithCurrManager']]


# ### Histogram
# Let's start by looking at a histogram. 

# In[ ]:


df_num.hist()


# Histogram Analysis:
# - Age follows a normal distribution
# - While monthly income follows the patter that we would expect where there are more people at lower ranks who earn less the daily rate, hourly rate and monthly rate do not follow that pattern 
# - Distance from home, monthly income, percent salary hike, years at company, and years since last promotion seem to follow roughly an exponential distribution indicating that will not be able to run a [regression analysis](http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/R/R5_Correlation-Regression/R5_Correlation-Regression4.html) on them

# ### Boxplots

# In[ ]:


for i in df_num.columns:
    df_num[[i]].boxplot()
    plt.show()


# The boxplots seem to confirm many of our observations from our historgrams. Something of note is that:
# - There are several outliers in monthly income, total working years, years at company, and years since last promotion

# ### Heatmap

# In[ ]:


sns.set(font_scale=2)
num_heatmap = sns.heatmap(df_num.corr(), annot=True, cmap='Blues')
num_heatmap.set_xticklabels(num_heatmap.get_yticklabels(), rotation=40)
plt.show()


# Heat Map Analysis:
# - Age, monthly income, total working years, years at company, years in current role, years since last promotion, and years with current manager all seem to have some correlation with each other. Most of this should be expected since as people age, they will gain experience which will make them more valuable, but also more specialized so it might make it more difficult for them to move out of the role they are in
# - Number of companies worked for doesn't quite fit into this cluster neatly, which also should be expected since the longer you work at one company, the less time you have to work at other companies
# - It's interesting to see that there isn't a strong correlation between number of companies worked for and monthly income because [research](https://www.forbes.com/sites/cameronkeng/2014/06/22/employees-that-stay-in-companies-longer-than-2-years-get-paid-50-less/#69b24daee07f) indicates that there is a correlation between switching jobs and an increase in pay. When we see in our data that there is a correlation between monthly income, years at company, years in current role, years since last promotion and years with current manager, we can surmise that the company strongly rewards loyalty.

# Since there are so many correlations, we will have to be careful about using a regression with this numerical data since so much [multicolinearity can cause instability](https://blog.exploratory.io/why-multicollinearity-is-bad-and-how-to-detect-it-in-your-regression-models-e40d782e67e) in our model. We will either have to feature engineer them out of our model or look at the categorical variables to create our model. 

# ### Categorical Analysis
# Now let's create a df for the categorial variables. This will also include ordinal values such as ratings from 0-5.

# In[ ]:


df_cat = df[['Attrition',
             'BusinessTravel',
             'Department',
             'Education',
             'EducationField',
             'EnvironmentSatisfaction',
             'JobInvolvement',
             'JobLevel',
             'JobRole',
             'JobSatisfaction',
             'MaritalStatus',
             'OverTime',
             'PerformanceRating',
             'RelationshipSatisfaction',
             ]]


# ### Heatmap

# In[ ]:


sns.set(font_scale=2)
num_heatmap = sns.heatmap(df_cat.corr(), annot=True, cmap='Blues')
num_heatmap.set_xticklabels(num_heatmap.get_yticklabels(), rotation=40)
plt.show()


# This time we find fewer correlations than with the numerical data. This indicates that we could use the categorical data to build our model.

# ### Barplot

# In[ ]:


for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    plt.show()


# Barplot Analysis:
# - Here we can see one of the most important categories that we want to look at is attrition which is ~ 14%. If we could collect more data we'd like to see how this compares in our industry and region to benchmark how well we are doing.
# - There are more single and divorced combined than married people at the company (the ethics and legalities of this data might preclude analyzing or acting on this information)
# - The vast majority of employees receive a 3 with none receiving a 1 or 2 and only ~ 15% receiving a 4. This is disappointing because a further analysis in employee churn is to see how effective it is. While a high amount of turnover is often bad for the company, very low rates of employee turnover could also be bad because the company is not moving on employees who are poor fits or poor performers. When you have a greater distribution of employee ratings it is easier to see how effective the turnover is. 

# ### Pivot Tables
# Now we will compare our categorical columns with attrition.
# Let's make a few more columns where we have counts of attrition yes and no.

# In[ ]:


df_cat['AttritYes'] = df_cat['Attrition'].apply(lambda x: 1 if x =='Yes' else 0)
df_cat['AttritNo'] = df_cat['Attrition'].apply(lambda x: 1 if x =='No' else 0)

p_columns = ['BusinessTravel',
             'Department',
             'Education',
             'EducationField',
             'EnvironmentSatisfaction',
             'JobInvolvement',
             'JobLevel',
             'JobRole',
             'JobSatisfaction',
             'MaritalStatus',
             'OverTime',
             'PerformanceRating',
             'RelationshipSatisfaction']

for i in p_columns:
    m = df_cat.pivot_table(columns=i, values = ['AttritYes','AttritNo'], aggfunc=np.sum)
    m.loc['PercentAttrit'] = 0
    for a in m:
        m.loc['PercentAttrit'][a] = ((m[a][1])/(m[a][0]+m[a][1]))*100
    print(m)
    print("")


# Pivot table analysis:
# 
# There is quite a bit of information here. Much of this should be used to look at the best and worst cases in order to determine how to respond to Attrition rates similar to how [Google approaches people analytics](https://www.amazon.com/dp/B00NLHJKBE/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1). For instance further data gathering such as employee interviews and employee sentiment surveys can be sent out to high attrition areas of the company such as those with low job involvement (33%), low job level (26%), and low environment satisfaction (25%) to determine what is causing people to leave. Conversely similar data gathering can also be conducted on low attrition areas such as Research Directors (2%), mid-high job level (4%) or even high Job Involvement (9%) to determine what is working well.

# # Model Building
# Due to the multicolinearity in the numerical variables we are going to look at the categorical variables. Since this is a classifcation problem where we are looking at a binary outcome between whether or not an employee with attrit, we will use a logistic regression, discriminant analysis, support vector machine and random forest for our analysis.

# In[ ]:


df_model = df[['Attrition',
              'BusinessTravel',
              'Department',
              'Education',
              'EducationField',
              'EnvironmentSatisfaction',
              'JobInvolvement',
              'JobLevel',
              'JobRole',
              'JobSatisfaction',
               'MaritalStatus',
              'OverTime',
              'PerformanceRating',
               'RelationshipSatisfaction',
              ]]


# Let's make these categorical variables dummy values so we can use them in our analysis.

# In[ ]:


df_dum = pd.get_dummies(df_model)


# Now, let's train the test splits. This will allow us to test the models that our analyses produce against real information from our dataset.

# In[ ]:


from sklearn.model_selection import train_test_split
X =df_dum.drop(['Attrition_Yes', 'Attrition_No'], axis=1) 
y = df_dum.Attrition_Yes.values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Logistic Regression
# Let's start with a logistic regression because we are attempting to predict the binary result of attrition.

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_score = lda.score(X_test, y_test)
print('Linear Discriminant accuracy: ', lda_score)


# Logistic Regression is able to predict attrition at an 87% accuracy. Let's see if we can get better accuracy with another model.
# ## Discriminant Analysis
# We can use the discriminant analysis here since the input variables will not result in proportional changes to the output analysis. Either the employee will attrit or will not attrit. There is no in between. 

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
pred_lda = lda.predict(X_test)
lda_score = lda.score(X_test, y_test)
print('Linear Discriminant accuracy: ', lda_score)


# Both the discriminant and the logistic regression have the same accuracy, so let's continue to see if another model is more accurate.

# ## Support Vector Machine
# SVM shows the difference between classes so we could use this to see predict whether an employee will attrit or not.

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
s_vm = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
s_vm.fit(X_train, y_train)
s_vm_score = s_vm.score(X_test, y_test)
print('Support Vector Machine accuracy: ', s_vm_score)


# Ding Ding Ding! We have a winner. SVM's accuracy is better than the other models but let's see if Random Forest can beat SVM.

# ## Random Forest
# We can use Random Forest here since this model predicts the likelihood between discrete problems like whether or not an employee will attrit.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=10, random_state=0)
rfc.fit(X_train, y_train)
rfc_score = rfc.score(X_test, y_test)
print('Random Forest accuracy: ', rfc_score)


# While better than logisitic regression, and Discriminant Analysis, Random Forest does not have a higher accuracy than SVM. 
#  
# Since SVM's accuracy is the highest at 88%, let's dig a little deeper by looking at precision and recall between attrition and retention.

# In[ ]:


from sklearn.metrics import classification_report
pred_s_vm = s_vm.predict(X_test)
print(classification_report(y_test, pred_s_vm))


# Looking first at employees retained (row with index of 0) we see that the precision is .89. In other words this model predicts an employee will stay 89% of the time. On the other hand, recall is 98%. In other words this model correctly identifies 98% of all employees retained.
#  
# Looking next at employees attrited (row with index of 1) we see that the precision is .64. In other words this model predicts an employee will leave 64% of the time. On the other hand, recall is 23%. In other words this model correctly identifies only 23% of all attrits.

# # Conclusion

# Through this exercise we have learned that we can create a model that achieves good precision of whether an individual will stay (89%) and an okay precision of who might leave (64%). A business leader can use this information to understand her team and better understand what staffing needs are. Additionally, she can use this to better predict the individuals who will stay or leave. While the 64% is not a high number of precision, the business leader can still use this model to better expect who will leave and pair this with qualitiative factors to identify who they should further invest in or who they should divest in. 
# 
# For next steps, the HR analytics team should: First, identify the quality of the attrits by identifying them as regretted or non-regretted attrits. Second, identify why people are leaving through an exit survey. Third, learn more about the high attrition areas and low attrition areas of the company to better understand the environmental trends impacting attrition. These analyses will help the business better understand how harmful attrition is and the different factors contributing to attrition. 

# In[ ]:




