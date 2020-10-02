#!/usr/bin/env python
# coding: utf-8

# # What is Employee attrition ?

# Employee attrition is defined as the natural process by which employees leave the workforce . 

# # okay let's think in General using our common sense !

# ## What caused employee to leave his or her workplace or company  ?
# 
#  * Being overworked ! 
#  * No worklife balance .
#  * May be employee hates travelling frequently .
#  * May be employee hates his/her boss .
#  * May be employee Dailywage is less for more work .
#  * May be employee hates to work in the current Department .
#  * May be Distance between home and workplace is more .
#  * May be employee not getting promotion for his/her work .
#  * May be Salary is low .
#  * May be his/her Personal problem .
#  * May be he/she got better job with better pacakage .
#  * May be employee's are not getting recognaized for his/her work .
#  * May be employee a frequently job changer (who never sticks with one company ).
#  * May be Lack of Growth and Progression  .

# # What caused employee to Stay in his or her workplace or company ?
# 
#  * worklife balance .
#  * work satisfaction .
#  * employee loves to work in the current Department .
#  * Distance between home and workplace is less .
#  * employee  getting promotion for his/her work .
#  * Salary is good .
#  * employees are  getting recognaized for his/her work .
#  * Growth and Progression  .

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data = pd.DataFrame(data)
data.head()


# # Data Preprocessing .

# In[ ]:


data.columns = [i.lower() for i in data.columns]


# In[ ]:


# dropping employeecount,over18,standardhours columns because these columns will not be 
# helpful in our analysis .
print("Features having only one value for all of it's index \n")
for i  in data.columns:
    if(data[i].nunique()==1):
        print(i)
print("\n")
data.drop(['employeecount','over18','standardhours',"employeenumber"],axis = 1,inplace = True)
data.columns


# In[ ]:


data.head()


# In[ ]:


# let's rename columns for improved readablity .

data.rename(columns = {
                       "businesstravel" :"travel",
                        "dailyrate":"daily_salary",
                        "distancefromhome":"dist_from_home",
                        "environmentsatisfaction" : "env_satisfaction",
                        "monthlyrate" : "monthly_salary" ,
                        "numcompaniesworked":"companies_worked",
                        "percentsalaryhike" : "hike",
                        "performancerating" : "performance",
                        "relationshipsatisfaction" : "relationship_satisfaction" ,
                        "totalworkingyears" : "experience",
                        "yearsatcompany" : "years_in_company" ,
                        "yearsincurrentrole" : "years_in_currentrole",
                       "yearssincelastpromotion" : "last_promotion" ,
                        "yearswithcurrmanager" :"years_with_current_manager ",
                        "trainingtimeslastyear" :"training"                                   
                      },inplace = True )


# In[ ]:


data.head(5)


# In[ ]:


data.drop(["daily_salary","monthlyincome"],inplace = True,axis = 1)


# # Checking for null values . 

# In[ ]:


print("we don't have missing values in our dataset : \n")
data.isnull().sum()


# In[ ]:


catgorical_data = data.select_dtypes(exclude = "number")
print("catgorical columns of dataset : ")
catgorical_data.columns


# In[ ]:


numerical_data = data.select_dtypes(include = "number")
print("numerical columns of dataset : ")
numerical_data.columns


# In[ ]:


# target re-aranging 
# 1 - means person will leave company .
# 0 - means person will not leave the company .
data.attrition =  data.attrition.map({"Yes":1,"No":0})


# # Let's see weather our target is balanced or not . 

# In[ ]:


print("Target count : \n")
print(data['attrition'].value_counts())
sns.countplot(data['attrition']).set_title("Target is not Balanced ")
plt.show()


# In[ ]:


# For our analysis purpose we are  consider equal number of target labels . (although this is not recommended )

turnover    = data[data["attrition"]==1]  
no_turnover = data[data["attrition"]==0].sample(len(turnover)) 
print(turnover.shape)
print(no_turnover.shape)


# # Explantory Data Analysis . 

# ## What caused employee to leave his or her (workplace) company ?

# ### 1 .  'attrition v/s travel '
#  here what i found from the below chart is that 
#    *  most of the people  who  leave the company are the one who travel Frequently compared to people       who don't   .
# 

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
pd.crosstab(turnover["attrition"],turnover["travel"]).plot(kind = 'barh',ax = axes[0]).set_title('attrition v/s travel ')
pd.crosstab(no_turnover["attrition"],no_turnover["travel"]).plot(kind = 'barh',ax = axes[1]).set_title('no - attrition v/s travel ')
plt.show()


# # 2. 'attrition v/s department' 
# 

# * Most of the people who left the company are from the research and development department 
#   and most of the people who are working in company are from the research and development department. 
#   
#   By this we conclude that which ever the company this dataset belongs has vast employees 
#   in R&D Department .

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
pd.crosstab(turnover["attrition"],turnover["department"]).plot(kind = 'barh',ax = axes[0]).set_title('attrition v/s department ')
pd.crosstab(no_turnover["attrition"],no_turnover["department"]).plot(kind = 'barh',ax = axes[1]).set_title('no - attrition v/s department ')
plt.show()


# # 3.  " Distance from home "
# insight : Most of the employee's , distance from home to workplace are in less than 10km . 

# In[ ]:


lt_10kms = data['dist_from_home'][data['dist_from_home']<=10].count()
gt_10kms = data['dist_from_home'][data['dist_from_home']>10].count()
print("Number of employee's whose distance from home to workplace less than or equal to 10kms : ",lt_10kms)
print("Number of employee's whose distance from home to workplace greater than 10kms   : ",gt_10kms)


# In[ ]:


sns.kdeplot(turnover['dist_from_home'],shade = False,color = "r",legend = False).set(xlim=(0))
sns.kdeplot(no_turnover['dist_from_home'],shade = False,color = "g").set(xlim=(0))
plt.legend(title='distance ', loc='upper right', labels=['turnover', 'no - turnover'])
plt.title("work home distance ")
plt.xlabel("work home distance in kms ")


# # 4 . "attrition v/s env_satisfaction "

# env_satisfaction : 
#     1 - bieng lowest .
#     4 - being highest .

# * insight : env_satisfaction  feature play's an important role ,because the working environment should be good,if it is not employee's tend to leave thier company .
# 
#           * from the below chart we conclude that employee with env_satistfcation  = 1
#           (being lowest) are the one who left the comapny most .
#             
#            # and again
#             
#           * from the below chart we conclude that employee with env_satistfcation  = 3 or 4
#           (being  highest) are the one who did not left the comapny  .
#             

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
pd.crosstab(turnover["attrition"],turnover["env_satisfaction"]).plot(kind = 'barh',ax = axes[0]).set_title('attrition v/s env_satisfaction ')
pd.crosstab(no_turnover["attrition"],no_turnover["env_satisfaction"]).plot(kind = 'barh',ax = axes[1]).set_title('no - attrition v/s env_satisfaction ')
plt.show()


# # 5. 'gender v/s attrition'  

# *  insight :  attrition percent is more for male compared to female. 

# In[ ]:


plt.rcParams['figure.figsize'] = (6,6)
px.pie(turnover,turnover["gender"].value_counts().index,turnover["gender"].value_counts(),hole = 0.6)


# # 6. ' jobsatisfaction v/s attrition '

# * insight : jobsatisfaction  feature play's an important role ,
#     
#          # say for example if you are not satisified with the work you are doing ,then probably you feel demotivated towards in whatever the work your doing . 
# 
#           * from the below chart we conclude that employee with jobsatisfaction  = 1
#           (being lowest) are the one who left the comapny most .
#             
#            # and again
#             
#           * from the below chart we conclude that employee with env_satistfcation  = 3 or 4
#           (being  highest) are the one who did not left the comapny  .
#             

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
pd.crosstab(turnover["attrition"],turnover["jobsatisfaction"]).plot(kind = 'barh',ax = axes[0]).set_title('attrition v/s jobsatisfaction ')
pd.crosstab(no_turnover["attrition"],no_turnover["jobsatisfaction"]).plot(kind = 'barh',ax = axes[1]).set_title('no - attrition v/s jobsatisfaction ')
plt.show()


# # 7 . ' ( jobsatisfaction + jobinvolvement ) v/s attrition '

# * insight : from the below distribution curve ,we can conclude that employee with high job interest are less likely to turnover .

# In[ ]:


turnover["job_interest"] = turnover['jobinvolvement'] + turnover['jobsatisfaction']
no_turnover["job_interest"] = no_turnover['jobinvolvement'] + no_turnover['jobsatisfaction']


# In[ ]:


# create one new feature 
data["job_interest"] = data['jobinvolvement'] + data['jobsatisfaction']
data.drop(['jobinvolvement','jobsatisfaction'],axis = 1,inplace = True)


# In[ ]:


plt.rcParams['figure.figsize'] = (10,6)
sns.kdeplot(turnover['job_interest'],shade = True,color = "r",legend = False).set(xlim=(0))
sns.kdeplot(no_turnover['job_interest'],shade = False,color = "green").set(xlim=(0))
plt.legend(title='distance ',loc = 'best', labels=['turnover_job-interest', 'no - turnover_job-interest'])
plt.title("overall job_interest ")
plt.xlabel('jobsatisfaction + jobinvolvement')
plt.ylabel('frequency')
plt.show()


# In[ ]:


data.drop(['stockoptionlevel','years_in_currentrole','years_with_current_manager ','last_promotion','hourlyrate','experience','training','years_in_company'],axis = 1,inplace = True)
data.columns


# # 8. 'relationship_satisfaction v/s attrition'

# * insight : employee with relationship satisfaction value are less likely to leave the company .

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
pd.crosstab(turnover["attrition"],turnover["relationship_satisfaction"]).plot(kind = 'barh',ax = axes[0]).set_title('attrition v/s relationship_satisfaction ')
pd.crosstab(no_turnover["attrition"],no_turnover["relationship_satisfaction"]).plot(kind = 'barh',ax = axes[1]).set_title('no - attrition v/s relationship_satisfaction ')
plt.show()


# # 9 .  ' life_work_satisfaction v/s attrition '

# In[ ]:


turnover["life_work_satisfaction"] = turnover['worklifebalance'] + turnover['relationship_satisfaction']
no_turnover["life_work_satisfaction"] = no_turnover['worklifebalance'] + no_turnover['relationship_satisfaction']


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
pd.crosstab(turnover["attrition"],turnover["life_work_satisfaction"]).plot(kind = 'barh',ax = axes[0]).set_title('attrition v/s life_work_satisfaction ')
pd.crosstab(no_turnover["attrition"],no_turnover["life_work_satisfaction"]).plot(kind = 'barh',ax = axes[1]).set_title('no - attrition v/s life_work_satisfaction ')
plt.show()


# In[ ]:


data['life_work_satisfaction'] = data['worklifebalance'] + data['relationship_satisfaction']
data.drop(['worklifebalance','relationship_satisfaction'],axis = 1,inplace = True)


# # 10. 'hike v/s attrition '

# * insight  :from the below pie chart we can conclude that most of the people who left the company is due to they are getting low hike .

# In[ ]:


plt.rcParams['figure.figsize'] = (6,6)
px.pie(turnover,turnover["hike"].value_counts().index,turnover["hike"].value_counts(),hole = 0.6,title = 'hike v/s attrition ')


# # Let's the correlation among the features 

# # insights 
#   * hike v/s performance has strong correlation . 
#   * joblevel v/s age has moderate correlation .  

# In[ ]:


plt.rcParams['figure.figsize'] = (10,6)
sns.heatmap(data.corr(),annot = True)


# In[ ]:


# drop either any one of the feature to avoid multicollinearity problem .
# i am dropping performance blindly !
data.drop(['performance'],axis = 1,inplace = True)


# In[ ]:


data.info()


# # Model Building  . 

# * our data is highly imbalanced ,let's make it balanced before providing it to our model .
#  * i'll use undersampling technique just for sake of model building ,
#    * in undersample we make decrease the majority class label up to count of minority class label .

# In[ ]:


attrition = data[data['attrition']==1]
no_attrition = data[data['attrition']==0].sample(len(attrition))
final = pd.concat([attrition,no_attrition],axis = 0)
print("attrition shape : ",attrition.shape)
print("no_attrition shape : ",no_attrition.shape)
print("final df  shape : ",final.shape)


# In[ ]:


plt.rcParams['figure.figsize'] = (6,3)
sns.countplot(final['attrition']).set_title("Target is Balanced")


# In[ ]:


final.info()


# In[ ]:


final = pd.get_dummies(final,drop_first = True)
final.shape


# In[ ]:


x = final.drop('attrition',axis = 1)
y = final['attrition']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,log_loss
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 33)


# # Random Forest Classifier .

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 150,min_samples_split = 20,min_samples_leaf = 5,random_state = 33)
rf.fit(xtrain,ytrain)


# In[ ]:


ypred = rf.predict(xtest)


# In[ ]:


print("classification - report ")
print(classification_report(ytest,ypred))


# In[ ]:


print("confusion matrix  - report ")
print(confusion_matrix(ytest,ypred))


# In[ ]:


print("cost function ")
print(log_loss(ytest,ypred))


# # Thank You : )

# any kind of constructive suggestions are welcome ,Please upvote this kernel if you find it useful .

# In[ ]:




