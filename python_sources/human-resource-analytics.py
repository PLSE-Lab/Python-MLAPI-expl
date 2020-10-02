#!/usr/bin/env python
# coding: utf-8

# # I am going to use random forest to predict whether a employee will leave company or not and will also find out the important feature which are contributing to prediction and will remove less important features . At the end we will find out which is the most contributing factor for  leaving job . 
# 
# # After that will visualise factors affecting decision of employe for leaving company .
# 

# ### Importing libraries 

# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib as mpl
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data

# In[ ]:


data = pd.read_csv('../input/HR_comma_sep.csv')
data["salary"] = data["salary"].replace(['low' , 'medium' , 'high'] , [0 , 1 , 2])

train=data.sample(frac=0.8,random_state=200)
test = data.drop(train.index)
data.head()


# ### I have taken all columns and tried to find out  how much a certain feature is contributing 
# 

# In[ ]:


traininput1 = train[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary']].values
traintarget = train[["left"]].values
testinput1 = test[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary']].values
testtarget = test[["left"]].values

forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(traininput1,traintarget)
print("Testing accuracy :" , my_forest.score(testinput1 , testtarget))
print("Training accuracy :" , my_forest.score(traininput1, traintarget))

features = list(my_forest.feature_importances_)
print("Importance of feattures:")
print("'satisfaction_level :{f[0]}', 'last_evaluation : {f[1]}', 'number_project: {f[2]}','average_montly_hours: {f[3]}', 'time_spend_company: {f[4]}', 'Work_accident: {f[5]}', 'promotion_last_5years: {f[6]}', 'salary: {f[7]}'".format(f = features))



# ### we found out that 'Work_accident' ,'promotion_last_5years', 'salary' are not useful so we will remove and check if there is any change in performance

# In[ ]:


traininput2= train[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company'
       ]].values
testinput2= test[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company'
       ]].values

forest2 = RandomForestClassifier( max_depth=10 , min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest2 = forest2.fit(traininput2,traintarget)
print("Testing accuracy :" , my_forest2.score(testinput2 , testtarget))
print("Training accuracy :" , my_forest2.score(traininput2, traintarget))

features1 =  list(my_forest2.feature_importances_)
print("How important a feature is :")
print("'satisfaction_level :{f[0]}', 'last_evaluation : {f[1]}', 'number_project: {f[2]}','average_montly_hours: {f[3]}','time_spend_company: {f[4]}',".format(f = features1))


# ## There is no much change but still it will help model from overfitting
# ## So we,  came to conclusion that "satisfaction level " contibuted mostly for leaving job.

# ## Lets plot some graphs 

# In[ ]:


plt.figure(figsize = (18,8))
plt.suptitle('Employees who left', fontsize=16)
plt.subplot(1,4,1)
plt.plot(data.satisfaction_level[data.left == 1],data.last_evaluation[data.left == 1],'o', alpha = 0.1)
plt.ylabel('Last Evaluation')
plt.title('Evaluation vs Satisfaction')
plt.xlabel('Satisfaction level')

plt.subplot(1,4,2)
plt.plot(data.satisfaction_level[data.left == 1],data.average_montly_hours[data.left == 1],'o', alpha = 0.1 )
plt.ylabel('Average Monthly Hours')
plt.title('Average hours vs Satisfaction ')
plt.xlabel('Satisfaction level')

plt.subplot(1,4,3)
plt.title('Salary vs Satisfaction ')
plt.plot(data.satisfaction_level[data.left == 1],data.salary[data.left == 1],'o', alpha = 0.1)
plt.xlim([0.4,1])
plt.ylabel('salary ')
plt.xlabel('Satisfaction level')

plt.subplot(1,4,4)
plt.title('Promotions vs Satisfaction ')
plt.plot(data.satisfaction_level[data.left == 1],data.promotion_last_5years[data.left == 1],'o', alpha = 0.1)
plt.xlim([0.4,1])
plt.ylabel('Promotion last 5years')
plt.xlabel('Satisfaction level')



# ## 1) from first two graph we found that employee with low satisfaction with high average monthly hours left company . In top right corner are the employees which may are getting better job in another company . Bottom one may have some personal reasons .
# ## 2)Employee with lower salary are more tend to leave .
# ## 3)Employe who are not promoted in 5 years are highly to leave company .

# ## Lets plot heat map . 

# In[ ]:


import seaborn as sns

correlation = data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='terrain')
plt.title('Correlation between different fearures' )


# ## We can see that employes with lower satisfaction , high working hours , low salary are causes for leaving company
# ## ploting probability of leaving company . We will use logistic regression for predicting probability

# In[ ]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression( C=1000)
logreg.fit(traininput2, traintarget)
probability = logreg.predict_proba(testinput2)
new =pd.DataFrame(list(probability) ,columns=['Stayed','left'])

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title("Employee who left vs Average hours ")
plt.xlabel("Probability of leaving")
plt.ylabel("Average hours")
plt.plot(new["left"] , test["average_montly_hours"] , 'o' , alpha= .3 )

plt.subplot(1,2,2)
plt.title("Employee who left vs Satisfaction level ")
plt.xlabel("Probability of leaving")
plt.ylabel("Satisfiction level")
plt.plot(new["left"] , test["satisfaction_level"] , 'o' , alpha= .3)


# # Lower satisfaction and high working hours are reasons for leaving

# In[ ]:




