#!/usr/bin/env python
# coding: utf-8

# # Human resource analytics 
# # importing data 
# 

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from statsmodels.formula.api import ols

df=pd.read_csv('hr.csv')
df.head()
#checkng null values 
print df.isnull().any()


# # data has no null values  now we will go checking correlation and outliers
# 

# In[ ]:


sns.heatmap(df.corr(),annot=True,fmt=".2f")
plt.show()


# # Renaming certain columns for better readability

# In[ ]:


df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })
front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)
print df.head()


# # printing statiscal report
# #turnover rate 

# In[ ]:


print " turnover rate for 0 an 1 as follows"
turnover_rate = df.turnover.value_counts() / len(df)
print turnover_rate
print "stats report "
print df.describe()
turnover_summary=df.groupby('turnover')
print turnover_summary.mean()


# # our report says that employess who left were actually having samne number      of yr 
# 
# # of exp , working than them on same number of projects , so might be the            reason can 
# 
# # low salaries , more working hrs and less promotions than them 
# 
# # conducting one sample t-test
# 

# In[ ]:


import scipy.stats as sts

emp_pop=df['satisfaction'][df['turnover']==0].mean()
emp_pop_turn=df['satisfaction'][df['turnover']==1].mean()
print 'The mean satisfaction for the employee population with no turnover is: ' + str(emp_pop)
print'The mean satisfaction for employees that had a turnover is: ' + str(emp_pop_turn) 

print sts.ttest_1samp(a=df[df['turnover']==1]['satisfaction'],popmean=emp_pop)


# # that value comes that lies in left quartiles and right or not
# 

# In[ ]:


degree_freedom = len(df[df['turnover']==1])

LQ = sts.t.ppf(0.025,degree_freedom)  # Left Quartile

RQ = sts.t.ppf(0.975,degree_freedom)  # Right Quartile

print 'The t-distribution left quartile range is: ' + str(LQ)
print 'The t-distribution right quartile range is: ' + str(RQ)


# # reject null hypothesis 
# 
# # now as we grouped data we can apply analytics to see 
# 
# # what employees were facing who left and 
# 

# In[ ]:


sns.distplot(df.satisfaction,color="orange")
plt.title("emp satisfaction dist" )
plt.xlabel("satifaction values")
plt.ylabel("emp count")
plt.show()
sns.distplot(df.evaluation,color="red")
plt.title("emp evaluation dist" )
plt.xlabel("evalution values")
plt.ylabel("emp count")
plt.show()
sns.distplot(df.averageMonthlyHours,color="blue")
plt.title("emp avg mon hrs dist" )
plt.xlabel("avg hrs values")
plt.ylabel("emp count")
plt.show()

sns.distplot(df.projectCount,color="green")
plt.title("emp project dist" )
plt.xlabel("project values")
plt.ylabel("emp count")
plt.show()
sns.distplot(df.promotion,color="violet")
plt.title("emp promotion dist" )
plt.xlabel("promotion values")
plt.ylabel("emp count")
plt.show()


# # satisfaction rate low many number of emp and high also for same 
# 
# # evaluation and avg monthly hrs have correlated graphs 
# 
# # which says that evaluation score inc with avg mon hrs
# 
# # porjects count has mostly 2,3,4,5 have correlation with hrs avg monthly 
# 
# # promotion and salary are major fac for leaving a company acc to the post 
# 
# # since we can see that promotion are given to very less emp and i think most      of them 
# # were senior to others that's why 
# 
# 
# # so we can see for our prospects salary , avg monthly hrs , projects will be the    major point 
# 
# # of leaving a company
# 
# # plotting graphs for turnover vs salary

# In[ ]:


type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]# Capture the original matplotlib rcParams

sns.countplot(x="salary",data=df,palette=['lightblue','orange'],hue="turnover")
plt.xlabel("salary")
plt.ylabel("emp count")
plt.title("salary vs turnover ")
plt.show()


# # our analytics show that low salary emp left more
# 
# # med salary less and high salary very less
# 
# # plotting graphs turnover vs porject count
# 
# 

# In[ ]:


sns.countplot(x="projectCount",data=df,palette=['#A8B820','#98D8D8'],hue="turnover")
plt.xlabel("project count ")
plt.ylabel("emp count")
plt.title("project vs turnover ")
plt.show()


# # less number of projects tend to leave company more 
# 
# 
# # means not valued enough , they are taken into consideration 
# 
# # no of pro with 3,4,5 their turnover is very less
# 
# # 2,6,7 are leaving more with 7 projets there is complete turnover
# 
# # so we can assume that 
# 

# In[ ]:


sns.countplot(x="yearsAtCompany",data=df,palette=['#705898','#7038F8'],hue="turnover")
plt.xlabel("year at company  ")
plt.ylabel("emp count")
plt.title("years at comp vs turnover ")
plt.show()


# # years at company 2 very less turnover is less
# 
# # with 3,4,5,6 are more 
# 
# # 7,8,8,10 nothing is there
# 
# # assumption is that emp with 3,4,5,6 years 
# 
# # may be less valued , may be salaries are less , and workaccident can be more
# 
# # number of workaccident with are very less and leaving is also
# 

# In[ ]:


sns.countplot(x="department",data=df,palette=type_colors)
plt.xlabel("workaccident")
plt.ylabel("emp count")
plt.title("workaccident vs turnover ")
plt.show()


# # sales area is more  conc than any other
# 

# In[ ]:


sns.countplot(x="department",data=df,palette=type_colors,hue="turnover")
plt.xlabel("workaccident")
plt.ylabel("emp count")
plt.title("workaccident vs turnover ")
plt.xticks(rotation=-45)
plt.show()


# # leaving is form sales ,technical , support 
# 
# 

# In[ ]:


sns.boxplot(x="projectCount",y="averageMonthlyHours",data=df,palette=['blue','orange'],hue="turnover")
plt.xlabel("project count")
plt.ylabel("avg monthly hours")
plt.title("project vs avg monthly hrs  ")
plt.show()


# # less projects less hrs given by some emp 
# 
# # emp worked with 200 hrs monthly stayed and less than 250hrs 
# 
# # emp worked less than 150 and less than that left the company 
# 
# # looks like that emp with 3,4,5,6 projects those who were 
# 
# # showing consistency of working hrs less left with inc no of projects 
# 
# # those who were made to overworked or who are doing are leaving 
# 
# # the company 

# In[ ]:


sns.boxplot(x="projectCount",y="evaluation",data=df,palette=['blue','orange'],hue="turnover")
plt.xlabel("project count")
plt.ylabel("evalutions")
plt.title("project vs evalution ")
plt.show()


# # those emp with 3,4,5,6 projects evaluated more and left 
# 
# # max leaving is with 3 no of projects 
# 
# # evalauted less than 0.6 left the company 
# 
# # 0.6 - 0.8 emp with such eva scores are not leaving 
# 

# In[ ]:


sns.lmplot(x="satisfaction",y="evaluation",data=df,palette=['blue','orange'],hue="turnover",fit_reg=False)
plt.xlabel("satisfaction")
plt.ylabel("evalutions")
plt.title("satisfaction vs evalution ")
plt.show()


# # more eva and less satisfied cluster 1 of hardworking and sad people
# 
# # cluster 2 less satified and less evaluated 
# 
# # cluster 3 is of happy people
# 
# # checking number of clusters needed

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df["department"] = df["department"].astype('category').cat.codes
df["salary"] = df["salary"].astype('category').cat.codes 
print df.head()
x=df.iloc[:,1:10]
print x 
y=df.iloc[:,0]
print y 
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)

tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
forest=RandomForestClassifier()
forest.fit(x_train,y_train)


# # feature importance by tree and forest
# 

# In[ ]:


import numpy as np
n_features=len(df.columns[1:10])


# # tree feature_importances

# In[ ]:


plt.barh(range(n_features),tree.feature_importances_,align="center")
plt.yticks(np.arange(n_features),df.columns[1:10])
plt.xlabel("score by tree")
plt.ylabel("features")
plt.show()


# # forest feature_importances

# In[ ]:


plt.barh(range(n_features),forest.feature_importances_,align="center")
plt.yticks(np.arange(n_features),df.columns[1:10])
plt.xlabel("score by forest")
plt.ylabel("features")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score,confusion_matrix,precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print df.head()
df = df.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })

logis = LogisticRegression(class_weight = "balanced")
logis.fit(x_train, y_train)
print ("\n\n ---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, logis.predict(x_test))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print(classification_report(y_test, logis.predict(x_test)))


# In[ ]:


# Decision Tree Model
dtree = DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(x_train,y_train)
print ("\n\n ---Decision Tree Model---")
dt_roc_auc = roc_auc_score(y_test, dtree.predict(x_test))
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(y_test, dtree.predict(x_test)))


# In[ ]:


# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
rf.fit(x_train, y_train)
print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(x_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(x_test)))


# Ada Boost
gbc = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1)
gbc.fit(x_train,y_train)
print ("\n\n ---GBCBoost Model---")
gbc_roc_auc = roc_auc_score(y_test, gbc.predict(x_test))
print ("GBC Boost model AUC = %2.2f" % gbc_roc_auc)
print(classification_report(y_test, gbc.predict(x_test)))
# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, logis.predict_proba(x_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:,1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(x_test)[:,1])
gbc_fpr, gbc_tpr, ada_thresholds = roc_curve(y_test, gbc.predict_proba(x_test)[:,1])


# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)

# Plot GradientBooseting Boost ROC
plt.plot(gbc_fpr, gbc_tpr, label='GradientBoostingclassifier (area = %0.2f)' % gbc_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




