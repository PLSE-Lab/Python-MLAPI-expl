#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/insurance.csv')


# In[ ]:


df.dtypes


# In[ ]:


df.head(10)


# In[ ]:


df['sex'].value_counts()


# In[ ]:


df['smoker'].value_counts()


# Looks like there are alot of non-smokers in our datset whereas for gender it is almost close. Before any analysis we would change the values as follows:
# 
# For Smoker column:
# 0 = no, 
# 1 = yes
# 
# For Sex column:
# 0 = female, 
# 1 = male
# 

# In[ ]:


df['smoker']=df['smoker'].apply(lambda x: 0 if x=='no' else 1)
df['sex']=df['sex'].apply(lambda x: 0 if x=='female' else 1)


# In[ ]:


df.head()


# First of all, i have less experience in visualization and learned about seaborn after writing this block

# In[ ]:


p=np.arange(len(df['smoker'].unique()))
sum_of_smokers=len(df['smoker'])
non_smokers=0
smokers=0
for x in df['smoker']:
    if(x==0):
        smokers +=1
    elif(x == 1):
        non_smokers +=1
        
sm=[smokers,non_smokers]


# In[ ]:


percentage_of_smokers= "{0:.2f}".format((smokers/float(sum_of_smokers))*100)
percentage_of_non_smokers= "{0:.2f}".format((non_smokers/float(sum_of_smokers))*100)


# In[ ]:


plt.bar(p,sm,color = ['r','b'])
plt.xticks(p,["Non-Smokers","Smokers"])
plt.text(0, 500 ,percentage_of_smokers+'%',color='blue',horizontalalignment='center',verticalalignment='center')
plt.text(1, 500 ,percentage_of_non_smokers+'%',color='red',horizontalalignment='center',verticalalignment='center')
plt.show()


# There might be less number of smokers as seen in bar chart above but as the distribution shows below that the people how enjoy puffs are for sure paying alot more than health concious non-smokers.

# In[ ]:


import seaborn as sns
sns.distplot(df[(df.smoker == 1)]["charges"],color='r')
plt.title('Distribution of charges for Smokers')
plt.show()
sns.distplot(df[(df.smoker == 0)]["charges"],color='b')
plt.title('Distribution of charges for Non-Smokers')
plt.show()


# In[ ]:


df['age'].describe()


# The minimum age in our dataset is 18 while maximum is 64. Now this code below is to count the number of people belonging to 
# diferent age groups to see the which is the dominant age group in the dataset.

# In[ ]:


columns=['Intervals','count']
df_for_age=pd.DataFrame(0,index=np.arange(7),columns=columns)
df_for_age['Intervals']=df_for_age['Intervals'].astype(str)
n=18
p=25
i=0
while i<7:
    df_for_age['Intervals'][i] = str(n)+'-'+str(p) 
    n=p
    p=7+p
    i=i+1
    
for x in df['age']:
    if(x<=25):
        df_for_age.ix[0,'count']+=1
    elif(x<=32):
         df_for_age.ix[1,'count']+=1
    elif(x<=39):
         df_for_age.ix[2,'count']+=1
    elif(x<=46):
         df_for_age.ix[3,'count']+=1
    elif(x<=53):
         df_for_age.ix[4,'count']+=1
    elif(x<=60):
         df_for_age.ix[5,'count']+=1
    elif(x<=67):
         df_for_age.ix[6,'count']+=1


# In[ ]:


df_for_age


# The graph and the table above clearly shows that our dataset has alot of young people in their 20's. Now let see how much smoker and non-smoker people from young age group spend on treatment.
# 

# In[ ]:


sns.distplot(df.age)
plt.show()


# In[ ]:


plt.title("Box plot for charges 18-25 years old smokers")
sns.boxplot(y="smoker", x="charges", data = df[(df.age <= 25 )] , orient="h", palette = 'Set2')


# The boxplot clearly show that the people who are of age 18-25 and smoke pay way more for treatments than non-smoker with the exception of the outliers and it shows that payment made by non-smokers varies way less than non-smokers as it can be seen by th size of Quartile range.
# 
# Now we will see the corelation between the feature using heatmap 
# 

# In[ ]:


corelation=df.corr()
print(corelation)


# In[ ]:


sns.heatmap(corelation)
plt.show()


# As we can there strong corelation between smoker and charges where age and bmi are positively corelated but the value less.
# So Now we will these features for Regression analysis

# In[ ]:


import statsmodels.api as stats

y=df['charges']
X=df[['age','smoker','bmi']]
est=stats.OLS(y,X).fit()


# In[ ]:


est.summary()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y ,test_size=0.20,random_state=42)
dtr=DecisionTreeRegressor(max_depth=4)
dtr.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error
print('MSE for training data: ',mean_squared_error(y_train,dtr.predict(X_train)))
print('MSE for testing data: ',mean_squared_error(y_test,dtr.predict(X_test)))
print('R^2 for training data: ',r2_score(y_train, dtr.predict(X_train)))
print('R^2 for testing data: ',r2_score(y_test, dtr.predict(X_test)))


# In[ ]:


import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=4)
xgb.fit(X_train,y_train)


# In[ ]:


print('MSE for training data: ',mean_squared_error(y_train,xgb.predict(X_train)))
print('MSE for testing data: ',mean_squared_error(y_test,xgb.predict(X_test)))
print('R^2 for training data: ',r2_score(y_train, xgb.predict(X_train)))
print('R^2 for testing data: ',r2_score(y_test, xgb.predict(X_test)))


# Our accuracy is around 86%. I think it is close to an optimal solution.
