#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Forecasting using ElasticNet 

# ## Importing Libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import shuffle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")


# In[ ]:


print(df.shape,"\n",df.head())


# ### Some countries where the provinces are not mentioned can be replaced .And some data preprocessing 

# In[ ]:


df["Province_State"].fillna("state", inplace = True)    
df["Country_Region"] = [country_name.replace("'","") for country_name in df["Country_Region"]]
print(df.shape,"\n",df.head())


# ## Dataset preparation

# ### Considering the past 7 days data to forecast the cases and fatalities on the 8th day

# In[ ]:


data=[]
countries=df.Country_Region.unique()
for country in countries:
    provinces=df[df.Country_Region==country].Province_State.unique()
    for province in provinces:
        temp_df=df[(df['Country_Region'] == country) & (df['Province_State']==province)]
        for i in range(0,77):
            Iday1=float(temp_df.iloc[i].ConfirmedCases)
            Iday2=float(temp_df.iloc[i+1].ConfirmedCases)
            Iday3=float(temp_df.iloc[i+2].ConfirmedCases)
            Iday4=float(temp_df.iloc[i+3].ConfirmedCases)
            Iday5=float(temp_df.iloc[i+4].ConfirmedCases)
            Iday6=float(temp_df.iloc[i+5].ConfirmedCases)
            Iday7=float(temp_df.iloc[i+6].ConfirmedCases)
            Fday1=float(temp_df.iloc[i].Fatalities)
            Fday2=float(temp_df.iloc[i+1].Fatalities)
            Fday3=float(temp_df.iloc[i+2].Fatalities)
            Fday4=float(temp_df.iloc[i+3].Fatalities)
            Fday5=float(temp_df.iloc[i+4].Fatalities)
            Fday6=float(temp_df.iloc[i+5].Fatalities)
            Fday7=float(temp_df.iloc[i+6].Fatalities)
            target_infection=float(temp_df.iloc[i+7].ConfirmedCases)
            target_fatal=float(temp_df.iloc[i+7].Fatalities)
            data.append({"Iday1":Iday1,"Iday2":Iday2,"Iday3":Iday3,"Iday4":Iday4,"Iday5":Iday5,
                         "Iday6":Iday6,"Iday7":Iday7,"Fday1":Fday1,"Fday2":Fday2,"Fday3":Fday3,
                         "Fday4":Fday4,"Fday5":Fday5,"Fday6":Fday6,"Fday7":Fday7,
                         "target_infection":target_infection,"target_fatal":target_fatal})        


# In[ ]:


new_data=pd.DataFrame(data)
print("The shape of new dataFrame:",new_data.shape,"\nThe columns are:",new_data.columns)
print(new_data.head(-5))


# ## Splitting the data into Train and test data

# In[ ]:


X_y=shuffle(new_data)
y_cases=X_y['target_infection']
y_fatal=X_y['target_fatal']
X=X_y.drop(['target_infection','target_fatal'],axis=1)
X_train_cases, X_test_cases, y_train_cases, y_test_cases = train_test_split(X, y_cases, test_size=0.33)
X_train_fatal, X_test_fatal, y_train_fatal, y_test_fatal = train_test_split(X, y_fatal, test_size=0.33)
print("Shape of infection train dataset:",(X_train_cases.shape,y_train_cases.shape))
print("Shape of infection test dataset:",(X_test_cases.shape,y_test_cases.shape))
print("Shape of fatal train dataset:",(X_train_fatal.shape,y_train_fatal.shape))
print("Shape of fatal test dataset:",(X_test_fatal.shape,y_test_fatal.shape))


# <h2>Training the Infection data using Elastic Net 

# ### RandomSearch is been done to find out the best parameter
# #### Note:Scaling didn't yield good output hence proceeded with the original data 
#     

# In[ ]:


reg_case=ElasticNet(random_state=42,l1_ratio=0.1,max_iter=2200)
params = [{'alpha': [10**-4,10**-3, 10**-2,10**-1, 10**0,10**1, 10**2,10**3,10**4]}]
clf = RandomizedSearchCV(reg_case, params, cv=4, scoring='neg_root_mean_squared_error',return_train_score=True)
search=clf.fit(X_train_cases, y_train_cases)
results = pd.DataFrame.from_dict(clf.cv_results_)


# In[ ]:


best_alpha=1000
best_itr=2400
final_reg_case=ElasticNet(random_state=42,alpha=best_alpha,l1_ratio=0.1,max_iter=best_itr)
final_reg_case.fit(X_train_cases,y_train_cases)


# #### Calculating the Root mean squared value .Since the data isn't noramlized we get a large value

# In[ ]:


pred=final_reg_case.predict(X_test_cases)
print("The RMSE value",(mean_squared_error(y_test_cases,pred))**0.5)


# ## Training the fatality data

# ### The same procedure is been followed on this data also

# In[ ]:


reg_fatal=ElasticNet(random_state=42,l1_ratio=0.1,max_iter=3500)
params = [{'alpha': [10**-4,10**-3, 10**-2,10**-1, 10**0,10**1, 10**2,10**3,10**4]}]
clf = RandomizedSearchCV(reg_fatal, params, cv=4, scoring='neg_root_mean_squared_error',return_train_score=True)
search=clf.fit(X_train_fatal, y_train_fatal)
results = pd.DataFrame.from_dict(clf.cv_results_)


# In[ ]:


best_alpha=100
best_iter=3500
final_reg_fatal = ElasticNet(random_state=42,alpha=best_alpha,l1_ratio=0.1,max_iter=best_iter)
final_reg_fatal.fit(X_train_fatal, y_train_fatal)


# In[ ]:


pred=final_reg_fatal.predict(X_test_fatal)
print("The RMSE value",(mean_squared_error(y_test_fatal,pred))**0.5)


# ## Featurization

# ### Considering the fact that the number of cases on a given day is influenced by the ratio of cases in the two previous days.Taking this into account we create two new features 'iratio' and 'fratio'

# In[ ]:


data=[]
countries=df.Country_Region.unique()
for country in countries:
    provinces=df[df.Country_Region==country].Province_State.unique()
    for province in provinces:
        temp_df=df[(df['Country_Region'] == country) & (df['Province_State']==province)]
        for i in range(0,77):
            Iday1=float(temp_df.iloc[i].ConfirmedCases)
            Iday2=float(temp_df.iloc[i+1].ConfirmedCases)
            Iday3=float(temp_df.iloc[i+2].ConfirmedCases)
            Iday4=float(temp_df.iloc[i+3].ConfirmedCases)
            Iday5=float(temp_df.iloc[i+4].ConfirmedCases)
            Iday6=float(temp_df.iloc[i+5].ConfirmedCases)
            Iday7=float(temp_df.iloc[i+6].ConfirmedCases)
            Fday1=float(temp_df.iloc[i].Fatalities)
            Fday2=float(temp_df.iloc[i+1].Fatalities)
            Fday3=float(temp_df.iloc[i+2].Fatalities)
            Fday4=float(temp_df.iloc[i+3].Fatalities)
            Fday5=float(temp_df.iloc[i+4].Fatalities)
            Fday6=float(temp_df.iloc[i+5].Fatalities)
            Fday7=float(temp_df.iloc[i+6].Fatalities)
            if Iday6==0 :
                iavg=1
            else:
                iavg=Iday7/(Iday6)
            if Fday6==0:
                favg=1
            else:    
                favg=Fday7/(Fday6)        
            target_infection=float(temp_df.iloc[i+7].ConfirmedCases)
            target_fatal=float(temp_df.iloc[i+7].Fatalities)
            data.append({"Iday1":Iday1,"Iday2":Iday2,"Iday3":Iday3,"Iday4":Iday4,"Iday5":Iday5,
                         "Iday6":Iday6,"Iday7":Iday7,"Fday1":Fday1,"Fday2":Fday2,"Fday3":Fday3,
                         "Fday4":Fday4,"Fday5":Fday5,"Fday6":Fday6,"Fday7":Fday7,'iratio':iavg,"fratio":favg,"target_infection":target_infection,"target_fatal":target_fatal})        


# ### All other procedures remain the same

# In[ ]:


featured=pd.DataFrame(data)
X_y_f=shuffle(featured)
y_cases_f=X_y_f['target_infection']
y_fatal_f=X_y_f['target_fatal']
X_f=X_y_f.drop(['target_infection','target_fatal'],axis=1)
X_train_cases_f, X_test_cases_f, y_train_cases_f, y_test_cases_f = train_test_split(X_f, y_cases_f, test_size=0.33)
X_train_fatal_f, X_test_fatal_f, y_train_fatal_f, y_test_fatal_f = train_test_split(X_f, y_fatal_f, test_size=0.33)
print("Shape of featurized infection train dataset:",(X_train_cases_f.shape,y_train_cases_f.shape))
print("Shape of featurized infection test dataset:",(X_test_cases_f.shape,y_test_cases_f.shape))
print("Shape of featurized fatal train dataset:",(X_train_fatal_f.shape,y_train_fatal_f.shape))
print("Shape of featurized fatal test dataset:",(X_test_fatal_f.shape,y_test_fatal_f.shape))


# In[ ]:


reg_case_f=ElasticNet(random_state=42,l1_ratio=0.1,max_iter=2200)
params = [{'alpha': [10**-4,10**-3, 10**-2,10**-1, 10**0,10**1, 10**2,10**3,10**4]}]
clf_f= RandomizedSearchCV(reg_case_f, params, cv=4, scoring='neg_root_mean_squared_error',return_train_score=True)
search_f=clf_f.fit(X_train_cases_f, y_train_cases_f)
results_f = pd.DataFrame.from_dict(clf_f.cv_results_)


# In[ ]:


best_alpha=10000
best_itr=4200
final_reg_case_f=ElasticNet(random_state=42,alpha=best_alpha,l1_ratio=0.1,max_iter=best_itr)
final_reg_case_f.fit(X_train_cases_f,y_train_cases_f)


# In[ ]:


pred_f=final_reg_case_f.predict(X_test_cases_f)
print("RMSE is:",(mean_squared_error(y_test_cases_f,pred_f))**0.5)


# In[ ]:


reg_fatal_f=ElasticNet(random_state=42,alpha=best_alpha,l1_ratio=0.1,max_iter=2200)
params = [{'alpha': [10**-4,10**-3, 10**-2,10**-1, 10**0,10**1, 10**2,10**3,10**4]}]
clf_f= RandomizedSearchCV(reg_fatal_f, params, cv=4, scoring='neg_root_mean_squared_error',return_train_score=True)
search_f=clf_f.fit(X_train_fatal_f, y_train_fatal_f)
results_f = pd.DataFrame.from_dict(clf_f.cv_results_)


# In[ ]:


best_alpha=100
best_itr=2400
final_reg_fatal_f=ElasticNet(random_state=42,alpha=best_alpha,l1_ratio=0.1,max_iter=best_itr)
final_reg_fatal_f.fit(X_train_fatal_f,y_train_fatal_f)


# In[ ]:


pred_f=final_reg_fatal_f.predict(X_test_fatal_f)
print("RMSE is:",(mean_squared_error(y_test_fatal_f,pred_f))**0.5)


# ## Forecasting the number of cases and fatalities

# In[ ]:


test["Province_State"].fillna("state", inplace = True)    
test["Country_Region"] = [country_name.replace("'","") for country_name in test["Country_Region"]]


# ### Method used :
# ### 1)Intialize the list with the data of previous seven days and predict the value for the next day.
# ### 2)Append this value to the list and use this updated list's latest data to predict the next one.(So on........)
# 

# In[ ]:


import math
import random
predicted_case=[]
predicted_fatal=[]
countries=df.Country_Region.unique()
for country in countries:
    provinces=df[df.Country_Region==country].Province_State.unique()
    for province in provinces:
        temp_df=df[(df['Country_Region'] == country) & (df['Province_State']==province)&(df['Date']>='2020-04-02')]
        ongoingCases=list(temp_df.ConfirmedCases.values)
        ongoingFatal=list(temp_df.Fatalities.values)
        predicted_case.extend(ongoingCases)
        predicted_fatal.extend(ongoingFatal)
        for _ in range(1,31):  
            if ongoingCases[-2]==0:
                iavg=ongoingCases[-1]
            else:
                iavg=ongoingCases[-1]/ongoingCases[-2]
            if ongoingFatal[-2]==0:
                favg=ongoingFatal[-1]
            else:    
                favg=ongoingFatal[-1]/ongoingFatal[-2] 
            point=ongoingCases[len(ongoingCases)-7:]+ongoingFatal[len(ongoingFatal)-7:]+[iavg,favg]
            # print(point)
            # print()
            randF=random.random()
            randI=random.random()
            predC=final_reg_case_f.predict([point])
            predF=final_reg_fatal_f.predict([point])
            predicted_case.append(int(predC[0]-(randI*predC[0]*0.002)))
            predicted_fatal.append(abs(int(predF[0]-(randF*predF[0]*0.0005))))
            ongoingCases.append(predC[0]-(randI*predC[0]*0.002))
            ongoingFatal.append(abs(predF[0]-(randF*predF[0]*0.0005)))


# ## Updating the values in the test data

# In[ ]:


test['ConfirmedCases']=list(map(int,predicted_case))
test['Fatalities']=list(map(int,predicted_fatal))


# In[ ]:


submission=test[['ForecastId','ConfirmedCases','Fatalities']]
submission=shuffle(submission)
submission.to_csv("submission.csv",index=False)

