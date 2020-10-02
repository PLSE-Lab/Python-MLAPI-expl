#!/usr/bin/env python
# coding: utf-8

# ## In the FIRST half of this Kernel we will try to VISUALIZE the Data and try to find out some Important Points which will be Crutial for a CANDIDATE to know before going for a JOB INTERVIEW.
# 
# ## In the SECOND half, we will create a MODEL and try to find out through that MODEL that whether a candidate can get a JOB or NOT.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

sns.set_style('whitegrid')
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()


# In[ ]:


def getting_ranges(df):
    
    bins_ssc = (39,50,60,70,80,90)
    group_names_ssc = ['40-50','50-60','60-70','70-80','80-90']
    
    bins_hsc = (29,40,50,60,70,80,90,100)
    group_names_hsc = ['30-40','40-50','50-60','60-70','70-80','80-90','90-100'] 
    
    bins_degree = (49,60,70,80,90,100)
    group_names_degree = ['50-60','60-70','70-80','80-90','90-100']
    
    bins_etest = (49,60,70,80,90,100)
    group_names_etest = ['50-60','60-70','70-80','80-90','90-100'] 
    
    bins_mba = (49,60,70,80)
    group_names_mba = ['50-60','60-70','70-80']
    
    bins_salary = (199999,300000,400000,500000,600000,700000,800000,900000,1000000)
    group_names_salary = ['2 Lakh-3 Lakh','3 Lakh-4 Lakh','4 Lakh-5 Lakh','5 Lakh-6 Lakh','6 Lakh-7 Lakh','7 Lakh-8 Lakh','8 Lakh-9 Lakh','9 Lakh-10 Lakh']    
  
    df['ssc_perc_range'] = pd.cut(df.ssc_p,bins_ssc,labels=group_names_ssc)
    df['hsc_perc_range'] = pd.cut(df.hsc_p,bins_hsc,labels=group_names_hsc)
    df['degree_perc_range'] = pd.cut(df.degree_p,bins_degree,labels=group_names_degree)
    df['etest_perc_range'] = pd.cut(df.etest_p,bins_etest,labels=group_names_etest)
    df['mba_perc_range'] = pd.cut(df.mba_p,bins_mba,labels=group_names_mba)
    df['salary_range'] = pd.cut(df.salary,bins_salary,labels=group_names_salary)


# In[ ]:


getting_ranges(df)


# In[ ]:


pd.set_option('max_columns',50)
df.head()


# # Let's see that what impact Candidates *PERCENTAGE OBTAINED* from Secondary School till MBA had on their Placement :

# # 1. SECONDARY EDUCATION

# In[ ]:


plt.figure(figsize=(9,5))
sns.countplot(data=df,x='ssc_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')


# # CONCLUSION :
# 
# ## 1. The Candidates having percentage 80+ in Secondary Education have 100% Placement Record.
# ## 2. The Candidates having percentage between 70-80 in Secondary Education have 90% Placement Record.
# ## 3. The Candidates having percentage below 60 have VERY LESS CHANCES of getting Placed. They need to work really HARD !!

# # 2. HIGHER EDUCATION

# In[ ]:


plt.figure(figsize=(9,5))
sns.countplot(data=df,x='hsc_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')


# # CONCLUSION :
# 
# ## 1. The Candidates having percentage 90+ in Higher Education have 100% Placement Record.
# ## 2. The Candidates having percentage between 70 and 90 in Higher Education have 90% Placement Record.
# ## 3. The Candidates having percentage between 60 and 70 also have a fair chance to get Placed.
# ## 4. The Candidates having percentage below 50 in Higher Education have 0% chance of getting Placed. So, to get Placed you should be having minimun of 50 percentage in Higher Education

#  # 3. GRADUATION

# In[ ]:


plt.figure(figsize=(9,5))
sns.countplot(data=df,x='degree_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')


# # CONCLUSION :
# 
# ## 1. The Candidates having percentage 80+ in GRADUATION have 100% Placement Record.
# ## 2. The Candidates having percentage between 70 and 80 in Higher Education have 90% Placement Record.
# ## 3. The Candidates having percentage between 60 and 70 also have a fair chance to get Placed.
# ## 4. The Candidates having percentage below 60 are having very LESS chance of getting Placed. They need to work really hard !

# # 4. MBA

# In[ ]:


plt.figure(figsize=(9,5))
sns.countplot(data=df,x='mba_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')


# # CONCLUSION :
# 
# ## In MBA, we haven't seen any strong relation between Percentage Obtained and Placement. Anyone who is having percentage 50+ in MBA can get Placed but still it seems like if a Candidate has 70+ perecntage he would be having a fairly Good Chance of getting Placed.

# # 5. ENTRANCE EXAM

# In[ ]:


plt.figure(figsize=(9,5))
sns.countplot(data=df,x='etest_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')


# # CONCLUSION :
# 
# ## There is no such strong relation found between the Entrance Test and Placement, but it seems a Candidate having marks 80+ has a slight edge over the candidates having marks below 70.

# # Should WORK EXPERIENCE also be a factor for getting a JOB to a CANDIDATE ??

# In[ ]:


plt.figure(figsize=(9,5))
sns.countplot(data=df,x='workex',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')


# # CONCLUSION :
# 
# ## It seems that the Candidates having WORK EXPERIENCE are having very STRONG CHANCES of getting placed as compared to Freshers.

# # How much Companies are Paying Off to HIRE the CANDIDATES ??

# In[ ]:


df.salary_range.value_counts().plot.bar(
figsize=(16,5),
color = 'green',
fontsize =14 
)


# # CONCLUSION :
# 
# ## It seems that most of the Companies are offering Packages ranging from 2 Lakh-3 Lakh to hire the Candidates and very few are willing to Pay 3Lakh+ Package to the Candidates.

# # MACHINE LEARNING - Let's Prepare a MODEL which can be Trained on this DATA and Later can predict that whether a New Applicant applying for a JOB can get it or not

# In[ ]:


data_ml = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data_ml.drop(['sl_no','gender','ssc_b','hsc_b'],inplace=True,axis=1)


# In[ ]:


data_ml.head()


# ### As Machine Learning Algorithm takes only NUMERIC features, So let's do some Feature Engineering and convert Columns having String values to Numeric form.

# In[ ]:


hsc_stream = pd.get_dummies(data_ml.hsc_s,prefix='hsc_stream',drop_first=True)
hsc_stream.head()


# In[ ]:


degree_trade = pd.get_dummies(data_ml.degree_t,prefix='degree_trade',drop_first=True)
degree_trade.head()


# In[ ]:


workex = pd.get_dummies(data_ml.workex,prefix='workex',drop_first=True)
workex.head()


# In[ ]:


specialisation = pd.get_dummies(data_ml.specialisation,prefix='specialisation',drop_first=True)
specialisation.head()


# In[ ]:


data_ml = pd.concat([data_ml,hsc_stream,degree_trade,workex,specialisation],axis=1)


# In[ ]:


data_ml.drop(['hsc_s','degree_t','workex','specialisation'],axis=1,inplace=True)


# In[ ]:


data_ml.head()


# In[ ]:


X = data_ml.drop(['status','salary'],axis=1)


# In[ ]:


X.head()


# In[ ]:


y = data_ml['status']
y.head()


# ### We will be using LOGISTIC REGRESSION Algorithm in this case for training the Data. You can try all the Models and choose the Best Performing MODEL out of all.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split


# In[ ]:


Log_model = LogisticRegression()


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20)

Log_model.fit(X_train,y_train)
y_pred = Log_model.predict(X_test)
print('Accuracy -> ',accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix : \n',confusion_matrix(y_test,y_pred))


# ### This accuracy can Fluctuate everytime we run this MODEL on the basis of the random selection of Train and Test data to train and test this Model.
# 
# ### So, let's use CROSS VALIDATION SCORE to get the mean value of Accuracy for this MODEL :

# In[ ]:


print('Accuracy -> ',cross_val_score(Log_model,X,y,cv=10).mean()*100,'%')


# ## Now, Let's try increasing the Accuracy of this MODEL by Tuning its Hyperparameters :

# In[ ]:


parameters = [{'penalty' : ['l2','l1'],
               'C' : np.logspace(0, 4, 10),
               'class_weight' : ['balanced',None],
               'multi_class' : ['ovr','auto'],
               'max_iter' : np.arange(50,130,10)},
              {'penalty' : ['l2'],
               'C' : np.logspace(0, 4, 10),
               'class_weight' : ['balanced',None],
               'max_iter' : np.arange(50,130,10),
               'solver' : ['newton-cg','saga','sag','liblinear'],
               'multi_class' : ['ovr','auto']}]


# In[ ]:


Log_grid = GridSearchCV(estimator=Log_model,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
Log_grid.fit(X,y)


# ### The BEST PARAMETERS obtained for Logistic Regression MODEL are :

# In[ ]:


Log_grid.best_params_


# #### Let's create a MODEL by using these HYPERARAMETER values :

# In[ ]:


Log_model_grid = LogisticRegression(C= 10000.0,
 class_weight= 'balanced',
 max_iter= 90,
 multi_class= 'ovr',
 penalty= 'l2')


# In[ ]:


print('Accuracy after doing HYPERPARAMETER Tuning -> ',cross_val_score(Log_model_grid,X,y,cv=10).mean()*100,'%')


# ### Now, as we can see their is Slight Increase in Accuracy after doing Hyperparameter Tuning.

# In[ ]:


Log_model_grid.fit(X,y)


# ## Let's Take 1 Candidate who has applied for a JOB and try to see if he will get PLACED or NOT according to our MODEL :
# 
# ### Here are the Details of the candidate who has applied for a JOB :
# 
# 1. ssc_p(Percentage Obtained in SECONDARY SCHOOL)                          = 80
# 2. hsc_p(Percentage Obtained in HIGH SCHOOL)                               = 75
# 3. degree_p(Percentage Obtained in GRADUATION)                             = 65
# 4. etest_p(Marks Obtained in ENTRANCE TEST)	                               = 64
# 5. mba_p(Marks Obtained in MBA)                                            = 63
# 6. hsc_stream_Commerce(Candidate is NOT from COMMERCE STREAM)              = 0
# 7. hsc_stream_Science(Candidate is from SCIENCE STREAM)                    = 1
# 8. degree_trade_Others(Candidate is NOT from OTHERS Degree Trade)          = 0
# 9. degree_trade_Sci&Tech(Candidate did GRADUATION in SCIENCE & TECHNOLOGY) = 1
# 10. workex_Yes(Candidate is having WORK EXPERIENCE)                        = 1	
# 11. specialisation_Mkt&HR(Candidate hasn't done specialization in Mkt&HR)  = 0

# In[ ]:


model_prediction = Log_model_grid.predict([[80,75,65,64,63,0,1,0,1,1,0]])
print('According to our MODEL, this CANDIDATE will get -> ',model_prediction)


# ## So, on seeing all his details our MODEL predicts that this CANDIDATE will get PLACED.
# 
# In the Similar way you can try running this MODEL by passing different values for the Candidate applying for the JOB and see if the person can get PLACED or Not.

# Thanks for watching this Kernel :)
