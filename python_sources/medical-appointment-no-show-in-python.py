#!/usr/bin/env python
# coding: utf-8

# <h1><center>Investigating on <br> 
#    Medical Appointment No Show Dataset</center></h1>

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data-Wrangling" data-toc-modified-id="Data-Wrangling-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Wrangling</a></span><ul class="toc-item"><li><span><a href="#import-required-python-libraries" data-toc-modified-id="import-required-python-libraries-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>import required python libraries</a></span></li><li><span><a href="#Read-table-and-cleaning-the-data" data-toc-modified-id="Read-table-and-cleaning-the-data-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Read table and cleaning the data</a></span></li></ul></li><li><span><a href="#Exploratory-Analysis" data-toc-modified-id="Exploratory-Analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploratory Analysis</a></span><ul class="toc-item"><li><span><a href="#Age-vs-No-Show" data-toc-modified-id="Age-vs-No-Show-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Age vs No Show</a></span><ul class="toc-item"><li><span><a href="#Is-there-any-relation-between-Age-and-No-show?" data-toc-modified-id="Is-there-any-relation-between-Age-and-No-show?-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Is there any relation between Age and No show?</a></span></li><li><span><a href="#plot-and-statistics" data-toc-modified-id="plot-and-statistics-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>plot and statistics</a></span></li><li><span><a href="#t-test" data-toc-modified-id="t-test-2.1.3"><span class="toc-item-num">2.1.3&nbsp;&nbsp;</span>t test</a></span></li></ul></li><li><span><a href="#Gender-vs-No-Show" data-toc-modified-id="Gender-vs-No-Show-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Gender vs No Show</a></span><ul class="toc-item"><li><span><a href="#Is-there-any-relation-between-Gender-and-No-show?" data-toc-modified-id="Is-there-any-relation-between-Gender-and-No-show?-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Is there any relation between Gender and No show?</a></span></li><li><span><a href="#plot-and-statistics" data-toc-modified-id="plot-and-statistics-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>plot and statistics</a></span></li><li><span><a href="#chi-squared-test" data-toc-modified-id="chi-squared-test-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>chi-squared test</a></span></li></ul></li><li><span><a href="#Scholarship,-Hypertension,-Diabetes,-Alcohol,-Handicap,-sms-received-vs-No-show" data-toc-modified-id="Scholarship,-Hypertension,-Diabetes,-Alcohol,-Handicap,-sms-received-vs-No-show-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Scholarship, Hypertension, Diabetes, Alcohol, Handicap, sms-received vs No show</a></span><ul class="toc-item"><li><span><a href="#What-are-the-relation-of--No-show-with-Scholarship,-Hypertension,-Diabetes,-Alcohol,-Handicap,-and-sms-received-?" data-toc-modified-id="What-are-the-relation-of--No-show-with-Scholarship,-Hypertension,-Diabetes,-Alcohol,-Handicap,-and-sms-received-?-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>What are the relation of  No show with Scholarship, Hypertension, Diabetes, Alcohol, Handicap, and sms-received ?</a></span></li><li><span><a href="#plot-by-looping" data-toc-modified-id="plot-by-looping-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>plot by looping</a></span></li><li><span><a href="#chi-squared-test" data-toc-modified-id="chi-squared-test-2.3.3"><span class="toc-item-num">2.3.3&nbsp;&nbsp;</span>chi-squared test</a></span></li></ul></li><li><span><a href="#Appointment-day-vs-No-Show" data-toc-modified-id="Appointment-day-vs-No-Show-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Appointment day vs No Show</a></span><ul class="toc-item"><li><span><a href="#plot" data-toc-modified-id="plot-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>plot</a></span></li><li><span><a href="#chi-squared-test" data-toc-modified-id="chi-squared-test-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>chi-squared test</a></span></li></ul></li><li><span><a href="#Neighbor-vs-No-Show" data-toc-modified-id="Neighbor-vs-No-Show-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Neighbor vs No Show</a></span><ul class="toc-item"><li><span><a href="#Plot" data-toc-modified-id="Plot-2.5.1"><span class="toc-item-num">2.5.1&nbsp;&nbsp;</span>Plot</a></span></li></ul></li></ul></li><li><span><a href="#Logistic-Regression-Model" data-toc-modified-id="Logistic-Regression-Model-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Logistic Regression Model</a></span><ul class="toc-item"><li><span><a href="#Logistics-Regression-using-statsmodel" data-toc-modified-id="Logistics-Regression-using-statsmodel-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Logistics Regression using statsmodel</a></span></li><li><span><a href="#Logistic-Regression-using-scikit-learn" data-toc-modified-id="Logistic-Regression-using-scikit-learn-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Logistic Regression using scikit-learn</a></span><ul class="toc-item"><li><span><a href="#Split-the-data" data-toc-modified-id="Split-the-data-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Split the data</a></span></li><li><span><a href="#Build-the-model" data-toc-modified-id="Build-the-model-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Build the model</a></span></li><li><span><a href="#Predictions" data-toc-modified-id="Predictions-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>Predictions</a></span></li><li><span><a href="#Evaluation" data-toc-modified-id="Evaluation-3.2.4"><span class="toc-item-num">3.2.4&nbsp;&nbsp;</span>Evaluation</a></span></li></ul></li></ul></li><li><span><a href="#Conclusions" data-toc-modified-id="Conclusions-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Conclusions</a></span></li></ul></div>

# ********

# Same project in **R**   [click here](https://www.kaggle.com/yousuf28/medical-appointment-no-show-in-r)

# If you want see same plot in **Tableau** <br> and want to see some **SAS** code 
# go to my GitHub page [GitHub](https://github.com/Yousuf28/udacity_data_analyst_nano_degree/blob/master/README.md)

# ## Data Wrangling

# ### import required python libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# figure set up
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# Following code for not to scroll figure, figure will be fixed in place.

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines){\n    return false;\n}')


# ### Read table and cleaning the data

# In[ ]:


#read the data
df = pd.read_csv('../input/KaggleV2-May-2016.csv')


# In[ ]:


# check the data
df.head()


# In[ ]:


# check structure of the data
df.info()


# In[ ]:


# change columns name

new_col_name = ['patient_id', 'appointment_id','gender','schedule_day','appointment_day','age','neighborhood',
               'scholarship','hypertension','diabetes','alcoholism','handicap',
               'sms_received','no_show']
df.columns = new_col_name


# In[ ]:


# check again
df.head(2)


# In[ ]:


# check missing value
df.isnull().sum()


# There is no missing value in data set. That's might not be case always.

# In[ ]:


# change data type of some columns
df['patient_id'] = df['patient_id'].astype('int64')
df['schedule_day']= pd.to_datetime(df['schedule_day'])
df['appointment_day']= pd.to_datetime(df['appointment_day'])


# In[ ]:


df.dtypes


# In[ ]:


# check summary statistics
df.describe()


# In[ ]:


# statistics for all columns
# df.describe(include= 'all');


# minimum age is $-1$ , so lets check all the value less than $0$

# In[ ]:


df[df['age']< 0]


# So there is one row that contain age less than $0$. So lets drop that row.

# In[ ]:


# drop row with condition
df.drop(df[df['age'] < 0].index, inplace =True)


# ****

# 
# ## Exploratory Analysis

# ### Age vs No Show

# #### Is there any relation between Age and No show?

# #### plot and statistics

# In[ ]:


# create mask 
showed_up = df['no_show'] == 'No'
not_showed_up = df['no_show'] == 'Yes'


# In the dataset description, it mention that in __no_show__ column **No** means patient showed up and **yes** means patient did not showed up.

# In[ ]:


# age versus showed up or not

df.age[showed_up].hist(label = 'showed up', alpha = 0.5, bins = 40)
df.age[not_showed_up].hist(label = 'not showed up', alpha = 0.5, bins =40)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age vs No Show Histogram')
plt.legend();


# It is very hard to tell from this plot the difference in age of patient who showed up or did not showed up. So let's explore more.

# Since there is confusion with no and yes with showed up or not, lets make it clear

# In[ ]:


# in no_show column No means patient showed up and yes means patient did not showed up
replacement = { 'No': 'Showed up', 
              'Yes': 'not showed up'}
df['no_show'].replace(replacement, inplace = True)


# In[ ]:


sns.boxplot(x ='no_show', y= 'age', data =df);
plt.title('Age vs No Show Boxplot')
plt.xlabel('No Show')
plt.ylabel('Age');


# From the box plot, the mean age is higher for those who showed up. Now let's see the exact value.

# In[ ]:


df.groupby('no_show')['age'].mean()


# In[ ]:


# more visulization


# Though it might not be appropriate for all case but in this case it is possible to visualize all age group.

# I have make contingency table below using  crosstab function. crosstab is like table function in R and when you use normalize parameter that's like prop.table function in R.

# In[ ]:


age_noshow = pd.crosstab(index = df['age'],
                        columns = df['no_show'], normalize = 'index')
age_noshow.head()


# In[ ]:


age_noshow.plot(kind = 'bar', stacked = True, figsize=(12,6), color = ['skyblue', 'red'], rot= 90);
plt.xlabel('Age')
plt.ylabel('Proportion')
plt.title('Age vs NO Show proportion in Bar Diagram')
plt.legend();


# From the above plot now it seems that proportion of showed up patients are higher in the age range 60 to 80 than patient age under 40.

# #### t test

# In[ ]:


from scipy.stats import ttest_ind


# In[ ]:


## make separate coulumn
showed_up = df[df.no_show == 'Showed up'].age
not_showed_up = df[df.no_show != 'Showed up'].age


# In[ ]:


test = ttest_ind(showed_up, not_showed_up)


# In[ ]:


print('p value for t test is: {:.5f}'.format(test.pvalue))


# In[ ]:


# original value 
test.pvalue


# So there is significant difference in age of patient those who showed up and those who did not show up.

# ### Gender vs No Show

# #### Is there any relation between Gender and No show?

# #### plot and statistics

# In[ ]:


#replace label, just for make it clear
replacement_gender = { 'F': 'Female', 
              'M': 'Male'}
df['gender'].replace(replacement_gender, inplace = True)


# In[ ]:


# plot of gender
sns.countplot(df['gender']);
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Bar Diagram');


# So there are more female patient than male patient in dataset.

# In[ ]:


# table
gender_noshow = pd.crosstab(index=df["gender"],
            columns=df["no_show"])
gender_noshow


# In[ ]:


# total pateints
df.no_show.value_counts()


# In[ ]:


# plot 
gender_noshow.plot(kind='bar',stacked = True)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender vs No Show Stacked Bar Diagram');


# showed up and not showed up both group is higher for female than male because more patients in data set is female. So its better to see proportionally.

# In[ ]:


## if you want to see propotionally
pd.crosstab(index=df["gender"], 
            columns=df["no_show"], normalize = 'index')


# In[ ]:


# plot of proportion data
pd.crosstab(index=df["gender"], 
            columns=df["no_show"], normalize = 'index').plot(kind='bar', 
                                     stacked = True);
plt.xlabel('Gender')
plt.ylabel('Proportion')
plt.title('Gender vs No Show Bar Diagram')
plt.legend(loc = 9);


# So there is not much difference between male and female.

# There is other way you can make contingency table, mostly I use crosstab but using groupby and unstack is faster than crosstab.

# In[ ]:



gender_noshow = pd.crosstab(index=df["gender"],
            columns=df["no_show"])
gender_noshow


# In[ ]:



df.groupby('gender')['no_show'].value_counts().unstack()


# #### chi-squared test

# In[ ]:


from scipy.stats import chi2_contingency


# In[ ]:


chi2, p, dof,ex = chi2_contingency(gender_noshow, correction=False)


# In[ ]:


print('chi-squared value is: {0:6.3f}'.format(chi2))


# In[ ]:


print('p value is: {0:6.3f}'.format(p))


# p value is more than 0.05, so gender difference is not significant.

# ### Scholarship, Hypertension, Diabetes, Alcohol, Handicap, sms-received vs No show

# #### What are the relation of  No show with Scholarship, Hypertension, Diabetes, Alcohol, Handicap, and sms-received ?

# #### plot by looping

# Number of the patients count in each group. For example hypertension,<br>
# 1 mean have hypertension and<br>
# 0 mean does not have hypertension.

# In[ ]:


column_plot = ['scholarship', 
               'hypertension',
               'diabetes', 
               'alcoholism', 
               'handicap', 
               'sms_received']


# In[ ]:


# count patient number in each group
fig = plt.figure(figsize= (8,12))
for number, column in enumerate(column_plot):
    axes = fig.add_subplot(3,2, number+1)
    axes.set_xlabel(column.capitalize())
    axes.set_title('Bar Diagram for ' + column.capitalize())
    df[column].value_counts().plot(kind = 'bar', ax = axes, color = ['skyblue', 'orange'])
    plt.ylabel('Count')
plt.tight_layout()


# In[ ]:


# this plot for each group with patient who showed up and not showed up

column_plot = ['scholarship', 
               'hypertension',
               'diabetes', 
               'alcoholism', 
               'handicap', 
               'sms_received']

fig = plt.figure(figsize= (8,12))
for number, column in enumerate(column_plot):
    axes = fig.add_subplot(3,2, number+1)
    axes.set_title(column.capitalize() + ' vs No Show')
    pd.crosstab(index = df[column], columns = df['no_show']).plot(kind = 'bar',
                                                                           stacked = True, 
                                                                           ax = axes, 
                                                                          color = ['skyblue','orange'])
    axes.set_xlabel(column.capitalize())
    plt.ylabel('Count')
    
plt.tight_layout()


# In[ ]:


# this plot is same as above one execpt it show proportion.
column_plot = ['scholarship', 
               'hypertension',
               'diabetes', 
               'alcoholism', 
               'handicap', 
               'sms_received']

fig = plt.figure(figsize= (8,12))
for number, column in enumerate(column_plot):
    axes = fig.add_subplot(3,2, number+1)
    axes.set_title(column.capitalize() + ' vs No Show')
    pd.crosstab(index = df[column], columns = df['no_show'],normalize = 'index').plot(kind = 'bar',
                                                                           stacked = True, 
                                                                           ax = axes, 
                                                                          color = ['skyblue','orange'])
    
    axes.set_xlabel(column.capitalize())
    plt.ylabel('Proportion')
plt.tight_layout() 
plt.legend(loc = 3);


# From all the above plots,
# chance of being showed up is higher for those who<br>
# have no scholarship, <br>
# have hypertension, <br>
# have diabetes, <br>
# and did not received sms (last one clearly strange)

# #### chi-squared test

# In[ ]:


chi_test =    ['scholarship', 
               'hypertension',
               'diabetes', 
               'alcoholism', 
               'handicap', 
               'sms_received']

for column in (chi_test):
    chi2, p, dof,ex = chi2_contingency(pd.crosstab(index=df[column], columns=df["no_show"]), correction=False)
    print ('chi-squared test- p value for {} is: {r:6.5f}'. format(column,r = p))

    
    


# p value is significant for scholarship, hypertension, diabetes, sms_received group.

# ### Appointment day vs No Show

# #### plot

# In[ ]:


# make new column  
df['day'] = df.appointment_day.dt.weekday_name


# In[ ]:


# plot number of appointment weekday
df['day'].value_counts().plot(kind = 'bar');
plt.xlabel('Days')
plt.ylabel('Count')
plt.title('Number of Appointment');


# Number of appointment differ across week. Some day like Wednesday and Tuesday make more appointment than other. Statistics given below to see exact number.

# In[ ]:


days = pd.crosstab(index = df['day'],
           columns = df['no_show'])
days


# In[ ]:


days_p = pd.crosstab(index = df['day'],
           columns = df['no_show'], normalize = 'index').plot(kind = 'bar', stacked = True)


# From above plot and below statistics, proportion of patient who showed up are higher in Wednesday and Thursday than Friday and Monday. The big difference in Saturday is due to less number of sample compared to others day.

# In[ ]:


pd.crosstab(index = df['day'],
           columns = df['no_show'], normalize = 'index')


# #### chi-squared test

# In[ ]:


chi2, p, dof,ex = chi2_contingency(days, correction=False)


# In[ ]:


print ('p value is : {r:5.5f}'.format(r = p))


# Since p value is significant that means showing up in appointment day is dependent on which day the appointment is.

# ### Neighbor vs No Show

# ####  Plot

# In[ ]:


neighbor = pd.crosstab(index=df["neighborhood"], 
            columns=df["no_show"])
neighbor_sort = neighbor.sort_values('Showed up', ascending= False)


# In[ ]:


neighbor_sort.plot(kind='bar',figsize = (12,6), fontsize = 6,
                                     stacked = True)

plt.xlabel('Neighborhood')
plt.ylabel('Count')
plt.title('Neighborhood vs No Show')

plt.legend();


# There are few neighborhood from where more than $4000$ appointment made. almost Half of all neighborhood make appointment less than $1000%. There are many neighborhood so it is hard to tell which neighborhood miss their appointment most.

# ***

# ## Logistic Regression Model

# In[ ]:


# check the dataset
df.head()


# Drop patient_id, appointment_id, schedule_day, neighborhood

# In[ ]:


## drop columns
df.drop(['patient_id', 'appointment_id','schedule_day',
         'appointment_day','neighborhood'], axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


# copy data set
df_2 = df.copy()


# change columns to category type

# In[ ]:


column_plot = ['gender', 'scholarship', 
               'hypertension',
               'diabetes', 
               'alcoholism', 
               'handicap', 
               'sms_received',
               'day',
               'no_show']


# In[ ]:


# change columns to category type using for loop
for cl in column_plot:
    df_2[cl] = df_2[cl].astype('category')


# In[ ]:


# can be done manually

# df['gender'] = df['gender'].astype('category')
# df['scholarship'] = df['scholarship'].astype('category')
# df['hypertension'] = df['hypertension'].astype('category')
# df['diabetes'] = df['diabetes'].astype('category')
# df['alcoholism'] = df['alcoholism'].astype('category')
# df['handicap'] = df['handicap'].astype('category')
# df['sms_received'] = df['sms_received'].astype('category')
# df['day'] = df['day'].astype('category')
# df['no_show'] = df['no_show'].astype('category')



# ### Logistic Regression using statsmodel

# If logistic regression part not work see this link.
# [alternate link for same  notebook](http://nbviewer.jupyter.org/github/Yousuf28/udacity_data_analyst_nano_degree/blob/master/part_02_investigate_dataset/investigating_dataset_final.ipynb)

# In[ ]:


import statsmodels.api as sm


# In[ ]:


log_model_1 = sm.formula.glm("no_show ~ age+gender+scholarship+hypertension+diabetes+alcoholism+handicap+sms_received+day", 
                           family = sm.families.Binomial(), data = df_2).fit()


# In[ ]:


log_model_1.summary2()


# Drop handicap and gender and build a model again.

# In[ ]:


log_model_2 = sm.formula.glm("no_show ~ age+scholarship+hypertension+diabetes+alcoholism+sms_received+day", 
                           family = sm.families.Binomial(), data = df_2).fit()


# In[ ]:


log_model_2.summary2()


# In[ ]:


df_3 = df.copy()


# In[ ]:


# 1 for showed up, and 0 for not showed up
df_3['no_show'] = df_3.no_show.map({'Showed up':1, 'not showed up':0})


# ### Logistic Regression using scikit-learn

#  Create dummy variable

# In[ ]:


# create dummy variable and save this in df_1
df_4 = pd.get_dummies(df_3, columns = ['gender',  'handicap', 'day'], drop_first = True)


# In[ ]:


df_4.head()


# In[ ]:


# import required function form scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score


# #### Split the data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_4.drop('no_show',axis=1), 
                                                    df_4['no_show'], test_size=0.30, 
                                                    random_state=101)


# #### Build the model

# In[ ]:


logit_model = LogisticRegression()
logit_model.fit(X_train,y_train);


# #### Predictions

# In[ ]:


predictions = logit_model.predict(X_test)


# #### Evaluation

# In[ ]:



print(classification_report(y_test,predictions));


# In[ ]:


confusion_matrix(y_test, predictions)


# In[ ]:


accuracy_score(y_test, predictions)


# Model is not like as we expected because there is class imbalance in between showed up and not showed up group. <br>There are $80$ % patients who show up and $20$ % those who don't show up. <br>
# So if I predict someone will show up based on data (without model) that is $80$ %.

# In[ ]:


# percent of patient who show up and not show up from data
df.no_show.value_counts()/df.no_show.value_counts().sum()


# ***

# ## Conclusions

# If someone make appointment there is already $80$% chance that this patient will show up. <br>
# even though we saw some independent variable(predictors) like age, hypertension , diabetes, sms-received are significant predictor but our model fail to predict accuracy more than $80$% because of class imbalance.
# So from this dataset it is very hard to tell that who is not going to show up.<br>
# There are techniques (like under-sampling and over-sampling) available to solve class imbalance problem but that beyond this project outline.

# *****
