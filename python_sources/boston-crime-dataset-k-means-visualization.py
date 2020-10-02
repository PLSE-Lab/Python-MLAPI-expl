#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from mpl_toolkits.basemap import Basemap


# *If folium is not installed please use the following code to install the same* - ***!conda install -c conda-forge folium***

# In[ ]:


import folium
from folium import plugins


# In[ ]:


os.chdir("/kaggle/input/crimes-in-boston/")
os.listdir()


# In[ ]:


df=pd.read_csv(r'crime.csv', encoding='unicode_escape',low_memory=False)
df.head()


# ***Define Pie plot function***

# In[ ]:


def pie_plot(list_number, list_unique):
    plt.figure(figsize=(20,10))
    plt.pie(list_unique, 
        labels=list_number,
        autopct='%1.1f%%', 
        shadow=True, 
        startangle=140)
 
    plt.axis('equal')
    plt.show()
    return 0


# ***Define bar chart function***

# In[ ]:


def bar_chart(list_number, list_unique,xlabel,ylabel):
    objects = list_unique
    y_pos = np.arange(len(objects))
    performance = list_number
 
    plt.figure(figsize=(20,10))    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel(ylabel) 
    plt.xlabel(xlabel)
    plt.show()
    
    return 0


# ***Group count of Incidents by day of the week***

# ***Day when most of the crimes occurred***

# In[ ]:


#df_dy_crim=pd.DataFrame(df.groupby(['DAY_OF_WEEK']).count().INCIDENT_NUMBER.transform(max)).reset_index()
df_dy_crim=df.groupby(['DAY_OF_WEEK']).count().INCIDENT_NUMBER.reset_index(name='Number_of_incidents')
dy_mst_crm=df_dy_crim.sort_values(by=['Number_of_incidents'],ascending=False).head(1)
dy_mst_crm


# ***The result above could be achieved using a single line of code as mentioned below as well***

# In[ ]:


df.groupby(['DAY_OF_WEEK']).count().INCIDENT_NUMBER.reset_index(name='Number_of_incidents').sort_values(by='Number_of_incidents', ascending=False).head(1)


# ***Highest number of incidents for a particular crime occured in each year***

# In[ ]:


crm_typ_yr=df.groupby(['OFFENSE_CODE_GROUP','YEAR']).size().reset_index(name='count').sort_values(by=['count'], ascending=False)
crm_typ_yr.head()


# ***Group Crime types by Street***

# In[ ]:


loc_crm_typ=df.loc[0:,['OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION','STREET']].groupby('STREET').agg(' ,'.join).reset_index()
loc_crm_typ['Lat']= df.loc[0:,['Lat']]
loc_crm_typ['Long']=df.loc[0:,['Long']]
loc_crm_typ.head()


# ***Year wise crime count***

# In[ ]:


df_inc_by_yr=df.groupby('YEAR').count().INCIDENT_NUMBER.reset_index(name="Number of Incidents")
df_inc_by_yr_lbl=df['YEAR'].unique()
df_inc_by_yr_lbl


# **A function to create lablel and data list for chart**

# In[ ]:


def create_list_number_crime(name_column, list_unique):
    # list_unique = df[name_column].unique()
    
    i = 0
    
    list_number = list()
    
    while i < len(list_unique):
        list_number.append(len(df.loc[df[name_column] == list_unique[i]]))
        i += 1
    
    return list_unique, list_number


# In[ ]:


create_list_number_crime('YEAR',df['YEAR'].unique())


# In[ ]:


pie_plot(list(df['YEAR'].unique()),df_inc_by_yr['Number of Incidents'])


# In[ ]:


bar_chart(df_inc_by_yr['Number of Incidents'],df_inc_by_yr['YEAR'],'Year','Number of Incidents')


# ***Function to drop Nan values in the input variables***

# In[ ]:


def drop_NaN_two_var(x, y):

    df1 = df[[x, y]].dropna()
    print(df1.shape)

    x_value = df1[x]
    y_value = df1[y]

    del df1
        
    print(x + ': ' + str(x_value.shape))
    print(y + ': ' + str(y_value.shape))
        
    return x_value, y_value


# ***Group by District and Year and plot the number of incidents***

# In[ ]:


df_dst_yr_crm=df.groupby(by=['DISTRICT','YEAR']).count().INCIDENT_NUMBER.reset_index(name='Number of Incidents')
sns.barplot(x='DISTRICT',y='Number of Incidents',hue='YEAR',data=df_dst_yr_crm)
plt.tight_layout()
plt.show()


# ***Month wise crime counts over the years***

# In[ ]:


df_mnth_crm = df.groupby('MONTH').count().INCIDENT_NUMBER.reset_index(name='Number of Incidents')
bar_chart(df_mnth_crm['Number of Incidents'],df_mnth_crm['MONTH'],'Month','Number of Incidents')


# ***Day wise crime count over the years***

# In[ ]:


df_dy_crim=df.groupby(['DAY_OF_WEEK']).count().INCIDENT_NUMBER.reset_index(name='num_of_incidents')
df_dy_crim.head()
day_of_week=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
bar_chart(df_dy_crim['num_of_incidents'],day_of_week,'Day of the week','Number of Incidents')


# ***Hour wise Crime count over the years***

# In[ ]:


hr_num_crm=df.groupby('HOUR').count().INCIDENT_NUMBER.reset_index(name='num_of_inc')
hr_num_crm
bar_chart(hr_num_crm['num_of_inc'],hr_num_crm['HOUR'],'Hour of the day','Number of Incidents')


# ***Hour wise crime by years***

# In[ ]:


hr_yr_crm=df.groupby(['HOUR','YEAR']).count().INCIDENT_NUMBER.reset_index(name='Number of Incidents')

sns.barplot(x='HOUR',y='Number of Incidents',hue='YEAR',data=hr_yr_crm)


# ***Offense Code wise number of incidents over the years***

# In[ ]:


crm_off_grp_df=df.groupby(['OFFENSE_CODE_GROUP']).count().INCIDENT_NUMBER.reset_index(name='Number of Incidents').sort_values(by='Number of Incidents', ascending=False)
fig, ax = plt.subplots()
fig.set_size_inches(10, 20)
sns.barplot('Number of Incidents','OFFENSE_CODE_GROUP',data=crm_off_grp_df,ax=ax)


# ***Fill na in Shooting column and change it to Numerical category by mapping 0 values to 0 and 'Y' values to 1***

# In[ ]:


df['SHOOTING'].fillna(0,inplace=True)

df['SHOOTING'] = df['SHOOTING'].map({
    0: 0,
    'Y':1
})

df['SHOOTING'].unique()


# ***Shooting occurences count and their percentage in the overall crime***

# In[ ]:


Shoot_True=len(df.loc[df['SHOOTING'] == 1])
Shoot_False=len(df.loc[df['SHOOTING'] == 0])

print('With shooting(num): ' + str(Shoot_True))
print('With shooting(%):   ' + str(round(Shoot_True*100/len(df),2))+'%')
print()
print('Without shooting(num): ' + str(Shoot_False))
print('Without shooting(%):   ' + str(round(Shoot_False*100/len(df),2))+'%')


# ***Shooting occurences over the years***

# In[ ]:


shoot_by_yr=df[df['SHOOTING']==1].groupby('YEAR').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_yr
bar_chart(shoot_by_yr['Number of Shootings'],shoot_by_yr['YEAR'],'Year','Number of Shootings')


# ***Month wise Shootings over the years***

# In[ ]:


shoot_by_mnth=df[df['SHOOTING']==1].groupby('MONTH').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_mnth

bar_chart(shoot_by_mnth['Number of Shootings'],shoot_by_mnth['MONTH'],'Month','Number of Shootings')


# ***Day wise Shooting occurences over the years***

# In[ ]:


shoot_by_day=df[df['SHOOTING']==1].groupby('DAY_OF_WEEK').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_day

bar_chart(shoot_by_day['Number of Shootings'],day_of_week,'Day of Week','Number of Shootings')


# In[ ]:


shoot_by_hour=df[df['SHOOTING']==1].groupby('HOUR').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_hour

bar_chart(shoot_by_hour['Number of Shootings'],shoot_by_hour['HOUR'],'Hour of the day','Number of Shootings')


# In[ ]:


shoot_by_district=df[df['SHOOTING']==1].groupby('DISTRICT').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_district

bar_chart(shoot_by_district['Number of Shootings'],shoot_by_district['DISTRICT'],'Hour of the day','Number of Shootings')


# ***District wise Shooting occurences over the years - Each district is highlighted in a different color using cm.spectral option!
# Two different ways to achieve the same output is also shown here***

# In[ ]:


plt.figure(figsize=[10,5])
color_dis=plt.cm.Spectral(np.linspace(0, 1, len(shoot_by_district['DISTRICT'])))
plt.bar(shoot_by_district['DISTRICT'],shoot_by_district['Number of Shootings'], color=color_dis)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
df_shoot=df[df['SHOOTING']==1]
df_shoot['DISTRICT'].value_counts().plot.bar(color=color_dis)
plt.show()


# ***Plotting Latitude and Longitudes of shooting crime locations in different kinds of joint plots to see if there is a pattern or concentration of crimes pertaining to a specific location***

# ***P.S. joinplot option in seaborn library creates its own figure and axes. Hence it does not have an 'ax' argument to allow us to add subplots***

# In[ ]:


shoot_location = df_shoot[['Lat','Long']]
shoot_location = shoot_location.dropna()

shoot_location.head()
shoot_location=shoot_location.loc[(shoot_location['Lat']>40) & (shoot_location['Long'] < -60)]  

x_shoot = shoot_location['Long']
y_shoot = shoot_location['Lat']

sns.jointplot(x_shoot, y_shoot, kind='scatter')
sns.jointplot(x_shoot, y_shoot, kind='hex')
sns.jointplot(x_shoot, y_shoot, kind='kde')
sns.jointplot(x_shoot,y_shoot,kind='reg')
plt.show()


# ***Plot UCR Part wise Shootings over the years***

# In[ ]:


shoot_by_UCR=df[df['SHOOTING']==1].groupby('UCR_PART').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')

plt.figure(figsize=(10,5))
color_ucr=plt.cm.Spectral(np.linspace(0, 1, len(shoot_by_UCR['UCR_PART'])))
df_shoot['UCR_PART'].value_counts().plot.bar(color=color_ucr)
plt.show()


# ***Exploring all the crime locations based on longitude and Latitude data and see if it is concentrated somewhere or if there is a pattern visible***

# In[ ]:


df[['Lat','Long']].describe()

location = df[['Lat','Long']]
location = location.dropna()

location = location.loc[(location['Lat']>40) & (location['Long'] < -60)]


# In[ ]:


x = location['Long']
y = location['Lat']
rand_colors = np.random.rand(len(x))
plt.figure(figsize=(20,20))
plt.scatter(x, y,c=rand_colors, alpha=0.5)
plt.show()


# In[ ]:


m = folium.Map([42.348624, -71.062492], zoom_start=11)


# In[ ]:


#generate various join plots to see if there is a pattern or trend
sns.jointplot(x, y, kind='scatter')
sns.jointplot(x, y, kind='hex')
sns.jointplot(x, y, kind='kde')


# In[ ]:


df.isnull().sum()


# ***Create a new category in the dataframe to indicate if the crime happened during Day or Night***

# ***Link to determine Day or Night based on Hour in Boston - https://www.timeanddate.com/sun/usa/boston ***

# ***We take the details on Day time for each of the 12 months from the above mentioned website and use the same to tag the said hours as Day***

# ***Then we Night using the logic that whenever Day==0, Night = 1***

# In[ ]:


df['Day']=0
df['Night']=0
# Day time for 1st month
df['Day'].loc[(df['MONTH'] == 1) & (df['HOUR'] >= 6) & (df['HOUR'] <= 18)] = 1

# Day time for 2st month
df['Day'].loc[(df['MONTH'] == 2) & (df['HOUR'] >= 6) & (df['HOUR'] <= 19)] = 1

# Day time for 3rd month
df['Day'].loc[(df['MONTH'] == 3) & (df['HOUR'] >= 6) & (df['HOUR'] <= 20)] = 1

# Day time for 4st month
df['Day'].loc[(df['MONTH'] == 4) & (df['HOUR'] >= 5) & (df['HOUR'] <= 20)] = 1

# Day time for 5th month
df['Day'].loc[(df['MONTH'] == 5) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1

# Day time for 6th month
df['Day'].loc[(df['MONTH'] == 6) & (df['HOUR'] >= 4) & (df['HOUR'] <= 21)] = 1

# Day time for 7th month
df['Day'].loc[(df['MONTH'] == 7) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1

# Day time for 8th month
df['Day'].loc[(df['MONTH'] == 8) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1

# Day time for 9th month
df['Day'].loc[(df['MONTH'] == 9) & (df['HOUR'] >= 6) & (df['HOUR'] <= 20)] = 1

# Day time for 10th month
df['Day'].loc[(df['MONTH'] == 10) & (df['HOUR'] >= 6) & (df['HOUR'] <= 19)] = 1

# Day time for 11th month
df['Day'].loc[(df['MONTH'] == 11) & (df['HOUR'] >= 6) & (df['HOUR'] <= 17)] = 1

# Day time for 12th month
df['Day'].loc[(df['MONTH'] == 12) & (df['HOUR'] >= 7) & (df['HOUR'] <= 17)] = 1


#Update Night as 1 where Day is 0
df['Night'].loc[df['Day']==0]=1


# ***Crime count on Day & Night***

# In[ ]:


plt.figure(figsize=(16,8))
color_DN=plt.cm.Spectral(np.linspace(0, 1, 2))
df['Night'].value_counts().plot.bar(color=color_DN)
plt.show()


# ***Try to see if we can classify the Offense code group by considering other columns independent variables' behaviour/ pattern***

# ***Y - OFFENSE_CODE_GROUP;X - 'DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK', 'HOUR','Lat','Long', 'OFFENSE_CODE_GROUP','Day','Night'***

# ***Top Offense code groups***

# In[ ]:


df['OFFENSE_CODE_GROUP'].value_counts().head(15)


# ***Place the top offense code groups in a list***

# In[ ]:


list_offense_code_group=('Motor Vehicle Accident Response',
                           'Larceny',
                           'Medical Assistance',
                           'Investigate Person',
                           'Other',
                           'Drug Violation',
                           'Simple Assault',
                           'Vandalism',
                           'Verbal Disputes',
                           'Towed',
                           'Investigate Property',
                           'Larceny From Motor Vehicle')
list_offense_code_group


# ***We are going to consider only top 15 offenses data for our model to minimize the noise***

# In[ ]:


df_model = pd.DataFrame()


# In[ ]:


i = 0

while i < len(list_offense_code_group):

    df_model = df_model.append(df.loc[df['OFFENSE_CODE_GROUP'] == list_offense_code_group[i]])
    
    i+=1


# In[ ]:


df_model.columns


# ***List only the needed columns and remove the rest***

# In[ ]:


list_column = ['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK',
               'HOUR','Lat','Long', 'OFFENSE_CODE_GROUP','Day','Night']


# In[ ]:


df_model=df_model[list_column]


# ***We are going to convert the available column values into numbers***

# ***DISTRICT Values are being assigned numbers - Each district will be identified by a number***

# In[ ]:


df_model['DISTRICT'].unique()


# In[ ]:


df_model['DISTRICT'] = df_model['DISTRICT'].map({
    'B3':1, 
    'E18':2, 
    'B2':3, 
    'E5':4, 
    'C6':5, 
    'D14':6, 
    'E13':7, 
    'C11':8, 
    'D4':9, 
    'A7':10, 
    'A1':11, 
    'A15':12
})

df_model['DISTRICT'].unique()


# ***Assign numbers to REPORTING_AREA column values or convert the string values to numerics***

# In[ ]:


df_model['REPORTING_AREA'] = pd.to_numeric(df_model['REPORTING_AREA'], errors='coerce')


# ***As MONTH column values are already numbers we dont have to alter them***

# In[ ]:


df_model['MONTH'].unique()


# ***Assign Day number to DAY_OF_WEEK Column values with the week starting from Monday i.e. Monday is assigned the value 1 and so on ***

# In[ ]:


df_model['DAY_OF_WEEK'] = df_model['DAY_OF_WEEK'].map({
    'Monday':1,
    'Tuesday':2,
    'Wednesday':3,
    'Thursday':4,
    'Friday':5,
    'Saturday':6, 
    'Sunday':7    
})

df_model['DAY_OF_WEEK'].unique()


# ***As HOUR column values are already numbers we dont have to alter them***

# In[ ]:


df_model['HOUR'].unique()


# ***LAT and LONG values are also in the expected format already***

# In[ ]:


df_model[['Lat', 'Long']].head()


# ***Fill nan in our model with 0***

# In[ ]:


df_model.fillna(0, inplace = True)


# ***Define our independent variable/s --> X***

# In[ ]:


x = df_model[['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night']]


# ***Define our dependent variable--> Y***

# In[ ]:


y = df_model['OFFENSE_CODE_GROUP']


# ***Check our dependent variable and assign unique numbers to each category or group***

# In[ ]:


y.unique()


# In[ ]:


y=y.map({
    'Motor Vehicle Accident Response':1, 
    'Larceny':2, 
    'Medical Assistance':3,
    'Investigate Person':4, 
    'Other':5, 
    'Drug Violation':6, 
    'Simple Assault':7,
    'Vandalism':8, 
    'Verbal Disputes':9, 
    'Towed':10, 
    'Investigate Property':11,
    'Larceny From Motor Vehicle':12
})


# ***Import sklearn libraries***

# In[ ]:


from sklearn.model_selection import train_test_split


# ***Generate test and train datasets using train_test_split module of sklearn library***

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# ***Import Classification modules from sklearn***

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
#!conda install -c conda-forge lightgbm --yes
from lightgbm import LGBMClassifier


# ***If lightgbm is not installed use the following command to get it installed - !conda install -c conda-forge lightgbm --yes***

# ***Import f1-score metric***

# In[ ]:


from sklearn.metrics import f1_score


# ***Define function to get mean, max and min that will be used to analyze the f1 score of different classifier algorithms***

# In[ ]:


def func_results(result):
    print('mean: ' + str(result.mean()))
    print('max: ' + str(result.max()))
    print('min: ' + str(result.min()))
    return result


# ***Decision tree classifier function to train, predict and compare with actual values***

# In[ ]:


def func_DecisionTreeClassifier(x_train, y_train):
    dec_tree = DecisionTreeClassifier()
    dec_tree = dec_tree.fit(x_train, y_train)

    dec_tree_pred = dec_tree.predict(x_test)

    dec_tree_score = f1_score(y_test, dec_tree_pred, average=None)
    return func_results(dec_tree_score)


# In[ ]:


func_DecisionTreeClassifier(x_train,y_train)


# ***Bernoulli classfier function to train, predict and compare with actual values***

# In[ ]:


def func_BernoulliNB(x_train, y_train):
    bernoulli = BernoulliNB()
    bernoulli = bernoulli.fit(x_train, y_train)

    bernoulli_pred = bernoulli.predict(x_test)

    bernoulli_score = f1_score(y_test, bernoulli_pred, average=None)
    return func_results(bernoulli_score)


# In[ ]:


func_BernoulliNB(x_train,y_train)


# ***Extra tree classifier function to train, predict and compare with actual values***

# In[ ]:


def func_ext_tree_cls(x_train,y_train):
    ext_tree=ExtraTreeClassifier()
    ext_tree=ext_tree.fit(x_train,y_train)
    ext_tree_pred=ext_tree.predict(x_test)
    ext_tree_score=f1_score(y_test,ext_tree_pred,average=None)
    return func_results(ext_tree_score)


# In[ ]:


func_ext_tree_cls(x_train,y_train)


# ***K Neighbor classifier function to train, predict and compare with actual values***

# ***For KNN Classifier K value needs to be passed, if not, default value - 5 is taken as K value***

# ***Here we are passing the n value as 5 and the f1 score is shown below***

# In[ ]:


def func_KNeighborsClassifier(x_train, y_train,n):
    Kneigh = KNeighborsClassifier(n_neighbors=n)
    Kneigh.fit(x_train, y_train) 

    Kneigh_pred = Kneigh.predict(x_test)

    Kneigh_score = f1_score(y_test, Kneigh_pred, average=None)
    return func_results(Kneigh_score),Kneigh_pred


# In[ ]:


KNN_score, KNN_pred=func_KNeighborsClassifier(x_train,y_train,5)
KNN_score


# In[ ]:


from sklearn import metrics


# ***Now let us see, if the score changes with the K value but here we are considering Accuracy metric instead of f1 score and calculate the score dynamically for each K value and we can see the same in a graph to choose the K value that gives us a better score***

# ***May be we can try using the K value with highest accuracy score above and see if f1 score improves as well***

# In[ ]:


Ks=20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[ ]:


mean_acc - 1 * std_acc


# In[ ]:


mean_acc + 1 * std_acc


# In[ ]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[ ]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# ***Gaussian classifier function to train, predict and compare with actual values***

# In[ ]:


def func_GaussianNB(x_train, y_train):
    gaussian = GaussianNB()
    gaussian = gaussian.fit(x_train, y_train)

    gaussian_pred = gaussian.predict(x_test)

    gaussian_score = f1_score(y_test, gaussian_pred, average=None)
    return func_results(gaussian_score)


# In[ ]:


func_GaussianNB(x_train,y_train)


# ***Random Forest classifier function to train, predict and compare with actual values***

# In[ ]:


def func_RandomForestClassifier(x_train, y_train):
    rfc = RandomForestClassifier()
    rfc = rfc.fit(x_train, y_train)

    rfc_pred = rfc.predict(x_test)

    rfc_score = f1_score(y_test, rfc_pred, average=None)
    return func_results(rfc_score)


# In[ ]:


func_RandomForestClassifier(x_train,y_train)


# ***LGBM classifier function to train, predict and compare with actual values***

# In[ ]:


def func_LGBMClassifier(x_train, y_train):
    lgbm = LGBMClassifier()
    lgbm = lgbm.fit(x_train, y_train)

    lgbm_pred = lgbm.predict(x_test)

    lgbm_score = f1_score(y_test, lgbm_pred, average=None)
    return func_results(lgbm_score)


# In[ ]:


func_LGBMClassifier(x_train,y_train)


# ***Create another model to see how well we can classify the data ***

# ***Y- DISTRICT
# 
# X - OFFENSE_CODE_GROUP, Month, Day of week, Hour, Day, Night***

# ***Create a Dataframe for 2nd model with the following columns***

# In[ ]:


df_model_2 = df[['OFFENSE_CODE', 'DISTRICT','MONTH','DAY_OF_WEEK','HOUR','Day','Night']]
df_model_2.head()


# ***Convert all the variables to the desired number format as we did before***

# In[ ]:


df_model_2['OFFENSE_CODE'] = pd.to_numeric(df_model_2['OFFENSE_CODE'], errors='coerce')


# ***DISTRICT value is mapped to numbers like we did in the previous model***

# In[ ]:


df_model_2['DISTRICT'] = df_model_2['DISTRICT'].map({
    'B3':1, 
    'E18':2, 
    'B2':3, 
    'E5':4, 
    'C6':5, 
    'D14':6, 
    'E13':7, 
    'C11':8, 
    'D4':9, 
    'A7':10, 
    'A1':11, 
    'A15':12
})

df_model_2['DISTRICT'].unique()


# ***Assign numbers to DAY_OF_WEEK column values with the week starting from Monday***

# In[ ]:


df_model_2['DAY_OF_WEEK'] = df_model_2['DAY_OF_WEEK'].map({
    'Tuesday':2, 
    'Saturday':6, 
    'Monday':1, 
    'Sunday':7, 
    'Thursday':4, 
    'Wednesday':3,
    'Friday':5
})

df_model_2['DAY_OF_WEEK'].unique()


# ***Check if there is any null value in our dataframe***

# In[ ]:


df_model_2.isnull().sum()


# ***Drop na and nan values from our dataframe***

# In[ ]:


df_model_2 = df_model_2.dropna()


# ***After dropping we could see that there is no null value in our data frame***

# In[ ]:


df_model_2.isnull().sum()


# In[ ]:


df_model_2.shape


# ***Declare our dependent and independent variables***

# In[ ]:


x = df_model_2[['OFFENSE_CODE','MONTH','DAY_OF_WEEK','HOUR','Day','Night']]
y = df_model_2['DISTRICT']


# ***Split our data into train and test sets***

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# ***Start applying the classifiers***

# ***Applying BernoulliNB classifier and the mean, max, min of the f1 scores along with actual f1 scores array is returned***

# In[ ]:


func_BernoulliNB(x_train,y_train)


# ***Applying Decision Tree classifier and the mean, max, min of the f1 scores along with actual f1 scores array is returned***

# In[ ]:


func_DecisionTreeClassifier(x_train,y_train)


# ***Applying External Tree classifier and the mean, max, min of the f1 scores along with actual f1 scores array is returned***

# In[ ]:


func_ext_tree_cls(x_train,y_train)


# ***Applying GaussianNB classifier and the mean, max, min of the f1 scores along with actual f1 scores array is returned***

# In[ ]:


func_GaussianNB(x_train,y_train)


# ***Applying KNN classifier and the mean, max, min of the f1 scores along with actual f1 scores array is returned***

# In[ ]:


func_KNeighborsClassifier(x_train,y_train,5)


# ***As we did before we are trying KNN Classifier with different K values but the accuracy score used below is not a great representation of accuracy for data sets where imbalanced class distribution exists like in this case***

# In[ ]:


Ks=10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[ ]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# ***Applying LGBM classifier and the mean, max, min of the f1 scores along with actual f1 scores array is returned***

# In[ ]:


func_LGBMClassifier(x_train,y_train)


# ***Applying Random Forest classifier and the mean, max, min of the f1 scores along with actual f1 scores array is returned***

# In[ ]:


func_RandomForestClassifier(x_train,y_train)


# ***Classifying UCR_PART variable***

# ***Y - UCR_PART***
# 
# ***X - DISTRICT, REPORTING_AREA, MONTH, DAY_OF_WEEK, HOUR, LATITUDE, LONGITUDE***

# ***We first standardise and normalise our data like we did in our previous models***

# In[ ]:


df_model3 = df[['DISTRICT','REPORTING_AREA', 'MONTH','DAY_OF_WEEK','HOUR','UCR_PART','Lat','Long']]


# In[ ]:


df_model3['DISTRICT'] = df_model3['DISTRICT'].map({
    'B3':1, 
    'E18':2, 
    'B2':3, 
    'E5':4, 
    'C6':5, 
    'D14':6, 
    'E13':7, 
    'C11':8, 
    'D4':9, 
    'A7':10, 
    'A1':11, 
    'A15':12
})


# In[ ]:


df_model3['REPORTING_AREA'] = pd.to_numeric(df_model3['REPORTING_AREA'], errors='coerce')


# In[ ]:


df_model3['DAY_OF_WEEK'] = df_model3['DAY_OF_WEEK'].map({
    'Tuesday':2, 
    'Saturday':6, 
    'Monday':1, 
    'Sunday':7, 
    'Thursday':4, 
    'Wednesday':3,
    'Friday':5
})


# In[ ]:


df_model3['UCR_PART'].unique()


# In[ ]:


df_model3['UCR_PART'] = df_model3['UCR_PART'].map({
    'Part Three':3, 
    'Part One':1, 
    'Part Two':2, 
#    'Other':4
})


# In[ ]:


df_model3 = df_model3.dropna()
print(df_model3.shape)
df_model3.isnull().sum()


# ***Define our x and y then split our data set into test and train sets***

# In[ ]:


x = df_model3[['DISTRICT','REPORTING_AREA', 'MONTH','DAY_OF_WEEK','HOUR','Lat','Long']]
y = df_model3['UCR_PART']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y, 
    test_size = 0.1,
    random_state=42
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# ***Apply all the classifiers like we did before***

# In[ ]:


func_BernoulliNB(x_train,y_train)


# In[ ]:


func_DecisionTreeClassifier(x_train,y_train)


# In[ ]:


func_ext_tree_cls(x_train,y_train)


# In[ ]:


func_GaussianNB(x_train,y_train)


# In[ ]:


func_KNeighborsClassifier(x_train,y_train,5)


# In[ ]:


func_LGBMClassifier(x_train,y_train)


# In[ ]:


func_RandomForestClassifier(x_train,y_train)


# ***Trying to cluster the locations of the crime***

# In[ ]:


location.isnull().sum()


# In[ ]:


location.shape


# ***Plot our location attributes as a scatter plot***

# In[ ]:


x = location['Long']
y = location['Lat']

colors = np.random.rand(len(location))

plt.figure(figsize=(20,20))
plt.scatter(x, y,c=colors, alpha=0.5)
plt.show()


# ***Set up our X variable on which clusters are to be determined***

# In[ ]:


X = location
X = X[~np.isnan(X)]


# ***Import KMeans clustering module from sklearn library***

# In[ ]:


from sklearn.cluster import KMeans


# ***Define K means fit and predict function that takes X and number of clusters as input***

# In[ ]:


def Kmeanscl(X, nclust):
    kmeansmodel = KMeans(nclust)
    kmeansmodel.fit(X)
    clust_labels = kmeansmodel.predict(X)
    cent = kmeansmodel.cluster_centers_
    return (clust_labels, cent)


# ***Call our K means model by setting the number of clusters as 2 and add the kmeans predicted cluster value as kmeans column in our X***

# In[ ]:


clust_labels, cent = Kmeanscl(X, 2)
kmeans = pd.DataFrame(clust_labels)
X.insert((X.shape[1]),'kmeans',kmeans)


# ***Plot the clusters***

# In[ ]:


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
scatter = ax.scatter(X['Long'],X['Lat'],
                     c=kmeans[0],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Long')
ax.set_ylabel('Lat')
plt.colorbar(scatter)


# ***Now we try the same thing with 3 clusters***

# In[ ]:


X = location
X = X[~np.isnan(X)]


# In[ ]:


clust_labels, cent = Kmeanscl(X, 3)
kmeans = pd.DataFrame(clust_labels)
X.insert((X.shape[1]),'kmeans',kmeans)


# In[ ]:


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
scatter = ax.scatter(X['Long'],X['Lat'],
                     c=kmeans[0],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Long')
ax.set_ylabel('Lat')
plt.colorbar(scatter)


# In[ ]:


#!conda install -c districtdatalabs yellowbrick --yes


# ***Install yellowbrick library to visualise KElbow to choose the best k value***

# In[ ]:


from yellowbrick.cluster import KElbowVisualizer


# In[ ]:


X = location
X = X[~np.isnan(X)]


# ***Apply KMeans and predict the best k value***

# In[ ]:


KMdl=KMeans()
visualizer = KElbowVisualizer(KMdl, k=(4,12),locate_elbow=True)

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.poof()  


# ***Difference here is that the timings curve is set as False so that is not plotted here and the metric to calculate the score is different here***

# In[ ]:


KMdl2=KMeans()
visualizer2 = KElbowVisualizer(KMdl2, k=(4,12), metric='calinski_harabasz',locate_elbow=True)
visualizer2.fit(X)        # Fit the data to the visualizer
visualizer2.poof()   


# ***Locate UCR_PART = 'Part One' Shooting Crimes alone around the city***

# In[ ]:


ucr_prt1_shoot_crm=df[(df['Lat']>=40) & (df['Long']<=-70) &(df['UCR_PART']=='Part One') & (df['SHOOTING']==1)].fillna(0).reset_index()


# In[ ]:


m = folium.Map( [42.3601,-71.0589],zoom_start=13, tiles='OpenStreetMap')
for i in range(0,len(ucr_prt1_shoot_crm)):
    #folium.Marker(ucr_prt1_crm.iloc[i]['Lat'], ucr_prt1_crm.iloc[i]['Long'], popup=ucr_prt1_crm.iloc[i]['OFFENSE_CODE_GROUP']).add_to(m)
    folium.Marker([ucr_prt1_shoot_crm.iloc[i]['Lat'], ucr_prt1_shoot_crm.iloc[i]['Long']], popup=ucr_prt1_shoot_crm.iloc[i]['OFFENSE_CODE_GROUP']).add_to(m)
    
m

