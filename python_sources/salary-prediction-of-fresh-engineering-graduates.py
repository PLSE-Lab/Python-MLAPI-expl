#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)   
pd.set_option('display.width', 1000)


# In[ ]:


train = pd.read_excel('train.xlsx') 
test = pd.read_excel('test.xlsx')
results = pd.read_excel('results.xlsx')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


data = pd.concat([train,test])


# In[ ]:


data.reset_index(inplace=True)


# In[ ]:


data.drop('index',axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.shape


# In[ ]:


data.columns


# ##### the values of collegeID an CollegecityID are same

# In[ ]:


data.drop(['ID','DOJ','DOL','Designation','JobCity','CollegeCityID'],axis=1,inplace=True)


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:





# ## Data Cleaning & Missing Value Treatment

# ### 10board

# In[ ]:


#treating for CBSE
data['10board'] = data['10board'].replace({'cbse':'CBSE',
                                             'central board of secondary education':'CBSE',
                                             'cbse board':'CBSE',
                                             'central board of secondary education, new delhi':'CBSE',
                                             'delhi public school':'CBSE',
                                             'c b s e':'CBSE',
                                             'central board of secondary education(cbse)':'CBSE',
                                             'dav public school,hehal':'CBSE',
                                             'cbse ':'CBSE',
                                             'cbse[gulf zone]':'CBSE',
                                             'cbsc':'CBSE'})


# In[ ]:


#treating for ICSE
data['10board'].replace({'icse':'ICSE',
                          'icse board':'ICSE',
                          'icse board , new delhi':'ICSE'},inplace=True)


# In[ ]:


#treating for State Board
stateboards = []

for i in data['10board']:
    if i!='CBSE' and i!='ICSE':
        stateboards.append(i)

#print(stateboards)


# In[ ]:


#replacing all values with 'State Board'
data['10board'].replace(stateboards,'State_Board',inplace=True)  


# In[ ]:





# In[ ]:





# In[ ]:


data['10board'].value_counts()


# In[ ]:


b = []
for i in data['10board']:
    b.append(i)
y = np.array(b)
print(np.unique(y))


# ### 12board

# In[ ]:


#treating for CBSE
data['12board'] = data['12board'].replace({'cbse':'CBSE',
                                             'central board of secondary education':'CBSE',
                                             'cbse board':'CBSE',
                                             'central board of secondary education, new delhi':'CBSE',
                                             'delhi public school':'CBSE',
                                             'c b s e':'CBSE',
                                             'central board of school education':'CBSE',
                                             'central board of secondary education(cbse)':'CBSE',
                                             'dav public school,hehal':'CBSE',
                                             'cbse ':'CBSE',
                                             'cbse[gulf zone]':'CBSE',
                                             'cbsc':'CBSE'})


# In[ ]:


#treating for ICSE
data['12board'].replace({'icse':'ICSE',
                          'icse board':'ICSE',
                          'icse board , new delhi':'ICSE'},inplace=True)


# In[ ]:


#treating for State Board
stateboardss = []

for i in data['12board']:
    if i!='CBSE' and i!='ICSE':
        stateboardss.append(i)

#print(stateboardss)


# In[ ]:


#replacing all values with 'State Board'
data['12board'].replace(stateboardss,'State_Board',inplace=True)  


# In[ ]:


data['12board'].value_counts()


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


c = []
for i in data['12board']:
    c.append(i)
z = np.array(c)
print(np.unique(z))


# ### adjust collegeGPA

# In[ ]:


len(data.loc[data['collegeGPA']>10])


# In[ ]:


len(data['collegeGPA'])


# In[ ]:


for i in range(5498):
    if data['collegeGPA'].values[i]>10:
        data.collegeGPA.values[i] = data.collegeGPA.values[i]/10


# In[ ]:


len(data.loc[data['collegeGPA']>10])


# In[ ]:





# ### Removing 6 variables more than 70% missing value

# In[ ]:


len(data.loc[data['ElectronicsAndSemicon']==-1])/len(data['ElectronicsAndSemicon'])*100


# In[ ]:


len(data.loc[data['ComputerScience']==-1])/len(data['ComputerScience'])*100


# In[ ]:


len(data.loc[data['MechanicalEngg']==-1])/len(data['MechanicalEngg'])*100


# In[ ]:


len(data.loc[data['ElectricalEngg']==-1])/len(data['ElectricalEngg'])*100


# In[ ]:


len(data.loc[data['TelecomEngg']==-1])/len(data['TelecomEngg'])*100


# In[ ]:


len(data.loc[data['CivilEngg']==-1])/len(data['CivilEngg'])*100


# In[ ]:


data.drop(['ElectronicsAndSemicon','ComputerScience','MechanicalEngg','ElectricalEngg','TelecomEngg','CivilEngg'],axis=1,inplace=True)


# In[ ]:


data.shape


# ##### ComputerProgramming & Domain

# In[ ]:


len(data.loc[data['ComputerProgramming']==-1])/len(data['ComputerProgramming'])*100


# In[ ]:


data['ComputerProgramming'].corr(data['Salary'])


# ComputerProgramming has 21.46% missing values

# In[ ]:





# In[ ]:


len(data.loc[data['Domain']==-1])/len(data['Domain'])*100


# In[ ]:


data['Domain'].corr(data['Salary'])


# ComputerProgramming has 6.04% missing values

# Both the columns Domain & ComputerProgramming are continuous value. So we are replacing with mean

# In[ ]:


data['ComputerProgramming'] = data['ComputerProgramming'].replace(-1,np.nan)


# In[ ]:


data['ComputerProgramming'].isnull().sum()


# In[ ]:


data['ComputerProgramming'] = data['ComputerProgramming'].replace(np.nan,data['ComputerProgramming'].mean())


# In[ ]:


data['ComputerProgramming'].isnull().sum()


# In[ ]:


data['ComputerProgramming'].corr(data['Salary'])


# In[ ]:





# In[ ]:





# In[ ]:


data['Domain'] = data['Domain'].replace(-1,np.nan)


# In[ ]:


data['Domain'].isnull().sum()


# In[ ]:


data['Domain'] = data['Domain'].replace(np.nan,data['Domain'].mean())


# In[ ]:


data['Domain'].isnull().sum()


# In[ ]:


data['Domain'].corr(data['Salary'])


# ### CollegeTier : replace 2 with 0 : 0 represents CollegeTier 2

# In[ ]:


data['CollegeTier'] = data['CollegeTier'].replace(2,0)


# In[ ]:


data['CollegeTier'].value_counts()


# We are replacing this because we have to gime more weightage to tier 1 colleges

# #### Extracting year from DOB

# In[ ]:


data['DOB'] = pd.to_datetime(data['DOB'])


# In[ ]:


data['DOB'] = pd.DatetimeIndex(data['DOB']).year


# In[ ]:


data['DOB'] = data['DOB'].astype(int)


# ### Specialization

# In[ ]:


data.Specialization = data.Specialization.replace({'construction technology and management':'civil engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'computer application':'computer science and engineering',
                                                     'electronics and computer engineering':'computer science and engineering',
                                                  'computer science and technology':'computer science and engineering',
                                                     'software engineering':'computer science and engineering',
                                                    'computer science':'computer science and engineering',
                                                   'computer engineering':'computer science and engineering',
                                                     'electronics and computer engineering':'computer science and engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'computer science & engineering':'computer science and engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'electronics engineering':'electronics and communication engineering',
                                                       'communication engineering':'electronics and communication engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'electronics':'electronics and communication engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'electronics & instrumentation eng':'electronics and instrumentation engineering',
                                                    'applied electronics and instrumentation':'electronics and instrumentation engineering',
                                                    'instrumentation engineering':'electronics and instrumentation engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'telecommunication engineering':'electronics and telecommunication engineering',
                                                   'electronics and telecommunication':'electronics and telecommunication engineering',
                                                   'electronics & telecommunications':'electronics and telecommunication engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'information science engineering':'information technology'})


# In[ ]:


data.Specialization = data.Specialization.replace({'information science':'information technology'})


# In[ ]:


data.Specialization = data.Specialization.replace({'information & communication technology':'information technology'})


# In[ ]:


data.Specialization = data.Specialization.replace({'vlsi design and cad':'computer aided design',
                                                     'cad / cam':'computer aided design'})


# In[ ]:


data.Specialization = data.Specialization.replace({'industrial & production engineering':'industrial and production engineering',
                                                    'industrial engineering and management':'industrial and production engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'biotechnology':'biotechnology engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'electronics and telecommunication engineering':'electronics and communication engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'electronics and instrumentation engineering':'electronics and communication engineering'})


# In[ ]:


data.Specialization = data.Specialization.replace({'electrical engineering':'electronics and electrical engineering'})


# In[ ]:





# In[ ]:


data['Specialization'].value_counts()


# In[ ]:


data.drop(data[data.Specialization == 'polymer technology'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'environment science'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'textile engineering'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'power systems and automation'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'operational research'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'aerospace engineering'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'embedded systems technology'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'ceramic engineering'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'mechatronics'].index, inplace=True)
data.drop(data[data.Specialization == 'computer aided design'].index, inplace=True)
data.drop(data[data.Specialization == 'electrical and power engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'automobile/automotive engineering'].index, inplace=True)


# In[ ]:


data.drop(data[data.Specialization == 'metallurgical engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'chemical engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'biomedical engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'aeronautical engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'automobile engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'mechatronics engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'biotechnology engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'industrial and production engineering'].index, inplace=True)
data.drop(data[data.Specialization == 'other'].index, inplace=True)


# In[ ]:


data['Specialization'].value_counts()


# In[ ]:


data.head()


# In[ ]:


data.shape


# #### CollegeCityTier

# In[ ]:


data['CollegeCityTier'].value_counts()


# In[ ]:


data['CollegeCityTier'] = data['CollegeCityTier'].replace({0:'A',1:'B'})


# In[ ]:


data['CollegeCityTier'].value_counts()


# #### Degree

# In[ ]:


data['Degree'].value_counts()


# In[ ]:


data['Degree'] = data['Degree'].replace({'B.Tech/B.E.':0,'MCA':1,'M.Sc. (Tech.)':3,'M.Tech./M.E.':4})


# In[ ]:


data['Degree'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data.info()


# In[ ]:





# ## Base-line-model

# In[ ]:





# In[ ]:


dummy_data = pd.get_dummies(data,drop_first=True)


# In[ ]:


dummy_data.shape


# In[ ]:


dummy_data.head()


# In[ ]:


new_train = dummy_data.loc[:3998]


# In[ ]:


new_train.shape


# In[ ]:


new_train.tail()


# In[ ]:


new_test = dummy_data.loc[3999:]


# In[ ]:


new_test.shape


# In[ ]:


new_test.tail()


# In[ ]:


x = new_train.drop(['Salary'], axis=1)
y = new_train['Salary']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size = 0.3, random_state = 3)


# In[ ]:


#xtrain = dummy_data.iloc[0:3898,1:]


# In[ ]:


#xtest = dummy_data.iloc[3999:,1:]


# In[ ]:


#ytrain = dummy_data.iloc[0:3898,0:1]


# In[ ]:


#ytest = dummy_data.iloc[3999:,0:1]


# In[ ]:


xtrain.shape


# In[ ]:


xtest.shape


# In[ ]:


ytrain.shape


# In[ ]:


ytest.shape


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(xtrain,ytrain)


# In[ ]:


ypred = model.predict(xtest)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r_sqrd = r2_score(ytest,ypred)
r_sqrd


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypred))


# In[ ]:


adj_r_sqrd = 1-(1-r_sqrd)*((1179-1)/(1179-57-1))
adj_r_sqrd


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


mape = mean_absolute_percentage_error(ytest,ypred)
mape


# In[ ]:





# ### outliers

# #### IQR of all numeric features

# In[ ]:


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# #### Getting percentile distribution for all numeric variables

# In[ ]:


data.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]).T


# #### outlier treatment

# ##### 10percentage

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,3))
sns.boxplot(x=data['10percentage'])
plt.show()


# In[ ]:


lower_limit = 72 - 1.5 * 13.6
lower_limit


# In[ ]:


data['10percentage']=data["10percentage"].map(lambda x:51 if x <51 else x)


# ##### 12percentage

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['12percentage'])


# In[ ]:


lower_limit = 66.4 - 1.5 * 16.4065
lower_limit


# In[ ]:


np.where(data['12percentage']<41.79)


# In[ ]:


data['12percentage'].min()


# In[ ]:


data['12percentage']=data["12percentage"].map(lambda x:41.79025 if x <41.79025 else x)


# ##### collegeGPA

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['collegeGPA'])


# In[ ]:


lower_limit = 6.69 - 1.5 * 0.95
lower_limit


# In[ ]:


np.where(data['collegeGPA']<5.265)


# In[ ]:


np.where(data['collegeGPA']<5)


# In[ ]:


upper_limit = 7.64 + 1.5 * 0.95
upper_limit


# In[ ]:


data['collegeGPA'].max()


# In[ ]:


np.where(data['collegeGPA']>9.065)


# In[ ]:


data['collegeGPA'].min()


# In[ ]:


data['collegeGPA']=data["collegeGPA"].map(lambda x:5 if x <5 else x)


# In[ ]:


#data['collegeGPA']=data["collegeGPA"].map(lambda x:9.99 if x >9.99 else x)


# ##### English

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['English'])


# In[ ]:


lower_limit = 425 - 1.5 * 145
lower_limit


# In[ ]:


np.where(data['English']<207.5)


# In[ ]:


upper_limit = 570 + 1.5 * 145
upper_limit


# In[ ]:


np.where(data['English']>787.5)


# In[ ]:


#data['English']=data["English"].map(lambda x:207.5 if x <207.5 else x)


# In[ ]:


#data['English']=data["English"].map(lambda x:787.5 if x >787.5 else x)


# In[ ]:





# ##### Quant

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['Quant'])


# In[ ]:


lower_limit = 430 - 1.5 * 165
lower_limit


# In[ ]:


upper_limit = 595 + 1.5 * 165
upper_limit


# In[ ]:


#data['Quant']=data["Quant"].map(lambda x:182.5 if x <182.5 else x)


# In[ ]:


#data['English']=data["English"].map(lambda x:842.5 if x >842.5 else x)


# In[ ]:





# In[ ]:





# ##### Logical

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['Logical'])


# In[ ]:


lower_limit = 445 - 1.5 * 120
lower_limit


# In[ ]:


upper_limit = 565 + 1.5 * 120
upper_limit


# In[ ]:


#data['Logical']=data["Logical"].map(lambda x:265 if x <265 else x)


# In[ ]:


#data['Logical']=data["Logical"].map(lambda x:745 if x >745 else x)


# In[ ]:





# ##### Domain

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['Domain'])


# ##### conscientiousness

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['conscientiousness'])


# In[ ]:


lower_limit = -0.726400 - 1.5 * 1.4291
lower_limit


# In[ ]:


upper_limit = 0.702700 + 1.5 * 1.4291
upper_limit


# In[ ]:


#data['conscientiousness']=data["conscientiousness"].map(lambda x:-2.87005 if x <-2.87005 else x)


# In[ ]:


#data['conscientiousness']=data["conscientiousness"].map(lambda x:2.84635 if x >2.84635 else x)


# In[ ]:





# ##### agreeableness

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['agreeableness'])


# In[ ]:


lower_limit = -0.287100 - 1.5 * 1.0999
lower_limit


# In[ ]:


upper_limit = 0.812800 + 1.5 * 1.0999
upper_limit


# In[ ]:


#data['agreeableness']=data["agreeableness"].map(lambda x:-1.93695 if x <-1.93695 else x)


# In[ ]:


#data['agreeableness']=data["agreeableness"].map(lambda x:2.46265 if x <2.46265 else x)


# In[ ]:





# ##### extraversion

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['extraversion'])


# In[ ]:


lower_limit = -0.7264 - 1.5 * 1.2768
lower_limit


# In[ ]:


upper_limit = 0.702700 + 1.5 * 1.2768
upper_limit


# In[ ]:


#data['extraversion']=data["extraversion"].map(lambda x:-2.6416 if x <-2.6416 else x)


# In[ ]:


#data['extraversion']=data["extraversion"].map(lambda x:2.6179 if x >2.6179 else x)


# In[ ]:





# ##### nueroticism

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['nueroticism'])


# In[ ]:


upper_limit = 0.5262 + 1.5 * 1.3944
upper_limit


# In[ ]:


#data['nueroticism']=data["nueroticism"].map(lambda x:2.6178 if x > 2.6178 else x)


# In[ ]:





# ##### openess_to_experience

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['openess_to_experience'])


# In[ ]:


lower_limit = -0.669200 - 1.5 * 1.1716
lower_limit


# In[ ]:


#data['openess_to_experience']=data["openess_to_experience"].map(lambda x:-2.4266 if x < -2.4266 else x)


# In[ ]:





# ##### ComputerProgramming

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=data['ComputerProgramming'])


# In[ ]:


lower_limit = 405 - 1.5 * 90
lower_limit


# In[ ]:


upper_limit = 495 + 1.5 * 90
upper_limit


# In[ ]:


#data['ComputerProgramming']=data["ComputerProgramming"].map(lambda x:270 if x < 270 else x)


# In[ ]:


#data['ComputerProgramming']=data["ComputerProgramming"].map(lambda x:630 if x > 630 else x)


# In[ ]:





# ### Feature engineering

# In[ ]:


data.columns


# In[ ]:


data['GraduationYear'] = data['GraduationYear'].astype(int)
data['DOB'] = data['DOB'].astype(int)
data['GraduationAge'] = data['GraduationYear'] - data['DOB']


# In[ ]:


data['GraduationAge'].value_counts()


# In[ ]:


data.info()


# In[ ]:


data['12_age'] = data['12graduation'] - data['DOB']


# In[ ]:


data['12graduation'].value_counts()


# In[ ]:


data.drop(['DOB'],axis=1,inplace = True)


# In[ ]:





# In[ ]:





# In[ ]:


new_data = pd.get_dummies(data,drop_first=True) 
new_data.shape


# In[ ]:


n_train = new_data.loc[:3998]


# In[ ]:


n_train.tail()


# In[ ]:


n_test = new_data.loc[3999:]


# In[ ]:


n_test.head()


# In[ ]:


x = n_train.drop(['Salary'],axis=1)
y = n_train['Salary']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 3,test_size=0.3)


# ### Distribution of Target Variable

# In[ ]:


sns.distplot(n_train['Salary'])


# ###### We can see data is not normally distributed and right skewed

# In[ ]:


n_train['Salary_log'] = np.log(n_train['Salary'])


# In[ ]:


sns.distplot(n_train['Salary_log'])


# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x=n_train['Salary'])


# In[ ]:


n_train.drop(n_train[n_train['Salary'] > 2000000].index, inplace = True)


# In[ ]:


sns.distplot(n_train['Salary'])


# In[ ]:


n_train['Salary_log'] = np.log(n_train['Salary'])


# In[ ]:


sns.distplot(n_train['Salary_log'])


# In[ ]:


n_train.info()


# In[ ]:


x = n_train.drop(['Salary_log','Salary'],axis=1)
y = n_train['Salary_log']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 3,test_size=0.3)


# In[ ]:





# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model1 = LinearRegression()
model1.fit(xtrain,ytrain)
model1_pred = model1.predict(xtest)


# In[ ]:


r_sqrd = r2_score(ytest,model1_pred)
r_sqrd


# In[ ]:


np.sqrt(mean_squared_error(ytest,model1_pred))


# In[ ]:


adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))
adj_r_sqrd


# In[ ]:


mape = mean_absolute_percentage_error(ytest,model1_pred)
mape


# In[ ]:





# In[ ]:





# In[ ]:





# ### RandomForest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


model2 = RandomForestRegressor()
model2.fit(xtrain,ytrain)
model2_pred = model2.predict(xtest)


# In[ ]:


r_sqrd = r2_score(ytest,model2_pred)
r_sqrd


# In[ ]:


np.sqrt(mean_squared_error(ytest,model2_pred))


# In[ ]:


adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))
adj_r_sqrd


# In[ ]:


mean_absolute_percentage_error(ytest,model2_pred)


# In[ ]:





# #### GradientBoostingRegressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


model3 = GradientBoostingRegressor()
model3.fit(xtrain,ytrain)
model3_pred = model3.predict(xtest)


# In[ ]:


r_sqrd =  r2_score(ytest,model3_pred)
r_sqrd


# In[ ]:


np.sqrt(mean_squared_error(ytest,model3_pred))


# In[ ]:


adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))
adj_r_sqrd


# In[ ]:


mean_absolute_percentage_error(ytest,model3_pred)


# In[ ]:


model3_pred


# In[ ]:


np.exp(model3_pred)


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


model4 = KNeighborsRegressor()
model4.fit(xtrain,ytrain)
model4_pred = model4.predict(xtest)


# In[ ]:


r_sqrd =  r2_score(ytest,model4_pred)
r_sqrd


# In[ ]:


adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))
adj_r_sqrd


# In[ ]:


np.sqrt(mean_squared_error(ytest,model4_pred))


# In[ ]:


mean_absolute_percentage_error(ytest,model4_pred)


# In[ ]:





# #### DecisionTreeRegressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


model5 = DecisionTreeRegressor()
model5.fit(xtrain,ytrain)
model5_pred = model5.predict(xtest)


# In[ ]:


r_sqrd =  r2_score(ytest,model5_pred)
r_sqrd


# In[ ]:


adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))
adj_r_sqrd


# In[ ]:


np.sqrt(mean_squared_error(ytest,model5_pred))


# In[ ]:


mean_absolute_percentage_error(ytest,model5_pred)


# In[ ]:





# #### PolynomialFeatures

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x_poly,y,test_size=0.3,random_state=3)


# In[ ]:


model6 = LinearRegression()
model6.fit(x_train, y_train)
model6_pred = model6.predict(x_test)


# In[ ]:


r_sqrd =  r2_score(y_test,model6_pred)
r_sqrd


# In[ ]:


adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))
adj_r_sqrd


# In[ ]:


np.sqrt(mean_squared_error(y_test,model6_pred))


# In[ ]:


mean_absolute_percentage_error(y_test,model6_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from scipy.stats import zscore


# In[ ]:


z_data = n_train.copy()


# In[ ]:


z_data.head()


# In[ ]:


X = z_data.drop(['Salary','Salary_log'],axis=1)


# In[ ]:


Y = z_data['Salary_log']


# In[ ]:


X[['10percentage', '12percentage', 'collegeGPA', 'English', 'Logical', 'Quant', 'Domain', 'ComputerProgramming', 'conscientiousness', 'agreeableness', 'extraversion','nueroticism', 'openess_to_experience']] = X[['10percentage', '12percentage', 'collegeGPA', 'English', 'Logical', 'Quant', 'Domain', 'ComputerProgramming', 'conscientiousness', 'agreeableness', 'extraversion','nueroticism', 'openess_to_experience']].apply(zscore)


# In[ ]:


X.head()


# In[ ]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,random_state = 3, test_size = 0.3)


# In[ ]:


Xtrain.head()


# In[ ]:


model7 = LinearRegression()
model7.fit(Xtrain,Ytrain)
model7_pred = model7.predict(Xtest)


# In[ ]:


r_sqrd =  r2_score(Ytest,model7_pred)
r_sqrd


# In[ ]:


adj_r_sqrd = 1-(1-r_sqrd)*((1178-1)/(1178-58-1))
adj_r_sqrd


# In[ ]:


np.sqrt(mean_squared_error(Ytest,model7_pred))


# In[ ]:


mean_absolute_percentage_error(Ytest,model7_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### VIF

# In[ ]:


def multi_collinearity(X):
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    VIF=pd.DataFrame()
    VIF['columns']=X.columns
    VIF['vif']=vif
    return(VIF)


# In[ ]:


v = multi_collinearity(n_train)


# In[ ]:


v[v['vif']>5]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


GradientBoostingRegressor
parameters = {
    'n_estimators': [50,100,150],
    'max_depth': [5,8,12,16],
    'min_samples_split' : [50,100,120],
    'min_samples_leaf': [20,30,50],
    'max_features' : [6,40,20,60,80,100,104]}
model9=GradientBoostingRegressor()
clf = GridSearchCV(model9,parameters,cv=10)
clf.fit(xtrain,ytrain)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[ ]:


num_data = data._get_numeric_data()


# In[ ]:


num_data.head()


# In[ ]:


X = add_constant(num_data)


# In[ ]:


vif = pd.DataFrame([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# In[ ]:


vif

