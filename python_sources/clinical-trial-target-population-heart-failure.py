
# coding: utf-8

# In[3]:

import requests
import pandas as pd


# In[4]:

url1 = "http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/admissions?limit=50000&offset=0"
conn1 = requests.get(url1).json()
file1 = conn1['admissions']
admissions =  pd.DataFrame.from_dict(file1, orient='columns')
admissions.head(1)


# In[5]:

admissions['diagnosis'].str.contains('HEART').value_counts()


# In[6]:

'''Dropping rows not having heart failure'''
heartpatients = admissions[admissions['diagnosis'].str.contains('heart', case = False) & admissions['diagnosis'].notnull()]


# In[7]:

url1 = "http://ec2-54-88-151-77.compute-1.amazonaws.com:3003/v1/conditions?limit=50000&offset=0"
conn1 = requests.get(url1).json()
file1 = conn1['conditions']
conditions =  pd.DataFrame.from_dict(file1, orient='columns')
conditions.head(1)


# In[8]:

conditions['name'].str.contains('HEART', case=False).value_counts()


# In[9]:

#Dataframe with conditions of heart alone 652 rows
heartconditions = conditions[conditions['name'].str.contains('HEART', case=False) & conditions['name'].notnull()]
heartconditions.head(1)


# In[10]:

url1 = "http://ec2-54-88-151-77.compute-1.amazonaws.com:3003/v1/eligibilities?limit=50000&offset=0"
conn1 = requests.get(url1).json()
file1 = conn1['eligibility']
eligibilities =  pd.DataFrame.from_dict(file1, orient='columns')
eligibilities.head(1)


# In[11]:

eligibilities = eligibilities.drop(['gender_based','gender_description','id','index','criteria','healthy_volunteers'],axis=1)    


# In[12]:

#missing : NONE
eligibilities.apply(lambda x: sum(x.isnull()))


# In[186]:

#eligibilities.sampling_method.unique()


# In[188]:

#eligibilities.population.str.contains('').sum() # dropping this as well


# In[13]:

eligibilities = eligibilities.drop(['population','sampling_method'],axis=1)


# In[14]:

clinicaldf = pd.merge(eligibilities,heartconditions, on=['nct_id'], how = 'inner')


# In[16]:

clinicaldf.head(1)


# In[62]:

target = clinicaldf.drop(['downcase_name','id','index','name'],axis=1)


# In[63]:


target.head(1)


# In[41]:

target.apply(lambda x: sum(x.isnull()))


# In[42]:

target.gender.value_counts()


# In[43]:

target.minimum_age.value_counts()


# In[44]:

target.maximum_age.value_counts()


# In[64]:

target[target.maximum_age=='30 Days']


# In[65]:

#replacing 30 Days with fraction of year = .08 Years (1 month)
target['maximum_age'] = target['maximum_age'].replace(['30 Days'], '0.08 Years')

'''Replacing N/A with 0 in minimum age'''
target['minimum_age'] = target['minimum_age'].replace(['N/A'], '0')

'''Replacing N/A with 150 in maximum age'''
target['maximum_age'] = target['maximum_age'].replace(['N/A'], '150 Years')

'''Split and Convert string to Float'''
target["minimum_age"] = target["minimum_age"].str.replace(" Years","").astype(float)

target["maximum_age"] = target["maximum_age"].str.replace(" Years","").astype(float)


# In[66]:

target.maximum_age.value_counts()


# In[54]:

heartpatients.head(1)


# In[56]:

#Importing patients and combining with heartpatients of admissions
url = "http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/patients?limit=50000&offset=0"
conn = requests.get(url).json()
file = conn['patients']
patients =  pd.DataFrame.from_dict(file, orient='columns')


# In[57]:

patientdf = pd.merge(patients,heartpatients, on=['subject_id'], how = 'inner')
#len(patientdf)


# In[58]:

#Selecting columns
inpatient = patientdf[['subject_id','dob','dod','gender']].copy()
inpatient['gender'] = inpatient['gender'].replace(['M'], '1')
inpatient['gender'] = inpatient['gender'].replace(['F'], '0')
inpatient['gender'] = inpatient.gender.astype(int)
inpatient['dod'] = inpatient.dod.fillna('')
inpatient['dod'] = inpatient['dod'].replace('', '2168-10-28 00:00:00')
inpatient['dob'] = pd.to_datetime(inpatient['dob'])
inpatient['dod'] = pd.to_datetime(inpatient['dod'])
inpatient['age'] = inpatient['dod'] - inpatient['dob']
inpatient['age'] = inpatient.age.dt.days
inpatient['age'] = inpatient['age']/365
inpatient = inpatient.drop(['dob','dod'],axis = 1)

#Dropping rows whose age is neagtive and whose age is above 150
inpatient = inpatient.drop(inpatient[inpatient.age < 0].index)
inpatient = inpatient.drop(inpatient[inpatient.age > 130].index)

inpatient.head(1)


# In[59]:

get_ipython().magic('matplotlib inline')
inpatient.age.hist(bins = 50)


# In[60]:

inpatient.gender.hist()


# In[67]:

#Replacing male with 1 in target| Replacing female with 1 in target| Replacing ALL with 2 in target
target['gender'] = target['gender'].replace(['Male'], '1')
target['gender'] = target['gender'].replace(['Female'], '0')
target['gender'] = target['gender'].replace(['All'], '2')
target.maximum_age = target.maximum_age.astype(float)
target.minimum_age = target.minimum_age.astype(float)
target.gender = target.gender.astype(int)


# In[68]:

target.gender.value_counts()


# In[72]:

#Selecting that specific study needing a female patient
#target[target.gender == 0]


# In[783]:

#adding a study NCTID to inpatient
#inpatient['NCT00007358'] = ''


# In[ ]:




# In[741]:

#Checking eligibility for study NCT00007358
def eligible(gender,age):
    if gender == 0:
        if age >=16 and age <=50:
            return 1
        else:
            return 0
    else:
        return 0


# In[73]:

target = (target.drop_duplicates())


# In[74]:

target[target['gender']==0]


# In[788]:

#for index, row in inpatient.iterrows():
#    (inpatient['NCT00007358'].iloc[index]) = eligible((inpatient['gender'].iloc[index]),(inpatient['age'].iloc[index]))
#    print(inpatient['NCT00007358'].iloc[index])

#Applying lambda to generate eligibility of a study
inpatient['NCT00007358'] = inpatient.apply(lambda x: eligible(x['gender'], x['age']), axis=1)


# In[791]:

womenheartstudy = inpatient[inpatient['NCT00007358']==1]


# In[870]:

womenheartstudy['age'].hist(bins=40)


# In[848]:

for x in target:
   print(x)
#target['nct_id'][2]


# In[855]:

inpatient = inpatient.drop(['NCT00382525','NCT00542854'],axis=1)


# In[857]:

#Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")


# In[864]:

sns.regplot(womenheartstudy['age'],womenheartstudy['NCT00007358'])


# In[874]:

sns.boxplot(womenheartstudy['age'])


# In[77]:

inpatient=inpatient.reset_index(drop=True)


# In[80]:

for x in range(0,inpatient.shape[0]):
    for j in range(0,target.shape[0]):
        if (target.loc[j,'gender']==2) & (target.loc[j,'minimum_age']<=inpatient.loc[x,'age']<=target.loc[j,'maximum_age']):
            inpatient.loc[x,target.loc[j,'nct_id']]=1
        elif (target.loc[j,'gender']==inpatient.loc[x,'gender']) & (target.loc[j,'minimum_age']<=inpatient.loc[x,'age']<=target.loc[j,'maximum_age']):
            inpatient.loc[x,target.loc[j,'nct_id']]=1 
 


# In[81]:

inpatient.head(5)


# In[82]:

target.head()


# In[84]:

target[target.gender==0]


# In[95]:

target[target.gender==1]


# In[99]:

#inpatient.drop_duplicates()


# In[100]:

inpatient.info()


# In[102]:

inpatient[inpatient.NCT00829842==1.0]


# In[103]:

target[target.nct_id == 'NCT00829842']


# In[108]:

inpatient['NCT00829842'].count()


# In[ ]:



