#!/usr/bin/env python
# coding: utf-8

# Descriptive Statistics on Employee Attrition

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
datafile= 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
data = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()


# The above is the sample data of the entire CSV

# In[ ]:


data.describe()


# For every Numberical Attribute the mean , standard deviation , minimum value , maximum value ,first quartile , median and Third quartile is given above

# In[ ]:


data.boxplot('Age',by='Attrition')


# Box plot is used to analyze the data distribution of the data more effectively.It also provides more simpler manner of avaoiding outliers.
# 
# From the above boxplot ,
# 75% of the people above age 32 tends to leave the company.

# In[ ]:


AttributeName='EducationField'
gdata=data.groupby([AttributeName,'Attrition']).count().reset_index()
gdata


# In[ ]:


prev=''
r1=[]
yes=[]
no=[]
for index, row in gdata.iterrows():
    if prev != row[AttributeName]:
        r1.append(row[AttributeName])
        prev=row[AttributeName]
    if row['Attrition'] == 'Yes':
        yes.append(row['Age'])
    else:
        no.append(row['Age'])


# In[ ]:


total=[]
ypercent=[]
npercent=[]
for i in range(len(yes)):
    total.append(int(yes[i])+int(no[i]))
    ypercent.append(int(yes[i])*100/total[i])
    npercent.append(int(no[i])*100/total[i])


# In[ ]:


ndf={AttributeName:r1,'Yes':yes , 'No' :no}
kdf= {AttributeName:r1,'Yes':ypercent , 'No':npercent}
fdata=pd.DataFrame(ndf)
fdata


# In[ ]:


fdata.plot.bar(x=AttributeName,stacked=True)


# The above is the bar graph of the educational field.From which we can clearly inference that there are large amount of people work is of Life Science background.Futher we have to normalise the above data for clear inference.

# In[ ]:


ndata=pd.DataFrame(kdf)
ndata


# In[ ]:


ndata.plot.bar(x=AttributeName,stacked=True)


# Clearly it can be scence that about 25% of people working in the Human Resource and Technical Degree tends to leave the company.By minimum 15% of the employee always tends to leave the company in each field.
