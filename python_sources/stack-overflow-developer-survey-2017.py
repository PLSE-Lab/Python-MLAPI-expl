#!/usr/bin/env python
# coding: utf-8

# Every year, Stack Overflow conducts a massive survey of people on the site, covering all sorts of information like programming languages, salary, code style and various other information. This year, they amassed more than 64,000 responses fielded from 213 countries.
# 
# ### Dataset
# 
# The dataset is collected from `Kaggle`

# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


pd.options.display.max_columns = 999
dataset = pd.read_csv("../input/so-survey-2017/survey_results_public.csv")
dataset.shape


# In[ ]:


dataset.head()


# In[ ]:


countries = dataset['Country'].unique()
print("Total Country: {0}".format(len(countries)))


# In[ ]:


country_freq = {}
for cnt in dataset['Country']:
  if cnt in country_freq:
    country_freq[cnt] += 1
  else:
    country_freq[cnt] =1


# ### Lets see from which country how many developers responsed for the survey

# In[ ]:


country_series = pd.Series(country_freq)
plt.figure(figsize=(40,15))
country_series.plot.bar()
plt.xlabel("Country Name")
plt.ylabel("Country Frequency")
plt.title("Country Frequency Graph from Stackoverflow dataset")
plt.show()


# In[ ]:


university = dataset['University'].value_counts()
university


# In[ ]:


plt.figure(figsize=(10,7))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'University', data=dataset)
plt.xlabel("University Education Type")
plt.ylabel("Education Type Frequency")
plt.title("Education Type Frequency Graph from Stackoverflow dataset")
plt.show()


# ### Here found 4 types of education which are No, Yes,full-time, Yes, part-time, I prefer not to say. Maximum developers has no degree from universities. Lets findout education type wise country frequnecy. The below historgram will show first 50 frequencies of countries where developer has no university degree.

# In[ ]:


no_university_cnt = dataset[dataset['University']=='No']['Country']
no_university_cnt_frq = no_university_cnt.value_counts()
no_university_cnt_frq
plt.figure(figsize=(80,40))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'Country', data=dataset[dataset['University']=='No'].iloc[:50])
plt.xlabel("Country")
plt.ylabel("'No' type University Frequency")
plt.title("'No' type University wise country frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# ### Below histogram will show country first 50 frequencies for the developer who went university for full time

# In[ ]:


no_university_cnt = dataset[dataset['University']=='Yes, full-time']['Country']
no_university_cnt_frq = no_university_cnt.value_counts()
no_university_cnt_frq
plt.figure(figsize=(40,10))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'Country', data=dataset[dataset['University']=='Yes, full-time'].iloc[:50])
plt.xlabel("Country")
plt.ylabel("'Yes, full-time' Type University Frequency")
plt.title("'Yes, full-time' type University wise country frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# ### Here we see that in India developers are very much carefull about their education. after India it is US, Germany and so on.

# In[ ]:


no_university_cnt = dataset[dataset['University']=='Yes, part-time']['Country']
no_university_cnt_frq = no_university_cnt.value_counts()
no_university_cnt_frq
plt.figure(figsize=(40,10))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'Country', data=dataset[dataset['University']=='Yes, part-time'].iloc[:50])
plt.xlabel("Country")
plt.ylabel("'Yes, part-time' Type University Frequency")
plt.title("'Yes, part-time' type University wise country frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


no_university_cnt = dataset[dataset['University']=='I prefer not to say']['Country']
no_university_cnt_frq = no_university_cnt.value_counts()
no_university_cnt_frq
plt.figure(figsize=(40,10))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'Country', data=dataset[dataset['University']=='I prefer not to say'].iloc[:50])
plt.xlabel("Country")
plt.ylabel("'I prefer not to say' Type University Frequency")
plt.title("'I prefer not to say' type University wise country frequency graph from Stackoverflow dataset")
plt.xticks


# ### Here we see the frequency graph for country where developer are not willing to expressa about their university or education. From the graph it is clear that India, Germany, US, Pakistan, Phillippines has so many people who are not so willing to say about their education.

# In[ ]:


dataset['EmploymentStatus'].value_counts()


# ### Lets see which country is the maximum for EmploymentStatus

# In[ ]:


EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Employed full-time']['Country'].value_counts()
print("Maximum Full time employee country: {0}".format(EmploymentStatusCnt.index[0]))


# In[ ]:


EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Independent contractor, freelancer, or self-employed']['Country'].value_counts()
print("Maximum (Independent contractor, freelancer, or self-employed) employee country: {0}".format(EmploymentStatusCnt.index[0]))


# In[ ]:


EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Employed part-time']['Country'].value_counts()
print("Maximum (Employed part-time) employee country: {0}".format(EmploymentStatusCnt.index[0]))


# In[ ]:


EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Not employed, and not looking for work']['Country'].value_counts()
print("Maximum (Not employed, and not looking for work ) employee country: {0}".format(EmploymentStatusCnt.index[0]))


# In[ ]:


EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Not employed, but looking for work']['Country'].value_counts()
print("Maximum (Not employed, but looking for work ) employee country: {0}".format(EmploymentStatusCnt.index[0]))


# In[ ]:


EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'I prefer not to say']['Country'].value_counts()
print("Maximum (I prefer not to say) employee country: {0}".format(EmploymentStatusCnt.index[0]))


# In[ ]:


EmploymentStatusCnt = dataset[dataset['EmploymentStatus']== 'Retired']['Country'].value_counts()
print("Maximum (Retired) employee country: {0}".format(EmploymentStatusCnt.index[0]))


# In[ ]:


FormalEducationFreq = dataset['FormalEducation'].value_counts(normalize=True)*100
FormalEducationFreq


# In[ ]:


data_labels = ["Bachelor's degree","Master's degree","Some college/university study without earning a bachelor's degree","Secondary school","Doctoral degree","I prefer not to answer","Primary/elementary school","Professional degree","I never completed any formal education "]
plt.figure(figsize=(10,8))
plt.pie(FormalEducationFreq,labels=data_labels,autopct='%1.1f%%',)


# ### The pie chart is for the developer who completed formal education, Most developer has only a bachelor degree and only 21.7% has a master degree. The amount of secondary school is 11.5%.

# In[ ]:


FormalEducationBachelor = dataset[dataset['FormalEducation'] == "Bachelor's degree"]["Country"].value_counts().iloc[:50]
plt.figure(figsize=(30,10))
sns.set(style="whitegrid")
FormalEducationBachelor.plot.bar()
plt.xlabel("FormalEducation")
plt.ylabel("Bachelor's degree Type Country Frequency")
plt.title("Bachelor's degree wise country frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


FormalEducationBachelor = dataset[dataset['FormalEducation'] == "Master's degree"]["Country"].value_counts().iloc[:50]
plt.figure(figsize=(30,10))
sns.set(style="whitegrid")
FormalEducationBachelor.plot.bar()
plt.xlabel("FormalEducation")
plt.ylabel("Master's degree Type Country Frequency")
plt.title("Master's degree wise country frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


MajorUndergradFreq = dataset['MajorUndergrad'].value_counts(normalize=True)*100
MajorUndergradFreq


# In[ ]:


data_labels = ["Computer science or software engineering","Computer engineering or electrical/electronics engineering","Computer programming or Web development","Information technology, networking, or system administration","A natural science","A non-computer-focused engineering discipline","Mathematics or statistics","Something else","A humanities discipline","A business discipline","Management information systems","Fine arts or performing arts","A social science","I never declared a major","Psychology","A health science"]
plt.figure(figsize=(25,8))
plt.pie(MajorUndergradFreq,labels=data_labels,autopct='%1.1f%%',)


# In[ ]:


MajorUndergradCSECountryFreq = dataset[dataset['MajorUndergrad'] == "Computer science or software engineering"]["Country"].value_counts().iloc[:40]
plt.figure(figsize=(30,10))
MajorUndergradCSECountryFreq.plot.bar()
plt.xlabel("Country")
plt.ylabel("CSE undergrade frequency")
plt.title("CSE undergrade frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


MajorUndergradCSECountryFreq = dataset[dataset['MajorUndergrad'] == "Computer engineering or electrical/electronics engineering"]["Country"].value_counts().iloc[:40]
plt.figure(figsize=(30,10))
MajorUndergradCSECountryFreq.plot.bar()
plt.xlabel("Country")
plt.ylabel("CSE/EEE undergrade frequency")
plt.title("CSE/EEE undergrade  frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


YearsCodedJobFreq = dataset["YearsCodedJob"].value_counts()
YearsCodedJobFreq


# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'YearsCodedJob', data=dataset)
plt.xlabel("YearsCodedJob")
plt.ylabel("YearsCodedJob Frequency")
plt.title("YearsCodedJob frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


dataset["DeveloperType"].isnull().sum()


# In[ ]:


dataset["DeveloperType"].fillna("No Type", inplace=True)
dataset["DeveloperType"].isnull().sum()


# In[ ]:


dataset["CareerSatisfaction"].isnull().sum()


# In[ ]:


dataset["CareerSatisfaction"].fillna(0, inplace=True)
dataset["CareerSatisfaction"].isnull().sum()


# In[ ]:


CareerSatisfactionFreq = dataset["CareerSatisfaction"].value_counts(normalize=True)*100
CareerSatisfactionFreq


# In[ ]:


data_labels = ["8.0","7.0","0.0","9.0","10.0","6.0","5.0","4.0","3.0","2.0","1.0"]
plt.figure(figsize=(25,8))
plt.pie(CareerSatisfactionFreq,labels=data_labels,autopct='%1.1f%%',)
plt.show()


# ### Lets find out country, people as their job statisfiction. For now lets findout people who are most satisfied and who are most less satisfied

# In[ ]:


MostCareerSatisfactionCountry = dataset[dataset["CareerSatisfaction"] == 8.0]["Country"].value_counts().iloc[:50]
MostCareerSatisfactionCountry


# In[ ]:


plt.figure(figsize=(30,10))
MostCareerSatisfactionCountry.plot.bar()
plt.xlabel("Country")
plt.ylabel("Most CareerSatisfaction Country frequency")
plt.title("Most CareerSatisfaction Country frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


MostCareerSatisfactionYearsCodedJob = dataset[dataset["CareerSatisfaction"] == 8.0]["YearsCodedJob"].value_counts().iloc[:50]
MostCareerSatisfactionYearsCodedJob


# ### Lets see an overall view of CareerSatisfiction and YearsCodeJob

# In[ ]:


plt.figure(figsize=(22,10))
sns.set(style="whitegrid")
ax = sns.boxplot(x = 'YearsCodedJob',y="CareerSatisfaction", data=dataset.sort_values(by="YearsCodedJob"))
plt.xlabel("YearsCodedJob")
plt.ylabel("CareerSatisfaction")
plt.title("Boxplot graph for CareerSatisfaction and YearsCodedJob from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# ### Most satisfied coders yearsCodeJob frequency

# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'YearsCodedJob', data=dataset[dataset["CareerSatisfaction"] == 8.0].sort_values(by="YearsCodedJob"))
plt.xlabel("YearsCodedJob")
plt.ylabel("CareerSatisfaction (8.0 )YearsCodedJob Frequency")
plt.title("Most CareerSatisfaction (8.0) YearsCodedJob frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# ### Find out most seniors country who are most satisfied

# In[ ]:


mostSeniorCarrerSatisfyCountry = dataset[(dataset["CareerSatisfaction"] == 8.0) & (dataset["YearsCodedJob"] == '20 or more years')]['Country'].value_counts(normalize=True)*100
mostSeniorCarrerSatisfyCountry


# In[ ]:


plt.figure(figsize=(25,10))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'Country', data=dataset[(dataset["CareerSatisfaction"] == 8.0) & (dataset["YearsCodedJob"] == '20 or more years')].sort_values(by="Country"))
plt.xlabel("Country")
plt.ylabel("Most senior (20 years or more)CareerSatisfaction (8.0 ) Frequency")
plt.title("Most senior (20 years or more)CareerSatisfaction (8.0 ) frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# ### Less satisfied coders yearsCodeJob frequency

# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'YearsCodedJob', data=dataset[dataset["CareerSatisfaction"] == 1.0].sort_values(by="YearsCodedJob"))
plt.xlabel("YearsCodedJob")
plt.ylabel("CareerSatisfaction (1.0 )YearsCodedJob Frequency")
plt.title("Less CareerSatisfaction (1.0) YearsCodedJob frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# ### From the above graph we can observe that who has 1-2 year experience or less then 1 year experience are less satisfy with their career.

# In[ ]:


plt.figure(figsize=(25,10))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'Country', data=dataset[(dataset["CareerSatisfaction"] == 1.0) & (dataset["YearsCodedJob"] == '1 to 2 years')].sort_values(by="Country"))
plt.xlabel("Country")
plt.ylabel("(1 to 2 years) CareerSatisfaction (1.0 ) Frequency")
plt.title("(1 to 2 years) CareerSatisfaction (1.0 ) frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


plt.figure(figsize=(28,11))
sns.set(style="whitegrid")
ax = sns.countplot(x = 'Country', data=dataset[(dataset["CareerSatisfaction"] == 1.0) & (dataset["YearsCodedJob"] == 'Less than a year')].sort_values(by="Country"))
plt.xlabel("Country")
plt.ylabel("(Less than a year) CareerSatisfaction (1.0 ) Frequency")
plt.title("(Less than a year) CareerSatisfaction (1.0 ) frequency graph from Stackoverflow dataset")
plt.xticks(rotation=45)
plt.show()


# ### From the graph we see that in India amount of who has one year experience of job and career satisfiction only 1 is maximum.
# 
# ### Now lets see about the JobSatisfaction frequency

# In[ ]:


JobSatisfactionFreq = dataset["JobSatisfaction"].value_counts(normalize=True)*100
JobSatisfactionFreq


# In[ ]:


data_labels = ["8.0","7.0","9.0","6.0","10.0","5.0","4.0","3.0","2.0","0.0","1.0"]
plt.figure(figsize=(25,8))
plt.pie(JobSatisfactionFreq,labels=data_labels,autopct='%1.1f%%',)
plt.show()


# ### Lets see which country has most job JobSatisfaction employee and lets explore some information about them as like (formal education, university type etc)

# In[ ]:


JobSatisfactionIndex = dataset["JobSatisfaction"].value_counts().index
JobSatisfactionIndex


# In[ ]:


JobSatisfactionDF = dataset[dataset['JobSatisfaction']== 10.0][['Country','FormalEducation','University','YearsCodedJob']]
JobSatisfactionDF.head()


# In[ ]:


JobSatisfactionDFCountryFreq = JobSatisfactionDF['Country'].value_counts(normalize=True)*100
JobSatisfactionDFCountryFreq.iloc[:20]


# In[ ]:


JobSatisfactionDFFormalEducationFreq = JobSatisfactionDF['FormalEducation'].value_counts(normalize=True)*100
JobSatisfactionDFFormalEducationFreq.iloc[:20]


# In[ ]:


JobSatisfactionDFFormalColFreq = JobSatisfactionDF['University'].value_counts(normalize=True)*100
JobSatisfactionDFFormalColFreq.iloc[:20]


# In[ ]:


JobSatisfactionDFFormalColFreq = JobSatisfactionDF['YearsCodedJob'].value_counts(normalize=True)*100
JobSatisfactionDFFormalColFreq.iloc[:20]


# ### The above four analysis is for who has most job satisfiction value. We have seen that `US` has the most job satisfiction people, also we see that bachelor's degree is most who have formal education. Now lets explore the country `US` and find out people percantage about formal education, university, yearcodejob etc

# In[ ]:


JobSatisfactionDFExplore = JobSatisfactionDF[JobSatisfactionDF['Country']=='United States'][['FormalEducation','University','YearsCodedJob']]
JobSatisfactionDFExplore.head()


# In[ ]:


JobSatisfactionDFExplore['FormalEducation'].value_counts(normalize=True)*100


# In[ ]:


JobSatisfactionDFExplore['University'].value_counts(normalize=True)*100


# In[ ]:


data_labels = ["No","Yes, full-time","Yes, part-time","I prefer not to say"]
plt.figure(figsize=(25,8))
plt.pie(JobSatisfactionDFExplore['University'].value_counts(normalize=True)*100,labels=data_labels,autopct='%1.1f%%',)
plt.show()


# In[ ]:


JobSatisfactionDFExplore['YearsCodedJob'].value_counts(normalize=True)*100


# ### Some observation
# 
# * Among so many job satisfcation value 8 is the most, 7 is second, 9 is third and 10 is 5th
# * Most Job satisfaction (10.0) country is United States
# * In United States who have most job satisfaction (10.0) and also have formal education , has maximum bachelor degree (55.1%) and after them college degree has maximum (20.23%)
# * In United States who have most job satisfaction (10.0), maximum of them has no university history ans the amount is 90% and aonly 5.76% has full time university history
# * In United States who have most job satisfaction (10.0), maximum of them has 20 or more years of experience of coding and after them 2-3 year 

# In[ ]:


JobSatisfactionDF_8 = dataset[dataset['JobSatisfaction']== 8.0][['Country','FormalEducation','University','YearsCodedJob']]
JobSatisfactionDF_8.head()


# In[ ]:


JobSatisfactionDFCountryFreq = JobSatisfactionDF_8['Country'].value_counts(normalize=True)*100
JobSatisfactionDFCountryFreq.iloc[:20]


# In[ ]:


JobSatisfactionDFFormalEducationFreq = JobSatisfactionDF_8['FormalEducation'].value_counts(normalize=True)*100
JobSatisfactionDFFormalEducationFreq.iloc[:20]


# In[ ]:


JobSatisfactionDFExplore['University'].value_counts(normalize=True)*100


# In[ ]:


data_labels = ["No","Yes, full-time","Yes, part-time","I prefer not to say"]
plt.figure(figsize=(25,8))
plt.pie(JobSatisfactionDFExplore['University'].value_counts(normalize=True)*100,labels=data_labels,autopct='%1.1f%%',)
plt.show()


# In[ ]:


JobSatisfactionDFExplore = JobSatisfactionDF_8[JobSatisfactionDF_8['Country']=='United States'][['FormalEducation','University','YearsCodedJob']]
JobSatisfactionDFExplore.head()


# In[ ]:


JobSatisfactionDFExplore['YearsCodedJob'].value_counts(normalize=True)*100


# In[ ]:


data_labels = ["20 or more years","2 to 3 years","1 to 2 years","3 to 4 years","4 to 5 years","Less than a year","5 to 6 years","6 to 7 years","10 to 11 years","9 to 10 years","7 to 8 years","8 to 9 years","15 to 16 years","14 to 15 years","11 to 12 years","12 to 13 years","16 to 17 years","13 to 14 years","19 to 20 years","17 to 18 years","18 to 19 years"]
plt.figure(figsize=(25,8))
plt.pie(JobSatisfactionDFExplore['YearsCodedJob'].value_counts(normalize=True)*100,labels=data_labels,autopct='%1.1f%%',)
plt.show()


# In[ ]:




