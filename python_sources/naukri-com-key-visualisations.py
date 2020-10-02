#!/usr/bin/env python
# coding: utf-8

# # 1. Importing libraries and datasets

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[ ]:


pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',None)
df=pd.read_csv('../input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')


# In[ ]:


df.head()


# # 2. Data wrangling

# Let us check the type of data which we are given with.

# In[ ]:


df.dtypes


# As we can see, all the data types have been encoded as objects. However, upon eyeballing into the data, we clearly see that we have some data that will be preferred to be in another form. For example, it will be better if we change the Crawl Timestamp into timestamp datatype. Similarly, the Job Experience column and salary can be shown as integer datatypes.

# ## Converting the crawl timestamp column into timestamp datatype

# Let us remove the unwanted **+0000** in the timestamp column first.

# In[ ]:


for i in range(len(df)):
    df['Crawl Timestamp'][i]=df['Crawl Timestamp'][i].replace('+0000','')
    i+=1


# In[ ]:


df.head()


# As we can see, the timestamp is in a more understandable form and can be converted into the required timestamp data frame using pandas.to_datetime method .

# In[ ]:


df['Crawl Timestamp']=pd.to_datetime(df['Crawl Timestamp'])


# In[ ]:


df['Crawl Timestamp'][:5]


# As we can see, the above Crawl Timestamp column is in the required datetime format.

# ## Missing values

# We need to check for the presence of any missing values and take care of these missing values. In this case, we will simply drop the missing values as we are primarily dealing with data visualisation and dropping few entries will not severly harm any calculations.

# In[ ]:


df.isna().any()


# Let us check the pattern and frequency of missing values in each column using a seaborn heatmap.

# In[ ]:


sns.heatmap(df.isnull(),cbar=True,cmap='gnuplot')


# As we can clearly see, columns such as key skills and role category have the most amount of missing values. Let us see how many exactly are missing.

# In[ ]:


cols=[ 'Job Title', 'Job Salary',
       'Job Experience Required', 'Key Skills', 'Role Category', 'Location',
       'Functional Area', 'Industry', 'Role']
empty_vals=[]
for col in cols:
    print('Number of missing values in {}: {}'.format(col,df[col].isna().value_counts()[1]))
    empty_vals.append(df[col].isna().value_counts()[1])
print('Total entries:{}'.format(len(df)))


# In[ ]:


missing_df=pd.DataFrame(columns=['Column','Missing values'])
missing_df['Column']=cols
missing_df['Missing values']=empty_vals
missing_df.sort_values(by='Missing values',inplace=True,ascending=False)
missing_df.index=missing_df.Column
missing_df.drop('Column',axis=1,inplace=True)
missing_df


# In[ ]:


my_colors = 'rgbkymc'  #red, green, blue, black, etc.
ax=missing_df.plot(kind='bar',figsize=(20,10),rot=90,width=0.8,color=my_colors)
ax.set_title("Number of missing values in the dataframe",size=20)
ax.set_ylabel('Number of missing values',size=18)
ax.set_xlabel('Column',size=18)


#For annotating the bars

for i in ax.patches:
    ax.text(i.get_x()+0.045,i.get_height()+2,str(round((i.get_height()), 2)),
            rotation=0,fontsize=15,color='black')
    


# The above plot makes it even easier to understand which fields have high missing values.
# 
# As we can see, the number of missing values in each column is not much. Even if we drop all the missing values, we should be able to get a good deptiction of the general data trend. Let us now drop all the missing values in the dataframe.

# In[ ]:


df.dropna(axis=0,inplace=True)


# In[ ]:


df.isna().any()


# In[ ]:


df.size


# As we can see, the size of the dataframe reduced from 30000 to 297055. The loss of data isn't much and can be worked with now.

# ## Job Title

# Let us check the job title field to see any intersting insights and perform the much required data cleaning which is to be done.

# In[ ]:


df['Job Title'].describe()


# In[ ]:


df['Job Title'].value_counts()[0:10]


# As we can see, the column is extremely unclean here. We must find some way to group these job titles into a common and easily understandable blocks.

# In[ ]:


df_temp=df.copy()


# Here, what we plan to do is find keywords such as engineer, analyst,HR, executive ,customer care and then replace the other words with these common job roles to align the data we have. Let us make the important list of keywords that we plan to use for sorting the data.
# 
# 

# In[ ]:


df_temp.loc[df_temp['Job Title'].str.contains('Planner', case=False), 'Cleaned Title'] = 'Planner'


# In[ ]:


df_temp.loc[df_temp['Job Title'].str.contains('Analyst', case=False), 'Cleaned Title'] = 'Analyst'
df_temp.loc[df_temp['Job Title'].str.contains('Analytics', case=False), 'Cleaned Title'] = 'Analyst'
df_temp.loc[df_temp['Job Title'].str.contains('Develop',case=False),'Cleaned Title']='Software/Web/App Developer'
df_temp.loc[df_temp['Job Title'].str.contains('Software',case=False),'Cleaned Title']='Software/Web/App Developer'
df_temp.loc[df_temp['Job Title'].str.contains('Web',case=False),'Cleaned Title']='Software/Web/App Developer'
df_temp.loc[df_temp['Job Title'].str.contains('App',case=False),'Cleaned Title']='Software/Web/App Developer'
df_temp.loc[df_temp['Job Title'].str.contains('Designer', case=False), 'Cleaned Title'] = 'Design and Creativity'
df_temp.loc[df_temp['Job Title'].str.contains('Animation', case=False), 'Cleaned Title'] = 'Design and Creativity'
df_temp.loc[df_temp['Job Title'].str.contains('Content', case=False), 'Cleaned Title'] = 'Design and Creativity'
df_temp.loc[df_temp['Job Title'].str.contains('Consultant', case=False), 'Cleaned Title'] = 'Consultancy'
df_temp.loc[df_temp['Job Title'].str.contains('Risk', case=False), 'Cleaned Title'] = 'Risk analyst'
df_temp.loc[df_temp['Job Title'].str.contains('Call', case=False), 'Cleaned Title'] = 'Customer service'
df_temp.loc[df_temp['Job Title'].str.contains('Support',case=False),'Cleaned Title']='Customer service'
df_temp.loc[df_temp['Job Title'].str.contains('Customer support',case=False),'Cleaned Title']='Customer service'
df_temp.loc[df_temp['Job Title'].str.contains('Engineer',case=False),'Cleaned Title']='Core engineering'
df_temp.loc[df_temp['Job Title'].str.contains('Tech',case=False),'Cleaned Title']='Core engineering'


# In[ ]:


df_temp.loc[df_temp['Job Title'].str.contains('Prof',case=False),'Cleaned Title']='Academic role'
df_temp.loc[df_temp['Job Title'].str.contains('Business',case=False),'Cleaned Title']='Business Developer/Intelligence'
df_temp.loc[df_temp['Job Title'].str.contains('Social Media',case=False),'Cleaned Title']='Public Relations'
df_temp.loc[df_temp['Job Title'].str.contains('HR',case=False),'Cleaned Title']='Human Resources'
df_temp.loc[df_temp['Job Title'].str.contains('HR Executive',case=False),'Cleaned Title']='Human Resources'
df_temp.loc[df_temp['Job Title'].str.contains('Manager',case=False),'Cleaned Title']='Managerial role'
df_temp.loc[df_temp['Job Title'].str.contains('Fresher',case=False),'Cleaned Title']='Fresher role'
df_temp.loc[df_temp['Job Title'].str.contains('Account',case=False),'Cleaned Title']='Accounting role'
df_temp.loc[df_temp['Job Title'].str.contains('Intern',case=False),'Cleaned Title']='Internships'
df_temp.loc[df_temp['Job Title'].str.contains('Placement',case=False),'Cleaned Title']='Placement & Liaison'
df_temp.loc[df_temp['Job Title'].str.contains('Liaison',case=False),'Cleaned Title']='Placement & Liaison'
df_temp.loc[df_temp['Job Title'].str.contains('Recruit',case=False),'Cleaned Title']='Placement & Liaison'
df_temp.loc[df_temp['Job Title'].str.contains('Data',case=False),'Cleaned Title']='Data Science'
df_temp.loc[df_temp['Job Title'].str.contains('Sale',case=False),'Cleaned Title']='Sales Executive'
df_temp.loc[df_temp['Job Title'].str.contains('Health',case=False),'Cleaned Title']='Health Care'
df_temp.loc[df_temp['Job Title'].str.contains('Quality',case=False),'Cleaned Title']='Quality Control'
df_temp.loc[df_temp['Job Title'].str.contains('Tele',case=False),'Cleaned Title']='Telemarketing'


# In[ ]:


df_temp['Cleaned Title'].value_counts()


# In[ ]:


df_temp['Cleaned Title'].isna().value_counts()


# As it can be seen, we were successful to clean about 20,000 entries by using keywords of each entry and entering them into various known roles. The remaining entries couldn't be captured into any of the known roles. Hence, we will simply drop these entries.

# In[ ]:


df_temp.dropna(inplace=True)


# Let us drop the unclean Job Title column and replace it by the new cleaned title column

# In[ ]:


df_temp.drop('Job Title',axis=1,inplace=True)


# In[ ]:


df_temp=df_temp[['Uniq Id', 'Crawl Timestamp', 'Cleaned Title','Job Salary', 'Job Experience Required',
       'Key Skills', 'Role Category', 'Location', 'Functional Area',
       'Industry', 'Role']]


# In[ ]:


df_temp.rename(columns={'Cleaned Title':'Job Title'},inplace=True)


# In[ ]:


df_temp.head()


# ## Job Experience

# Let us check how clean is the job salary section.

# In[ ]:


df_temp.reset_index(inplace=True,drop=True)
df=df_temp.copy() #Checkpoint


# In[ ]:


df_temp['Job Experience Required'].value_counts()[0:10]


# Since all the experiences are required in years, hence we can remove the years from the column.

# In[ ]:


for i in range(len(df_temp)):
    df_temp['Job Experience Required'][i]=df_temp['Job Experience Required'][i].replace('yrs','')
    df_temp['Job Experience Required'][i]=df_temp['Job Experience Required'][i].replace('years','')
    df_temp['Job Experience Required'][i]=df_temp['Job Experience Required'][i].replace('Years','')

    i+=1


# In[ ]:


df_temp['Job Experience Required'].value_counts()[0:10]


# The data is not too clean again. Since almost each entry is unique, it will be difficult to clean this data manually. Instead, we will visualise the top most required job experiences to get a general idea of what is required by the industry.

# In[ ]:


top_job_exp=df_temp['Job Experience Required'].value_counts()[0:10]
top_job_exp


# To make the data more intuitive in nature, we shall label the job experiences as follows:
# 
# * 0-1 : Freshers
# * 1-5 : Early professionals
# * 5-10 : Expereinced professionals
# 

# In[ ]:


exp_df=pd.DataFrame(top_job_exp)
exp_df.reset_index(inplace=True)


# In[ ]:


exp_df.rename(columns={'index':'Job Experience','Job Experience Required':'Count'},inplace=True)


# In[ ]:


exp_df


# In[ ]:


exp_df.loc[exp_df['Job Experience'].str.contains('2 - 5',case=False),'Sorted Experience']='Early Professionals'
exp_df.loc[exp_df['Job Experience'].str.contains('5 - 10',case=False),'Sorted Experience']='Expereinced Professionals'
exp_df.loc[exp_df['Job Experience'].str.contains('2 - 7',case=False),'Sorted Experience']='Early Professionals'
exp_df.loc[exp_df['Job Experience'].str.contains('3 - 8',case=False),'Sorted Experience']='Expereinced Professionals'
exp_df.loc[exp_df['Job Experience'].str.contains('1 - 3',case=False),'Sorted Experience']='Early Professionals'

exp_df.loc[exp_df['Job Experience'].str.contains('3 - 5',case=False),'Sorted Experience']='Early Professionals'

exp_df.loc[exp_df['Job Experience'].str.contains('1 - 6',case=False),'Sorted Experience']='Early Professionals'

exp_df.loc[exp_df['Job Experience'].str.contains('1 - 5',case=False),'Sorted Experience']='Early Professionals'

exp_df.loc[exp_df['Job Experience'].str.contains('0 - 1',case=False),'Sorted Experience']='Freshers'
exp_df.loc[exp_df['Job Experience'].str.contains('2 - 4',case=False),'Sorted Experience']='Early Professionals'


# In[ ]:


exp_cat=exp_df.copy()
exp_cat.drop('Job Experience',axis=1,inplace=True)
exp_cat.rename(columns={'Sorted Experience':'Experience category'},inplace=True)
exp_cat=exp_cat[['Experience category','Count']]
exp_cat


# In[ ]:


grouped_df=exp_cat.groupby('Experience category').sum()


# In[ ]:


grouped_df.reset_index(inplace=True)
grouped_df


# ## Location
# 
# 
# Let us check the cleanliness of the location data 

# In[ ]:


locs_df=pd.DataFrame(df_temp['Location'])
locs_df.head()


# As we can see, this columns is quite clean and doesn't require any external data wrangling. Let us group these to get an idea of the number of jobs in a particular location.

# In[ ]:


locs_df['Count']=1
group_locs=locs_df.groupby('Location').sum().reset_index()


# In[ ]:


group_locs.sort_values(by='Count',ascending=False,inplace=True)


# In[ ]:


group_locs_top=group_locs.head(12)
group_locs_top


# With this, we are done with the data cleaning and wrangling portion. We can no move forward with visualising the data to get good insights into the data.

# # 3. Data Visualisation

# Let us check how the various job titles are distributed in our data using a seaborn barplot.

# In[ ]:


df_titles=pd.DataFrame(df_temp['Job Title'],columns=['Job Title','Count'])
                    


# In[ ]:


df_titles['Count']=1


# In[ ]:


df_titles=df_titles.groupby('Job Title').sum()
df_titles


# In[ ]:


df_titles.reset_index(inplace=True)
df_titles.sort_values('Count',ascending=False,inplace=True)
df_titles


# ## Job title

# In[ ]:


sns.catplot('Job Title','Count',data=df_titles,kind='bar',aspect=2,height=6,palette='summer')
plt.xticks(rotation=90)
plt.xlabel('Job Title',size=15)
plt.ylabel('Number of jobs available',size=15)
plt.title('Distribution of job titles',size=25)


# As we can see from the barplot above, the roles such as software/web/app developers are the maximum followed closely by Core engineering fields and then by Managerial jobs.

# ## Job experience

# Let us check what kind of job experiences are most commonly advertised. For better intuition, we divided the various ranges as follows:
# 
# * 0-1 : Freshers
# * 1-5 : Early professionals
# * 5-10 : Expereinced professionals
# 
# Let us take a quick look at the dataframe first.

# In[ ]:


grouped_df['Count']=grouped_df['Count'].astype(int)
grouped_df


# In[ ]:


plt.figure(figsize=(10,8))
ax=sns.barplot('Experience category','Count',data=grouped_df)
plt.xlabel('Category',size=15)
plt.ylabel('Number of vacancies',size=15)
plt.title('Expereince wise vacancies',size=20)

for i in ax.patches:
    ax.text(i.get_x()+.25,i.get_height()+2.3,str(int((i.get_height()))),
            rotation=0,fontsize=15,color='black')


# As we can see, most job posting required professionals having about 1-5 years of experience. Fresher jobs with 0-1 year of experience was much lower than the rest of the postings. 

# ## Job Locations

# Let us check the top most cities where the job offers are high.

# In[ ]:


group_locs_top


# In[ ]:


plt.figure(figsize=(10,8))
ax=sns.barplot('Location','Count',data=group_locs_top,palette='winter')
plt.xlabel('Location',size=15)
plt.ylabel('Number of vacancies',size=15)
plt.title('Expereince wise vacancies',size=20)
plt.xticks(rotation=45)

for i in ax.patches:
    ax.text(i.get_x(),i.get_height()+2.3,str(int((i.get_height()))),
            rotation=0,fontsize=15,color='black')


# As we can see, job postings in Bengaluru was found to be much higher than the other cities. Most of the jobs are located in the metro cities.

# ## Role category

# Let us check the buzz words in the role category section through a world cloud. Words that appear in larger size means these are roles with higher vacancies.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud is installed and imported!')


# In[ ]:


imp_words = df_temp['Role Category'].to_list()

wordcloud = WordCloud(width = 500, height = 500, 
                background_color ='white', 
                min_font_size = 10).generate(str(imp_words))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# As we can see, some of the role categories easily separate itself from the rest of the category such as Prgramming, Design, Retail Sales, HR, etc.

# ## Top skills

# For top skills aswell, we will prefer to create a wordcloud to understand which are the various skills that are required for jobs.

# In[ ]:


imp_words = df_temp['Key Skills'].to_list()

wordcloud = WordCloud(width = 500, height = 500, 
                background_color ='green', 
                min_font_size = 10).generate(str(imp_words))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# As it is quite clearly visible, some of the top skills are Business Development, customer servive, web technologies, netoworking.

# In[ ]:




