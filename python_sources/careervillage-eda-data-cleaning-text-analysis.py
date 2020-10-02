#!/usr/bin/env python
# coding: utf-8

# # Data Science for Good: CareerVillage.org
#     Match career advice questions with professionals in the field
# 
# In the following analysis, what I intend is to make a complete exploration of the data to provide the best possible recommendation to "CareerVillage.org" using techniques such as data cleaning, merge between dataframes, visualization, text analysis, time series between others for this reason, I show you the order in which each of the data provided was addressed:
# 
# **Outline**
# 
# 1. Answer by year
#     * Top 10 users with more answer
#     * How many collaborators we have register ?
# 2. How many collaborators have more than 100 answer
# 3. Evaluation "Answers" and "Questions"
# 4. Time response of our collaborators
#     * Time series
# 5. Collaborators who respond in less than 24 hours
# 6. Top 10 users with response time less than 1 hour 
# 7. Top 10 users with response time less than 1 day
# 8. How quickly do our volunteers respond ?
# 9. Memberships 
# 10. Merge groups and membership
# 11. porfessionals
# 12. Students 
#     * Cleaning Student location
#     * Top 15 origin place of our students 
# 13. Tag questions
#     * Top 15- What are interest of our students ?
# 14. Text Anlysis -- "Question title"
#     * Top 20 -- Common words - Question Title
# 15. Top 20 collaborators more actives -- Global Information 
# 
# 16. **SUMMARY**
# 17. **Criteria for Measuring Solutions**
# 

# # Import Libraries 

# In[ ]:


import pandas as pd
import datetime as dt
import seaborn as sns 
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Files 

# In[ ]:


df_answer=pd.read_csv('../input/answers.csv') 
df_questions=pd.read_csv('../input/questions.csv')
df_comments=pd.read_csv('../input/comments.csv')
df_emails=pd.read_csv('../input/emails.csv')
df_memberships=pd.read_csv('../input/group_memberships.csv')
df_groups=pd.read_csv('../input/groups.csv')
df_matches=pd.read_csv('../input/matches.csv')
df_professionals=pd.read_csv('../input/professionals.csv')
df_school_mem=pd.read_csv('../input/school_memberships.csv')
df_student=pd.read_csv('../input/students.csv')
df_tag_questions=pd.read_csv('../input/tag_questions.csv')
df_tag_users=pd.read_csv('../input/tag_users.csv')
df_tags=pd.read_csv('../input/tags.csv')


# In[ ]:


df_answer.head(2) 


# In[ ]:


df_answer.info()


# In[ ]:


# We separate the date and time to have a better handling of our data


# In[ ]:


date=[]
time_added=[]
for i in range(51123):
    date.append(df_answer['answers_date_added'][i].split()[0])
    time_added.append(df_answer['answers_date_added'][i].split()[1])


# In[ ]:


# We  add the new columns to our dataframe  


# In[ ]:


df_answer['Date']=date
df_answer['Time']=time_added
df_answer=df_answer.drop(columns='answers_date_added')


# In[ ]:


df_answer.head(2)


# In[ ]:


# Convertion of datetime 


# In[ ]:


df_answer['Date'] = df_answer['Date'].apply(lambda x:  dt.datetime.strptime(x,'%Y-%m-%d'))
df_answer['Time'] = df_answer['Time'].apply(lambda x: dt.datetime.strptime(x,'%H:%M:%S'))


# In[ ]:


df_answer_merge=df_answer


# In[ ]:


# We create the columns Year,Month,Hour of the answers


# In[ ]:


Year=[]
Month=[]
Hour=[]
for i in range(51123):
    Year.append(df_answer['Date'][i].year)
    Month.append(df_answer['Date'][i].month)
    Hour.append(df_answer['Time'][i].time)


# In[ ]:


df_answer['Year']=Year
df_answer['Month']=Month
df_answer['Hour']=Hour


# In[ ]:


# New DataFrame
df_answer=df_answer.drop(columns=['Date','Time'])


# In[ ]:


df_answer['Year'].value_counts()


# # 1) Answer By Year

# In[ ]:


sns.set_style(style='darkgrid')
plt.title('Answer by Year')
df_answer['Year'].value_counts().plot(kind='bar')


# **ANALYSIS:** As we can see from the year 2017 there is a significant increase in the number of responses that our collaborators have had to offer, which invites us to look for mechanisms to give a quick response to our students

# # Top 10 users with more answers

# In[ ]:


df_answer['answers_author_id'].value_counts().head(10)


# # How many collaborator we have register ?

# In[ ]:


df_answer['answers_author_id'].nunique()


# # Analysis amount answers by user

# In[ ]:


Answer_By_User=pd.DataFrame(df_answer['answers_author_id'].value_counts())
test=pd.DataFrame(Answer_By_User)
Answer_By_User=test.reset_index(inplace=False)


# In[ ]:


Answer_By_User=Answer_By_User.rename(columns={'answers_author_id':'Amount_Answer'})
Answer_By_User=Answer_By_User.rename(columns={'index':'professionals_id'})


# # 2) How many collaborator  have more than 100 answer?
# 
# Of the 10169 records of collaborators we have `23` that are above the 100 answers, indicating that they are our collaborator more active .

# In[ ]:


Answer_By_User[Answer_By_User['Amount_Answer']>100]


# # Visualization 

# In[ ]:


Answer_ByUser=Answer_By_User[Answer_By_User['Amount_Answer']>100]
sns.barplot(data=Answer_ByUser,x='Amount_Answer',y='professionals_id')
plt.title('How many collaborator have more than 100 answer?')


# # Read  "`Question.csv`"

# In[ ]:


df_questions.info()


# In[ ]:


dateQ=[]
time_addedQ=[]
for i in range(23931):
    dateQ.append(df_questions['questions_date_added'][i].split()[0])
    time_addedQ.append(df_questions['questions_date_added'][i].split()[1])


# In[ ]:


df_questions['Date Question']=dateQ
df_questions['Time Question']=time_addedQ


# In[ ]:


df_questions.head(2)


# In[ ]:


# Convertion of datetime 


# In[ ]:


df_questions['Time Question'] = df_questions['Time Question'].apply(lambda x: dt.datetime.strptime(x,'%H:%M:%S'))
df_questions['Date Question'] = df_questions['Date Question'].apply(lambda x:  dt.datetime.strptime(x,'%Y-%m-%d'))


# In[ ]:


df_questions=df_questions.drop(columns='questions_date_added')


# In[ ]:


df_questions.tail(1)


# In[ ]:


tiempo=[]
for i in range(23931):
    tiempo.append(df_questions['Time Question'][i].time())


# In[ ]:


df_questions['Time of day']=tiempo # We create a new column


# In[ ]:


df_questions.head(1)


# In[ ]:


df_questions_merge=df_questions


# # 3) Evaluation  "Answers" and "Questions"

# In[ ]:


df_answer_merge.tail(1) # 51123 logs 


# In[ ]:


tiempoAnswer=[]
for i in range(51123):
    tiempoAnswer.append(df_answer_merge['Time'][i].time())


# In[ ]:


df_answer_merge['Time of day Answer']=tiempoAnswer


# In[ ]:


# We edit column name to make merge 


# In[ ]:


df_answer_merge['questions_id']=df_answer['answers_question_id']


# In[ ]:


df_answer_merge=df_answer_merge.drop(columns=['Time','Year','Month','Hour'])


# In[ ]:


df_answer_merge.head(2)


# In[ ]:


# We restructure the DataFrame at our convenience


# In[ ]:


df_questions_merge=df_questions_merge[['questions_id','questions_title','questions_body','Date Question','Time of day']]


# In[ ]:


df_questions_merge.head()


# # We merge answer and question 

# In[ ]:


New_answ_quest=pd.merge(df_answer_merge,df_questions_merge,how='inner',on='questions_id')


# In[ ]:


New_answ_quest.head()


# In[ ]:


respon_time=New_answ_quest[['answers_author_id','Date','Date Question','Time of day','Time of day Answer']]


# In[ ]:


respon_time['time_by_colaborator']=respon_time['Date']-respon_time['Date Question']


# # 4) Time response of our collaborators

# In[ ]:


# We restructure the DataFrame at our convenience


# In[ ]:


respon_time=respon_time[['answers_author_id','Date Question','Date','time_by_colaborator','Time of day','Time of day Answer']]


# In[ ]:


respon_time.head()


# In[ ]:


# We convert to integer "time_by_colaborator"


# In[ ]:


c=[]
for i in range(51123):
    c.append(int(str(respon_time['time_by_colaborator'][i]).split()[0]))


# In[ ]:


respon_time['TimeAvg_By_colab_Day']=c


# In[ ]:


respon_time=respon_time[['answers_author_id','Date Question','Date','TimeAvg_By_colab_Day','Time of day','Time of day Answer']]


# In[ ]:


respon_time.head()


# # Users who answer our users quickly
# 
# In this first instance we will evaluate the users who answer in less than 24 hours to our users, for this we extract the users that have their response time equal to `0 days` , 
# the column "Speed time" refers to the response time in hours of users within a period of 1 day

# In[ ]:


df=respon_time
df=df[df['TimeAvg_By_colab_Day']==0] # Respuestas inmediatas
df=df.drop(columns=['Date Question','Date','TimeAvg_By_colab_Day'])

test=df
df=test.reset_index(inplace=False) # Reconfiguramos los ejes 


# In[ ]:


# Creacion nueva columna tiempo de respuesta en minutos 


# In[ ]:


time_res=[]
time_ques=[]
for i in range(8429):
    time_res.append(df['Time of day Answer'][i].hour*60+df['Time of day Answer'][i].minute+df['Time of day Answer'][i].second/60)
    time_ques.append(df['Time of day'][i].hour*60+df['Time of day'][i].minute+df['Time of day'][i].second/60)


# In[ ]:


c=[]
for i in range(8429):
    c.append(time_res[i]-time_ques[i])


# In[ ]:


df['Speed_Time_Minute']=c


# In[ ]:


Speed_time=df


# In[ ]:


Speed_time.head(1)


# In[ ]:





# In[ ]:


Speed_time=Speed_time.drop(columns=['Time of day','Time of day Answer'])


# In[ ]:


Speed_time.head()


# In[ ]:


test=Speed_time
Speed_time=test.reset_index(inplace=False)


# In[ ]:


Speed_time=Speed_time.drop(columns='index')


# In[ ]:


Speed_time.head()


# # 5) Users who respond in less than 24 hours  "*Minutes*"
# 
# We have a total of 8429 volunteers who respond in less than 24 hours

# In[ ]:


Speed_time=Speed_time[Speed_time['Speed_Time_Minute']>0]


# In[ ]:


Speed_time[['answers_author_id','Speed_Time_Minute']].sort_values(by='Speed_Time_Minute',ascending=True)


# In[ ]:


# We calculate the average of users who respond in less than 24 hours


# In[ ]:


test=Speed_time[['answers_author_id','Speed_Time_Minute']].groupby(by='answers_author_id').mean().sort_values(by='Speed_Time_Minute',ascending=True)
avg_speed_time=test.reset_index(inplace=False)


# In[ ]:


# Collaborators who respond in less than 1 hour


# In[ ]:


prom_less_60=avg_speed_time[avg_speed_time['Speed_Time_Minute']<=60]


# In[ ]:


# Collaborators who respond after 1 hour


# In[ ]:


prom_greater_60=avg_speed_time[avg_speed_time['Speed_Time_Minute']>60]


# In[ ]:


Answer_By_User=Answer_By_User.rename(columns={'professionals_id':'answers_author_id'})


# # 6) Top 10 users with response time less than 1 hour 

# In[ ]:


pd.merge(Answer_By_User,prom_less_60,how='inner',on='answers_author_id').head(10)


# # 7) Top 15 users with response time less than 1 day
# 
# We can see in the following table, users who have more than 100 answers respond on average in less than a week

# In[ ]:


pd.merge(Answer_By_User,prom_greater_60,how='inner',on='answers_author_id').head(15)


# In[ ]:


# Average response less than a week


# In[ ]:


respon_time=respon_time[respon_time['TimeAvg_By_colab_Day']>0]


# In[ ]:


respon_time=respon_time[['answers_author_id','TimeAvg_By_colab_Day']]


# In[ ]:


respon_time[(respon_time['TimeAvg_By_colab_Day']>0)&(respon_time['TimeAvg_By_colab_Day']<=7)].head()


# In[ ]:


# Number of volunteers who respond normally in less than a week


# In[ ]:


respon_time[(respon_time['TimeAvg_By_colab_Day']>0)&(respon_time['TimeAvg_By_colab_Day']<=7)].count()


# In[ ]:


# Volunteers who respond in less than 1 month


# In[ ]:


respon_time[(respon_time['TimeAvg_By_colab_Day']>7)&(respon_time['TimeAvg_By_colab_Day']<=30)].count()


# In[ ]:


#  Volunteers who respond in less than 6 months


# In[ ]:


respon_time[(respon_time['TimeAvg_By_colab_Day']>30)&(respon_time['TimeAvg_By_colab_Day']<=180)].count()


# In[ ]:


#  Volunteers who respond after 6 months


# In[ ]:


respon_time[respon_time['TimeAvg_By_colab_Day']>180].count()


# # 8) How quickly do our volunteers respond?
# 
# * We create a dictionary with the figures previously found
# * We created a DataFrame to observe in a better way the found findings
# * We create our graph

# In[ ]:


Num_dict={'tiempo_respues':[' < 1 week',' < 1 month' , ' < 6 months ', ' > 6 months'],'Num_cola' : [21226,6056,11523,12302] }


# In[ ]:


# Number of volunteers by response times


# In[ ]:


Num_by_time_rest=pd.DataFrame(Num_dict)


# In[ ]:


Num_by_time_rest


# In[ ]:


plt.figure(figsize=(7,8))
plt.pie(Num_by_time_rest.Num_cola,autopct='%1.1f%%',labels=Num_by_time_rest.tiempo_respues,shadow=True,radius=0.75)
plt.legend(loc=0)


# **NOTE:** We can clearly observe that our volunteers respond in less than a week, although we must pay special attention to the volunteers who respond to our users in a time exceeding one month

# In[ ]:


df_comments.head(1)


# In[ ]:


df_emails.nunique()


# In[ ]:


df_emails['emails_frequency_level'].value_counts()


# In[ ]:


df_emails.head()


# In[ ]:


df_emails[['emails_id','emails_recipient_id','emails_frequency_level']].groupby(by=['emails_recipient_id','emails_frequency_level'],group_keys=True,sort=True).count().head(10)


# In[ ]:


df_emails.head()


# # 9) Memberships

# In[ ]:


# Amount groups 


# In[ ]:


df_memberships['group_memberships_group_id'].nunique()


# # Top 10 Amount of user by group

# In[ ]:


Group_mem=pd.DataFrame(df_memberships['group_memberships_group_id'].value_counts())
test=pd.DataFrame(Group_mem)
Group_mem=test.reset_index(inplace=False)
Group_mem=Group_mem.rename(columns={'group_memberships_group_id':'Amount_members'})
Group_mem['groups_id']=Group_mem['index']
Group_mem=Group_mem[['groups_id','Amount_members']]
Group_mem.head(10)


# In[ ]:


df_groups.head()


# In[ ]:


df_groups['groups_group_type'].value_counts()


# In[ ]:


plt.title('Group type')
df_groups['groups_group_type'].value_counts().plot(kind='bar')


# # 10 ) Merge Groups and membership

# In[ ]:


New_Group_member=pd.merge(Group_mem, df_groups, how='inner',on='groups_id')
New_Group_member.head(5)


# # Analysis General groups

# In[ ]:


New_Group_member=New_Group_member.groupby('groups_group_type').sum()
test=New_Group_member
New_Group_member=test.reset_index(inplace=False)


# In[ ]:


sns.set_style(style='darkgrid')
plt.title('Amount members by group_type')
data=New_Group_member.sort_values(by='Amount_members',ascending=False)
sns.barplot(x='Amount_members',y='groups_group_type',palette='viridis',data=data)


# **NOTE:** Individually the groups of `"Cause"` have more users but if we analyze together the youth groups are the ones that in general have the largest community

# In[ ]:


df_matches.nunique()


# In[ ]:


df_matches.head(2)


# In[ ]:


df_matches[df_matches['matches_email_id']==2337714]


# # 11) Professionals

# In[ ]:


df_professionals.info()


# In[ ]:


# Missing Data


# In[ ]:


sns.heatmap(df_professionals.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


df_professionals[df_professionals['professionals_id']=='36ff3b3666df400f956f8335cf53e09e']


# In[ ]:


Answer_By_User.head()


# In[ ]:


Answer_By_User['professionals_id']=Answer_By_User['answers_author_id']
Answer_By_User=Answer_By_User[['professionals_id','Amount_Answer']]


# In[ ]:


#  Merge analysis between Answer_By_User and df_professionals


# In[ ]:


New_volunteers=pd.merge(Answer_By_User,df_professionals,how='inner',on='professionals_id')


# In[ ]:


# Number of volunteers who register more than 80 answers in our database

New_volunteers=New_volunteers[New_volunteers['Amount_Answer']>80] # 38 people
New_volunteers.head()


# In[ ]:


a=[]
for i in range(38):
    a.append(New_volunteers['professionals_date_joined'][i].split()[0].split('-')[0])

    # Create new columns 
New_volunteers['Year']=a


# In[ ]:


test=New_volunteers[['professionals_id','Amount_Answer','Year']].groupby(by='Year').sum().sort_values(by='Amount_Answer',ascending=False)
Volu_By_Year=test.reset_index(inplace=False)
Volu_By_Year


# **NOTE:** 
# 
# In this section we can see that our volunteers with the most answers belong mostly to the **`technology industry`** but we must focus our attention on the volunteers who registered in the years 2016-2015 and 2018, who are our volunteers with the greatest tendency to respond to our users. 

# # 12) Students

# In[ ]:


df_school_mem.nunique()


# In[ ]:


df_student.info()


# In[ ]:


# Remove null values
df_student=df_student.dropna()
df_student=df_student.reset_index()


# In[ ]:


df_student=df_student.drop(columns='index')


# In[ ]:


df_student['students_location'][27].split(',')[-1]


# # Cleaning student location 

# In[ ]:


b=[]
for i in range(28938):    
    b.append(df_student['students_location'][i].split(',')[-1])


# In[ ]:


df_student['students_location']=b


# In[ ]:


Student_by_loc=pd.DataFrame(df_student['students_location'].value_counts())


# # Top 15 origin place of our students  

# In[ ]:


top_Stud=Student_by_loc['students_location'].head(15)
top_Stud=pd.DataFrame(top_Stud)
plt.title('Top 15 origin place of our students')
sns.barplot(x='students_location',y=top_Stud.index,data=top_Stud)


# **NOTE:** We have around **`3053`** foreign students who belong mostly to **`India and Canada`**, which is a significant number of students, but we must not forget our most important sectors: California, Texas and the rest of the United States.

# # 13) Tag questions

# In[ ]:


df_tag_questions.head(2)


# In[ ]:


df_tag_users=df_tag_users.rename(columns={'tag_users_tag_id':'tag_id'})


# In[ ]:


df_tags=df_tags.rename(columns={'tags_tag_id':'tag_id'})


# # Top 15 - What are interest of our students? 

# In[ ]:


New_tags_by_file=pd.merge(df_tag_users,df_tags,how='inner',on='tag_id')
New_tags_by_file=New_tags_by_file[['tags_tag_name','tag_id']].groupby(by='tags_tag_name').count()


# In[ ]:


test=New_tags_by_file
New_tags_by_file=test.reset_index(inplace=False)


# In[ ]:


New_tags_by_file.sort_values(by='tag_id',ascending=False).head(15)


# In[ ]:


interest=New_tags_by_file.sort_values(by='tag_id',ascending=False).head(15)
sns.barplot(data=interest,x='tag_id',y='tags_tag_name')
plt.title("Common tags according to the student's question")


# In[ ]:


df_tags_q=df_tag_questions.rename(columns={'tag_questions_tag_id':'tag_id'})


# In[ ]:


merge_users_tag=pd.merge(df_tag_users,df_tags,how='inner',on='tag_id')


# In[ ]:


merge_users_tag.tail()


# # 14) Text analysis --- `"Question Title"`

# In[ ]:


respuestas=df_answer
preguntas=df_questions


# In[ ]:


respuestas=respuestas[['answers_author_id','answers_question_id','answers_body']]
preguntas=preguntas[['questions_id','questions_title','questions_body']]


# **Tokenize:** We create the data that we are going to tokenize **`"Questions Title"`**

# In[ ]:


texto=''
for i in range(23931):
    texto= texto + ' ' + preguntas['questions_title'][i]


# In[ ]:


# Delete repeated words 


# In[ ]:


stopWords = set(stopwords.words('english'))
words = word_tokenize(texto)
wordsFiltered = []
 
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)


# In[ ]:


diccio=nltk.Counter(wordsFiltered)
hola=dict(diccio) # We convert to dictionary


# In[ ]:


valores=hola.values()
filas=hola.keys()
filas=list(filas) # We convert to list 
valores=list(valores) # We convert to list


# In[ ]:


# We create to DataFrame common words
df_pal = pd.DataFrame([[key, hola[key]] for key in hola.keys()], columns=['Word','Frequency_words'])


# # We eliminate words that are obvious from a question like:
# 
# * What
# * I
# * How
# * ,
# * Is
# * .
# * 's
# * If

# In[ ]:


df_pal=df_pal[(df_pal['Word'] != 'What') & (df_pal['Word'] != 'I') & (df_pal['Word'] != 'How') & (df_pal['Word'] != ',') & (df_pal['Word'] != 'Is') & (df_pal['Word'] != '.') & (df_pal['Word'] != 'If') & (df_pal['Word'] != "'s") & (df_pal['Word'] != '?')]


# # Top 20 common words - Question Title

# In[ ]:


top_20_Qword=df_pal.sort_values(by='Frequency_words',ascending=False).head(20)
top_20_Qword


# In[ ]:


plt.title('Common words of our students')
sns.barplot(data=top_20_Qword,x='Frequency_words',y='Word')


# # 15) Top 20 collaborators more actives- Global information

# In[ ]:


col_pro=pd.merge(Answer_ByUser,df_professionals,how='inner',on='professionals_id')


# In[ ]:


d=[]
for i in range (20):
    d.append(col_pro['professionals_date_joined'][i].split()[0].split('-')[0])


# In[ ]:


col_pro['Year_Joined']=d


# In[ ]:


col_pro[['professionals_id','Amount_Answer','professionals_industry','professionals_headline','Year_Joined']]


# 
# # 16) SUMMARY
# 
# After all this tour of the data I will report the following findings:
# 
# 
# 1. Our most active collaborators are those who joined our community in years exceeding 2015, being the years 2015 and 2016 the most representative in terms of volunteers with high vocation of response to our users.
# 
# 
# 2. Of the 10169 records of collaborators we have 23 that are above the 100 answers, indicating that they are our collaborator more active .
# 
# 
# 3. In general terms we observed that our volunteers respond in less than a week, although we must pay special attention to the volunteers who respond to our users in a time exceeding one month because if we add this amount we obtain a figure higher than **`40%`** of responses registered after one month.
# 
# 
# 4. Individually the groups of **`"Cause"`** have more users but if we analyze together the youth groups are the ones that in general have the largest community
# 
# 
# 5. We can see that our volunteers with the most answers belong mostly to the **`technology industry`** but we must focus our attention on the volunteers who registered in the years 2016-2015 and 2018, who are our volunteers with the greatest tendency to respond to our users.
# 
# 
# 6. We have around **`3053`** foreign students who belong mostly to **`India and Canada`**, which is a significant number of students, but we must not forget our most important sectors: California, Texas and the rest of the United States.
# 
# 
# 7. At the time of making the analysis of the text **"Question Title"**, we can see a correct total correspondence between the frequent words in our students' questions and the labels registered in our database, our students are interested in asking about issues related to to ***`"college, telecommunications, computer-software, information-technology-and-services, technology, business, etc."`***
# 
# 
# 
# 

# # 17) Criteria for Measuring Solutions
# 
# 
# * **How well does the solution match professionals to the questions they would be  motivated to answer?**
#     
#     **Rta:** *We could show that our most active volunteers belong to the technology industry so we can infer that as volunteers are or have worked in this industry they will be more motivated to respond, which means that we must organize all our volunteers according to the field that have more experience so that our students get a faster response to their questions*
# 
# 
# 

# **FINAL**
# 
# If this analysis was very useful, I thank you for being reciprocated with one upvote, you are free to comment on this notebook, if you have any questions I will be happy to answer your questions.

# In[ ]:




