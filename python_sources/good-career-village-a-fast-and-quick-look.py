#!/usr/bin/env python
# coding: utf-8

# # Data Science for Good: **CareerVillage**.org
# 

# In[ ]:


NUM_ROWS = None # enter None for full data run / or 5000 for test rows


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


# In[ ]:


df_answer=pd.read_csv('../input/answers.csv', nrows= NUM_ROWS) 
df_questions=pd.read_csv('../input/questions.csv', nrows= NUM_ROWS)
df_comments=pd.read_csv('../input/comments.csv', nrows= NUM_ROWS)
df_emails=pd.read_csv('../input/emails.csv', nrows= NUM_ROWS)
df_memberships=pd.read_csv('../input/group_memberships.csv', nrows= NUM_ROWS)
df_groups=pd.read_csv('../input/groups.csv', nrows= NUM_ROWS)
df_matches=pd.read_csv('../input/matches.csv', nrows= NUM_ROWS)
df_professionals=pd.read_csv('../input/professionals.csv', nrows= NUM_ROWS)
df_school_mem=pd.read_csv('../input/school_memberships.csv', nrows= NUM_ROWS)
df_student=pd.read_csv('../input/students.csv', nrows= NUM_ROWS)
df_tag_questions=pd.read_csv('../input/tag_questions.csv', nrows= NUM_ROWS)
df_tag_users=pd.read_csv('../input/tag_users.csv', nrows= NUM_ROWS)
df_tags=pd.read_csv('../input/tags.csv', nrows= NUM_ROWS)


# In[ ]:


df_answer_len    = len(df_answer)
df_questions_len = len(df_questions)
df_comments_len  = len(df_comments)
df_emails_len    = len(df_emails)
df_memberships_len = len(df_memberships)
df_groups_len      = len(df_groups)
df_matches_len     = len(df_matches)
df_professionals_len = len(df_professionals)
df_school_mem_len    = len(df_school_mem)
df_student_len       = len(df_student)
df_tag_questions_len = len(df_tag_questions)
df_tag_users_len     = len(df_tag_users)
df_tags_len          = len(df_tags)


# In[ ]:


df_answer.head(2) 


# In[ ]:


df_answer.info()


# In[ ]:


def get_range(rows1=0, rows2=0):
    if (rows1>0 and rows2>0):
        return rows1 + rows2
    elif (rows1>0):
        return rows1
    else:
        return NUM_ROWS + NUM_ROWS


# In[ ]:


#separate the date and time to handle better
date=[]
time_added=[]   #51123
range01 = get_range(df_answer_len, rows2=0)
for i in range(range01):
    date.append(df_answer['answers_date_added'][i].split()[0])
    time_added.append(df_answer['answers_date_added'][i].split()[1])


# In[ ]:


df_answer['Date']=date
df_answer['Time']=time_added
df_answer=df_answer.drop(columns='answers_date_added')

df_answer.head(2)


# In[ ]:


# Convertion of datetime 
df_answer['Date'] = df_answer['Date'].apply(lambda x:  dt.datetime.strptime(x,'%Y-%m-%d'))
df_answer['Time'] = df_answer['Time'].apply(lambda x: dt.datetime.strptime(x,'%H:%M:%S'))

df_answer_merge=df_answer
# We create the columns Year,Month,Hour of the answers

Year=[]
Month=[]
Hour=[]
for i in range(range01):
    Year.append(df_answer['Date'][i].year)
    Month.append(df_answer['Date'][i].month)
    Hour.append(df_answer['Time'][i].time)


# In[ ]:


df_answer['Year']=Year
df_answer['Month']=Month
df_answer['Hour']=Hour

# New DataFrame
df_answer=df_answer.drop(columns=['Date','Time'])


# In[ ]:


df_answer['Year'].value_counts()


# In[ ]:


# 1) Answer By Year
sns.set_style(style='darkgrid')
plt.title('Answer by Year')
df_answer['Year'].value_counts().plot(kind='bar')


# In[ ]:


# Top 10 users with more answers
df_answer['answers_author_id'].value_counts().head(10)


# In[ ]:


# How many collaborator we have register ?
df_answer['answers_author_id'].nunique()


# In[ ]:


# Analysis amount answers by user
Answer_By_User=pd.DataFrame(df_answer['answers_author_id'].value_counts())
test=pd.DataFrame(Answer_By_User)
Answer_By_User=test.reset_index(inplace=False)
Answer_By_User.head(2)


# In[ ]:


Answer_By_User=Answer_By_User.rename(columns={'answers_author_id':'Amount_Answer'})
Answer_By_User=Answer_By_User.rename(columns={'index':'professionals_id'})


# In[ ]:


# 2) How many collaborator  have more than 100 answer?
Answer_By_User[Answer_By_User['Amount_Answer']>100]
Answer_By_User.head(2)


# In[ ]:


Answer_ByUser=Answer_By_User[Answer_By_User['Amount_Answer']>100]
sns.barplot(data=Answer_ByUser,x='Amount_Answer',y='professionals_id')
plt.title('How many collaborator have more than 100 answer?')


# In[ ]:


# Read  "`Question.csv`"
df_questions.info()


# In[ ]:


dateQ=[]
time_addedQ=[]
range02 = df_questions_len
for i in range(range02):
    dateQ.append(df_questions['questions_date_added'][i].split()[0])
    time_addedQ.append(df_questions['questions_date_added'][i].split()[1])


# In[ ]:


df_questions['Date Question']=dateQ
df_questions['Time Question']=time_addedQ

df_questions.head(2)


# In[ ]:


df_questions['Time Question'] = df_questions['Time Question'].apply(lambda x: dt.datetime.strptime(x,'%H:%M:%S'))
df_questions['Date Question'] = df_questions['Date Question'].apply(lambda x:  dt.datetime.strptime(x,'%Y-%m-%d'))

df_questions=df_questions.drop(columns='questions_date_added')


# In[ ]:


df_questions.tail(1)


# In[ ]:


tiempo=[]
for i in range(range02):
    tiempo.append(df_questions['Time Question'][i].time())


# In[ ]:


df_questions['Time of day']=tiempo # We create a new column
df_questions.head(1)


# In[ ]:


df_questions_merge=df_questions

# 3) Evaluate  "Answers" and "Questions"
df_answer_merge.tail(1) # 51123 logs 


# In[ ]:


tiempoAnswer=[]
for i in range(range01):
    tiempoAnswer.append(df_answer_merge['Time'][i].time())


# In[ ]:


df_answer_merge['Time of day Answer']=tiempoAnswer

# edit column name to make merge 
df_answer_merge['questions_id']=df_answer['answers_question_id']


# In[ ]:


df_answer_merge=df_answer_merge.drop(columns=['Time','Year','Month','Hour'])


# In[ ]:


df_answer_merge.head(2)


# In[ ]:


# restructure the DataFrame at our convenience
df_questions_merge=df_questions_merge[['questions_id','questions_title','questions_body','Date Question','Time of day']]
df_questions_merge.head(2)


# In[ ]:


# We merge answer and question 
New_answ_quest=pd.merge(df_answer_merge,df_questions_merge,how='inner',on='questions_id')
New_answ_quest.head(2)


# In[ ]:


respon_time=New_answ_quest[['answers_author_id','Date','Date Question','Time of day','Time of day Answer']]


# In[ ]:


respon_time['time_by_colaborator']=respon_time['Date']-respon_time['Date Question']


# In[ ]:


# 4) Time response of our collaborators
# We restructure the DataFrame at our convenience
respon_time=respon_time[['answers_author_id','Date Question','Date','time_by_colaborator','Time of day','Time of day Answer']]
respon_time.head(2)


# In[ ]:


# convert to integer "time_by_colaborator"
c=[]
for i in range(51123):
    c.append(int(str(respon_time['time_by_colaborator'][i]).split()[0]))


# In[ ]:


respon_time['TimeAvg_By_colab_Day']=c
respon_time=respon_time[['answers_author_id','Date Question','Date','TimeAvg_By_colab_Day','Time of day','Time of day Answer']]
respon_time.head(2)


# In[ ]:


# Users who answer our users quickly  -- In this first instance we will evaluate the users who answer in less than 24 hours to our users, for this we extract the users that have their response time equal to `0 days` , 
#the column "Speed time" refers to the response time in hours of users within a period of 1 day

df=respon_time
df=df[df['TimeAvg_By_colab_Day']==0] # Respuestas inmediatas
df=df.drop(columns=['Date Question','Date','TimeAvg_By_colab_Day'])

test=df
df=test.reset_index(inplace=False) # Reconfiguramos los ejes 


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


# In[ ]:


# 5) Users who respond in less than 24 hours  "*Minutes*"
Speed_time=Speed_time[Speed_time['Speed_Time_Minute']>0]


# In[ ]:


Speed_time[['answers_author_id','Speed_Time_Minute']].sort_values(by='Speed_Time_Minute',ascending=True)


# In[ ]:


# calculate the average of users who respond in less than 24 hours
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


# In[ ]:


# 6) Top 10 users with response time less than 1 hour
pd.merge(Answer_By_User,prom_less_60,how='inner',on='answers_author_id').head(10)


# In[ ]:


# 7) Top 15 users with response time less than 1 day
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


# In[ ]:


# 8) How quickly do our volunteers respond?
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


# In[ ]:


# 9) Memberships
#   ---- Amount groups 


# In[ ]:


df_memberships['group_memberships_group_id'].nunique()


# In[ ]:


# Top 10 Amount of user by group
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


# In[ ]:


# 10 ) Merge Groups and membership
New_Group_member=pd.merge(Group_mem, df_groups, how='inner',on='groups_id')
New_Group_member.head(5)


# In[ ]:


# Analyze General groups
New_Group_member=New_Group_member.groupby('groups_group_type').sum()
test=New_Group_member
New_Group_member=test.reset_index(inplace=False)


# In[ ]:


sns.set_style(style='darkgrid')
plt.title('Amount members by group_type')
data=New_Group_member.sort_values(by='Amount_members',ascending=False)
sns.barplot(x='Amount_members',y='groups_group_type',palette='viridis',data=data)


# In[ ]:


df_matches.nunique()


# In[ ]:


df_matches.head(2)


# In[ ]:


df_matches[df_matches['matches_email_id']==2337714]


# In[ ]:


# 11) analyze Professionals
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


# In[ ]:


# 12) Students
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


# In[ ]:


# Cleaning student location 
b=[]
for i in range(28938):    
    b.append(df_student['students_location'][i].split(',')[-1])


# In[ ]:


df_student['students_location']=b


# In[ ]:


Student_by_loc=pd.DataFrame(df_student['students_location'].value_counts())


# In[ ]:


# Top 15 origin place of our students  
top_Stud=Student_by_loc['students_location'].head(15)
top_Stud=pd.DataFrame(top_Stud)
plt.title('Top 15 origin place of our students')
sns.barplot(x='students_location',y=top_Stud.index,data=top_Stud)


# In[ ]:


# 13) Tag questions analysis
df_tag_questions.head(2)


# In[ ]:


df_tag_users=df_tag_users.rename(columns={'tag_users_tag_id':'tag_id'})


# In[ ]:


df_tags=df_tags.rename(columns={'tags_tag_id':'tag_id'})


# In[ ]:


# Top 15 - What are interest of our students? 
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


# In[ ]:


# 14) Text analysis --- `"Question Title"`
respuestas=df_answer
preguntas=df_questions


# In[ ]:


respuestas=respuestas[['answers_author_id','answers_question_id','answers_body']]
preguntas=preguntas[['questions_id','questions_title','questions_body']]


# In[ ]:


#**Tokenize:** We create the data that we are going to tokenize **`"Questions Title"`**
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


# Eliminate words that are obvious from a question like:
# 
# * What  * I  * How  * ,  * Is  * .  * 's  * If

# In[ ]:


df_pal=df_pal[(df_pal['Word'] != 'What') & (df_pal['Word'] != 'I') & (df_pal['Word'] != 'How') & (df_pal['Word'] != ',') & (df_pal['Word'] != 'Is') & (df_pal['Word'] != '.') & (df_pal['Word'] != 'If') & (df_pal['Word'] != "'s") & (df_pal['Word'] != '?')]


# In[ ]:


# Top 20 common words - Question Title
top_20_Qword=df_pal.sort_values(by='Frequency_words',ascending=False).head(20)
top_20_Qword


# In[ ]:


plt.title('Common words of our students')
sns.barplot(data=top_20_Qword,x='Frequency_words',y='Word')


# In[ ]:


# 15) Top 20 collaborators more actives- Global information
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
# SUMMARY - The most active collaborators are those who joined our community in years after 2015. 
# 
# Thank you
