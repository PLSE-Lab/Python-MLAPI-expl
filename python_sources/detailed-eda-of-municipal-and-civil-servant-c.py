#!/usr/bin/env python
# coding: utf-8

# # Detailed EDA of municipal and civil servant costs
# ### Analysing civil servant jobs, contracts and salaries from Campo Alegre - AL - Brazil

# ### Introduction
# 
# Campo Alegre is a municipality located in the Brazilian state of Alagoas. The Municipality has a strong influence of religion in cultural manifestations, one of the main ones being the Corpus Christi festival with the making of street rugs for the passage of the Corpus Christi procession, including various materials such as sawdust, flour, sand and flowers for the ornamentation of the main public roads of the city, this party is very traditional in Campo Alegre and also in several cities of strong Catholic influence in Alagoas and Brazil.

# <img src='https://i.imgur.com/Gl2w6rT.jpg' style='width:1750px;height:350px'/>
# 
# **Figure 1**: Campo Alegre city entrance. Source: http://www.campoalegre.al.gov.br/

# ### Objectives
# 
# The cost of the state is certainly one of the most heated debates in politics, a significant part of this cost is related to payroll. In Brazil The 2017 numbers shows that we have a total of 11.5 million civil servants and a cost of R $ 725 billion, in 2017, the Brazilian public service accounts for 10.7 percent of the country's Gross Domestic Product (GDP). According to http://www.ipea.gov.br/atlasestado/ ***"The civil service in the country has expanded in the last three decades and the expansion has concentrated at the municipal level".***
# 
# Here we will analyze these costs related with jobs, contract levels and salaries in one of the many Brazilian municipalities and try to answer the following questions:
# 
#  - **The total payroll cost grown with passing years?**
#  
#  - **The salarys grown with passing years?**
#  
#  - **What the most frequent job position and how much they earn?**
#  
#  - **What job positions have higher salaries?**
#  
#  - **What contract levels have higher salaries?**

# ### About the Dataset
# 
# The used dataset was extracted from transparency portal, the dataset contains payroll information from all public workers of campo alegre city hall located on Brazillian state of Alagoas.
# 
# **Columns information:**
# 
#    - **date:** describe the month and year of public worker payment in MONTH YEAR format
#    
#    - **nome:** public worker name
# 
#    - **matricula:** the registration number
# 
#    - **cargo:** job position in portuguese
# 
#    - **nivel:** contract level in portuguese
# 
#    - **valor_base:** base salary in Reais
# 
#    - **proventos:** earnings in Reais
# 
#    - **descontos:** discounts in Reais
# 
#    - **liquido:** liquid salary in Reais
#    
# You can find the original Source of data here: https://www.municipioonline.com.br/al/prefeitura/campoalegre/cidadao/servidor

# ### Loading Required Librarys and Dataset

# In[ ]:


import pandas as pd # data manipulation
import numpy as np # linear algebra
import re # regular expression
import string # string manipulation
from datetime import datetime # date maniputation
import seaborn as sns # visualization
import matplotlib.pyplot as plt # visualization

from nltk.corpus import stopwords # text mining
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # wordcloud

import warnings # ignore warnings

warnings.filterwarnings("ignore")


# In[ ]:


# loading our dataset
payroll_data = pd.read_csv("../input/city-hall-payroll-from-campo-alegre-al-brazil/payroll_dataset.csv")

# seeing the first 6 rows
payroll_data.head()


# In[ ]:


# verifying dimenssions
payroll_data.shape


# Now that we already know what our data looks like, let's see basics informations about our data looking for missing values and seeing our dataset structure

# In[ ]:


# seeing features information
payroll_data.info()


# In[ ]:


# verifying missing values
payroll_data.isna().sum()


# ### Exploratory Data Analysis

# Let's start taking a look into our numerical variables and the relation between them using a pairplot. All numerical variables except "matricula" seems to have similar distributions, "proventos" and "liquido" and "liquido" and "descontos" seems to have a intersting correlation, let's take a look into this latter.

# In[ ]:


# selecting numerical variables to pairplots
num_vars = ['matricula','valor_base'
            ,'proventos','descontos','liquido']

# pairplot with numerical variables in payroll_data
sns.pairplot(payroll_data[num_vars],aspect=0.3*5);


# Clearly the "matricula" variable has no relation with any numerical variables our salarys, let's drop this variable of latter analysis, let's now take a look into Base salary, Liquid salary and discounts distribuitions and relation separatly with histograms and latter with scatterplots

# In[ ]:


# defyning the plotsize
plt.figure(figsize=(18,10))

# creating the distribution plot
sns.distplot(payroll_data['valor_base']
             , bins=100
             , color='blue')

plt.title("Base salary distribution",size=30)
plt.xlabel("valor_base",size=20) # defyning xlabel
plt.xticks(size=12) # defyning xticks size
plt.yticks(size=12); # defyning yticks size


# In[ ]:


# defyning the plotsize
plt.figure(figsize=(18,10))

# creating the distribution plot
sns.distplot(payroll_data['liquido']
             , bins=100
             , color='green')

plt.title("Liquid salary distribution",size=30)
plt.xlabel("Liquido",size=20) # defyning xlabel
plt.xticks(size=12) # defyning xticks size
plt.yticks(size=12); # defyning yticks size


# In[ ]:


# defyning the plotsize
plt.figure(figsize=(18,10))

# creating the distribution plot
sns.distplot(payroll_data['descontos']
             , bins=100
             , color='red')

plt.title("Discounts distribution",size=30)
plt.xlabel("descontos",size=20) # defyning xlabel
plt.xticks(size=12) # defyning xticks size
plt.yticks(size=12); # defyning yticks size


# "liquido" and "descontos" seems the have a good relation but clearly some high salaries have low discounts wilhe some small salaries have high discounts, probably this occurred because of human error on data collection

# In[ ]:


# defyning the plotsize
plt.figure(figsize=(18,10))

# creating the plot
sns.scatterplot(x='liquido' # defynin x_axis
               , y='descontos' # defyning y_axis
               , data=payroll_data) # defyning the data base

plt.title('Relation Between Liquid salary and discounts',size=30)
plt.xlabel("liquido",size=20) # defyning xlabel
plt.ylabel("descontos",size=20) # defyning ylabel
plt.xticks(size=12) # defyning xticks size
plt.yticks(size=12); # defyning yticks size


# Now let's take a look into relation between "valor_base" and "liquido", as we saw on pairplots this features seems to be not positive correlated, not necessarily when "valor_base" is higher "liquido" is higher

# In[ ]:


# defyning the plotsize
plt.figure(figsize=(18,10))

# creating the plot
sns.scatterplot(x='valor_base' # defynin x_axis
                , y='liquido' # defyning y_axis
                , data=payroll_data) # defyning the data base

plt.title('Relation Between Base salary and Liquid salary',size=30)
plt.xlabel("valor_base",size=20) # defyning xlabel
plt.ylabel("liquido",size=20) # defyning ylabel
plt.xticks(size=12) # defyning xticks size
plt.yticks(size=12); # defyning yticks size


# Now let's take a look into numerical variables distribution with boxplots where we can see a great number of outliers and the majority of salaries concentrated below 5000 Reais

# In[ ]:


# selecting numerical variables
num_vars = ['valor_base','proventos','descontos','liquido']

# defyning plotsize
plt.figure(figsize=(20,10))

# plotting multiple boxplots
sns.boxplot(data=payroll_data[num_vars], palette="bright")

plt.title('Numerical features boxplots distributions and outliers analysis',size=30)
plt.xlabel("Salaries variables",size=20); # defyning xlabel
plt.ylabel("Salaries distribution",size=20); # defyning ylabel
plt.xticks(size=12) # defyning xticks size
plt.yticks(size=12); # defyning yticks size


# Now to confirm correlations we can take a look into Pearson Correlations between numerical variables and see that "valor_base" and "liquido" seems to have nagative correlation also as "liquido" and "descontos"

# In[ ]:


colormap = plt.cm.RdBu # defyning colormap

plt.figure(figsize=(12,12)) # difyning plot size

# creating correlation plot
sns.heatmap(payroll_data[num_vars].astype(float).corr()
            , linewidths=0.5,vmax=1
            , square=True
            , cmap=colormap
            , linecolor='white'
            , annot=True)

plt.title('Pearson Correlation between salary variables', y=1.05, size=20)
plt.xticks(size=12) # defyning xticks size
plt.yticks(size=12); # defyning yticks size


# #### The most frequent names on "nome" variable
# 
# "silva" and "santos" are commom last names in Brazil and here in my city this are not diferent, the majority of public workers in Campo Alegre also have this last names, "maria" and "jose" and commom first and middle names.

# In[ ]:


# setting all names to lowercase
payroll_data['nome'] = payroll_data['nome'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# joing all names from "nome" column
names = " ".join(name for name in payroll_data.nome)

# setting stopwords
stopwords = set(STOPWORDS)

# removing conjunctions
stopwords.update(["DA",'DE','DOS'])

# create and generate a wordcloud:
wordcloud = WordCloud(width=1920
                      , height=1080
                      , stopwords=stopwords
                      , max_font_size=250
                      , max_words=100
                      , background_color="white").generate(names)

# plotting the wordcloud:
plt.figure(figsize=(16,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Taking a look into the most frequent full names in Campo Alegre payroll we can see that "maria","jose","silva" and "santos appear in almost all names

# In[ ]:


# difyning plot size
plt.figure(figsize=(25,10))

# here i use .value_counts() to count the frequency that each category occurs of dataset
payroll_data['nome'].value_counts(ascending=True)[payroll_data['nome'].value_counts(ascending=True)>100].plot(kind='bar')

plt.title('Most Frequent Full Names',size=30)
plt.xlabel('Names',size=20) # defyning xlabel title
plt.ylabel('Frequency',size=20) # defyning ylabel title
plt.xticks(rotation=45,size=12) # defyning xticks rotation and size
plt.yticks(size=12); # defyning yticks size


# Plotting most commom full names liquid salarys we can see that the majority of workers with the two most frequent names earn less then 2000 Reais

# In[ ]:


# first let's select most commom names
names = ['maria cicera dos santos','maria quiteria da silva'
         ,'maria de lourdes dos santos','maria jose dos santos silva'
         ,'alex sandro de souza','maria jose da silva','maria jose dos santos']

# selecting names
names = payroll_data['nome'][payroll_data['nome'].isin(names)]

# selecting salaries by most commom names
salary = payroll_data['liquido'][payroll_data['nome'].isin(names)]

# joing names and salaries
dados = pd.concat([names,salary],axis=1)

# defyning plotsize
plt.figure(figsize=(25,10))

# creating the plot
sns.boxplot(x='nome'
            , y='liquido' # defyning y axis
            , data=dados # defyning the dataset
            , palette="bright") # defyning the color palette

plt.title('Most frequent names salary distribution',size=30)
plt.xlabel('Names',size=20)
plt.ylabel('Liquid Salary',size=20)
plt.xticks(rotation=45,size=12)
plt.yticks(size=12);


# #### Extracting weekly workload
# 
# The weekly workload information was distributed into "cargo" and "nivel" variables, soo i use text mining tecnics to extract this information, when the information was not available i defyne the workload as "unknow"

# In[ ]:


# fist let's create 6 new variables to indicate the workers that have 40, 35, 30, 25, 20 or 15 weekly workload
payroll_data['weekly_workload_40'] = np.where((payroll_data['cargo'].str.extract(r'(40)')=='40') | 
                                              (payroll_data['nivel'].str.extract(r'(40)')=='40') | 
                                              (payroll_data['nivel'].str.extract(r'(40)')=='40 horas'),1,0)

payroll_data['weekly_workload_35'] = np.where((payroll_data['cargo'].str.extract(r'(35)')=='35') | 
                                              (payroll_data['nivel'].str.extract(r'(35)')=='35') | 
                                              (payroll_data['nivel'].str.extract(r'(35)')=='35 horas'),1,0)

payroll_data['weekly_workload_30'] = np.where((payroll_data['cargo'].str.extract(r'(30)')=='30') | 
                                              (payroll_data['nivel'].str.extract(r'(30)')=='30') | 
                                              (payroll_data['nivel'].str.extract(r'(30)')=='30 horas'),1,0)

payroll_data['weekly_workload_25'] = np.where((payroll_data['cargo'].str.extract(r'(25)')=='25') | 
                                              (payroll_data['nivel'].str.extract(r'(25)')=='25') | 
                                              (payroll_data['nivel'].str.extract(r'(25)')=='25 horas'),1,0)

payroll_data['weekly_workload_20'] = np.where((payroll_data['cargo'].str.extract(r'(20)')=='20') | 
                                              (payroll_data['nivel'].str.extract(r'(20)')=='20') | 
                                              (payroll_data['nivel'].str.extract(r'(20)')=='20 horas'),1,0)

payroll_data['weekly_workload_15'] = np.where((payroll_data['cargo'].str.extract(r'(15)')=='15') | 
                                              (payroll_data['nivel'].str.extract(r'(15)')=='15') | 
                                              (payroll_data['nivel'].str.extract(r'(15)')=='15 horas'),1,0)

# now let's create a new categorical variable using the variables above
payroll_data['weekly_workload'] = np.where(payroll_data['weekly_workload_40']==1
                                           ,'40H','UNKNOW')

payroll_data['weekly_workload'] = np.where(payroll_data['weekly_workload_35']==1
                                           ,'35H',payroll_data['weekly_workload'])

payroll_data['weekly_workload'] = np.where(payroll_data['weekly_workload_30']==1
                                           ,'30H',payroll_data['weekly_workload'])

payroll_data['weekly_workload'] = np.where(payroll_data['weekly_workload_25']==1
                                           ,'25H',payroll_data['weekly_workload'])

payroll_data['weekly_workload'] = np.where(payroll_data['weekly_workload_20']==1
                                           ,'20H',payroll_data['weekly_workload'])

payroll_data['weekly_workload'] = np.where(payroll_data['weekly_workload_15']==1
                                           ,'15H',payroll_data['weekly_workload'])

# let's remove the binary variables from our dataset and see the first 5 rows
payroll_data = payroll_data.drop(['weekly_workload_40','weekly_workload_35'
                                  ,'weekly_workload_30','weekly_workload_25'
                                  ,'weekly_workload_20','weekly_workload_15'],axis=1)

payroll_data.head(5)


# #### Weekly workload distribution and Average liquid salaries by weekly workload
# 
# Clearly the majority of workers have 40 hours per week contracts...

# In[ ]:


# difyning plot size
plt.figure(figsize=(15,15))

 # here i use .value_counts() to count the frequency that each category occurs of dataset
payroll_data['weekly_workload'].value_counts().plot(kind='pie'
                                                    , colormap='Set2'
                                                    , autopct='%1.1f%%' # adding percentagens
                                                    , shadow=True
                                                    , startangle=140)
plt.legend();


# Curiously workers that have 30 hours weekly workload earn on average more than a double that who have a 40 hours weekly workload

# In[ ]:


# taking weekly workload data
workload = payroll_data['weekly_workload']

# taking liquid salary data
salary = payroll_data['liquido']

# joing
dados = pd.concat([workload,salary],axis=1)

# defyning plotsize
plt.figure(figsize=(25,10))

# creating the plot
sns.barplot(x='weekly_workload'
            , y='liquido' # defyning y axis
            , data=dados # defyning the dataset
            , palette="bright") # defyning the color palette

plt.title('Average liquid salary per weekly workload',size=30)
plt.xlabel('Weekly workload',size=20) # setting xlabel title
plt.ylabel('Average liquid salary',size=20) # setting ylabel title
plt.xticks(rotation=45,size=12) # defyning xticks rotation and size
plt.yticks(size=12); # defyning yticks size


# #### But what job positions are more frequent in a 30-hour week journey?
# 
# "PROFESSOR","MEDICO" and "ASSISTENTE SOCIAL" are the most frequent job positions with this journey

# In[ ]:


# selecting job positions that have a 30-hour week journey
workload_30H_jobs = payroll_data['cargo'][payroll_data['weekly_workload']=='30H']

# difyning plot size
plt.figure(figsize=(25,10))

# here i use .value_counts() to count the frequency that each category occurs of dataset
workload_30H_jobs.value_counts(ascending=True)[workload_30H_jobs.value_counts(ascending=True)>5].plot(kind='bar', color='green')

plt.title('Job Positions that have a 30-hour week journey',size=30)
plt.xlabel('Job Positions',size=20)
plt.ylabel('Frequency',size=20)
plt.xticks(rotation=45,size=12)
plt.xticks(size=12);


# To finish this part of our analysis let's see the relation between numericl features clustered by weekly workload

# In[ ]:


num_vars = ['valor_base','proventos','descontos','liquido','weekly_workload']

# pair plot with numerical variables in iris dataset
sns.pairplot(payroll_data[num_vars],hue='weekly_workload',aspect=0.3*5);


# #### The most frequent job positions
# 
# Here I dicide to start removing punctuation, weekly workload infomation and contract information from the data to clean and doo a more clean wordcloud. In the wordcloud we can see that the most frequent job positions in Campo Alegre city are "professors", "aux servicos", "servicos gerais" and "auxiliar turma". it's very interesting see "professors" leading job positions on our city, let's analyse this better below.

# In[ ]:


# converting all text to lowercase
payroll_data['cargo'] = payroll_data['cargo'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# removing punctuation
payroll_data['cargo'] = payroll_data['cargo'].str.replace('[^\w\s]','')

# passing words to be removed as stopwords
stop = ["40h",'25h','20h','cc','c','e','ni','nii','niii'
        ,'b','f','i','g','a','contratado','contrato','de'
       ,'niv','da','do','cc5','mun','d','nvi','30h','cc4'
        ,'nv','h','n','cc7','nmag','3','ccs','art','7'
        ,'cc6','cc3','cc2','cc1']

# removing stopwords
payroll_data['cargo'] = payroll_data['cargo'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[ ]:


# joing all text data into cargo variable
cargo = " ".join(name for name in payroll_data.cargo)

# create and generate a wordcloud:
wordcloud = WordCloud(width=1920
                      , height=1080
                      , stopwords=stopwords
                      , max_font_size=250
                      , max_words=100
                      , background_color="white").generate(cargo)

# plotting the wordcloud
plt.figure(figsize=(16,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# #### Professor: Teachers dominating the public sector
# 
# Taking a look into most frequent job positions we can see "professor" leading with wide slack appearing more then 25000 times in our data

# In[ ]:


# difyning plot size
plt.figure(figsize=(25,10))

# here i use .value_counts() to count the frequency that each category occurs of dataset
payroll_data['cargo'].value_counts(ascending=True)[payroll_data['cargo'].value_counts(ascending=True)>1000].plot(kind='bar')

plt.title('Most frequent job positions',size=30)
plt.xlabel('Job Positions',size=20)
plt.ylabel('Frequency',size=20)
plt.xticks(rotation=45,size=12)
plt.yticks(size=12);


# Seeing salaries per most frequent jobs we can see that "enfermeiro" have the higher average salary, some "professor" earn more then 5000 Reais but the majority earn less then 2500 Reais

# In[ ]:


# selecting job positions
cargo = ['professor','aux servicos gerais','auxiliar turma'
         ,'agente comunitario saude','vigilante','gari','enfermeiro'
         ,'aux serv educacionais','aux servicos gerais externos'
         ,'monitor','agente administrativo','motorista'
         ,'vigilante escolar','assistente administrativo'
        ,'tecnico enfermagem','assist adm educacional'
         ,'aux serv gerais internos','agente endemias']

# selecting job positions
cargos = payroll_data['cargo'][payroll_data['cargo'].isin(cargo)]

# selecting liquid salaries by the selected job positions
salary = payroll_data['liquido'][payroll_data['cargo'].isin(cargo)]

# joing
dados = pd.concat([cargos,salary],axis=1)

# defyning plotsize
plt.figure(figsize=(25,10))

# creating the plot
sns.boxplot(x='cargo', # defyning x axis
            y='liquido' # defyning y axis
            ,data=dados # defyning the dataset
            ,palette="bright") # defyning the color palette

plt.title('Most common job positions salary distribution',size=30)
plt.xlabel('Job Position',size=20)
plt.ylabel('Liquid Salary',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# The great majority of teachers earns less then 5000 Reais and only a few outliers pass this salary, based on my expirience about teachers salaries here in my city i deduce that this professor that earn more then 5000 Reais are coordinators and directors registered as teachers.

# In[ ]:


teachers = payroll_data['liquido'][payroll_data['cargo']=="professor"]

# defyning the plotsize
plt.figure(figsize=(18,10))

# creating the distribution plot
sns.distplot(teachers
             , bins=50
             , color='blue')

plt.title("Teachers salaries distribution",size=30)
plt.xlabel("Salaries",size=20) # defyning xlabel
plt.xticks(size=15) # defyning xticks size
plt.yticks(size=15); # defyning yticks size


# Teachers that have 40 hours weekly workload have higher average salaries followed bye 25 hours, above we saw that jobs with 30 hours weekly workload seems to have higher salarys than jobs that have 40 hours, this seems to be not the case of teachers.

# In[ ]:


# defyning the plotsize
plt.figure(figsize=(25,10))

# creating the distribution plot
sns.barplot(y=teachers
            , x=payroll_data['weekly_workload']
            , palette="bright")

plt.title("Teachers salaries per weekly workload",size=30)
plt.xlabel("Weekly workload",size=20); # defyning xlabel
plt.ylabel("Average Liquid Salary",size=20); # defyning ylabel
plt.xticks(rotation=45, size=15) # defyning xticks size
plt.yticks(size=15); # defyning yticks size


# How this teachers salarys grown with passing years?

# In[ ]:


# converting date column to date format
payroll_data['date'] = payroll_data['date'].apply(lambda x: datetime.strptime(x, '%b %Y'))

# defyning the plotsize
plt.figure(figsize=(25,10))

# creating the distribution plot
sns.lineplot(y=teachers
             , x=payroll_data['date']
             , marker='o'
             , color='green')

plt.title("Teachers salaries over the years",size=30)
plt.xlabel("Time",size=20); # defyning xlabel
plt.ylabel("Average Liquid Salary",size=20) # defyning ylabel
plt.xticks(size=15) # defyning xticks size
plt.yticks(size=15); # defyning yticks size


# #### Most frequent contract levels
# 
# The majority of contracts are "contrato", here in Brazil it's very commom contract someone by contract instead of CLT regime to avoid costs and problems with justice of work. First let's treat our data then generate a wordcloud.

# In[ ]:


# converting to lowercase
payroll_data['nivel'] = payroll_data['nivel'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# removing punctuatiion
payroll_data['nivel'] = payroll_data['nivel'].str.replace('[^\w\s]','')

# passing words to be removed as stopwords
stop2 = ["40h",'25h','20h','cc','c','e','ni','nii','niii'
        ,'b','f','i','g','a','de','niv','da','do','cc5'
        ,'mun','d','nvi','30h','cc4','nv','h','nmag','3'
        ,'ccs','art','7','cc6','cc7','cc3','cc2','cc1','iii'
       ,'ii','iv','vi','v','n','mag','6','5','2','cc8','cc9'
       ,'10','11','12','13','l','4','ic','iic','iiic','ie'
         ,'iie','iiie','if','iif','iiif','ig','iig','iiig']

# removing stopwords
payroll_data['nivel'] = payroll_data['nivel'].apply(lambda x: " ".join(x for x in x.split() if x not in stop2))

# replacing empty contract levels per "other"
payroll_data['nivel'] = np.where(payroll_data['nivel']=='','other',payroll_data['nivel'])


# In[ ]:


# joing all words from "nivel" column
nivel = " ".join(name for name in payroll_data.nivel)

# create a wordcloud:
wordcloud = WordCloud(width=1920
                      , height=1080
                      , stopwords=stopwords
                      , max_font_size=250
                      , max_words=100
                      , background_color="white").generate(nivel)

# plotting the wordcloud:
plt.figure(figsize=(16,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Clearly "contrato" is the most frequent contract level present in our data, let's see how many workers are employed by "contrato"

# In[ ]:


# difyning plot size
plt.figure(figsize=(25,10))

# here i use .value_counts() to count the frequency that each category occurs of dataset
payroll_data['nivel'].value_counts(ascending=True)[payroll_data['nivel'].value_counts(ascending=True)>1000].plot(kind='bar')

plt.title('Most frequent contract level',size=30)
plt.xlabel('contract type',size=20)
plt.ylabel('Frequency',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# How much each of this contract levels earns?

# In[ ]:


# selecting the most frequent contracts
level = ['nivel','gari','contratado','magisterio'
         ,'comissionado','other','contrato']

# taking contract levels
levels = payroll_data['nivel'][payroll_data['nivel'].isin(level)]

# taking salaries that match with most frequent contract levels
salary = payroll_data['liquido'][payroll_data['nivel'].isin(level)]

# joing
dados = pd.concat([levels,salary],axis=1)

# defyning plotsize
plt.figure(figsize=(25,10))

# creating the plot
sns.boxplot(x='nivel'
            , y='liquido'
            , data=dados 
            , palette="bright")

plt.title('Most common contract level salary distribution',size=30)
plt.xlabel('Contract type',size=20)
plt.ylabel('Liquid Salary',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# #### The most payed jobs
# 
# When we take a look into average salary from jobs that earn 10000 Reais or more we can see what jobs earns more money, "diretor geral medico hospitalar", "diretor medico hospitalar" and "contator" are the most payed jobs in Campo Alegre City.

# In[ ]:


# separating job positions where the salarys are equal or above them 10000 Reais
cargo = payroll_data['cargo'][(payroll_data['liquido']>=10000)]

# separating salarys that are equal or above them 10000 Reais
salary = payroll_data['liquido'][(payroll_data['liquido']>=10000)]

# defyning plotsize
plt.figure(figsize=(30,10))

# creating the plot
salary.groupby(cargo).mean().sort_values(ascending=True).plot(kind='bar',color="blue")

plt.title('Job position average salary that earn 10000 Reais or more',size=30)
plt.xlabel('Job Position',size=20)
plt.ylabel('Mean Salary',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# What contract types this jobs are?

# In[ ]:


# separating job positions where the salarys are equal or above them 10000 Reais
levels = payroll_data['nivel'][(payroll_data['liquido']>=10000)]

# separating salarys that are equal or above them 10000 Reais
salary = payroll_data['liquido'][(payroll_data['liquido']>=10000)]

# defyning plotsize
plt.figure(figsize=(30,10))

# creating the plot
salary.groupby(levels).mean().sort_values(ascending=True).plot(kind='bar',color="blue")

plt.title('Contract levels average salary that earn 10000 Reais or more',size=30)
plt.xlabel('Contract level',size=20)
plt.ylabel('Mean Salary',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# #### The grown and decreasing of salaries with passing years
# 
# Here I decide to analyse salaries evolution in time using two groups, first with salaries equal or above 10000 Reais and the second with salaries below 10000 Reais, I decide to separate in this two groups to undertand if only one of this groups have salarie increases.

# Let's start with salaries equal or above 10000 Reais, clearly we have a fall of average salaries from 2016 to 2018 and then the average was estabilized

# In[ ]:


# taking dates that match salaries equal or above 10000
cargo = payroll_data['date'][(payroll_data['liquido']>=10000)]

# taking below 10000
salary = payroll_data['liquido'][(payroll_data['liquido']>=10000)]

# joing
dados = pd.concat([cargo,salary],axis=1)

# defyning plotsize
plt.figure(figsize=(30,10))

# creating the plot
sns.lineplot(x='date'
            , y='liquido'
             , marker='o'
            , data=dados
            , palette="bright") 

plt.title('Drop in average wages above 10000 Reais over the years',size=30)
plt.xlabel('Time',size=20)
plt.ylabel('Average Liquid Salary',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# Now let's see salaries below 10000 Reais, the salaries seems to have grown with passing years always with pikes in every year start

# In[ ]:


# taking dates that match salaries equal or above 10000
cargo = payroll_data['date'][(payroll_data['liquido']<10000)]

# taking salaries below 10000
salary = payroll_data['liquido'][(payroll_data['liquido']<10000)]

# joing
dados = pd.concat([cargo,salary],axis=1)

# defyning plotsize
plt.figure(figsize=(30,10))

# creating the plot
sns.lineplot(x='date'
             , y='liquido'
             , marker='o'
             , data=dados
             , color='green')

plt.title('Average salary growth over the years of salaries below 10000 Reais',size=30)
plt.xlabel('Time',size=20)
plt.ylabel('Average Liquid Salary',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# And if we look to all salaries average by date?

# In[ ]:


# taking dates
date = payroll_data['date']

# taking discounts
salary = payroll_data['liquido']

# joing
dados = pd.concat([date,salary],axis=1)

# defyning plotsize
plt.figure(figsize=(30,10))

# creating the plot
sns.lineplot(x='date'
             , y='liquido'
             , marker='o'
             , data=dados
             , color='purple')

plt.title('Average salary over the years',size=30)
plt.xlabel('Time',size=20)
plt.ylabel('Average Liquid Salary',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# In[ ]:


# selecting total salarys per year
salaries_2016 = np.mean(payroll_data['liquido'][payroll_data['date']=="2016"])
salaries_2017 = np.mean(payroll_data['liquido'][payroll_data['date']=="2017"])
salaries_2018 = np.mean(payroll_data['liquido'][payroll_data['date']=="2018"])
salaries_2019 = np.mean(payroll_data['liquido'][payroll_data['date']=="2019"])

years = ['2016','2017','2018','2019']
salaries = [salaries_2016,salaries_2017,salaries_2018,salaries_2019]

# defyning the plotsize
plt.figure(figsize=(25,10))

# creating the plot
sns.lineplot(y=salaries
             , x=years
             , marker='o'
             , palette="bright")

plt.title("Average wage growth over the years",size=30)
plt.xlabel("Years",size=20); # defyning xlabel
plt.ylabel("Average salary",size=20) # defyning ylabel
plt.xticks(rotation=45, size=15) # defyning xticks size
plt.yticks(size=15); # defyning yticks size


# One interesting variables is the discounts, let's see how this variable change with passing years

# In[ ]:


# taking dates
date = payroll_data['date']

# taking discounts
discounts = payroll_data['descontos']

# joing
dados = pd.concat([date,discounts],axis=1)

# defyning plotsize
plt.figure(figsize=(30,10))


sns.lineplot(x='date'
             , y='descontos'
             , marker='o'
             , data=dados
             , color="red")

plt.title('Average discounts over the years',size=30) 
plt.xlabel('Time',size=20)
plt.ylabel('Average Discounts',size=20)
plt.xticks(rotation=45,size=15)
plt.yticks(size=15);


# How the total costs with payroll have grown with passing years

# In[ ]:


# selecting total cost per year
costs_2016 = np.sum(payroll_data['liquido'][payroll_data['date']=="2016"])
costs_2017 = np.sum(payroll_data['liquido'][payroll_data['date']=="2017"])
costs_2018 = np.sum(payroll_data['liquido'][payroll_data['date']=="2018"])
costs_2019 = np.sum(payroll_data['liquido'][payroll_data['date']=="2019"])

# creating two lists with years and costs
years = ['2016','2017','2018','2019']
costs = [costs_2016,costs_2017,costs_2018,costs_2019]

plt.figure(figsize=(25,10)) # defyning the plotsize

# creating the plot
sns.lineplot(y=costs
             , x=years
             , marker='o'
             , palette="bright")

plt.title("Cost with payroll over the years",size=30)
plt.xlabel("Years",size=20)
plt.ylabel("Costs in Reais",size=20)
plt.xticks(rotation=45, size=15)
plt.yticks(size=15);


# ### Conclusions
# 
# **The total payroll cost grown with passing years?**
#    - The payroll cost fell from 2016 to 2017 and then started to grow continuously until 2019, at the moment we still don't have data to observe a growth in 2020.
#  
# **The salaries grown with passing years?**
#    - On average, wages grew over the years, falling from 2016 to 2017 and then increased again until 2019, wages above 10000 Reals fell from 2016 to 2018 and then stabilized, while wages below 10000 have been increasing since 2016 to today.
#  
# **What the most frequent job position and how much they earn?**
#    - "professor" are the most frequent job position in Campo Alegre, earn em average 2058 Reais and with majority workers earning between 1000 and 5000 Reais, some outliers with "professor" job position earns more then 5000 Reais, probably this workers are not teachers, but school directors and coordenators.
#  
# **What job positions have higher salaries?**
#    - "diretor geral medico hospitalar", "diretor medico hospitalar", "contador", "procurador 2 classe" and "medico" are the jobs with higher salaries in Campo Alegre earning more than 10000 Reais a month and some of then earning more than 20000 Reais a month.
#  
# **What contract levels have higher salaries?**
#    - "procurador classe", "magisterio", "comissionado", "prefeito", "contrato/contratado" and "medico autorizador" are the contract levels with higher salaries, all of then earning more then 10000 Reais a month.

# ## ***Thank you very much for read this kernel, if you think that this kernel was useful please give a upvote, i really appriciate that :)***

# In[ ]:




