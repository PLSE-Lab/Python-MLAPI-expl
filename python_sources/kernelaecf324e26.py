#!/usr/bin/env python
# coding: utf-8

# # WebScraping - project-3 website : https://influence.co/category/riyadh

# ## Problem statment
# - This dataset is for local (Saudi Arabia) social media influencers, and the dataset is built using web scraping to get influencers information from https://influence.co . The dataset focused on Instagram influencers in Saudi Arabia and contains 5 attributes and 243 rows. In particular, the dataset has the Instagram id for the influencers,number of followers, the category name that they belong to and level of impact of influencers on Instagramwhich is the avg engagement rate.
# 
# ### Data Set Information: 
# - IG_id, The influencer Instagram id, object.
# - No_followers,  The number of followers the influencer have, int64.
# - Category_name, the category which the influencer belongs to (# Here I assumed that when there were other social media platforms, I would replace them with the name of persons in these programs, for example,  'snapchat - lifestyle', youtube - vlogger','Facebook, Blogger'), object.
# - Locations, the influencer location (based city), object.
# - engagment_rate_avg , the engagement rate the influencer have in % ,float64.
# ### The data can be used to clustering the influencers according to their Category_name, their avg engagement rate and also the number of followers they have.

# ## Import moduels

# In[ ]:


import re
import requests
import pandas as pd
from scrapy.selector import Selector
from time import sleep
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")


# ## Web scraping

# In[ ]:


# lists for saving values after scraping
ig_id_list=[]
ig_name_list=[]
ig_name_list=[]
category_name=[]
locations=[]
followers_no=[]
engagment_rate_avg=[]
engagment_rate_link=[]


# In[ ]:


# Links list to request the page (some pages didnot work so they need to be called directly using request)
links_list=['https://influence.co/category/riyadh/1','https://influence.co/category/riyadh/2'
            ,'https://influence.co/category/riyadh/3','https://influence.co/category/riyadh/4'
            ,'https://influence.co/category/riyadh/5','https://influence.co/category/riyadh/6'
            ,'https://influence.co/category/riyadh/7','https://influence.co/category/riyadh/8'
            ,'https://influence.co/category/riyadh/9','https://influence.co/category/riyadh/10'
            ,'https://influence.co/category/riyadh/11','https://influence.co/category/riyadh/12']


# In[ ]:


# Starting the web scraping for each link using get_influencers(link)
for link in links_list:
    get_influencers(link)


# In[ ]:


# The function scraping_influencers(item_lst) will use xpath and regx to extract valuse from each page
def scraping_influencers(item_lst):
    for section in item_lst:
        """"
        get the infuencer ig_id,ig_name,
        catogray of the infuencer,location,
        number of followers.
        +
        the engagment_rate is on other page so in the function 
        it will save the link of that page to request it later.
        """

        exp_ig_link ='div[@class="middle"]/h4/a/@href'
        exp_ig_name='div[@class="middle"]/h4/a/text()'
        exp_ig_id ='div[@class="middle"]/p/a/text()'
        exp_category='div[@class="middle"]/p[2]/span[@class="category-name"]/a/text()'
        exp_location= 'div[@class="middle"]/p[3][@style="margin-bottom: 0px;"]/text()'
        exp_followers='div[@class="bottom clearfix"]/p[@class="pull-left"]/text()'
        ig_link= section.xpath(exp_ig_link).extract()[0]
        ig_id_list.append(str(section.xpath(exp_ig_id).extract()[0]).replace("/",'@'))
        ig_name_list.append(section.xpath(exp_ig_name).extract()[0])
        category_name.append(str(section.xpath(exp_category).extract()).split(','))
        locations.append(str(section.xpath(exp_location).extract()[1]).split(',')[0])
        followers_no.append(re.findall("\d+\.?\d+\w",section.xpath(exp_followers).extract()[1])[0])
        engagment_rate_link.append('https://influence.co'+ig_link)
       
        
        


# In[ ]:


# The function get_influencers(link) will do the request for each main link
def get_influencers(link): 
    """"
    create selector and choose the part that containes the 
    influencer informations and called  scraping_influencers(item_lst) to scrap the page 
    and extract infos
    """"
    response=requests.get(link)
    response.status_code
    HTML=response.text
    sel = Selector(text=HTML)
    exp = '//div[contains(@class,"influencer-card styled")]'
    item_lst = sel.xpath(exp)
    print("Done")
    scraping_influencers(item_lst)


# In[ ]:


# The function get_influencers_engagment_rate(link) will do the request for the engagment rate page
def get_influencers_engagment_rate(link):
    """"
    get the engagment rate page for each infeluner using the engagment_rate_link and
    requst that page
    +
    The engagment rate page  have tow styles:
    1. if the website knows the engagment rate for this influencer(the page with specific html design)
    2. if the website doesn't know the engagment rate for this influencer it
    will give a engagment rate similer to the same ifluncer in terms of catagory and so on 
    (the page with different html design)
    to deal with this differentiation there will be two  exp_for_page_style and if satatment 
    to know each page belonges to which style
    """"
    rate=0
    response=requests.get(link)
    HTML=response.text
    sel = Selector(text=HTML)
    exp = '/html'
    item_lst = sel.xpath(exp)

    for section in item_lst:
        exp_for_page_style1='//*[@id="header-section"]/div[2]/div/div/div/div/div[2]/div/div[1]/div/div/div[2]/p/span/text()'
        exp_for_page_style2='/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div[2]/article/div/p/strong/text()'
        rate1= section.xpath(exp_for_page_style1).extract()
        rate2= section.xpath(exp_for_page_style2).extract()
        if rate1 != []:
            rate=re.findall("\d+?\.?\d+?\w?",str(rate1))[0]
        elif rate2 != []:
            rate=re.findall("\d+?\.?\d+?\w?",str(rate2))[0]
    return rate
 


# In[ ]:


# Loop throgh the  engagment_rate_link list to call get_influencers_engagment_rate(link) function wich will do the engagment rate scraping
for link in engagment_rate_link:
    
    engagment_rate_avg.append(get_influencers_engagment_rate(link))
    # to get all values lcorrectly  the program needs to sleep 3 seconds
    sleep(3)


# In[ ]:


# check the engagment_rate_avg list 
len(engagment_rate_avg)


# In[ ]:


# create a df
Influncers_Dataset= pd.DataFrame({'IG_id':ig_id_list,'IG_name':ig_name_list,'No_followers':followers_no,'Category_name':category_name,'Locations':locations,'engagment_rate_avg':engagment_rate_avg})


# In[ ]:


# get df head
Influncers_Dataset.head()


# In[ ]:


# save it to csv to keep data save and no need to wait for scraping if something get wrong later
Influncers_Dataset.to_csv('Top Riyadh Influencers');


# ### Start Cleaning Data

# In[ ]:


# get data from the csv file
Influncers_Dataset=pd.read_csv('Top Riyadh Influencers')


# In[ ]:


# check the first 10 rows 
Influncers_Dataset.head(10)


# In[ ]:


# Clean the Category_name by removing Riyadh and other special charecters
Influncers_Dataset['Category_name'] = Influncers_Dataset['Category_name'].map(lambda x: str(re.findall("\w+",x)).replace('Riyadh',"").strip('[]'))


# In[ ]:


# reducing words to their word stem by using PorterStemmer
stemmer= PorterStemmer()


# In[ ]:


# domwnload punkt to use stemmer
import nltk
nltk.download('punkt')


# In[ ]:


# domwnload stopwords to use in nlp cleaning
import nltk
nltk.download('stopwords')


# In[ ]:


# Cleaning the Category_name column and using stemming to reduce each catagry to their word stem and remove stop words  
for i in range(600):
    new_word=''
    input_str=Influncers_Dataset['Category_name'].iloc[i]
    stop_words = set(stopwords.words('english'))
    input_str=word_tokenize(input_str)
    for word in input_str:
        if word not in stop_words:
            new_word+=stemmer.stem(word)
    Influncers_Dataset['Category_name'].iloc[i]=re.findall("\w+",new_word)


# In[ ]:


# Remove repeated catagray for same influncer after doing stemming using set
for i in range(600):
    Influncers_Dataset['Category_name'].iloc[i]=list(set(Influncers_Dataset['Category_name'].iloc[i]))


# In[ ]:


# Here I assumed that when there were other social media platforms,
# I would replace them with the name of persons in these programs foe example = 'snapchat - lifestyle' ,'youtub - vlogger','facebook,Blogger'
for i in range(600):
    Influncers_Dataset['Category_name'].iloc[i]=str(Influncers_Dataset['Category_name'].iloc[i]).replace('snapchat','lifestyle')
    Influncers_Dataset['Category_name'].iloc[i]=str(Influncers_Dataset['Category_name'].iloc[i]).replace('youtub','vlogger')
    Influncers_Dataset['Category_name'].iloc[i]=str(Influncers_Dataset['Category_name'].iloc[i]).replace('facebook','Blogger')


# In[ ]:


# Remove list bracets 
for i in range(600):
    Influncers_Dataset['Category_name'].iloc[i]=Influncers_Dataset['Category_name'].iloc[i].strip('[]')


# In[ ]:


# unique Category_name values 
Influncers_Dataset['Category_name'][Influncers_Dataset['Category_name']!= ''].unique()


# In[ ]:


# There is 357 accounts out of 600 with empty catagry_name 
len(Influncers_Dataset[Influncers_Dataset['Category_name']== ''])


# In[ ]:


# Delete the rows with empty catagry_name 
Influncers_Dataset=Influncers_Dataset[Influncers_Dataset['Category_name']!= '']


# In[ ]:


# Transform engagment_rate_avg to float
Influncers_Dataset['engagment_rate_avg']=Influncers_Dataset['engagment_rate_avg'].apply( lambda x:float(x))


# In[ ]:


type(Influncers_Dataset['engagment_rate_avg'].iloc[0])


# In[ ]:


# delete useless columns (unused)
useless_column=['Unnamed: 0','Unnamed: 0.1','IG_name']
Influncers_Dataset=Influncers_Dataset.drop(useless_column,axis=1)


# In[ ]:


# Change 'No_followers' columns from number k to float and add zeros 


# In[ ]:


Influncers_Dataset['No_followers']=Influncers_Dataset['No_followers'].apply( lambda x:x.strip('k'))


# In[ ]:


Influncers_Dataset['No_followers']=Influncers_Dataset['No_followers'].apply( lambda x:re.findall("\d+",x))


# In[ ]:


Influncers_Dataset['No_followers']=Influncers_Dataset['No_followers'].apply( lambda x:str(x).strip('[]'))


# In[ ]:


Influncers_Dataset['No_followers']=Influncers_Dataset['No_followers'].apply( lambda x:x.strip("'").strip(''))


# In[ ]:


for i in range(243):
    
    if len(Influncers_Dataset['No_followers'].iloc[i])==6:
        number =Influncers_Dataset['No_followers'].iloc[i]
        number=number[0]+number[5]+'00'
        Influncers_Dataset['No_followers'].iloc[i]=float(number)
    elif len(Influncers_Dataset['No_followers'].iloc[i])==7:
        number =Influncers_Dataset['No_followers'].iloc[i]
        number=number[0]+number[1]+number[6]+'00'
        Influncers_Dataset['No_followers'].iloc[i]=float(number)
    else:
        
        number =Influncers_Dataset['No_followers'].iloc[i]
        number =number+'000'
        Influncers_Dataset['No_followers'].iloc[i]=float(number)
       


# In[ ]:


# Save to Top Riyadh Influencers.csv
Influncers_Dataset.to_csv('Top Riyadh Influencers',index=False)


# In[ ]:


# read results
Influncers_Dataset_final=pd.read_csv('Top Riyadh Influencers')


# In[ ]:


# check results
Influncers_Dataset_final


# In[ ]:


Influncers_Dataset_final.dtypes


# In[ ]:


# Heatmap to see correlation between numerical values (No_followers ,engagment_rate_avg ) (No correlation)
fig, ax = plt.subplots(figsize=(10, 8))
corr = Influncers_Dataset_final[["No_followers" ,"engagment_rate_avg" ]].corr()
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)


# In[ ]:


# see the relation between x="Locations",y="No_followers"
fig, ax = plt.subplots(figsize=(200,100))
sns.barplot(x="Locations",y="No_followers",data=Influncers_Dataset_final,ax=ax)
 


# In[ ]:


# see relation between x="Locations",y="engagment_rate_avg"
fig, ax = plt.subplots(figsize=(200,100))
sns.barplot(x="Locations",y="engagment_rate_avg",data=Influncers_Dataset_final,ax=ax)
 


# In[ ]:


# see realtion between x="Category_name",y="engagment_rate_avg"
fig, ax = plt.subplots(figsize=(200,100))
sns.barplot(x="Category_name",y="engagment_rate_avg",data=Influncers_Dataset_final,ax=ax)
 


# In[ ]:


# see relation between x="engagment_rate_avg",y="No_followers"
fig, ax = plt.subplots(figsize=(200,100))
sns.barplot(x="engagment_rate_avg",y="No_followers",data=Influncers_Dataset_final,ax=ax)
 

