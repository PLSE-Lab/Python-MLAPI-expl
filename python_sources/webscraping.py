#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is  to find the most common words found in the job advertisments for "data scientist" positions. 
# I wrote this code snippets in my jupyter notebook and uploaded it here later. 
# The code snippets will go over how to create a new environment using Anaconda prompt, install required packages in it, plays around with `beautifulsoup` and `requests` and uses `selenium` to scrap the web. Finaly using `collections` and `nltk` to find the most frequent words. 

# In[ ]:


#Viewing a list of environment, insert these line in the Anacond Prompt
conda info --envs  #OR
conda env list
# Viewing a list of packages in an enviroinment
conda list -n myenv
# Creating a new wnviroinment with the same version of python that currently using
conda create --name myenv
# Create new environment with different version of python
conda create -n myenv python=3.4
# Create new env with a specific version of package
conda create -n myenv scipy=0.15.0
# It is better to create all the packages you need all at once to prevent dependency problems

conda create -n webscap bs4 requests scrapy lxml selenium html5lib json re pandas # was not possible to install bs4, json, re
deactivate #deactivate base environment
activate webscap
conda config --set changeps1 true # To show the name of env in () after the prompt
conda config --env --add channels conda-forge # to add the conda-forge channel to the active enviroinment
conda install -n webscap -c conda-forge beautifulsoup4 # or add the channel manually to the conda install


# In[15]:


# make sure the enviroiment is right
get_ipython().system('conda env list ')


# In[17]:


# Reason I have to do the following:
# In the new environment I created `webscap` I was not able to `import selenium` in jupyter but I was able to do it in python prompt
# It turned out the `sys.path` was different in jupyter ans python prompt. In jupyter there was no path 
# looking at the `\\envs\\webscap\\`, therefore what I am doing in the following is that I am adding the python sys.path to 
# the jupyter sys.path

import sys
# this is the path where jupyter looks when executing the notebook or doing lots of other stuff
sys.path
for element in r'C:\Users\zahrae\Anaconda3\envs\webscap\python36.zip,C:\Users\zahrae\Anaconda3\envs\webscap\DLLs,C:\Users\zahrae\Anaconda3\envs\webscap\lib,C:\Users\zahrae\Anaconda3\envs\webscap,C:\Users\zahrae\Anaconda3\envs\webscap\lib\site-packages,C:\Users\zahrae\Anaconda3\envs\webscap\lib\site-packages\win32,C:\Users\zahrae\Anaconda3\envs\webscap\lib\site-packages\win32\lib,C:\Users\zahrae\Anaconda3\envs\webscap\lib\site-packages\Pythonwin'.split(','):
    sys.path.append(element)
sys.path


# In[1]:


# Playing around with beautifulsoup and requests
import bs4
import requests
import re
url='https://www.indeed.com/q-data-scientist-jobs.html'
url='https://www.indeed.com/viewjob?jk=89eb429be568276a&tk=1c7de7r141ek306c&from=serp&alid=3&advn=8876452989351355'
page=requests.get(url)
page.status_code #status code starting with a 2 generally indicates success,
                  #  and a code starting with a 4 or a 5 indicates an error.
#page.content
soup=bs4.BeautifulSoup(page.content,'html.parser')
#print(soup.prettify())
print([type(item) for item in list(soup.children)])
html = list(soup.children)[-2]


# In[19]:


# using requests and bs4 does not work because the attribute of the html page such as "id" and "class_" 
# are dynamic because of java script
container=soup.find(id="job-content").find_next("td").find_next("table").get_text()
container=soup.find(id='job_summary').get_text().lower()
#container?
re.sub(re.compile(r'[^a-zA-Z]+'),' ',container)
#alljobs=container.find_all('div',class_="row result clickcard") #returns an empty list
#links=alljobs.find_all("a")
#for job in alljobs:
#test=soup.find('span',class_='company')

#test.find("a")["href"]
#print(type(test.a),type((test.find("a"))))
#test.select("a[href]")


# In[20]:


import bs4 
import selenium
import time
import pandas as pd
import re
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# maximum number of pages to browse
num_pages_max = 7        
url='https://www.indeed.com'

# make sure to download the chromebrowser.exe and put it somewhere in the sys.path directory, 
# or give the path to it as the first argument of Chrome, i.e., Chrome(\path\to\chromebrowser.exe)
browser = selenium.webdriver.Chrome()
browser.delete_all_cookies()
browser.get(url)

delay=10 
search_key = 'data scientist'

#first clear the box and then write down the search key into it
# because I was having problem sometimes, I decided to clicke, clear and send_keys
try:
    WebDriverWait(browser,delay).until(EC.presence_of_element_located((By.ID,'text-input-what')))
    browser.find_element_by_id('text-input-what').click()
    browser.find_element_by_id('text-input-what').clear()
    browser.find_element_by_id('text-input-what').send_keys(search_key)
    browser.find_element_by_id('text-input-where').click()
    browser.find_element_by_id('text-input-where').clear()
    browser.find_element_by_id('text-input-where').send_keys('')
    browser.find_element_by_tag_name('button').click()
except TimeoutException:
    print('Browser took more than {} sec to load'.format(delay))
        
# with having all the above lines I couldn't remove the location out of search. bizzare! so I did the following:
print(browser.current_url)
url=re.sub(re.compile(r'&l=.*'),'',browser.current_url)

# the following function goes through desired number of pages to scrap the job summary in each page
def section_job_summary(browser):
    soup = bs4.BeautifulSoup(browser.page_source,"lxml")
    sectionDF=pd.DataFrame()
    alljobs = soup.find_all('div',class_='row result clickcard')
    if alljobs:
        titles=[]
        companies=[]
        locations=[]
        links=[]
        for job in alljobs:
            # job.a is equal to job.find('a')
            titles.append(job.find('a')['title'])
            links.append(job.find('a')['href'])
            companies.append(job.find('span',class_='company').get_text())
            locations.append(job.find('span',class_='location').get_text())
        prefix=r'http://www.indeed.com'
        links=[(prefix+link) for link in links if not link.startswith(prefix)]
        sectionDF=pd.DataFrame({'Title':titles,'Company':companies,'Location':locations,'Link':links})
    return sectionDF

jobsDF = pd.DataFrame()
num_pages = 1
while num_pages <= num_pages_max:
    
    browser.get(url)
    previous_url = url
    print('Processing page {0} from {1}'.format(num_pages, num_pages_max))
    
    sectionDF = section_job_summary(browser)
    jobsDF = pd.concat([jobsDF,sectionDF],axis='index',ignore_index=True)
    
    ## find the next button to go to the next page
    #next_button = browser.find_element_by_class_name('np')
    ## clicking the next button was tricky because .click() won't work.
    ##ActionChains(browser).move_to_element(next_button).click(next_button)
    ## I found a solution here: https://stackoverflow.com/questions/37879010/selenium-debugging-
    ##element-is-not-clickable-at-point-x-y-with-firefox-driver
    #browser.execute_script("arguments[0].click();", next_button)
    ##WebDriverWait(browser,delay).until(EC.url_to_be(url))
    #url=browser.current_url
    #print(url)
    
    if num_pages == 1:
        url=url+'&start={0:d}'.format(10)
    else:
        url=re.sub('&start.*','&start={0:d}'.format(num_pages*10),url)
    print(url)
    num_pages += 1
    
browser.quit() 


# In[21]:


print(jobsDF.shape)
jobsDF
# furthur cleaning of the raw jobsDF
jobsDF_refined=pd.DataFrame({'Title':jobsDF.Title.str.replace(r'^\d+/?\d+','').str.strip(),
                             'Company':jobsDF.Company.str.strip(),
                             'Location':jobsDF.Location.str.replace(r'\d+','').str.strip(),
                             'Link':jobsDF.Link})
jobsDF_refined.drop_duplicates(['Title','Company','Location'],keep='first',inplace=True)
print(jobsDF_refined.shape)
#jobsDF_refined[jobsDF_refined.Company=='Tesla']
#jobsDF_refined.Company


# In[493]:


#for each link, we want to go to the link and append the job description to a text file
# improvement: do a vectorized function
def write_job_descrp(links,text_file_name):
    import os
    if not os.path.isfile(os.getcwd()+'\\'+text_file_name):
        url_flag=[]
        browser = webdriver.Chrome()
        fh=open(text_file_name,'a+')
        for link in links: 
            if link:
                browser.get(link)    
                soup=bs4.BeautifulSoup(browser.page_source,'lxml')
                # problem: these webpages come in different flavours, not sure if the regex helps
                # you can also use selenium methods to do the same things
                # job_descrp_tag = browser.find_element_by_id(id='job_summary')
                # format 1)
                job_descrp_tag = soup.find(id='job_summary')
                if job_descrp_tag:
                    url_flag.append(1)
                    job_descrp=job_descrp_tag.get_text().lower()
                    #fh.write(re.sub(re.compile(r'[^a-zA-Z]+'),' ',job_descrp))
                # format 2)
                else:
                    job_descrp=''
                    job_descrp_tag=soup.find_all('div',attrs={'class':'jobDetailDescription'})
                    if job_descrp_tag:
                        # we can be more fussy and have flag=1 if all the sections below exist
                        url_flag.append(1)
                        for tag in job_descrp_tag:
                            section_tag = tag.find_next().find_next()
                            if section_tag:
                                section = section_tag.get_text().lower()
                            job_descrp=job_descrp + section                 
                    else:
                        url_flag.append(0)
                        continue
                fh.write(re.sub(re.compile(r'[^a-zA-Z]+'),' ',job_descrp))
            else:
                url_flag.append(0)
                continue
        browser.quit()
    else:
        raise BaseException('There is a file with the same name in the current working directory.')
    return {'flags' : url_flag, 'filehandle' : fh}
        
jobdescp_txt = write_job_descrp(jobsDF_refined.Link,'jobdescrp2.txt')
jobdescp_txt['filehandle'].close()

if sum(jobdescp_txt['flags'])==len(jobsDF_refined.Link):
    print('All the webpages are written successfully ')
else:
    print('Not all the webpages are written successfully. search for ~.flags = 0 to find unsuccessful instances')


# In[4]:


import nltk
nltk.download("stopwords")


# In[7]:


from collections import Counter
from nltk.corpus import stopwords

stopwords = set(stopwords.words("english"))
words = Counter(open('jobdescrp2.txt','r').read().split()).most_common()

words_nonstop=[]
for word in words:
    if word[0] not in stopwords:
        words_nonstop.append(word)
words_nonstop


# In[495]:


pd.set_option('display.max_colwidth', -1)
jobsDF_refined.iloc[0,:].Link
jobsDF_refined.iloc[-1,:]
jobsDF_refined[jobsDF_refined.Company=='CVS Health']

