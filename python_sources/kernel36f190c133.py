# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from bs4 import BeautifulSoup
import requests
import json

import csv


def crawl_kompas(url):
    
    result = []
   
# Looping through page 1 to 8

    for i in range(1, 8):
       
       
        req = requests.get(url + str(i))
    
        soup = BeautifulSoup(req.text, "lxml")
        news_links = soup.find_all('div', class_='group-right')
        # looping through Pages !
        
        for idx, news in enumerate(news_links):
               
            news_dict = {}
           
            # find news title
            title_news = news.find('div', class_='field field-name-title-qs')
            des = news.find('div', class_='field field-name-field-course-summary')
            url_news = news.find('div', class_='field field-name-title-qs')
            if None in (url_news, title_news):
                  continue
            print("Is it working ?")
            print(title_news.text.strip())
            q = title_news.text.strip();
            
            y = url_news.text.strip()
            print(url_news.find('a')['href'])
            x = url_news.find('a')['href']
            c = des.text.strip()
            
            print(c)
            
           # Find required Syllabus and description
           
            req_news = requests.get("https://online-learning.harvard.edu" + x)  

          # Syllabus  
            print("I am  here")
        soup_news = BeautifulSoup(req_news.text, "lxml")
        news_content = soup_news.find("ul")
        news_content = soup_news.find("div", {'class':'field-name-field-course-nutshell field field-name-field-course-nutshell field-type-text-long field-label-above'})
        p = news_content.find_all('li')
        content = ' \ '.join(item .text for item in p)
           
        news_content = content.encode('utf8', 'replace')
        #
        soup_des = BeautifulSoup(req_news.text, "lxml")
        des_content = soup_des.find("ul")
        des_content = soup_des.find("div", {'class':'field-name-field-course-nutshell field field-name-field-course-nutshell field-type-text-long field-label-above'})
        p = des_content.find_all('li')
        Des = ' \ '.join(item .text for item in p)
           
        des_content = Des.encode('utf8', 'replace')
        print (des_content)

        print(news_content)
        news_dict['id'] = idx
        news_dict['url'] = y
        news_dict['title'] = q
        news_dict['Syllabus'] = news_content

        news_dict['Description'] = des_content
        result.append(news_dict)
        print()
        csvRow = [q, news_content, des_content]
        csvfile = "data.csv"
        with open(csvfile, "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(csvRow)
       
    return result


url = 'https://online-learning.harvard.edu/CATALOG?page='
crawl = crawl_kompas(url)
