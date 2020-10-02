#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bs4
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup
import pandas as pd
import csv


# In[ ]:


my_url = 'https://www.cel.ro/placi-video/' 


# In[ ]:


uClient = ureq(my_url)


# In[ ]:


page_html = uClient.read()


# In[ ]:


uClient.close()


# In[ ]:


page_soup = soup(page_html,"html.parser")


# In[ ]:


page_soup.h1


# In[ ]:


containers = page_soup.findAll("div",{"class":"product_data productListing-tot"})


# In[ ]:


len(containers)


# In[ ]:


containers[0]


# In[ ]:


container = containers[0]


# In[ ]:


productTitle = container.findAll("h2",{"class":"productTitle"})


# In[ ]:


productTitle[0].text.strip()


# In[ ]:


pret_n = container.findAll("div",{"class":"pret_n"})


# In[ ]:


pret_n[0].text.strip()


# In[ ]:


pret_v = container.findAll("div",{"class":"pret_v"})


# In[ ]:


pret_v[0].text.strip()


# In[ ]:


caract_scurte = container.findAll("div",{"class":"caract_scurte"})


# In[ ]:


caract_scurte[0].text.strip()


# In[ ]:


oferte_alternative = container.findAll("div",{"class":"oferte_alternative"})


# In[ ]:


oferte_alternative[0].text.strip()


# In[ ]:


title = []
price_n = []
price_o = []
mem_gb = []
vendor = []


# In[ ]:


for container in containers : 
    productTitle = container.findAll("h2",{"class":"productTitle"})
    titlu = productTitle[0].text.strip().replace("Placa video ","")
    
    pret_n = container.findAll("div",{"class":"pret_n"})
    pretNou = pret_n[0].text.strip().replace("lei","") 
    
    try:
        pret_v = container.findAll("div",{"class":"pret_v"})
        pretVechi = pret_v[0].text.strip().replace(" lei","")
    except:
        pretVechi = -1
    
    caract_scurte = container.findAll("div",{"class":"caract_scurte"})
    crt = caract_scurte[0].text.strip().replace("Capacitate memorie (GB): ","").replace(" GB","")
    
    oferte_alternative = container.findAll("div",{"class":"oferte_alternative"})
    ofAlt = oferte_alternative[0].text.strip()
    
    title.append(titlu)
    price_n.append(pretNou)
    price_o.append(pretVechi)
    mem_gb.append(crt)
    vendor.append(ofAlt[10:])
    
    print("Titlu : " + titlu)
    print("Pret nou : " + pretNou)
    print("Pret vechi : " + str(pretVechi))
    print("Capacitate memorie (GB) : " + crt)
    print("Vandut de : " + ofAlt[10:])
    


# In[ ]:


page_start = 2
page_end = 8
page_url = "https://www.cel.ro/placi-video/0a-"


# In[ ]:


def page_scrape(page_start , page_end , page_url , title , price_n , price_o , mem_gb , vendor):
    for i in range (page_start,page_end + 1):
        my_url = page_url + str(i)
        uClient = ureq(my_url)
        page_html = uClient.read()
        uClient.close()
        page_soup = soup(page_html,"html.parser")
        containers = page_soup.findAll("div",{"class":"product_data productListing-tot"})

        for container in containers : 
            productTitle = container.findAll("h2",{"class":"productTitle"})
            titlu = productTitle[0].text.strip().replace("Placa video ","")

            pret_n = container.findAll("div",{"class":"pret_n"})
            pretNou = pret_n[0].text.strip().replace("lei","") 

            try:
                pret_v = container.findAll("div",{"class":"pret_v"})
                pretVechi = pret_v[0].text.strip().replace(" lei","")
            except:
                pretVechi = -1

            caract_scurte = container.findAll("div",{"class":"caract_scurte"})
            crt = caract_scurte[0].text.strip().replace("Capacitate memorie (GB): ","").replace(" GB","")

            oferte_alternative = container.findAll("div",{"class":"oferte_alternative"})
            ofAlt = oferte_alternative[0].text.strip()

            title.append(titlu)
            price_n.append(pretNou)
            price_o.append(pretVechi)
            mem_gb.append(crt)
            vendor.append(ofAlt[10:])
            
    return title,price_n,price_o,mem_gb,vendor
    


# In[ ]:


title ,price_n ,price_o ,mem_gb ,vendor = page_scrape(page_start ,page_end ,page_url ,title ,price_n ,price_o , mem_gb ,vendor)


# In[ ]:


for i in range(0, len(price_n)): 
    price_n[i] = int(price_n[i]) 
    
for i in range(0, len(price_o)): 
    price_o[i] = int(price_o[i]) 
   
for i in range(0, len(mem_gb)):
    try:
        mem_gb[i] = int(mem_gb[i])
    except:
        mem_gb[i] = -1
        

    


# In[ ]:


products = pd.DataFrame({
        "title": title,
        "price_n": price_n,
        "price_o": price_o,
        "mem_gb": mem_gb,
        "vendor": vendor
    })


# In[ ]:


products.to_csv("products.csv",index=False)

