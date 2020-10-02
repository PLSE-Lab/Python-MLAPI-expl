from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import json
import codecs
import multiprocessing as mp

# stock quote
quote_dict = {  "Apple": "aapl", 
                "Tesla": "tsla", 
                "Microsoft": "msft", 
                "AT&T": "t", 
                "UnitedAirlines": "ual", 
                "FedEx": "fdx", 
                "Facebook": "fb", 
                "IBM": "ibm"}

# crawler function
def crawl(key, value):
    try:
        file_name = './Texts/Nasdaq/News/nasdaq' + key + 'News.json'
        f = open(file_name, 'w', encoding='utf-8')
        # number of pages to be crawled
        no_of_pages = 10
        news_counter = 0
        news_dict = {}
        for i in range(1, no_of_pages + 1):
            req = requests.get('https://www.nasdaq.com/symbol/' + value + '/news-headlines?page=' + str(i))
            soup = bs(req.text, 'lxml')

            target = soup.find('div', {'class': 'news-headlines'})

            target_headlines = target.find_all('a', {'target': '_self'})
            no_of_news = no_of_pages * 10

            for j in range(0, 10):
                print('appending ' + str(news_counter + 1) + ' news out of ' + str(no_of_news))
                if(i > 10):
                    news_dict["news" + str(i - 1) + str(j)] = {}
                    news_dict["news" + str(i - 1) + str(j)]["Heading"] = str(target_headlines[j].text.strip())
                else:
                    news_dict["news0" + str(i - 1) + str(j)] = {}
                    news_dict["news0" + str(i - 1) + str(j)]["Heading"] = str(target_headlines[j].text.strip())
                try:
                    req2 = requests.get(target_headlines[j].get('href'))
                    soup2 = bs(req2.text, 'lxml')
                    target2 = soup2.find('div', {'id': 'articleText'})
                    target_text = target2.find_all('p')
                    no_of_text = len(target_text)
                    paragraph = ''
                    for k in range(no_of_text):
                        paragraph = paragraph + str(target_text[k].text)

                except AttributeError:
                    req2 = requests.get(target_headlines[j].get('href'))
                    soup2 = bs(req2.text, 'lxml')
                    target2 = soup2.find('div', {'id': 'articlebody'})
                    target_text = target2.find_all('p')
                    no_of_text = len(target_text)
                    paragraph = ''
                    for k in range(no_of_text):
                        paragraph = paragraph + str(target_text[k].text) + " "
                if(i > 10):
                    news_dict["news" + str(i - 1) + str(j)]["Content"] = paragraph
                else:
                    news_dict["news0" + str(i - 1) + str(j)]["Content"] = paragraph

                news_counter = news_counter + 1
        f.write(json.dumps(news_dict, ensure_ascii=False))
        f.close()
    except:
        pass


output = mp.Queue()
processes = [mp.Process(target=crawl, args=(key, value)) for key, value in quote_dict.items()]

for p in processes:
    p.start()

for p in processes:
    p.join()

print("process completed")
