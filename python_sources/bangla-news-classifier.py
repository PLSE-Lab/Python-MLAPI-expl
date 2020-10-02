from rafikamalstemmer import RafiStemmer
import csv
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import logging
import time
from newspaper import Article
import random
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dateutil.parser import parse
import json
import requests

class Scrapper:
    _header = {'Accept-Encoding': 'gzip, deflate, sdch',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/41.0.2272.118 Safari/537.36'}

    def __init__(self, url):
        self._url = url

    def scrape_site(self):
        """Scrape raw data from this url"""
        if self._url is not None:
            res = requests.get(self._url, headers=self._header, verify=False)
            if res.status_code == 200:
                print("status code 200!!!")
                return res
            else:
                print(res.status_code)
                return
        else:
            print("Must give a valid url!")

def get_epoch_time(time_obj):
    if time_obj:
        return time_obj.timestamp()
    return None

def crawl_link_article(id_type, url):
    name = ''.join(random.choice("012345") for i in range(7))
    id = id_type + "_" + name
    url = url.strip()

    url_parse_result = urlparse(url)
    base_url = url_parse_result.netloc

    result_json = None
    try:
        if 'http' not in url:
            if url[0] == '/':
                url = url[1:]
            try:
                article = Article('http://' + url)
                article.download()
                time.sleep(2)
                article.parse()
                flag = True
            except:
                logging.exception("Exception in getting data from url {}".format(url))
                flag = False
                pass
            if flag == False:
                try:
                    article = Article('https://' + url)
                    article.download()
                    time.sleep(2)
                    article.parse()
                    flag = True
                except:
                    logging.exception("Exception in getting data from url {}".format(url))
                    flag = False
                    pass
            if flag == False:
                return None
        else:
            try:
                article = Article(url)
                article.download()
                time.sleep(2)
                article.parse()
            except:
                logging.exception("Exception in getting data from url {}".format(url))
                return None

        if not article.is_parsed:
            return None

        visible_text = article.text
        top_image = article.top_image
        images = article.images
        keywords = article.keywords
        authors = article.authors
        canonical_link = article.canonical_link
        title = article.title
        meta_data = article.meta_data
        movies = article.movies
        publish_date = article.publish_date
        source = article.source_url
        summary = article.summary
        text = meta_data['description']

        if base_url == 'www.prothomalo.com':
            prothomalo = Prothomalo(url).get_news_dict()
            if prothomalo['body'] != "":
                text = prothomalo['body']
            else:
                text = meta_data['description']
    except:
        logging.exception("Exception in fetching article form URL : {}".format(url))

    return [id, title, text, url, top_image, authors, source, get_epoch_time(publish_date), movies, images,
            canonical_link, meta_data]

def collect_news_articles(news_list, label):
    with open("{}/dataset_{}.csv".format(base_path, label), 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        t = 0
        writer.writerow(["id", "title", "text", "url", "top_image", "authors", "source","publish_date", "movies", "images","canonical_link", "meta_data"])
        for source in news_list:
            t+=1
            if t%50==0:
                time.sleep(30)
            x = crawl_link_article(label,source.link)
            while x is None:
                x = crawl_link_article(label, source.link)
            writer.writerow(x)
            print(t, " : ok")


class NewsContentCollector:
    def __init__(self, news_list, label):
        self.news_list = news_list
        self.label = label
        create_dir(base_path)

    def collect_data(self):
        collect_news_articles(self.news_list, self.label)


class SingleNewsCollector:
    def collect_single_news(self, url):
        news_data = crawl_link_article("", url)
        return (news_data[1], news_data[2])

class Prothomalo:
    def __init__(self, url):
        scrapper = Scrapper(url)
        raw_data = scrapper.scrape_site()
        self.soup = BeautifulSoup(raw_data.text, "html.parser")

    def get_title(self):
        """Parsing the title form news"""
        title = self.soup.title.string
        return title

    def get_category(self):
        """Parsing the category form news"""
        category = self.soup.find('div', attrs={'class': 'breadcrumb'})
        news_category = category.ul.find_all('li').pop().strong.string
        # print(news_category)
        return news_category

    def get_body_images(self):
        """This function will parse the news body and all images"""
        try:
            article = self.soup.find('article')
            article_body = article.find_all('p')
            images = []
            news_body = ''
            clean = re.compile('<.*?>')
            for p in article_body:
                if p.string is None:
                    if p.img is not None:
                        images.append(p.img['src'])
                        if p.img.next_sibling is not None:
                            news_body += p.img.next_sibling
                    else:
                        cleaned_p = re.sub(clean, '', str(p))
                        news_body += cleaned_p

                else:
                    news_body += p.string
                    # news_body += '\n'

            return news_body, images

        except:
            return "", ""

    def get_date(self):
        """This function will parse the news body and all images"""
        date = self.soup.find('span', attrs={"itemprop": "datePublished"})
        date_bn = date.string
        datetime = parse(date['content'])
        return datetime, date_bn

    def get_news_dict(self):
        """This function will generate news dictionary"""
        body, images = self.get_body_images()
        date, date_bn = self.get_date()
        news_dict = {
            'title': self.get_title(),
            'category': self.get_category(),
            'body': body,
            'images': images,
            'date': str(date),
            # 'date_bn': str(date_bn),
        }
        # print(news_dict)
        return news_dict

    def get_news_json(self):
        """This function will return news JSON"""
        news_json = json.dumps(self.get_news_dict()).encode('utf8').decode('unicode-escape')
        return news_json


def generator(input_document, cl_output, batch_size):
    #generator function not implemented here, loading whole training set at a time
    x = np.zeros((batch_size, len(vocab_set)), dtype=np.bool)
    y = np.zeros(batch_size, dtype=np.bool)
    for i in range(batch_size):
        for w in input_document[i]:
            x[i, word_indices[w]] = 1
        y[i] = bool(cl_output[i])
    return (x, y)

def sd_vectorizer(input_document):
    #generator not implemented yet
    x = np.zeros( len(vocab_set), dtype=np.bool)
    for w in input_document:
        if w in word_indices:
            x[word_indices[w]] = 1
    return x

def vectorizer(input_document):
    x = np.zeros((len(input_document), len(vocab_set)), dtype=np.bool)
    for i in range(len(input_document)):
        for w in input_document[i]:
            x[i, word_indices[w]] = 1
    return x

def shuffle_training_set(doc_orig, class_orig):
    tmp_sentences = []
    tmp_class = []
    for i in np.random.permutation(len(doc_orig)):
        tmp_sentences.append(doc_orig[i])
        tmp_class.append(class_orig[i])
    
    return (tmp_sentences, tmp_class)

corpus_fake_text = []
with open('/kaggle/input/data-fake/dataset_fake.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            doc_text = row[1] + '\n' + row[2]
            line_count += 1
            doc_words = re.findall(r'[\u0980-\u09FF]+', doc_text)
            corpus_fake_text.append(doc_words)
corpus_fake_text = [[re.sub(r'\d+', ' ', word) for word in document]for document in corpus_fake_text]

corpus_real_text = []
with open('/kaggle/input/data-real/dataset_real.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            doc_text = row[1] + '\n' + row[2]
            line_count += 1
            doc_words = re.findall(r'[\u0980-\u09FF]+', doc_text)
            corpus_real_text.append(doc_words)
corpus_real_text = [[re.sub(r'\d+', ' ', word) for word in document]for document in corpus_real_text]

rules = open('/kaggle/input/stemming-rules/common.rules', 'r', encoding='utf-8')
my_stemmer = RafiStemmer(rules)
vocab = []

document_fake = []
for doc in corpus_fake_text:
    doc_fake = []
    for word in doc:
        stemmed_word = my_stemmer.stem_word(word)
        if len(stemmed_word) < 2:
            continue
        doc_fake.append(stemmed_word)
        vocab.append(stemmed_word)
    document_fake.append(doc_fake)

document_real = []
for doc in corpus_real_text:
    doc_real = []
    for word in doc:
        stemmed_word = my_stemmer.stem_word(word)
        if len(stemmed_word) < 2:
            continue
        doc_real.append(stemmed_word)
        vocab.append(stemmed_word)
    document_real.append(doc_real)

vocab_set = sorted(set(vocab))
#Building Dictionary
word_indices = dict((c, i) for i, c in enumerate(vocab_set))

print('**** Vocabulary Ready - Preparing Classifier Input ****')

document = document_fake + document_real
classification = []
for doc in document_fake:
    classification.append(0)
for doc in document_real:
    classification.append(1)

document, classification = shuffle_training_set(document, classification)
(X, Y) = generator(document, classification, len(document))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size =  0.2)

print('**** Training Models - Split Ration 80-20 ****')

chosen_models = {}
chosen_models['LogisticRegression'] = LogisticRegression(max_iter=200)
chosen_models['RandomForest'] = RandomForestClassifier(max_depth=2, random_state=0)
chosen_models['GausNB.sav'] = GaussianNB()
#chosen_models['KNeigobor'] = KNeighborsClassifier(n_neighbors=2)
#chosen_models['DecisionTree'] = DecisionTreeClassifier()

for name, model in chosen_models.items():
    print("working with - " + name)
    model.fit(X_train, y_train)
    #predictions = model.predict(X_test)
    #print(classification_report(predictions, y_test))
    #print("*************")

url = "prothomalo.com/sports/article/1631115"
news_collector = SingleNewsCollector()
title, body = news_collector.collect_single_news(url)

document = title + ' ' + body
document = re.findall("[\u0980-\u09FF']+", document)
clean_doc = []

for word in document:
    stemmed_word = my_stemmer.stem_word(word)
    if len(stemmed_word) > 2:
        clean_doc.append(stemmed_word)
clean_doc = set(clean_doc)

X_Test = sd_vectorizer(clean_doc).reshape(1, -1)

for name, model in chosen_models.items():
    log_prediction = model.predict(X_Test)
    log_proba = model.predict_proba(X_Test)
    print('='*20 + name + 'Prediction')
    print(log_prediction)
    print(log_proba)





