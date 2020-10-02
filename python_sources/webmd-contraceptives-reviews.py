import os
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import math
import string
import collections


url ='https://www.webmd.com/drugs/2/condition-3454/pregnancy%20contraception'
base_url = 'https://www.webmd.com'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc, "html5lib")

# extract links to review pages
links = [link.get('href') for link in soup.find_all('a', href=re.compile("drugreview"))]
# add base_url
links = [base_url + link for link in links]
# only keep links that have a review
links = links[0:269]
# 269 products

def num_pages(product_links):
    num_list = []
    remove_list = []
    for link in product_links:
        r = requests.get(link)
        html_doc = r.text
        soup = BeautifulSoup(html_doc, "html5lib")
        try:
            # get one string because duplicates
            num_page = [post.get_text() for post in soup.find_all(class_='postPaging')][0]
            start = 'of'
            end = 'Next'
            # number of reviews
            num = num_page[(num_page.find(start)+len(start)):num_page.rfind(end)]
            # number of review pages, given each page has 5 reviews
            num = math.ceil(int(num)/5)
        except:
            remove_link = product_links.index(link)
            remove_list.append(int(remove_link))
            continue
        num_list.append(int(num))

    return num_list, remove_list

# get list of number of pages for each product
product_page_num, r_list = num_pages(links)

# sum(product_page_num)
r_list = [198, 243]
# remove the products with no review page
links = np.delete(links, r_list).tolist()
# 267


def extract_reviews(review_link):
    """
    given a review link, this function will go to the page and extract comment, rating
    :param review_link: string
    :return: comments and ratings
    """
    # navigate to the review page to scrape reviews
    r = requests.get(review_link)
    html_doc = r.text
    soup = BeautifulSoup(html_doc, "html5lib")

    # extract full reviews
    full_reviews = [post.get_text() for post in soup.find_all('p', class_='comment',
                                                                id=['comFull1', 'comFull2', 'comFull3', 'comFull4',
                                                                    'comFull5'])]
    # remove words in the front and the back
    full_reviews = [post[8:-17] for post in full_reviews]

    # extract ratings
    ratings = [post.get_text() for post in soup.find_all('p', class_='inlineRating starRating')]
    # get ratings for 'Satisfaction' only: exclude the first 3 items as they're dummies, then take the 3rd reviews
    ratings = ratings[3::3]
    # remove words in the front and only get the ratings
    ratings = [rating[-1] for rating in ratings]

    return full_reviews, ratings

def get_review_link(link_product, num_page):
    """
    Function to get review links for each product. 299 is the highest number of review pages (Mirena)
    :param link_product: link of each product
    :param num_page: number of review pages for each product
    :return: list of links to all review pages of the product
    """
    link_product_list = []
    # eg. there are 299 pages for Mirena
    for i in range(num_page):
        link = link_product + '&pageIndex=' + str(i)
        link_product_list.append(link)
    return link_product_list

def combine_reviews(product_list, product_page_n):
    """
    crawl through all product links to get all reviews and combine into list of reviews and list of ratings
    :param product_list: list of contraceptive product links to be scraped
    :return review_df: dataframe of reviews and ratings
    """
    reviews = []
    ratings = []
    for i, link in enumerate(product_list):
        # list of review pages for each product
        link_list = get_review_link(link, product_page_n[i])
        for link_review in link_list:
            review, rating = extract_reviews(link_review)
            reviews.append(review)
            ratings.append(rating)
    flat_list = [item for sublist in reviews for item in sublist]
    flat_list_ratings = [item for sublist in ratings for item in sublist]
    review_df = pd.DataFrame({'ratings':flat_list_ratings, 'reviews': flat_list })
    return review_df

# scrape all reviews for all products
df = combine_reviews(links, product_page_num)
# export to csv file
df.to_csv('reviews.csv', index=False )