#!/usr/bin/env python
# coding: utf-8

# In this simple kernal, I illustrate how to we can use `BeautifulSoup` to scrape the International Movies Database (IMDB) at [imdb.com](https://imdb.com) for top films released in year 2018 with the highest US box office. 
# 
# I am organizing the final results as a dataframe with below elements:
# 
# * `name` - title of the movie, 
# * `year` - release year of the movie, 
# * `imdb` - IMDB score of the movie, 
# * `m_score` - meta score of the movie, 
# * `vote` - number of votes.

# First, we import the requried packages

# In[ ]:


import bs4
import requests
import time
import random as ran
import sys
import pandas as pd


# Now I search for the for [top 1000 films released in year of 2018 at imdb.com](https://www.imdb.com/search/title?release_date=2018&sort=boxoffice_gross_us,desc&start=1) and scrape results from the first page

# In[ ]:


url = 'https://www.imdb.com/search/title?release_date=2018&sort=boxoffice_gross_us,desc&start=1'

source = requests.get(url).text
soup = bs4.BeautifulSoup(source,'html.parser')


# Since above code extracts all data on the first page, below code is run only to extract movie information on it.

# In[ ]:


movie_blocks = soup.findAll('div',{'class':'lister-item-content'})


# Before extracting information across all movies, I first examine one of the extracted block to identify the elements that we need to scrape.
# 
# Below I've extracted the elements from the first movie block

# In[ ]:


mname = movie_blocks[0].find('a').get_text() # Name of the movie

m_reyear = int(movie_blocks[0].find('span',{'class': 'lister-item-year'}).contents[0][1:-1]) # Release year

m_rating = float(movie_blocks[0].find('div',{'class':'inline-block ratings-imdb-rating'}).get('data-value')) #rating

m_mscore = float(movie_blocks[0].find('span',{'class':'metascore favorable'}).contents[0].strip()) #meta score

m_votes = int(movie_blocks[0].find('span',{'name':'nv'}).get('data-value')) # votes

print("Movie Name: " + mname,
      "\nRelease Year: " + str(m_reyear),
      "\nIMDb Rating: " + str(m_rating),
      "\nMeta score: " + str(m_mscore),
      "\nVotes: " + '{:,}'.format(m_votes)

)


# Once you examine the resulting pages of the imbd search that we initially did , it's obvious that by editing the html link it is possible to view all search results. Thus I will be using this feature during the scrape to iterate through all pages.

# Now since scraping the data is an iterative process, I define separate functions for each purpose.
# 
# First I am going to define a function which will extract the targeted elements from a 'movie block list' (discussed above)

# In[ ]:


def scrape_mblock(movie_block):
    
    movieb_data ={}
  
    try:
        movieb_data['name'] = movie_block.find('a').get_text() # Name of the movie
    except:
        movieb_data['name'] = None

    try:    
        movieb_data['year'] = str(movie_block.find('span',{'class': 'lister-item-year'}).contents[0][1:-1]) # Release year
    except:
        movieb_data['year'] = None

    try:
        movieb_data['rating'] = float(movie_block.find('div',{'class':'inline-block ratings-imdb-rating'}).get('data-value')) #rating
    except:
        movieb_data['rating'] = None

    try:
        movieb_data['m_score'] = float(movie_block.find('span',{'class':'metascore favorable'}).contents[0].strip()) #meta score
    except:
        movieb_data['m_score'] = None

    try:
        movieb_data['votes'] = int(movie_block.find('span',{'name':'nv'}).get('data-value')) # votes
    except:
        movieb_data['votes'] = None

    return movieb_data
    


# Then I create the below function to scrape all movie blocks within a single search result page

# In[ ]:


def scrape_m_page(movie_blocks):
    
    page_movie_data = []
    num_blocks = len(movie_blocks)
    
    for block in range(num_blocks):
        page_movie_data.append(scrape_mblock(movie_blocks[block]))
    
    return page_movie_data


# Now we built functions to extract all movie data from a single page.
# 
# Next function will be created to iterate the above made function through all pages of the search result untill we scrape data for the targeted number of movies

# In[ ]:


def scrape_this(link,t_count):
    
    #from IPython.core.debugger import set_trace

    base_url = link
    target = t_count
    
    current_mcount_start = 0
    current_mcount_end = 0
    remaining_mcount = target - current_mcount_end 
    
    new_page_number = 1
    
    movie_data = []
    
    
    while remaining_mcount > 0:

        url = base_url + str(new_page_number)
        
        #set_trace()
        
        source = requests.get(url).text
        soup = bs4.BeautifulSoup(source,'html.parser')
        
        movie_blocks = soup.findAll('div',{'class':'lister-item-content'})
        
        movie_data.extend(scrape_m_page(movie_blocks))   
        
        current_mcount_start = int(soup.find("div", {"class":"nav"}).find("div", {"class": "desc"}).contents[1].get_text().split("-")[0])

        current_mcount_end = int(soup.find("div", {"class":"nav"}).find("div", {"class": "desc"}).contents[1].get_text().split("-")[1].split(" ")[0])

        remaining_mcount = target - current_mcount_end
        
        print('\r' + "currently scraping movies from: " + str(current_mcount_start) + " - "+str(current_mcount_end), "| remaining count: " + str(remaining_mcount), flush=True, end ="")
        
        new_page_number = current_mcount_end + 1
        
        time.sleep(ran.randint(0, 10))
    
    return movie_data
    
    


# Finally, I put together all functions created above to scrape the top 150 movies on the list

# In[ ]:


base_scraping_link = "https://www.imdb.com/search/title?release_date=2018-01-01,2018-12-31&sort=boxoffice_gross_us,desc&start="

top_movies = 150 #input("How many movies do you want to scrape?")
films = []

films = scrape_this(base_scraping_link,int(top_movies))

print('\r'+"List of top " + str(top_movies) +" movies:" + "\n", end="\n")
pd.DataFrame(films)


# That's it! Likewise, the same principle can be used to scrape data from any page!
# 
# I believe if you follow a bottom-up approach when scraping data, i.e., to build functions to extract data from the smallest pieces and work yourself up to higher levels, it'll make your life much easier than tryin to extract a lot of imformation at once.
# 
# Thanks for reading! Please let me know what you think and appreciate any feedback.
