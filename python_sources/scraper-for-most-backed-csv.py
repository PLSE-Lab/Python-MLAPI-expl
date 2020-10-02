import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import urllib
import re
#from IPython.display import display, HTML
import HTMLParser

proj_link_list = []
proj_title_list = []
proj_blurb_list = []
proj_by_list = []

for num in range(200):

    link = urllib.urlopen('https://www.kickstarter.com/discover/advanced?state=successful&sort=most_backed&seed=2462842&page='+str(num+1)).read()
    soup = BeautifulSoup(link, 'html.parser')

    proj_title = soup.find_all("div", class_="project-profile-title text-truncate-xs")
    # project link
    title_str = HTMLParser.HTMLParser().unescape(str(proj_title)) # remove the &quot
    proj_title_list += re.findall('target="">(.*?)</a>',title_str)
    proj_link_list += re.findall('href="(/projects/.*?)[\?]ref=most_backed"', title_str)
    # project blurb
    proj_blurb = soup.find_all("p", class_="project-profile-blurb type-12")
    proj_blurb_list += [s.string for s in proj_blurb]
    proj_by = soup.find_all("p", class_="project-profile-byline type-12")
    proj_by_str = HTMLParser.HTMLParser().unescape(str(proj_by))
    proj_by_list += re.findall('by (.*?) and \d+[,\d+]* backers',proj_by_str)

numBackers_list = []
currency_type_list = []
amt_pledged_list = []
goal_list = []
location_list = []
category_list = []
pledge_tier_list = []
numBackers_tier_list = []


for url in proj_link_list:

    url = 'https://www.kickstarter.com'+url
    link = urllib.urlopen(url).read()
    soup = BeautifulSoup(link,'html.parser')

    # project stats
    stat = soup.find("div", class_="NS_campaigns__spotlight_stats")
    stat_str = HTMLParser.HTMLParser().unescape(str(stat))
    # number of backers
    numBackers = re.findall('<b>(.*?) backers</b>',stat_str)
    numBackers_list.append(int(re.sub(',','',numBackers[0])))
    currency_type = re.findall('<span class="money (.*?) [no\-code]*">', stat_str)
    currency_type_list.append(currency_type[0])
    amt_pledged = re.findall(' [no\-code]*">\D+(.*?)</span>', stat_str)
    amt_pledged_list.append(float(re.sub(',','',amt_pledged[0])))

    # project goal
    goal = soup.find("div", class_="col-right col-4 py3 border-left")
    goal_str = HTMLParser.HTMLParser().unescape(str(goal))
    goal = re.findall('>\D+(.*?)</span> goal',goal_str)
    goal_list.append(float(re.sub(',','',goal[0])))

    # project location and category
    temp = soup.find("div", class_="NS_projects__category_location ratio-16-9")
    temp = str(temp)
    location = re.findall('<span aria-hidden="true" class="ksr-icon__location"></span>\n(.*?)\n</a>',temp)
    location_list.append(location[0])
    category = re.findall('<span aria-hidden="true" class="ksr-icon__tag"></span>\n(.*?)\n</a>',temp)
    category_list.append(category[0])

    # pledge tiers and corresponding number of backers
    pledge_tier = []
    numBackers_tier = []
    pledge_info = soup.find_all("div", class_="pledge__info")

    for i_pledge in pledge_info:
        pledge_str = str(i_pledge)
        amt = re.findall('About <span>\D+(.*?) USD',pledge_str)
        # note that pledge_tier is already in USD
        pledge_tier.append(float(re.sub(',','',amt[0])))
        backers = re.findall('pledge__backer-count">\n(.*?) backer',pledge_str)
        numBackers_tier.append(int(re.sub(',','',backers[0])))

    pledge_tier_list.append(pledge_tier)
    numBackers_tier_list.append(numBackers_tier)

df = pd.DataFrame({'by':proj_by_list, 'title':proj_title_list, 'url':proj_link_list, 'blurb':proj_blurb_list, 'num.backers':numBackers_list, 'currency':currency_type_list,'amt.pledged':amt_pledged_list,'goal':goal_list,'location':location_list,'category':category_list,'pledge.tier':pledge_tier_list,'num.backers.tier':numBackers_tier_list})

#df.info()
#display(df)
df.to_csv('most_backed.csv',encoding = 'UTF-8')
