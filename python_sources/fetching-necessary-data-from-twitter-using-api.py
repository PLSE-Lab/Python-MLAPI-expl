"""
didn't follow coding standards
"""

import base64
import pandas as pd
import requests

def get_hashtags(user):
	hashtag1 = []
	
	hashtag1.extend([item["text"] for item in user["entities"]["hashtags"]])
	if "retweeted_status" in user:
		hashtag1.extend([item["text"] for item in user["retweeted_status"]["entities"]["hashtags"]])
	return list(set(hashtag1))
	
def get_user_mentions(user):
	user_mentions = []
	
	user_mentions.extend([item["screen_name"] for item in user["entities"]["user_mentions"]])
	if "retweeted_status" in user:
		user_mentions.extend([item["screen_name"] for item in user["retweeted_status"]["entities"]["user_mentions"]])
	return list(set(user_mentions))
	
#'expanded_url'
def get_mentioned_urls(user):
	urls = []
	if "media" not in user["entities"]:
		return urls
	urls.extend([item["expanded_url"] for item in user["entities"]["media"]])
	if "retweeted_status" in user:
		urls.extend([item["expanded_url"] for item in user["retweeted_status"]["entities"]["media"]])
	return list(set(urls))


def get_date_format(date):
	date = date.split()
	date = date[1:3]+date[-1:]
	print("date : " + str(date))
	return "-".join(date)
	
def get_time_format(time):
	time = time.split()
	print("time : " + time[3])
	return time[3]
	
	
class Twitter():
	token = "-1"
	base_url = 'https://api.twitter.com/'
	def __init__(self,client_key = "",client_secret = ""):
		self.client_key = ''#insert your Client Key
		self.client_secret = ''#insert your Client Secret Key
	def set_urls(self):
		
		auth_url = '{}oauth2/token'.format(self.base_url)
		return auth_url
	def b64_encoded_key(self):
		key_secret = '{}:{}'.format(self.client_key, self.client_secret).encode('ascii')
		b64_encoded_key = base64.b64encode(key_secret)
		b64_encoded_key = b64_encoded_key.decode('ascii')
		return b64_encoded_key
	def auth_headers(self):
		auth_headers = {
    'Authorization': 'Basic {}'.format(self.b64_encoded_key()),\
    'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'\
	}
		return auth_headers
	def auth_data(self):
		auth_data = {
    'grant_type': 'client_credentials'\
	}
		return auth_data
	def get_token(self):
		auth_resp = requests.post(self.set_urls(), headers=self.auth_headers(), data=self.auth_data())
		try:
			auth_resp.json()
			access_token = auth_resp.json()['access_token']
			self.token = access_token
			return access_token
		except:
			return -1
	def search_request(self,q = "@urstrulyMahesh", result_type = "recent", count = 3):
		search_headers = {
    'Authorization': 'Bearer {}'.format(self.token)\
	}
		search_params = {
			'q': q,
			'result_type': result_type,
			'tweet_mode' : "extended",
			'count': count\
			}

		search_url = '{}1.1/search/tweets.json'.format(self.base_url)

		search_resp = requests.get(search_url, headers=search_headers, params=search_params)
		return search_resp
item = Twitter()
print(item.get_token())

metro_polices = ["@DelhiPolice","@BlrCityPolice","@KolkataPolice","@HYDTraffic","@blrcitytraffic","@CCTPolice_Alert","@hydcitypolice","@dtptraffic","@MumbaiPolice","@jaipur_police","@TheKeralaPolice"]
city_police = ["Delhi","Bengaluru","Kolkata","Hyderabad","Bengaluru","Chennai","Hyderabad","Delhi","Mumbai","jaipur","kerala"]
from pprint import pprint
#[pprint(item["text"]) for item in police["statuses"]]
public_requests_police = pd.DataFrame()
for index,search_query in enumerate(metro_polices):
	police = item.search_request(search_query, count = 500).json()
	rows = []
	
	for user in police["statuses"]:
		print("*****************")
		print("User Description : "+user["user"]["description"])
		print("User Name : "+user["user"]["name"])
		print("Twitter Name : "+user["user"]["screen_name"])
		#print("profile URL : "+user["user"]["url"])
		print("Tweet Text : "+user["full_text"])
		print("Link to tweet : https://twitter.com/statuses/"+user["id_str"])
		print("*****************")

		
		if "retweeted_status" in user:
			is_retweeted = "yes"
			who_retweeted = user["retweeted_status"]["user"]["screen_name"]
		else:
			is_retweeted = "no"
			who_retweeted = "NA"
		rows.append([user["user"]["name"],user["user"]["screen_name"],user["user"]["location"],user["user"]["description"]\
		,user["full_text"],city_police[index],str(get_mentioned_urls(user)),user["lang"],"https://twitter.com/statuses/"+user["id_str"],is_retweeted,who_retweeted\
		,str(get_hashtags(user)),str(get_user_mentions(user)), get_date_format(user["created_at"]), get_time_format(user["created_at"])])

	tweets = pd.DataFrame(rows,columns = ["User_Name", "Twitter_Name","Location", "Bio", "Tweet","city_police","media","Language","Tweet_Link",\
	"is_Retweeted","Original_Tweet_by","Hashtags","User_Mentions","tweet_date","tweet_time"])
	#print(rows)
	public_requests_police = public_requests_police.append(tweets,ignore_index = True)
	tweets.to_excel(search_query+".xlsx")
	pprint(police)	
public_requests_police.to_excel("public_requests_police.xlsx")
			