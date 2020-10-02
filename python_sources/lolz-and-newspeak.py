import sqlite3
import pandas as pd
import re
import requests

sql_conn = sqlite3.connect('../input/database.sqlite')

res = pd.read_sql("SELECT body FROM May2015 WHERE  LOWER(body) LIKE 'lol%' ", sql_conn)
#new_res = pd.read_sql("SELECT body FROM May2015 WHERE LOWER(body) LIKE 'lol%' AND (LOWER(body) LIKE '% u %' OR LOWER(body) LIKE '% r %')", sql_conn)
new_res = pd.read_sql("SELECT body FROM May2015 WHERE LOWER(body) LIKE 'lol%' OR LOWER(body) LIKE '% u %' OR LOWER(body) LIKE '% r %'", sql_conn)
cur = sql_conn.cursor()
cur.execute("SELECT COUNT(body) FROM May2015;")
total_comments = cur.fetchone()[0]

total_LOLs = res['body'].shape[0]
total_LOLS_n_Newspeak = new_res['body'].shape[0]
percent_LOLs = total_LOLs/total_comments*100
percent_LOLs_n_newspeak = total_LOLS_n_Newspeak/total_comments*100
print (total_comments)
print (total_LOLs)
print (total_LOLS_n_Newspeak)
print (str("%.2f" % round(percent_LOLs,2)) + "% of comments LOLed")
print (str("%.5f" % round(percent_LOLs_n_newspeak,2)) + "% of comments use Newspeak")

'''
max_word_in_comment_list=[]

for i in range (0, total_LOLs):
    comment = res['body'][i]
    split_comments = comment.replace('\n', ' ').replace('\r', ' ')
    
    split_comments = re.sub("[^a-zA-Z ]+", "", split_comments)
    split_comments = split_comments.split(' ')
    for word in split_comments:
        if 'http' in word or 'lol' in word.lower():
            split_comments.remove(word)
    if len(split_comments) > 0:
        max_word_in_comment_list.append(max(split_comments, key=len))
longest_word=max(max_word_in_comment_list, key=len)

try:
    url = 'http://services.aonaware.com/DictService/DictService.asmx/Define?word=%s' % longest_word
    r = requests.post('http://services.aonaware.com/DictService/DictService.asmx/Define', data={'test'})
    print (r.text)
except ConnectionError as e:
    print (e)
#print ("Longest word is: " + max(max_word_in_comment_list, key=len))
'''