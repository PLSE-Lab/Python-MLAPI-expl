# The script will not run here as is, as Kaggle restrict from opening external URLs
# This script will try to do a sentiment analysis with text-processing.com  
# It will select all email texts and send it to text-processing.com to get the sentiment:
# {neutral, pos, neg}
# The results will be saved into a local table called santiments
# to create run: con.execute("CREATE TABLE Sentiment(id int primary key, sentiment tinyint)")

import sqlite3
import urllib
import json

# You can read in the SQLite datbase like this
con = sqlite3.connect('../output/database.sqlite')
c = con.cursor()
sql="""
SELECT e.Id as id, p.Name as sender, MetadataDateSent as datesent, ExtractedSubject as subject, ExtractedBodyText as text, RawText
FROM Emails e
INNER JOIN Persons p ON e.SenderPersonId=p.Id
where p.Name = 'Hillary Clinton'
and ExtractedBodyText <> ''
order by MetadataDateSent limit 1
"""

sentiment_label_dict = {"neg": -1, "neutral": 0, "pos": 1}
sentiment_res = []

for row in c.execute(sql):
#    print row[4]
    try:
        data = urllib.parse.urlencode({"text": row[4]}).encode('utf-8') 
    except UnicodeEncodeError:
        pass
    print(data)
    req = urllib.request.Request("http://text-processing.com/api/sentiment/", data)
    with urllib.request.urlopen(req) as f:
        the_page=f.read().decode('utf-8')
    the_page = u.read()
    res_label = sentiment_label_dict[json.loads(the_page)["label"]]
    print(res_label)
    sentiment_res.append((row[0],res_label))

print(sentiment_res)
c.executemany('''replace into Sentiment (id, sentiment) VALUES (?,?)''', sentiment_res)  
con.commit()

con.close()