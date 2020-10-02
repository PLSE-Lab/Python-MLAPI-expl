import pandas as pd
import sqlite3
import string
import random

con = sqlite3.connect('../input/database.sqlite')
#e = pd.read_sql_query("Select ExtractedBodyText From Emails Where length(ExtractedBodyText)>9 ",con)
#print(e[5:10])
#print(len(e))

hilldic = {}
for row in con.execute('Select ExtractedBodyText From Emails Where length(ExtractedBodyText)>9'):
  row = row[0].replace("\n", " ")
  row = row.split(' ')
  tmpL = len(row)
  if tmpL > 4:
    tmpL = tmpL - 2
    for j in range(tmpL):
      tmpR1 = row[j].lower()
      tmpR2 = row[j+1].lower()
      if len(tmpR1) > 1 and len(tmpR2)>1:
        if hilldic.get(tmpR1) is None:
          hilldic[tmpR1] = [tmpR2]
        else:
          hilldic[tmpR1] += [tmpR2]          


#print(hilldic.items())

#start hillchain via random word
#startword = "president"
startword = "obama"
speech = [startword]
for j in range(30):
  if hilldic.get(startword) is None:
    break
  else:
    tmpword = hilldic[startword][random.randint(0,len(hilldic[startword])-1)]
    print(tmpword)
    speech += [tmpword]
    startword = tmpword
    

print(' '.join(speech))