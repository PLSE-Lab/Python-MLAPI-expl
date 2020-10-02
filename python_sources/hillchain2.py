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
    tmpL = tmpL - 3
    for j in range(tmpL):
      tmpR1a = row[j].lower()
      tmpR1b = row[j+1].lower()
      tmpR2a = row[j+2].lower()
      tmpR2b = row[j+3].lower()
      if len(tmpR1a) > 1 and len(tmpR2a)>1 and len(tmpR1b)>1 and len(tmpR2b):
        tmpW = tmpR1a+' '+tmpR1b
        if hilldic.get(tmpW) is None:
          hilldic[tmpW] = [tmpR2a+' '+tmpR2b]
        else:
          hilldic[tmpW] += [tmpR2a+' '+tmpR2b]          


#print(hilldic.items())

#start hillchain via random word
#startword = "president"
startword = "the benghazi"
speech = [startword]
for j in range(60):
  if hilldic.get(startword) is None:
    break
  else:
    tmpword = hilldic[startword][random.randint(0,len(hilldic[startword])-1)]
    print(tmpword)
    speech += [tmpword]
    startword = tmpword
    

print(' '.join(speech))