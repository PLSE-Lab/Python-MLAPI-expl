# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT INSTNM,
       COSTT4_A AverageCostOfAttendance,
       Year
FROM Scorecard
WHERE INSTNM='Duke University'""", con)
print(sample)

# You can read a CSV file like this
df = pd.read_csv("../input/Scorecard.csv")
#print(df.head())

# It's yours to take from here!
df1=df.loc[df['CONTROL']=='Public',['UNITID','INSTNM','Year','COSTT4_A']].dropna()
df1=df1.pivot(index='UNITID', columns='Year', values='COSTT4_A')
df1.to_csv('cost.tsv',sep='\t')

df2=df.loc[df['CONTROL']=='Public',['UNITID','INSTNM','Year','md_earn_wne_p6']].dropna()
df2.loc[df2['md_earn_wne_p6']=='PrivacySuppressed','md_earn_wne_p6']=None
df2['md_earn_wne_p6']=pd.to_numeric(df2['md_earn_wne_p6'])
df2=df2.pivot(index='UNITID', columns='Year', values='md_earn_wne_p6')
df2=df2.apply(lambda x: pd.qcut(x, 2, labels=['No','Yes']), axis=0)
df2.to_csv('earnings.tsv', sep='\t')

df2=df.loc[df['CONTROL']=='Public',['UNITID','INSTNM','Year','CDR2']].dropna()
df2=df2.pivot(index='UNITID', columns='Year', values='CDR2')
df2=(df2==0)
df2.to_csv('default.tsv', sep='\t')