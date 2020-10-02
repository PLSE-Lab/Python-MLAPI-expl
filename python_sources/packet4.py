import glob
import pandas
import pandasql

def find_file():
 return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
 return pandasql.sqldf(query, globals())

PRICES = pandas.read_excel(find_file(), sheet_name="PRICES")
print(PRICES)

PriceID = run_sql("""
 select *
 from PRICES
 where PriceID='1'""")
print(PriceID)