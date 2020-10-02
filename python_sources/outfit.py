import glob
import pandas
import pandasql
def find_file():
 return glob.glob("../input/**/*.xlsx", recursive=True)[0]
def run_sql(query):
    
 return pandasql.sqldf(query, globals())
Shoe = pandas.read_excel(find_file(), sheet_name="Shoe")
print(Shoe)
white = run_sql("""
 select ShoeID, Brand
 from Shoe
 where Color = 'white'
""")

print(white)

Accessory = pandas.read_excel(find_file(), sheet_name="Accessory")
print(Accessory)
gold = run_sql("""
 select AccessoryID, Brand
 from Accessory
 where Color = 'gold'
""")

print(gold)
