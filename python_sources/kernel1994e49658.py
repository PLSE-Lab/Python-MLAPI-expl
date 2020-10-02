import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

Restaurant = pandas.read_excel(find_file(), sheet_name="Restaurant")
print(Restaurant)

Monument = pandas.read_excel(find_file(), sheet_name="Monument")
print(Monument)

Vicinity = pandas.read_excel(find_file(), sheet_name="Vicinity ")
print(Vicinity)

FoodItems = pandas.read_excel(find_file(), sheet_name="FoodItems")
print(FoodItems)

OsteriaFoods = run_sql("""
    select FoodItemName
    from FoodItems
    where RestaurantID=1
""")
print(OsteriaFoods)

David = run_sql("""
    select *
    from Vicinity
    where MonumentID=3
""")
print(David)

