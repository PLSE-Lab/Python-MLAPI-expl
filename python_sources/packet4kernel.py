import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

restaurantsNorth = pandas.read_excel(find_file(), sheet_name="NorthRestaurants")
restaurantsSouth = pandas.read_excel(find_file(), sheet_name="SouthRestaurants")
neighbor  = pandas.read_excel(find_file(), sheet_name="Neighborhoods")

query1 = run_sql("""
    select NorthRestaurantName, NeighborhoodName
    from restaurantsNorth, neighbor
    where restaurantsNorth.NeighborhoodID = neighbor.NeighborhoodID
""")

print(query1)

query2 = run_sql("""
    select AVG(NorthRestaurantRating), AverageHouseholdIncome
    from restaurantsNorth, neighbor
    where restaurantsNorth.NeighborhoodID = 51 and neighbor.NeighborhoodID = 51
""")

print(query2)

query3 = run_sql("""
    select AVG(SouthRestaurantRating), AverageHouseholdIncome
    from restaurantsSouth, neighbor
    where restaurantsSouth.NeighborhoodID = 61 and neighbor.NeighborhoodID = 61
""")
print(query3)

