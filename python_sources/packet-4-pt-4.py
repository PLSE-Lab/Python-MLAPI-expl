import glob
import pandas
import pandasql

def find_file():
    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

def run_sql(query):
    return pandasql.sqldf(query, globals())

seasons = pandas.read_excel(find_file(), sheet_name="Seasons")
print(seasons)

episodes = pandas.read_excel(find_file(), sheet_name="Episodes")
print(episodes)

characters = pandas.read_excel(find_file(), sheet_name="Characters")
print(characters)

actors = pandas.read_excel(find_file(), sheet_name="Actors")
print(actors)

characters = run_sql("""
    select *
    from characters
""")

print(seasons)